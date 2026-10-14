"""Phần cần GPU: nạp Qwen3-VL-4B QLoRA 4-bit, encode + mask loss, collate, setup E1/E2/E3, suy luận."""
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
from PIL import Image  # noqa: E402

from lab.common import MODEL_ID, write_jsonl  # noqa: E402

# 36 lớp LLM x 7 = 252 module; 24 block ViT x 4 = 96 module (E3 tổng 348).
LLM_TARGETS = r"model\.language_model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)"
VIT_TARGETS = r"model\.visual\.blocks\.\d+\.(attn\.(qkv|proj)|mlp\.(linear_fc1|linear_fc2))"
PROJECTOR_KEYS = ("visual.merger", "visual.deepstack_merger_list")


def bnb_config():
    from transformers import BitsAndBytesConfig
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        # ViT (gồm cả merger/deepstack) và lm_head giữ fp16: merger phải là tham số thật mới mở full được.
        # transformers 5.x so khớp bằng re.match neo ở ĐẦU tên đầy đủ ("model.visual.merger..."), nên
        # "visual" một mình KHÔNG khớp -> cả ViT bị lượng tử hoá âm thầm. Thêm "model.visual" (xem checks.py).
        llm_int8_skip_modules=["visual", "model.visual", "lm_head"],
    )


def assert_vit_not_quantized(model):
    bad = [n for n, m in model.named_modules()
           if ".visual." in f".{n}." and type(m).__name__ in ("Linear4bit", "Linear8bitLt")]
    assert not bad, f"ViT/projector bị lượng tử hoá ({len(bad)} module, vd {bad[:2]}): sửa llm_int8_skip_modules"


def load_model(model_id=MODEL_ID):
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id, quantization_config=bnb_config(), dtype=torch.float16, device_map={"": 0})
    assert_vit_not_quantized(model)
    processor = AutoProcessor.from_pretrained(model_id)
    processor.tokenizer.padding_side = "left"
    return model, processor


def build_messages(image, prompt, answer=None):
    msgs = [{"role": "user", "content": [{"type": "image", "image": image},
                                         {"type": "text", "text": prompt}]}]
    if answer is not None:
        msgs.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
    return msgs


def load_image(entry, data_dir="data"):
    return Image.open(Path(data_dir) / entry["image_path"]).convert("RGB")


SEQ_KEYS = ("input_ids", "attention_mask", "mm_token_type_ids")


def encode_example(processor, image, prompt, answer):
    """Chuỗi đầy đủ (ảnh + prompt + đáp án) và labels: -100 cho mọi token ảnh/prompt,
    chỉ giữ token của câu trả lời assistant (JSON + <|im_end|>)."""
    kw = dict(tokenize=True, return_dict=True, return_tensors="pt")
    p = processor.apply_chat_template(build_messages(image, prompt), add_generation_prompt=True, **kw)
    f = processor.apply_chat_template(build_messages(image, prompt, answer), add_generation_prompt=False, **kw)
    n = p["input_ids"].shape[1]
    assert torch.equal(f["input_ids"][0, :n], p["input_ids"][0]), "prompt không phải tiền tố của chuỗi đầy đủ"
    ids = f["input_ids"][0]
    labels = ids.clone()
    labels[:n] = -100
    # template thêm "\n" sau <|im_end|>; không cần học token này
    im_end = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
    last = (ids == im_end).nonzero()[-1].item()
    labels[last + 1:] = -100
    out = {k: (v[0] if k in SEQ_KEYS else v) for k, v in f.items()}
    out["labels"] = labels
    return out


def collate(batch, pad_id):
    """Pad các tensor theo token (labels đệm -100); tensor ảnh nối theo chiều 0, KHÔNG pad."""
    L = max(len(b["input_ids"]) for b in batch)

    def pad(key, val):
        return torch.stack([torch.cat([b[key], torch.full((L - len(b[key]),), val, dtype=b[key].dtype)])
                            for b in batch])

    out = {"input_ids": pad("input_ids", pad_id), "attention_mask": pad("attention_mask", 0),
           "labels": pad("labels", -100)}
    if "mm_token_type_ids" in batch[0]:
        out["mm_token_type_ids"] = pad("mm_token_type_ids", 0)
    out["pixel_values"] = torch.cat([b["pixel_values"] for b in batch], dim=0)
    out["image_grid_thw"] = torch.cat([b["image_grid_thw"] for b in batch], dim=0)
    return out


def lora_targets(setup):
    return LLM_TARGETS if setup in ("E1", "E2") else f"({LLM_TARGETS}|{VIT_TARGETS})"


def apply_setup(model, setup, r=16, alpha=32, dropout=0.05):
    """E1: LoRA LLM. E2: E1 + mở full projector (float32). E3: E2 + LoRA trên ViT."""
    from peft import LoraConfig, get_peft_model
    for p in model.parameters():
        p.requires_grad = False
    model.enable_input_require_grads()
    target = lora_targets(setup)
    matched = [n for n, _ in model.named_modules() if re.fullmatch(target, n)]
    model = get_peft_model(model, LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout,
                                             target_modules=target, bias="none", task_type="CAUSAL_LM"))
    if setup in ("E2", "E3"):
        for n, p in model.named_parameters():
            if any(k in n for k in PROJECTOR_KEYS) and "lora_" not in n:
                p.data = p.data.to(torch.float32)
                p.requires_grad = True
    return model, matched


def count_params(model):
    groups = {"lora_llm": 0, "lora_vit": 0, "projector": 0, "other": 0}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "lora_" in n:
            groups["lora_vit" if ".visual." in n else "lora_llm"] += p.numel()
        elif any(k in n for k in PROJECTOR_KEYS):
            groups["projector"] += p.numel()
        else:
            groups["other"] += p.numel()
    return sum(groups.values()), groups


def projector_state(model):
    return {n: p.detach().cpu() for n, p in model.named_parameters()
            if p.requires_grad and any(k in n for k in PROJECTOR_KEYS)}


def load_trained(adapter_dir, projector_path=None, model_id=MODEL_ID):
    """Model gốc 4-bit -> nạp adapter LoRA -> nạp lại projector đã huấn luyện (E2/E3)."""
    from peft import PeftModel
    model, processor = load_model(model_id)
    model = PeftModel.from_pretrained(model, adapter_dir)
    if projector_path and Path(projector_path).exists():
        sd = torch.load(projector_path, map_location="cpu")
        own = dict(model.named_parameters())
        missing = [k for k in sd if k not in own]
        assert not missing, f"tham số projector không khớp tên: {missing[:3]}"
        with torch.no_grad():
            for k, v in sd.items():
                own[k].copy_(v.to(own[k].dtype))
        print(f"nạp lại {len(sd)} tensor projector ({sum(v.numel() for v in sd.values()):,} tham số)")
    model.eval()
    return model, processor


@torch.no_grad()
def predict(model, processor, entries, prompt, out_path, bs=8, max_new_tokens=256, data_dir="data"):
    model.eval()
    rows, t0 = [], time.time()
    for i in range(0, len(entries), bs):
        chunk = entries[i:i + bs]
        msgs = [build_messages(load_image(e, data_dir), prompt) for e in chunk]
        inputs = processor.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                               return_dict=True, return_tensors="pt",
                                               padding=True).to(model.device)
        with torch.autocast("cuda", dtype=torch.float16):
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        texts = processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        rows += [{"image_id": e["image_id"], "raw": t} for e, t in zip(chunk, texts)]
        print(f"  {len(rows)}/{len(entries)} ảnh, {time.time() - t0:.0f}s", flush=True)
    write_jsonl(out_path, rows)
    return rows
