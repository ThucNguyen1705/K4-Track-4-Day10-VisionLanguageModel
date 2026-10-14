"""Kiểm tra bước 4–5 trước khi huấn luyện.

python -m lab.checks            # cần mạng để tải processor/config; model dựng trên meta device (không cần GPU)

- Ảnh 640x360 -> image_grid_thw -> số token ảnh (phải là 220).
- Giải mã các token có labels != -100: phải đúng JSON đáp án + <|im_end|>.
- Collate 2 ví dụ: pad đúng, tensor ảnh nối theo chiều 0.
- Quy mô tham số từng khối và số module LoRA (252 cho E1/E2, 348 cho E3), số tham số huấn luyện E1/E2/E3.
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from lab.common import MODEL_ID, PROMPT, objects_to_answer, read_jsonl  # noqa: E402
from lab.vlm import (PROJECTOR_KEYS, apply_setup, collate, count_params, encode_example,  # noqa: E402
                     load_image, lora_targets)


def check_encoding(processor, entries):
    tok = processor.tokenizer
    e = entries[0]
    answer = objects_to_answer(e["objects"])
    ex = encode_example(processor, load_image(e), PROMPT, answer)
    thw = ex["image_grid_thw"][0].tolist()
    merge = processor.image_processor.merge_size
    n_vis = thw[0] * thw[1] * thw[2] // merge ** 2
    n_pad = int((ex["input_ids"] == tok.convert_tokens_to_ids("<|image_pad|>")).sum())
    n_ans = int((ex["labels"] != -100).sum())
    decoded = tok.decode(ex["input_ids"][ex["labels"] != -100])
    print(f"ảnh {e['width']}x{e['height']} -> image_grid_thw={thw}, patch={processor.image_processor.patch_size}, "
          f"merge={merge} -> {n_vis} token ảnh (đếm <|image_pad|> = {n_pad})")
    print(f"pixel_values {tuple(ex['pixel_values'].shape)}; tổng {len(ex['input_ids'])} token: "
          f"{len(ex['input_ids']) - n_ans} bị mask (-100), {n_ans} token tính loss")
    print("labels != -100 giải mã:", repr(decoded))
    assert n_vis == 220 and n_pad == 220, "ảnh 640x360 phải ra 220 token"
    assert decoded == answer + "<|im_end|>", "phần tính loss phải đúng là JSON đáp án + <|im_end|>"
    full_text = tok.decode(ex["input_ids"])
    print("chuỗi đầy đủ (rút gọn token ảnh):",
          repr(re.sub(r"(<\|image_pad\|>)+", "<|image_pad|>x220", full_text)))

    b = collate([ex, encode_example(processor, load_image(entries[1]), PROMPT,
                                    objects_to_answer(entries[1]["objects"]))], tok.pad_token_id)
    print("collate:", {k: tuple(v.shape) for k, v in b.items()})
    assert b["pixel_values"].shape[0] == 2 * ex["pixel_values"].shape[0]
    return {"image_grid_thw": thw, "visual_tokens": n_vis, "answer_tokens": n_ans,
            "total_tokens": len(ex["input_ids"])}


def block_sizes(model):
    sizes = {"visual_total": 0, "visual.merger": 0, "visual.deepstack_merger_list": 0,
             "language_model": 0, "lm_head": 0}
    for n, p in model.named_parameters():
        if ".visual." in f".{n}":
            sizes["visual_total"] += p.numel()
            for k in PROJECTOR_KEYS:
                if k in n:
                    sizes[k] += p.numel()
        elif "language_model" in n:
            sizes["language_model"] += p.numel()
        elif "lm_head" in n:
            sizes["lm_head"] += p.numel()
    sizes["vit_without_projector"] = (sizes["visual_total"] - sizes["visual.merger"]
                                      - sizes["visual.deepstack_merger_list"])
    sizes["total"] = sum(p.numel() for p in model.parameters())
    return sizes


def check_skip_modules(model):
    """Bao nhiêu nn.Linear của ViT/projector sẽ bị bitsandbytes lượng tử hoá với từng danh sách skip."""
    from transformers.quantizers.quantizers_utils import should_convert_module
    from lab.vlm import bnb_config
    linears = [n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)]
    out = {}
    for pats in (["visual", "lm_head"], bnb_config().llm_int8_skip_modules):
        conv = [n for n in linears if should_convert_module(n, pats)]
        vis = [n for n in conv if n.startswith("model.visual.")]
        proj = [n for n in vis if any(k in n for k in PROJECTOR_KEYS)]
        out[str(pats)] = {"linear_quantized": len(conv), "visual_quantized": len(vis),
                          "projector_quantized": len(proj), "lm_head_quantized": "lm_head" in conv}
        print(f"llm_int8_skip_modules={pats}: {len(vis)} Linear của ViT bị 4-bit "
              f"(projector {len(proj)}), LLM {len(conv) - len(vis)}, lm_head 4-bit: {'lm_head' in conv}")
    return out


def meta_model():
    from accelerate import init_empty_weights
    from transformers import AutoConfig, Qwen3VLForConditionalGeneration
    cfg = AutoConfig.from_pretrained(MODEL_ID)
    with init_empty_weights():
        return Qwen3VLForConditionalGeneration._from_config(cfg, dtype=torch.float16), cfg


def main():
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    report = {"encoding": check_encoding(processor, read_jsonl("data/splits/train.jsonl"))}

    model, cfg = meta_model()
    report["block_params"] = block_sizes(model)
    report["tie_word_embeddings"] = bool(getattr(cfg, "tie_word_embeddings", None)
                                         or getattr(cfg.text_config, "tie_word_embeddings", False))
    print("tham số từng khối:", json.dumps(report["block_params"]))
    print("deepstack_visual_indexes:", cfg.vision_config.deepstack_visual_indexes,
          "| ViT depth:", cfg.vision_config.depth, "| LLM layers:", cfg.text_config.num_hidden_layers)
    names = [n for n, _ in model.named_modules()]
    report["skip_modules"] = check_skip_modules(model)
    for setup in ("E1", "E3"):
        print(setup, "module khớp regex LoRA:", sum(bool(re.fullmatch(lora_targets(setup), n)) for n in names))

    report["trainable"] = {}
    for setup in ("E1", "E2", "E3"):
        m, _ = meta_model()
        m, matched = apply_setup(m, setup)
        n, groups = count_params(m)
        report["trainable"][setup] = {"lora_modules": len(matched), "trainable_params": n, **groups}
        print(setup, len(matched), "module LoRA,", f"{n:,}", "tham số huấn luyện", groups)
    assert report["trainable"]["E1"]["lora_modules"] == 252
    assert report["trainable"]["E3"]["lora_modules"] == 348
    Path("report").mkdir(exist_ok=True)
    Path("report/checks.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("wrote report/checks.json")


if __name__ == "__main__":
    main()
