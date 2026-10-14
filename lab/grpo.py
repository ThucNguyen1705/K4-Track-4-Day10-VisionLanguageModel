"""E5 (tuỳ chọn): GRPO với reward IoU kiểm chứng được, khởi tạo từ adapter SFT tốt nhất (GUIDE bước 9).

python -m lab.grpo --init work/E1 --out work/E5
"""
import argparse
import json
import random
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
import yaml  # noqa: E402

from lab.common import PROMPT, compute_reward, gt_boxes, parse_output, read_jsonl  # noqa: E402
from lab.vlm import count_params, load_image, load_model  # noqa: E402


class RewardIoU:
    """Reward cho GRPOTrainer + nhật ký dấu hiệu reward hacking (trả [], box khổng lồ, JSON hỏng)."""
    __name__ = "iou_reward"

    def __init__(self, log_path):
        self.log = open(log_path, "w", encoding="utf-8")
        self.calls = 0

    def __call__(self, completions, gt, **kw):
        texts = [c[-1]["content"] if isinstance(c, list) else c for c in completions]
        rewards, n_pred, n_empty, n_bad, n_giant = [], [], 0, 0, 0
        for t, g in zip(texts, gt):
            g = [{"label": x["label"], "bbox": tuple(x["bbox"])} for x in json.loads(g)]
            rewards.append(compute_reward(t, g))
            p = parse_output(t)
            if p is None:
                n_bad += 1
                continue
            n_pred.append(len(p))
            n_empty += not p
            n_giant += sum((b["bbox"][2] - b["bbox"][0]) * (b["bbox"][3] - b["bbox"][1]) > 0.25e6 for b in p)
        self.calls += 1
        n = len(texts)
        rec = {"call": self.calls, "n": n, "reward_mean": sum(rewards) / n,
               "invalid_rate": n_bad / n, "empty_rate": n_empty / n, "giant_boxes": n_giant,
               "pred_boxes_mean": sum(n_pred) / max(1, len(n_pred)),
               "gt_boxes_mean": sum(len(json.loads(g)) for g in gt) / n,
               "sample": texts[0][:200]}
        self.log.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self.log.flush()
        return rewards


def main(argv=None):
    from datasets import Dataset
    from peft import PeftModel
    from trl import GRPOConfig, GRPOTrainer

    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True, help="thư mục SFT có adapter/ (và projector.pt nếu E2/E3)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-steps", type=int)
    ap.add_argument("--max-boxes", type=int, default=4,
                    help="chỉ dùng ảnh train có <= N box để đáp án vừa max_completion_length=128")
    a = ap.parse_args(argv)
    cfg = yaml.safe_load(Path("configs/e5.yaml").read_text(encoding="utf-8"))["grpo"]
    init, out = Path(a.init), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    random.seed(42)
    torch.manual_seed(42)

    model, processor = load_model()
    model = PeftModel.from_pretrained(model, str(init / "adapter"), is_trainable=True)
    proj = init / "projector.pt"
    if proj.exists():   # projector đã huấn luyện ở SFT: nạp lại, giữ đóng băng trong GRPO
        own = dict(model.named_parameters())
        with torch.no_grad():
            for k, v in torch.load(proj, map_location="cpu").items():
                own[k].copy_(v.to(own[k].dtype))
        shutil.copy(proj, out / "projector.pt")

    entries = [e for e in read_jsonl("data/splits/train.jsonl") if len(e["objects"]) <= a.max_boxes]
    ds = Dataset.from_list([{"prompt": [{"role": "user", "content": PROMPT}],
                             "image": load_image(e), "gt": json.dumps(gt_boxes(e))} for e in entries])
    print(f"GRPO: {len(ds)} ảnh train (<= {a.max_boxes} box), khởi tạo từ {init}", flush=True)

    args = GRPOConfig(
        output_dir=str(out / "trainer"), max_steps=a.max_steps or cfg["max_steps"],
        learning_rate=cfg["learning_rate"], beta=cfg["beta"], num_generations=cfg["num_generations"],
        max_completion_length=cfg["max_completion_length"], loss_type="grpo", temperature=1.0,
        per_device_train_batch_size=4, gradient_accumulation_steps=2,      # 2 ảnh x 4 lần sinh mỗi bước
        fp16=True, bf16=False, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=1, save_strategy="no", report_to="none", seed=42, remove_unused_columns=False,
    )
    reward = RewardIoU(out / "reward_log.jsonl")
    trainer = GRPOTrainer(model=model, reward_funcs=reward, args=args, train_dataset=ds,
                          processing_class=processor)
    # TRL ép adapter QLoRA sang bf16; T4 không có bf16 gốc và với lr 5e-6 bước cập nhật nhỏ hơn
    # độ phân giải của bf16 (~3.9e-3 tương đối) -> đưa adapter về fp32, forward vẫn fp16 (autocast).
    for p in trainer.model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    n_train, groups = count_params(trainer.model)
    print(f"tham số huấn luyện GRPO: {n_train:,} {groups}", flush=True)

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    trainer.train()
    minutes = (time.time() - t0) / 60
    trainer.model.save_pretrained(out / "adapter")
    (out / "grpo_log.json").write_text(json.dumps(trainer.state.log_history, indent=1), encoding="utf-8")
    run = {"setup": "E5-GRPO", "init": str(init), "trainable_params": n_train,
           "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
           "train_minutes": round(minutes, 1), "train_images": len(ds), **cfg,
           "per_device_train_batch_size": 4, "gradient_accumulation_steps": 2, "loss_type": "grpo"}
    (out / "run.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    print("RUN", json.dumps(run), flush=True)


if __name__ == "__main__":
    main()
