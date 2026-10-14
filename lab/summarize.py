"""Bảng tổng hợp lấy thẳng từ report/evidence/*.json (để REPORT.md khớp 100% với bằng chứng).

python -m lab.summarize
"""
import json
import sys
from pathlib import Path

SETUPS = {"e0": "zero-shot", "e1": "LoRA LLM", "e2": "+ projector", "e3": "+ LoRA ViT", "e5": "GRPO (IoU)"}


def row(key, d):
    m, pc, meta = d["micro"], d["per_class"], d.get("meta", {})
    setup = SETUPS.get(key, key.replace("e4_", "E4 · ").replace("_", " = "))
    params = meta.get("trainable_params", 0)
    return (f"| {d['name']} | {setup} | {params:,} | {meta.get('peak_vram_gb', '–')} | "
            f"{meta.get('train_minutes', '–')} | {m['f1']:.3f} | [{d['f1_ci95'][0]:.3f}, {d['f1_ci95'][1]:.3f}] | "
            f"{d['fp_per_image']:.2f} | {pc['pothole']['f1']:.3f} | {pc['crack']['f1']:.3f} | "
            f"{pc['manhole']['f1']:.3f} |")


def main(ev_dir="report/evidence"):
    files = sorted(Path(ev_dir).glob("e*.json"), key=lambda p: (p.stem[:2], p.stem))
    print("| Mã | Setup | Tham số huấn luyện | VRAM đỉnh (GB) | Phút | F1 | CI95 | FP/ảnh | "
          "F1 pothole | F1 crack | F1 manhole |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for p in files:
        print(row(p.stem, json.loads(p.read_text(encoding="utf-8"))))
    print()
    print("| Mã | parse_valid | P | R | TP | FP | FN | mean IoU | missing |")
    print("|---|---|---|---|---|---|---|---|---|")
    for p in files:
        d = json.loads(p.read_text(encoding="utf-8"))
        m = d["micro"]
        print(f"| {d['name']} | {d['parse_valid_rate']:.3f} | {m['precision']:.3f} | {m['recall']:.3f} | "
              f"{m['tp']} | {m['fp']} | {m['fn']} | {d['mean_iou']:.3f} | {d['missing_predictions']} |")


if __name__ == "__main__":
    main(*sys.argv[1:])
