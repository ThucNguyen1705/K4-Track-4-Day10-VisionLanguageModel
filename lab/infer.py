"""Suy luận trên một split -> predictions.jsonl cho scripts/score.py (GUIDE bước 3 và 7).

python -m lab.infer --out work/e0.jsonl                                  # E0 zero-shot
python -m lab.infer --adapter work/E2/adapter --projector work/E2/projector.pt --out work/e2.jsonl
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lab.common import prompt_for, read_jsonl  # noqa: E402
from lab.vlm import load_model, load_trained, predict  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter")
    ap.add_argument("--projector")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--limit", type=int, help="chỉ suy luận N ảnh đầu (gỡ lỗi trên val)")
    ap.add_argument("--coord", default="norm1000", choices=["norm1000", "pixel"])
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    entries = read_jsonl(Path(a.data_dir) / "splits" / f"{a.split}.jsonl")[: a.limit]
    if a.adapter:
        model, processor = load_trained(a.adapter, a.projector)
    else:
        model, processor = load_model()
    rows = predict(model, processor, entries, prompt_for(a.coord), a.out, bs=a.bs,
                   max_new_tokens=a.max_new_tokens, data_dir=a.data_dir)
    for r in rows[:5]:
        print(r["image_id"], "->", repr(r["raw"][:300]))


if __name__ == "__main__":
    main()
