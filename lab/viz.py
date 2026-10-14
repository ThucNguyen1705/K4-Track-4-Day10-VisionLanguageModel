"""Vẽ box GT / dự đoán lên ảnh dashcam.

python -m lab.viz gt   --split train --n 6 --out report/figures/gt_samples.jpg
python -m lab.viz pred --pred work/e1.jsonl --ids id1,id2 --out report/figures/e1_x.jpg
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from lab.common import gt_boxes, parse_output, read_jsonl  # noqa: E402

COLORS = {"pothole": (230, 57, 70), "crack": (255, 183, 3), "manhole": (58, 134, 255)}
PRED_COLOR = (0, 230, 118)


def _font(size=14):
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw(entry, data_dir="data", pred=None, title=None):
    """GT: khung màu theo lớp, nét liền. Dự đoán (nếu có): khung xanh lá, nhãn 'P:'."""
    im = Image.open(Path(data_dir) / entry["image_path"]).convert("RGB")
    W, H = im.size
    d = ImageDraw.Draw(im)
    f = _font(13)
    for g in gt_boxes(entry):
        x1, y1, x2, y2 = (g["bbox"][0] * W / 1000, g["bbox"][1] * H / 1000,
                          g["bbox"][2] * W / 1000, g["bbox"][3] * H / 1000)
        d.rectangle([x1, y1, x2, y2], outline=COLORS.get(g["label"], (255, 255, 255)), width=2)
        d.text((x1 + 2, max(0, y1 - 14)), g["label"], fill=COLORS.get(g["label"]), font=f)
    for p in pred or []:
        x1, y1, x2, y2 = (p["bbox"][0] * W / 1000, p["bbox"][1] * H / 1000,
                          p["bbox"][2] * W / 1000, p["bbox"][3] * H / 1000)
        d.rectangle([x1, y1, x2, y2], outline=PRED_COLOR, width=2)
        d.text((x1 + 2, min(H - 14, y2 + 1)), "P:" + p["label"], fill=PRED_COLOR, font=f)
    if title:
        d.rectangle([0, 0, W, 18], fill=(0, 0, 0))
        d.text((4, 2), title, fill=(255, 255, 255), font=f)
    return im


def grid(images, cols=2, pad=4):
    w, h = images[0].size
    rows = (len(images) + cols - 1) // cols
    out = Image.new("RGB", (cols * w + (cols - 1) * pad, rows * h + (rows - 1) * pad), (255, 255, 255))
    for k, im in enumerate(images):
        out.paste(im, ((k % cols) * (w + pad), (k // cols) * (h + pad)))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["gt", "pred"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pred")
    ap.add_argument("--ids", help="danh sách image_id, cách nhau dấu phẩy")
    ap.add_argument("--cols", type=int, default=2)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    entries = read_jsonl(Path(a.data_dir) / "splits" / f"{a.split}.jsonl")
    by_id = {e["image_id"]: e for e in entries}
    if a.ids:
        chosen = [by_id[i] for i in a.ids.split(",")]
    else:
        chosen = random.Random(a.seed).sample(entries, a.n)
    preds = {}
    if a.pred:
        preds = {r["image_id"]: r["raw"] for r in read_jsonl(a.pred)}
    ims = []
    for e in chosen:
        p = parse_output(preds[e["image_id"]]) if a.pred else None
        n_gt = len(e["objects"])
        t = f"{e['image_id']} | GT={n_gt}" + (f" | pred={len(p) if p is not None else 'INVALID'}"
                                              if a.pred else "")
        ims.append(draw(e, a.data_dir, p, t))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    grid(ims, a.cols).save(a.out, quality=88)
    print("wrote", a.out, json.dumps([e["image_id"] for e in chosen]))


if __name__ == "__main__":
    main()
