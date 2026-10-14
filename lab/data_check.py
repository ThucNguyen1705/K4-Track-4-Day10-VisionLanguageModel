"""Bước 1–2: kiểm tra split theo phiên quay và quy đổi nhãn YOLO -> JSON 0–1000 -> pixel.

python -m lab.data_check
"""
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lab.common import CLASSES, box_to_pixel, objects_to_answer, read_jsonl, yolo_to_box  # noqa: E402


def main(data_dir="data"):
    sd = Path(data_dir) / "splits"
    meta = json.loads((sd / "meta.json").read_text(encoding="utf-8"))
    print("dataset_version:", meta["dataset_version"], "| salt:", meta["salt"])
    sessions = {}
    for s in ("train", "val", "test"):
        rows = read_jsonl(sd / f"{s}.jsonl")
        sessions[s] = collections.Counter(r["session"] for r in rows)
        nb = collections.Counter(o["label"] for r in rows for o in r["objects"])
        print(f"{s:5s} {len(rows):5d} ảnh | " + " ".join(f"{c}={nb[c]}" for c in CLASSES)
              + f" | {len(sessions[s])} phiên")
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        inter = set(sessions[a]) & set(sessions[b])
        assert not inter, f"rò rỉ phiên giữa {a} và {b}: {inter}"
    print("Không phiên nào xuất hiện ở hai split.")
    for s in ("val", "test"):
        print(f"  phiên {s}:", dict(sessions[s]))

    # quy đổi YOLO -> JSON 0–1000 -> pixel, so từng object với toạ độ pixel gốc (đã kẹp vào khung)
    worst, n_box, n_out = 0.0, 0, 0
    for s in ("train", "val", "test"):
        for r in read_jsonl(sd / f"{s}.jsonl"):
            W, H = r["width"], r["height"]
            ans = json.loads(objects_to_answer(r["objects"]))
            assert sorted(b["label"] for b in ans) == sorted(o["label"] for o in r["objects"])
            assert ans == sorted(ans, key=lambda b: (b["bbox_2d"][1], b["bbox_2d"][0]))
            for o in r["objects"]:
                xc, yc, w, h = o["yolo"]
                raw = ((xc - w / 2) * W, (yc - h / 2) * H, (xc + w / 2) * W, (yc + h / 2) * H)
                n_box += 1
                n_out += any(v < 0 for v in raw) or raw[2] > W or raw[3] > H
                ref = (min(max(raw[0], 0), W), min(max(raw[1], 0), H),
                       min(max(raw[2], 0), W), min(max(raw[3], 0), H))
                got = box_to_pixel(yolo_to_box(o["label"], o["yolo"])["bbox_2d"], W, H)
                worst = max(worst, max(abs(u - v) for u, v in zip(ref, got)))
    print(f"{n_box} box, {n_out} box vượt mép ảnh (bị kẹp về [0,1000]).")
    print(f"Sai số quy đổi YOLO -> 0–1000 -> pixel lớn nhất: {worst:.3f} px "
          f"(1 đơn vị = 0.64 px theo x, 0.36 px theo y)")
    ex = read_jsonl(sd / "train.jsonl")[0]
    print("Ví dụ đáp án:", ex["image_id"], objects_to_answer(ex["objects"]))


if __name__ == "__main__":
    main()
