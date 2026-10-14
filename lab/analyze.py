"""Phân tích lỗi một file dự đoán trên test (Câu 2, 3, 9) và chọn ảnh minh hoạ.

python -m lab.analyze --pred work/e0.jsonl --name E0
python -m lab.analyze --pred work/e2.jsonl --name E2 --figs     # vẽ ảnh đúng/sai vào report/figures/
"""
import argparse
import collections
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lab.common import CLASSES, gt_boxes, iou_xyxy, parse_output, read_jsonl  # noqa: E402
from scripts.score import evaluate  # noqa: E402


def match_image(gt, pred, thr=0.5):
    """Ghép tham lam như score.py; trả (tp_pairs, fp_idx, fn_idx)."""
    tp, used_g, used_p = [], set(), set()
    for lab in {g["label"] for g in gt} | {p["label"] for p in pred}:
        pairs = sorted(((iou_xyxy(g["bbox"], p["bbox"]), i, j) for i, g in enumerate(gt)
                        for j, p in enumerate(pred) if g["label"] == lab and p["label"] == lab), reverse=True)
        for v, i, j in pairs:
            if v < thr:
                break
            if i in used_g or j in used_p:
                continue
            used_g.add(i)
            used_p.add(j)
            tp.append((i, j, v))
    return tp, [j for j in range(len(pred)) if j not in used_p], [i for i in range(len(gt)) if i not in used_g]


def area(b):
    return (b[2] - b[0]) * (b[3] - b[1]) / 1e6


def size_bucket(a):
    return "nhỏ <0.5%" if a < 0.005 else ("vừa 0.5–2%" if a < 0.02 else "lớn ≥2%")


def analyze(entries, preds, coord="norm1000"):
    raws = [preds[e["image_id"]] for e in entries]
    out = {"n": len(entries)}
    # định dạng output
    out["format"] = {
        "invalid_json": sum(parse_output(r) is None for r in raws),
        "markdown_fence": sum("```" in r for r in raws),
        "extra_text": sum(not r.strip().startswith(("[", "```")) for r in raws),
        "empty_list": sum(parse_output(r) == [] for r in raws),
        "chars_mean": round(st.mean(len(r) for r in raws), 1),
    }
    fp_kind = collections.Counter()
    recall_size = collections.defaultdict(lambda: [0, 0])
    per_img, pred_labels, n_pred, full_width = [], collections.Counter(), [], 0
    for e, r in zip(entries, raws):
        gt = gt_boxes(e)
        pred = parse_output(r) or []
        if coord == "pixel":
            pred = [{"label": p["label"], "bbox": (p["bbox"][0] / e["width"] * 1000, p["bbox"][1] / e["height"] * 1000,
                                                    p["bbox"][2] / e["width"] * 1000, p["bbox"][3] / e["height"] * 1000)}
                    for p in pred]
        n_pred.append(len(pred))
        pred_labels.update(p["label"] for p in pred)
        full_width += sum(p["bbox"][2] - p["bbox"][0] >= 900 for p in pred)
        tp, fp, fn = match_image(gt, pred)
        for i in range(len(gt)):
            key = (gt[i]["label"], size_bucket(area(gt[i]["bbox"])))
            recall_size[key][1] += 1
        for i, _, _ in tp:
            recall_size[(gt[i]["label"], size_bucket(area(gt[i]["bbox"])))][0] += 1
        for j in fp:
            p = pred[j]
            best_same = max([iou_xyxy(g["bbox"], p["bbox"]) for g in gt if g["label"] == p["label"]] or [0])
            best_other = max([iou_xyxy(g["bbox"], p["bbox"]) for g in gt if g["label"] != p["label"]] or [0])
            if p["label"] not in CLASSES:
                fp_kind["nhãn ngoài 3 lớp"] += 1
            elif best_other >= 0.5:
                fp_kind["sai lớp (IoU≥0.5 với GT lớp khác)"] += 1
            elif best_same >= 0.1:
                fp_kind["lệch vị trí (0.1≤IoU<0.5 cùng lớp)"] += 1
            elif area(p["bbox"]) >= 0.25:
                fp_kind["box khổng lồ (≥25% khung)"] += 1
            else:
                fp_kind["bịa box (không chồng GT)"] += 1
        per_img.append({"image_id": e["image_id"], "gt": len(gt), "pred": len(pred), "tp": len(tp),
                        "fp": len(fp), "fn": len(fn), "valid": parse_output(r) is not None})
    out["pred_boxes_per_image"] = round(st.mean(n_pred), 2)
    out["gt_boxes_per_image"] = round(st.mean(len(e["objects"]) for e in entries), 2)
    out["pred_labels"] = dict(pred_labels.most_common())
    out["boxes_width_ge_900"] = full_width
    out["fp_kinds"] = dict(fp_kind.most_common())
    out["recall_by_size"] = {f"{c} | {s}": f"{v[0]}/{v[1]} = {v[0] / v[1]:.2f}"
                             for (c, s), v in sorted(recall_size.items())}
    gt_area = collections.defaultdict(list)
    for e in entries:
        for g in gt_boxes(e):
            gt_area[g["label"]].append(area(g["bbox"]) * 100)
    out["gt_area_median_pct"] = {c: round(st.median(v), 2) for c, v in gt_area.items()}
    return out, per_img


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--coord", default="norm1000")
    ap.add_argument("--figs", action="store_true")
    a = ap.parse_args(argv)
    entries = read_jsonl(f"data/splits/{a.split}.jsonl")
    preds = {r["image_id"]: r["raw"] for r in read_jsonl(a.pred)}
    entries = [e for e in entries if e["image_id"] in preds]
    out, per_img = analyze(entries, preds, a.coord)
    out["score"] = {k: v for k, v in evaluate(entries, preds, coord=a.coord, n_boot=200).items()
                    if k in ("parse_valid_rate", "micro", "fp_per_image")}
    Path("report/analysis").mkdir(parents=True, exist_ok=True)
    Path(f"report/analysis/{a.name.lower()}.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "score"}, ensure_ascii=False, indent=1))
    if a.figs:
        from lab.viz import main as viz
        good = sorted((r for r in per_img if r["fp"] == 0 and r["fn"] == 0 and r["gt"] >= 2),
                      key=lambda r: -r["gt"])[:2] or sorted(per_img, key=lambda r: -r["tp"])[:2]
        fns = sorted((r for r in per_img if r["fn"] >= 2 and r["tp"] == 0), key=lambda r: -r["fn"])[:2]
        fps = sorted((r for r in per_img if r["fp"] >= 2), key=lambda r: -r["fp"])[:2]
        for tag, rows in (("dung", good), ("bo_sot", fns), ("du_thua", fps)):
            if rows:
                viz(["pred", "--pred", a.pred, "--split", a.split, "--ids", ",".join(r["image_id"] for r in rows),
                     "--out", f"report/figures/{a.name.lower()}_{tag}.jpg"])


if __name__ == "__main__":
    main()
