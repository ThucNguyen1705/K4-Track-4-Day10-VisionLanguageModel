"""Phần thuần Python (không cần GPU): prompt, nhãn YOLO -> JSON grounding, reward IoU.

Dùng chung cho notebook, train.py, infer.py, grpo.py và unit test.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.labcommon import CLASSES, read_jsonl, write_jsonl  # noqa: E402,F401
from scripts.score import iou_xyxy, parse_output  # noqa: E402

MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"

PROMPT = (
    "Locate road damage in this image. "
    "Detect only objects of classes: pothole, crack, manhole. "
    "Output a JSON list of objects with label and bbox_2d coordinates "
    "normalized on a 0-1000 scale: "
    '[{"label": "...", "bbox_2d": [x1, y1, x2, y2]}]. '
    "If no damage is found, output []."
)

# Chỉ dùng nếu chọn E4 = coord_scale (đổi cả nhãn, prompt và --coord pixel khi chấm).
PROMPT_PIXEL = (
    "Locate road damage in this image. "
    "Detect only objects of classes: pothole, crack, manhole. "
    "Output a JSON list of objects with label and bbox_2d coordinates "
    "in absolute pixels of the {w}x{h} image: "
    '[{{"label": "...", "bbox_2d": [x1, y1, x2, y2]}}]. '
    "If no damage is found, output []."
)


def yolo_to_box(label, yolo, coord="norm1000", width=640, height=360):
    """YOLO [xc, yc, w, h] chuẩn hoá [0,1] -> {"label", "bbox_2d": [x1, y1, x2, y2]} số nguyên."""
    xc, yc, w, h = yolo
    sx, sy = (1000, 1000) if coord == "norm1000" else (width, height)
    x1 = int(round(max(0.0, min(sx, (xc - w / 2) * sx))))
    y1 = int(round(max(0.0, min(sy, (yc - h / 2) * sy))))
    x2 = int(round(max(0.0, min(sx, (xc + w / 2) * sx))))
    y2 = int(round(max(0.0, min(sy, (yc + h / 2) * sy))))
    # box quá mỏng sau khi làm tròn: nới 1 đơn vị để x2 > x1, y2 > y1 (score.py coi box suy biến là hỏng)
    if x2 <= x1:
        x1, x2 = (x1, x1 + 1) if x1 < sx else (x1 - 1, x1)
    if y2 <= y1:
        y1, y2 = (y1, y1 + 1) if y1 < sy else (y1 - 1, y1)
    return {"label": label, "bbox_2d": [x1, y1, x2, y2]}


def objects_to_answer(objects, coord="norm1000", width=640, height=360):
    """Danh sách object của một ảnh -> chuỗi JSON gọn, box sắp theo (y1, x1)."""
    boxes = [yolo_to_box(o["label"], o["yolo"], coord, width, height) for o in objects]
    boxes.sort(key=lambda b: (b["bbox_2d"][1], b["bbox_2d"][0]))
    return json.dumps(boxes, separators=(",", ":"))


def prompt_for(coord="norm1000", width=640, height=360):
    return PROMPT if coord == "norm1000" else PROMPT_PIXEL.format(w=width, h=height)


def box_to_pixel(bbox, width, height):
    """bbox_2d thang 0–1000 -> pixel (để vẽ/kiểm tra quy đổi ngược)."""
    x1, y1, x2, y2 = bbox
    return (x1 / 1000 * width, y1 / 1000 * height, x2 / 1000 * width, y2 / 1000 * height)


def gt_boxes(entry):
    """GT của một ảnh ở thang 0–1000 (float), cùng định dạng parse_output: {"label","bbox"}."""
    out = []
    for o in entry["objects"]:
        xc, yc, w, h = o["yolo"]
        out.append({"label": o["label"], "bbox": (
            (xc - w / 2) * 1000, (yc - h / 2) * 1000, (xc + w / 2) * 1000, (yc + h / 2) * 1000)})
    return out


def compute_reward(completion, gt, iou_thr=0.5):
    """Reward kiểm chứng được cho GRPO (E5).

    0.0 nếu JSON hỏng; 1.0 nếu GT và dự đoán đều rỗng;
    còn lại 0.2 + 0.8 * (tổng IoU các cặp khớp cùng lớp, IoU >= 0.5) / max(n_gt, n_pred, 1).
    `gt` là list {"label", "bbox"} (xem gt_boxes). Mẫu số max(...) phạt box bịa thêm,
    ngưỡng IoU 0.5 triệt tiêu box khổng lồ.
    """
    pred = parse_output(completion)
    if pred is None:
        return 0.0
    if not gt and not pred:
        return 1.0
    total = 0.0
    for lab in {g["label"] for g in gt} | {p["label"] for p in pred}:
        g = [x["bbox"] for x in gt if x["label"] == lab]
        p = [x["bbox"] for x in pred if x["label"] == lab]
        pairs = sorted(((iou_xyxy(a, b), i, j) for i, a in enumerate(g)
                        for j, b in enumerate(p)), reverse=True)
        used_g, used_p = set(), set()
        for v, i, j in pairs:
            if v < iou_thr:
                break
            if i in used_g or j in used_p:
                continue
            used_g.add(i)
            used_p.add(j)
            total += v
    return 0.2 + 0.8 * total / max(len(gt), len(pred), 1)
