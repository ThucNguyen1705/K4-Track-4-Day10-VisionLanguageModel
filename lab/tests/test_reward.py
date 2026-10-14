"""Unit test cho reward IoU của GRPO (E5). Chạy: python -m pytest lab/tests -q"""
import json

import pytest

from lab.common import compute_reward, gt_boxes

GT1 = [{"label": "pothole", "bbox": (100.0, 200.0, 300.0, 400.0)}]


def _out(*boxes):
    return json.dumps([{"label": lab, "bbox_2d": list(bb)} for lab, bb in boxes])


def test_perfect():
    assert compute_reward(_out(("pothole", (100, 200, 300, 400))), GT1) == pytest.approx(1.0)


def test_broken_json():
    assert compute_reward('[{"label": "pothole", "bbox_2d": [100, 200', GT1) == 0.0
    assert compute_reward("Based on the image, there is a pothole.", GT1) == 0.0


def test_wrong_label():
    assert compute_reward(_out(("crack", (100, 200, 300, 400))), GT1) == pytest.approx(0.2)


def test_extra_box_penalised():
    out = _out(("pothole", (100, 200, 300, 400)), ("pothole", (600, 600, 700, 700)))
    assert compute_reward(out, GT1) == pytest.approx(0.6)


def test_both_empty():
    assert compute_reward("[]", []) == 1.0


def test_empty_prediction_when_gt_exists_gets_format_reward_only():
    assert compute_reward("[]", GT1) == pytest.approx(0.2)


def test_giant_box_is_not_rewarded():
    gt = GT1 + [{"label": "pothole", "bbox": (700.0, 700.0, 760.0, 760.0)}]
    assert compute_reward(_out(("pothole", (0, 0, 1000, 1000))), gt) == pytest.approx(0.2)


def test_partial_iou_and_threshold():
    # IoU ≈ 0.667 (khớp, được tính); IoU = 0.45 < 0.5 (không khớp)
    good = _out(("pothole", (100, 200, 300, 333.3333333)))
    assert compute_reward(good, GT1) == pytest.approx(0.2 + 0.8 * (133.3333333 / 200), rel=1e-6)
    bad = _out(("pothole", (100, 200, 300, 290)))
    assert compute_reward(bad, GT1) == pytest.approx(0.2)


def test_one_prediction_cannot_match_two_gt():
    gt = GT1 + [{"label": "pothole", "bbox": (100.0, 200.0, 300.0, 400.0)}]
    assert compute_reward(_out(("pothole", (100, 200, 300, 400))), gt) == pytest.approx(0.6)


def test_markdown_fence_is_accepted_like_score_py():
    fenced = "```json\n" + _out(("pothole", (100, 200, 300, 400))) + "\n```"
    assert compute_reward(fenced, GT1) == pytest.approx(1.0)


def test_gt_boxes_from_yolo():
    e = {"objects": [{"label": "crack", "yolo": [0.5, 0.5, 0.2, 0.1]}]}
    assert gt_boxes(e) == [{"label": "crack", "bbox": pytest.approx((400.0, 450.0, 600.0, 550.0))}]
