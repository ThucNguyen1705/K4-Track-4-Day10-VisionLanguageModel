"""Bootstrap GHÉP CẶP cho hiệu F1 giữa hai setup trên cùng ảnh test (Câu 7, 8, 10).

python -m lab.paired_bootstrap
"""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.labcommon import read_jsonl  # noqa: E402
from scripts.score import _gt_boxes, _match, parse_output  # noqa: E402

M, X = "work/kaggle_out/lab10-main/work", "work/kaggle_out/lab10-e5/work"
PAIRS = [("E2", f"{M}/e2.jsonl", "E1", f"{M}/e1.jsonl"), ("E3", f"{M}/e3.jsonl", "E2", f"{M}/e2.jsonl"),
         ("E3", f"{M}/e3.jsonl", "E1", f"{M}/e1.jsonl"),
         ("E4 data=1.0", f"{M}/e1.jsonl", "E4 data=0.1", f"{M}/e4_data_0.1.jsonl"),
         ("E5", f"{X}/e5.jsonl", "E1r", f"{X}/e1r.jsonl"), ("E1r", f"{X}/e1r.jsonl", "E1", f"{M}/e1.jsonl")]


def rows(entries, path):
    preds = {r["image_id"]: r["raw"] for r in read_jsonl(path)}
    out = []
    for e in entries:
        per, _ = _match(_gt_boxes(e), parse_output(preds[e["image_id"]]) or [])
        out.append(tuple(sum(v[k] for v in per.values()) for k in range(3)))
    return out


def f1(rs, idx):
    tp = sum(rs[i][0] for i in idx)
    fp = sum(rs[i][1] for i in idx)
    fn = sum(rs[i][2] for i in idx)
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


def main(n_boot=2000):
    entries = read_jsonl("data/splits/test.jsonl")
    res, rng, n = [], random.Random(0), len(entries)
    for a, pa, b, pb in PAIRS:
        ra, rb = rows(entries, pa), rows(entries, pb)
        full = list(range(n))
        d = []
        for _ in range(n_boot):
            idx = [rng.randrange(n) for _ in range(n)]
            d.append(f1(ra, idx) - f1(rb, idx))
        d.sort()
        r = {"pair": f"{a} − {b}", "delta_f1": round(f1(ra, full) - f1(rb, full), 3),
             "ci95": [round(d[int(0.025 * n_boot)], 3), round(d[int(0.975 * n_boot) - 1], 3)],
             "p_delta_le_0": round(sum(x <= 0 for x in d) / n_boot, 3)}
        res.append(r)
        print(r)
    Path("report/analysis/paired_bootstrap.json").write_text(json.dumps(res, ensure_ascii=False, indent=1),
                                                            encoding="utf-8")


if __name__ == "__main__":
    main()
