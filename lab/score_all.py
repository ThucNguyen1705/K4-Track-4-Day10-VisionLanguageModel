"""Chấm mọi dự đoán bằng trọng tài scripts/score.py, meta lấy thẳng từ run.json của từng lần huấn luyện.

python -m lab.score_all --root work/kaggle_out/lab10-main/work [--e5-root work/kaggle_out/lab10-e5/work]
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

# (tên evidence, tên hiển thị, file dự đoán, thư mục run.json hoặc None)
JOBS = [
    ("e0", "E0", "e0.jsonl", None),
    ("e1", "E1", "e1.jsonl", "E1"),
    ("e2", "E2", "e2.jsonl", "E2"),
    ("e3", "E3", "e3.jsonl", "E3"),
    ("e4_data_0.1", "E4 data_size=0.1", "e4_data_0.1.jsonl", "E4_data_0.1"),
    ("e4_data_1.0", "E4 data_size=1.0", "e1.jsonl", "E1"),   # 100% train = chính run E1 (setup gốc của E4)
]


def score(pred, name, out, run_dir=None):
    cmd = [sys.executable, "scripts/score.py", "--pred", str(pred), "--name", name, "--out", out]
    if run_dir:
        run = json.loads((Path(run_dir) / "run.json").read_text(encoding="utf-8"))
        for k in ("trainable_params", "peak_vram_gb", "train_minutes"):
            cmd += ["--meta", f"{k}={run[k]}"]
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="work/kaggle_out/lab10-main/work")
    ap.add_argument("--e5-root")
    a = ap.parse_args()
    root = Path(a.root)
    for key, name, pred, run in JOBS:
        if (root / pred).exists():
            score(root / pred, name, f"report/evidence/{key}.json", root / run if run else None)
    if a.e5_root:
        r = Path(a.e5_root)
        if (r / "e5.jsonl").exists():
            score(r / "e5.jsonl", "E5 GRPO", "report/evidence/e5.json", r / "E5")
        if (r / "e1r.jsonl").exists():   # adapter SFT khởi tạo của GRPO (chạy lại E1 trong cùng kernel)
            score(r / "e1r.jsonl", "E1r (init GRPO)", "report/analysis/e1r_score.json", r / "E1r")


if __name__ == "__main__":
    main()
