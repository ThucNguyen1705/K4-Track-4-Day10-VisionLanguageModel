"""Biểu đồ nhật ký huấn luyện: loss E1–E4 (SFT) và reward GRPO (E5).

python -m lab.plots --sft work/kaggle_out/lab10-main/work --grpo work/kaggle_out/lab10-e5/work/E5
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# thứ tự màu categorical cố định (không xoay vòng): slot 1..4
SERIES = [("E1", "E1 · LoRA LLM", "#2a78d6"), ("E2", "E2 · + projector", "#eb6834"),
          ("E3", "E3 · + LoRA ViT", "#1baf7a"), ("E4_data_0.1", "E4 · E1 với 10% train", "#eda100")]
INK, MUTED, GRID = "#2b2b2b", "#6b6b6b", "#e6e6e6"


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def smooth(v, k=10):
    return [sum(v[max(0, i - k + 1):i + 1]) / len(v[max(0, i - k + 1):i + 1]) for i in range(len(v))]


def plot_sft(root, out):
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=130)
    _style(ax)
    for key, label, color in SERIES:
        p = Path(root) / key / "train_log.jsonl"
        if not p.exists():
            continue
        rows = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
        if not rows:
            continue
        steps, loss = [r["step"] for r in rows], smooth([r["loss"] for r in rows])
        ax.plot(steps, loss, color=color, linewidth=2, label=label)
        ax.annotate(f"{loss[-1]:.3f}", (steps[-1], loss[-1]), xytext=(4, 0), textcoords="offset points",
                    va="center", fontsize=8, color=INK)
    ax.set_xlabel("bước tối ưu (batch hiệu dụng 8 ảnh)", color=INK, fontsize=9)
    ax.set_ylabel("loss (trung bình trượt 10 bước)", color=INK, fontsize=9)
    ax.set_title("Loss huấn luyện SFT (chỉ trên token trả lời)", color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(out)
    print("wrote", out)


def plot_grpo(run_dir, out):
    rows = [json.loads(x) for x in (Path(run_dir) / "reward_log.jsonl").read_text(encoding="utf-8").splitlines()
            if x.strip()]
    x = [r["call"] for r in rows]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.6), dpi=130)
    for ax in (a1, a2):
        _style(ax)
        ax.set_xlabel("lượt sinh (mỗi lượt 2 ảnh x 4 completion)", color=INK, fontsize=8.5)
    a1.plot(x, [r["reward_mean"] for r in rows], color="#2a78d6", linewidth=1, alpha=0.35)
    a1.plot(x, smooth([r["reward_mean"] for r in rows]), color="#2a78d6", linewidth=2)
    a1.set_title("Reward IoU trung bình", color=INK, fontsize=10, loc="left")
    for key, label, color in (("invalid_rate", "JSON hỏng", "#2a78d6"), ("empty_rate", "trả []", "#eb6834")):
        a2.plot(x, smooth([r[key] for r in rows]), color=color, linewidth=2, label=label)
    a2.set_title("Tỉ lệ dấu hiệu hacking (trượt 10)", color=INK, fontsize=10, loc="left")
    a2.set_ylim(0, 1)
    a2.legend(frameon=False, fontsize=8.5, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(out)
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft")
    ap.add_argument("--grpo")
    a = ap.parse_args()
    Path("report/figures").mkdir(parents=True, exist_ok=True)
    if a.sft:
        plot_sft(a.sft, "report/figures/loss_sft.png")
    if a.grpo:
        plot_grpo(a.grpo, "report/figures/grpo_reward.png")


if __name__ == "__main__":
    main()
