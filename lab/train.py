"""Huấn luyện QLoRA cho E1/E2/E3/E4 (GUIDE bước 5–6).

python -m lab.train --setup E1 --out work/E1
python -m lab.train --setup E2 --out work/E4_data_0.1 --train-fraction 0.1
"""
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
import yaml  # noqa: E402

from lab.common import objects_to_answer, prompt_for, read_jsonl  # noqa: E402
from lab.vlm import (apply_setup, collate, count_params, encode_example, load_image,  # noqa: E402
                     load_model, projector_state)


def batches(entries, bs, seed):
    """Vô hạn: xáo lại mỗi epoch (seed cố định), cắt thành batch bs ảnh."""
    rng, epoch = random.Random(seed), 0
    while True:
        order = entries[:]
        rng.shuffle(order)
        for i in range(0, len(order) - bs + 1, bs):
            yield epoch, order[i:i + bs]
        epoch += 1


def save(model, out, setup):
    model.save_pretrained(out / "adapter")          # chỉ chứa LoRA
    if setup in ("E2", "E3"):                        # projector huấn luyện full: KHÔNG nằm trong adapter
        torch.save(projector_state(model), out / "projector.pt")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--setup", required=True, choices=["E1", "E2", "E3"])
    ap.add_argument("--config", help="mặc định configs/e{1,2,3}.yaml theo setup")
    ap.add_argument("--out", required=True)
    ap.add_argument("--r", type=int)
    ap.add_argument("--alpha", type=int)
    ap.add_argument("--train-fraction", type=float, default=1.0)
    ap.add_argument("--coord", default="norm1000", choices=["norm1000", "pixel"])
    ap.add_argument("--max-steps", type=int)
    ap.add_argument("--save-every", type=int, default=100)
    ap.add_argument("--data-dir", default="data")
    a = ap.parse_args(argv)

    cfg = yaml.safe_load(Path(a.config or f"configs/{a.setup.lower()}.yaml").read_text(encoding="utf-8"))
    r = a.r or cfg["lora"]["r"]
    alpha = a.alpha or cfg["lora"]["alpha"]
    max_steps = a.max_steps or cfg["max_steps"]
    bs, accum, lr, seed = cfg["batch_size"], cfg["grad_accum"], cfg["lr"], cfg["seed"]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    random.seed(seed)
    torch.manual_seed(seed)

    entries = read_jsonl(Path(a.data_dir) / "splits" / "train.jsonl")
    if a.train_fraction < 1.0:
        entries = random.Random(seed).sample(entries, round(len(entries) * a.train_fraction))
    print(f"{a.setup}: {len(entries)} ảnh train, r={r} alpha={alpha} lr={lr} bs={bs}x{accum} "
          f"steps={max_steps} coord={a.coord}", flush=True)

    model, processor = load_model()
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model, matched = apply_setup(model, a.setup, r=r, alpha=alpha, dropout=cfg["lora"]["dropout"])
    n_train, groups = count_params(model)
    print(f"LoRA khớp {len(matched)} module; tham số huấn luyện {n_train:,} {groups}", flush=True)
    dtypes = {str(p.dtype) for p in model.parameters() if p.requires_grad}
    assert dtypes == {"torch.float32"}, f"tham số huấn luyện phải float32 (GradScaler): {dtypes}"

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    warm = max(1, round(0.05 * max_steps))

    def lr_lambda(s):
        if s < warm:
            return (s + 1) / warm
        return 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, max_steps - warm)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    scaler = torch.amp.GradScaler("cuda")
    pad_id = processor.tokenizer.pad_token_id
    prompt = prompt_for(a.coord)
    it = batches(entries, bs, seed)

    torch.cuda.reset_peak_memory_stats()
    model.train()
    log_f = open(out / "train_log.jsonl", "w", encoding="utf-8")
    t0 = time.time()
    for step in range(max_steps):
        step_loss = 0.0
        for _ in range(accum):
            epoch, chunk = next(it)
            batch = collate([encode_example(processor, load_image(e, a.data_dir), prompt,
                                            objects_to_answer(e["objects"], a.coord, e["width"], e["height"]))
                             for e in chunk], pad_id)
            batch = {k: v.to(model.device) for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.float16):
                loss = model(**batch).loss / accum
            if not torch.isfinite(loss):
                raise RuntimeError(f"loss không hữu hạn ở bước {step}: {loss.item()} (tràn số fp16?)")
            scaler.scale(loss).backward()
            step_loss += loss.item()
        scaler.unscale_(opt)
        gnorm = torch.nn.utils.clip_grad_norm_(params, 1.0).item()
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        sched.step()
        rec = {"step": step + 1, "epoch": epoch, "loss": round(step_loss, 4), "grad_norm": round(gnorm, 3),
               "lr": sched.get_last_lr()[0], "min": round((time.time() - t0) / 60, 2),
               "vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
        log_f.write(json.dumps(rec) + "\n")
        log_f.flush()
        if step % 10 == 0 or step + 1 == max_steps:
            print(json.dumps(rec), flush=True)
        if (step + 1) % a.save_every == 0 and step + 1 < max_steps:
            save(model, out, a.setup)
    train_minutes = (time.time() - t0) / 60
    save(model, out, a.setup)
    run = {"setup": a.setup, "trainable_params": n_train, "trainable_groups": groups,
           "lora_modules": len(matched), "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
           "train_minutes": round(train_minutes, 1), "r": r, "alpha": alpha, "lr": lr, "batch_size": bs,
           "grad_accum": accum, "max_steps": max_steps, "train_images": len(entries),
           "train_fraction": a.train_fraction, "coord": a.coord, "seed": seed,
           "gpu": torch.cuda.get_device_name(0)}
    (out / "run.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    print("RUN", json.dumps(run), flush=True)


if __name__ == "__main__":
    main()
