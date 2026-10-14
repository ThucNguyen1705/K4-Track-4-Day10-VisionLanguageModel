"""Chạy các bước GPU của lab dưới dạng kernel Kaggle chạy theo lô (T4 x2, không cần Colab).

Kernel nhận mã (lab/, scripts/, configs/) đóng gói base64, tự cài thư viện đã ghim, chia lại split
từ dataset gắn sẵn (cùng dataset_version), rồi chạy hai "làn" lệnh song song: làn i dùng GPU i.
Kết quả nằm ở /kaggle/working/{work,logs} và tải về bằng `fetch`. Không có khoá nào trong kernel.

python tools/kaggle_job.py push  --slug lab10-smoke --lane "lệnh 1" --lane "lệnh 2" [--from-kernel slug]
python tools/kaggle_job.py fetch --slug lab10-smoke [--pattern REGEX]
"""
import argparse
import base64
import io
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET = "lorenzoarcioni/road-damage-dataset-potholes-cracks-and-manholes"
PACK = ["lab", "scripts", "configs", "requirements-colab.txt"]
SUFFIX = "-2a202603014"

KERNEL = r'''
import base64, glob, io, json, os, shutil, subprocess, sys, tarfile, time
W = "/kaggle/working"
os.chdir(W)
CODE = "__CODE__"
LANES = json.loads(__LANES__)
FROM = json.loads(__FROM__)
tarfile.open(fileobj=io.BytesIO(base64.b64decode(CODE)), mode="r:gz").extractall(W, filter="data")
os.makedirs("logs", exist_ok=True)
os.makedirs("work", exist_ok=True)

def sh(cmd):
    print(">>>", cmd, flush=True)
    return subprocess.run(cmd, shell=True).returncode

sh("nvidia-smi --query-gpu=name,memory.total --format=csv; python -V")
sh(f"{sys.executable} -m pip install -q 'transformers==5.19.0' 'peft==0.21.2' 'trl==1.9.2' "
   "'accelerate==1.15.0' 'bitsandbytes==0.50.2'")
sh(f"{sys.executable} -m pip uninstall -y -q torchao")  # torchao 0.10 của Kaggle làm peft lỗi khi LoRA trên nn.Linear fp16
sh("python -c 'import torch,transformers,peft,trl,bitsandbytes as b;"
   "print(torch.__version__,torch.version.cuda,transformers.__version__,peft.__version__,trl.__version__,b.__version__)'")
root = next(os.path.dirname(p) for p in glob.glob("/kaggle/input/**/labels-YOLO", recursive=True))
assert sh(f"python scripts/prepare_data.py --source {root} --out data --work /tmp/w") == 0
for slug in FROM:  # kết quả của kernel trước (adapter, projector) để dùng tiếp
    for src in glob.glob(f"/kaggle/input/**/{slug}/work", recursive=True) + glob.glob(f"/kaggle/input/{slug}/work"):
        sh(f"cp -rn {src}/. work/")
sh("ls -R work | head -50")
sh("python -c \"from huggingface_hub import snapshot_download as s; print(s('Qwen/Qwen3-VL-4B-Instruct'))\"")

procs = []
for i, cmds in enumerate(LANES):
    script = " ; ".join(f"echo '>>> {c}'; {c}; echo \"<<< rc=$? ({c[:60]})\"" for c in cmds)
    procs.append(subprocess.Popen(["bash", "-c", script], stdout=open(f"logs/lane{i}.log", "w"),
                                  stderr=subprocess.STDOUT,
                                  env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(i), PYTHONUNBUFFERED="1")))
t0 = time.time()
while any(p.poll() is None for p in procs):
    time.sleep(120)
    for i in range(len(procs)):
        lines = open(f"logs/lane{i}.log", errors="replace").read().replace("\r", "\n").splitlines()
        keep = [l for l in lines if l.startswith((">>>", "<<<", "{", "RUN", "LoRA", "E", "Traceback"))
                or "Error" in l or "error" in l]
        print(f"[{(time.time() - t0) / 60:.0f} phút] lane{i}:", " | ".join((keep or lines)[-3:])[-600:], flush=True)
for i in range(len(procs)):
    print(f"===== lane{i} (rc={procs[i].returncode}) tail =====")
    print("\n".join(open(f"logs/lane{i}.log", errors="replace").read().replace("\r", "\n").splitlines()[-60:]))
for d in ("data", "lab", "scripts", "configs"):
    shutil.rmtree(d, ignore_errors=True)
sh("du -sh work/* logs 2>/dev/null")
'''


def pack():
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name in PACK:
            tar.add(ROOT / name, arcname=name,
                    filter=lambda t: None if "__pycache__" in t.name or t.name.endswith(".pyc") else t)
    return base64.b64encode(buf.getvalue()).decode()


def kaggle(*args, **kw):
    return subprocess.run(["kaggle", *args], check=True, **kw)


def user():
    out = subprocess.run(["kaggle", "config", "view"], capture_output=True, text=True).stdout
    return next(ln.split(":", 1)[1].strip() for ln in out.splitlines() if "username" in ln)


def push(a):
    lanes = [[c.strip() for c in lane.split(";;") if c.strip()] for lane in a.lane]
    kdir = ROOT / "work" / "kaggle_jobs" / a.slug
    kdir.mkdir(parents=True, exist_ok=True)
    src = (KERNEL.replace("__CODE__", pack())
                 .replace("__LANES__", repr(json.dumps(lanes)))
                 .replace("__FROM__", repr(json.dumps([s + SUFFIX for s in a.from_kernel]))))
    (kdir / "job.py").write_text(src, encoding="utf-8")
    u = user()
    meta = {"id": f"{u}/{a.slug}{SUFFIX}", "title": f"{a.slug}{SUFFIX}", "code_file": "job.py",
            "language": "python", "kernel_type": "script", "is_private": True,
            "enable_gpu": True, "enable_tpu": False, "enable_internet": True,
            "machine_shape": "NvidiaTeslaT4", "dataset_sources": [DATASET],
            "kernel_sources": [f"{u}/{s}{SUFFIX}" for s in a.from_kernel],
            "competition_sources": [], "model_sources": []}
    (kdir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("lanes:", json.dumps(lanes, ensure_ascii=False, indent=1))
    kaggle("kernels", "push", "-p", str(kdir), "-t", str(a.timeout))


def fetch(a):
    dest = ROOT / "work" / "kaggle_out" / a.slug
    dest.mkdir(parents=True, exist_ok=True)
    args = ["kernels", "output", f"{user()}/{a.slug}{SUFFIX}", "-p", str(dest), "-o"]
    if a.pattern:
        args += ["--file-pattern", a.pattern]
    kaggle(*args)
    print("->", dest)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("op", choices=["push", "fetch"])
    ap.add_argument("--slug", required=True)
    ap.add_argument("--lane", action="append", default=[], help="các lệnh của một làn, ngăn bằng ';;'")
    ap.add_argument("--from-kernel", action="append", default=[])
    ap.add_argument("--timeout", type=int, default=6 * 3600)
    ap.add_argument("--pattern", default=r".*\.(jsonl|json|log|txt)$")
    a = ap.parse_args()
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    push(a) if a.op == "push" else fetch(a)


if __name__ == "__main__":
    sys.exit(main())
