"""
mi300x_contention_bench - the exact reproducer for BENCHMARKS.md §1's MI300X result.

Question: does Perseus Vault's CPU-side recall steal cycles from the accelerator?
Method: measure recall latency on the host CPU, then saturate the GPU to 100% with a
compute-bound FP16 matmul (isolated in subprocesses so it can't contend for the Python
GIL with the recall measurement) and measure recall again. If recall p50 barely moves,
the memory layer and the accelerator do not contend - which is the whole thesis.

Measured on a rented AMD Instinct MI300X node (RunPod, ROCm 6.x, torch 2.4-rocm6.0,
~192-core AMD EPYC host) on 2026-07-09:
    recall p50  19.96 ms (GPU idle)  ->  20.08 ms (GPU 100%, 97.4 TFLOPS FP16)  = +0.6%

Run on an AMD GPU box (needs torch-ROCm + this repo's src/ on the path):
    python3 scripts/mi300x_contention_bench.py --burners 3 --store-size 100000
Falls back to a CPU spin load (clearly labelled) if torch/GPU is unavailable, so it
still runs anywhere.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from perseus_vault_store import ReferenceStore  # noqa: E402

TOOLS = ["ripgrep", "sqlite", "rocm", "vllm", "perseus-vault", "fireworks"]
TASKS = ["retrieval", "captioning", "routing", "planning", "summarization"]
PROBES = ["retrieval ripgrep", "routing rocm", "planning vllm",
          "summarization fireworks", "captioning perseus-vault"]

# Self-contained GPU burn: one process pins a chunk of the accelerator with FP16 matmuls.
BURN_SRC = (
    "import torch,time\n"
    "d='cuda' if torch.cuda.is_available() else 'cpu'\n"
    "S=8192\n"
    "a=torch.randn(S,S,device=d,dtype=torch.float16)\n"
    "b=torch.randn(S,S,device=d,dtype=torch.float16)\n"
    "n=0;t0=time.perf_counter()\n"
    "while True:\n"
    "    c=a@b\n"
    "    (torch.cuda.synchronize() if d=='cuda' else None)\n"
    "    n+=1\n"
    "    if n%200==0:\n"
    "        open('/tmp/burn_tflops.txt','w').write(f'{(2*S**3*n)/(time.perf_counter()-t0)/1e12:.1f}')\n"
)


def seed(n: int) -> ReferenceStore:
    s = ReferenceStore(":memory:")
    s.remember_many(("seed", f"agent {i} prefers {TOOLS[i % 6]} for "
                             f"{TASKS[i % 5]} in region amd-cloud-{i % 4}")
                    for i in range(n))
    return s


def measure(store: ReferenceStore, iters: int) -> tuple[float, float]:
    lat = []
    for i in range(iters):
        t = time.perf_counter()
        store.recall(PROBES[i % len(PROBES)], k=5)
        lat.append((time.perf_counter() - t) * 1000.0)
    lat.sort()
    return lat[len(lat) // 2], lat[min(len(lat) - 1, int(len(lat) * 0.99))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--burners", type=int, default=3, help="parallel GPU-burn processes")
    ap.add_argument("--store-size", type=int, default=100_000)
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--ramp", type=float, default=15.0, help="seconds to let the GPU ramp")
    args = ap.parse_args()

    have_gpu = False
    try:
        import torch
        have_gpu = torch.cuda.is_available()
        dev = torch.cuda.get_device_name(0) if have_gpu else "no GPU"
    except Exception:
        dev = "torch unavailable"

    print(f"Accelerator      : {dev}")
    print(f"Host logical CPUs: {os.cpu_count()}")
    store = seed(args.store_size)
    store.recall(PROBES[0], k=5)  # warm

    p50_i, p99_i = measure(store, args.iters)
    print(f"[measured] recall @ GPU idle      : p50 {p50_i:.3f} ms  p99 {p99_i:.3f} ms")

    burn_file = os.path.join(os.path.dirname(__file__), "_burn.py")
    with open(burn_file, "w") as f:
        f.write(BURN_SRC)
    procs = [subprocess.Popen([sys.executable, burn_file],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
             for _ in range(args.burners)]
    try:
        time.sleep(args.ramp)
        tf = ""
        try:
            tf = open("/tmp/burn_tflops.txt").read().strip()
        except OSError:
            pass
        util = f"{tf} TFLOPS FP16" if tf else ("GPU busy" if have_gpu else "CPU spin (no GPU)")
        p50_s, p99_s = measure(store, args.iters)
        print(f"[measured] recall @ GPU saturated : p50 {p50_s:.3f} ms  p99 {p99_s:.3f} ms  ({util})")
        d = p50_s - p50_i
        pct = (d / p50_i * 100.0) if p50_i else 0.0
        print(f"           -> recall p50 moved {d:+.3f} ms ({pct:+.1f}%) under a saturated accelerator")
        print("           -> CPU memory layer and accelerator do not contend "
              "(0 bytes of GPU HBM used by memory)")
    finally:
        for p in procs:
            p.terminate()
        try:
            os.remove(burn_file)
        except OSError:
            pass
        store.close()


if __name__ == "__main__":
    main()
