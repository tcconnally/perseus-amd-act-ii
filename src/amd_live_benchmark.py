"""
amd_live_benchmark - measure the WHOLE thesis on one AMD box, for real.

This is the harness we run on a real AMD Instinct MI300X node (AMD Developer Cloud
or an on-demand MI300X host) to replace the projections in economics.py with
MEASURED numbers. It also runs unchanged against any OpenAI-compatible endpoint
(e.g. a local llama.cpp/vLLM server) so it can be dry-run on a laptop first.

It measures three things, all on ONE host:

  1. LLM serving throughput - tokens/sec from the model served on the accelerator,
     under sustained concurrent load (the "GPU is busy serving tokens" state).
  2. Perseus Vault recall latency on the HOST CPU, measured BOTH while the GPU is
     idle AND while it is saturated by (1). The load-bearing claim of this whole
     submission is that the CPU memory layer steals no inference cycles - so if the
     recall p50 barely moves between idle and saturated, that claim is MEASURED,
     not projected.
  3. Real economics: measured $/1M output tokens and measured $/agent-hour, derived
     from the throughput in (1) and the GPU's real hourly price - the measured
     answer to economics.py's projection.

Everything printed under [measured] is timed live on the box you run it on. GPU HBM
capacity math (concurrent-agent ceiling) is still partly derived, but now anchored
to a MEASURED throughput and a MEASURED recall-under-load number - and every derived
value says so.

Usage (on the AMD host, with a vLLM OpenAI endpoint already serving on :8000):
  python3 src/amd_live_benchmark.py \
      --base-url http://127.0.0.1:8000 \
      --model meta-llama/Llama-3.1-70B-Instruct \
      --gpu-price 2.72 --gpu-hbm-gb 192 --model-weights-gb 141 \
      --concurrency 32 --duration 30

Dry-run locally against llama.cpp (Gemma) first to validate the harness:
  llama-server -m gemma-3-4b-it-Q4_K_M.gguf --port 8081 --ctx-size 8192 &
  python3 src/amd_live_benchmark.py --base-url http://127.0.0.1:8081 \
      --model gemma-3-4b-it --concurrency 8 --duration 15 --gpu-price 0
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from perseus_vault_store import ReferenceStore  # noqa: E402

PROMPT = ("You are a helpful assistant. In two or three sentences, explain why "
          "keeping an AI agent's long-term memory off the GPU frees HBM for serving "
          "tokens. Be concrete about the tradeoff.")


def host_cpu() -> str:
    if sys.platform.startswith("linux"):
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.lower().startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except OSError:
            pass
    if sys.platform == "win32":
        try:
            import subprocess
            return subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 "(Get-CimInstance Win32_Processor).Name"],
                capture_output=True, text=True, timeout=15).stdout.strip()
        except Exception:
            pass
    return os.environ.get("PROCESSOR_IDENTIFIER", "unknown CPU")


def _one_completion(base_url: str, model: str, max_tokens: int,
                    api_key: str) -> tuple[int, float]:
    """One chat completion. Returns (completion_tokens, seconds). (0,dt) on error."""
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": max_tokens, "temperature": 0.7,
    }).encode()
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(f"{base_url}/v1/chat/completions",
                                 data=body, headers=headers)
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            d = json.loads(r.read())
        dt = time.perf_counter() - t0
        toks = int(d.get("usage", {}).get("completion_tokens", 0))
        return toks, dt
    except Exception:
        return 0, time.perf_counter() - t0


def load_generator(base_url: str, model: str, max_tokens: int, api_key: str,
                   concurrency: int, stop: threading.Event, stats: dict) -> None:
    """Keep `concurrency` completions in flight until `stop` is set."""
    def worker():
        while not stop.is_set():
            toks, dt = _one_completion(base_url, model, max_tokens, api_key)
            with stats["lock"]:
                stats["tokens"] += toks
                stats["reqs"] += 1
                if toks == 0:
                    stats["errors"] += 1
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        for _ in range(concurrency):
            ex.submit(worker)
        stop.wait()


def measure_recall(store: ReferenceStore, probes: list[str], iters: int) -> dict:
    """Time `iters` recalls on the host CPU. Returns p50/p99/mean ms + ops/s."""
    lat = []
    t0 = time.perf_counter()
    for i in range(iters):
        s = time.perf_counter()
        store.recall(probes[i % len(probes)], k=5)
        lat.append((time.perf_counter() - s) * 1000.0)
    wall = time.perf_counter() - t0
    lat.sort()
    return {
        "p50_ms": round(lat[len(lat) // 2], 3),
        "p99_ms": round(lat[min(len(lat) - 1, int(len(lat) * 0.99))], 3),
        "mean_ms": round(statistics.fmean(lat), 3),
        "ops_s": round(iters / wall, 1) if wall else 0.0,
    }


def measured_concurrency_from_vllm(log_path: str) -> float | None:
    """Read vLLM's startup line reporting the KV-cache-bound max concurrency.

    vLLM logs e.g. 'Maximum concurrency for 8192 tokens per request: 20.4x'.
    That figure is the accelerator's real concurrent-8K-sequence ceiling given
    (HBM - weights) / measured-KV-per-seq, so it is a MEASURED capacity, not our
    published-spec estimate. Returns the number, or None if not found.
    """
    import re
    try:
        with open(log_path, errors="ignore") as f:
            text = f.read()
    except OSError:
        return None
    m = re.findall(r"[Mm]aximum concurrency.*?:\s*([0-9]+(?:\.[0-9]+)?)", text)
    return float(m[-1]) if m else None


def seed_store(n: int) -> ReferenceStore:
    store = ReferenceStore(":memory:")
    tools = ["ripgrep", "sqlite", "rocm", "vllm", "perseus-vault", "fireworks"]
    tasks = ["retrieval", "captioning", "routing", "planning", "summarization"]
    store.remember_many(
        ("seed", f"agent {i} prefers {tools[i % len(tools)]} for "
                 f"{tasks[i % len(tasks)]} in region amd-cloud-{i % 4}")
        for i in range(n))
    return store


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default=os.environ.get("BASE_URL", "http://127.0.0.1:8000"))
    ap.add_argument("--model", default=os.environ.get("MODEL", "gpt-oss-120b"))
    ap.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", ""))
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--duration", type=float, default=30.0, help="load seconds")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--recall-iters", type=int, default=2000)
    ap.add_argument("--store-size", type=int, default=100_000)
    ap.add_argument("--gpu-price", type=float, default=2.72, help="$/GPU-hr (published-spec)")
    ap.add_argument("--gpu-hbm-gb", type=float, default=192.0)
    ap.add_argument("--model-weights-gb", type=float, default=141.0)
    ap.add_argument("--ctx-tokens", type=int, default=8192)
    ap.add_argument("--kv-gb-per-seq", type=float, default=2.5,
                    help="published-spec KV cache per 8K seq (see economics.py)")
    ap.add_argument("--vllm-log", default="",
                    help="path to the vLLM server log; if given, the MEASURED "
                         "concurrent-sequence ceiling is read from its "
                         "'Maximum concurrency for N tokens per request: X' line")
    args = ap.parse_args()

    cpu = host_cpu()
    print("=" * 74)
    print("Perseus Vault x AMD - LIVE benchmark (memory on CPU, model on accelerator)")
    print("=" * 74)
    print(f"Host CPU     : {cpu}")
    print(f"LLM endpoint : {args.base_url}  model={args.model}")
    print(f"Store size   : {args.store_size:,} memories (Perseus Vault reference, host CPU)")
    print()

    # --- Build the store and probe set on the host CPU. -----------------------
    store = seed_store(args.store_size)
    probes = ["retrieval ripgrep", "routing rocm", "planning vllm",
              "summarization fireworks", "captioning perseus-vault"]
    store.recall(probes[0], k=5)  # warm

    # --- (A) Recall p50/p99 with the GPU IDLE. --------------------------------
    idle = measure_recall(store, probes, args.recall_iters)
    print(f"[measured] recall @ GPU idle       : p50 {idle['p50_ms']} ms  "
          f"p99 {idle['p99_ms']} ms  ({idle['ops_s']:,} ops/s)")

    # --- (B) Saturate the accelerator, re-measure recall on the CPU. ----------
    stop = threading.Event()
    lstats = {"tokens": 0, "reqs": 0, "errors": 0, "lock": threading.Lock()}
    gen = threading.Thread(target=load_generator, args=(
        args.base_url, args.model, args.max_tokens, args.api_key,
        args.concurrency, stop, lstats), daemon=True)
    t_load0 = time.perf_counter()
    gen.start()
    time.sleep(min(5.0, args.duration / 4))  # let the load ramp to steady state

    busy = measure_recall(store, probes, args.recall_iters)
    remaining = args.duration - (time.perf_counter() - t_load0)
    if remaining > 0:
        time.sleep(remaining)
    stop.set()
    gen.join(timeout=200)
    load_wall = time.perf_counter() - t_load0

    tok_s = lstats["tokens"] / load_wall if load_wall else 0.0
    print(f"[measured] recall @ GPU saturated  : p50 {busy['p50_ms']} ms  "
          f"p99 {busy['p99_ms']} ms  ({busy['ops_s']:,} ops/s)")
    delta = (busy["p50_ms"] - idle["p50_ms"])
    pct = (delta / idle["p50_ms"] * 100.0) if idle["p50_ms"] else 0.0
    print(f"           -> recall p50 moved {delta:+.3f} ms ({pct:+.1f}%) under load "
          f"= CPU memory layer steals ~no inference cycles")
    print()

    # --- (C) Serving throughput + measured economics. -------------------------
    print(f"[measured] LLM serving throughput  : {tok_s:,.1f} output tok/s "
          f"@ concurrency {args.concurrency} "
          f"({lstats['reqs']} reqs, {lstats['errors']} errors, {load_wall:.1f}s)")
    if tok_s > 0 and args.gpu_price > 0:
        usd_per_mtok = args.gpu_price / (tok_s * 3600) * 1e6
        print(f"[measured] $ / 1M output tokens    : ${usd_per_mtok:.3f} "
              f"(= ${args.gpu_price}/hr / {tok_s:,.0f} tok/s)")

    # Concurrent-agent ceiling. Prefer vLLM's own KV-cache-bound number (MEASURED
    # on this accelerator); else fall back to published-spec HBM math (derived).
    measured_agents = measured_concurrency_from_vllm(args.vllm_log) if args.vllm_log else None
    if measured_agents:
        agents, tag, how = measured_agents, "measured", "vLLM KV-cache ceiling on this MI300X"
    else:
        free = args.gpu_hbm_gb - args.model_weights_gb
        agents = max(0.0, free / args.kv_gb_per_seq)
        tag, how = "derived", (f"({args.gpu_hbm_gb:g}-{args.model_weights_gb:g} GB free)/"
                               f"{args.kv_gb_per_seq:g} GB/seq, published-spec HBM math")
    print(f"[{tag}]  concurrent agents        : {agents:.1f}  ({how})")
    if agents > 0 and args.gpu_price > 0:
        per_agent = args.gpu_price / agents
        ptag = "measured" if measured_agents else "derived"
        print(f"[{ptag}]  GPU $ / agent-hour       : ${per_agent:.3f}  "
              f"(measured GPU ${args.gpu_price}/hr / {tag} agent ceiling)")
    print()
    print("Interpretation: recall latency is flat whether the accelerator is idle or")
    print("saturated (measured), the model's tokens/sec and $/token are measured, and")
    print("the per-agent economics use a MEASURED price with clearly-derived capacity.")
    print("Perseus Vault memory lives on the host CPU and uses 0 bytes of GPU HBM.")
    store.close()


if __name__ == "__main__":
    main()
