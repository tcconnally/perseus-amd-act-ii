# Benchmarks

> **WARNING — read this first.** Every number below is tagged with a `data_source`:
> **`measured`** (timed live, reproducible), **`published-spec`** (vendor datasheet /
> cloud price list, cited), or **`projection`** (derived from published-spec inputs
> with stated assumptions). We rented a real **AMD Instinct MI300X** and a real
> **2× NVIDIA H100 SXM** and measured the load-bearing claims on both sides ourselves
> (§1, §3a, §3b). Only the A100 row remains `published-spec`/`projection`, labelled
> as such. See [§4](#4-what-we-measured-on-real-amd-hardware--and-whats-next).

Reproduce §1–§2 yourself: `python3 src/benchmark.py` (add `--quick` to skip 100K).

---

## 1. Recall throughput & latency — `data_source: measured`

The agent's recall path (BM25 lexical + recency fusion over SQLite/FTS5) at three
context sizes. Latency stays low and predictable as the store grows 100×.

**Reference implementation** (bundled `ReferenceStore`, run on an AMD-CPU laptop,
Python 3.14, single process — reproduce with `src/benchmark.py`):

| Entries | Recall ops/s | p50 (ms) | p99 (ms) | Insert ops/s |
|---|---|---|---|---|
| 1,000   | 5,081 | 0.20  | 0.39  | 72,917 |
| 10,000  | 953   | 1.14  | 1.44  | 72,217 |
| 100,000 | 92    | 11.87 | 15.67 | 68,276 |

**Shipping engine** (Perseus Vault v2.19.x Rust binary, measured on AMD 16-core CPU
"Family 26", Windows 11 — source: [PERF.md](https://github.com/Perseus-Computing-LLC/perseus-vault/blob/main/PERF.md)
and the repo's `benchmark/scale/` gate):

| Recall mode @ 100K | p50 (ms) | p99 (ms) | `data_source` |
|---|---|---|---|
| FTS5 (lexical)   | 17.0  | 19.4  | measured |
| Dense (sign-bit) | 24.9  | 29.3  | measured |
| Hybrid (RRF)     | 308.4 | 325.3 | measured |
| Bulk insert (100K) | — | — | 98,732 entities/s (measured) |

*The reference implementation's 11.9 ms p50 @100K tracks the shipping engine's
17 ms FTS5 p50 — the bundled store is a faithful shape of the real recall path.*

### Measured on a real MI300X node — recall is unaffected while the GPU is saturated — `measured`

We rented a single **AMD Instinct MI300X OAM (192 GB)** node (RunPod, ROCm 6.x, PyTorch
2.4-rocm6.0, ~192-core AMD EPYC host CPU) and measured the load-bearing claim directly:
does the CPU-side memory layer steal cycles from the accelerator? We drove the **MI300X to
100% utilization** (a compute-saturating FP16 matmul, **97.4 TFLOPS** sustained, confirmed
by `rocm-smi`) and measured Perseus Vault recall on the host CPU **before and during** that
saturation (100K-memory store, `src/perseus_vault_store.py`, 2026-07-09):

| Condition | MI300X util | Recall p50 (ms) | Recall p99 (ms) | `data_source` |
|---|---|---|---|---|
| GPU idle       | 0%   | 19.96 | 21.38 | measured (MI300X node) |
| GPU saturated  | **100%** (97.4 TFLOPS FP16) | 20.08 | 22.20 | measured (MI300X node) |

**Recall p50 moved +0.12 ms (+0.6%) while the MI300X ran flat-out.** The memory layer lives
on the host CPU and uses **0 bytes of GPU HBM**, so a fully-loaded accelerator does not slow
recall and recall does not slow the accelerator — the two never contend. Reproduce with
`src/amd_live_benchmark.py` (point `--base-url` at a vLLM endpoint) or
`scripts/mi300x_contention_bench.py` (the GPU-burn + recall probe used for this run).

*Honest scope: the GPU load in this section is a synthetic compute-saturating matmul —
it isolates the "does CPU memory work contend with a busy GPU?" question (answer: no).
We have since repeated the measurement under a **real vLLM serving load** (Qwen2.5-72B)
with the same result — see [§3a](#3a-measured-on-a-real-mi300x--data_source-measured).
The cross-vendor `$/agent-hour` comparison is now **measured too** (§3b, 2×H100);
only the A100 row remains a `projection`.*

---

## 2. Memory footprint — `data_source: measured`

**Reference implementation** (`src/benchmark.py`):

| Entries | DB file (MB) | Decay tick (s) |
|---|---|---|
| 1,000   | 0.31  | 0.000 |
| 10,000  | 2.61  | 0.003 |
| 100,000 | 25.95 | 0.036 |

**Shipping engine** (perseus-vault
[README stress test](https://github.com/Perseus-Computing-LLC/perseus-vault#stress-test-100k-entities),
measured, AMD CPU):

| Metric @ 100K entities | Result | `data_source` |
|---|---|---|
| Process RSS   | ~85 MB   | measured |
| DB file size  | ~45 MB   | measured |
| Decay tick    | 1.317 s (batched, transactional) | measured |
| 100K insert   | 1.01 s (98,732/s) | measured |

**Footprint is the point:** a full 100K-memory agent is ~85 MB RAM + ~45 MB disk —
on the **host**, not in GPU HBM. Thousands of per-agent stores fit in ordinary
system RAM.

### Bonus — whole agent on one AMD chip (Gemma 3 + Perseus Vault) — `measured`

For the hackathon's Gemma partner challenge, `src/gemma_on_amd.py` runs the same
recall→infer loop with **Gemma 3 (4B-it, Q4_K_M) served locally by llama.cpp** beside
the memory layer — one AMD processor, no GPU, no cloud, no API key:

| Host | Recall p50 | Gemma 3 4B generation (wall-clock) | `data_source` |
|---|---|---|---|
| AMD Ryzen 7 9800X3D (8-core) | 0.21 ms | ~13 tok/s | measured (2026-07-08) |

(Gemma on Fireworks is on-demand — deploy-it-yourself — and even the cheapest option
bills ~$7/hour while idle. Self-hosting Gemma on AMD silicon costs $0 and runs with no
idle meter — the more on-thesis answer, and no download or key required.)

---

## 3. Cost economics — one accelerator serves N agents

### 3a. Measured on a real MI300X — `data_source: measured`

We served **Qwen2.5-72B-Instruct** (bf16) on one rented **AMD Instinct MI300X**
(vLLM 0.19.1 + ROCm 7.13, host = **AMD EPYC 9474F**, $2.19/GPU-hr retail, 2026-07-09)
and measured the whole loop — 6 harness runs plus 3 saturation runs (`vllm bench serve`,
1,024-token inputs / 512-token outputs, max-concurrency 64). Medians reported:

| Metric | Measured | `data_source` |
|---|---|---|
| Model weights on GPU | 135.5 GiB | measured (vLLM) |
| KV-cache budget | 38.36 GiB → 125,696 tokens | measured (vLLM; matches Qwen2.5-72B KV arithmetic, 320 KiB/token) |
| **Concurrent agents (8K-token ctx) / card** | **15.3** (15.2–15.3 across configs) | measured (vLLM KV ceiling) |
| **GPU $/agent-hour** | **$0.143** ($0.143–0.144) | measured price ÷ measured ceiling |
| Sustained serving throughput | **658 output tok/s** (1,973 total tok/s) | measured (3 runs: 649–660) |
| Peak output throughput | 1,088 tok/s | measured (identical in all 3 runs) |
| Per-stream decode speed | 83 ms/token ≈ **12 tok/s per agent stream** | measured (median TPOT) |
| **$ / 1M output tokens** | **$0.92** ($0.31 / 1M total tokens) | measured throughput × rental price |
| **Recall p50 — GPU idle vs. serving 72B** | **18.7 → 18.8 ms (±0.6% median, 6 runs)** | measured (100K store, EPYC host) |

The load-bearing claim, now proven under **real 72B inference load** (not just the
synthetic matmul in §1): recall on the host CPU is unaffected while the MI300X serves —
across 6 idle-vs-serving comparisons the p50 delta ranged −0.4% to +1.1% (median ≈ +0.5%).
The projection below is validated: we projected $0.133/agent-hr, **measured $0.143**.
Measured concurrency (15.3) came in below the idealized projection (20.4) because real
vLLM reserves HBM for activations/overhead beyond the raw (HBM−weights)/KV arithmetic —
the measured number is the honest one. Independent corroboration: the saturation runs
peaked at 80–87 in-flight requests, matching the KV-budget prediction for ~1.5K-token
requests (125,696 ÷ ~1,536 ≈ 81).

**Scope, stated plainly:** this is **untuned, out-of-the-box vLLM** — bf16 weights (no
FP8), no AITER kernels, no speculative decoding. Even so, **$0.92/1M output tokens on a
single retail-rented card is at parity with managed-API list pricing** for 70B-class
models — treat these numbers as a floor, not a ceiling. (Earlier single-process runs
(`VLLM_ENABLE_V1_MULTIPROCESSING=0`) gave the same sustained throughput but 133 ms TPOT
and long queue times; the multiprocess figures above are the correct serving shape.)

Reproduce: serve with `vllm serve Qwen/Qwen2.5-72B-Instruct --max-model-len 8192
--gpu-memory-utilization 0.92` (remount `/dev/shm` to ≥16G first on container hosts),
then `python3 src/amd_live_benchmark.py --base-url http://localhost:8000` for the
agent-ceiling + recall-under-load metrics, and `vllm bench serve --model
Qwen/Qwen2.5-72B-Instruct --dataset-name random --random-input-len 1024
--random-output-len 512 --num-prompts 256 --max-concurrency 64` for the throughput row.

### 3b. Measured cross-vendor baseline: 2× NVIDIA H100 SXM — `data_source: measured`

We then rented **2× H100 SXM 80 GB** (Lambda, $8.38/hr, NVLink, host Intel Xeon
Platinum 8480+) and ran the **same model, same vLLM version (0.19.1), same bench
parameters** as §3a. Measured, 2026-07-09:

| Step | Result | `data_source` |
|---|---|---|
| 1× H100 (80 GB), load attempt | **Fails — CUDA OOM** ("79.19 GiB total, 583 MiB free"; vLLM suggests `--tensor-parallel-size`) | measured |
| 2× H100, util 0.92 (identical to §3a config) | **Cannot serve one 8K request** — 0.67 GiB KV available vs 1.25 needed; vLLM caps context at 4,368 | measured |
| 2× H100, util 0.97 | KV math allows 3.70 sessions — **but OOM-crashes on startup** (CUDA-graph capture, 78.9/79.19 GiB) | measured |
| 2× H100, util 0.97 + `--enforce-eager` (**best case that boots**) | Serves: **“Maximum concurrency for 8,192 tokens per request: 5.00×”** | measured |
| Saturation throughput (3 runs, median) | 679.8 output tok/s (2,039 total), peak 1,153, TPOT 38.7 ms | measured |

**Measured-vs-measured economics (same model, same vLLM, n=3 medians both sides):**

| | 1× MI300X | 2× H100 SXM (best case) | Advantage |
|---|---|---|---|
| Holds Qwen2.5-72B bf16 | ✅ one card, 38 GiB KV to spare | ❌ 1 card can't load it; 2 required | — |
| Concurrent 8K agents | **15.3** (std settings, graphs on) | **5.0** (eager-only, 97% util redline) | **3.1×** |
| Rental $/hr | $2.19 | $8.38 | 3.8× |
| **GPU $/agent-hour** | **$0.143** | **$1.68** | **11.7×** |
| Sustained output tok/s | 658 | 680 | ≈parity |
| **$ / 1M output tokens** | **$0.92** | **$3.42** | **3.7×** |
| Per-stream decode (TPOT) | 83 ms | 39 ms | H100 2.1× faster |

Honest notes: (a) the H100 figures are its **best case** — the only configuration that
boots at all sacrifices CUDA graphs at 97% utilization, while the MI300X ran standard
settings; at the *identical* configuration the comparison is not 11.7×, it is
**undefined, because the H100 pair serves zero 8K requests**. (b) H100 wins per-stream
decode latency (39 vs 83 ms TPOT) — if single-stream speed is your constraint, the H100
is faster; if fleet capacity per dollar is (the agent-fleet case), the MI300X wins
11.7×. (c) Our earlier projection said ~7.8×; **measurement says 11.7×** — the
projection *understated* the MI300X advantage, because it didn't model the H100's
graph-capture overhead and activation headroom. Every row reproduces from the repo.

### 3c. Cross-accelerator projection (A100) — `data_source: projection`

*(The A100 row below remains a projection — we rented MI300X and H100, not A100.)*

**Inputs (`published-spec`, cited in the README):**

| Accelerator | HBM | Bandwidth | FP16 (peak) | TDP | Cloud $/hr (2026) |
|---|---|---|---|---|---|
| **AMD Instinct MI300X** | **192 GB HBM3** | 5.325 TB/s | 1,307 TFLOPS | 750 W | $2.72 (median) |
| NVIDIA H100 SXM | 80 GB HBM3 | 3.35 TB/s | 989 TFLOPS | 700 W | $3.93 |
| NVIDIA A100 80GB SXM | 80 GB HBM2e | 2.04 TB/s | 312 TFLOPS | 400 W | $1.80 |

**Workload assumption (`projection`, stated openly):** serve **Llama-3.1-70B in
FP16** (~141 GB weights), fill remaining HBM with 8K-token KV cache (~2.5 GB/seq),
and count concurrent agent sessions. Perseus Vault memory (~85 MB/agent) lives on
the host CPU and consumes **0 bytes of HBM**.

| Accelerator | Cards for 70B weights | Concurrent agents | GPU $/hr | **GPU $/agent-hr** | Perseus Vault mem $/agent-hr |
|---|---|---|---|---|---|
| **AMD Instinct MI300X** | **1** | **20.4** | $2.72 | **$0.133** | $0.00042 |
| NVIDIA H100 SXM | 2 | 7.6 | $7.86 | $1.034 | $0.00042 |
| NVIDIA A100 80GB SXM | 2 | 7.6 | $3.60 | $0.474 | $0.00042 |

**Takeaway:** the projection said **~7.8× cheaper than 2×H100** — the §3b
**measurement came in at 11.7×** (the projection understated the MI300X advantage;
it didn't model the H100's graph-capture overhead and activation headroom). The
A100 comparison (~3.6×) remains a `projection`. Perseus Vault keeps memory on the
CPU (~$0.0004/agent-hr, **~0.3% of the agent's hourly cost**), so none of that
192 GB is wasted storing what the agent knows instead of serving tokens.

> **On the FP16 assumption (honest caveat).** We model **unquantized** Llama-3.1-70B.
> The single-card claim is arithmetic from the cited datasheets, not a quote: FP16
> weights are ~141 GB (70.6B params × 2 bytes), which fits inside the MI300X's 192 GB
> of HBM3 and does not fit inside the 80 GB of an H100 or A100. Quantize to FP8 and a
> 70B fits on a single H100 too — the MI300X edge then shifts from *"fits on one card"*
> to *"far more KV-cache headroom → more concurrent agents per card,"* a **smaller but
> still real** $/agent-hour advantage. Either way the load-bearing claim holds:
> Perseus Vault memory stays on the CPU and consumes **0 bytes of HBM**.

Reproduce the model: `python3 src/economics.py`.

---

## 4. What we measured on real AMD hardware — and what's next

We rented real MI300X time (twice) and knocked out the most important items; the rest
stay on the list for a longer run.

1. **✅ DONE — Recall while the accelerator is saturated.** Measured twice on real
   MI300X nodes: **+0.6%** (19.96 → 20.08 ms p50) under a synthetic 100%-utilization
   matmul (97.4 TFLOPS FP16 — see §1), and **±0.6% median across 6 runs**
   (18.7 → 18.8 ms p50) under a **real vLLM serving load of Qwen2.5-72B** — see §3a.
   The CPU memory layer and the accelerator do not contend.
2. **✅ DONE — True concurrent-agent ceiling.** Measured from vLLM's own KV-cache
   budget while serving Qwen2.5-72B bf16: **15.3 concurrent 8K-token agents** on one
   MI300X (vs the idealized §3b projection of ~20) — see §3a.
3. **End-to-end agent-turn latency.** recall (CPU) + prompt assembly + generation
   (MI300X) as one number, versus a vector-DB baseline that puts embedding + ANN on
   the critical path.
4. **✅ DONE — $/agent-hour and $/token, measured on both vendors.** MI300X:
   $2.19 ÷ 15.3 = **$0.143/agent-hr**, 658 tok/s sustained → **$0.92/1M output
   tokens** (§3a). 2×H100 baseline: best case 5.0 agents → **$1.68/agent-hr**,
   $3.42/1M output tokens (§3b) → **11.7× / 3.7× measured-vs-measured**.
5. **Optional ROCm offload.** Prototype Perseus Vault's dense re-rank (sign-bit /
   vector ops) on HIP to quantify whether an idle GPU slice can accelerate hybrid
   recall without hurting inference — a genuine open question we would answer with
   data, not claim.
