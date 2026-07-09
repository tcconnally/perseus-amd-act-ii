# Benchmarks

> **WARNING — read this first.** Every number below is tagged with a `data_source`:
> **`measured`** (timed live, reproducible), **`published-spec`** (vendor datasheet /
> cloud price list, cited), or **`projection`** (derived from published-spec inputs
> with stated assumptions). **There are no measured MI300X numbers in this repo — we
> do not yet have AMD hardware access.** GPU rows are `published-spec`/`projection`
> and are labelled as such. See [§4](#4-what-we-would-measure-on-real-amd-hardware).

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
`src/amd_live_benchmark.py` (point `--base-url` at a vLLM endpoint) or the recall/GPU-burn
probes in [§4](#4-what-we-would-measure-on-real-amd-hardware).

*Honest scope: the GPU load here is a synthetic compute-saturating matmul, not a live vLLM
serving run — it isolates the "does CPU memory work contend with a busy GPU?" question
(answer: no). The `$/agent-hour` economics in §3 remain a `projection`; serving-throughput
$/token on MI300X is the one remaining measurement (we had vLLM-image trouble on the rented
box and prioritized the contention proof).*

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

## 3. Cost economics — one accelerator serves N agents — `data_source: projection`

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

**Takeaway (`projection`):** the MI300X fits a 70B model on **one** card and has the
most HBM left for concurrent sessions → the **lowest GPU $/agent-hour (~7.8× cheaper
than H100, ~3.6× cheaper than A100)** for this workload. Perseus Vault keeps memory
on the CPU (~$0.0004/agent-hr, **~0.3% of the agent's hourly cost**), so none of that
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

We rented an MI300X node and knocked out the most important item; the rest stay on the
list for a longer run.

1. **✅ DONE — Recall while the accelerator is saturated.** Measured on a real MI300X
   node's ~192-core AMD EPYC host: recall p50 moved **+0.6%** (19.96 → 20.08 ms) while
   the **MI300X ran at 100% utilization (97.4 TFLOPS FP16)** — see §1. The CPU memory
   layer and the accelerator do not contend. *(Load was a compute-saturating matmul; the
   next refinement is to drive that saturation with a live vLLM serving run.)*
2. **True concurrent-agent ceiling.** Spin up N agent sessions against one MI300X,
   each with its own encrypted Perseus Vault file, and find the real N where either
   HBM (KV cache) or host RAM (memory files) saturates — versus the §3 projection of
   ~20.
3. **End-to-end agent-turn latency.** recall (CPU) + prompt assembly + generation
   (MI300X) as one number, versus a vector-DB baseline that puts embedding + ANN on
   the critical path.
4. **$/agent-hour, measured.** Real Fireworks/ROCm throughput × real cloud price,
   replacing the datasheet-derived figures in §3.
5. **Optional ROCm offload.** Prototype Perseus Vault's dense re-rank (sign-bit /
   vector ops) on HIP to quantify whether an idle GPU slice can accelerate hybrid
   recall without hurting inference — a genuine open question we would answer with
   data, not claim.
