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

**Measured on AMD hardware** (AMD Developer Cloud notebook — **AMD EPYC 9334 32-core**
host CPU, ROCm 7.2 image, `src/perseus_vault_store.py` reference store, 2026-07-08):

| Entries | Recall p50 (ms) | Recall p99 (ms) | Recall ops/s | `data_source` |
|---|---|---|---|---|
| 10,000 | 2.6 | 3.0 | ~372 | measured (AMD EPYC 9334) |

This is the memory layer running on exactly the kind of AMD server CPU that sits next to
an Instinct accelerator — sub-3 ms recall, on the host, using **0 bytes of GPU HBM**.
(The pool's GPU is a virtualized RDNA3 slice, so we do not report a GPU-load figure; the
CPU number is what matters — the memory layer never touches the accelerator.)

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

Reproduce the model: `python3 src/economics.py`.

---

## 4. What we would measure on real AMD hardware

Cloud credits did not arrive before the deadline, so §3 is a projection. Given an
MI300X node (AMD Developer Cloud) we would replace the projections with measurements:

1. **Recall under real inference load.** Run Perseus Vault on the node's AMD EPYC
   host CPU while an MI300X serves Llama-3.1-70B via ROCm/vLLM, and measure recall
   p50/p99 *while the GPU is saturated* — confirming the CPU memory layer does not
   steal cycles from inference.
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
