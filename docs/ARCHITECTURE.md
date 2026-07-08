# Architecture

## The problem

An AI agent is only as good as what it remembers. But the way most agents "remember"
is the LLM context window — which vanishes when the session ends. The common fix is
to bolt on a vector database: embed everything, store the vectors, do an approximate
nearest-neighbour search at recall time. That buys persistence at a steep price:

- **A second system of record** (Postgres/pgvector, Pinecone, a Docker sidecar) that
  can drift out of sync with the agent's structured state.
- **An embedding model on the hot path** for every write and every query — GPU
  cycles and latency spent before the *actual* model even runs.
- **HBM pressure**: if the vector index or embedding model shares the accelerator,
  it eats the exact memory you wanted for weights and KV cache.

On an expensive accelerator like the AMD Instinct MI300X, that last point is the
whole game. Its 192 GB of HBM3 is the scarce resource. Every gigabyte spent storing
what the agent knows is a gigabyte not serving tokens.

## The solution: keep memory off the GPU

**Perseus Vault** is a single Rust binary that gives an agent durable memory over the
Model Context Protocol (MCP). Its recall path is **SQLite + FTS5 hybrid search** —
BM25 lexical ranking blended with a recency/decay prior, with an optional dense mode.
It needs **no embedding model, no external vector database, and no GPU**.

That is not a limitation — it is the design. The memory layer runs on the host CPU
next to the accelerator, so:

- **100% of the MI300X's 192 GB HBM3 stays available for weights + KV cache.**
- Recall adds no GPU work and does not compete with inference for the accelerator.
- One GPU can back many concurrent agents, each with its own encrypted memory file,
  because those files live in host RAM/disk (measured ~85 MB RSS + ~45 MB on disk
  per 100K-memory agent), not in HBM.

```
                          AMD Instinct MI300X (192 GB HBM3)
                       +--------------------------------------+
   user turn  ───────► |  LLM weights + KV cache (inference)  |
       ▲               |  served via Fireworks AI / vLLM / ROCm|
       │               +--------------------------------------+
       │                              ▲   │ tokens
       │ recall(query) THEN infer     │   ▼
       │                        ┌───────────────┐
       └────────────────────────┤  Agent loop   │
                                └───────────────┘
                                   ▲        │ remember() / recall() / decay()
                        grounding  │        ▼   (MCP stdio, CPU only)
                                ┌─────────────────────────────┐
                                │  Perseus Vault (Rust binary) │
                                │  SQLite + FTS5, AES-256-GCM  │
                                │  one portable .db file/agent │  ── 0 bytes HBM
                                └─────────────────────────────┘
                                     host CPU + RAM + disk
```

## The agent loop (what the demo does)

1. **Session 1 — learn.** The agent stores durable facts (`remember`). They persist
   to a single encrypted file and outlive the session.
2. **Session 2 — recall, then infer.** A fresh session has an empty context window.
   The agent `recall`s the relevant facts from Perseus Vault (BM25 + recency on the
   CPU), injects them as grounding, and only then calls the model on the MI300X.
3. **Load.** Under a burst of recalls against a growing store, throughput and latency
   stay flat (measured — see [BENCHMARKS.md](BENCHMARKS.md)).
4. **Decay.** A periodic tick ages rarely-recalled memories and archives noise
   (never deletes; the journal is immutable and auditable), so signal survives and
   recall quality does not rot as the store grows.

## Why FTS5 / BM25 instead of vectors

- **Zero dependencies, zero GPU.** No embedding model to host, no vector DB to run.
- **Deterministic and auditable.** Lexical matches are explainable; there is no
  opaque similarity score to debug, which matters for regulated/on-prem deployments.
- **Tiny footprint.** A 100K-entity store is ~26–45 MB on disk (measured), so
  thousands of per-agent stores fit in ordinary host RAM.
- **Encrypted + local-first.** AES-256-GCM at rest, one portable file, no cloud
  round-trip — the memory never leaves the box the GPU is in.

Perseus Vault does also offer a dense/hybrid mode; this submission leans on the
lexical path precisely because it keeps the accelerator free for inference.

## Where AMD hardware fits

- **Inference on MI300X via ROCm.** The container is built `FROM rocm/dev-ubuntu-22.04`
  so it is ready to serve an open-weight model on an Instinct GPU (through the
  Fireworks AI API or a local vLLM/ROCm stack) the moment a GPU is attached.
- **The economic argument** (see [BENCHMARKS.md](BENCHMARKS.md) §3): the MI300X's
  192 GB HBM3 fits a 70B model on a *single* card and leaves the most room for
  concurrent agent sessions — and Perseus Vault ensures none of that HBM is wasted
  on memory storage. That combination is what makes the lowest $/agent-hour.
