# AGENTS.md

Guidance for AI coding agents (and humans) working in this repository, plus a
description of the agent this project actually ships.

## The agent this repo ships

`src/agent_memory_demo.py` is a minimal but complete **stateful agent**:

1. **Inference** is an open-weight model behind the Fireworks AI API (target
   deployment: AMD Instinct via ROCm/vLLM; no serving API attests which accelerator
   handles a request). Inference is stateless — the context window dies with the
   session.
2. **Memory** is provided by **Perseus Vault**, an MCP-native, local-first,
   encrypted memory engine (single Rust binary). The agent `remember`s durable
   facts, then in a later session `recall`s them *before* prompting the model, so
   knowledge survives across sessions.
3. **Lifecycle**: a `decay` tick ages rarely-used memories and archives noise
   (never deletes — the journal stays auditable), so recall quality holds as the
   store grows.

The key architectural claim: **memory lives on the host CPU, not in GPU HBM.** See
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Repo map

| Path | What it is |
|---|---|
| `src/perseus_vault_store.py` | The memory interface. `ReferenceStore` (SQLite/FTS5, CPU, always runs) and `BinaryStore` (bridge to a real `perseus-vault` binary when `PERSEUS_VAULT_BIN` is set). |
| `src/agent_memory_demo.py` | End-to-end agent: learn → new session → recall → infer → load → decay. |
| `src/benchmark.py` | Measured throughput + footprint tables (1K/10K/100K) and published-spec economics. |
| `src/economics.py` | The "one MI300X serves N agents" math. Every value tagged `published-spec` / `projection` / `measured`. |
| `docs/` | Architecture, benchmarks, and the pre-filled lablab submission. |

## Ground rules for agents editing this repo

- **The honesty rule is load-bearing.** Never present a GPU/MI300X number as
  measured. Every benchmark row carries a `data_source` of `measured`,
  `published-spec`, or `projection`. If you add a number, tag it and cite it.
- **No secrets in code.** Credentials come from `.env` (gitignored) via
  `.env.example`. Never inline an API key or token.
- **Keep the core stdlib-only.** The demo and benchmark must run with no
  third-party install so any judge can reproduce them. Optional deps go in
  `requirements.txt` behind clear comments.
- **Everything must stay runnable offline.** No network call may be required for
  the demo to complete; the Fireworks path is opt-in via `FIREWORKS_API_KEY`.
- **MIT-licensed and original.** This is a hackathon submission requirement.

## Reproduce everything

```bash
python3 src/agent_memory_demo.py        # the agent, end to end
python3 src/benchmark.py                 # measured tables + economics
python3 src/economics.py                 # just the cost model
docker build -t perseus-amd-act-ii .     # ROCm-based container
```
