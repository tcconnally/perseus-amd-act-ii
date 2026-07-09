# lablab.ai Submission — pre-filled form fields

**Hackathon:** AMD Developer Hackathon: Act II
**Track:** Unicorn (Open) Track — judged on creativity, originality, completeness, use
of AMD platforms, and product/market potential.
**Deadline:** July 11, 2026.

Copy each field below into the matching field on the lablab submission form. Fields
that require a hosted artifact (video, slides) are marked **[ACTION NEEDED]** with a
ready-to-use script/outline so nothing is fabricated.

---

## Field: Project name / Submission title
```
Perseus Vault × AMD Instinct — Encrypted Agent Memory That Stays Off the GPU
```

## Field: Tagline / Short description (one line)
```
Persistent, encrypted memory for AI agents that keeps 100% of MI300X HBM free for
inference — no embeddings, no vector DB, ~$0.13 per agent-hour.
```

## Field: Track / Category
```
Unicorn Track (Open)
```

## Field: Cover image
```
assets/thumbnail.png  (dark-themed architecture diagram, 3:2, in the repo)
```

## Field: Description of your project

**The problem.** An AI agent is only as smart as what it remembers, but its memory is
usually just the LLM context window — gone when the session ends. The standard fix,
bolting on a vector database, adds a second system of record that drifts out of sync,
puts an embedding model on the hot path, and — worst of all on an expensive
accelerator — competes for GPU memory. On an AMD Instinct MI300X, the scarce resource
is its 192 GB of HBM3. Every gigabyte spent storing what the agent *knows* is a
gigabyte not serving tokens.

**Our solution: Perseus Vault.** A single Rust binary that gives agents durable memory
over the Model Context Protocol (MCP). Its recall path is SQLite + FTS5 hybrid search
(BM25 lexical ranking fused with a recency/decay prior) — no embedding model, no
external vector database, and no GPU. The memory layer runs on the host CPU right next
to the accelerator, so 100% of the MI300X's HBM3 stays available for weights and KV
cache, recall adds zero GPU work, and one GPU can back many concurrent agents — each
with its own AES-256-GCM-encrypted, ~85 MB memory file that lives in host RAM/disk, not
in HBM.

**What we built for Act II.** A complete, runnable stateful agent that demonstrates the
full loop: it learns durable facts in one session, then in a brand-new session (empty
context window) recalls them from Perseus Vault *before* prompting an open-weight model
via the Fireworks AI API (the hackathon's designated inference partner), keeps recall
latency low under a burst of load, and runs a decay tick that ages noise into an auditable archive so recall
quality survives as the store grows. The whole thing is containerized on a ROCm base
image and ships with a reproducible benchmark harness.

**Verifiable data (honestly labelled).** We did not get MI300X cloud credits before the
deadline, so we drew a hard line: every number is tagged `measured`, `published-spec`,
or `projection`, and no projection is ever shown as a measurement — the demo prints that
warning on every run.

- *Measured (reproducible now):* recall p50 scales from 0.20 ms @1K to 11.87 ms @100K;
  the shipping Rust engine does FTS5 recall in 17 ms p50 @100K and bulk-inserts 98,732
  entities/s; a 100K-memory agent is ~85 MB RAM + ~45 MB disk.
- *Projection from published specs:* serving Llama-3.1-70B, the MI300X's 192 GB HBM3
  fits the model on ONE card and holds ~20 concurrent agents at **~$0.13/agent-hour —
  about 7.8× cheaper than a 2×H100 deployment** for the same workload — while Perseus
  Vault memory costs ~$0.0004/agent-hour on the CPU and consumes 0 bytes of HBM.

**Why it's a Unicorn.** Agent memory is a real, growing market (Mem0, Letta, Zep) — but
every incumbent is cloud- or vector-DB-bound. Perseus Vault is the only memory engine
that is simultaneously MCP-native, local-first, zero-dependency, and encrypted, and this
submission reframes that as an *AMD economics story*: the cheapest way to give a fleet
of agents durable memory is to keep memory off the accelerator entirely and let the
MI300X do what it's best at — hold a big model and serve tokens. **The strategic point
for AMD:** by removing memory as a reason to reach for NVIDIA, Perseus Vault turns AMD
Instinct into the economical home for stateful agent fleets — an adoption wedge for
Instinct, not just another tool.

**Bonus — Gemma on AMD (partner challenge).** The hackathon's Fireworks account doesn't
expose Gemma models (verified: every catalog Gemma ID returns NOT_FOUND), so we
self-hosted the partner building block on AMD silicon instead: `src/gemma_on_amd.py`
runs the same recall→infer loop with **Gemma 3 4B served locally by llama.cpp on an AMD
Ryzen 7 9800X3D**, beside the Perseus Vault memory layer — recall 0.21 ms, ~13 tok/s,
no GPU, no cloud, no API key, all `measured`. One architecture across the AMD lineup:
Gemma on a Ryzen/EPYC host for single-agent boxes, a 70B-class model on MI300X for
fleets — the memory layer never moves.

**This is a shipping product, not a weekend build.** Perseus Vault is at v2.19 with
**32 releases** — a single ~8 MB Rust binary with 55 MCP tools, AES-256-GCM at rest. It's
distributed where agents actually live: **five framework adapters on PyPI** (LangChain,
CrewAI, PydanticAI, Haystack, Google ADK) and listed in the **MCP registry, Smithery, and
Glama**. It runs in production today, including behind the live demo above. We brought a
real product *to* AMD — that's why the memory layer is production-grade, not a prototype.

## Field: How did you use AMD products / platforms?

- **AMD Instinct MI300X** is the target inference accelerator. The container is built
  `FROM rocm/dev-ubuntu-22.04:6.2` and is ready to serve an open-weight model on an
  Instinct GPU the moment one is attached (`--device=/dev/kfd --device=/dev/dri`). Our
  entire economic thesis is built around the MI300X's 192 GB HBM3 advantage — it fits a
  70B model on one card where H100/A100 need two.
- **ROCm** is the software stack the container targets for GPU inference (via
  vLLM/Fireworks), and it's the runtime we'd use to prototype offloading Perseus Vault's
  optional dense re-rank to an idle GPU slice.
- **Fireworks AI API** — the hackathon's designated inference partner, now
  [partnering with AMD](https://fireworks.ai/blog/fireworks-amd-ai-infrastructure-partnership)
  to serve on Instinct accelerators — serves the open-weight LLM; the agent calls it
  after recalling grounding from memory. (No serving API attests which accelerator
  handles a request, so we don't claim a specific one.) Opt-in via
  `FIREWORKS_API_KEY`; the demo runs fully offline without it.
- **AMD Developer Cloud** is where we would run the measurements listed under "What we
  would measure on real AMD hardware" to replace every projection with a real number.

## Field: Technologies used (tags)
```
Perseus Vault, MCP (Model Context Protocol), Rust, SQLite, FTS5, BM25,
AES-256-GCM, AMD Instinct MI300X, AMD Ryzen, ROCm, Fireworks AI, Gemma 3,
llama.cpp, Llama-3.1-70B, Python, Docker
```

## Field: GitHub / Source code
```
https://github.com/tcconnally/perseus-amd-act-ii
```
Built on the production engine: https://github.com/Perseus-Computing-LLC/perseus-vault

## Field: Live demo (Demo Application URL)
```
https://amd-demo.perseus.observer
```
A hosted, clickable version of `webdemo/` — the repo's SQLite + FTS5 recall path
(CPU reference implementation) running live on the **host CPU** (0 bytes of GPU HBM).
Visitors teach the agent facts (Session 1), start a fresh session and recall them
(Session 2), then an **open-weight LLM (gpt-oss-120b) via the Fireworks AI API** (the
hackathon's designated inference partner) answers using only the recalled facts — the
full recall→infer loop, live. A decay tick and the MI300X-vs-H100-vs-A100 economics
table round it out. Honest scope: recall/footprint are measured on CPU; the LLM answer
is real inference via Fireworks (with a per-day budget cap; it falls back to a labelled
memory-grounded composition when the cap is hit — never a fabricated generation); the
MI300X economics are `projection`.
Each visitor gets an isolated, rate-limited, auto-evicted store. Served from a
container (`--restart unless-stopped`, key via `--env-file`) behind a Cloudflare tunnel.

## Field: Demo video  **[DONE — file uploaded]**
Suggested ≤ 3-minute script (all steps run from a clean clone; no fabricated output):
1. `python3 src/agent_memory_demo.py` — narrate Session 1 (agent learns facts) → Session
   2 (new session, empty context, recalls from Perseus Vault, then answers) → LOAD
   (throughput under a burst) → DECAY (noise archived, signal kept). Point at the
   "Published-spec estimates" warning banner and the "0 GPU HBM used" line.
2. `python3 src/benchmark.py --quick` — show the measured latency/footprint tables and
   the economics table, calling out `data_source` tags.
3. `docker build -t perseus-amd-act-ii .` — show the ROCm base pulling and the FTS5
   build-time check passing.
4. Close on the one-liner: "Keep memory off the GPU; let the MI300X serve tokens."

`demo_video.mp4` (repo root) is uploaded directly to the Video Presentation field —
lablab hosts the file; no external URL is used.

## Field: Presentation / Slides  **[DONE — PDF uploaded]**
Slide deck (PDF) uploaded to the Slide Presentation field. The three-panel story
mirrors this repo: (1) Problem — memory eats HBM; (2) Solution — Perseus Vault keeps
memory on the CPU; (3) Data — the economics table + honesty banner.

## Field: Team members
```
Perseus Computing LLC (Wyoming)
- Thomas Connally — Founder  (GitHub: tcconnally)
```

## Field: What's next / roadmap
```
1. Run on AMD Developer Cloud and replace every projection with a measured number
   (recall under saturated MI300X inference; true concurrent-agent ceiling; measured
   $/agent-hour).
2. Ship a ready-to-deploy "MI300X + Perseus Vault" agent memory reference stack
   (compose file + vLLM/ROCm serving + N per-agent encrypted stores).
3. Prototype an optional ROCm/HIP dense re-rank offload for hybrid recall and quantify
   whether an idle GPU slice helps without hurting inference.
```

---

## Pre-submit checklist
- [x] Public GitHub repo, MIT license visible at top of README
- [x] Original work, MIT-compliant
- [x] Containerized (`docker build .` on a ROCm base image)
- [x] Runnable demo (`python3 src/agent_memory_demo.py`, stdlib-only, offline)
- [x] Verifiable data with honest `data_source` tags; no projection shown as measured
- [x] `.env.example` only — no real credentials committed
- [x] Demo video produced (`demo_video.mp4`, ~91s, AMD-specific; built by
      `generate_video.py` + `narration.py`) — lablab hosts the file directly, so it is
      uploaded to the Video Presentation field (not a URL)
- [x] Slide deck (PDF) uploaded to the Slide Presentation field
- [x] Cover image (`assets/thumbnail.png`) uploaded to the Cover Image field
- [x] Unicorn (Open) Track selected on the lablab form
- [x] Step 3 (Application) completed: GitHub repo URL; Docker Image = "N/A" (Unicorn
      Track, not Track 1/2); honest measured-vs-projection note in Additional Information
- [x] Live hosted demo deployed: https://amd-demo.perseus.observer (CPU-only memory
      engine, per-visitor sandbox, smoke-tested over TLS end-to-end 2026-07-08)
- [x] Demo Application URL on lablab Step 3 set to https://amd-demo.perseus.observer;
      Platform = "Other" (done 2026-07-08)
- [x] Final human check + Submit clicked on lablab (done 2026-07-08, ahead of the
      Jul 11 deadline)

> lablab submission form: https://lablab.ai/ai-hackathons/amd-developer-hackathon-act-ii/perseus/submission
> The Video Presentation field takes an uploaded file (it displays the clip inline), not
> a YouTube/URL link — upload `demo_video.mp4` from the repo root.
