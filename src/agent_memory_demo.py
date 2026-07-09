"""
agent_memory_demo - an agent that remembers, across sessions, using Perseus Vault.

The story: the agent's LLM is designed to run on an AMD Instinct MI300X (vLLM on
ROCm, or behind an API such as Fireworks AI). That inference is stateless - the
moment a session ends, the context window is gone. Perseus Vault is the durable
memory layer that lets the agent carry knowledge from one session into the next,
WITHOUT consuming any GPU HBM.

This script demonstrates the full loop end-to-end and runs anywhere:
  * Memory  : Perseus Vault reference store (SQLite/FTS5, CPU) - or the real
              perseus-vault binary if PERSEUS_VAULT_BIN is set.
  * Inference: Fireworks AI (real HTTP call) if FIREWORKS_API_KEY is set;
              otherwise a deterministic offline stand-in so the demo always runs.
              (No serving API attests which accelerator handles a request, so we
              never claim a specific one.)

HONESTY: no GPU is touched unless you supply FIREWORKS_API_KEY. Any MI300X/ROCm
performance number this repo prints is a published-spec estimate, never a
measurement - the banner below says so on every run.
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from perseus_vault_store import open_store  # noqa: E402

WARNING = ("WARNING: Published-spec estimates for MI300X/ROCm. "
           "Real MI300X data pending AMD hardware access.")

# Open-weight model served via the Fireworks AI API (target deployment: Instinct/ROCm).
FIREWORKS_MODEL = os.environ.get(
    "FIREWORKS_MODEL", "accounts/fireworks/models/llama-v3p1-70b-instruct")


def infer(prompt: str) -> str:
    """Call the LLM via the Fireworks AI API, or fall back to an offline stand-in.

    Returns model text. The fallback is clearly a stand-in - it never pretends to
    be a real generation.
    """
    key = os.environ.get("FIREWORKS_API_KEY")
    if not key:
        return ("[offline stand-in - set FIREWORKS_API_KEY to run real inference "
                "via the Fireworks AI API]")
    try:
        import json
        import urllib.request
        req = urllib.request.Request(
            "https://api.fireworks.ai/inference/v1/chat/completions",
            data=json.dumps({
                "model": FIREWORKS_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 200,
            }).encode(),
            headers={"Authorization": f"Bearer {key}",
                     "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as r:
            data = json.loads(r.read())
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:  # pragma: no cover - network dependent
        return f"[inference error: {e}]"


def hr(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def main() -> None:
    print(WARNING)
    print(f"Inference target : Fireworks AI API (hackathon inference partner) - {FIREWORKS_MODEL}")

    store = open_store(os.environ.get("PERSEUS_VAULT_DB", ":memory:"))
    print(f"Memory backend   : {store.backend} (data_source={store.data_source})")

    # --- Session 1: the agent learns durable facts about its user/project ---
    hr("SESSION 1  -  the agent learns things worth remembering")
    now = time.time()
    facts = [
        ("preference", "The user prefers concise answers and works in the CET timezone."),
        ("project", "Project Perseus deploys inference to AMD Instinct MI300X via ROCm."),
        ("preference", "The user wants all GPU numbers labelled measured vs published-spec."),
        ("contact", "The AMD Developer Cloud POC for the team is on the amd-cloud-1 region."),
        ("decision", "We chose Fireworks AI to serve Llama-3.1-70B on MI300X for the demo."),
    ]
    for cat, text in facts:
        store.remember(cat, text, now=now)
        print(f"  remembered [{cat}] {text}")
    print(f"  -> {store.count()} memories persisted (survive session end; 0 GPU HBM used)")

    # --- Session 2: fresh context window, agent recalls before answering ---
    hr("SESSION 2  -  new session, empty context window; recall THEN infer")
    question = "Where do we run inference, and how should I present GPU numbers?"
    print(f"  user asks: {question!r}")
    hits = store.recall("inference GPU numbers present", k=3, now=now + 3600)
    print("  Perseus Vault recall (BM25 + recency, CPU-side):")
    for h in hits:
        print(f"    - {h.memory.text}")
    grounding = "\n".join(f"- {h.memory.text}" for h in hits)
    answer = infer(f"Using only these remembered facts:\n{grounding}\n\n"
                   f"Answer briefly: {question}")
    print("\n  agent answer (open-weight LLM via Fireworks AI):")
    print(f"    {answer}")

    # --- Load: recall under a burst, the way a busy multi-agent host behaves ---
    hr("LOAD  -  recall under a burst  [data_source: measured, CPU]")
    for i in range(2000):
        store.remember("bulk", f"log line {i}: routed task {i % 5} via tool {i % 6}",
                       now=now)
    n = store.count()
    probes = ["inference AMD", "user timezone preference", "deploy region",
              "serve Llama Fireworks", "concise answers"]
    t0 = time.perf_counter()
    iters = 500
    for i in range(iters):
        store.recall(probes[i % len(probes)], k=5, now=now + 7200)
    dt = time.perf_counter() - t0
    print(f"  burst inserted    : 2000 bulk memories")
    print(f"  store size        : {n} active (engine-reported)")
    if store.backend != "reference":
        print("  note              : the shipping engine keeps a bounded buffer of recent")
        print("                      items by design (noise control); the reference store")
        print("                      keeps everything until decay.")
    print(f"  recalls           : {iters}")
    print(f"  wall time         : {dt*1000:.1f} ms")
    print(f"  throughput        : {iters/dt:,.0f} recall ops/sec  [measured, this CPU]")
    print(f"  mean latency      : {dt/iters*1000:.3f} ms/recall  [measured, this CPU]")

    # --- Decay: noise fades, signal stays; audit trail preserved ---
    hr("DECAY  -  rarely-recalled memories fade and archive (not deleted)")
    before = store.count()
    archived = store.decay(now=now + 86400 * 120)  # jump 120 days ahead
    after = store.count()
    print(f"  active before     : {before}")
    print(f"  archived this tick: {archived}")
    print(f"  active after      : {after}  (durable, high-signal facts retained)")

    store.close()

    hr("WHY THIS MATTERS ON AMD INSTINCT")
    print("  The MI300X's 192 GB HBM3 is best spent on model weights + KV cache, not on")
    print("  storing what the agent knows. Perseus Vault keeps memory on the host CPU:")
    print("  every byte of HBM stays available to serve tokens, and one MI300X can back")
    print("  many concurrent agents. See docs/ARCHITECTURE.md and `python src/benchmark.py`.")
    print("\n" + WARNING)


if __name__ == "__main__":
    main()
