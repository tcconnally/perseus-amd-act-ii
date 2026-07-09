"""
gemma_on_amd - the whole agent on one AMD chip: Gemma + Perseus Vault, zero cloud.

The main submission shows the fleet-scale story: big open-weight models serve tokens
from MI300X HBM while Perseus Vault keeps every agent's memory on the host CPU. This
script shows the same architecture scaled DOWN for the hackathon's Gemma building
block (Track 3 "Best AMD-Hosted Gemma Project"): a lightweight open Gemma model and
the encrypted memory layer running side by side on a single AMD processor - no GPU,
no cloud, no API key.

  * Memory   : Perseus Vault reference store (SQLite/FTS5) - or the real
               perseus-vault binary if PERSEUS_VAULT_BIN is set. Host CPU.
  * Inference: Gemma 3 (default gemma-3-4b-it, Q4_K_M GGUF) served locally by
               llama.cpp's OpenAI-compatible server - on the SAME host CPU.

HONESTY: everything this script prints is measured live on the machine you run it
on; it prints the detected CPU so "AMD-hosted" is a fact you can see, not a claim.
Gemma on Fireworks is on-demand (deploy-it-yourself) and bills ~$7/hour even while
idle; rather than pay to keep a model warm, we self-host Gemma on AMD silicon for $0
- which is exactly the off-the-GPU economics point of this submission.

Run it:
  1) Install llama.cpp (e.g. `winget install ggml.llamacpp` / `brew install llama.cpp`)
  2) Download an open Gemma GGUF, e.g.
     https://huggingface.co/ggml-org/gemma-3-4b-it-GGUF/resolve/main/gemma-3-4b-it-Q4_K_M.gguf
  3) llama-server -m gemma-3-4b-it-Q4_K_M.gguf --port 8081 --ctx-size 8192
  4) python3 src/gemma_on_amd.py
Env: GEMMA_BASE_URL (default http://127.0.0.1:8081), GEMMA_MODEL (default gemma-3-4b-it).
"""

from __future__ import annotations

import json
import os
import platform
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from perseus_vault_store import open_store  # noqa: E402

GEMMA_BASE_URL = os.environ.get("GEMMA_BASE_URL", "http://127.0.0.1:8081").rstrip("/")
GEMMA_MODEL = os.environ.get("GEMMA_MODEL", "gemma-3-4b-it")


def host_cpu() -> str:
    """Best-effort human-readable CPU name, so 'AMD-hosted' is visible fact."""
    name = os.environ.get("PROCESSOR_IDENTIFIER", "")
    if sys.platform == "win32":
        try:
            import subprocess
            out = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 "(Get-CimInstance Win32_Processor).Name"],
                capture_output=True, text=True, timeout=15).stdout.strip()
            if out:
                return out
        except Exception:
            pass
    if sys.platform.startswith("linux"):
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.lower().startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except OSError:
            pass
    return name or platform.processor() or platform.machine()


def gemma_chat(prompt: str, max_tokens: int = 120) -> tuple[str, int, float]:
    """Call the local Gemma server. Returns (text, completion_tokens, seconds)."""
    body = json.dumps({
        "model": GEMMA_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens, "temperature": 0.3,
    }).encode()
    req = urllib.request.Request(
        f"{GEMMA_BASE_URL}/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=120) as r:
        d = json.loads(r.read())
    dt = time.perf_counter() - t0
    text = (d["choices"][0]["message"].get("content") or "").strip()
    toks = int(d.get("usage", {}).get("completion_tokens", 0))
    return text, toks, dt


def main() -> None:
    cpu = host_cpu()
    print("=" * 72)
    print("GEMMA ON AMD - the whole agent on one chip (memory + open-weight LLM)")
    print("=" * 72)
    print(f"Host CPU        : {cpu}")
    print(f"Gemma endpoint  : {GEMMA_BASE_URL}  model={GEMMA_MODEL} (local llama.cpp)")

    try:
        with urllib.request.urlopen(f"{GEMMA_BASE_URL}/health", timeout=5) as r:
            r.read()
    except (urllib.error.URLError, OSError):
        print("\nNo local Gemma server found. Start one first (see module docstring):")
        print("  llama-server -m gemma-3-4b-it-Q4_K_M.gguf --port 8081 --ctx-size 8192")
        sys.exit(1)

    store = open_store(os.environ.get("PERSEUS_VAULT_DB", ":memory:"))
    print(f"Memory backend  : {store.backend} (host CPU, 0 bytes of any GPU's memory)")

    # Session 1: learn.
    facts = [
        ("preference", "The user prefers concise answers and works in the CET timezone."),
        ("project", "Project Perseus targets AMD Instinct MI300X for fleet-scale inference."),
        ("decision", "For single-host agents we pair Gemma 3 with Perseus Vault on one AMD CPU."),
        ("constraint", "All GPU numbers must be labelled measured versus published-spec."),
    ]
    print("\nSESSION 1 - learn:")
    for cat, text in facts:
        store.remember(cat, text)
        print(f"  remembered [{cat}] {text}")

    # Session 2: recall on the CPU, then ask Gemma - on the same CPU.
    question = "What model pairing do we use for single-host agents, and why?"
    print(f"\nSESSION 2 - new session; recall then infer:\n  user asks: {question!r}")
    t0 = time.perf_counter()
    hits = store.recall("single-host agents Gemma pairing", k=3)
    recall_ms = (time.perf_counter() - t0) * 1000.0
    for h in hits:
        print(f"  recalled: {h.memory.text}")
    grounding = "\n".join(f"- {h.memory.text}" for h in hits)

    answer, toks, dt = gemma_chat(
        f"Using ONLY these remembered facts:\n{grounding}\n\n"
        f"Answer in one or two sentences: {question}")
    print(f"\n  Gemma answers (local, {GEMMA_MODEL}):\n    {answer}")

    print("\nMEASURED on this host [data_source: measured]:")
    print(f"  recall latency  : {recall_ms:.2f} ms  (Perseus Vault, {store.backend})")
    print(f"  generation      : {toks} tokens in {dt:.2f}s = {toks/dt:,.1f} tok/s "
          f"({GEMMA_MODEL}, llama.cpp, CPU)")
    print(f"  host            : {cpu}")
    print("\nEvery number above was measured live on this machine. If the CPU line")
    print("says AMD, this was an AMD-hosted Gemma agent - memory and model on one chip.")
    store.close()


if __name__ == "__main__":
    main()
