"""
economics - the "one MI300X serves N agents" math for Perseus Vault memory.

EVERY number in this file is tagged with a data_source:
    published-spec : vendor datasheet / cloud price list (cited in README)
    projection     : derived here from published-spec inputs + stated assumptions
    measured       : reproducible on real hardware (see benchmark.py)

MEASURED VALIDATION (2026-07-09): we rented a real AMD Instinct MI300X and measured
the deployment shape this module projects — serving Qwen2.5-72B bf16 on one card via
vLLM 0.19.1/ROCm 7.13 gave 15.3 concurrent 8K-token agents at $0.143/agent-hour
($2.19/GPU-hr), vs the ~$0.133/agent-hr projected below (docs/BENCHMARKS.md §3a).
We then rented 2x NVIDIA H100 SXM and measured the cross-vendor claim too: a single
H100 cannot load the model; the pair's best-boot case serves 5.0 agents at
$1.68/agent-hour -> 11.7x measured-vs-measured (§3b; the ~7.8x projection below
UNDERSTATED the advantage). We also rented and measured 8x A100 40GB (§3d): 57.9 agents
at $0.275/agent-hr — 8 cards vs 1, so MI300X still wins 1.9x. And 2x A100 80GB (§3c,
eager@0.97): 6.37 agents / $0.436 (MI300X 3.0x cheaper). EVERY cross-vendor row is now
measured — no projection remains.
The thesis this module quantifies is deliberately CPU-side and therefore honest:
Perseus Vault keeps the agent memory layer OFF the GPU, so 100% of an accelerator's
HBM and FLOPs go to inference while durable memory is served for cents on the host
CPU. The tables below make that trade explicit and reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GPU:
    name: str
    hbm_gb: float           # published-spec
    bandwidth_tbs: float    # published-spec
    fp16_tflops: float      # published-spec (peak theoretical)
    tdp_w: int              # published-spec
    usd_per_hr: float       # published-spec (2026 cloud median/typical, cited)


# --- published-spec inputs (see README "Published-Spec Estimates" for citations) ---
MI300X = GPU("AMD Instinct MI300X", 192.0, 5.325, 1307.4, 750, 2.72)
H100_SXM = GPU("NVIDIA H100 SXM", 80.0, 3.35, 989.0, 700, 3.93)
A100_80 = GPU("NVIDIA A100 80GB SXM", 80.0, 2.039, 312.0, 400, 1.80)

# --- workload assumption (projection inputs, stated openly) ---
# Serving one large open-weight model in FP16 and asking how many concurrent agent
# sessions each accelerator can hold. Llama-3.1-70B is the canonical "fits on one
# MI300X, does not fit on one H100/A100" case - that single-GPU headroom is the
# well-documented MI300X advantage we are leaning on.
MODEL_NAME = "Llama-3.1-70B (FP16)"
MODEL_WEIGHTS_GB = 141.0        # 70.6B params x 2 bytes
CONTEXT_TOKENS = 8192

# --- MEASURED deployment (2026-07-09, Qwen2.5-72B-Instruct bf16, vLLM 0.19.1) ---
# The live cost table below is MEASURED on real rented hardware, not projected:
# one AMD Instinct MI300X (§3a) and one 2x NVIDIA H100 SXM pair (§3b) in
# docs/BENCHMARKS.md, plus 8x A100 40GB (§3d) and 2x A100 80GB (§3c). Every row is measured.
MEASURED_MODEL_NAME = "Qwen2.5-72B-Instruct (bf16)"
MEASURED_WEIGHTS_GB = 135.5     # measured on GPU (vLLM)
# KV cache per token for Llama-3.1-70B with GQA (80 layers, 8 KV heads, head_dim 128,
# fp16, k+v): 2 * 80 * 8 * 128 * 2 bytes = 327,680 B ~= 0.3125 MiB/token.
KV_BYTES_PER_TOKEN = 2 * 80 * 8 * 128 * 2
KV_GB_PER_SEQ = (KV_BYTES_PER_TOKEN * CONTEXT_TOKENS) / (1024 ** 3)   # ~2.5 GiB/seq

# --- Perseus Vault memory footprint (MEASURED, upstream repo, AMD CPU) ---
# Source: perseus-vault README "Stress Test: 100K entities" + PERF.md, measured on
# an AMD 16-core CPU (AMD Family 26), Windows 11. Per-agent store of 100K memories.
PV_RSS_MB_PER_AGENT = 85.0      # measured
PV_DISK_MB_PER_AGENT = 45.0     # measured
HOST_RAM_USD_PER_GB_HR = 0.005  # published-spec (typical cloud host RAM price)


def agents_per_card(gpu: GPU) -> float:
    """Concurrent agent sessions a single card can hold for the model above.

    projection: (HBM - weights) / KV-per-sequence, floored at 0 when the weights
    do not even fit on one card.
    """
    free = gpu.hbm_gb - MODEL_WEIGHTS_GB
    if free <= 0:
        return 0.0
    return free / KV_GB_PER_SEQ


def cards_needed(gpu: GPU) -> int:
    """How many of this card it takes just to hold the weights (projection)."""
    import math
    return max(1, math.ceil(MODEL_WEIGHTS_GB / gpu.hbm_gb))


def agents_per_deployment(gpu: GPU):
    """Return (n_cards, total_agents, usd_per_hr, usd_per_agent_hr) - projection.

    Uses the smallest card count that fits the weights, then fills remaining HBM
    across those cards with KV cache.
    """
    n = cards_needed(gpu)
    total_hbm = gpu.hbm_gb * n
    free = total_hbm - MODEL_WEIGHTS_GB
    agents = max(0.0, free / KV_GB_PER_SEQ)
    cost = gpu.usd_per_hr * n
    per_agent = cost / agents if agents > 0 else float("inf")
    return n, agents, cost, per_agent


def perseus_vault_cost_per_agent_hr() -> float:
    """Host-side $/agent-hr for Perseus Vault memory (projection from measured RSS).

    This is what the memory layer costs - and it is spent on the CPU host, not the
    GPU, so it does NOT reduce the KV-cache budget above.
    """
    return (PV_RSS_MB_PER_AGENT / 1024.0) * HOST_RAM_USD_PER_GB_HR


def economics_rows():
    """Yield dict rows for the live cost table.

    MI300X, both H100 rows, and the 8x A100 40GB row are MEASURED on real rented
    hardware (Qwen2.5-72B-Instruct bf16, vLLM 0.19.1 — docs/BENCHMARKS.md §3a/§3b/§3d).
    The 2x A100 80GB row is now measured too (eager@0.97, matches 2x H100). Every
    cross-vendor row is measured — no projection remains. ``concurrent_agents`` is
    None when the deployment cannot even load the model (1x H100 → CUDA OOM).
    """
    pv = round(perseus_vault_cost_per_agent_hr(), 5)

    def row(gpu, hbm, cards, agents, gpu_hr, per_agent, source, note):
        return {
            "gpu": gpu,
            "hbm_gb": hbm,
            "cards_for_weights": cards,
            "concurrent_agents": agents,
            "gpu_usd_per_hr": gpu_hr,
            "gpu_usd_per_agent_hr": per_agent,
            "pv_memory_usd_per_agent_hr": pv,
            "data_source": source,
            "note": note,
        }

    # MI300X — one card holds 72B bf16 with KV headroom to spare (§3a).
    yield row("AMD Instinct MI300X", 192.0, 1, 15.3, 2.19, 0.143,
              "measured", "one card holds 72B + 38 GiB KV to spare")
    # 1x H100 — cannot load the model at all (CUDA OOM) (§3b).
    yield row("NVIDIA H100 SXM (1 card)", 80.0, 1, None, 3.93, None,
              "measured", "cannot load the model — CUDA OOM")
    # 2x H100 — best case that boots: eager-only at 97% util (§3b).
    yield row("NVIDIA H100 SXM (2 cards)", 80.0, 2, 5.0, 8.38, 1.68,
              "measured", "best case that boots (eager, 97% util)")
    # 8x A100 40GB — MEASURED on Lambda (§3d). 320 GB is heavily overprovisioned for a
    # 72B (~136 GB), so huge KV headroom → high per-agent count, but it's 8 cards vs 1.
    yield row("NVIDIA A100 40GB SXM4 (8 cards)", 40.0, 8, 57.9, 15.92, 0.275,
              "measured", "8 cards, 320 GB — overprovisioned; 7.2 agents/card")
    # 2x A100 80GB — MEASURED on RunPod (§3c), eager@0.97 to match the 2x H100 best-case
    # (projection was 7.6 agents / $0.474 — validated: measured 6.37 / $0.436).
    yield row("NVIDIA A100 80GB SXM4 (2 cards)", 80.0, 2, 6.37, 2.78, 0.436,
              "measured", "eager@0.97, matches 2x H100 methodology")


if __name__ == "__main__":
    print(f"Live table model: {MEASURED_MODEL_NAME}  weights={MEASURED_WEIGHTS_GB} GB "
          f"[measured: MI300X + 2xH100 + 8xA100 40GB + 2xA100 80GB — no projection rows]")
    print(f"Projection model:  {MODEL_NAME}  weights={MODEL_WEIGHTS_GB} GB  "
          f"ctx={CONTEXT_TOKENS}  KV/seq={KV_GB_PER_SEQ:.2f} GB")
    print(f"Perseus Vault memory: {PV_RSS_MB_PER_AGENT} MB RSS / agent  "
          f"(~${perseus_vault_cost_per_agent_hr():.5f}/agent-hr, host CPU) [measured footprint]")
    print()
    for row in economics_rows():
        print(row)
