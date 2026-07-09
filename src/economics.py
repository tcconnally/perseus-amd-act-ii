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
Cross-accelerator (H100/A100) rows remain projections — we rented only MI300X.
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
    """Yield dict rows for the cost table. Tagged so callers can print sources."""
    for gpu in (MI300X, H100_SXM, A100_80):
        n, agents, cost, per_agent = agents_per_deployment(gpu)
        yield {
            "gpu": gpu.name,
            "hbm_gb": gpu.hbm_gb,
            "cards_for_weights": n,
            "concurrent_agents": round(agents, 1),
            "gpu_usd_per_hr": round(cost, 2),
            "gpu_usd_per_agent_hr": (round(per_agent, 3) if per_agent != float("inf") else None),
            "pv_memory_usd_per_agent_hr": round(perseus_vault_cost_per_agent_hr(), 5),
            "data_source": "projection",
        }


if __name__ == "__main__":
    print(f"Model: {MODEL_NAME}  weights={MODEL_WEIGHTS_GB} GB  "
          f"ctx={CONTEXT_TOKENS}  KV/seq={KV_GB_PER_SEQ:.2f} GB  [projection]")
    print(f"Perseus Vault memory: {PV_RSS_MB_PER_AGENT} MB RSS / agent  "
          f"(~${perseus_vault_cost_per_agent_hr():.5f}/agent-hr, host CPU) [measured footprint]")
    print()
    for row in economics_rows():
        print(row)
