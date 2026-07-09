"""
benchmark - measured throughput + footprint tables, plus published-spec economics.

Run:  python src/benchmark.py            # 1K / 10K / 100K, prints tables
      python src/benchmark.py --quick    # 1K / 10K only (fast, for CI/Docker)

Everything under "MEASURED" is timed live on the CPU you run this on and is fully
reproducible. Everything under "PUBLISHED-SPEC ESTIMATES" comes from vendor
datasheets and cloud price lists (cited in README) - it is NOT measured. The banner
below prints on every run so no reader can mistake a projection for a measurement.
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from economics import (  # noqa: E402
    MODEL_NAME, CONTEXT_TOKENS, KV_GB_PER_SEQ, MI300X, H100_SXM, A100_80,
    economics_rows, perseus_vault_cost_per_agent_hr, PV_RSS_MB_PER_AGENT,
    PV_DISK_MB_PER_AGENT,
)
from perseus_vault_store import ReferenceStore  # noqa: E402

WARNING = (
    "NOTE: the GPU table below is the cross-vendor PROJECTION from published-spec "
    "inputs. The MI300X deployment shape itself was measured on a rented MI300X "
    "(15.3 agents, $0.143/agent-hr; docs/BENCHMARKS.md S3a). CPU rows here are "
    "measured live on this machine."
)

CORPUS_TEMPLATES = [
    "user {i} prefers {tool} for {task} and works in the {tz} timezone",
    "project {i} deploys to {region} with owner contact dr {name}",
    "meeting note {i}: decided to ship {tool} for {task} by quarter {q}",
    "fact {i}: the {task} pipeline depends on service {name} in {region}",
    "preference {i}: agent should summarize {task} using {tool}",
]
TOOLS = ["ripgrep", "sqlite", "rocm", "vllm", "perseus-vault", "fireworks"]
TASKS = ["retrieval", "captioning", "routing", "planning", "summarization"]
REGIONS = ["us-east-1", "eu-west-1", "amd-cloud-1", "us-west-2"]
NAMES = ["thorne", "vela", "okafor", "singh", "romano", "chen"]
TZS = ["UTC", "PST", "CET", "IST"]
QS = ["Q1", "Q2", "Q3", "Q4"]


def _row(i: int) -> tuple[str, str]:
    t = CORPUS_TEMPLATES[i % len(CORPUS_TEMPLATES)]
    return ("seed", t.format(i=i, tool=TOOLS[i % len(TOOLS)], task=TASKS[i % len(TASKS)],
                             region=REGIONS[i % len(REGIONS)], name=NAMES[i % len(NAMES)],
                             tz=TZS[i % len(TZS)], q=QS[i % len(QS)]))


def _rss_mb() -> float | None:
    """Best-effort resident-set size in MB (Linux/macOS via resource; else None)."""
    try:
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux reports KB, macOS reports bytes.
        return peak / 1024.0 if sys.platform != "darwin" else peak / (1024.0 * 1024.0)
    except Exception:
        return None


def bench_size(n: int, queries: int = 200) -> dict:
    """Load n memories, then measure recall throughput + footprint. All measured."""
    gc.collect()
    store = ReferenceStore(":memory:")

    t0 = time.perf_counter()
    store.remember_many((_row(i) for i in range(n)))
    load_s = time.perf_counter() - t0
    insert_ops = n / load_s if load_s else 0.0

    # Warm one query, then time a batch of realistic recalls.
    probes = [f"{TASKS[i % len(TASKS)]} {TOOLS[i % len(TOOLS)]}" for i in range(queries)]
    store.recall(probes[0], k=5)
    lat = []
    t0 = time.perf_counter()
    for q in probes:
        s = time.perf_counter()
        store.recall(q, k=5)
        lat.append((time.perf_counter() - s) * 1000.0)
    total_s = time.perf_counter() - t0
    lat.sort()
    p50 = lat[len(lat) // 2]
    p99 = lat[min(len(lat) - 1, int(len(lat) * 0.99))]
    recall_ops = queries / total_s if total_s else 0.0

    # Decay tick over the whole store.
    t0 = time.perf_counter()
    archived = store.decay()
    decay_s = time.perf_counter() - t0

    # Footprint.
    db_bytes = store.db.execute(
        "SELECT page_count * page_size FROM pragma_page_count(), pragma_page_size()"
    ).fetchone()[0]
    rss = _rss_mb()
    store.close()

    return {
        "entries": n,
        "insert_ops_s": round(insert_ops, 0),
        "recall_ops_s": round(recall_ops, 0),
        "recall_p50_ms": round(p50, 3),
        "recall_p99_ms": round(p99, 3),
        "decay_tick_s": round(decay_s, 3),
        "decay_archived": archived,
        "db_mb": round(db_bytes / (1024 * 1024), 2),
        "rss_mb": round(rss, 1) if rss is not None else None,
        "data_source": "measured",
    }


def _fmt(rows: list[dict], cols: list[tuple[str, str]]) -> str:
    header = " | ".join(h for _, h in cols)
    sep = " | ".join("---" for _ in cols)
    lines = [f"| {header} |", f"| {sep} |"]
    for r in rows:
        cells = []
        for key, _ in cols:
            v = r.get(key)
            cells.append("n/a" if v is None else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="skip the 100K row")
    args = ap.parse_args()

    sizes = [1_000, 10_000] if args.quick else [1_000, 10_000, 100_000]

    print("=" * 78)
    print("Perseus Vault  x  AMD Instinct - agentic memory benchmark")
    print("=" * 78)
    print(WARNING)
    print(f"Backend: bundled reference store, SQLite/FTS5 (CPU, {sys.platform}, "
          f"py{sys.version_info.major}.{sys.version_info.minor}). Shipping-engine numbers "
          "come from the upstream repo's PERF.md (see docs/BENCHMARKS.md).")
    print()

    results = [bench_size(n) for n in sizes]

    print("### 1. Recall throughput & latency  [data_source: measured]")
    print(_fmt(results, [
        ("entries", "Entries"),
        ("recall_ops_s", "Recall ops/s"),
        ("recall_p50_ms", "p50 ms"),
        ("recall_p99_ms", "p99 ms"),
        ("insert_ops_s", "Insert ops/s"),
    ]))
    print()

    print("### 2. Memory footprint  [data_source: measured]")
    print(_fmt(results, [
        ("entries", "Entries"),
        ("rss_mb", "Process RSS (MB)"),
        ("db_mb", "DB file (MB)"),
        ("decay_tick_s", "Decay tick (s)"),
        ("decay_archived", "Archived"),
    ]))
    print()

    print("### 3. Cost economics - one accelerator serves N agents  "
          "[data_source: published-spec inputs -> projection]")
    print(f"Model: {MODEL_NAME}  |  context: {CONTEXT_TOKENS} tok  |  "
          f"KV/seq: {KV_GB_PER_SEQ:.2f} GB  |  "
          f"Perseus Vault memory: {PV_RSS_MB_PER_AGENT:.0f} MB RSS + "
          f"{PV_DISK_MB_PER_AGENT:.0f} MB disk / agent (measured, host CPU)")
    econ = list(economics_rows())
    print(_fmt(econ, [
        ("gpu", "Accelerator"),
        ("hbm_gb", "HBM (GB)"),
        ("cards_for_weights", "Cards for weights"),
        ("concurrent_agents", "Concurrent agents"),
        ("gpu_usd_per_hr", "GPU $/hr"),
        ("gpu_usd_per_agent_hr", "GPU $/agent-hr"),
        ("pv_memory_usd_per_agent_hr", "PV mem $/agent-hr"),
    ]))
    print()
    print("Takeaway [projection]: the MI300X's 192 GB HBM3 fits Llama-3.1-70B on ONE "
          "card and\nstill leaves room for the most concurrent agents, giving the "
          "lowest GPU $/agent-hr.\nPerseus Vault memory lives on the host CPU "
          f"(~${perseus_vault_cost_per_agent_hr():.5f}/agent-hr), so it consumes\nZERO "
          "HBM - none of that 192 GB is spent storing memory instead of serving tokens.")
    print()
    print(WARNING)


if __name__ == "__main__":
    main()
