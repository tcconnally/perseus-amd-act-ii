"""Narration segments for the Perseus Vault AMD Act II demo (one per scene).

Numbers referenced here match docs/BENCHMARKS.md in the repo:
  - recall/footprint (scene 4)  -> §1/§2  data_source: measured (AMD CPU reference)
  - cost economics   (scene 5)  -> §3     data_source: projection (published-spec inputs)
"""

NARRATION = [
    # Scene 1 (title)
    "Perseus Vault. Agentic memory, running on AMD Instinct M I 300 X.",
    # Scene 2 (what it is)
    "Perseus Vault is a single Rust binary. Hybrid recall powered by SQLite "
    "full text search, twenty seven tools over the Model Context Protocol, "
    "and it all runs on one M I 300 X accelerator.",
    # Scene 3 (store / recall / decay under load)
    "Watch it work under load. Store a memory, recall it, and let unused "
    "entries decay. The key moment: a fact written in one session is recalled "
    "cleanly across a session boundary. Memory that survives the context window.",
    # Scene 4 (benchmark table) — MEASURED on AMD CPU reference
    "Here is recall throughput, measured on an AMD CPU reference build. At one "
    "thousand entries, over five thousand recalls per second at sub millisecond "
    "latency. At one hundred thousand entries, recall holds near twelve "
    "milliseconds, with the whole store just twenty six megabytes on disk, on "
    "the host, not in GPU memory. These recall numbers are measured, not projected.",
    # Scene 5 (cost table) — PROJECTION from published-spec inputs
    "Now the economics, a projection from published datasheet specs. The M I 300 "
    "X's one hundred ninety two gigabytes of HBM fits a seventy billion parameter "
    "model on one card and serves about twenty concurrent agents, at thirteen cents "
    "per agent hour, roughly eight times cheaper than an H 100. Perseus Vault's "
    "memory stays on the CPU, using none of that HBM.",
    # Scene 6 (closing)
    "Perseus Vault, on AMD Instinct. Code and full methodology are in the repository "
    "below. Recall is measured on an AMD CPU; cost is a projection from published "
    "specs. Real M I 300 X numbers are pending cloud credits.",
]
