"""Narration segments for the Perseus Vault AMD Act II demo (one per scene).

Numbers referenced here match docs/BENCHMARKS.md in the repo:
  - MI300X contention (scene 4) -> §1  data_source: measured (real MI300X)
  - recall/footprint  (scene 5) -> §1/§2  data_source: measured (AMD CPU reference)
  - cost economics    (scene 6) -> §3  data_source: projection (published-spec inputs)
"""

NARRATION = [
    # Scene 1 (title)
    "Perseus Vault. Encrypted agent memory that stays off the GPU, so the AMD "
    "Instinct M I 300 X serves tokens, not storage.",
    # Scene 2 (problem + what it is)
    "Agents forget everything when the session ends. Bolt on a vector database and "
    "it eats the very HBM you bought for inference. Perseus Vault is a single Rust "
    "binary, SQLite full text recall, fifty five tools over the Model Context "
    "Protocol, and it keeps memory on the host CPU. Two markets need this: teams "
    "burning tokens re-feeding context, and regulated teams who can't put memory in "
    "the cloud at all.",
    # Scene 3 (store / recall / decay under load)
    "Watch it work. Store a memory, recall it, and let unused entries decay. The key "
    "moment: a fact written in one session is recalled cleanly across a session "
    "boundary. Memory that survives the context window.",
    # Scene 4 (MEASURED on a real MI300X) — the hero result
    "Here is the proof, measured on a real AMD Instinct M I 300 X. We drove the "
    "accelerator to one hundred percent utilization, ninety seven teraflops, and "
    "measured recall on the host CPU. It moved zero point six percent, from nineteen "
    "point nine six to twenty point one milliseconds. The memory layer and the "
    "accelerator never compete. Measured, on real AMD hardware, and reproducible.",
    # Scene 5 (throughput + footprint) — MEASURED on AMD CPU reference
    "Recall scales, too. On an AMD CPU reference build, one hundred thousand memories "
    "recall near twelve milliseconds, the whole store just twenty six megabytes on "
    "disk, on the host, not in GPU memory. Measured, not projected.",
    # Scene 6 (cost table) — PROJECTION from published-spec inputs
    "The economics, a projection from published datasheet specs. The M I 300 X's "
    "one hundred ninety two gigabytes of HBM fits a seventy billion parameter model "
    "on one card and serves about twenty concurrent agents, at thirteen cents per "
    "agent hour, roughly eight times cheaper than an H 100. Perseus Vault's memory "
    "stays on the CPU, using none of that HBM.",
    # Scene 7 (closing)
    "Perseus Vault, on AMD Instinct. Recall is measured on a real M I 300 X; the cost "
    "economics are a projection from published specs, clearly labelled. Code and full "
    "methodology are in the repository below.",
]
