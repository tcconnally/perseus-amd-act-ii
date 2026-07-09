"""Narration segments for the Perseus Vault AMD Act II demo (one per scene).

Numbers referenced here match docs/BENCHMARKS.md in the repo:
  - MI300X recall-under-serving (scene 4) -> §3a  data_source: measured (real MI300X)
  - recall/footprint  (scene 5) -> §1/§2  data_source: measured (AMD CPU reference)
  - cost economics    (scene 6) -> §3a + §3b, MEASURED on both vendors
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
    "Here is the proof, measured on a real AMD Instinct M I 300 X while it served a "
    "seventy two billion parameter model on V L L M and ROCm. With the accelerator "
    "saturated, recall on the host CPU moved zero point six percent, eighteen point "
    "seven to eighteen point eight milliseconds, median of six runs. The memory layer "
    "and the accelerator never compete. Measured, under real inference load, and "
    "reproducible from the repository.",
    # Scene 5 (throughput + footprint) — MEASURED on AMD CPU reference
    "Recall scales, too. On an AMD CPU reference build, one hundred thousand memories "
    "recall near twelve milliseconds, the whole store just twenty six megabytes on "
    "disk, on the host, not in GPU memory. Measured, not projected.",
    # Scene 6 (cost) — MEASURED on BOTH vendors (§3a MI300X + §3b 2×H100)
    "The economics, measured on both sides. One M I 300 X holds fifteen point three "
    "concurrent seventy two B agents at fourteen cents per agent hour. We rented two "
    "H 100s and ran the same model, same V L L M: a single H 100 cannot load it at "
    "all, and the pair's best case is five agents at a dollar sixty eight. That is "
    "eleven point seven times cheaper per agent on the M I 300 X. Measured, not "
    "projected. Perseus Vault's memory stays on the CPU, using none of that HBM.",
    # Scene 7 (closing)
    "Perseus Vault, on AMD Instinct. Concurrency, cost per agent hour, recall under "
    "load, and the two H 100 comparison are all measured, and all reproduce from the "
    "repository below with one command.",
]
