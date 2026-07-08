# Perseus Vault x AMD Instinct - Act II submission
#
# Base image is an official AMD ROCm developer image so this container is ready to
# run inference on AMD Instinct (MI300X) GPUs the moment one is attached. The memory
# layer itself (Perseus Vault) is CPU-resident and needs no GPU - see docs/ARCHITECTURE.md
# for why that separation is the whole economic argument of this submission.
#
# Build:  docker build -t perseus-amd-act-ii .
# Run  :  docker run --rm perseus-amd-act-ii                 # runs the agent demo
#         docker run --rm perseus-amd-act-ii python3 src/benchmark.py --quick
# On an AMD GPU host, add:  --device=/dev/kfd --device=/dev/dri --group-add video
#
# ROCm base (pick a tag your host's ROCm/driver supports). rocm/dev-ubuntu-22.04
# ships ROCm + HIP; we add Python and run entirely on CPU unless a GPU is present.
FROM rocm/dev-ubuntu-22.04:6.2

LABEL org.opencontainers.image.title="perseus-amd-act-ii"
LABEL org.opencontainers.image.description="Perseus Vault - encrypted local-first memory layer for AI agents on AMD Instinct"
LABEL org.opencontainers.image.licenses="MIT"
LABEL org.opencontainers.image.source="https://github.com/tcconnally/perseus-amd-act-ii"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Python 3 (Ubuntu's libsqlite3 is built WITH FTS5, which the memory layer needs).
RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-pip python3-venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first for layer caching. Core demo is stdlib-only; requirements.txt
# holds only the optional Fireworks SDK, so this stays tiny.
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt || true

# App
COPY src/ ./src/
COPY docs/ ./docs/
COPY .env.example ./.env.example
COPY README.md LICENSE AGENTS.md ./

# Prove FTS5 is available at build time (fails the build loudly if not).
RUN python3 -c "import sqlite3; c=sqlite3.connect(':memory:'); \
c.execute('CREATE VIRTUAL TABLE t USING fts5(x)'); \
print('FTS5 OK, sqlite', sqlite3.sqlite_version)"

# Default: run the end-to-end agentic-memory demo (prints the published-spec warning).
CMD ["python3", "src/agent_memory_demo.py"]
