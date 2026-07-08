# Perseus Vault x AMD Instinct — live interactive demo (CPU-only web app).
# Build from the REPO ROOT:  docker build -f webdemo/Dockerfile -t perseus-amd-demo .
# Run:                       docker run --rm -p 8080:8080 perseus-amd-demo
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8080

WORKDIR /app

COPY webdemo/requirements.txt ./webdemo/requirements.txt
RUN pip install --no-cache-dir -r webdemo/requirements.txt

COPY src/ ./src/
COPY webdemo/ ./webdemo/

# Prove FTS5 is present at build time (the recall path needs it).
RUN python -c "import sqlite3; c=sqlite3.connect(':memory:'); \
c.execute('CREATE VIRTUAL TABLE t USING fts5(x)'); \
print('FTS5 OK, sqlite', sqlite3.sqlite_version)"

EXPOSE 8080
# Single worker + single thread: the per-visitor in-memory stores live in-process,
# and SQLite connections stay on one thread. Fine for a low-traffic demo.
CMD ["gunicorn", "-w", "1", "--threads", "1", "-b", "0.0.0.0:8080", \
     "--chdir", "/app/webdemo", "--access-logfile", "-", "app:app"]
