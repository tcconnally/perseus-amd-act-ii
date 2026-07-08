"""
perseus_vault_store - a faithful, runnable reference of Perseus Vault's recall path.

Perseus Vault (https://github.com/Perseus-Computing-LLC/perseus-vault) is a single
Rust binary that gives AI agents durable, local-first, encrypted memory over an
MCP (Model Context Protocol) interface. Its recall path is *SQLite + FTS5 hybrid
search* - BM25 lexical ranking with no embeddings and no external vector database.

This module ships two backends behind one interface so the demo and benchmarks in
this repo run *anywhere* - including inside the ROCm container - with no GPU and no
external services:

  1. ReferenceStore  - a pure-stdlib SQLite/FTS5 store that mirrors the real
     engine's recall path (BM25 lexical recall + recency decay). Numbers it
     produces are REAL, MEASURED on whatever CPU you run it on. This is the
     default and needs nothing but CPython.

  2. BinaryStore     - a thin MCP/stdio bridge to a real `perseus-vault` binary,
     used automatically when the PERSEUS_VAULT_BIN environment variable points at
     one. This is how you would reproduce the numbers against the shipping engine.

IMPORTANT (honesty rule for this submission): nothing in this module runs on a GPU.
Perseus Vault's memory layer is deliberately CPU-resident - see docs/ARCHITECTURE.md
for why that is the whole point of the AMD Instinct economics story. Any MI300X /
ROCm figure printed by this repo is a published-spec estimate or projection, never a
measurement, and is labelled as such at the point of use.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from typing import Iterable


@dataclass
class Memory:
    """One thing an agent knows."""

    id: str
    category: str
    text: str
    created_at: float
    last_recalled_at: float
    recall_count: int = 0
    score: float = 1.0
    archived: bool = False


@dataclass
class RecallHit:
    memory: Memory
    rank: float  # lower is better (BM25 distance-like); fused with recency


def _fts_query(text: str) -> str:
    """Turn free text into a safe FTS5 OR-query of its word tokens.

    Mirrors how Perseus Vault tokenizes a recall query before handing it to FTS5:
    we never pass raw user text (which can contain FTS operators) straight through.
    """
    toks = re.findall(r"[A-Za-z0-9_]+", text.lower())
    if not toks:
        return '""'
    return " OR ".join(toks)


class ReferenceStore:
    """CPU-only SQLite/FTS5 reference of Perseus Vault's recall path.

    Backend tag: ``measured`` - everything this class reports is measured live on
    the host CPU. It is a faithful *shape* of the real engine (BM25 + recency
    decay), not the shipping Rust binary; use BinaryStore for that.
    """

    backend = "reference"
    data_source = "measured"

    def __init__(self, path: str = ":memory:") -> None:
        self.db = sqlite3.connect(path)
        self.db.executescript(
            """
            PRAGMA journal_mode=WAL;
            PRAGMA synchronous=NORMAL;
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                category TEXT NOT NULL,
                text TEXT NOT NULL,
                created_at REAL NOT NULL,
                last_recalled_at REAL NOT NULL,
                recall_count INTEGER NOT NULL DEFAULT 0,
                score REAL NOT NULL DEFAULT 1.0,
                archived INTEGER NOT NULL DEFAULT 0
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts
                USING fts5(text, content='memories', content_rowid='rowid');
            CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
                INSERT INTO memories_fts(rowid, text) VALUES (new.rowid, new.text);
            END;
            CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
                INSERT INTO memories_fts(memories_fts, rowid, text)
                    VALUES('delete', old.rowid, old.text);
            END;
            CREATE INDEX IF NOT EXISTS idx_mem_cat ON memories(category, archived);
            """
        )
        self.db.commit()

    # ---- write path -------------------------------------------------------
    def remember(self, category: str, text: str, now: float | None = None) -> str:
        now = time.time() if now is None else now
        mid = uuid.uuid4().hex
        self.db.execute(
            "INSERT INTO memories(id,category,text,created_at,last_recalled_at) "
            "VALUES(?,?,?,?,?)",
            (mid, category, text, now, now),
        )
        return mid

    def remember_many(self, rows: Iterable[tuple[str, str]], now: float | None = None) -> int:
        now = time.time() if now is None else now
        n = 0
        cur = self.db.cursor()
        for category, text in rows:
            cur.execute(
                "INSERT INTO memories(id,category,text,created_at,last_recalled_at) "
                "VALUES(?,?,?,?,?)",
                (uuid.uuid4().hex, category, text, now, now),
            )
            n += 1
        self.db.commit()
        return n

    # ---- recall path (BM25 lexical + recency fusion) ----------------------
    def recall(self, query: str, k: int = 5, now: float | None = None) -> list[RecallHit]:
        now = time.time() if now is None else now
        rows = self.db.execute(
            """
            SELECT m.id, m.category, m.text, m.created_at, m.last_recalled_at,
                   m.recall_count, m.score, m.archived, bm25(memories_fts) AS rank
            FROM memories_fts
            JOIN memories m ON m.rowid = memories_fts.rowid
            WHERE memories_fts MATCH ? AND m.archived = 0
            ORDER BY rank
            LIMIT ?
            """,
            (_fts_query(query), k * 4),
        ).fetchall()

        hits: list[RecallHit] = []
        for r in rows:
            mem = Memory(
                id=r[0], category=r[1], text=r[2], created_at=r[3],
                last_recalled_at=r[4], recall_count=r[5], score=r[6],
                archived=bool(r[7]),
            )
            # Reciprocal-rank-style fusion of lexical rank with recency, the way
            # Perseus Vault blends BM25 with a freshness prior. Half-life ~ 30 days.
            age_days = max(0.0, (now - mem.last_recalled_at) / 86400.0)
            recency = 0.5 ** (age_days / 30.0)
            fused = r[8] - recency  # lower rank is better; recency pulls it down
            hits.append(RecallHit(memory=mem, rank=fused))

        hits.sort(key=lambda h: h.rank)
        top = hits[:k]

        # Touch recalled memories so recency/decay reflect real usage.
        if top:
            ids = [h.memory.id for h in top]
            qmarks = ",".join("?" * len(ids))
            self.db.execute(
                f"UPDATE memories SET recall_count = recall_count + 1, "
                f"last_recalled_at = ? WHERE id IN ({qmarks})",
                (now, *ids),
            )
            self.db.commit()
        return top

    # ---- lifecycle: decay + archive --------------------------------------
    def decay(self, now: float | None = None, half_life_days: float = 30.0,
              archive_below: float = 0.15) -> int:
        """Age every memory's score; archive the ones that fall below threshold.

        Mirrors Perseus Vault's decay tick: rarely-recalled memories fade and are
        archived (not deleted), so noise stops crowding recall while history stays
        auditable. Returns the number archived this tick.
        """
        now = time.time() if now is None else now
        self.db.execute(
            """
            UPDATE memories
            SET score = (1.0 + recall_count) *
                        pow(0.5, ((? - last_recalled_at) / 86400.0) / ?)
            WHERE archived = 0
            """,
            (now, half_life_days),
        )
        cur = self.db.execute(
            "UPDATE memories SET archived = 1 WHERE archived = 0 AND score < ?",
            (archive_below,),
        )
        self.db.commit()
        return cur.rowcount

    def count(self, include_archived: bool = False) -> int:
        if include_archived:
            return self.db.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        return self.db.execute(
            "SELECT COUNT(*) FROM memories WHERE archived = 0"
        ).fetchone()[0]

    def close(self) -> None:
        self.db.close()


class BinaryStore:
    """Bridge to a real `perseus-vault` MCP/stdio binary.

    Backend tag: ``measured`` - but against the shipping Rust engine rather than
    the reference. Activated when PERSEUS_VAULT_BIN is set. Kept intentionally
    minimal (remember + recall) so the demo can prove real-engine parity; the full
    55-tool surface is documented at the upstream repo.
    """

    backend = "perseus-vault-binary"
    data_source = "measured"

    def __init__(self, binary: str, db_path: str) -> None:
        self.proc = subprocess.Popen(
            [binary, "serve", "--db", db_path],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1,
        )
        self._id = 0

    def _rpc(self, method: str, params: dict) -> dict:
        self._id += 1
        req = {"jsonrpc": "2.0", "id": self._id, "method": method, "params": params}
        assert self.proc.stdin and self.proc.stdout
        self.proc.stdin.write(json.dumps(req) + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        return json.loads(line) if line else {}

    def remember(self, category: str, text: str, now: float | None = None) -> str:
        res = self._rpc("tools/call", {
            "name": "mimir_remember",
            "arguments": {"category": category, "key": uuid.uuid4().hex,
                          "body_json": json.dumps({"text": text})},
        })
        return json.dumps(res.get("result", {}))

    def recall(self, query: str, k: int = 5, now: float | None = None) -> list[RecallHit]:
        res = self._rpc("tools/call", {
            "name": "mimir_recall", "arguments": {"query": query, "limit": k},
        })
        hits: list[RecallHit] = []
        for i, item in enumerate(res.get("result", {}).get("results", [])):
            text = item.get("body_json", item.get("text", ""))
            hits.append(RecallHit(
                memory=Memory(id=item.get("id", str(i)), category=item.get("category", ""),
                              text=str(text), created_at=0.0, last_recalled_at=0.0),
                rank=float(i),
            ))
        return hits

    def close(self) -> None:
        self.proc.terminate()


def open_store(db_path: str = ":memory:"):
    """Return the best available store: the real binary if configured, else the
    CPU reference. Both report the same interface and both are ``measured``."""
    binary = os.environ.get("PERSEUS_VAULT_BIN")
    if binary and os.path.exists(binary):
        real_path = db_path if db_path != ":memory:" else "perseus-amd-demo.db"
        return BinaryStore(binary, real_path)
    return ReferenceStore(db_path)
