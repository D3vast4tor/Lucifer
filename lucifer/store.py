"""Versioned SQLite document store. All updates are transactions."""
from __future__ import annotations
import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def uid(prefix: str = "id") -> str:
    return f"{prefix}_{uuid4().hex[:16]}"


class Store:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.lock = threading.RLock()
        # One backend owns this database. Revisions invalidate in-memory UI
        # snapshots without re-reading unchanged registries on every heartbeat.
        self._revisions: dict[str, int] = {}
        self._savepoint_serial = 0
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.executescript('''
        CREATE TABLE IF NOT EXISTS documents (
          kind TEXT NOT NULL, id TEXT NOT NULL, data TEXT NOT NULL,
          PRIMARY KEY(kind,id));
        CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        PRAGMA user_version=1;
        ''')
        self.conn.commit()
        self.path.chmod(0o600)

    @contextmanager
    def transaction(self):
        """Atomic nested document changes, including their cache revisions."""
        with self.lock:
            self._savepoint_serial += 1
            name = f"lucifer_store_{self._savepoint_serial}"
            revisions = dict(self._revisions)
            self.conn.execute(f"SAVEPOINT {name}")
            try:
                yield self
            except BaseException:
                self.conn.execute(f"ROLLBACK TO SAVEPOINT {name}")
                self.conn.execute(f"RELEASE SAVEPOINT {name}")
                self._revisions = revisions
                raise
            else:
                self.conn.execute(f"RELEASE SAVEPOINT {name}")

    def put(self, kind: str, data: dict) -> dict:
        data = dict(data)
        data.setdefault("id", uid(kind.rstrip("s")))
        data.setdefault("created_at", now())
        data["updated_at"] = now()
        with self.transaction():
            self.conn.execute("INSERT INTO documents VALUES(?,?,?) ON CONFLICT(kind,id) DO UPDATE SET data=excluded.data", (kind, data["id"], json.dumps(data)))
            self._revisions[kind] = self._revisions.get(kind, 0) + 1
        return data

    def get(self, kind: str, ident: str) -> dict:
        with self.lock:
            row = self.conn.execute("SELECT data FROM documents WHERE kind=? AND id=?", (kind, ident)).fetchone()
        if row is None:
            raise KeyError(f"{kind}: {ident} not found")
        return json.loads(row[0])

    def list(self, kind: str, **filters) -> list[dict]:
        with self.lock:
            rows = self.conn.execute("SELECT data FROM documents WHERE kind=? ORDER BY rowid", (kind,)).fetchall()
        docs = [json.loads(row[0]) for row in rows]
        return [d for d in docs if all(d.get(k) == v for k, v in filters.items())]

    def put_many(self, kind: str, documents: list[dict]):
        records = []
        for doc in documents:
            data = dict(doc)
            data.setdefault("id", uid(kind.rstrip("s")))
            data.setdefault("created_at", now())
            data["updated_at"] = now()
            records.append((kind, data["id"], json.dumps(data)))
        with self.transaction():
            self.conn.executemany("INSERT INTO documents VALUES(?,?,?) ON CONFLICT(kind,id) DO UPDATE SET data=excluded.data", records)
            if records:
                self._revisions[kind] = self._revisions.get(kind, 0) + 1

    def update(self, kind: str, ident: str, **changes) -> dict:
        with self.lock:
            return self.put(kind, {**self.get(kind, ident), **changes})

    def delete(self, kind: str, ident: str):
        with self.transaction():
            self.conn.execute("DELETE FROM documents WHERE kind=? AND id=?", (kind, ident))
            self._revisions[kind] = self._revisions.get(kind, 0) + 1

    def setting(self, key: str, default=None):
        with self.lock:
            row = self.conn.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def set_setting(self, key: str, value):
        with self.transaction():
            self.conn.execute("INSERT INTO settings VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, json.dumps(value)))
            self._revisions["settings"] = self._revisions.get("settings", 0) + 1

    def revision(self, *kinds: str) -> tuple[int, ...]:
        with self.lock:
            return tuple(self._revisions.get(kind, 0) for kind in kinds)

    def audit(self, action: str, **details):
        return self.put("audit", {"action": action, **details})

    def close(self):
        with self.lock:
            self.conn.close()
