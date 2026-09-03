"""SQLite storage for InfraPulse. Plain sqlite3 — no ORM, nothing to misconfigure."""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timezone

DB_PATH = os.environ.get("INFRAPULSE_DB", "infrapulse.db")

# The status pipeline, in order. A complaint may only ever move forward along it.
STATUSES = ["Submitted", "Assigned", "In Progress", "Resolved"]
LIVE_STATUSES = ["Submitted", "Assigned", "In Progress"]   # everything except Resolved


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=15)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    email         TEXT NOT NULL UNIQUE,
    name          TEXT NOT NULL,
    password_hash TEXT NOT NULL,
    role          TEXT NOT NULL CHECK (role IN ('user', 'staff')),
    category      TEXT,                      -- staff only: their assigned category
    created_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS complaints (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id        INTEGER NOT NULL REFERENCES users(id),
    reporter_name  TEXT NOT NULL,
    address        TEXT NOT NULL,
    description    TEXT NOT NULL,
    photo          TEXT NOT NULL,            -- filename inside the uploads folder

    defect         TEXT NOT NULL,            -- machine name, e.g. cracked_tiles
    defect_name    TEXT NOT NULL,            -- display name, e.g. Cracked Tiles
    category       TEXT NOT NULL,            -- Structural | Functional | Performance
    confidence     REAL NOT NULL,

    severity       REAL NOT NULL,            -- 0..1, from the photograph
    extent         REAL NOT NULL,            -- 0..1, from the photograph
    priority_score REAL NOT NULL,
    priority_band  TEXT NOT NULL,
    explanation    TEXT NOT NULL,            -- the arithmetic, shown in the UI

    status         TEXT NOT NULL,
    created_at     TEXT NOT NULL,
    updated_at     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS status_history (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    complaint_id INTEGER NOT NULL REFERENCES complaints(id),
    status       TEXT NOT NULL,
    changed_by   TEXT NOT NULL,
    changed_at   TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_queue
    ON complaints(category, status, priority_score DESC, created_at ASC);
CREATE INDEX IF NOT EXISTS idx_user ON complaints(user_id, created_at DESC);
"""


def init_db() -> None:
    with connect() as conn:
        conn.executescript(SCHEMA)


# ------------------------------------------------------------------ users
def create_user(email: str, name: str, password_hash: str,
                role: str = "user", category: str | None = None) -> int:
    with connect() as conn:
        cur = conn.execute(
            "INSERT INTO users (email, name, password_hash, role, category, created_at)"
            " VALUES (?,?,?,?,?,?)",
            (email.strip().lower(), name.strip(), password_hash, role, category, now()),
        )
        return int(cur.lastrowid)


def get_user_by_email(email: str):
    with connect() as conn:
        return conn.execute("SELECT * FROM users WHERE email = ?",
                            (email.strip().lower(),)).fetchone()


def get_user(user_id: int):
    with connect() as conn:
        return conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()


# ------------------------------------------------------------------ complaints
def create_complaint(user_id: int, reporter_name: str, address: str, description: str,
                     photo: str, analysis: dict, priority: dict) -> int:
    ts = now()
    with connect() as conn:
        cur = conn.execute(
            """INSERT INTO complaints
               (user_id, reporter_name, address, description, photo,
                defect, defect_name, category, confidence,
                severity, extent, priority_score, priority_band, explanation,
                status, created_at, updated_at)
               VALUES (?,?,?,?,?, ?,?,?,?, ?,?,?,?,?, ?,?,?)""",
            (user_id, reporter_name, address, description, photo,
             analysis["defect"], analysis["defect_name"], analysis["category"],
             analysis["confidence"], analysis["severity"], analysis["extent"],
             priority["priority_score"], priority["priority_band"], priority["explanation"],
             "Submitted", ts, ts),
        )
        cid = int(cur.lastrowid)
        conn.execute(
            "INSERT INTO status_history (complaint_id, status, changed_by, changed_at)"
            " VALUES (?,?,?,?)",
            (cid, "Submitted", reporter_name, ts),
        )
        return cid


def queue_for_category(category: str) -> list:
    """
    The live queue for one staff category.

    Ordered by priority_score descending — highest priority served first — with equal
    scores broken by submission time, oldest first. Resolved complaints are excluded, so
    they leave the live queue the moment staff mark them done.
    """
    placeholders = ",".join("?" for _ in LIVE_STATUSES)
    with connect() as conn:
        return conn.execute(
            f"""SELECT * FROM complaints
                WHERE category = ? AND status IN ({placeholders})
                ORDER BY priority_score DESC, created_at ASC""",
            (category, *LIVE_STATUSES),
        ).fetchall()


def resolved_for_category(category: str, limit: int = 50) -> list:
    with connect() as conn:
        return conn.execute(
            """SELECT * FROM complaints WHERE category = ? AND status = 'Resolved'
               ORDER BY updated_at DESC LIMIT ?""",
            (category, limit),
        ).fetchall()


def complaints_for_user(user_id: int) -> list:
    """Every complaint this user filed, resolved ones included — their full history."""
    with connect() as conn:
        return conn.execute(
            "SELECT * FROM complaints WHERE user_id = ? ORDER BY created_at DESC",
            (user_id,),
        ).fetchall()


def get_complaint(complaint_id: int):
    with connect() as conn:
        return conn.execute("SELECT * FROM complaints WHERE id = ?",
                            (complaint_id,)).fetchone()


def queue_position(complaint_id: int) -> int | None:
    """Where this complaint currently sits in its category's live queue (1 = next up)."""
    row = get_complaint(complaint_id)
    if row is None or row["status"] == "Resolved":
        return None
    for i, item in enumerate(queue_for_category(row["category"]), start=1):
        if item["id"] == complaint_id:
            return i
    return None


def update_status(complaint_id: int, new_status: str, changed_by: str) -> tuple[bool, str]:
    """Move a complaint forward through the pipeline. Forward-only, one step at a time."""
    if new_status not in STATUSES:
        return False, f"Unknown status '{new_status}'."

    row = get_complaint(complaint_id)
    if row is None:
        return False, "Complaint not found."

    current, target = STATUSES.index(row["status"]), STATUSES.index(new_status)
    if target <= current:
        return False, f"Cannot move from {row['status']} back to {new_status}."
    if target != current + 1:
        return False, (f"Status must advance one step at a time: "
                       f"{row['status']} -> {STATUSES[current + 1]}.")

    ts = now()
    with connect() as conn:
        conn.execute("UPDATE complaints SET status = ?, updated_at = ? WHERE id = ?",
                     (new_status, ts, complaint_id))
        conn.execute(
            "INSERT INTO status_history (complaint_id, status, changed_by, changed_at)"
            " VALUES (?,?,?,?)",
            (complaint_id, new_status, changed_by, ts),
        )
    return True, f"Complaint #{complaint_id} moved to {new_status}."


def history_for(complaint_id: int) -> list:
    with connect() as conn:
        return conn.execute(
            "SELECT * FROM status_history WHERE complaint_id = ? ORDER BY id ASC",
            (complaint_id,),
        ).fetchall()


def next_status(current: str) -> str | None:
    i = STATUSES.index(current)
    return STATUSES[i + 1] if i + 1 < len(STATUSES) else None


def stats() -> dict:
    with connect() as conn:
        total = conn.execute("SELECT COUNT(*) c FROM complaints").fetchone()["c"]
        resolved = conn.execute(
            "SELECT COUNT(*) c FROM complaints WHERE status='Resolved'").fetchone()["c"]
        by_cat = {
            r["category"]: r["c"]
            for r in conn.execute(
                "SELECT category, COUNT(*) c FROM complaints"
                " WHERE status != 'Resolved' GROUP BY category")
        }
    return {"total": total, "resolved": resolved, "live_by_category": by_cat}
