"""
lifecycle.py -- Standalone knowledge-record lifecycle state machine.

Ported from Hermes curator's apply_automatic_transitions logic.
Three states: active -> stale -> archived, with reactivation on reuse.

State transitions (per tick):
    active   + older than archive_cutoff  -> archived
    active   + older than stale_cutoff    -> stale
    stale    + newer than stale_cutoff    -> active   (reactivation)
    archived                               -> stays archived
    pinned records are never transitioned.
"""

from __future__ import annotations

import random
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Optional

# -- State constants --------------------------------------------------------
STATE_ACTIVE = "active"
STATE_STALE = "stale"
STATE_ARCHIVED = "archived"

# -- LifecycleManager ----------------------------------------------------
class LifecycleManager:
    """Manage knowledge-record lifecycle with SQLite persistence."""

    def __init__(self, db_path: str, stale_days: int = 30, archive_days: int = 90):
        self.db_path = db_path
        self.stale_days = stale_days
        self.archive_days = archive_days
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    # -- schema --------------------------------------------------------------
    def _init_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS records (
                record_id        TEXT PRIMARY KEY,
                content          TEXT NOT NULL,
                state            TEXT NOT NULL DEFAULT 'active',
                pinned           INTEGER NOT NULL DEFAULT 0,
                created_at       TEXT NOT NULL,
                last_referenced_at TEXT
            );
            """
        )
        self._conn.commit()

    # -- public API ----------------------------------------------------------
    def add_record(
        self,
        record_id: str,
        content: str,
        created_at: str,
        pinned: bool = False,
    ) -> None:
        """Insert a new knowledge record (defaults to active state)."""
        self._conn.execute(
            "INSERT OR IGNORE INTO records (record_id, content, created_at, pinned) "
            "VALUES (?, ?, ?, ?)",
            (record_id, content, created_at, int(pinned)),
        )
        self._conn.commit()

    def touch_record(self, record_id: str, referenced_at: Optional[str] = None) -> None:
        """Mark a record as referenced -- updates last_referenced_at."""
        ts = referenced_at or datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            "UPDATE records SET last_referenced_at = ? WHERE record_id = ?",
            (ts, record_id),
        )
        self._conn.commit()

    def tick(self, now: Optional[str] = None) -> dict:
        """Run state transitions across all records.

        Returns counts: {checked, marked_stale, archived, reactivated}
        """
        now_dt = datetime.fromisoformat(now) if now else datetime.now(timezone.utc)
        stale_cutoff = now_dt - timedelta(days=self.stale_days)
        archive_cutoff = now_dt - timedelta(days=self.archive_days)

        counts = {"checked": 0, "marked_stale": 0, "archived": 0, "reactivated": 0}

        rows = self._conn.execute(
            "SELECT record_id, state, pinned, created_at, last_referenced_at "
            "FROM records"
        ).fetchall()

        for row in rows:
            counts["checked"] += 1
            # Pinned records are never transitioned.
            if row["pinned"]:
                continue
            # Anchor: last_referenced_at if present, else created_at.
            anchor_str = row["last_referenced_at"] or row["created_at"]
            anchor = datetime.fromisoformat(anchor_str)
            if anchor.tzinfo is None:
                anchor = anchor.replace(tzinfo=timezone.utc)

            current_state = row["state"]
            if anchor <= archive_cutoff and current_state != STATE_ARCHIVED:
                self._set_state(row["record_id"], STATE_ARCHIVED)
                counts["archived"] += 1
            elif anchor <= stale_cutoff and current_state == STATE_ACTIVE:
                self._set_state(row["record_id"], STATE_STALE)
                counts["marked_stale"] += 1
            elif anchor > stale_cutoff and current_state == STATE_STALE:
                self._set_state(row["record_id"], STATE_ACTIVE)
                counts["reactivated"] += 1

        self._conn.commit()
        return counts

    def get_state(self, record_id: str) -> Optional[str]:
        """Return current state of a record, or None if not found."""
        row = self._conn.execute(
            "SELECT state FROM records WHERE record_id = ?", (record_id,)
        ).fetchone()
        return row["state"] if row else None

    def get_all_states(self) -> list[dict]:
        """Return all records with their current state."""
        rows = self._conn.execute(
            "SELECT record_id, state, pinned, created_at, last_referenced_at "
            "FROM records"
        ).fetchall()
        return [dict(r) for r in rows]

    def get_stats(self) -> dict:
        """Return counts by state and active_ratio (active / total)."""
        rows = self._conn.execute(
            "SELECT state, COUNT(*) as cnt FROM records GROUP BY state"
        ).fetchall()
        counts = {r["state"]: r["cnt"] for r in rows}
        total = sum(counts.values())
        active = counts.get(STATE_ACTIVE, 0)
        return {
            "total": total,
            "active": active,
            "stale": counts.get(STATE_STALE, 0),
            "archived": counts.get(STATE_ARCHIVED, 0),
            "active_ratio": round(active / total, 3) if total else 0.0,
        }

    # -- internal ------------------------------------------------------------
    def _set_state(self, record_id: str, state: str) -> None:
        self._conn.execute(
            "UPDATE records SET state = ? WHERE record_id = ?", (state, record_id)
        )

    def close(self) -> None:
        self._conn.close()

# -- Test-data generator --------------------------------------------------

def generate_test_data(n: int = 1000, days_span: int = 180) -> list[dict]:
    """Generate n test records with power-law distributed reference frequencies.

    10% of records receive ~80% of references (high-frequency group).
    High-freq: referenced every 1-5 days. Low-freq: every 30-90 days.
    Each dict: record_id, content, created_at, pinned, references (list of ISO timestamps).
    """
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    high_freq_count = max(1, n // 10)  # 10% are high-frequency
    records: list[dict] = []

    for i in range(n):
        created = base + timedelta(days=random.uniform(0, days_span * 0.3))

        is_high_freq = i < high_freq_count
        pinned = random.random() < 0.02  # ~2% pinned
        references: list[str] = []
        t = created
        end = base + timedelta(days=days_span)
        while t < end:
            if is_high_freq:
                gap = random.randint(1, 5)
            else:
                gap = random.randint(30, 90)
            t = t + timedelta(days=gap)
            if t < end:
                references.append(t.isoformat())

        records.append(
            {
                "record_id": f"rec-{i:04d}",
                "content": f"Test record {i} ({'high' if is_high_freq else 'low'} freq)",
                "created_at": created.isoformat(),
                "pinned": pinned,
                "references": references,
            }
        )
    return records


# -- Control groups -------------------------------------------------------

class SimpleExpiryManager:
    """Control A: simple time-based expiry. No reactivation, no stale state."""

    def __init__(self, db_path: str, expire_days: int = 30):
        self.db_path = db_path
        self.expire_days = expire_days
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS records (
                record_id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                state TEXT NOT NULL DEFAULT 'active',
                created_at TEXT NOT NULL,
                last_referenced_at TEXT
            );
            """
        )
        self._conn.commit()

    def add_record(self, record_id: str, content: str, created_at: str) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO records (record_id, content, state, created_at) VALUES (?,?,?,?)",
            (record_id, content, "active", created_at),
        )
        self._conn.commit()

    def touch_record(self, record_id: str, referenced_at: str | None = None) -> None:
        ref = referenced_at or datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            "UPDATE records SET last_referenced_at=? WHERE record_id=?",
            (ref, record_id),
        )
        self._conn.commit()

    def tick(self, now: str | None = None) -> dict:
        now_dt = datetime.fromisoformat(now) if now else datetime.now(timezone.utc)
        cutoff = now_dt - timedelta(days=self.expire_days)
        counts = {"checked": 0, "expired": 0}

        rows = self._conn.execute("SELECT record_id, state, created_at, last_referenced_at FROM records WHERE state='active'").fetchall()
        for row in rows:
            counts["checked"] += 1
            anchor_str = row["last_referenced_at"] or row["created_at"]
            anchor = datetime.fromisoformat(anchor_str)
            if anchor.tzinfo is None:
                anchor = anchor.replace(tzinfo=timezone.utc)
            if anchor < cutoff:
                self._conn.execute("UPDATE records SET state='expired' WHERE record_id=?", (row["record_id"],))
                counts["expired"] += 1
        self._conn.commit()
        return counts

    def get_stats(self) -> dict:
        rows = self._conn.execute("SELECT state, COUNT(*) as cnt FROM records GROUP BY state").fetchall()
        counts = {r["state"]: r["cnt"] for r in rows}
        total = sum(counts.values())
        return {**counts, "total": total, "active_ratio": counts.get("active", 0) / max(total, 1)}


class NoOpManager:
    """Control B: no lifecycle management at all."""

    def __init__(self, db_path: str):
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS records (
                record_id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                state TEXT NOT NULL DEFAULT 'active',
                created_at TEXT NOT NULL
            );
            """
        )
        self._conn.commit()

    def add_record(self, record_id: str, content: str, created_at: str) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO records (record_id, content, state, created_at) VALUES (?,?,?,?)",
            (record_id, content, "active", created_at),
        )
        self._conn.commit()

    def touch_record(self, record_id: str, referenced_at: str | None = None) -> None:
        pass  # No-op

    def tick(self, now: str | None = None) -> dict:
        return {"checked": 0}  # No-op

    def get_stats(self) -> dict:
        total = self._conn.execute("SELECT COUNT(*) FROM records").fetchone()[0]
        return {"active": total, "total": total, "active_ratio": 1.0}


# -- Simulation helper ----------------------------------------------------

def simulate_timeline(manager, days: int = 180, reference_schedule: dict | None = None) -> list[dict]:
    """Simulate daily ticks over `days` period. Returns list of daily snapshots.

    reference_schedule: dict mapping date_str ("2026-01-15") -> list of record_ids
    to simulate real-time references during the simulation.
    """
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    timeseries = []

    for day in range(days):
        now_dt = base + timedelta(days=day)
        now = now_dt.isoformat()
        date_str = now[:10]

        # Apply scheduled references before tick
        if reference_schedule and date_str in reference_schedule:
            for rid in reference_schedule[date_str]:
                manager.touch_record(rid, now)

        result = manager.tick(now)
        stats = manager.get_stats()
        timeseries.append({
            "day": day,
            "date": date_str,
            **stats,
            "total_transitions": result.get("marked_stale", 0) + result.get("archived", 0) + result.get("expired", 0) + result.get("reactivated", 0),
            "reactivations": result.get("reactivated", 0),
        })

    return timeseries


def build_reference_schedule(records: list[dict]) -> dict:
    """Convert records' reference lists into a date -> [record_ids] schedule."""
    schedule: dict[str, list[str]] = {}
    for r in records:
        for ref_ts in r.get("references", []):
            date_str = ref_ts[:10]
            schedule.setdefault(date_str, []).append(r["record_id"])
    return schedule
