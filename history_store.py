"""Local, single-user persistence for completed FreshLens verifications."""

from __future__ import annotations

import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from schemas import FreshnessReport


DEFAULT_HISTORY_DB_PATH = Path(__file__).resolve().parent / ".freshlens_history.sqlite3"
_SQLITE_TIMEOUT_SECONDS = 10.0


@dataclass(frozen=True)
class HistoryRecord:
    """One persisted report and the submitted answer that produced it."""

    record_id: str
    created_at_utc: datetime
    original_answer: str
    report: FreshnessReport


class HistoryStore:
    """Persist successful reports in a local SQLite database for one user."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = Path(db_path) if db_path is not None else DEFAULT_HISTORY_DB_PATH
        self.db_path = self.db_path.expanduser().resolve()
        self.initialize()

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(
            self.db_path,
            timeout=_SQLITE_TIMEOUT_SECONDS,
        )
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialize(self) -> None:
        """Create the local database and schema if they do not exist yet."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS verification_history (
                    record_id TEXT PRIMARY KEY NOT NULL,
                    created_at_utc TEXT NOT NULL,
                    original_answer TEXT NOT NULL,
                    report_json TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_verification_history_created_at
                ON verification_history (created_at_utc DESC, record_id DESC)
                """
            )

    @staticmethod
    def _utc_timestamp(value: datetime | None) -> datetime:
        timestamp = value if value is not None else datetime.now(timezone.utc)
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise ValueError("created_at_utc must include a timezone")
        return timestamp.astimezone(timezone.utc)

    @staticmethod
    def _from_row(row: sqlite3.Row) -> HistoryRecord:
        report = FreshnessReport.model_validate_json(row["report_json"])
        # FreshnessReport's shared config strips string whitespace on normal
        # validation, while the pipeline preserves the submitted answer exactly.
        report = report.model_copy(update={"original_answer": row["original_answer"]})
        return HistoryRecord(
            record_id=row["record_id"],
            created_at_utc=datetime.fromisoformat(row["created_at_utc"]).astimezone(
                timezone.utc
            ),
            original_answer=row["original_answer"],
            report=report,
        )

    def save(
        self,
        report: FreshnessReport,
        *,
        created_at_utc: datetime | None = None,
    ) -> HistoryRecord:
        """Save a successful report and return its persisted history record."""
        timestamp = self._utc_timestamp(created_at_utc)
        record_id = str(uuid.uuid4())
        original_answer = report.original_answer
        report_json = report.model_dump_json()
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO verification_history (
                    record_id, created_at_utc, original_answer, report_json
                ) VALUES (?, ?, ?, ?)
                """,
                (
                    record_id,
                    timestamp.isoformat(),
                    original_answer,
                    report_json,
                ),
            )
        return HistoryRecord(
            record_id=record_id,
            created_at_utc=timestamp,
            original_answer=original_answer,
            report=report,
        )

    def list_recent(self, limit: int = 20) -> list[HistoryRecord]:
        """Return newest records first, capped at the requested positive limit."""
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("limit must be a positive integer")
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT record_id, created_at_utc, original_answer, report_json
                FROM verification_history
                ORDER BY created_at_utc DESC, record_id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [self._from_row(row) for row in rows]

    def get(self, record_id: str) -> HistoryRecord | None:
        """Load one previous report without running claim verification again."""
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT record_id, created_at_utc, original_answer, report_json
                FROM verification_history
                WHERE record_id = ?
                """,
                (record_id,),
            ).fetchone()
        return self._from_row(row) if row is not None else None

    def delete(self, record_id: str) -> bool:
        """Delete a single record and report whether it existed."""
        with self._connection() as connection:
            cursor = connection.execute(
                "DELETE FROM verification_history WHERE record_id = ?",
                (record_id,),
            )
        return cursor.rowcount > 0

    def clear_all(self) -> int:
        """Delete all stored history records and return the number removed."""
        with self._connection() as connection:
            cursor = connection.execute("DELETE FROM verification_history")
        return cursor.rowcount


__all__ = ["HistoryRecord", "HistoryStore"]
