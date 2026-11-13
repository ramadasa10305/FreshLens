"""Tests for local persistent FreshLens verification history."""

from datetime import datetime, timezone
import sqlite3

from schemas import (
    Claim,
    ClaimEvidence,
    ClaimStatus,
    ClaimVerification,
    FreshnessReport,
    FreshnessSummary,
)
from history_store import HistoryStore


def _report(answer: str = "  A submitted answer, exactly as entered.  ") -> FreshnessReport:
    claim = Claim(
        claim_id="claim-1",
        text="The answer contains a current factual claim.",
        is_time_sensitive=True,
        category="general",
        time_sensitivity_reason="This fact can change.",
    )
    evidence = ClaimEvidence(
        title="Verification source",
        url="https://example.test/source",
        excerpt="Evidence excerpt for the claim.",
        published_date="2026-11-13",
        relevance_score=0.9,
    )
    verification = ClaimVerification(
        claim_id=claim.claim_id,
        status=ClaimStatus.POTENTIALLY_OUTDATED,
        rationale="The source suggests the claim may have changed.",
        confidence=0.8,
        newer_information="Newer information is available.",
        evidence=[evidence],
        freshness_score=0.0,
    )
    summary = FreshnessSummary(
        total_claims=1,
        time_sensitive_claims=1,
        supported_claims=0,
        potentially_outdated_claims=1,
        conflicting_claims=0,
        insufficient_evidence_claims=0,
        overall_freshness_score=0.0,
        evidence_coverage=1.0,
    )
    return FreshnessReport.model_construct(
        original_answer=answer,
        claims=[claim],
        verifications=[verification],
        summary=summary,
    )


def test_initializes_database_and_history_schema(tmp_path):
    database_path = tmp_path / "freshlens-history.sqlite3"

    store = HistoryStore(database_path)

    assert database_path.is_file()
    with sqlite3.connect(database_path) as connection:
        tables = connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
        columns = connection.execute(
            "PRAGMA table_info(verification_history)"
        ).fetchall()
    assert ("verification_history",) in tables
    assert [column[1] for column in columns] == [
        "record_id",
        "created_at_utc",
        "original_answer",
        "report_json",
    ]
    assert store.list_recent() == []


def test_save_creates_unique_records_with_utc_timestamps(tmp_path):
    store = HistoryStore(tmp_path / "history.sqlite3")
    submitted_at = datetime(2026, 11, 13, 12, 30, tzinfo=timezone.utc)

    first = store.save(_report(), created_at_utc=submitted_at)
    second = store.save(_report("A second answer."), created_at_utc=submitted_at)

    assert first.record_id != second.record_id
    assert first.created_at_utc == submitted_at
    assert first.created_at_utc.utcoffset().total_seconds() == 0
    assert first.original_answer == "  A submitted answer, exactly as entered.  "


def test_retrieval_is_newest_first_and_honors_limit(tmp_path):
    store = HistoryStore(tmp_path / "history.sqlite3")
    older_at = datetime(2026, 11, 12, tzinfo=timezone.utc)
    newer_at = datetime(2026, 11, 13, tzinfo=timezone.utc)
    older = store.save(_report("Older answer."), created_at_utc=older_at)
    newer = store.save(_report("Newer answer."), created_at_utc=newer_at)

    recent = store.list_recent(limit=1)

    assert recent == [newer]
    assert store.list_recent(limit=2) == [newer, older]


def test_report_serialization_and_reopening_a_previous_record(tmp_path):
    database_path = tmp_path / "history.sqlite3"
    expected_report = _report()
    first_store = HistoryStore(database_path)
    saved = first_store.save(expected_report)

    reopened_store = HistoryStore(database_path)
    reopened = reopened_store.get(saved.record_id)

    assert reopened is not None
    assert reopened.original_answer == expected_report.original_answer
    assert reopened.report == expected_report
    assert reopened.report.verifications[0].status is ClaimStatus.POTENTIALLY_OUTDATED
    assert reopened.report.verifications[0].evidence[0].title == "Verification source"
    assert reopened.report.summary.evidence_coverage == 1.0


def test_delete_removes_only_the_requested_record(tmp_path):
    store = HistoryStore(tmp_path / "history.sqlite3")
    first = store.save(_report("First answer."))
    second = store.save(_report("Second answer."))

    assert store.delete(first.record_id) is True
    assert store.delete(first.record_id) is False
    assert store.get(first.record_id) is None
    assert store.get(second.record_id) is not None


def test_clear_all_deletes_records_and_is_repeatable(tmp_path):
    store = HistoryStore(tmp_path / "history.sqlite3")
    store.save(_report("First answer."))
    store.save(_report("Second answer."))

    assert store.clear_all() == 2
    assert store.list_recent() == []
    assert store.clear_all() == 0


def test_history_survives_store_reinitialization(tmp_path):
    database_path = tmp_path / "history.sqlite3"
    saved = HistoryStore(database_path).save(_report("Persisted answer."))

    restarted_store = HistoryStore(database_path)

    assert restarted_store.list_recent()[0].record_id == saved.record_id
    assert restarted_store.get(saved.record_id).original_answer == "Persisted answer."
