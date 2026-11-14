"""Streamlit submit-handler tests for the shared answer-size limit."""

from __future__ import annotations

import sys
import types
from datetime import datetime, timezone
from pathlib import Path

import history_store
import pytest
from history_store import HistoryRecord
from schemas import FreshnessReport, FreshnessSummary
from streamlit.testing.v1 import AppTest


class _FakeHistoryStore:
    def __init__(self, records: list[HistoryRecord] | None = None) -> None:
        self.records = list(records or [])
        self.deleted_ids: list[str] = []
        self.clear_calls = 0
        self.saved_reports: list[FreshnessReport] = []

    def list_recent(self, limit: int = 20) -> list[object]:
        return self.records[:limit]

    def save(self, report: FreshnessReport) -> None:
        self.saved_reports.append(report)

    def get(self, record_id: str) -> HistoryRecord | None:
        return next((item for item in self.records if item.record_id == record_id), None)

    def delete(self, record_id: str) -> bool:
        self.deleted_ids.append(record_id)
        before = len(self.records)
        self.records = [item for item in self.records if item.record_id != record_id]
        return len(self.records) < before

    def clear_all(self) -> int:
        self.clear_calls += 1
        count = len(self.records)
        self.records.clear()
        return count


def _report(answer: str) -> FreshnessReport:
    return FreshnessReport(
        original_answer=answer,
        summary=FreshnessSummary(
            total_claims=0,
            time_sensitive_claims=0,
            supported_claims=0,
            potentially_outdated_claims=0,
            conflicting_claims=0,
            insufficient_evidence_claims=0,
            overall_freshness_score=None,
            evidence_coverage=0.0,
        ),
    )


def _history_record(record_id: str, answer: str) -> HistoryRecord:
    return HistoryRecord(
        record_id=record_id,
        created_at_utc=datetime(2026, 11, 14, tzinfo=timezone.utc),
        original_answer=answer,
        report=_report(answer),
    )


def _app_path() -> str:
    return str(Path(__file__).resolve().parents[1] / "app.py")


def _install_fake_pipeline(monkeypatch: pytest.MonkeyPatch, pipeline_type: type) -> None:
    module = types.ModuleType("pipeline")
    module.FreshnessPipeline = pipeline_type
    monkeypatch.setitem(sys.modules, "pipeline", module)


def _click_verify_freshness(app: AppTest) -> None:
    button = next(
        (item for item in app.button if item.label == "Verify freshness"),
        None,
    )
    assert button is not None
    button.click().run()


@pytest.mark.parametrize(
    ("answer", "expected_error"),
    [
        (" \t\n", "Paste an answer before starting a check."),
        ("x" * 12_001, "Answer must be 12,000 characters or fewer."),
    ],
)
def test_submit_rejects_empty_or_oversized_answer_without_calling_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    answer: str,
    expected_error: str,
) -> None:
    pipeline_constructions: list[object] = []

    class ForbiddenPipeline:
        def __init__(self) -> None:
            pipeline_constructions.append(self)

    _install_fake_pipeline(monkeypatch, ForbiddenPipeline)
    monkeypatch.setattr(history_store, "HistoryStore", _FakeHistoryStore)

    app = AppTest.from_file(_app_path()).run()
    assert not app.exception

    answer_widget = app.text_area(key="answer_input")
    answer_widget.set_value(answer)
    _click_verify_freshness(app)

    assert not app.exception
    assert app.text_area(key="answer_input").value == answer
    assert app.session_state["freshness_error"] == expected_error
    assert [item.value for item in app.error] == [expected_error]
    assert pipeline_constructions == []


def test_manual_submit_calls_pipeline_and_saves_successful_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _FakeHistoryStore()
    monkeypatch.setattr(history_store, "HistoryStore", lambda: store)
    calls: list[str] = []

    class FakePipeline:
        def analyze(self, answer: str) -> FreshnessReport:
            calls.append(answer)
            return _report(answer)

    _install_fake_pipeline(monkeypatch, FakePipeline)
    app = AppTest.from_file(_app_path()).run()
    answer = "A normal answer with one factual claim."
    app.text_area(key="answer_input").set_value(answer)
    _click_verify_freshness(app)

    assert not app.exception
    assert calls == [answer]
    assert len(store.saved_reports) == 1
    assert store.saved_reports[0].original_answer == answer
    assert app.session_state["freshness_report"].original_answer == answer


@pytest.mark.parametrize(
    ("button_key", "expected_answer"),
    [
        ("example_microsoft_leadership", "Microsoft's CEO is Satya Nadella."),
        (
            "example_python_version",
            "Python 3.12 is the latest stable feature release series.",
        ),
    ],
)
def test_example_buttons_only_populate_answer_and_clear_error(
    monkeypatch: pytest.MonkeyPatch,
    button_key: str,
    expected_answer: str,
) -> None:
    pipeline_constructions: list[object] = []

    class ForbiddenPipeline:
        def __init__(self) -> None:
            pipeline_constructions.append(self)

    store = _FakeHistoryStore()
    monkeypatch.setattr(history_store, "HistoryStore", lambda: store)
    _install_fake_pipeline(monkeypatch, ForbiddenPipeline)
    app = AppTest.from_file(_app_path()).run()
    app.session_state["freshness_error"] = "Previous validation error"

    app.button(key=button_key).click().run()

    assert not app.exception
    assert app.session_state["answer_input"] == expected_answer
    assert app.text_area(key="answer_input").value == expected_answer
    assert app.session_state["freshness_error"] is None
    assert app.session_state["freshness_report"] is None
    assert pipeline_constructions == []
    assert store.saved_reports == []


def test_example_answer_remains_editable_and_requires_manual_submission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _FakeHistoryStore()
    calls: list[str] = []
    monkeypatch.setattr(history_store, "HistoryStore", lambda: store)

    class FakePipeline:
        def analyze(self, answer: str) -> FreshnessReport:
            calls.append(answer)
            return _report(answer)

    _install_fake_pipeline(monkeypatch, FakePipeline)
    app = AppTest.from_file(_app_path()).run()
    app.button(key="example_microsoft_leadership").click().run()

    assert calls == []
    assert app.session_state["answer_input"] == "Microsoft's CEO is Satya Nadella."

    edited_answer = "Microsoft's CEO is Satya Nadella. Edited by the user."
    app.text_area(key="answer_input").set_value(edited_answer)
    _click_verify_freshness(app)

    assert not app.exception
    assert calls == [edited_answer]
    assert len(store.saved_reports) == 1


def test_history_reopening_and_individual_deletion_use_saved_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = _history_record("record-1", "Previously verified answer.")
    store = _FakeHistoryStore([record])

    class ForbiddenPipeline:
        def analyze(self, answer: str) -> FreshnessReport:
            raise AssertionError("Opening history must not rerun verification")

    monkeypatch.setattr(history_store, "HistoryStore", lambda: store)
    _install_fake_pipeline(monkeypatch, ForbiddenPipeline)
    app = AppTest.from_file(_app_path()).run()

    assert any("Previously verified answer." in str(item.value) for item in app.get("text"))
    app.button(key="open_history_record-1").click().run()
    assert app.session_state["freshness_report"].original_answer == record.original_answer

    app.button(key="delete_history_record-1").click().run()
    assert store.deleted_ids == [record.record_id]
    assert store.records == []


def test_clear_history_requires_confirmation_and_clears_saved_records(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _FakeHistoryStore([_history_record("record-2", "Saved answer.")])

    class ForbiddenPipeline:
        def analyze(self, answer: str) -> FreshnessReport:
            raise AssertionError("History controls must not run verification")

    monkeypatch.setattr(history_store, "HistoryStore", lambda: store)
    _install_fake_pipeline(monkeypatch, ForbiddenPipeline)
    app = AppTest.from_file(_app_path()).run()
    app.button(key="request_clear_all_history").click().run()

    assert app.session_state["confirm_clear_history"] is True
    app.button(key="confirm_delete_all_history").click().run()

    assert store.clear_calls == 1
    assert store.records == []
    assert app.session_state["confirm_clear_history"] is False
