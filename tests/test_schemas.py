"""Unit tests for FreshLens shared Pydantic data models in schemas.py."""

import json
import math
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from schemas import (
    Claim,
    ClaimEvidence,
    ClaimStatus,
    ClaimVerification,
    FreshnessReport,
    FreshnessSummary,
)


# =====================================================================
# 1. Valid Claim creation
# =====================================================================

def test_valid_claim_creation():
    claim_ts = Claim(
        claim_id="claim-001",
        text="The current CEO of OpenAI is Sam Altman.",
        is_time_sensitive=True,
        category="leadership",
        time_sensitivity_reason="Corporate executive positions frequently change over time.",
    )
    assert claim_ts.claim_id == "claim-001"
    assert claim_ts.text == "The current CEO of OpenAI is Sam Altman."
    assert claim_ts.is_time_sensitive is True
    assert claim_ts.category == "leadership"
    assert claim_ts.time_sensitivity_reason == "Corporate executive positions frequently change over time."

    claim_static = Claim(
        claim_id="claim-002",
        text="Water is composed of two hydrogen atoms and one oxygen atom.",
        is_time_sensitive=False,
        category="science",
    )
    assert claim_static.claim_id == "claim-002"
    assert claim_static.is_time_sensitive is False
    assert claim_static.time_sensitivity_reason is None


# =====================================================================
# 2. Valid ClaimEvidence creation
# =====================================================================

def test_valid_claim_evidence_creation():
    evidence_full = ClaimEvidence(
        title="OpenAI Leadership Directory",
        url="https://openai.com/about",
        excerpt="Sam Altman serves as Chief Executive Officer of OpenAI.",
        published_date="2025-02-01",
        relevance_score=0.96,
    )
    assert evidence_full.title == "OpenAI Leadership Directory"
    assert evidence_full.url == "https://openai.com/about"
    assert evidence_full.excerpt == "Sam Altman serves as Chief Executive Officer of OpenAI."
    assert evidence_full.published_date == "2025-02-01"
    assert evidence_full.relevance_score == 0.96

    evidence_minimal = ClaimEvidence(
        title="OpenAI News",
        url="https://example.com/news",
        excerpt="News excerpt.",
    )
    assert evidence_minimal.published_date is None
    assert evidence_minimal.relevance_score is None


# =====================================================================
# 3. All four ClaimStatus values
# =====================================================================

def test_all_four_claim_status_values():
    assert ClaimStatus.SUPPORTED == "supported"
    assert ClaimStatus.POTENTIALLY_OUTDATED == "potentially_outdated"
    assert ClaimStatus.CONFLICTING == "conflicting"
    assert ClaimStatus.INSUFFICIENT_EVIDENCE == "insufficient_evidence"

    expected_statuses = {
        "supported",
        "potentially_outdated",
        "conflicting",
        "insufficient_evidence",
    }
    assert {s.value for s in ClaimStatus} == expected_statuses
    assert len(ClaimStatus) == 4

    # Instantiation from exact string value
    assert ClaimStatus("supported") is ClaimStatus.SUPPORTED
    assert ClaimStatus("potentially_outdated") is ClaimStatus.POTENTIALLY_OUTDATED
    assert ClaimStatus("conflicting") is ClaimStatus.CONFLICTING
    assert ClaimStatus("insufficient_evidence") is ClaimStatus.INSUFFICIENT_EVIDENCE

    # Rejection of invalid status strings
    with pytest.raises(ValueError):
        ClaimStatus("fresh")
    with pytest.raises(ValueError):
        ClaimStatus("stale")
    with pytest.raises(ValueError):
        ClaimStatus("unverified")


# =====================================================================
# 4. Valid ClaimVerification creation
# =====================================================================

def test_valid_claim_verification_creation():
    evidence = ClaimEvidence(
        title="Source",
        url="https://example.com",
        excerpt="Relevant quote.",
        relevance_score=0.9,
    )
    verification_full = ClaimVerification(
        claim_id="claim-001",
        status=ClaimStatus.SUPPORTED,
        rationale="Evidence directly corroborates the claim statement.",
        confidence=0.95,
        newer_information="Confirmed current as of 2025.",
        evidence=[evidence],
        freshness_score=95.0,
    )
    assert verification_full.claim_id == "claim-001"
    assert verification_full.status == ClaimStatus.SUPPORTED
    assert verification_full.confidence == 0.95
    assert verification_full.newer_information == "Confirmed current as of 2025."
    assert len(verification_full.evidence) == 1
    assert verification_full.freshness_score == 95.0

    verification_minimal = ClaimVerification(
        claim_id="claim-002",
        status=ClaimStatus.POTENTIALLY_OUTDATED,
        rationale="Newer reports indicate a leadership transition.",
        confidence=0.8,
    )
    assert verification_minimal.evidence == []
    assert verification_minimal.newer_information is None
    assert verification_minimal.freshness_score is None


# =====================================================================
# 5. Valid FreshnessSummary creation
# =====================================================================

def test_valid_freshness_summary_creation():
    summary_full = FreshnessSummary(
        total_claims=5,
        time_sensitive_claims=4,
        supported_claims=3,
        potentially_outdated_claims=1,
        conflicting_claims=0,
        insufficient_evidence_claims=0,
        overall_freshness_score=75.0,
        evidence_coverage=1.0,
    )
    assert summary_full.total_claims == 5
    assert summary_full.time_sensitive_claims == 4
    assert summary_full.supported_claims == 3
    assert summary_full.potentially_outdated_claims == 1
    assert summary_full.overall_freshness_score == 75.0
    assert summary_full.evidence_coverage == 1.0

    summary_no_score = FreshnessSummary(
        total_claims=0,
        time_sensitive_claims=0,
        supported_claims=0,
        potentially_outdated_claims=0,
        conflicting_claims=0,
        insufficient_evidence_claims=0,
        evidence_coverage=0.0,
    )
    assert summary_no_score.overall_freshness_score is None
    assert summary_no_score.evidence_coverage == 0.0


# =====================================================================
# 6. Valid FreshnessReport creation
# =====================================================================

def test_valid_freshness_report_creation():
    claim = Claim(
        claim_id="c-1",
        text="Python 3.14 is the current release.",
        is_time_sensitive=True,
        category="technology",
    )
    evidence = ClaimEvidence(
        title="Python Releases",
        url="https://python.org",
        excerpt="Python 3.13 is the current stable release.",
        relevance_score=0.99,
    )
    verification = ClaimVerification(
        claim_id="c-1",
        status=ClaimStatus.POTENTIALLY_OUTDATED,
        rationale="Official site reports 3.13 as current stable release.",
        confidence=0.98,
        newer_information="Python 3.13 is the active stable release.",
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

    report = FreshnessReport(
        original_answer="Python 3.14 is the current release.",
        claims=[claim],
        verifications=[verification],
        summary=summary,
    )

    assert report.original_answer == "Python 3.14 is the current release."
    assert len(report.claims) == 1
    assert len(report.verifications) == 1
    assert report.summary.overall_freshness_score == 0.0


# =====================================================================
# 7. Invalid empty required fields
# =====================================================================

@pytest.mark.parametrize(
    "empty_val",
    ["", "   ", "\t\n"],
)
def test_invalid_empty_claim_fields(empty_val):
    with pytest.raises(ValidationError):
        Claim(claim_id=empty_val, text="Valid", is_time_sensitive=True, category="cat")
    with pytest.raises(ValidationError):
        Claim(claim_id="c1", text=empty_val, is_time_sensitive=True, category="cat")
    with pytest.raises(ValidationError):
        Claim(claim_id="c1", text="Valid", is_time_sensitive=True, category=empty_val)


@pytest.mark.parametrize(
    "empty_val",
    ["", "   ", "\t\n"],
)
def test_invalid_empty_verification_fields(empty_val):
    with pytest.raises(ValidationError):
        ClaimVerification(
            claim_id=empty_val,
            status=ClaimStatus.SUPPORTED,
            rationale="Rationale",
            confidence=0.8,
        )
    with pytest.raises(ValidationError):
        ClaimVerification(
            claim_id="c1",
            status=ClaimStatus.SUPPORTED,
            rationale=empty_val,
            confidence=0.8,
        )


@pytest.mark.parametrize(
    "empty_val",
    ["", "   ", "\t\n"],
)
def test_invalid_empty_report_fields(empty_val):
    summary = FreshnessSummary(
        total_claims=0,
        time_sensitive_claims=0,
        supported_claims=0,
        potentially_outdated_claims=0,
        conflicting_claims=0,
        insufficient_evidence_claims=0,
        evidence_coverage=0.0,
    )
    with pytest.raises(ValidationError):
        FreshnessReport(original_answer=empty_val, summary=summary)


# =====================================================================
# 8. Invalid score and confidence ranges
# =====================================================================

@pytest.mark.parametrize("invalid_conf", [-0.01, -1.0, 1.01, 2.0, 100.0])
def test_invalid_confidence_range(invalid_conf):
    with pytest.raises(ValidationError):
        ClaimVerification(
            claim_id="c1",
            status=ClaimStatus.SUPPORTED,
            rationale="Valid rationale",
            confidence=invalid_conf,
        )


@pytest.mark.parametrize("invalid_freshness", [-0.1, -10.0, 100.1, 150.0])
def test_invalid_verification_freshness_score_range(invalid_freshness):
    with pytest.raises(ValidationError):
        ClaimVerification(
            claim_id="c1",
            status=ClaimStatus.SUPPORTED,
            rationale="Valid rationale",
            confidence=0.5,
            freshness_score=invalid_freshness,
        )


@pytest.mark.parametrize("invalid_rel", [-0.01, -0.5, 1.01, 10.0])
def test_invalid_relevance_score_range(invalid_rel):
    with pytest.raises(ValidationError):
        ClaimEvidence(
            title="Title",
            url="https://example.com",
            excerpt="Excerpt",
            relevance_score=invalid_rel,
        )


@pytest.mark.parametrize("invalid_overall", [-0.1, -5.0, 100.1, 200.0])
def test_invalid_summary_overall_freshness_range(invalid_overall):
    with pytest.raises(ValidationError):
        FreshnessSummary(
            total_claims=1,
            time_sensitive_claims=1,
            supported_claims=1,
            potentially_outdated_claims=0,
            conflicting_claims=0,
            insufficient_evidence_claims=0,
            overall_freshness_score=invalid_overall,
            evidence_coverage=1.0,
        )


@pytest.mark.parametrize("invalid_cov", [-0.01, -1.0, 1.01, 2.0])
def test_invalid_evidence_coverage_range(invalid_cov):
    with pytest.raises(ValidationError):
        FreshnessSummary(
            total_claims=1,
            time_sensitive_claims=1,
            supported_claims=1,
            potentially_outdated_claims=0,
            conflicting_claims=0,
            insufficient_evidence_claims=0,
            overall_freshness_score=80.0,
            evidence_coverage=invalid_cov,
        )


@pytest.mark.parametrize("bool_val", [True, False])
def test_reject_boolean_for_numeric_fields(bool_val):
    # Confidence in ClaimVerification
    with pytest.raises(ValidationError):
        ClaimVerification(
            claim_id="c1",
            status=ClaimStatus.SUPPORTED,
            rationale="Valid",
            confidence=bool_val,
        )

    # Freshness score in ClaimVerification
    with pytest.raises(ValidationError):
        ClaimVerification(
            claim_id="c1",
            status=ClaimStatus.SUPPORTED,
            rationale="Valid",
            confidence=0.5,
            freshness_score=bool_val,
        )

    # Relevance score in ClaimEvidence
    with pytest.raises(ValidationError):
        ClaimEvidence(
            title="T",
            url="U",
            excerpt="E",
            relevance_score=bool_val,
        )

    # Summary overall score and coverage
    with pytest.raises(ValidationError):
        FreshnessSummary(
            total_claims=1,
            time_sensitive_claims=1,
            supported_claims=1,
            potentially_outdated_claims=0,
            conflicting_claims=0,
            insufficient_evidence_claims=0,
            overall_freshness_score=bool_val,
            evidence_coverage=0.5,
        )
    with pytest.raises(ValidationError):
        FreshnessSummary(
            total_claims=1,
            time_sensitive_claims=1,
            supported_claims=1,
            potentially_outdated_claims=0,
            conflicting_claims=0,
            insufficient_evidence_claims=0,
            evidence_coverage=bool_val,
        )

    # Summary counts
    with pytest.raises(ValidationError):
        FreshnessSummary(
            total_claims=bool_val,
            time_sensitive_claims=0,
            supported_claims=0,
            potentially_outdated_claims=0,
            conflicting_claims=0,
            insufficient_evidence_claims=0,
            evidence_coverage=0.5,
        )


# =====================================================================
# 9. NaN and infinity rejection
# =====================================================================

@pytest.mark.parametrize("non_finite", [math.nan, math.inf, -math.inf])
def test_reject_nan_and_infinity(non_finite):
    # Confidence
    with pytest.raises(ValidationError):
        ClaimVerification(
            claim_id="c1",
            status=ClaimStatus.SUPPORTED,
            rationale="Valid",
            confidence=non_finite,
        )

    # Verification freshness score
    with pytest.raises(ValidationError):
        ClaimVerification(
            claim_id="c1",
            status=ClaimStatus.SUPPORTED,
            rationale="Valid",
            confidence=0.5,
            freshness_score=non_finite,
        )

    # Relevance score
    with pytest.raises(ValidationError):
        ClaimEvidence(
            title="T",
            url="U",
            excerpt="E",
            relevance_score=non_finite,
        )

    # Summary overall score and coverage
    with pytest.raises(ValidationError):
        FreshnessSummary(
            total_claims=1,
            time_sensitive_claims=1,
            supported_claims=1,
            potentially_outdated_claims=0,
            conflicting_claims=0,
            insufficient_evidence_claims=0,
            overall_freshness_score=non_finite,
            evidence_coverage=0.5,
        )
    with pytest.raises(ValidationError):
        FreshnessSummary(
            total_claims=1,
            time_sensitive_claims=1,
            supported_claims=1,
            potentially_outdated_claims=0,
            conflicting_claims=0,
            insufficient_evidence_claims=0,
            overall_freshness_score=50.0,
            evidence_coverage=non_finite,
        )


# =====================================================================
# 10. Invalid negative counts
# =====================================================================

@pytest.mark.parametrize(
    "count_kwargs",
    [
        {"total_claims": -1},
        {"time_sensitive_claims": -1},
        {"supported_claims": -1},
        {"potentially_outdated_claims": -1},
        {"conflicting_claims": -1},
        {"insufficient_evidence_claims": -1},
    ],
)
def test_invalid_negative_counts(count_kwargs):
    valid_args = {
        "total_claims": 0,
        "time_sensitive_claims": 0,
        "supported_claims": 0,
        "potentially_outdated_claims": 0,
        "conflicting_claims": 0,
        "insufficient_evidence_claims": 0,
        "evidence_coverage": 0.0,
    }
    valid_args.update(count_kwargs)
    with pytest.raises(ValidationError):
        FreshnessSummary(**valid_args)


# =====================================================================
# 11. JSON serialization
# =====================================================================

def test_json_serialization_and_roundtrip():
    claim = Claim(
        claim_id="claim-101",
        text="The current prime minister of the UK is Keir Starmer.",
        is_time_sensitive=True,
        category="politics",
        time_sensitivity_reason="Political leadership positions change.",
    )
    evidence = ClaimEvidence(
        title="UK Government",
        url="https://gov.uk",
        excerpt="Keir Starmer is the Prime Minister of the United Kingdom.",
        published_date="2025-01-01",
        relevance_score=0.98,
    )
    verification = ClaimVerification(
        claim_id="claim-101",
        status=ClaimStatus.SUPPORTED,
        rationale="Official UK government portal confirms Starmer as PM.",
        confidence=0.99,
        newer_information=None,
        evidence=[evidence],
        freshness_score=100.0,
    )
    summary = FreshnessSummary(
        total_claims=1,
        time_sensitive_claims=1,
        supported_claims=1,
        potentially_outdated_claims=0,
        conflicting_claims=0,
        insufficient_evidence_claims=0,
        overall_freshness_score=100.0,
        evidence_coverage=1.0,
    )
    report = FreshnessReport(
        original_answer="Keir Starmer is the Prime Minister of the UK.",
        claims=[claim],
        verifications=[verification],
        summary=summary,
    )

    # Verify JSON string generation
    json_str = report.model_dump_json()
    assert isinstance(json_str, str)

    parsed_dict = json.loads(json_str)
    assert parsed_dict["original_answer"] == "Keir Starmer is the Prime Minister of the UK."
    assert parsed_dict["verifications"][0]["status"] == "supported"
    assert parsed_dict["summary"]["overall_freshness_score"] == 100.0

    # Verify roundtrip deserialization
    rehydrated = FreshnessReport.model_validate_json(json_str)
    assert rehydrated == report
    assert rehydrated.verifications[0].status is ClaimStatus.SUPPORTED

