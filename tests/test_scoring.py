"""Unit tests for deterministic freshness scoring in scoring.py."""

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from schemas import (
    Claim,
    ClaimEvidence,
    ClaimStatus,
    ClaimVerification,
    FreshnessSummary,
)
from scoring import (
    calculate_claim_freshness_score,
    score_claims,
    score_freshness,
)


def _make_claim(claim_id: str, *, is_time_sensitive: bool = True) -> Claim:
    """Helper to create a valid Claim instance."""
    return Claim(
        claim_id=claim_id,
        text=f"Factual statement for {claim_id}",
        is_time_sensitive=is_time_sensitive,
        category="technology" if is_time_sensitive else "science",
        time_sensitivity_reason="Changes over time" if is_time_sensitive else None,
    )


def _make_evidence(url: str = "https://example.com/test") -> ClaimEvidence:
    """Helper to create a valid ClaimEvidence instance."""
    return ClaimEvidence(
        title="Test Source",
        url=url,
        excerpt="Test evidence excerpt.",
        relevance_score=0.95,
    )


def _make_verification(
    claim_id: str,
    status: ClaimStatus = ClaimStatus.SUPPORTED,
    *,
    confidence: float = 1.0,
    evidence: list[ClaimEvidence] | None = None,
    freshness_score: float | None = None,
) -> ClaimVerification:
    """Helper to create a valid ClaimVerification instance."""
    return ClaimVerification(
        claim_id=claim_id,
        status=status,
        rationale=f"Rationale for {claim_id}",
        confidence=confidence,
        evidence=evidence if evidence is not None else [],
        freshness_score=freshness_score,
    )


# =====================================================================
# 1. Every ClaimStatus
# =====================================================================

def test_every_claim_status_scoring():
    # Direct function tests
    assert calculate_claim_freshness_score(ClaimStatus.SUPPORTED, is_time_sensitive=True) == 100.0
    assert calculate_claim_freshness_score(ClaimStatus.POTENTIALLY_OUTDATED, is_time_sensitive=True) == 0.0
    assert calculate_claim_freshness_score(ClaimStatus.CONFLICTING, is_time_sensitive=True) == 0.0
    assert calculate_claim_freshness_score(ClaimStatus.INSUFFICIENT_EVIDENCE, is_time_sensitive=True) == 0.0

    # End-to-end score_freshness with all 4 statuses
    claims = [
        _make_claim("c-sup", is_time_sensitive=True),
        _make_claim("c-out", is_time_sensitive=True),
        _make_claim("c-con", is_time_sensitive=True),
        _make_claim("c-ins", is_time_sensitive=True),
    ]
    verifications = [
        _make_verification("c-sup", ClaimStatus.SUPPORTED),
        _make_verification("c-out", ClaimStatus.POTENTIALLY_OUTDATED),
        _make_verification("c-con", ClaimStatus.CONFLICTING),
        _make_verification("c-ins", ClaimStatus.INSUFFICIENT_EVIDENCE),
    ]

    scored, summary = score_freshness(claims, verifications)

    scores_by_id = {v.claim_id: v.freshness_score for v in scored}
    assert scores_by_id["c-sup"] == 100.0
    assert scores_by_id["c-out"] == 0.0
    assert scores_by_id["c-con"] == 0.0
    assert scores_by_id["c-ins"] == 0.0


# =====================================================================
# 2. Time-sensitive and stable claims
# =====================================================================

def test_time_sensitive_and_stable_claims():
    claims = [
        _make_claim("c-ts", is_time_sensitive=True),
        _make_claim("c-stable", is_time_sensitive=False),
    ]
    verifications = [
        _make_verification("c-ts", ClaimStatus.SUPPORTED),
        _make_verification("c-stable", ClaimStatus.SUPPORTED),
    ]

    scored, summary = score_freshness(claims, verifications)

    scores_by_id = {v.claim_id: v.freshness_score for v in scored}
    assert scores_by_id["c-ts"] == 100.0
    assert scores_by_id["c-stable"] is None

    # Overall score considers only the time-sensitive claim
    assert summary.overall_freshness_score == 100.0
    assert summary.time_sensitive_claims == 1
    assert summary.total_claims == 2


# =====================================================================
# 3. Correct individual freshness scores
# =====================================================================

@pytest.mark.parametrize(
    ("status", "is_time_sensitive", "expected_score"),
    [
        (ClaimStatus.SUPPORTED, True, 100.0),
        (ClaimStatus.POTENTIALLY_OUTDATED, True, 0.0),
        (ClaimStatus.CONFLICTING, True, 0.0),
        (ClaimStatus.INSUFFICIENT_EVIDENCE, True, 0.0),
        (ClaimStatus.SUPPORTED, False, None),
        (ClaimStatus.POTENTIALLY_OUTDATED, False, None),
        (ClaimStatus.CONFLICTING, False, None),
        (ClaimStatus.INSUFFICIENT_EVIDENCE, False, None),
    ],
)
def test_correct_individual_freshness_scores(status, is_time_sensitive, expected_score):
    claim = _make_claim("c-1", is_time_sensitive=is_time_sensitive)
    verif = _make_verification("c-1", status=status)

    scored, _ = score_freshness([claim], [verif])
    assert scored[0].freshness_score == expected_score


# =====================================================================
# 4. Correct overall average
# =====================================================================

def test_correct_overall_average_multiple_claims():
    # 2 supported, 1 outdated, 1 conflicting -> (100 + 100 + 0 + 0) / 4 = 50.0
    claims = [
        _make_claim("c1", is_time_sensitive=True),
        _make_claim("c2", is_time_sensitive=True),
        _make_claim("c3", is_time_sensitive=True),
        _make_claim("c4", is_time_sensitive=True),
        _make_claim("c-stable", is_time_sensitive=False),
    ]
    verifications = [
        _make_verification("c1", ClaimStatus.SUPPORTED),
        _make_verification("c2", ClaimStatus.SUPPORTED),
        _make_verification("c3", ClaimStatus.POTENTIALLY_OUTDATED),
        _make_verification("c4", ClaimStatus.CONFLICTING),
        _make_verification("c-stable", ClaimStatus.SUPPORTED),
    ]

    _, summary = score_freshness(claims, verifications)
    assert summary.overall_freshness_score == 50.0


def test_correct_overall_average_fractional():
    # 1 supported, 2 outdated -> 100 / 3 = 33.333333333333336
    claims = [
        _make_claim("c1", is_time_sensitive=True),
        _make_claim("c2", is_time_sensitive=True),
        _make_claim("c3", is_time_sensitive=True),
    ]
    verifications = [
        _make_verification("c1", ClaimStatus.SUPPORTED),
        _make_verification("c2", ClaimStatus.POTENTIALLY_OUTDATED),
        _make_verification("c3", ClaimStatus.POTENTIALLY_OUTDATED),
    ]

    _, summary = score_freshness(claims, verifications)
    assert summary.overall_freshness_score == pytest.approx(100.0 / 3.0)


# =====================================================================
# 5. No time-sensitive claims
# =====================================================================

def test_no_time_sensitive_claims():
    claims = [
        _make_claim("c1", is_time_sensitive=False),
        _make_claim("c2", is_time_sensitive=False),
    ]
    verifications = [
        _make_verification("c1", ClaimStatus.SUPPORTED),
        _make_verification("c2", ClaimStatus.INSUFFICIENT_EVIDENCE),
    ]

    scored, summary = score_freshness(claims, verifications)

    assert all(v.freshness_score is None for v in scored)
    assert summary.overall_freshness_score is None
    assert summary.evidence_coverage == 0.0
    assert summary.total_claims == 2
    assert summary.time_sensitive_claims == 0
    assert summary.supported_claims == 1
    assert summary.insufficient_evidence_claims == 1


# =====================================================================
# 6. Empty claims and empty verifications
# =====================================================================

def test_empty_claims_and_empty_verifications():
    scored, summary = score_freshness([], [])

    assert scored == []
    assert summary.total_claims == 0
    assert summary.time_sensitive_claims == 0
    assert summary.supported_claims == 0
    assert summary.potentially_outdated_claims == 0
    assert summary.conflicting_claims == 0
    assert summary.insufficient_evidence_claims == 0
    assert summary.overall_freshness_score is None
    assert summary.evidence_coverage == 0.0


# =====================================================================
# 7. Evidence coverage with and without evidence
# =====================================================================

def test_evidence_coverage_partial():
    claims = [
        _make_claim("c1", is_time_sensitive=True),
        _make_claim("c2", is_time_sensitive=True),
        _make_claim("c3", is_time_sensitive=True),
        _make_claim("c4", is_time_sensitive=True),
    ]
    verifications = [
        _make_verification("c1", ClaimStatus.SUPPORTED, evidence=[_make_evidence(), _make_evidence("https://b.com")]),
        _make_verification("c2", ClaimStatus.POTENTIALLY_OUTDATED, evidence=[_make_evidence()]),
        _make_verification("c3", ClaimStatus.CONFLICTING, evidence=[]),
        _make_verification("c4", ClaimStatus.INSUFFICIENT_EVIDENCE, evidence=[]),
    ]

    _, summary = score_freshness(claims, verifications)
    assert summary.evidence_coverage == 0.5  # 2 of 4 have evidence


def test_evidence_coverage_full_and_none():
    # All with evidence
    claims_all = [_make_claim("c1", is_time_sensitive=True), _make_claim("c2", is_time_sensitive=True)]
    verifs_all = [
        _make_verification("c1", evidence=[_make_evidence()]),
        _make_verification("c2", evidence=[_make_evidence()]),
    ]
    _, summary_all = score_freshness(claims_all, verifs_all)
    assert summary_all.evidence_coverage == 1.0

    # None with evidence
    claims_none = [_make_claim("c1", is_time_sensitive=True), _make_claim("c2", is_time_sensitive=True)]
    verifs_none = [
        _make_verification("c1", evidence=[]),
        _make_verification("c2", evidence=[]),
    ]
    _, summary_none = score_freshness(claims_none, verifs_none)
    assert summary_none.evidence_coverage == 0.0


def test_stable_claims_do_not_distort_evidence_coverage():
    # 1 TS claim with evidence, 1 stable claim without evidence -> coverage should be 1.0
    claims = [
        _make_claim("c-ts", is_time_sensitive=True),
        _make_claim("c-stable", is_time_sensitive=False),
    ]
    verifications = [
        _make_verification("c-ts", evidence=[_make_evidence()]),
        _make_verification("c-stable", evidence=[]),
    ]

    _, summary = score_freshness(claims, verifications)
    assert summary.evidence_coverage == 1.0


# =====================================================================
# 8. Correct summary counts
# =====================================================================

def test_correct_summary_counts():
    claims = [
        _make_claim("c1", is_time_sensitive=True),   # SUPPORTED
        _make_claim("c2", is_time_sensitive=False),  # SUPPORTED (stable)
        _make_claim("c3", is_time_sensitive=True),   # POTENTIALLY_OUTDATED
        _make_claim("c4", is_time_sensitive=True),   # CONFLICTING
        _make_claim("c5", is_time_sensitive=False),  # INSUFFICIENT_EVIDENCE (stable)
    ]
    verifications = [
        _make_verification("c1", ClaimStatus.SUPPORTED),
        _make_verification("c2", ClaimStatus.SUPPORTED),
        _make_verification("c3", ClaimStatus.POTENTIALLY_OUTDATED),
        _make_verification("c4", ClaimStatus.CONFLICTING),
        _make_verification("c5", ClaimStatus.INSUFFICIENT_EVIDENCE),
    ]

    _, summary = score_freshness(claims, verifications)

    assert summary.total_claims == 5
    assert summary.time_sensitive_claims == 3
    assert summary.supported_claims == 2
    assert summary.potentially_outdated_claims == 1
    assert summary.conflicting_claims == 1
    assert summary.insufficient_evidence_claims == 1

    # Sum of status counts must represent all claims
    total_status = (
        summary.supported_claims
        + summary.potentially_outdated_claims
        + summary.conflicting_claims
        + summary.insufficient_evidence_claims
    )
    assert total_status == summary.total_claims


# =====================================================================
# 9. Missing verification
# =====================================================================

def test_missing_verification_raises_value_error():
    claims = [
        _make_claim("c1"),
        _make_claim("c2"),
    ]
    verifications = [
        _make_verification("c1"),
    ]

    with pytest.raises(ValueError, match="Missing verification"):
        score_freshness(claims, verifications)


# =====================================================================
# 10. Duplicate claim IDs
# =====================================================================

def test_duplicate_claim_ids_raises_value_error():
    claims = [
        _make_claim("c1"),
        _make_claim("c1"),
    ]
    verifications = [
        _make_verification("c1"),
    ]

    with pytest.raises(ValueError, match="Duplicate claim ID"):
        score_freshness(claims, verifications)


# =====================================================================
# 11. Duplicate verification IDs
# =====================================================================

def test_duplicate_verification_ids_raises_value_error():
    claims = [
        _make_claim("c1"),
    ]
    verifications = [
        _make_verification("c1"),
        _make_verification("c1"),
    ]

    with pytest.raises(ValueError, match="Duplicate verification claim ID"):
        score_freshness(claims, verifications)


# =====================================================================
# 12. Unknown verification claim IDs
# =====================================================================

def test_unknown_verification_claim_ids_raises_value_error():
    claims = [
        _make_claim("c1"),
    ]
    verifications = [
        _make_verification("c1"),
        _make_verification("c999"),
    ]

    with pytest.raises(ValueError, match="Unknown verification claim ID"):
        score_freshness(claims, verifications)


# =====================================================================
# 13. Different confidence values must not change freshness scores
# =====================================================================

@pytest.mark.parametrize("confidence", [0.0, 0.1, 0.5, 0.85, 0.99, 1.0])
def test_confidence_does_not_affect_freshness_scores(confidence):
    claims = [
        _make_claim("c-sup", is_time_sensitive=True),
        _make_claim("c-out", is_time_sensitive=True),
        _make_claim("c-con", is_time_sensitive=True),
        _make_claim("c-ins", is_time_sensitive=True),
    ]
    verifications = [
        _make_verification("c-sup", ClaimStatus.SUPPORTED, confidence=confidence),
        _make_verification("c-out", ClaimStatus.POTENTIALLY_OUTDATED, confidence=confidence),
        _make_verification("c-con", ClaimStatus.CONFLICTING, confidence=confidence),
        _make_verification("c-ins", ClaimStatus.INSUFFICIENT_EVIDENCE, confidence=confidence),
    ]

    scored, summary = score_freshness(claims, verifications)

    scores_by_id = {v.claim_id: v.freshness_score for v in scored}
    assert scores_by_id["c-sup"] == 100.0
    assert scores_by_id["c-out"] == 0.0
    assert scores_by_id["c-con"] == 0.0
    assert scores_by_id["c-ins"] == 0.0
    assert summary.overall_freshness_score == 25.0


# =====================================================================
# 14. Input models must remain unchanged
# =====================================================================

def test_input_models_are_not_mutated():
    claim = _make_claim("c1", is_time_sensitive=True)
    verif = _make_verification("c1", ClaimStatus.SUPPORTED, freshness_score=None)

    # Verification before scoring has freshness_score=None
    assert verif.freshness_score is None

    scored, summary = score_freshness([claim], [verif])

    # Returned verification is scored
    assert scored[0].freshness_score == 100.0

    # Input verification is completely unchanged
    assert verif.freshness_score is None
    assert scored[0] is not verif

    # Input claim is completely unchanged
    assert claim.claim_id == "c1"


def test_existing_freshness_score_on_input_is_overwritten_in_copy():
    # If an input verification arrived with an arbitrary score, it must be
    # deterministically recomputed in the output copy, while input stays untouched
    claim = _make_claim("c1", is_time_sensitive=True)
    verif = _make_verification("c1", ClaimStatus.POTENTIALLY_OUTDATED, freshness_score=99.0)

    scored, _ = score_freshness([claim], [verif])

    assert verif.freshness_score == 99.0  # input untouched
    assert scored[0].freshness_score == 0.0  # policy applied to copy


# =====================================================================
# 15. Boundary cases and invalid inputs
# =====================================================================

def test_none_inputs_rejected():
    with pytest.raises(ValueError, match="must not be None"):
        score_freshness(None, [])

    with pytest.raises(ValueError, match="must not be None"):
        score_freshness([], None)


def test_invalid_element_types_rejected():
    claim = _make_claim("c1")
    verif = _make_verification("c1")

    with pytest.raises(TypeError, match="not an instance of Claim"):
        score_freshness(["not_a_claim"], [verif])

    with pytest.raises(TypeError, match="not an instance of ClaimVerification"):
        score_freshness([claim], ["not_a_verif"])


def test_invalid_status_in_calculate_claim_freshness_score():
    with pytest.raises(ValueError, match="Invalid ClaimStatus"):
        calculate_claim_freshness_score("invalid_status_string")

    with pytest.raises(ValueError, match="Invalid ClaimStatus"):
        calculate_claim_freshness_score(12345)


def test_invalid_status_rejected_for_stable_claim():
    with pytest.raises(ValueError, match="Invalid ClaimStatus"):
        calculate_claim_freshness_score("invalid_status_string", is_time_sensitive=False)


@pytest.mark.parametrize("is_time_sensitive", ["false", 1])
def test_non_boolean_time_sensitivity_rejected(is_time_sensitive):
    with pytest.raises(TypeError, match="is_time_sensitive must be a bool"):
        calculate_claim_freshness_score(
            ClaimStatus.SUPPORTED,
            is_time_sensitive=is_time_sensitive,
        )


def test_valid_status_for_stable_claim_returns_none():
    assert calculate_claim_freshness_score(
        ClaimStatus.SUPPORTED,
        is_time_sensitive=False,
    ) is None
    assert calculate_claim_freshness_score(
        "supported",
        is_time_sensitive=False,
    ) is None


def test_alias_score_claims_matches_score_freshness():
    claims = [_make_claim("c1")]
    verifs = [_make_verification("c1")]

    res1 = score_freshness(claims, verifs)
    res2 = score_claims(claims, verifs)

    assert res1 == res2
