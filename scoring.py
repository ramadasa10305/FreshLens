"""Deterministic freshness scoring engine for FreshLens claims and summaries."""

from __future__ import annotations

from typing import Sequence

from schemas import Claim, ClaimStatus, ClaimVerification, FreshnessSummary

# Freshness scores for time-sensitive claims based on verification status
_CLAIM_FRESHNESS_SCORES: dict[ClaimStatus, float] = {
    ClaimStatus.SUPPORTED: 100.0,
    ClaimStatus.POTENTIALLY_OUTDATED: 0.0,
    ClaimStatus.CONFLICTING: 0.0,
    ClaimStatus.INSUFFICIENT_EVIDENCE: 0.0,
}


def calculate_claim_freshness_score(
    status: ClaimStatus,
    *,
    is_time_sensitive: bool = True,
) -> float | None:
    """Return the deterministic claim-level freshness score.

    Policy:
    - Stable (non-time-sensitive) claims have freshness_score = None.
    - Time-sensitive claims:
        SUPPORTED = 100.0
        POTENTIALLY_OUTDATED = 0.0
        CONFLICTING = 0.0
        INSUFFICIENT_EVIDENCE = 0.0

    A score of 0.0 means freshness has not been established. It does not
    necessarily mean the claim is false or definitely outdated.
    """
    if not isinstance(is_time_sensitive, bool):
        raise TypeError("is_time_sensitive must be a bool.")

    if not isinstance(status, ClaimStatus):
        try:
            status = ClaimStatus(status)
        except (ValueError, TypeError):
            raise ValueError(f"Invalid ClaimStatus: {status!r}") from None

    if not is_time_sensitive:
        return None

    return _CLAIM_FRESHNESS_SCORES[status]


def score_freshness(
    claims: Sequence[Claim],
    verifications: Sequence[ClaimVerification],
) -> tuple[list[ClaimVerification], FreshnessSummary]:
    """Score claim verifications and compute FreshnessSummary deterministically.

    Validates 1-to-1 correspondence between claims and verifications.
    Rejects duplicate claim IDs, duplicate verification IDs, missing
    verifications, and unknown verification IDs.

    Does not mutate input models. Returns newly constructed scored
    ClaimVerification copies alongside the FreshnessSummary.

    Scoring Policy:
    1. Claim-level freshness score:
       - SUPPORTED = 100.0
       - POTENTIALLY_OUTDATED = 0.0
       - CONFLICTING = 0.0
       - INSUFFICIENT_EVIDENCE = 0.0
       - Stable (non-time-sensitive) claims = None
    2. Overall freshness score:
       - Arithmetic mean of time-sensitive claim freshness scores.
       - None if there are no time-sensitive claims.
       - Confidence is not used as a weighting factor.
    3. Evidence coverage:
       - Proportion of time-sensitive claims with at least one evidence item.
       - 0.0 if there are no time-sensitive claims.
    4. Summary status counts:
       - Represent all claims (both time-sensitive and stable).
    """
    if claims is None or verifications is None:
        raise ValueError("claims and verifications must not be None.")

    # 1. Validate element types and claim ID uniqueness
    claim_map: dict[str, Claim] = {}
    for idx, c in enumerate(claims):
        if not isinstance(c, Claim):
            raise TypeError(
                f"Item at claims[{idx}] is not an instance of Claim: {type(c).__name__}"
            )
        if c.claim_id in claim_map:
            raise ValueError(f"Duplicate claim ID found in claims: '{c.claim_id}'.")
        claim_map[c.claim_id] = c

    # 2. Validate verification element types and ID uniqueness
    verif_map: dict[str, ClaimVerification] = {}
    for idx, v in enumerate(verifications):
        if not isinstance(v, ClaimVerification):
            raise TypeError(
                f"Item at verifications[{idx}] is not an instance of ClaimVerification: {type(v).__name__}"
            )
        if v.claim_id in verif_map:
            raise ValueError(
                f"Duplicate verification claim ID found in verifications: '{v.claim_id}'."
            )
        verif_map[v.claim_id] = v

    # 3. Validate claim-verification correspondence
    claim_ids = set(claim_map.keys())
    verif_ids = set(verif_map.keys())

    missing_verifs = claim_ids - verif_ids
    if missing_verifs:
        missing_sorted = sorted(missing_verifs)
        raise ValueError(f"Missing verification for claim IDs: {missing_sorted}.")

    unknown_verifs = verif_ids - claim_ids
    if unknown_verifs:
        unknown_sorted = sorted(unknown_verifs)
        raise ValueError(
            f"Unknown verification claim IDs not matching any claim: {unknown_sorted}."
        )

    # 4. Score verifications (aligned with claims order)
    scored_verifications: list[ClaimVerification] = []
    ts_scores: list[float] = []
    ts_claims_count = 0
    ts_with_evidence_count = 0

    # Summary counts across ALL claims (including stable)
    supported_count = 0
    potentially_outdated_count = 0
    conflicting_count = 0
    insufficient_evidence_count = 0

    for c in claims:
        v = verif_map[c.claim_id]

        # Count verification statuses across all claims
        if v.status == ClaimStatus.SUPPORTED:
            supported_count += 1
        elif v.status == ClaimStatus.POTENTIALLY_OUTDATED:
            potentially_outdated_count += 1
        elif v.status == ClaimStatus.CONFLICTING:
            conflicting_count += 1
        elif v.status == ClaimStatus.INSUFFICIENT_EVIDENCE:
            insufficient_evidence_count += 1

        # Calculate claim-level freshness score
        claim_score = calculate_claim_freshness_score(
            v.status,
            is_time_sensitive=c.is_time_sensitive,
        )

        if c.is_time_sensitive:
            ts_claims_count += 1
            if claim_score is not None:
                ts_scores.append(claim_score)
            if len(v.evidence) > 0:
                ts_with_evidence_count += 1

        # Create copy with updated freshness_score without mutating input
        scored_v = v.model_copy(update={"freshness_score": claim_score})
        scored_verifications.append(scored_v)

    # 5. Overall freshness score
    if ts_claims_count == 0:
        overall_freshness_score = None
    else:
        overall_freshness_score = sum(ts_scores) / ts_claims_count

    # 6. Evidence coverage
    if ts_claims_count == 0:
        evidence_coverage = 0.0
    else:
        evidence_coverage = ts_with_evidence_count / ts_claims_count

    # 7. Construct FreshnessSummary
    summary = FreshnessSummary(
        total_claims=len(claims),
        time_sensitive_claims=ts_claims_count,
        supported_claims=supported_count,
        potentially_outdated_claims=potentially_outdated_count,
        conflicting_claims=conflicting_count,
        insufficient_evidence_claims=insufficient_evidence_count,
        overall_freshness_score=overall_freshness_score,
        evidence_coverage=evidence_coverage,
    )

    return scored_verifications, summary


# Public aliases
score_claims = score_freshness


__all__ = [
    "calculate_claim_freshness_score",
    "score_claims",
    "score_freshness",
]
