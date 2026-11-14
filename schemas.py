"""Shared Pydantic data models for FreshLens claim analysis and freshness scoring."""

from __future__ import annotations

from enum import Enum
from pydantic import BaseModel, ConfigDict, Field


MAX_ANSWER_CHARS = 12_000


def validate_answer_length(answer: str) -> None:
    """Reject answer strings that exceed the shared request-size limit."""
    if isinstance(answer, str) and len(answer) > MAX_ANSWER_CHARS:
        raise ValueError(
            f"Answer must be {MAX_ANSWER_CHARS:,} characters or fewer."
        )


class ClaimStatus(str, Enum):
    """Verification status for an individual time-sensitive claim."""

    SUPPORTED = "supported"
    POTENTIALLY_OUTDATED = "potentially_outdated"
    CONFLICTING = "conflicting"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class Claim(BaseModel):
    """An individual atomic claim extracted from an AI-generated answer."""

    model_config = ConfigDict(str_strip_whitespace=True)

    claim_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    is_time_sensitive: bool = Field(strict=True)
    category: str = Field(min_length=1)
    time_sensitivity_reason: str | None = None


class ClaimEvidence(BaseModel):
    """Retrieved search evidence evaluating the veracity/freshness of a claim."""

    model_config = ConfigDict(str_strip_whitespace=True)

    title: str
    url: str
    excerpt: str
    published_date: str | None = None
    relevance_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        strict=True,
        allow_inf_nan=False,
    )


class ClaimVerification(BaseModel):
    """Verification result connecting a claim with retrieved evidence and a status."""

    model_config = ConfigDict(str_strip_whitespace=True)

    claim_id: str = Field(min_length=1)
    status: ClaimStatus
    rationale: str = Field(min_length=1)
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        strict=True,
        allow_inf_nan=False,
    )
    newer_information: str | None = None
    evidence: list[ClaimEvidence] = Field(default_factory=list)
    freshness_score: float | None = Field(
        default=None,
        ge=0.0,
        le=100.0,
        strict=True,
        allow_inf_nan=False,
    )


class FreshnessSummary(BaseModel):
    """Aggregated freshness metrics across all claims extracted from an answer."""

    model_config = ConfigDict(str_strip_whitespace=True)

    total_claims: int = Field(ge=0, strict=True)
    time_sensitive_claims: int = Field(ge=0, strict=True)
    supported_claims: int = Field(ge=0, strict=True)
    potentially_outdated_claims: int = Field(ge=0, strict=True)
    conflicting_claims: int = Field(ge=0, strict=True)
    insufficient_evidence_claims: int = Field(ge=0, strict=True)
    overall_freshness_score: float | None = Field(
        default=None,
        ge=0.0,
        le=100.0,
        strict=True,
        allow_inf_nan=False,
    )
    evidence_coverage: float = Field(
        ge=0.0,
        le=1.0,
        strict=True,
        allow_inf_nan=False,
    )


class FreshnessReport(BaseModel):
    """Comprehensive freshness evaluation report for an AI-generated answer."""

    model_config = ConfigDict(str_strip_whitespace=True)

    original_answer: str = Field(min_length=1)
    claims: list[Claim] = Field(default_factory=list)
    verifications: list[ClaimVerification] = Field(default_factory=list)
    summary: FreshnessSummary


__all__ = [
    "Claim",
    "ClaimEvidence",
    "ClaimStatus",
    "ClaimVerification",
    "FreshnessReport",
    "FreshnessSummary",
]
