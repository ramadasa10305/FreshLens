"""Mocked end-to-end regression tests for the M8 FreshLens workflow."""

from __future__ import annotations

import json
from typing import Any

import pytest

from clients.tavily_client import TavilySearchResponse, TavilySearchResult
from pipeline import FreshnessPipeline
from schemas import ClaimStatus


_EXTRACTED_CLAIMS = [
    {
        "text": "The Nova phone is currently in stock.",
        "is_time_sensitive": True,
        "category": "product availability",
        "time_sensitivity_reason": "Availability changes.",
    },
    {
        "text": "The Orion app is currently version 4.2.",
        "is_time_sensitive": True,
        "category": "technology",
        "time_sensitivity_reason": "Software versions change.",
    },
    {
        "text": "Water is composed of hydrogen and oxygen.",
        "is_time_sensitive": False,
        "category": "science",
        "time_sensitivity_reason": None,
    },
]


def _tavily_result(query: str) -> TavilySearchResult:
    slug = query.lower().split()[1]
    return TavilySearchResult(
        title=f"Source for {slug}",
        url=f"https://evidence.example/{slug}",
        content=f"Unique evidence for {query}",
        published_date="2026-10-01",
        score=0.8,
    )


def _verification_payload(
    status: str,
    confidence: float,
    rationale: str,
) -> str:
    return json.dumps(
        {
            "status": status,
            "rationale": rationale,
            "confidence": confidence,
            "newer_information": None,
        }
    )


class FakeTavilyClient:
    """Implements the injected Tavily search interface without network access."""

    def __init__(self, results_by_query: dict[str, list[TavilySearchResult]]):
        self.results_by_query = results_by_query
        self.search_calls: list[tuple[str, dict[str, Any]]] = []
        self.error: Exception | None = None

    def search(self, query: str, **kwargs: Any) -> TavilySearchResponse:
        self.search_calls.append((query, kwargs))
        if self.error is not None:
            raise self.error
        return TavilySearchResponse(
            query=query,
            results=tuple(self.results_by_query.get(query, [])),
        )


class FakeBifrostClient:
    """Implements Bifrost text generation for extraction and verification."""

    def __init__(self, *, confidence: float = 0.2):
        self.confidence = confidence
        self.calls: list[tuple[str, str]] = []
        self.verification_inputs: list[tuple[str, list[dict[str, Any]]]] = []
        self.error: Exception | None = None

    def generate_text(
        self,
        prompt: str,
        system_prompt: str = "You are a helpful AI assistant.",
    ) -> str:
        self.calls.append((prompt, system_prompt))
        if self.error is not None:
            raise self.error

        if "extract all atomic factual claims" in prompt:
            return json.dumps({"claims": _EXTRACTED_CLAIMS})

        claim_line = next(line for line in prompt.splitlines() if line.startswith("Claim: "))
        claim = json.loads(claim_line.removeprefix("Claim: "))
        evidence_line = next(
            line for line in prompt.splitlines() if line.startswith("Retrieved evidence: ")
        )
        evidence = json.loads(evidence_line.removeprefix("Retrieved evidence: "))
        self.verification_inputs.append((claim, evidence))

        if claim.startswith("The Nova"):
            return _verification_payload(
                ClaimStatus.SUPPORTED.value,
                self.confidence,
                "The claim-specific source confirms availability.",
            )
        if claim.startswith("The Orion"):
            return _verification_payload(
                ClaimStatus.POTENTIALLY_OUTDATED.value,
                self.confidence,
                "The source indicates a newer Orion version.",
            )
        return _verification_payload(
            ClaimStatus.CONFLICTING.value,
            self.confidence,
            "The source contradicts the claim wording.",
        )


def _complete_pipeline(*, confidence: float = 0.2):
    results = {
        claim["text"]: [_tavily_result(claim["text"])]
        for claim in _EXTRACTED_CLAIMS
    }
    bifrost = FakeBifrostClient(confidence=confidence)
    tavily = FakeTavilyClient(results)
    answer = "  Nova is in stock. Orion is version 4.2. Water contains hydrogen and oxygen.  "
    report = FreshnessPipeline(bifrost, tavily).analyze(answer)
    return report, bifrost, tavily, answer


def test_complete_mocked_workflow_preserves_association_and_scoring():
    report, bifrost, tavily, answer = _complete_pipeline()

    assert report.original_answer == answer
    assert [claim.text for claim in report.claims] == [c["text"] for c in _EXTRACTED_CLAIMS]
    assert [query for query, _ in tavily.search_calls] == [c["text"] for c in _EXTRACTED_CLAIMS]
    assert all(options["max_results"] == 5 for _, options in tavily.search_calls)
    assert [claim for claim, _ in bifrost.verification_inputs] == [c["text"] for c in _EXTRACTED_CLAIMS]

    for claim, evidence in bifrost.verification_inputs:
        assert len(evidence) == 1
        assert evidence[0]["excerpt"] == f"Unique evidence for {claim}"

    assert [v.claim_id for v in report.verifications] == [c.claim_id for c in report.claims]
    assert [v.status for v in report.verifications] == [
        ClaimStatus.SUPPORTED,
        ClaimStatus.POTENTIALLY_OUTDATED,
        ClaimStatus.CONFLICTING,
    ]
    assert [v.evidence[0].excerpt for v in report.verifications] == [
        f"Unique evidence for {claim['text']}" for claim in _EXTRACTED_CLAIMS
    ]
    assert [v.freshness_score for v in report.verifications] == [100.0, 0.0, None]
    assert report.summary.total_claims == 3
    assert report.summary.time_sensitive_claims == 2
    assert report.summary.supported_claims == 1
    assert report.summary.potentially_outdated_claims == 1
    assert report.summary.conflicting_claims == 1
    assert report.summary.insufficient_evidence_claims == 0
    assert report.summary.overall_freshness_score == 50.0
    assert report.summary.evidence_coverage == 1.0


def test_empty_extraction_returns_valid_empty_report():
    bifrost = FakeBifrostClient()
    tavily = FakeTavilyClient({})

    # Supply a dedicated extractor response without mocking pipeline components.
    bifrost.generate_text = lambda prompt, system_prompt="": json.dumps({"claims": []})
    report = FreshnessPipeline(bifrost, tavily).analyze("A greeting without factual claims.")

    assert report.claims == []
    assert report.verifications == []
    assert report.summary.total_claims == 0
    assert report.summary.time_sensitive_claims == 0
    assert report.summary.overall_freshness_score is None
    assert report.summary.evidence_coverage == 0.0
    assert tavily.search_calls == []


def test_no_evidence_only_skips_model_call_for_that_claim():
    claims = _EXTRACTED_CLAIMS[:2]
    bifrost = FakeBifrostClient()
    tavily = FakeTavilyClient({claims[0]["text"]: [_tavily_result(claims[0]["text"])]})
    report = FreshnessPipeline(bifrost, tavily).analyze("Nova is in stock; Orion is version 4.2.")

    assert len(bifrost.calls) == 2  # extraction and one evidence-backed verification
    assert report.verifications[1].status is ClaimStatus.INSUFFICIENT_EVIDENCE
    assert report.verifications[1].confidence == 0.0
    assert report.verifications[1].evidence == []
    assert report.verifications[1].freshness_score == 0.0


@pytest.mark.parametrize("failing_client", ["tavily", "bifrost"])
def test_external_client_failures_propagate(failing_client: str):
    bifrost = FakeBifrostClient()
    tavily = FakeTavilyClient({claim["text"]: [_tavily_result(claim["text"])] for claim in _EXTRACTED_CLAIMS})
    if failing_client == "tavily":
        tavily.error = RuntimeError("mock Tavily failure")
    else:
        bifrost.error = RuntimeError("mock Bifrost failure")

    with pytest.raises(RuntimeError, match="mock"):
        FreshnessPipeline(bifrost, tavily).analyze("Answer with factual claims.")


def test_confidence_does_not_change_deterministic_scores():
    low_report, _, _, _ = _complete_pipeline(confidence=0.05)
    high_report, _, _, _ = _complete_pipeline(confidence=0.99)

    assert [v.confidence for v in low_report.verifications] == [0.05, 0.05, 0.05]
    assert [v.confidence for v in high_report.verifications] == [0.99, 0.99, 0.99]
    assert [v.freshness_score for v in low_report.verifications] == [
        v.freshness_score for v in high_report.verifications
    ]
    assert low_report.summary.overall_freshness_score == high_report.summary.overall_freshness_score == 50.0
