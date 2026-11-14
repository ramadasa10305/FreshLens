"""MCP tool tests using the official SDK's in-memory client."""

from __future__ import annotations

import asyncio
import importlib
import subprocess
import sys
import types
from typing import Any

import pytest
from mcp.client import Client
from mcp.server import MCPServer

import mcp_server
from schemas import (
    Claim,
    ClaimEvidence,
    ClaimStatus,
    ClaimVerification,
    FreshnessReport,
    FreshnessSummary,
)


def _sample_report(answer: str = "  Original answer, unchanged.  ") -> FreshnessReport:
    claims = [
        Claim(
            claim_id="claim-001",
            text="The Atlas service is available.",
            is_time_sensitive=True,
            category="technology",
            time_sensitivity_reason="Availability changes.",
        ),
        Claim(
            claim_id="claim-002",
            text="Water contains hydrogen and oxygen.",
            is_time_sensitive=False,
            category="science",
            time_sensitivity_reason=None,
        ),
    ]
    evidence = ClaimEvidence(
        title="Atlas status",
        url="https://example.com/atlas",
        excerpt="Atlas is currently available.",
        published_date="2026-11-01",
        relevance_score=0.95,
    )
    verifications = [
        ClaimVerification(
            claim_id="claim-001",
            status=ClaimStatus.SUPPORTED,
            rationale="The current status page supports availability.",
            confidence=0.91,
            newer_information=None,
            evidence=[evidence],
            freshness_score=100.0,
        ),
        ClaimVerification(
            claim_id="claim-002",
            status=ClaimStatus.CONFLICTING,
            rationale="The evidence contradicts the claim.",
            confidence=0.72,
            newer_information="The source gives different details.",
            evidence=[],
            freshness_score=None,
        ),
    ]
    summary = FreshnessSummary(
        total_claims=2,
        time_sensitive_claims=1,
        supported_claims=1,
        potentially_outdated_claims=0,
        conflicting_claims=1,
        insufficient_evidence_claims=0,
        overall_freshness_score=100.0,
        evidence_coverage=1.0,
    )
    # The exact input can include edge whitespace; preserve it through model construction.
    return FreshnessReport.model_construct(
        original_answer=answer,
        claims=claims,
        verifications=verifications,
        summary=summary,
    )


class FakePipeline:
    def __init__(self, *, fail: bool = False):
        self.calls: list[str] = []
        self.fail = fail

    def analyze(self, answer: str) -> FreshnessReport:
        self.calls.append(answer)
        if self.fail:
            raise RuntimeError("pipeline failed with protected detail")
        return _sample_report(answer)


def _run(coro):
    return asyncio.run(coro)


async def _list_tools(server: MCPServer):
    async with Client(server) as client:
        listed = await client.list_tools()
        return client.server_info, listed.tools


async def _call_tool(server: MCPServer, answer: str):
    async with Client(server) as client:
        return await client.call_tool("verify_answer_freshness", {"answer": answer})


def test_server_name_and_exactly_one_tool_contract():
    server = mcp_server.create_mcp_server(FakePipeline())
    info, tools = _run(_list_tools(server))

    assert server.name == "FreshLens"
    assert info.name == "FreshLens"
    assert len(tools) == 1
    tool = tools[0]
    assert tool.name == "verify_answer_freshness"
    assert "freshness" in tool.description.lower()
    assert tool.input_schema["type"] == "object"
    assert tool.input_schema["properties"]["answer"]["type"] == "string"
    assert tool.input_schema["required"] == ["answer"]
    assert "original_answer" in tool.output_schema["properties"]
    assert "claims" in tool.output_schema["properties"]
    assert "verifications" in tool.output_schema["properties"]
    assert "summary" in tool.output_schema["properties"]


def test_tool_invocation_returns_complete_structured_report_and_forwards_answer():
    pipeline = FakePipeline()
    server = mcp_server.create_mcp_server(pipeline)
    answer = "\n  Keep this answer exactly.  \n"

    result = _run(_call_tool(server, answer))

    assert not result.is_error
    assert pipeline.calls == [answer]
    assert result.structured_content == _sample_report(answer).model_dump(mode="json")
    assert result.structured_content["original_answer"] == answer
    assert len(result.structured_content["claims"]) == 2
    assert len(result.structured_content["verifications"]) == 2
    assert result.structured_content["summary"]["total_claims"] == 2


def test_oversized_answer_is_rejected_before_pipeline_invocation():
    pipeline = FakePipeline()
    server = mcp_server.create_mcp_server(pipeline)

    result = _run(_call_tool(server, "x" * 12_001))

    assert result.is_error
    assert pipeline.calls == []


def test_injected_pipeline_is_reused_for_multiple_calls():
    pipeline = FakePipeline()
    server = mcp_server.create_mcp_server(pipeline)

    first = _run(_call_tool(server, "first answer"))
    second = _run(_call_tool(server, "second answer"))

    assert not first.is_error and not second.is_error
    assert pipeline.calls == ["first answer", "second answer"]


def test_default_pipeline_is_instantiated_lazily_and_reused(monkeypatch):
    constructions: list[object] = []

    class ConstructedPipeline(FakePipeline):
        def __init__(self):
            super().__init__()
            constructions.append(self)

    fake_pipeline_module = types.ModuleType("pipeline")
    fake_pipeline_module.FreshnessPipeline = ConstructedPipeline
    monkeypatch.setitem(sys.modules, "pipeline", fake_pipeline_module)

    server = mcp_server.create_mcp_server()
    assert constructions == []
    first = _run(_call_tool(server, "one"))
    second = _run(_call_tool(server, "two"))

    assert not first.is_error and not second.is_error
    assert len(constructions) == 1
    assert constructions[0].calls == ["one", "two"]


def test_module_import_needs_no_api_config_or_credentials():
    code = """
import importlib.abc, sys
class BlockConfig(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'config':
            raise AssertionError('config must not be imported at module import')
sys.meta_path.insert(0, BlockConfig())
import mcp_server
assert mcp_server.mcp.name == 'FreshLens'
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == ""


def test_import_does_not_start_server_and_emits_no_stdout(monkeypatch, capsys):
    run_calls: list[Any] = []
    monkeypatch.setattr(MCPServer, "run", lambda self, *args, **kwargs: run_calls.append(self))

    reloaded = importlib.reload(mcp_server)
    output = capsys.readouterr()

    assert run_calls == []
    assert output.out == ""
    assert reloaded.mcp.name == "FreshLens"


def test_pipeline_failure_is_reported_as_tool_error_without_secret_detail():
    server = mcp_server.create_mcp_server(FakePipeline(fail=True))

    result = _run(_call_tool(server, "answer"))

    assert result.is_error
    rendered = " ".join(getattr(block, "text", "") for block in result.content)
    assert "Error executing tool verify_answer_freshness" in rendered
    assert "protected detail" not in rendered


def test_tool_call_does_not_write_diagnostics_to_stdout(capsys):
    server = mcp_server.create_mcp_server(FakePipeline())
    result = _run(_call_tool(server, "no stdout please"))

    assert not result.is_error
    assert capsys.readouterr().out == ""
