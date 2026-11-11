"""STDIO MCP server exposing FreshLens answer freshness verification."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from mcp.server import MCPServer
from schemas import FreshnessReport

if TYPE_CHECKING:
    from pipeline import FreshnessPipeline
    from schemas import FreshnessReport

logger = logging.getLogger(__name__)


def create_mcp_server(pipeline: FreshnessPipeline | None = None) -> MCPServer:
    """Create the FreshLens MCP server with an optional injected pipeline."""
    pipeline_instance = pipeline
    server = MCPServer("FreshLens")

    def get_pipeline() -> FreshnessPipeline:
        nonlocal pipeline_instance
        if pipeline_instance is None:
            # Import only when first invoked: pipeline clients read credentials
            # through config.py during construction.
            from pipeline import FreshnessPipeline as Pipeline

            pipeline_instance = Pipeline()
        return pipeline_instance

    @server.tool()
    def verify_answer_freshness(answer: str) -> FreshnessReport:
        """Analyze an AI-generated answer and return its structured freshness report."""
        logger.info("Verifying freshness for submitted answer")
        return get_pipeline().analyze(answer)

    return server


mcp = create_mcp_server()


if __name__ == "__main__":
    mcp.run()
