"""
Agent tools for Sherlock.

Each tool has a JSON-schema definition (sent to the LLM so it knows how to call
it) and an executor. A SessionContext bundles the per-investigation state (parsed
logs + RAG vector store) the tools operate on.
"""

from sherlock.models import LogEntry
from sherlock.rag import VectorStore
from sherlock.ingest import log_stats


class SessionContext:
    """Per-investigation state shared by all tools."""

    def __init__(self, logs: list[LogEntry], store: VectorStore):
        self.logs = logs
        self.store = store


# ── Tool schemas (provider-agnostic; converted per-provider in llm.py) ──

TOOL_SCHEMAS = [
    {
        "name": "search_logs",
        "description": "Search the ingested application logs for entries matching a keyword or phrase. Returns matching log lines with their level and service. Use this to find errors, warnings, or specific events.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Keyword or phrase to search for, e.g. 'timeout', 'error', '500'"},
                "level": {"type": "string", "description": "Optional level filter: ERROR, WARN, INFO, CRITICAL, DEBUG"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "retrieve_runbook",
        "description": "Semantically search the runbook and documentation for guidance relevant to the incident. Returns the most relevant passages. Use this to find known causes, fixes, or operational procedures.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to look up, e.g. 'redis connection failure handling'"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_log_stats",
        "description": "Get summary statistics of the ingested logs: total entries, counts by level, counts by service, and total error count. Use this first to understand the overall shape of the incident.",
        "parameters": {"type": "object", "properties": {}},
    },
]


def execute_tool(name: str, tool_input: dict, ctx: SessionContext) -> str:
    """Dispatch a tool call to its implementation and return a string observation."""
    if name == "search_logs":
        return _search_logs(tool_input.get("query", ""), tool_input.get("level"), ctx)
    if name == "retrieve_runbook":
        return _retrieve_runbook(tool_input.get("query", ""), ctx)
    if name == "get_log_stats":
        return _get_log_stats(ctx)
    return f"OBSERVATION: unknown tool '{name}'"


def _search_logs(query: str, level: str | None, ctx: SessionContext) -> str:
    q = query.lower()
    matches: list[LogEntry] = []
    for e in ctx.logs:
        if level and e.level != level.upper():
            continue
        if q in e.message.lower() or (e.service and q in e.service.lower()) or q in e.level.lower():
            matches.append(e)
    if not matches:
        return f"OBSERVATION: no log entries matched '{query}'."
    lines = []
    for e in matches[:15]:
        svc = f" {e.service}" if e.service else ""
        ts = f"{e.timestamp} " if e.timestamp else ""
        lines.append(f"{ts}{e.level}{svc}: {e.message}")
    suffix = f" (showing 15 of {len(matches)})" if len(matches) > 15 else ""
    return f"OBSERVATION: found {len(matches)} matching log(s){suffix}:\n" + "\n".join(lines)


def _retrieve_runbook(query: str, ctx: SessionContext) -> str:
    results = ctx.store.search(query, k=3)
    if not results:
        return "OBSERVATION: no runbook content available."
    out = []
    for i, (doc, score) in enumerate(results, 1):
        snippet = doc[:400] + ("..." if len(doc) > 400 else "")
        out.append(f"[{i}] (relevance {score:.2f}) {snippet}")
    return "OBSERVATION: top runbook passages:\n" + "\n".join(out)


def _get_log_stats(ctx: SessionContext) -> str:
    stats = log_stats(ctx.logs)
    by_level = ", ".join(f"{k}={v}" for k, v in stats["by_level"].items())
    by_service = ", ".join(f"{k}={v}" for k, v in stats["by_service"].items()) or "none tagged"
    return (
        f"OBSERVATION: {stats['total']} total log entries. "
        f"Levels: {by_level}. Services: {by_service}. "
        f"Total errors (ERROR+CRITICAL): {stats['error_count']}."
    )
