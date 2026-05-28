"""Tests for log ingestion and agent tools."""

from sherlock.ingest import parse_logs, log_stats
from sherlock.tools import SessionContext, execute_tool
from sherlock.rag import VectorStore, MockEmbedder


class TestParseLogs:
    def test_empty(self):
        assert parse_logs("") == []

    def test_full_format(self):
        logs = parse_logs("2026-01-01T12:00:00 ERROR api: connection refused")
        assert len(logs) == 1
        assert logs[0].level == "ERROR"
        assert logs[0].service == "api"
        assert "connection refused" in logs[0].message
        assert logs[0].timestamp == "2026-01-01T12:00:00"

    def test_bracket_level_format(self):
        logs = parse_logs("[WARN] disk usage at 90%")
        assert len(logs) == 1
        assert logs[0].level == "WARN"
        assert "disk usage" in logs[0].message

    def test_plain_line_defaults_info(self):
        logs = parse_logs("just a plain message")
        assert len(logs) == 1
        assert logs[0].level == "INFO"

    def test_warning_normalized_to_warn(self):
        logs = parse_logs("[WARNING] something")
        assert logs[0].level == "WARN"

    def test_skips_blank_lines(self):
        logs = parse_logs("line one\n\n\nline two")
        assert len(logs) == 2


class TestLogStats:
    def test_counts_by_level(self):
        logs = parse_logs("ERROR: a\nERROR: b\nINFO: c")
        stats = log_stats(logs)
        assert stats["total"] == 3
        assert stats["by_level"]["ERROR"] == 2
        assert stats["by_level"]["INFO"] == 1

    def test_error_count_includes_critical(self):
        logs = parse_logs("ERROR: a\nCRITICAL: b\nINFO: c")
        stats = log_stats(logs)
        assert stats["error_count"] == 2

    def test_by_service(self):
        logs = parse_logs("2026-01-01T00:00:00 ERROR api: x\n2026-01-01T00:00:01 INFO db: y")
        stats = log_stats(logs)
        assert stats["by_service"]["api"] == 1
        assert stats["by_service"]["db"] == 1


def _make_ctx():
    logs = parse_logs(
        "2026-01-01T00:00:00 INFO api: started\n"
        "2026-01-01T00:00:05 ERROR api: redis connection timeout after 30s\n"
        "2026-01-01T00:00:06 ERROR api: falling back to database\n"
        "2026-01-01T00:00:10 WARN cache: high latency"
    )
    store = VectorStore(MockEmbedder())
    store.add(["When redis fails, the service should fall back to postgres within 1 second using a circuit breaker."])
    return SessionContext(logs=logs, store=store)


class TestTools:
    def test_search_logs_finds_matches(self):
        ctx = _make_ctx()
        out = execute_tool("search_logs", {"query": "redis"}, ctx)
        assert "redis" in out.lower()
        assert "OBSERVATION" in out

    def test_search_logs_no_match(self):
        ctx = _make_ctx()
        out = execute_tool("search_logs", {"query": "nonexistent-term-xyz"}, ctx)
        assert "no log entries matched" in out

    def test_search_logs_level_filter(self):
        ctx = _make_ctx()
        out = execute_tool("search_logs", {"query": "api", "level": "ERROR"}, ctx)
        # Should only include ERROR lines, not the INFO "started" line
        assert "started" not in out

    def test_get_log_stats(self):
        ctx = _make_ctx()
        out = execute_tool("get_log_stats", {}, ctx)
        assert "total log entries" in out
        assert "ERROR" in out

    def test_retrieve_runbook(self):
        ctx = _make_ctx()
        out = execute_tool("retrieve_runbook", {"query": "redis fallback"}, ctx)
        assert "circuit breaker" in out.lower() or "redis" in out.lower()

    def test_unknown_tool(self):
        ctx = _make_ctx()
        out = execute_tool("does_not_exist", {}, ctx)
        assert "unknown tool" in out
