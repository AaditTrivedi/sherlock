"""Tests for the agent loop and LLM abstraction."""

from sherlock.llm import MockLLM, LLMResponse, get_llm
from sherlock.agent import run_investigation
from sherlock.tools import SessionContext
from sherlock.ingest import parse_logs
from sherlock.rag import VectorStore, MockEmbedder


def _make_ctx():
    logs = parse_logs(
        "2026-01-01T00:00:05 ERROR api: redis connection timeout after 30s\n"
        "2026-01-01T00:00:06 ERROR api: falling back to database"
    )
    store = VectorStore(MockEmbedder())
    store.add(["Redis failures should trigger a circuit breaker and fall back to postgres."])
    return SessionContext(logs=logs, store=store)


class TestLLMResponse:
    def test_wants_tools_true(self):
        r = LLMResponse(tool_calls=[{"id": "1", "name": "x", "input": {}}])
        assert r.wants_tools is True

    def test_wants_tools_false(self):
        r = LLMResponse(text="done")
        assert r.wants_tools is False


class TestMockLLM:
    def test_first_turn_calls_tool(self):
        llm = MockLLM()
        from sherlock.tools import TOOL_SCHEMAS
        resp = llm.chat([{"role": "user", "content": "why did it fail?"}], TOOL_SCHEMAS, "sys")
        assert resp.wants_tools

    def test_eventually_produces_text(self):
        llm = MockLLM()
        from sherlock.tools import TOOL_SCHEMAS
        msgs = [{"role": "user", "content": "why?"}]
        # Drive several turns; it must eventually stop calling tools
        produced_text = False
        for _ in range(5):
            resp = llm.chat(msgs, TOOL_SCHEMAS, "sys")
            if not resp.wants_tools:
                produced_text = True
                break
            msgs.append({"role": "user", "content": "OBSERVATION: data"})
        assert produced_text


class TestAgentLoop:
    def test_investigation_completes(self):
        result = run_investigation(MockLLM(), _make_ctx(), "Why did the API fail?", max_steps=6)
        assert result.root_cause
        assert result.question == "Why did the API fail?"
        assert len(result.steps) >= 1

    def test_investigation_uses_tools(self):
        result = run_investigation(MockLLM(), _make_ctx(), "Why?", max_steps=6)
        assert result.tool_call_count >= 1

    def test_steps_are_numbered(self):
        result = run_investigation(MockLLM(), _make_ctx(), "Why?", max_steps=6)
        step_numbers = [s.step for s in result.steps]
        assert step_numbers == sorted(step_numbers)

    def test_respects_max_steps(self):
        # Even with a tiny budget, it must return a result without crashing
        result = run_investigation(MockLLM(), _make_ctx(), "Why?", max_steps=1)
        assert result.root_cause


class TestGetLLM:
    def test_falls_back_to_mock_without_keys(self, monkeypatch):
        for key in ("GROQ_API_KEY", "GEMINI_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
            monkeypatch.delenv(key, raising=False)
        llm = get_llm()
        assert llm.mode == "mock"

    def test_groq_selected_when_key_present(self, monkeypatch):
        for key in ("GEMINI_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
            monkeypatch.delenv(key, raising=False)
        monkeypatch.setenv("GROQ_API_KEY", "test-key")
        llm = get_llm()
        assert llm.provider == "groq"
        assert llm.mode == "live"
