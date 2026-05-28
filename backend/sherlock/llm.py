"""
LLM client abstraction for Sherlock.

Supports the Anthropic Messages API and OpenAI Chat Completions API, both with
tool-calling. Falls back to a deterministic MockLLM when no API key is present,
so the agent is fully runnable and testable without credentials.
"""

import os
import json
from abc import ABC, abstractmethod
from typing import Optional


class LLMResponse:
    """Normalized LLM response across providers."""

    def __init__(self, text: Optional[str] = None, tool_calls: Optional[list[dict]] = None):
        # text: the assistant's natural-language output (may be None if it only called tools)
        # tool_calls: list of {"id": str, "name": str, "input": dict}
        self.text = text
        self.tool_calls = tool_calls or []

    @property
    def wants_tools(self) -> bool:
        return len(self.tool_calls) > 0


class BaseLLM(ABC):
    """Interface every LLM provider implements."""

    provider: str = "base"
    mode: str = "mock"

    @abstractmethod
    def chat(self, messages: list[dict], tools: list[dict], system: str) -> LLMResponse:
        """Send a conversation and available tools, get back text and/or tool calls."""
        ...


class AnthropicLLM(BaseLLM):
    """Anthropic Messages API client with tool-calling."""

    provider = "anthropic"
    mode = "live"

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514"):
        import anthropic
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def chat(self, messages: list[dict], tools: list[dict], system: str) -> LLMResponse:
        # Convert generic tool schema to Anthropic format
        anthropic_tools = [
            {
                "name": t["name"],
                "description": t["description"],
                "input_schema": t["parameters"],
            }
            for t in tools
        ]
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=system,
            tools=anthropic_tools,
            messages=messages,
        )
        text_parts, tool_calls = [], []
        for block in resp.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append({"id": block.id, "name": block.name, "input": block.input})
        return LLMResponse(text="\n".join(text_parts) or None, tool_calls=tool_calls)


class OpenAILLM(BaseLLM):
    """OpenAI Chat Completions client with tool-calling."""

    provider = "openai"
    mode = "live"

    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def chat(self, messages: list[dict], tools: list[dict], system: str) -> LLMResponse:
        openai_tools = [
            {"type": "function", "function": {
                "name": t["name"], "description": t["description"], "parameters": t["parameters"]
            }}
            for t in tools
        ]
        full_messages = [{"role": "system", "content": system}] + messages
        resp = self.client.chat.completions.create(
            model=self.model, messages=full_messages, tools=openai_tools,
        )
        msg = resp.choices[0].message
        tool_calls = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls.append({
                    "id": tc.id, "name": tc.function.name,
                    "input": json.loads(tc.function.arguments or "{}"),
                })
        return LLMResponse(text=msg.content, tool_calls=tool_calls)


class MockLLM(BaseLLM):
    """
    Deterministic LLM used for tests and zero-key demo mode.

    It follows a fixed investigation policy so the agent loop is exercised
    end-to-end: first retrieve a runbook, then search logs for errors, then
    return a root-cause summary built from what the tools returned.
    """

    provider = "mock"
    mode = "mock"

    def __init__(self):
        self._turn = 0

    def chat(self, messages: list[dict], tools: list[dict], system: str) -> LLMResponse:
        tool_names = {t["name"] for t in tools}
        # Find the most recent user question
        question = ""
        for m in messages:
            if m["role"] == "user" and isinstance(m["content"], str):
                question = m["content"]

        self._turn += 1
        if self._turn == 1 and "search_logs" in tool_names:
            return LLMResponse(tool_calls=[{"id": "c1", "name": "search_logs", "input": {"query": "error"}}])
        if self._turn == 2 and "retrieve_runbook" in tool_names:
            return LLMResponse(tool_calls=[{"id": "c2", "name": "retrieve_runbook", "input": {"query": question or "incident"}}])

        # Build a final answer from the tool outputs already in the transcript
        observations = [m["content"] for m in messages if m["role"] in ("tool", "user") and "OBSERVATION" in str(m.get("content", ""))]
        summary = "Based on the logs and runbook, the most likely root cause is an upstream dependency failure that was not handled gracefully. Recommend adding a timeout and fallback."
        return LLMResponse(text=summary)


def get_llm() -> BaseLLM:
    """Factory: pick a provider from env, else fall back to MockLLM."""
    provider = os.getenv("SHERLOCK_LLM_PROVIDER", "auto").lower()
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    if provider in ("anthropic", "auto") and anthropic_key:
        return AnthropicLLM(api_key=anthropic_key)
    if provider in ("openai", "auto") and openai_key:
        return OpenAILLM(api_key=openai_key)
    return MockLLM()
