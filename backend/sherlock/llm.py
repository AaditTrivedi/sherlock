"""
LLM client abstraction for Sherlock.

Supports the Anthropic Messages API and any OpenAI-compatible Chat Completions
API (OpenAI, Groq, Google Gemini), all with tool-calling. Falls back to a
deterministic MockLLM when no API key is present, so the agent is fully runnable
and testable without credentials.

Provider selection (first match wins), via environment variables:
  GROQ_API_KEY       -> Groq (free, fast; default model llama-3.3-70b-versatile)
  GEMINI_API_KEY     -> Google Gemini (free tier; default model gemini-2.0-flash)
  ANTHROPIC_API_KEY  -> Anthropic Claude
  OPENAI_API_KEY     -> OpenAI
  (none)             -> MockLLM
"""

import os
import json
from abc import ABC, abstractmethod
from typing import Optional


class LLMResponse:
    """Normalized LLM response across providers."""

    def __init__(self, text: Optional[str] = None, tool_calls: Optional[list[dict]] = None):
        self.text = text
        self.tool_calls = tool_calls or []

    @property
    def wants_tools(self) -> bool:
        return len(self.tool_calls) > 0


class BaseLLM(ABC):
    provider: str = "base"
    mode: str = "mock"

    @abstractmethod
    def chat(self, messages: list[dict], tools: list[dict], system: str) -> LLMResponse:
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
        anthropic_tools = [
            {"name": t["name"], "description": t["description"], "input_schema": t["parameters"]}
            for t in tools
        ]
        kwargs = {"model": self.model, "max_tokens": 1024, "system": system, "messages": messages}
        if anthropic_tools:
            kwargs["tools"] = anthropic_tools
        resp = self.client.messages.create(**kwargs)
        text_parts, tool_calls = [], []
        for block in resp.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append({"id": block.id, "name": block.name, "input": block.input})
        return LLMResponse(text="\n".join(text_parts) or None, tool_calls=tool_calls)


class OpenAICompatibleLLM(BaseLLM):
    """
    Client for any OpenAI-compatible Chat Completions API with tool-calling.
    Works with OpenAI, Groq, and Google Gemini by varying base_url and model.
    """

    mode = "live"

    def __init__(self, api_key: str, model: str, provider: str = "openai", base_url: Optional[str] = None):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
        self.model = model
        self.provider = provider

    def _to_openai_messages(self, messages: list[dict]) -> list[dict]:
        """
        Convert the agent's provider-neutral transcript (which uses Anthropic-style
        content blocks) into OpenAI chat format with tool_calls / tool messages.
        """
        out: list[dict] = []
        for m in messages:
            role, content = m["role"], m["content"]
            if isinstance(content, str):
                out.append({"role": role, "content": content})
                continue
            if role == "assistant":
                text_parts, tool_calls = [], []
                for block in content:
                    if block.get("type") == "text":
                        text_parts.append(block["text"])
                    elif block.get("type") == "tool_use":
                        tool_calls.append({
                            "id": block["id"],
                            "type": "function",
                            "function": {"name": block["name"], "arguments": json.dumps(block["input"])},
                        })
                msg = {"role": "assistant", "content": "\n".join(text_parts) or None}
                if tool_calls:
                    msg["tool_calls"] = tool_calls
                out.append(msg)
            elif role == "user":
                pending_text = []
                for block in content:
                    if block.get("type") == "tool_result":
                        out.append({
                            "role": "tool",
                            "tool_call_id": block["tool_use_id"],
                            "content": block["content"],
                        })
                    elif block.get("type") == "text":
                        pending_text.append(block["text"])
                if pending_text:
                    out.append({"role": "user", "content": "\n".join(pending_text)})
        return out

    def chat(self, messages: list[dict], tools: list[dict], system: str) -> LLMResponse:
        openai_tools = [
            {"type": "function", "function": {
                "name": t["name"], "description": t["description"], "parameters": t["parameters"]
            }}
            for t in tools
        ]
        full = [{"role": "system", "content": system}] + self._to_openai_messages(messages)
        kwargs = {"model": self.model, "messages": full}
        if openai_tools:
            kwargs["tools"] = openai_tools
        resp = self.client.chat.completions.create(**kwargs)
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
    """Deterministic LLM for tests and zero-key demo mode."""

    provider = "mock"
    mode = "mock"

    def __init__(self):
        self._turn = 0

    def chat(self, messages: list[dict], tools: list[dict], system: str) -> LLMResponse:
        tool_names = {t["name"] for t in tools}
        question = ""
        for m in messages:
            if m["role"] == "user" and isinstance(m["content"], str):
                question = m["content"]

        self._turn += 1
        if self._turn == 1 and "search_logs" in tool_names:
            return LLMResponse(tool_calls=[{"id": "c1", "name": "search_logs", "input": {"query": "error"}}])
        if self._turn == 2 and "retrieve_runbook" in tool_names:
            return LLMResponse(tool_calls=[{"id": "c2", "name": "retrieve_runbook", "input": {"query": question or "incident"}}])

        summary = "Based on the logs and runbook, the most likely root cause is an upstream dependency failure that was not handled gracefully. Recommend adding a timeout and fallback."
        return LLMResponse(text=summary)


# Default models per provider (override with SHERLOCK_LLM_MODEL)
_GROQ_DEFAULT = "llama-3.3-70b-versatile"
_GEMINI_DEFAULT = "gemini-2.0-flash"
_OPENAI_DEFAULT = "gpt-4o-mini"
_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
_GROQ_BASE_URL = "https://api.groq.com/openai/v1"


def get_llm() -> BaseLLM:
    """Factory: pick a provider from env (first match wins), else MockLLM."""
    model_override = os.getenv("SHERLOCK_LLM_MODEL")

    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        return OpenAICompatibleLLM(
            api_key=groq_key, model=model_override or _GROQ_DEFAULT,
            provider="groq", base_url=_GROQ_BASE_URL,
        )

    gemini_key = os.getenv("GEMINI_API_KEY")
    if gemini_key:
        return OpenAICompatibleLLM(
            api_key=gemini_key, model=model_override or _GEMINI_DEFAULT,
            provider="gemini", base_url=_GEMINI_BASE_URL,
        )

    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    if anthropic_key:
        return AnthropicLLM(api_key=anthropic_key, model=model_override or "claude-sonnet-4-20250514")

    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        return OpenAICompatibleLLM(
            api_key=openai_key, model=model_override or _OPENAI_DEFAULT, provider="openai",
        )

    return MockLLM()
