"""
Data models for Sherlock — AI Incident Investigation Agent.
"""

from pydantic import BaseModel, Field
from typing import Optional, Literal
from datetime import datetime, timezone


class LogEntry(BaseModel):
    """A single parsed log line."""
    timestamp: Optional[str] = None
    level: str = "INFO"  # INFO, WARN, ERROR, DEBUG, CRITICAL
    service: Optional[str] = None
    message: str


class IngestRequest(BaseModel):
    """Request to ingest logs and optional runbook documents."""
    logs: str = Field(..., description="Raw log text, one entry per line")
    runbooks: list[str] = Field(default_factory=list, description="Runbook/doc texts for RAG context")


class IngestResponse(BaseModel):
    session_id: str
    log_count: int
    runbook_chunks: int


class InvestigateRequest(BaseModel):
    """Request to investigate an incident."""
    session_id: str
    question: str = Field(..., description="The incident question, e.g. 'why did the API start returning 500s?'")
    max_steps: int = Field(default=6, ge=1, le=12)


class ToolCall(BaseModel):
    """A single tool invocation made by the agent."""
    tool: str
    input: dict
    output: str


class AgentStep(BaseModel):
    """One reasoning step in the agent loop."""
    step: int
    thought: Optional[str] = None
    tool_calls: list[ToolCall] = Field(default_factory=list)


class InvestigateResponse(BaseModel):
    """Final investigation result."""
    session_id: str
    question: str
    root_cause: str
    steps: list[AgentStep]
    tool_call_count: int
    generated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class HealthResponse(BaseModel):
    status: str
    llm_provider: str
    llm_mode: Literal["live", "mock"]
    embedder_mode: Literal["live", "mock"]
