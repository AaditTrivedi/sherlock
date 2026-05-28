"""
Sherlock API — AI Incident Investigation Agent.

Endpoints:
  POST /ingest       — load logs + runbooks, returns a session_id
  POST /investigate  — run the agent on a question for a session
  GET  /health       — service + LLM/embedder status
"""

import uuid
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from sherlock.models import (
    IngestRequest, IngestResponse, InvestigateRequest, InvestigateResponse, HealthResponse,
)
from sherlock.ingest import parse_logs
from sherlock.rag import VectorStore, get_embedder, chunk_text
from sherlock.tools import SessionContext
from sherlock.agent import run_investigation
from sherlock.llm import get_llm


app = FastAPI(
    title="Sherlock",
    description="AI agent that investigates system failures and diagnoses root causes.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session store (swap for Redis/Postgres in production)
_SESSIONS: dict[str, SessionContext] = {}


@app.get("/health", response_model=HealthResponse)
def health():
    llm = get_llm()
    embedder = get_embedder()
    return HealthResponse(
        status="healthy",
        llm_provider=llm.provider,
        llm_mode=llm.mode,
        embedder_mode=embedder.mode,
    )


@app.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest):
    logs = parse_logs(req.logs)
    if not logs:
        raise HTTPException(status_code=400, detail="No parseable log entries found.")

    embedder = get_embedder()
    store = VectorStore(embedder)
    chunk_count = 0
    for doc in req.runbooks:
        chunks = chunk_text(doc)
        chunk_count += store.add(chunks)

    session_id = uuid.uuid4().hex[:12]
    _SESSIONS[session_id] = SessionContext(logs=logs, store=store)

    return IngestResponse(session_id=session_id, log_count=len(logs), runbook_chunks=chunk_count)


@app.post("/investigate", response_model=InvestigateResponse)
def investigate(req: InvestigateRequest):
    ctx = _SESSIONS.get(req.session_id)
    if ctx is None:
        raise HTTPException(status_code=404, detail="Session not found. Call /ingest first.")

    llm = get_llm()
    result = run_investigation(llm, ctx, req.question, max_steps=req.max_steps)
    result.session_id = req.session_id
    return result
