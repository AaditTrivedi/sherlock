# Sherlock

**An AI agent that investigates system failures and diagnoses root causes.**

Feed Sherlock your application logs and an incident question — *"why did checkout start failing at 2am?"* — and an LLM-powered agent **autonomously investigates**. It calls tools to analyze log statistics, search for errors, and retrieve relevant runbooks via semantic search (RAG), reasoning step by step until it produces a root-cause analysis with supporting evidence and a recommended fix.

This isn't a chatbot that answers in one shot. It's an **agent** that drives a multi-step investigation on its own, deciding which tools to call and when.

![status](https://img.shields.io/badge/tests-38%20passing-brightgreen) ![python](https://img.shields.io/badge/python-3.12-blue) ![react](https://img.shields.io/badge/react-18-blue) ![license](https://img.shields.io/badge/license-MIT-green)

## Why I built this

This is the third project in a trilogy about building and operating reliable systems:

1. **[URL Shortener](https://github.com/AaditTrivedi/url-shortener)** — I build backend systems (FastAPI, PostgreSQL, Redis, circuit breaker).
2. **[Dock Chaos](https://github.com/AaditTrivedi/dock-chaos)** — I build tools to break them (chaos engineering that injects failures and finds resilience bugs).
3. **Sherlock** — I build AI agents to diagnose what broke.

The three connect: Dock Chaos generates exactly the kind of failure data Sherlock investigates. Break a system on purpose, capture the logs, then let an AI agent figure out the root cause. That's the full reliability loop.

## What makes it "agentic"

Most LLM apps do single-turn retrieval: stuff context into a prompt, get one answer. Sherlock runs an **agent loop** — the model is given tools and decides, on each turn, what to do next:

```
User: "Why did checkout fail at 2am?"

  → Agent calls get_log_stats()      → "47 entries, 4 ERROR, 1 CRITICAL, service=checkout"
  → Agent calls search_logs("error") → "redis connection timeout after 30s..."
  → Agent calls retrieve_runbook(    → "[RAG] Redis failures need a circuit breaker
        "redis timeout")                 with a 1s timeout; a 30s hang means it's missing"
  → Agent reasons and concludes:
        "Root cause: Redis became unavailable and the checkout service lacked a
         circuit breaker, so it hung on the default 30s timeout, cascading into
         queue overflow. Fix: add a 1s timeout + circuit breaker with PG fallback."
```

The agent chose that sequence itself. With a different incident, it investigates differently.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│            React Frontend (Vite)                         │
│   Log input · runbook input · live investigation view    │
└──────────────────────────┬──────────────────────────────┘
                          │ REST
┌──────────────────────────▼──────────────────────────────┐
│                  FastAPI Backend                         │
│   /ingest   →  parse logs + embed runbooks               │
│   /investigate → run the agent loop                      │
└──────────────────────────┬──────────────────────────────┘
                          │
              ┌───────────▼───────────┐
              │      Agent Loop       │
              │  (LLM tool-calling)   │
              └───────────┬───────────┘
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
  ┌─────▼─────┐    ┌──────▼──────┐   ┌──────▼──────┐
  │ search_   │    │ retrieve_   │   │ get_log_    │
  │ logs      │    │ runbook(RAG)│   │ stats       │
  └───────────┘    └──────┬──────┘   └─────────────┘
                          │
                  ┌───────▼────────┐
                  │  Vector Store  │
                  │ (embeddings +  │
                  │  cosine search)│
                  └────────────────┘
                          │
              ┌───────────▼───────────┐
              │   LLM Provider        │
              │ Anthropic / OpenAI    │
              │ (or MockLLM offline)  │
              └───────────────────────┘
```

## Tech stack

- **Backend:** Python, FastAPI, Pydantic
- **AI:** LLM tool-calling (Anthropic Claude / OpenAI), agent loop, RAG with vector embeddings and cosine-similarity search
- **Frontend:** React 18, Vite
- **Testing:** pytest (38 tests), GitHub Actions CI
- **Design:** provider-agnostic LLM and embedder interfaces; runs fully offline with deterministic mocks (no API key needed to demo)

## Quick start

### Backend

```bash
cd backend
pip install -r requirements.txt

# Optional: use a real LLM (otherwise runs in deterministic mock mode).
# Free options — no credit card required:
export GROQ_API_KEY=gsk_...          # free + fast, runs Llama 3.3 70B  (recommended)
# export GEMINI_API_KEY=...          # free tier, runs Gemini 2.0 Flash
# Or paid providers:
# export ANTHROPIC_API_KEY=sk-...    # Claude
# export OPENAI_API_KEY=sk-...       # GPT

uvicorn sherlock.main:app --reload
```

Sherlock auto-detects which provider to use from whichever key is set (Groq → Gemini → Anthropic → OpenAI → offline mock). Override the model with `SHERLOCK_LLM_MODEL` if you want a different one.

### Frontend

```bash
cd frontend
npm install
npm run dev
# open http://localhost:5173
```

The UI ships with a sample incident pre-loaded — click **Investigate** to watch the agent work.

### Run without any API key

Sherlock is built to run offline. With no key set, it uses a deterministic `MockLLM` and a hashing-based embedder, so the entire agent loop, RAG retrieval, and UI work end-to-end for demos and CI. Add a real key to get genuine LLM reasoning.

## API

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/ingest` | Load logs + runbooks, returns a `session_id` |
| POST | `/investigate` | Run the agent on a question for a session |
| GET | `/health` | Service + LLM/embedder status |

Example:

```bash
# 1. Ingest
curl -X POST localhost:8000/ingest -H "Content-Type: application/json" -d '{
  "logs": "2026-01-01T02:14:32 ERROR checkout: redis connection timeout after 30s",
  "runbooks": ["Redis failures need a circuit breaker with a 1s timeout."]
}'
# → {"session_id": "abc123...", "log_count": 1, "runbook_chunks": 1}

# 2. Investigate
curl -X POST localhost:8000/investigate -H "Content-Type: application/json" -d '{
  "session_id": "abc123...",
  "question": "Why did checkout fail?"
}'
```

## Project structure

```
sherlock/
├── backend/
│   ├── sherlock/
│   │   ├── main.py        # FastAPI endpoints
│   │   ├── agent.py       # Agent loop (LLM tool-calling orchestration)
│   │   ├── llm.py         # Provider abstraction: Anthropic / OpenAI / Mock
│   │   ├── rag.py         # Embeddings + vector store + semantic search
│   │   ├── tools.py       # Agent tools: search_logs, retrieve_runbook, get_log_stats
│   │   ├── ingest.py      # Log parsing + statistics
│   │   └── models.py      # Pydantic schemas
│   └── tests/             # 38 tests (agent, RAG, tools, parsing)
├── frontend/
│   └── src/
│       ├── App.jsx        # Main UI
│       ├── api.js         # Backend client
│       └── components/
│           └── StepCard.jsx   # Renders each agent step + tool output
├── .github/workflows/ci.yml
└── README.md
```

## Design decisions

- **Agent over single-shot RAG.** Real diagnosis is iterative. An agent that picks its own tools and follows the evidence mirrors how an SRE actually debugs, and surfaces reasoning you can audit step by step.
- **Provider-agnostic.** The `BaseLLM` and `BaseEmbedder` interfaces mean Anthropic, OpenAI, or a local model are drop-in. No vendor lock-in.
- **Runs offline.** Deterministic mocks make the whole system testable in CI and demoable without spending a cent on API calls — and keep tests fast and reproducible.
- **Pluggable vector store.** The in-memory cosine-similarity store is intentionally simple; swap in FAISS, Chroma, or pgvector for scale without touching the agent.
- **Tools return auditable observations.** Every tool call and its output is captured and shown in the UI, so the investigation is transparent, not a black box.

## Testing

```bash
cd backend && python -m pytest tests/ -v
```

38 tests cover log parsing, RAG retrieval, every agent tool, the LLM abstraction, and full agent-loop investigations (with a mocked LLM so they run anywhere).

## License

MIT
