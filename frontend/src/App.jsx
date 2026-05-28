import React, { useState, useEffect } from "react";
import { ingest, investigate, health } from "./api";
import StepCard from "./components/StepCard";

const SAMPLE_LOGS = `2026-01-01T02:00:00 INFO checkout: service started
2026-01-01T02:14:32 ERROR checkout: redis connection timeout after 30s
2026-01-01T02:14:33 ERROR checkout: payment validation failed - cache unavailable
2026-01-01T02:14:35 WARN checkout: falling back to direct DB reads
2026-01-01T02:15:00 CRITICAL checkout: request queue overflow - 5000 pending`;

const SAMPLE_RUNBOOK = `If Redis becomes unavailable, the checkout service should use a circuit breaker with a 1-second timeout and fall back to PostgreSQL. A 30-second hang indicates the circuit breaker is missing or misconfigured. Queue overflow downstream is a symptom of the upstream Redis stall.`;

export default function App() {
  const [logs, setLogs] = useState(SAMPLE_LOGS);
  const [runbook, setRunbook] = useState(SAMPLE_RUNBOOK);
  const [question, setQuestion] = useState("Why did checkout start failing at 2am?");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [status, setStatus] = useState(null);

  useEffect(() => {
    health().then(setStatus).catch(() => {});
  }, []);

  async function runInvestigation() {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const { session_id } = await ingest(logs, runbook ? [runbook] : []);
      const res = await investigate(session_id, question);
      setResult(res);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="app">
      <header className="header">
        <div className="logo">
          <span className="logo-mark">🕵️</span>
          <div>
            <h1>Sherlock</h1>
            <p className="tagline">AI agent that investigates system failures</p>
          </div>
        </div>
        {status && (
          <div className="status-pill" title="LLM provider status">
            <span className={`dot ${status.llm_mode === "live" ? "live" : "mock"}`} />
            {status.llm_provider} · {status.llm_mode}
          </div>
        )}
      </header>

      <div className="layout">
        <section className="panel input-panel">
          <h2>Incident Data</h2>

          <label>Application Logs</label>
          <textarea
            className="mono"
            value={logs}
            onChange={(e) => setLogs(e.target.value)}
            rows={10}
            placeholder="Paste raw logs, one entry per line..."
          />

          <label>Runbook / Docs (optional, used for RAG)</label>
          <textarea
            value={runbook}
            onChange={(e) => setRunbook(e.target.value)}
            rows={4}
            placeholder="Paste runbook or documentation..."
          />

          <label>Incident Question</label>
          <input
            type="text"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="e.g. why did the API start returning 500s?"
          />

          <button className="investigate-btn" onClick={runInvestigation} disabled={loading}>
            {loading ? "Investigating…" : "🔎 Investigate"}
          </button>

          {error && <div className="error-box">{error}</div>}
        </section>

        <section className="panel results-panel">
          <h2>Investigation</h2>

          {!result && !loading && (
            <div className="empty">
              <p>The agent will investigate step by step:</p>
              <ol>
                <li>Analyze log statistics</li>
                <li>Search logs for errors</li>
                <li>Retrieve relevant runbooks (RAG)</li>
                <li>Reason to a root cause</li>
              </ol>
            </div>
          )}

          {loading && <div className="loading">🕵️ Sherlock is investigating…</div>}

          {result && (
            <>
              <div className="steps">
                {result.steps.map((s) => (
                  <StepCard key={s.step} step={s} />
                ))}
              </div>

              <div className="root-cause">
                <h3>🎯 Root Cause Analysis</h3>
                <p>{result.root_cause}</p>
                <div className="meta">
                  {result.tool_call_count} tool calls · {result.steps.length} reasoning steps
                </div>
              </div>
            </>
          )}
        </section>
      </div>
    </div>
  );
}
