// Thin API client for the Sherlock backend.
// Uses the Vite dev proxy (/api -> http://localhost:8000).

const BASE = "/api";

export async function ingest(logs, runbooks) {
  const res = await fetch(`${BASE}/ingest`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ logs, runbooks }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || "Ingest failed");
  }
  return res.json();
}

export async function investigate(sessionId, question, maxSteps = 6) {
  const res = await fetch(`${BASE}/investigate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, question, max_steps: maxSteps }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || "Investigation failed");
  }
  return res.json();
}

export async function health() {
  const res = await fetch(`${BASE}/health`);
  return res.json();
}
