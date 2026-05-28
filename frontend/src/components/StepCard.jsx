import React, { useState } from "react";

const TOOL_LABELS = {
  search_logs: "🔍 Searched logs",
  retrieve_runbook: "📖 Retrieved runbook",
  get_log_stats: "📊 Analyzed log stats",
};

export default function StepCard({ step }) {
  const [open, setOpen] = useState(false);
  const hasTools = step.tool_calls && step.tool_calls.length > 0;

  return (
    <div className="step-card">
      <div className="step-header">
        <span className="step-badge">Step {step.step}</span>
        {hasTools ? (
          <span className="step-summary">
            {step.tool_calls.map((tc) => TOOL_LABELS[tc.tool] || tc.tool).join(" · ")}
          </span>
        ) : (
          <span className="step-summary final">✓ Reached conclusion</span>
        )}
      </div>

      {step.thought && hasTools && <p className="step-thought">{step.thought}</p>}

      {hasTools && (
        <>
          <button className="toggle-btn" onClick={() => setOpen(!open)}>
            {open ? "Hide" : "Show"} tool output
          </button>
          {open && (
            <div className="tool-output">
              {step.tool_calls.map((tc, i) => (
                <div key={i} className="tool-block">
                  <div className="tool-name">
                    {tc.tool}
                    <span className="tool-input">({JSON.stringify(tc.input)})</span>
                  </div>
                  <pre className="tool-result">{tc.output}</pre>
                </div>
              ))}
            </div>
          )}
        </>
      )}
    </div>
  );
}
