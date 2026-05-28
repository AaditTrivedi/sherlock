"""
The Sherlock agent loop.

Implements an agentic investigation: the LLM is given the incident question and a
set of tools. It reasons, calls tools, observes results, and repeats until it
produces a root-cause analysis or hits the step limit. This is the "agentic AI"
core — the model drives a multi-step investigation autonomously.
"""

from sherlock.llm import BaseLLM
from sherlock.tools import TOOL_SCHEMAS, execute_tool, SessionContext
from sherlock.models import AgentStep, ToolCall, InvestigateResponse


SYSTEM_PROMPT = """You are Sherlock, an expert Site Reliability Engineer investigating a production incident.

You have tools to inspect application logs and search runbooks. Investigate methodically:
1. Start by getting an overview with get_log_stats.
2. Search the logs for errors and relevant events with search_logs.
3. Look up relevant guidance with retrieve_runbook.
4. Once you have enough evidence, stop calling tools and write your final root-cause analysis.

Your final answer must be concise and structured: state the most likely root cause, cite the
specific log evidence that supports it, and recommend a concrete fix. Do not call tools once you
are ready to give the final answer."""


def run_investigation(
    llm: BaseLLM,
    ctx: SessionContext,
    question: str,
    max_steps: int = 6,
) -> InvestigateResponse:
    """Run the agent loop and return a structured investigation result."""
    messages: list[dict] = [{"role": "user", "content": question}]
    steps: list[AgentStep] = []
    total_tool_calls = 0
    final_text = ""

    for step_num in range(1, max_steps + 1):
        response = llm.chat(messages=messages, tools=TOOL_SCHEMAS, system=SYSTEM_PROMPT)

        if not response.wants_tools:
            # Agent produced a final answer
            final_text = response.text or "No conclusion reached."
            steps.append(AgentStep(step=step_num, thought=final_text))
            break

        # Record the assistant's tool-calling turn in the transcript
        _append_assistant_tool_turn(messages, response)

        step_tool_calls: list[ToolCall] = []
        for call in response.tool_calls:
            observation = execute_tool(call["name"], call["input"], ctx)
            total_tool_calls += 1
            step_tool_calls.append(ToolCall(tool=call["name"], input=call["input"], output=observation))
            _append_tool_result(messages, call, observation)

        steps.append(AgentStep(step=step_num, thought=response.text, tool_calls=step_tool_calls))
    else:
        # Loop exhausted without a final answer: ask once more for a conclusion
        messages.append({"role": "user", "content": "Based on everything above, give your final root-cause analysis now. Do not call any more tools."})
        response = llm.chat(messages=messages, tools=[], system=SYSTEM_PROMPT)
        final_text = response.text or "Investigation inconclusive within step budget."
        steps.append(AgentStep(step=max_steps + 1, thought=final_text))

    return InvestigateResponse(
        session_id="",  # filled in by caller
        question=question,
        root_cause=final_text,
        steps=steps,
        tool_call_count=total_tool_calls,
    )


def _append_assistant_tool_turn(messages: list[dict], response) -> None:
    """Append the assistant's tool-use turn in a provider-neutral shape."""
    content = []
    if response.text:
        content.append({"type": "text", "text": response.text})
    for call in response.tool_calls:
        content.append({"type": "tool_use", "id": call["id"], "name": call["name"], "input": call["input"]})
    messages.append({"role": "assistant", "content": content})


def _append_tool_result(messages: list[dict], call: dict, observation: str) -> None:
    """Append a tool result the next LLM turn can read."""
    messages.append({
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": call["id"], "content": observation}],
    })
