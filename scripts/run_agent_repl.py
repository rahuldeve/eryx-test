"""Let a pydantic-ai agent analyse a host DataFrame through the eryx REPL capability.

The weather frame from `run_dataframe_bokeh` is passed in as a preloaded variable,
so the agent's first cell can use `df` directly.
Two questions are asked as two `agent.run()` calls sharing one conversation and
one eryx session. `ClearToolResults` is set to clear aggressively so that, by the
second question, early cell outputs are gone from the conversation; the model can
still find them through the cell index in the instructions and `repl_history`.
Instruction changes are appended (`on_change="append"`), so the prompt cache
survives each new cell.

Needs ANTHROPIC_API_KEY (or set AGENT_MODEL to another pydantic-ai model).
"""

import os

from pydantic_ai import Agent, capture_run_messages
from pydantic_ai.messages import ModelRequest, ToolReturnPart
from pydantic_ai_harness.compaction import ClearToolResults
from run_dataframe_bokeh import make_weather
from run_session_bokeh import LIMITS

from eryx_sandbox.factory import BOKEH_PACKAGES, load_factory
from eryx_sandbox.pydantic_capability import EryxRepl

MODEL = os.environ.get("AGENT_MODEL", "anthropic:claude-sonnet-5-5")
CLEARED = "[tool result cleared]"  # ClearToolResults' default placeholder

QUESTIONS = [
    (
        "`df` holds daily weather for three cities. Which city had the most rain, "
        "and on how many days did it rain there?"
    ),
    "For that city, what was the warmest week (7-day rolling mean of temp)?",
]


def main() -> None:
    factory = load_factory(
        "bokeh", BOKEH_PACKAGES, imports=["numpy", "pandas", "bokeh.plotting"]
    )
    repl = EryxRepl(
        factory,
        variables={"df": make_weather()},
        resource_limits=LIMITS,
        on_change="append",
    )
    agent = Agent(
        MODEL,
        instructions="Answer questions about the data by running code in the REPL.",
        capabilities=[ClearToolResults(max_messages=6, keep_pairs=2), repl],
    )

    history = None
    for question in QUESTIONS:
        print(f"\n### {question}")
        with capture_run_messages() as messages:
            result = agent.run_sync(question, message_history=history)
        history = result.all_messages()
        print(result.output)
        cleared = [
            part.tool_call_id
            for message in messages
            if isinstance(message, ModelRequest)
            for part in message.parts
            if isinstance(part, ToolReturnPart) and part.content == CLEARED
        ]
        print(f"[{len(cleared)} REPL results cleared so far]")
        usage = result.usage
        print(
            f"[tokens: {usage.input_tokens} input, "
            f"{usage.cache_read_tokens} read from cache]"
        )

    print("\n### Cells run")
    for cell in repl.cells:
        print(f"--- cell {cell.number} ({cell.duration_ms:.0f}ms)")
        print(cell.code.rstrip())
        print(">>>", cell.output().rstrip() or "(no output)")


if __name__ == "__main__":
    main()
