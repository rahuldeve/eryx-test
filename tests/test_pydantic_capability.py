"""The eryx REPL capability, driven by a scripted model (no API key needed).

Uses a bare `eryx.Session` rather than a package factory, so these run in seconds.
"""

import gc
import pickle
import warnings
from typing import Any

import eryx
import pytest
from pydantic_ai import Agent
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai_harness.compaction import ClearToolResults

from eryx_sandbox.pydantic_capability import EryxRepl, prepare_cell

Call = tuple[str, dict[str, object]]


class Script:
    """A model that makes `calls` one per request, then answers, recording what it saw."""

    def __init__(self, calls: list[Call]) -> None:
        self.calls = list(calls)
        self.requests: list[tuple[list[ModelMessage], AgentInfo]] = []

    def __call__(self, messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        self.requests.append((messages, info))
        if self.calls:
            name, args = self.calls.pop(0)
            return ModelResponse(parts=[ToolCallPart(name, args)])
        return ModelResponse(parts=[TextPart("done")])

    @property
    def returns(self) -> dict[str, str]:
        """The last request's tool results, by tool call id."""
        messages, _ = self.requests[-1]
        return {
            part.tool_call_id: str(part.content)
            for m in messages
            if isinstance(m, ModelRequest)
            for part in m.parts
            if isinstance(part, ToolReturnPart)
        }


def run_cells(*codes: str) -> list[Call]:
    return [("run_code", {"code": code}) for code in codes]


@pytest.fixture
def repl() -> EryxRepl:
    return EryxRepl(execution_timeout_ms=2000)


def make_agent(
    repl: EryxRepl, script: Script, *capabilities: AbstractCapability[Any]
) -> Agent:
    return Agent(FunctionModel(script), capabilities=[*capabilities, repl])


def tool_returns(messages: list[ModelMessage]) -> list[str]:
    return [
        str(part.content)
        for m in messages
        if isinstance(m, ModelRequest)
        for part in m.parts
        if isinstance(part, ToolReturnPart)
    ]


def test_output_and_errors(repl: EryxRepl) -> None:
    script = Script(
        run_cells(
            'import sys\nprint("out")\nprint("err", file=sys.stderr)',
            'print("before")\n1/0',
            "pass",
        )
    )
    result = make_agent(repl, script).run_sync("go")

    ok, raised, empty = tool_returns(result.all_messages())
    assert ok == "[cell 1]\nout\nerr\n"
    # stdout printed before the exception is kept, eryx's wrapper frame is not
    assert raised.startswith("[cell 2]\nbefore\nTraceback (most recent call last):\n")
    assert '  File "<user>", line 2, in <module>\n' in raised
    assert raised.endswith("ZeroDivisionError: division by zero\n")
    assert "_eryx_exec" not in raised
    assert empty == "[cell 3]\n(no output)\n"
    assert [c.error for c in repl.cells] == [None, "ZeroDivisionError", None]


@pytest.mark.parametrize(
    ("code", "printed"),
    [
        ("x = 2\nx * 21", "42\n"),
        ("{\n    'a': 1,\n}", "{'a': 1}\n"),  # multi-line last expression
        ("'text'", "'text'\n"),
        ("None", ""),
        ("x = 1", ""),
        ("print('p')", "p\n"),  # its None result isn't echoed
        ("x = 1; x", "1\n"),  # last expression shares a line
        ("s = 'é'; s  # comment", "'é'\n"),  # col offsets are in bytes
    ],
)
def test_last_expression_is_echoed(repl: EryxRepl, code: str, printed: str) -> None:
    make_agent(repl, Script(run_cells(code))).run_sync("go")
    assert repl.cells[0].stdout == printed
    assert repl.cells[0].code == code


def test_echo_keeps_line_numbers() -> None:
    code = "a = 1\nb = (\n    a +\n    undefined\n)\nb"
    assert prepare_cell(code).splitlines()[:5] == code.splitlines()[:5]


def test_state_persists_across_calls_and_runs(repl: EryxRepl) -> None:
    script = Script(run_cells("x = 40", "x += 1"))
    agent = make_agent(repl, script)
    first = agent.run_sync("go")

    script.calls = run_cells("x + 1")
    agent.run_sync("again", message_history=first.all_messages())
    assert repl.cells[-1].number == 3
    assert repl.cells[-1].stdout == "42\n"


def test_instructions_index_cells(repl: EryxRepl) -> None:
    script = Script(run_cells("x = 1\ny = 2", "1/0"))
    make_agent(repl, script).run_sync("go")

    _, info = script.requests[-1]
    assert info.instructions is not None
    assert info.instructions.endswith(
        "Cells run so far:\n1 ok: x = 1\n2 error (ZeroDivisionError): 1/0"
    )


def test_long_output_is_clipped_but_kept(repl: EryxRepl) -> None:
    repl.max_output_chars = 100
    script = Script(
        [
            *run_cells("print('x' * 500)"),
            ("repl_history", {"cell": 1, "output": True}),
        ]
    )
    result = make_agent(repl, script).run_sync("go")

    clipped, history = tool_returns(result.all_messages())
    assert clipped.endswith(
        "... 401 more characters, see repl_history(cell=1, output=True)\n"
    )
    assert "x" * 500 + "\n" in history


def test_history(repl: EryxRepl) -> None:
    script = Script(
        [
            *run_cells("print(1)", "1/0", "print(3)"),
            ("repl_history", {"cell": 2}),
            ("repl_history", {"cell": 2, "output": True}),
            ("repl_history", {"last": 2}),
            ("repl_history", {"last": 2, "output": True}),
            ("repl_history", {"cell": 9}),
        ]
    )
    result = make_agent(repl, script).run_sync("go")

    code, with_output, last_two, last_two_output, missing = tool_returns(
        result.all_messages()
    )[3:]
    assert code == "[cell 2, ZeroDivisionError]\n>>> code\n1/0\n"
    assert with_output.startswith(
        "[cell 2, ZeroDivisionError]\n>>> code\n1/0\n>>> output\n"
    )
    assert "ZeroDivisionError: division by zero" in with_output
    assert last_two == (
        "[cell 2, ZeroDivisionError]\n>>> code\n1/0\n\n\n[cell 3]\n>>> code\nprint(3)\n"
    )
    assert last_two_output.endswith("\n\n[cell 3]\n>>> code\nprint(3)\n>>> output\n3\n")
    assert missing == "No cell 9: cells run so far are 1-3."


@pytest.mark.parametrize("clear_first", [True, False])
def test_clearing_in_any_order(repl: EryxRepl, clear_first: bool) -> None:
    script = Script(
        [
            *run_cells("print('one')", "print('two')", "print('three')"),
            ("repl_history", {"cell": 1, "output": True}),
        ]
    )
    clear = ClearToolResults(max_messages=2, keep_pairs=1)
    capabilities = [clear, repl] if clear_first else [repl, clear]
    Agent(FunctionModel(script), capabilities=capabilities).run_sync("go")

    seen = list(script.returns.values())
    assert seen[:3] == ["[tool result cleared]"] * 3
    assert seen[-1] == "[cell 1]\n>>> code\nprint('one')\n>>> output\none\n"
    _, info = script.requests[-1]
    assert info.instructions is not None
    assert "Cells run so far:\n1 ok: print('one')" in info.instructions


def test_append_instructions() -> None:
    repl = EryxRepl(on_change="append")
    script = Script(run_cells("x = 1", "y = 2"))
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message="Instruction block")
        make_agent(repl, script).run_sync("go")

    messages, info = script.requests[-1]
    # the prompt prefix keeps the index as it was when the run started
    assert info.instructions is not None
    assert "Cells run so far" not in info.instructions
    updates = [
        str(part.content)
        for m in messages
        if isinstance(m, ModelRequest)
        for part in m.parts
        if isinstance(part, UserPromptPart) and "eryx_repl" in str(part.content)
    ]
    assert "1 ok: x = 1" in updates[0]
    assert "<context_diff" in updates[-1] and "+2 ok: y = 2" in updates[-1]


def test_default_id() -> None:
    assert EryxRepl().id == "eryx_repl"
    assert EryxRepl(id="analysis").id == "analysis"


def test_timeout_restarts_session(repl: EryxRepl) -> None:
    script = Script(run_cells("x = 1", "while True: pass", "x", "y = 2\ny"))
    make_agent(repl, script).run_sync("go")

    _, timed_out, lost, fresh = repl.cells
    assert timed_out.error == (
        "timed out after 2000 ms; the sandbox was restarted "
        "and variables from earlier cells are gone"
    )
    assert lost.error == "NameError"
    assert fresh.stdout == "2\n"


def test_snapshot_is_restored_on_restart() -> None:
    session = eryx.Session()
    session.execute("preloaded = 'yes'")
    repl = EryxRepl(snapshot=session.snapshot_state(), execution_timeout_ms=1000)
    script = Script(run_cells("preloaded", "while True: pass", "preloaded"))
    make_agent(repl, script).run_sync("go")

    first, timed_out, after = repl.cells
    assert first.stdout == after.stdout == "'yes'\n"
    assert timed_out.error is not None
    assert timed_out.error.endswith("(the preloaded state was restored)")


class HostOnly:
    """A class the sandbox can't import, so it can't unpickle instances of it."""


def test_variables_are_preloaded() -> None:
    repl = EryxRepl(variables={"x": 1, "data": {"a": [1, 2]}})
    script = Script(
        run_cells("x, data", "[n for n in ('_eryx_load', 'pickle') if n in globals()]")
    )
    make_agent(repl, script).run_sync("go")

    values, leaked = repl.cells
    assert values.stdout == "(1, {'a': [1, 2]})\n"
    assert leaked.stdout == "[]\n"
    _, info = script.requests[-1]
    assert info.instructions is not None
    assert "Preloaded variables: x (int), data (dict)" in info.instructions


def test_variables_are_restored_on_restart() -> None:
    repl = EryxRepl(variables={"x": 1}, execution_timeout_ms=1000)
    script = Script(run_cells("x = 2", "while True: pass", "x"))
    make_agent(repl, script).run_sync("go")

    _, timed_out, after = repl.cells
    assert timed_out.error is not None
    assert timed_out.error.endswith("(the preloaded state was restored)")
    assert after.stdout == "1\n"


def test_variables_override_snapshot() -> None:
    session = eryx.Session()
    session.execute("x = 'old'\ny = 'kept'")
    repl = EryxRepl(snapshot=session.snapshot_state(), variables={"x": "new"})
    make_agent(repl, Script(run_cells("x, y"))).run_sync("go")

    assert repl.cells[0].stdout == "('new', 'kept')\n"


@pytest.mark.parametrize(
    ("variables", "error"),
    [
        ({"not-a-name": 1}, ValueError),
        ({"class": 1}, ValueError),
        ({"f": lambda: 0}, (pickle.PicklingError, AttributeError)),
    ],
)
def test_bad_variables_fail_at_construction(
    variables: dict[str, object], error: type[Exception]
) -> None:
    with pytest.raises(error):
        EryxRepl(variables=variables)


def test_variables_the_sandbox_cannot_load() -> None:
    repl = EryxRepl(variables={"obj": HostOnly()})
    with pytest.raises(RuntimeError, match="loading EryxRepl variables failed"):
        make_agent(repl, Script(run_cells("obj"))).run_sync("go")


def test_reset(repl: EryxRepl) -> None:
    make_agent(repl, Script(run_cells("x = 1"))).run_sync("go")
    repl.reset()
    make_agent(repl, Script(run_cells("x"))).run_sync("go")
    assert [c.number for c in repl.cells] == [1]
    assert repl.cells[0].error == "NameError"


def test_parallel_calls_run_in_order(repl: EryxRepl) -> None:
    calls = [
        *run_cells("x = 1", "x += 1", "x"),
        ("repl_history", {"last": 3, "output": True}),
    ]
    responses = [
        ModelResponse(parts=[ToolCallPart(name, args) for name, args in calls]),
        ModelResponse(parts=[TextPart("done")]),
    ]

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return responses.pop(0)

    result = Agent(FunctionModel(model), capabilities=[repl]).run_sync("go")

    assert [(c.number, c.code) for c in repl.cells] == [
        (1, "x = 1"),
        (2, "x += 1"),
        (3, "x"),
    ]
    assert repl.cells[-1].stdout == "2\n"
    assert (
        "[cell 3]\n>>> code\nx\n>>> output\n2\n"
        in tool_returns(result.all_messages())[-1]
    )


def test_bound_to_one_agent(repl: EryxRepl) -> None:
    script = Script([])
    agent = make_agent(repl, script)
    with pytest.raises(ValueError, match="already used by agent"):
        make_agent(repl, script)

    # binding again to the same agent is fine
    agent.run_sync("go")
    with agent.override(model=FunctionModel(script)):
        agent.run_sync("go")
    # a run capability is bound to the run's agent
    other = Agent(FunctionModel(script))
    with pytest.raises(ValueError, match="already used by agent"):
        other.run_sync("go", capabilities=[repl])


def test_rebinds_after_agent_is_gone(repl: EryxRepl) -> None:
    make_agent(repl, Script([]))
    gc.collect()
    make_agent(repl, Script(run_cells("1 + 1"))).run_sync("go")
    assert repl.cells[0].stdout == "2\n"
