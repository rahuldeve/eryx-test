"""The eryx REPL capability, driven by a scripted model (no API key needed).

Uses a bare `eryx.Session` rather than a package factory, so these run in seconds.
"""

import asyncio
import copy
import dataclasses
import gc
import pathlib
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
from eryx_sandbox.variables import VariableStore

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
        ("\x0c\nx = 1\nx", "1\n"),  # form feed isn't a line break to ast
        ("s = 'a\u2028b'\nx = 2\nx", "2\n"),  # nor is U+2028 in a string
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


def test_result_is_an_ordinary_variable(repl: EryxRepl) -> None:
    # eryx deletes its result variable after each cell; by default that's `result`
    make_agent(repl, Script(run_cells("result = 42", "result"))).run_sync("go")
    assert repl.cells[-1].stdout == "42\n"


def test_instructions_index_cells(repl: EryxRepl) -> None:
    script = Script(run_cells("x = 1\ny = 2", "1/0"))
    make_agent(repl, script).run_sync("go")

    _, info = script.requests[-1]
    assert info.instructions is not None
    assert info.instructions.endswith(
        "Cells run so far:\n1 ok: x = 1\n2 error (ZeroDivisionError): 1/0"
    )


def test_instructions_index_no_cells() -> None:
    repl = EryxRepl(index_cells=0)
    script = Script(run_cells("x = 1", "x"))
    make_agent(repl, script).run_sync("go")

    _, info = script.requests[-1]
    assert info.instructions is not None
    assert info.instructions.endswith(
        "Cells run so far:\n(2 earlier cells, see repl_history)"
    )


def test_negative_index_cells() -> None:
    with pytest.raises(ValueError, match="index_cells"):
        EryxRepl(index_cells=-1)


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


@pytest.mark.parametrize(
    ("limits", "error"),
    [(None, None), (eryx.ResourceLimits(max_memory_bytes=64 << 20), "MemoryError")],
)
def test_resource_limits_without_factory(
    limits: eryx.ResourceLimits | None, error: str | None
) -> None:
    # eryx's default (128 MB) allows 100 MB, so a MemoryError comes from `limits`
    repl = EryxRepl(resource_limits=limits)
    make_agent(repl, Script(run_cells("x = bytearray(100 << 20)"))).run_sync("go")
    assert repl.cells[0].error == error


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


def test_cancelled_cell_finishes_before_the_next(repl: EryxRepl) -> None:
    # the cancelled cell keeps running in its thread; the next one must wait for it
    slow = (
        "import time\nt = time.time()\nwhile time.time() - t < 1: pass\nprint('slow')"
    )

    async def main() -> str:
        task = asyncio.create_task(repl.run_code(None, slow))  # type: ignore[arg-type]
        await asyncio.sleep(0.2)
        task.cancel()
        return await repl.run_code(None, "print(1)")  # type: ignore[arg-type]

    assert asyncio.run(main()) == "[cell 2]\n1\n"
    slow_cell, _ = repl.cells
    assert slow_cell.stdout == "slow\n"


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


def instructions(script: Script) -> str:
    _, info = script.requests[-1]
    assert info.instructions is not None
    return info.instructions


def test_host_variables() -> None:
    repl = EryxRepl(variables={"x": 1, "data": {"a": [1, 2]}})
    script = Script(
        run_cells(
            "host.x, host['data']",
            "[n for n in ('_eryx_setup', 'Host', 'pickle', 'json') if n in globals()]",
        )
    )
    make_agent(repl, script).run_sync("go")

    values, leaked = repl.cells
    assert values.stdout == "(1, {'a': [1, 2]})\n"
    assert leaked.stdout == "[]\n"
    assert (
        "Host variables (read-only): x (int), data (dict). Read one with `host.NAME`, "
        "e.g. `x = host.x`; each read returns a fresh copy."
    ) in instructions(script)


def test_changing_the_passed_dict_changes_nothing() -> None:
    # issue #6: the instructions read the live dict, the sandbox had a stale copy
    variables: dict[str, object] = {"x": 1}
    repl = EryxRepl(variables=variables)
    variables["x"] = [1, 2, 3]
    variables["y"] = "added later"
    script = Script(run_cells("host.x, 'y' in host"))
    make_agent(repl, script).run_sync("go")

    assert repl.cells[0].stdout == "(1, False)\n"
    assert "Host variables (read-only): x (int)." in instructions(script)


def test_no_host_variables() -> None:
    repl = EryxRepl()
    script = Script(run_cells("list(host)"))
    make_agent(repl, script).run_sync("go")

    assert repl.cells[0].stdout == "[]\n"
    assert "Host variables" not in instructions(script)


def test_model_variable_and_host_variable_are_independent() -> None:
    repl = EryxRepl(variables={"x": 1})
    make_agent(repl, Script(run_cells("x = host.x\nx += 1", "x, host.x"))).run_sync(
        "go"
    )
    assert repl.cells[-1].stdout == "(2, 1)\n"


def test_host_variables_survive_a_restart() -> None:
    repl = EryxRepl(variables={"x": 1}, execution_timeout_ms=1000)
    script = Script(run_cells("y = 2", "while True: pass", "'y' in globals(), host.x"))
    make_agent(repl, script).run_sync("go")

    _, timed_out, after = repl.cells
    assert timed_out.error is not None
    assert timed_out.error.endswith(
        "variables from earlier cells are gone; host variables are unchanged"
    )
    assert after.stdout == "(False, 1)\n"


def test_snapshot_and_host_variables_coexist() -> None:
    session = eryx.Session()
    session.execute("x = 'old'\ny = 'kept'")
    repl = EryxRepl(snapshot=session.snapshot_state(), variables={"x": "new"})
    make_agent(repl, Script(run_cells("x, y, host.x"))).run_sync("go")

    assert repl.cells[0].stdout == "('old', 'kept', 'new')\n"


def test_shared_store() -> None:
    with VariableStore({"x": 1}) as store:
        for repl in (EryxRepl(store=store), EryxRepl(store=store)):
            make_agent(repl, Script(run_cells("host.x"))).run_sync("go")
            assert repl.cells[0].stdout == "1\n"


def test_store_and_variables_together() -> None:
    with VariableStore() as store, pytest.raises(ValueError, match="not both"):
        EryxRepl(store=store, variables={"x": 1})


@pytest.mark.parametrize("private", [True, False])
def test_store_and_variables_cannot_be_reassigned(private: bool) -> None:
    # issue #6: a running session keeps the store it mounted, so the instructions
    # would list one store while `host` reads another
    variables = {"x": 1}
    repl = (
        EryxRepl(variables=variables)
        if private
        else EryxRepl(store=VariableStore(variables))
    )
    store = repl.store
    make_agent(repl, Script(run_cells("host.x"))).run_sync("go")

    with pytest.raises(AttributeError, match="store can't be changed"):
        repl.store = VariableStore({"y": 2})
    with pytest.raises(AttributeError, match="only an init argument"):
        repl.variables = {"y": 2}

    assert repl.store is store
    make_agent(repl, Script(run_cells("host.x"))).run_sync("go")
    assert repl.cells[-1].stdout == "1\n"


def test_copies_keep_the_store() -> None:
    repl = EryxRepl(variables={"x": 1})
    assert copy.copy(repl).store is repl.store
    assert dataclasses.replace(repl).store is repl.store


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


def test_variable_the_sandbox_cannot_load() -> None:
    # unpickled when a cell reads it, so the model sees the error in that cell
    repl = EryxRepl(variables={"obj": HostOnly(), "x": 1})
    make_agent(repl, Script(run_cells("host.obj", "host.x"))).run_sync("go")

    failed, ok = repl.cells
    assert failed.error == "ModuleNotFoundError"
    assert ok.stdout == "1\n"


def test_session_that_cannot_start() -> None:
    repl = EryxRepl(snapshot=b"garbage")
    with pytest.raises(RuntimeError, match="starting the EryxRepl session failed"):
        make_agent(repl, Script(run_cells("1"))).run_sync("go")
    assert repl.cells == []


@pytest.mark.parametrize(
    ("damage", "message"),
    [
        ("close", "starting the EryxRepl session failed: Failed to mount"),
        ("remove host.py", "setting up EryxRepl host variables failed"),
    ],
)
def test_broken_store(damage: str, message: str) -> None:
    store = VariableStore({"x": 1})
    if damage == "close":
        store.close()
    else:
        pathlib.Path(store.volume[0], "host.py").unlink()
    repl = EryxRepl(store=store)
    with pytest.raises(RuntimeError, match=message):
        make_agent(repl, Script(run_cells("1"))).run_sync("go")
    assert repl.cells == []


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
