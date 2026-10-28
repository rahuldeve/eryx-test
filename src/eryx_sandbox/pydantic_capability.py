"""A persistent eryx session as a pydantic-ai capability: a sandboxed Python REPL.

`EryxRepl` gives the model two tools:

- `run_code` runs a cell in one eryx session that lives as long as the capability,
  so variables carry over between calls and across `agent.run()` calls. It returns
  the cell's stdout and stderr (a traceback if the code raised), and echoes the
  value of a bare last expression, as a REPL would.
- `repl_history` returns the code and full output of earlier cells.

Every cell is kept on the host, and the instructions carry a short index of them,
so the model can see what it ran even after compaction blanks old tool results.
For pydantic-ai-harness's `ClearToolResults`, pass `CLEARED_MARKER` as its
`placeholder` and list it *before* `EryxRepl`: just before each model request
`EryxRepl` replaces the marker on its own tools' results with a pointer to the
cell, e.g. `[output of cell 3 cleared: call repl_history(cell=3)]`. The pointer
only goes into the request; the stored history keeps the marker, which is how
`ClearToolResults` recognises results it has already cleared.

    repl = EryxRepl(factory, snapshot=snapshot)
    agent = Agent(
        "anthropic:claude-sonnet-5-5",
        capabilities=[
            ClearToolResults(max_tokens=50_000, placeholder=CLEARED_MARKER),
            repl,
        ],
    )
"""

import ast
import asyncio
import re
import time
import weakref
from collections.abc import Sequence
from dataclasses import KW_ONLY, dataclass, field, replace
from typing import Any, Self

import eryx
from pydantic_ai import FunctionToolset, RunContext
from pydantic_ai.agent import AbstractAgent
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ModelMessage, ModelRequest, ToolReturnPart
from pydantic_ai.models import ModelRequestContext

CLEARED_MARKER = "[eryx-repl: tool result cleared]"

INSTRUCTIONS = """\
You have a persistent, sandboxed Python REPL. Run code with `run_code`: \
variables, imports and functions carry over between calls. Each call returns the \
cell's stdout and stderr, and the value of a bare last expression is printed. \
Use `repl_history` to see the code and full output of earlier cells, for example \
when an old tool result has been cleared."""

# eryx runs user code from its own wrapper, which shows up as a traceback frame
_WRAPPER_FRAME = re.compile(r'  File "<string>", line \d+, in _eryx_exec\n')
_ECHO_VAR = "__eryx_repl_value"
# A session that trapped (timeout, out of memory) fails every later call with this
_WASM_TRAP = "WASM execution error"


@dataclass
class Cell:
    """One `run_code` call: the code as the model wrote it and what it printed."""

    number: int
    code: str
    stdout: str
    stderr: str
    error: str | None
    """The exception type for code that raised, or why eryx stopped it."""
    duration_ms: float

    def output(self) -> str:
        parts = [self.stdout, self.stderr]
        if self.error and self.error not in self.stderr:
            parts.append(self.error + "\n")
        return "".join(parts)

    def summary(self, width: int = 80) -> str:
        """The cell's index line: number, ok/error and its first line of code."""
        first = next((line for line in self.code.splitlines() if line.strip()), "")
        if len(first) > width:
            first = first[: width - 3] + "..."
        status = f"error ({self.error})" if self.error else "ok"
        return f"{self.number} {status}: {first}"


def prepare_cell(code: str) -> str:
    """Return `code` with a bare last expression rewritten to print its repr.

    Source lines are spliced rather than unparsed, so line numbers in tracebacks
    still match the code as written. Code that doesn't parse is returned unchanged
    so the sandbox reports the SyntaxError.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    if not tree.body or not isinstance(tree.body[-1], ast.Expr):
        return code
    last = tree.body[-1]
    lines = code.splitlines()
    start, end = last.lineno - 1, last.end_lineno or last.lineno
    # what precedes the expression on its line, e.g. `x = 1; ` (offsets are in bytes)
    head = lines[start].encode()[: last.col_offset].decode()
    expr = ast.get_source_segment(code, last)
    echo = (
        f"{head}{_ECHO_VAR} = ({expr}\n)\n"
        f"if {_ECHO_VAR} is not None:\n    print(repr({_ECHO_VAR}))\n"
        f"del {_ECHO_VAR}"
    )
    return "\n".join([*lines[:start], echo, *lines[end:]])


@dataclass
class EryxRepl(AbstractCapability[Any]):
    """Capability giving the model a persistent, sandboxed Python REPL (see module docs).

    The eryx session is created on the first `run_code` call and kept, with the
    cell history, until `reset()`. An instance belongs to the first agent it is
    given to; another agent gets a `ValueError`. It should also serve one
    conversation at a time, which isn't checked: concurrent runs of its agent
    share variables and history. Tool calls in one model response run one at a
    time, in order.
    """

    factory: eryx.SandboxFactory | None = None
    """Factory with packages pre-installed; `None` uses a bare `eryx.Session`."""

    _: KW_ONLY

    resource_limits: eryx.ResourceLimits | None = None
    """Passed to `factory.create_session` (needs `factory`)."""

    volumes: Sequence[tuple[str, str, bool]] | None = None
    """Host directories to mount, as `(host_path, sandbox_path, read_only)`."""

    snapshot: bytes | None = None
    """State from `Session.snapshot_state()` to restore when the session starts."""

    execution_timeout_ms: int | None = 60_000
    """Time limit for one cell; `None` removes it."""

    max_output_chars: int = 20_000
    """Longer cell output is clipped in the `run_code` return; history keeps it all."""

    index_cells: int = 50
    """Most recent cells listed in the instructions."""

    cleared_marker: str = CLEARED_MARKER
    """The `placeholder` given to `ClearToolResults`."""

    _session: eryx.Session | None = field(default=None, init=False, repr=False)
    _cells: list[Cell] = field(default_factory=list, init=False, repr=False)
    _call_cells: dict[str, list[int]] = field(
        default_factory=dict, init=False, repr=False
    )
    _stdout: list[bytes] = field(default_factory=list, init=False, repr=False)
    _stderr: list[bytes] = field(default_factory=list, init=False, repr=False)
    _agent: weakref.ref[AbstractAgent[Any, Any]] | None = field(
        default=None, init=False, repr=False
    )

    def __post_init__(self) -> None:
        # Each call runs alone, in the order the model made them: the session can
        # only run one cell at a time, and later cells may use earlier ones' variables
        self._toolset = FunctionToolset[Any](
            [self.run_code, self.repl_history], sequential=True
        )

    def for_agent(self, agent: AbstractAgent[Any, Any]) -> Self:
        """Bind to `agent`, refusing any other while it is alive."""
        bound = self._agent() if self._agent else None
        if bound is not None and bound is not agent:
            raise ValueError(
                f"this EryxRepl is already used by agent {bound.name or bound!r}; "
                "create one EryxRepl per agent"
            )
        self._agent = weakref.ref(agent)
        return self

    @property
    def cells(self) -> list[Cell]:
        return list(self._cells)

    def reset(self) -> None:
        """Drop the session and the cell history; the next cell starts fresh."""
        self._session = None
        self._cells.clear()
        self._call_cells.clear()

    def get_toolset(self) -> FunctionToolset[Any]:
        return self._toolset

    def get_instructions(self) -> Any:
        return self._instructions

    async def run_code(self, ctx: RunContext[Any], code: str) -> str:
        """Run Python code in the persistent REPL and return what it printed.

        Args:
            code: Python source. Variables, imports and functions persist between
                calls; a bare last expression is printed, as in a REPL.
        """
        cell = await asyncio.to_thread(self._execute, code)
        if ctx.tool_call_id:
            self._call_cells[ctx.tool_call_id] = [cell.number]
        output = cell.output() or "(no output)\n"
        if len(output) > self.max_output_chars:
            output = (
                output[: self.max_output_chars]
                + f"\n... {len(output) - self.max_output_chars} more characters, "
                f"see repl_history(cell={cell.number})\n"
            )
        return f"[cell {cell.number}]\n{output}"

    async def repl_history(
        self, ctx: RunContext[Any], cell: int | None = None, last: int = 5
    ) -> str:
        """Show the code and full output of earlier REPL cells.

        Args:
            cell: The cell number to show. Omit it to show the most recent cells.
            last: How many of the most recent cells to show when `cell` is omitted.
        """
        if cell is not None:
            if not 1 <= cell <= len(self._cells):
                return f"No cell {cell}: cells run so far are 1-{len(self._cells)}."
            shown = [self._cells[cell - 1]]
        else:
            shown = self._cells[-max(last, 1) :]
        if not shown:
            return "No cells have been run yet."
        if ctx.tool_call_id:
            self._call_cells[ctx.tool_call_id] = [c.number for c in shown]
        return "\n\n".join(
            f"[cell {c.number}{f', {c.error}' if c.error else ''}]\n"
            f">>> code\n{c.code.rstrip()}\n>>> output\n{c.output() or '(no output)'}"
            for c in shown
        )

    async def before_model_request(
        self, ctx: RunContext[Any], request_context: ModelRequestContext
    ) -> ModelRequestContext:
        """Point cleared results of this capability's tools at `repl_history`."""
        messages = [self._rewrite_cleared(m) for m in request_context.messages]
        return replace(request_context, messages=messages)

    def _instructions(self, ctx: RunContext[Any]) -> str:
        if not self._cells:
            return INSTRUCTIONS
        shown = self._cells[-self.index_cells :]
        lines = [c.summary() for c in shown]
        if earlier := len(self._cells) - len(shown):
            lines.insert(0, f"({earlier} earlier cells, see repl_history)")
        return INSTRUCTIONS + "\n\nCells run so far:\n" + "\n".join(lines)

    def _rewrite_cleared(self, message: ModelMessage) -> ModelMessage:
        if not isinstance(message, ModelRequest):
            return message
        parts = [
            replace(part, content=self._cleared_pointer(part))
            if isinstance(part, ToolReturnPart)
            and part.tool_name in self._toolset.tools
            and part.content == self.cleared_marker
            else part
            for part in message.parts
        ]
        if all(new is old for new, old in zip(parts, message.parts)):
            return message
        return replace(message, parts=parts)

    def _cleared_pointer(self, part: ToolReturnPart) -> str:
        numbers = self._call_cells.get(part.tool_call_id)
        if not numbers:  # e.g. history from before this capability was created
            return "[output cleared: call repl_history to see earlier cells]"
        if len(numbers) == 1:
            return (
                f"[output of cell {numbers[0]} cleared: "
                f"call repl_history(cell={numbers[0]})]"
            )
        cells = ", ".join(map(str, numbers))
        return f"[output of cells {cells} cleared: call repl_history(cell=N) for each]"

    def _get_session(self) -> eryx.Session:
        if self._session is None:
            options: dict[str, Any] = {
                "on_stdout": self._stdout.append,
                "on_stderr": self._stderr.append,
            }
            if self.volumes is not None:
                options["volumes"] = list(self.volumes)
            if self.factory is not None:
                session = self.factory.create_session(
                    resource_limits=self.resource_limits, **options
                )
            else:
                session = eryx.Session(**options)
            session.execution_timeout_ms = self.execution_timeout_ms
            if self.snapshot is not None:
                session.restore_state(self.snapshot)
            self._session = session
        return self._session

    def _restart(self) -> str:
        """Drop a session eryx can't run again, e.g. after a timeout traps it."""
        self._session = None
        return _restart_note(self.snapshot is not None)

    def _execute(self, code: str) -> Cell:
        """Run one cell, collecting output through the session's stream callbacks.

        The callbacks are needed because `ExecutionError` keeps stderr but not
        what the cell printed to stdout before it raised.
        """
        session = self._get_session()
        self._stdout.clear()
        self._stderr.clear()
        error = None
        start = time.perf_counter()
        try:
            session.execute(prepare_cell(code))
        except eryx.TimeoutError:
            error = f"timed out after {self.execution_timeout_ms} ms" + self._restart()
        except eryx.ResourceLimitError as e:
            error = f"resource limit exceeded: {e}" + self._restart()
        except eryx.ExecutionError as e:
            if str(e).startswith(_WASM_TRAP):
                error = f"sandbox crashed: {e}" + self._restart()
            else:
                error = _exception_name(str(e))
        stderr = _WRAPPER_FRAME.sub("", b"".join(self._stderr).decode(errors="replace"))
        cell = Cell(
            number=len(self._cells) + 1,
            code=code,
            stdout=b"".join(self._stdout).decode(errors="replace"),
            stderr=stderr,
            error=error,
            duration_ms=(time.perf_counter() - start) * 1000,
        )
        self._cells.append(cell)
        return cell


def _restart_note(snapshot: bool) -> str:
    return "; the sandbox was restarted and variables from earlier cells are gone" + (
        " (the preloaded state was restored)" if snapshot else ""
    )


def _exception_name(message: str) -> str:
    """Return the exception type from a traceback, e.g. `ZeroDivisionError`."""
    last = message.strip().splitlines()[-1] if message.strip() else ""
    name = last.split(":", 1)[0]
    return name if name.isidentifier() or "." in name else "error"
