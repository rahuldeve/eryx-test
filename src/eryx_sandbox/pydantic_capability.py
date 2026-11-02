"""A persistent eryx session as a pydantic-ai capability: a sandboxed Python REPL.

`EryxRepl` gives the model two tools:

- `run_code` runs a cell in one eryx session that lives as long as the capability,
  so variables carry over between calls and across `agent.run()` calls. It returns
  the cell's stdout and stderr (a traceback if the code raised), and echoes the
  value of a bare last expression, as a REPL would.
- `repl_history` returns the code of earlier cells, and their output on request.

Every cell is kept on the host, and the instructions carry an index of them built
from that record rather than from the conversation. So whatever a compaction
strategy does to old tool calls and results (clears, summarises or drops them),
the model can still see what it ran and fetch it again with `repl_history`.
Compaction capabilities can be listed in any order relative to `EryxRepl`.

    repl = EryxRepl(factory, variables={"df": df}, on_change="append")
    agent = Agent(
        "anthropic:claude-sonnet-5-5",
        capabilities=[ClearToolResults(max_tokens=50_000), repl],
    )
"""

import ast
import asyncio
import re
import threading
import time
import weakref
from collections.abc import Mapping, Sequence
from dataclasses import KW_ONLY, InitVar, dataclass, field
from typing import Any, Self

import eryx
from pydantic_ai import FunctionToolset, RunContext
from pydantic_ai.agent import AbstractAgent
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import InstructionChangePolicy, InstructionPart

from eryx_sandbox.variables import VariableStore

INSTRUCTIONS = """\
You have a persistent, sandboxed Python REPL. Run code with `run_code`: \
variables, imports and functions carry over between calls. Each call returns the \
cell's stdout and stderr, and the value of a bare last expression is printed. \
Earlier calls and their results may be cleared, summarised or dropped from this \
conversation, but the list of cells below is complete: `repl_history(cell=N)` shows \
a cell's code, and `repl_history(cell=N, output=True)` also what it printed."""

# eryx runs user code from its own wrapper, which shows up as a traceback frame
_WRAPPER_FRAME = re.compile(r'  File "<string>", line \d+, in _eryx_exec\n')
_ECHO_VAR = "__eryx_repl_value"
# eryx captures this global after each cell and then deletes it; by default it is
# `result`, which a model's own `result` would lose between cells
_RESULT_VAR = "__eryx_repl_result"
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
    # only the line breaks ast counts: str.splitlines() also splits on \x0c, \u2028...
    lines = re.split(r"\r\n|\r|\n", code)
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
    cell history, until `reset()`. Every new session (the first, and one after a
    restart or `reset()`) restores `snapshot`, mounts `store` and defines `host`
    to read it; if that fails, `run_code` raises `RuntimeError`, which ends the
    run. Host variables live on the host, so they survive restarts. An
    instance belongs to the first agent it is given to; another agent gets a `ValueError`. It should also serve one
    conversation at a time, which isn't checked: concurrent runs of its agent
    share variables and history. Tool calls in one model response run one at a
    time, in order. Cancelling `run_code` doesn't stop its cell (it runs in a
    thread): the next cell, and `reset()`, wait for it to finish or time out, and
    it is still recorded. That wait counts against a pydantic-ai tool timeout
    on `run_code`, so set one above `execution_timeout_ms`, or the model's retry
    can time out too.
    """

    factory: eryx.SandboxFactory | None = None
    """Factory with packages pre-installed; `None` uses a bare `eryx.Session`."""

    _: KW_ONLY

    resource_limits: eryx.ResourceLimits | None = None
    """Limits for the session; `None` uses eryx's defaults (128 MB of memory).
    Their `execution_timeout_ms` is replaced by this capability's own."""

    volumes: Sequence[tuple[str, str, bool]] | None = None
    """Host directories to mount, as `(host_path, sandbox_path, read_only)`."""

    snapshot: bytes | None = None
    """State from `Session.snapshot_state()` to restore when the session starts."""

    store: VariableStore | None = None
    """Host values the model reads as `host.NAME` (see `eryx_sandbox.variables`).

    `None` creates one from `variables`. A store passed in belongs to the caller,
    and can be shared by several REPLs. It can't be replaced afterwards."""

    variables: InitVar[Mapping[str, Any] | None] = None
    """Shorthand for `store=VariableStore(variables)`. The values are copied and
    pickled when the capability is created, so later changes on the host aren't
    seen. Unpickling happens in the sandbox, when a cell reads one, so it needs the
    values' modules (e.g. pandas from the factory) and can't load classes defined
    only on the host. Only an init argument: read `store` afterwards, as this
    attribute stays `None`."""

    execution_timeout_ms: int | None = 60_000
    """Time limit for one cell; `None` removes it."""

    max_output_chars: int = 20_000
    """Longer cell output is clipped in the `run_code` return; history keeps it all."""

    index_cells: int = 50
    """Most recent cells listed in the instructions; `0` lists none, only how many
    there are."""

    on_change: InstructionChangePolicy = "rewrite"
    """How changes to the instructions (the cell index) reach the model.

    `'rewrite'` resends them in the prompt prefix on each request. `'append'` keeps
    the prefix and adds each change to history (as a diff where shorter), so the
    prompt cache survives new cells; the full text is resent after compaction. See
    pydantic-ai's `InstructionPart.on_change`."""

    _session: eryx.Session | None = field(default=None, init=False, repr=False)
    _cells: list[Cell] = field(default_factory=list, init=False, repr=False)
    _stdout: list[bytes] = field(default_factory=list, init=False, repr=False)
    _stderr: list[bytes] = field(default_factory=list, init=False, repr=False)
    # held while a cell runs, so a cancelled cell can't overlap the next one
    _lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False
    )
    _agent: weakref.ref[AbstractAgent[Any, Any]] | None = field(
        default=None, init=False, repr=False
    )

    def __setattr__(self, name: str, value: Any) -> None:
        # A running session keeps the store it mounted, so a new one would put the
        # instructions and `host` out of step
        if name == "store" and self.__dict__.get("store") is not None:
            raise AttributeError("EryxRepl.store can't be changed after creation")
        if name == "variables":
            raise AttributeError("EryxRepl.variables is only an init argument")
        super().__setattr__(name, value)

    def __post_init__(self, variables: Mapping[str, Any] | None) -> None:
        # Appended instruction changes are addressed by the capability's id
        if self.id is None:
            self.id = "eryx_repl"
        if self.index_cells < 0:
            raise ValueError(f"index_cells must be >= 0, got {self.index_cells}")
        if self.store is not None and variables is not None:
            raise ValueError("pass variables or store, not both")
        if self.store is None:
            # Pickled now so an unpicklable value fails here, not on the first call
            self.store = VariableStore(variables)
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
        with self._lock:
            self._session = None
            self._cells.clear()

    @property
    def _host(self) -> VariableStore:
        assert self.store is not None, "set in __post_init__"
        return self.store

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
        output = cell.output() or "(no output)\n"
        if len(output) > self.max_output_chars:
            output = (
                output[: self.max_output_chars]
                + f"\n... {len(output) - self.max_output_chars} more characters, "
                f"see repl_history(cell={cell.number}, output=True)\n"
            )
        return f"[cell {cell.number}]\n{output}"

    async def repl_history(
        self,
        ctx: RunContext[Any],
        cell: int | None = None,
        last: int = 5,
        output: bool = False,
    ) -> str:
        """Show the code of earlier REPL cells, and optionally their full output.

        Args:
            cell: The cell number to show. Omit it to show the most recent cells.
            last: How many of the most recent cells to show when `cell` is omitted.
            output: Also show each cell's full output (stdout, stderr and error).
        """
        if cell is not None:
            if not 1 <= cell <= len(self._cells):
                return f"No cell {cell}: cells run so far are 1-{len(self._cells)}."
            shown = [self._cells[cell - 1]]
        else:
            shown = self._cells[-max(last, 1) :]
        if not shown:
            return "No cells have been run yet."
        return "\n\n".join(_render_cell(c, output) for c in shown)

    def _instructions(self, ctx: RunContext[Any]) -> str | InstructionPart:
        text = self._instructions_text()
        if self.on_change == "append":
            return InstructionPart(text, name="repl", on_change="append")
        return text

    def _instructions_text(self) -> str:
        text = INSTRUCTIONS
        if names := self._host.names():
            listed = ", ".join(f"{n} ({self._host.info(n).type})" for n in names)
            text += (
                f"\n\nHost variables (read-only): {listed}. Read one with "
                f"`host.NAME`, e.g. `{names[0]} = host.{names[0]}`; each read "
                "returns a fresh copy."
            )
        if not self._cells:
            return text
        # -0 would slice from the start and list every cell
        shown = self._cells[-self.index_cells :] if self.index_cells else []
        lines = [c.summary() for c in shown]
        if earlier := len(self._cells) - len(shown):
            lines.insert(0, f"({earlier} earlier cells, see repl_history)")
        return text + "\n\nCells run so far:\n" + "\n".join(lines)

    def _get_session(self) -> eryx.Session:
        if self._session is None:
            self._session = self._create_session(
                [*(self.volumes or []), self._host.volume]
            )
        return self._session

    def _create_session(self, volumes: list[tuple[str, str, bool]]) -> eryx.Session:
        options: dict[str, Any] = {
            "resource_limits": self.resource_limits,
            "on_stdout": self._stdout.append,
            "on_stderr": self._stderr.append,
            "result_variable": _RESULT_VAR,
            "volumes": volumes,
        }
        # Failures here are host setup problems (a bad snapshot, a module missing in
        # the sandbox, a broken mount), not something for the model to see as a
        # cell error
        try:
            if self.factory is not None:
                session = self.factory.create_session(**options)
            else:
                session = eryx.Session(**options)
            session.execution_timeout_ms = self.execution_timeout_ms
            if self.snapshot is not None:
                session.restore_state(self.snapshot)
        except eryx.EryxError as e:
            raise RuntimeError(f"starting the EryxRepl session failed: {e}") from e
        try:
            session.execute(self._host.setup_code)
        except eryx.EryxError as e:
            raise RuntimeError(f"setting up EryxRepl host variables failed: {e}") from e
        return session

    def _restart(self) -> str:
        """Drop a session eryx can't run again, e.g. after a timeout traps it."""
        self._session = None
        return _restart_note(
            snapshot=self.snapshot is not None, host=bool(self._host.names())
        )

    def _execute(self, code: str) -> Cell:
        """Run one cell, collecting output through the session's stream callbacks.

        The callbacks are needed because `ExecutionError` keeps stderr but not
        what the cell printed to stdout before it raised.
        """
        with self._lock:
            session = self._get_session()
            self._stdout.clear()
            self._stderr.clear()
            error = None
            start = time.perf_counter()
            try:
                session.execute(prepare_cell(code))
            except eryx.TimeoutError:
                error = (
                    f"timed out after {self.execution_timeout_ms} ms" + self._restart()
                )
            except eryx.ResourceLimitError as e:
                error = f"resource limit exceeded: {e}" + self._restart()
            except eryx.ExecutionError as e:
                if str(e).startswith(_WASM_TRAP):
                    error = f"sandbox crashed: {e}" + self._restart()
                else:
                    error = _exception_name(str(e))
            stderr = _WRAPPER_FRAME.sub(
                "", b"".join(self._stderr).decode(errors="replace")
            )
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


def _render_cell(cell: Cell, output: bool) -> str:
    text = (
        f"[cell {cell.number}{f', {cell.error}' if cell.error else ''}]\n"
        f">>> code\n{cell.code.rstrip()}\n"
    )
    if output:
        text += f">>> output\n{cell.output() or '(no output)'}"
    return text


def _restart_note(snapshot: bool, host: bool) -> str:
    note = "; the sandbox was restarted and variables from earlier cells are gone"
    if snapshot:
        note += " (the preloaded state was restored)"
    if host:
        note += "; host variables are unchanged"
    return note


def _exception_name(message: str) -> str:
    """Return the exception type from a traceback, e.g. `ZeroDivisionError`."""
    last = message.strip().splitlines()[-1] if message.strip() else ""
    name = last.split(":", 1)[0]
    return name if name.isidentifier() or "." in name else "error"
