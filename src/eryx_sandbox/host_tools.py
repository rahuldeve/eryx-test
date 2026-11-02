"""Agent tools that sandboxed code calls as `await host.tools.NAME(...)`.

`SandboxedToolset` hides the selected tools from the model's tool list and describes
them on `run_code` instead. Each call from a cell goes through one eryx callback,
`DISPATCH_CALLBACK`, to `CellTools`, which runs the tool on the agent's event loop
with the agent's hooks, validation and `deps`.

Why one dispatcher rather than one eryx callback per tool
---------------------------------------------------------
Registering each tool with `CallbackRegistry` looks simpler, and is what eryx's docs
show. These pyeryx 0.12.1 behaviours rule it out (re-check them when upgrading):

- Positional arguments can silently go to the wrong parameter. eryx maps them onto
  the schema's properties, whose keys arrive sorted, so for `send(to, body)` the call
  `await send("bob", "hi")` runs with `to="hi", body="bob"`. `host.tools` refuses
  positional arguments, so this can't happen.
- A tool's result often can't cross. eryx converts it to a JSON value itself, where
  one NaN anywhere fails the whole result, and `datetime`, `Decimal` and pydantic
  models are unsupported types. The dispatcher returns a JSON string that pydantic
  has already encoded, which eryx passes through untouched.
- Callbacks are fixed when the session is created, but an `EryxRepl` session outlives
  agent runs, and a run's tools can change at every step. The dispatcher looks a tool
  up on each call, so the session never has to be rebuilt to match.
- Callbacks become sandbox globals. The model's variables can replace one, a name
  like `foo-bar` can't be written in Python, and a reserved name like `print` isn't
  defined at all. `host.tools.NAME` has none of these problems, and only needs the
  sanitized names `partition_callable_tools` gives it.

Why the dispatcher does its own waiting and time limits
-------------------------------------------------------
These apply however tools are registered, so a dispatcher only makes them one place
to handle:

- eryx runs each callback on a thread of its own, and an async one in a new event
  loop. Tools can hold things tied to the agent's loop, such as an MCP connection or
  an HTTP client, so each call is sent back to that loop.
- `callback_timeout_ms` (10 s by default) only stops eryx waiting: the sandbox sees a
  timeout while the tool keeps running on the host. `EryxRepl` turns it off.
- The cell's time limit doesn't interrupt a pending callback, and fires only once the
  callback returns. Alone, a hung tool would hang the cell for good, so `CellTools`
  cancels tools at the limit. The sandbox then gets a `ToolError`, and because it
  arrives before eryx's clock fires, the session and its variables survive.

Everything is enforced here on the host. Sandboxed code can call the dispatcher
directly with `invoke`, which only gets it the checks that `host.tools` gets.

Limits: tools only work with `await`, so plain functions and `df.apply` can't call
them. Their time counts towards the cell's limit, as eryx's clock keeps running
during a callback. A tool that needs approval fails, as a cell can't pause for it. Sequential
tools run one at a time, but other tools can overlap them. `ToolReturn.content`
doesn't reach the model. A tool's error message is shown to the sandboxed code.
"""

import asyncio
import json
import time
from collections.abc import Mapping
from concurrent.futures import CancelledError, Future
from contextlib import nullcontext
from contextvars import ContextVar, copy_context
from dataclasses import dataclass, replace
from typing import Any

from pydantic import ValidationError
from pydantic_ai import RunContext, ToolDefinition, WrapperToolset
from pydantic_ai.exceptions import ModelRetry, UsageLimitExceeded
from pydantic_ai.function_signature import FunctionSignature
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.tool_manager import ToolManager
from pydantic_ai.tools import ToolSelector, matches_tool_selector
from pydantic_ai.toolsets.abstract import ToolsetTool

# Private to the harness, which shares them between its own code modes. A rename there
# fails on import, whereas a copy here would drift without anyone noticing
from pydantic_ai_harness._code_tools import (
    can_be_sandboxed,
    partition_callable_tools,
    run_nested_call,
    signatures_and_conflicts,
)
from pydantic_core import PydanticSerializationError, to_json

DISPATCH_CALLBACK = "__eryx_host_tool"
DISPATCH_SCHEMA = {
    "type": "object",
    "properties": {"tool": {"type": "string"}, "args": {"type": "string"}},
    "required": ["tool", "args"],
}
TOOLS_SETUP_CODE = f"host._enable_tools(invoke, {DISPATCH_CALLBACK!r})\n"
"""Run after a store's `setup_code` to give `host` its `tools`."""

RUN_CODE = "run_code"
# The REPL's own tools always stay with the model
_REPL_TOOLS = {RUN_CODE, "repl_history"}
# Tools are stopped this long before the cell's limit, as eryx's clock fires as soon
# as a callback returns after it. That would end the cell with a timeout, and restart
# the session, rather than raise a `ToolError` the model can read
_DEADLINE_MARGIN_S = 0.1

_CATALOG_HEADER = (
    "Code in a cell can call these host functions as `await host.tools.NAME(...)`, "
    "with keyword arguments only. They can only be awaited, so a plain function or "
    "`df.apply` can't call them: use `async def` and `asyncio.gather`, which runs calls "
    "concurrently. Arguments and results are JSON: datetimes arrive as strings and "
    "bytes as URL-safe base64. A failed call raises `host.tools.ToolError`. Time spent "
    "in these functions counts towards the cell's time limit."
)

current_cell_tools: ContextVar["CellTools | None"] = ContextVar(
    "eryx_cell_tools", default=None
)
"""Hands a `run_code` call its `CellTools`, from the toolset that made them."""


class CellTools:
    """What the dispatcher needs for one `run_code` call: the run's loop, context and tools.

    The session outlives each run, but these don't, so they are looked up per cell. A
    cell holds these only while it holds the REPL's lock: a cancelled cell that is
    still running must not reach the next cell's run, with that run's `deps`.
    """

    def __init__(
        self,
        manager: ToolManager[Any],
        names: Mapping[str, str],
        parent_call_id: str,
        max_result_bytes: int,
    ) -> None:
        self._loop = asyncio.get_running_loop()
        self._context = copy_context()
        self._manager = manager
        self._names = names
        self._parent_call_id = parent_call_id
        self._max_result_bytes = max_result_bytes
        self._deadline: float | None = None
        self._calls = 0
        self._closed = False
        self._pending: set[Future[str]] = set()
        self._sequential = asyncio.Lock()
        self.usage_limit_exceeded: UsageLimitExceeded | None = None
        """Set when a call hit `tool_calls_limit`, which ends the run after the cell."""

    def start(self, timeout_ms: int | None) -> None:
        """Start the cell's clock, just before eryx starts its own."""
        if timeout_ms is not None:
            self._deadline = time.monotonic() + timeout_ms / 1000 - _DEADLINE_MARGIN_S

    def close(self) -> None:
        """Cancel running tools and refuse new ones, as `run_code` itself was cancelled."""
        self._closed = True
        for future in list(self._pending):
            future.cancel()

    def dispatch(self, tool: str, args: str) -> str:
        """Run one call and return its JSON reply; runs on a thread of eryx's."""
        if self._closed:
            return error_reply("this cell was cancelled, so its tool calls don't run")
        timeout = None
        if self._deadline is not None:
            timeout = max(self._deadline - time.monotonic(), 0)
        call = self._call(tool, args)
        try:
            # In a copy of the run's context, so a tool sees its tracing spans and any
            # other context variables, as it would when called by the model
            future = self._context.copy().run(
                asyncio.run_coroutine_threadsafe, call, self._loop
            )
        except RuntimeError:  # the run's event loop has closed
            call.close()
            return error_reply("the agent run that started this cell has ended")
        self._pending.add(future)
        if self._closed:  # close() may have missed it
            future.cancel()
        try:
            return future.result(timeout)
        except TimeoutError:
            future.cancel()
            return error_reply("the cell's time limit ran out during this tool call")
        except CancelledError:
            return error_reply("this tool call was cancelled")
        finally:
            self._pending.discard(future)

    async def _call(self, name: str, args_json: str) -> str:
        original = self._names.get(name)
        if original is None:
            available = ", ".join(self._names) or "none"
            return error_reply(f"no host tool {name!r}; available: {available}")
        self._calls += 1
        part = ToolCallPart(
            original,
            json.loads(args_json),
            tool_call_id=f"{self._parent_call_id}__{self._calls}",
        )
        one_at_a_time = (
            self._manager.is_sequential(part)
            or self._manager.get_parallel_execution_mode() == "sequential"
        )
        async with self._sequential if one_at_a_time else nullcontext():
            outcome = await run_nested_call(self._manager, part)
        if outcome.usage_limit_exceeded is not None:
            self.usage_limit_exceeded = outcome.usage_limit_exceeded
        if outcome.error is not None:
            return error_reply(_describe_error(outcome.error))
        try:
            # NaN stays valid for the sandbox's json.loads. pydantic's base64 is the
            # URL-safe alphabet, which the catalog tells the model
            value = to_json(
                outcome.content, inf_nan_mode="constants", bytes_mode="base64"
            )
        except PydanticSerializationError as e:
            return error_reply(f"{name}'s result can't be sent as JSON: {e}")
        if len(value) > self._max_result_bytes:
            # Big enough to run the sandbox out of memory, which crashes it and loses
            # its variables, rather than raising MemoryError
            return error_reply(
                f"{name}'s result is {len(value)} bytes, over the limit of "
                f"{self._max_result_bytes}; ask for less of it"
            )
        return f'{{"value": {value.decode()}}}'


def error_reply(message: str) -> str:
    """A dispatcher reply that raises `ToolError(message)` in the sandbox."""
    return json.dumps({"error": message})


def _describe_error(error: Exception) -> str:
    if isinstance(error, ModelRetry):
        return error.message
    if isinstance(error, ValidationError):
        # On one line and without the docs URLs, so the traceback ends in the error
        problems = "; ".join(
            f"{'.'.join(map(str, e['loc']))}: {e['msg']}"
            for e in error.errors(include_url=False, include_input=False)
        )
        return f"invalid arguments for {error.title}: {problems}"
    return f"{type(error).__name__}: {error}"


@dataclass(kw_only=True)
class _RunCodeTool(ToolsetTool[Any]):
    """`run_code` for one run step, with the tools its cells can call in that step."""

    inner: ToolsetTool[Any]
    sandboxed: dict[str, ToolsetTool[Any]]
    names: dict[str, str]


@dataclass
class SandboxedToolset(WrapperToolset[Any]):
    """Moves the tools `selector` picks from the model's tool list into `host.tools`."""

    selector: ToolSelector[Any]
    max_result_bytes: int

    async def get_tools(self, ctx: RunContext[Any]) -> dict[str, ToolsetTool[Any]]:
        tools = await self.wrapped.get_tools(ctx)
        run_code = tools.get(RUN_CODE)
        if run_code is None:  # filtered out, so nothing could call them from a cell
            return tools
        native: dict[str, ToolsetTool[Any]] = {}
        sandboxed: dict[str, ToolsetTool[Any]] = {}
        for name, tool in tools.items():
            selected = (
                name not in _REPL_TOOLS
                and can_be_sandboxed(tool.tool_def)
                and ctx.is_tool_available(tool.tool_def)
                and await matches_tool_selector(self.selector, ctx, tool.tool_def)
            )
            (sandboxed if selected else native)[name] = tool
        defs, renamed = partition_callable_tools(
            sandboxed, owner="EryxRepl", reserved_name=RUN_CODE
        )
        native[RUN_CODE] = _RunCodeTool(
            toolset=self,
            tool_def=_with_catalog(run_code.tool_def, defs),
            max_retries=run_code.max_retries,
            args_validator=run_code.args_validator,
            args_validator_func=run_code.args_validator_func,
            inner=run_code,
            sandboxed=sandboxed,
            names={safe: renamed.get(safe, safe) for safe in defs},
        )
        return native

    async def call_tool(
        self,
        name: str,
        tool_args: dict[str, Any],
        ctx: RunContext[Any],
        tool: ToolsetTool[Any],
    ) -> Any:
        if not isinstance(tool, _RunCodeTool):
            return await self.wrapped.call_tool(name, tool_args, ctx, tool)
        assert ctx.tool_manager is not None, "set by the agent for a tool call"
        cell_tools = CellTools(
            # Goes through the same validation and hooks as a call from the model
            ToolManager(
                toolset=self.wrapped,
                root_capability=ctx.tool_manager.root_capability,
                ctx=ctx,
                tools=tool.sandboxed,
            ),
            tool.names,
            ctx.tool_call_id or RUN_CODE,
            self.max_result_bytes,
        )
        token = current_cell_tools.set(cell_tools)
        try:
            result = await self.wrapped.call_tool(name, tool_args, ctx, tool.inner)
        except asyncio.CancelledError:
            cell_tools.close()
            raise
        finally:
            current_cell_tools.reset(token)
        if cell_tools.usage_limit_exceeded is not None:
            raise cell_tools.usage_limit_exceeded
        return result


def _with_catalog(
    tool_def: ToolDefinition, defs: dict[str, ToolDefinition]
) -> ToolDefinition:
    if not defs:
        return tool_def
    signatures, conflicting = signatures_and_conflicts(defs)
    blocks = [
        *FunctionSignature.render_type_definitions(signatures, conflicting),
        *(
            td.render_signature(
                "...", is_async=True, conflicting_type_names=conflicting
            )
            for td in defs.values()
        ),
    ]
    catalog = "\n\n".join(blocks)
    return replace(
        tool_def,
        description=(
            f"{tool_def.description}\n\n{_CATALOG_HEADER}\n\n```python\n{catalog}\n```"
        ),
    )
