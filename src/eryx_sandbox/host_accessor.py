"""The sandbox side of a `VariableStore`: the `host` object that reads it.

The store copies this file into its read-only mount, and its `setup_code` runs it
there, so it uses only the standard library. Every access reads `view.json` again;
the store never changes a file once written, so a read can't see a half-written
one. An `EryxRepl` with tools also gives `host` a `tools` attribute to call them.
"""

import json
import pickle
from collections.abc import Iterator
from typing import Any


class ToolError(Exception):
    """A host tool failed, or couldn't be called; the message says why."""


def _encode(value: Any) -> Any:
    """Turn what a cell commonly holds into JSON: numpy values and dates."""
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    raise TypeError(f"{type(value).__name__} can't be passed to a host tool")


class Tools:
    """The host's tools, as async functions: `await host.tools.search(query="x")`.

    Any name gives a function, and the host says if there is no such tool, so this
    needs no list of them. `eryx_sandbox.host_tools` explains why every call goes
    through one callback, and has the host's side of it.
    """

    __slots__ = ("_callback", "_invoke")

    ToolError = ToolError

    def __init__(self, invoke: Any, callback: str) -> None:
        self._invoke = invoke
        self._callback = callback

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__"):
            raise AttributeError(name)

        async def call(*args: Any, **kwargs: Any) -> Any:
            if args:
                raise TypeError(f"host.tools.{name}() takes keyword arguments only")
            reply = json.loads(
                await self._invoke(
                    self._callback,
                    tool=name,
                    args=json.dumps(kwargs, default=_encode),
                )
            )
            if "error" in reply:
                raise ToolError(reply["error"])
            return reply["value"]

        call.__name__ = name
        return call

    def __repr__(self) -> str:
        return "host tools: see the run_code description, or ask for one by name"


class Host:
    """Read-only access to the host's variables: `host.df` or `host["df"]`.

    Each read loads the value again, so it returns a fresh copy: changing it
    changes neither the host's value nor the next read. `host.tools` holds the
    host's tools, if it was given any.
    """

    __slots__ = ("_root", "_tools")

    def __init__(self, root: str) -> None:
        object.__setattr__(self, "_root", root)
        object.__setattr__(self, "_tools", None)

    def _enable_tools(self, invoke: Any, callback: str) -> None:
        object.__setattr__(self, "_tools", Tools(invoke, callback))

    @property
    def tools(self) -> Tools:
        # Without tools this falls through to __getattr__, so a variable named `tools`
        # still reads as usual
        if self._tools is None:
            raise AttributeError("tools")
        return self._tools

    def _entries(self) -> dict[str, dict[str, Any]]:
        with open(f"{self._root}/view.json") as f:
            return json.load(f)["entries"]

    def _entry(self, name: str) -> dict[str, Any]:
        entries = self._entries()
        if name not in entries:
            available = ", ".join(entries) or "none"
            raise KeyError(f"no host variable {name!r}; available: {available}")
        return entries[name]

    def __getitem__(self, name: str) -> Any:
        entry = self._entry(name)
        with open(f"{self._root}/blobs/{entry['blob']}", "rb") as f:
            data = f.read()
        if entry["codec"] == "pickle":
            return pickle.loads(data)
        raise ValueError(f"host variable {name!r} has unknown codec {entry['codec']!r}")

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(e.args[0]) from None

    def info(self, name: str) -> dict[str, Any]:
        """The variable's entry: its type, size, version and how it is stored."""
        return dict(self._entry(name))

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("host variables are read-only")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("host variables are read-only")

    def __setitem__(self, name: str, value: Any) -> None:
        raise TypeError("host variables are read-only")

    def __delitem__(self, name: str) -> None:
        raise TypeError("host variables are read-only")

    def __contains__(self, name: object) -> bool:
        return name in self._entries()

    def __iter__(self) -> Iterator[str]:
        return iter(list(self._entries()))

    def __len__(self) -> int:
        return len(self._entries())

    def __dir__(self) -> list[str]:
        return [*self._entries(), "info", *(["tools"] if self._tools else [])]

    def __repr__(self) -> str:
        entries = self._entries()
        listed = ", ".join(f"{n} ({e['type']})" for n, e in entries.items())
        return f"host variables: {listed or 'none'}"
