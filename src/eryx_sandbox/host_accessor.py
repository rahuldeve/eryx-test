"""The sandbox side of a `VariableStore`: the `host` object that reads it.

The store copies this file into its read-only mount, and its `setup_code` runs it
there, so it uses only the standard library. Every access reads `view.json` again;
the store never changes a file once written, so a read can't see a half-written
one.
"""

import json
import pickle
from collections.abc import Iterator
from typing import Any


class Host:
    """Read-only access to the host's variables: `host.df` or `host["df"]`.

    Each read loads the value again, so it returns a fresh copy: changing it
    changes neither the host's value nor the next read.
    """

    __slots__ = ("_root",)

    def __init__(self, root: str) -> None:
        object.__setattr__(self, "_root", root)

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
        return [*self._entries(), "info"]

    def __repr__(self) -> str:
        entries = self._entries()
        listed = ", ".join(f"{n} ({e['type']})" for n, e in entries.items())
        return f"host variables: {listed or 'none'}"
