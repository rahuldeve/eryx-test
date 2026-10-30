"""Host values the sandbox reads as `host.NAME`, from files on a read-only mount.

`VariableStore` pickles each value into its own file in a temporary directory,
mounted read-only at `STORE_MOUNT`. Its `setup_code`, run once in the sandbox,
defines `host`, which loads a value when it is read:

    with VariableStore({"df": df}) as store:
        sandbox = factory.create_sandbox(volumes=[store.volume])
        result = sandbox.execute(store.setup_code + "result = len(host.df)")

Every read unpickles the file again, so it returns a fresh copy. The store can't
be changed after it is created. Unpickling in the sandbox is safe for the host. It
needs the values' modules in the sandbox (e.g. pandas from the factory, matching
the host's version) and can't load classes defined only on the host.

Layout of the store's directory (`shared/` is what the sandbox sees):

    shared/view.json      {"format": 1, "epoch": 1, "entries": {name: entry}}
    shared/host.py        the sandbox side, a copy of `host_accessor.py`
    shared/blobs/<hash>   one value's pickle, named by its sha256; never changed
"""

import hashlib
import json
import keyword
import os
import pathlib
import pickle
import shutil
import tempfile
import weakref
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from types import TracebackType
from typing import Any, Self

# Where the sandbox finds the store
STORE_MOUNT = "/eryx-store"
_ACCESSOR = pathlib.Path(__file__).with_name("host_accessor.py")
_FORMAT = 1

# A throwaway function keeps the setup's names out of the sandbox's globals
SETUP_CODE = f"""
def _eryx_setup():
    path = "{STORE_MOUNT}/host.py"
    namespace = {{"__name__": "eryx_host"}}
    with open(path) as f:
        exec(compile(f.read(), path, "exec"), namespace)
    return namespace["Host"]("{STORE_MOUNT}")
host = _eryx_setup()
del _eryx_setup
"""


@dataclass(frozen=True)
class Entry:
    """One variable in the store, as the sandbox's `host.info(name)` also sees it."""

    name: str
    version: int
    codec: str
    """How `blob` is decoded in the sandbox: `"pickle"`."""
    blob: str
    """The sha256 of the stored bytes, which is also their file name."""
    type: str
    """The value's type name on the host, e.g. `DataFrame`."""
    size: int


class VariableStore:
    """Host values for the sandbox to read as `host.NAME` (see module docs).

    `variables` is copied and each value pickled when the store is created, so an
    unpicklable value fails here and later changes to the dict aren't seen. The
    directory is removed by `close()`, at the end of a `with` block, or when the
    store is garbage collected.
    """

    setup_code = SETUP_CODE
    """Run once in each new sandbox or session to define `host`."""

    def __init__(self, variables: Mapping[str, Any] | None = None) -> None:
        values = dict(variables or {})
        bad = [n for n in values if not _is_name(n)]
        if bad:
            raise ValueError(f"variable names must be Python identifiers: {bad}")
        pickled = {name: pickle.dumps(value) for name, value in values.items()}
        self._dir = pathlib.Path(tempfile.mkdtemp(prefix="eryx-store-"))
        self._finalizer = weakref.finalize(
            self, shutil.rmtree, self._dir, ignore_errors=True
        )
        try:
            self._entries = self._write(
                {
                    name: (data, type(values[name]).__name__)
                    for name, data in pickled.items()
                }
            )
        except BaseException:
            self._finalizer()
            raise

    def _write(self, pickled: dict[str, tuple[bytes, str]]) -> dict[str, Entry]:
        shared = self._dir / "shared"
        blobs = shared / "blobs"
        blobs.mkdir(parents=True)
        entries = {}
        for name, (data, type_name) in pickled.items():
            digest = hashlib.sha256(data).hexdigest()
            if not (blobs / digest).exists():
                _atomic_write(blobs / digest, data)
            entries[name] = Entry(name, 1, "pickle", digest, type_name, len(data))
        _atomic_write(shared / "host.py", _ACCESSOR.read_bytes())
        view = {
            "format": _FORMAT,
            "epoch": 1,
            "entries": {name: asdict(entry) for name, entry in entries.items()},
        }
        # Written last: the sandbox finds nothing until every blob is in place
        _atomic_write(shared / "view.json", json.dumps(view, indent=1).encode())
        return entries

    def names(self) -> list[str]:
        """The variables' names, in the order they were given."""
        return list(self._entries)

    def info(self, name: str) -> Entry:
        return self._entries[name]

    @property
    def volume(self) -> tuple[str, str, bool]:
        """The read-only mount for `create_session`/`create_sandbox`'s `volumes`."""
        return (str(self._dir / "shared"), STORE_MOUNT, True)

    def close(self) -> None:
        """Remove the store's directory. Sandboxes using it can no longer read it."""
        self._finalizer()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def __repr__(self) -> str:
        listed = ", ".join(f"{e.name}: {e.type}" for e in self._entries.values())
        return f"VariableStore({listed})"


def _atomic_write(path: pathlib.Path, data: bytes) -> None:
    """Write `data` to `path` so that a reader sees either the old file or the new."""
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def _is_name(name: object) -> bool:
    return isinstance(name, str) and name.isidentifier() and not keyword.iskeyword(name)
