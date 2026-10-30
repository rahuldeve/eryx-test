"""Host values preloaded as sandbox globals.

The values are pickled into a host directory mounted read-only at
`VARIABLES_MOUNT`, and `LOAD_VARIABLES`, run in the sandbox, unpickles them into
its globals. Once that has run, the directory is no longer needed.

    with mounted_variables({"df": df}) as volume:
        sandbox = factory.create_sandbox(volumes=[volume])
        result = sandbox.execute(LOAD_VARIABLES + "result = len(df)")

Unpickling in the sandbox is safe for the host. It needs the values' modules in
the sandbox (e.g. pandas from the factory, matching the host's version) and can't
load classes defined only on the host.
"""

import pathlib
import pickle
import tempfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any

# Where the sandbox finds the pickled variables
VARIABLES_MOUNT = "/eryx-variables"
# A throwaway function keeps the loader's names out of the sandbox's globals
LOAD_VARIABLES = f"""
def _eryx_load():
    import pickle
    with open("{VARIABLES_MOUNT}/variables.pkl", "rb") as f:
        return pickle.load(f)
globals().update(_eryx_load())
del _eryx_load
"""


def write_variables(
    directory: str | pathlib.Path, pickled: bytes
) -> tuple[str, str, bool]:
    """Write `pickled` (a pickled dict of name -> value) into `directory`.

    Returns the volume that mounts `directory` where `LOAD_VARIABLES` reads it.
    """
    pathlib.Path(directory, "variables.pkl").write_bytes(pickled)
    return (str(directory), VARIABLES_MOUNT, True)


@contextmanager
def mounted_variables(variables: Mapping[str, Any]) -> Iterator[tuple[str, str, bool]]:
    """Pickle `variables` into a temporary directory and yield the volume mounting it.

    The directory is removed when the block exits, whether or not its code raised.
    """
    with tempfile.TemporaryDirectory(prefix="eryx-variables-") as tmp:
        yield write_variables(tmp, pickle.dumps(dict(variables)))
