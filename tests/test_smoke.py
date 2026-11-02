"""Smoke test: a bare eryx sandbox runs Python and is isolated from the host."""

import eryx
import pytest

from eryx_sandbox.factory import NUMPY_PACKAGES, load_factory, out


@pytest.fixture(scope="module")
def sandbox() -> eryx.Sandbox:
    return eryx.Sandbox()


def test_runs_python(sandbox: eryx.Sandbox) -> None:
    result = sandbox.execute(
        """
import sys
print("Hello from the sandbox!")
result = {"answer": 2 + 2, "version": sys.version_info[:2], "platform": sys.platform}
"""
    )
    assert out(result.stdout) == "Hello from the sandbox!\n"
    assert result.result == {"answer": 4, "version": [3, 14], "platform": "wasi"}


def test_host_filesystem_is_hidden(sandbox: eryx.Sandbox) -> None:
    result = sandbox.execute(
        """
import os
try:
    os.listdir("/home")
    result = "visible"
except FileNotFoundError:
    result = "hidden"
"""
    )
    assert result.result == "hidden"


def test_no_third_party_packages(sandbox: eryx.Sandbox) -> None:
    result = sandbox.execute(
        """
import importlib.util
result = [m for m in ("numpy", "pandas", "bokeh") if importlib.util.find_spec(m)]
"""
    )
    assert result.result == []


def test_copy_numpy_scalar() -> None:
    # numpy's __copy__ has too few parameters, which traps on wasm unless patched
    # (build_packages.sh). Scalars' copy is the path users can actually reach.
    factory = load_factory("numpy", NUMPY_PACKAGES, imports=["numpy"])
    result = factory.create_sandbox().execute(
        """
import copy
import numpy as np
result = float(copy.copy(np.float64(1.5)))
"""
    )
    assert result.result == 1.5
