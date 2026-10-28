"""Smoke test: a bare eryx sandbox runs Python and is isolated from the host."""

import eryx
import pytest

from eryx_sandbox.factory import out


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
