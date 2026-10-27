"""Smoke test: run a few snippets inside a basic eryx sandbox."""

import eryx


def out(result: eryx.ExecuteResult) -> str:
    # pyeryx 0.11 returns stdout as bytes
    s = result.stdout
    return s.decode() if isinstance(s, bytes) else s


def main() -> None:
    sandbox = eryx.Sandbox()
    result = sandbox.execute(
        """
import sys, platform
print("Hello from the sandbox!")
print("python", sys.version.split()[0], "on", sys.platform)
result = {"answer": 2 + 2}
"""
    )
    print(out(result), end="")
    print("result:", result.result)
    print(f"took {result.duration_ms:.1f}ms")

    # Confirm isolation: host filesystem is not visible
    result = sandbox.execute(
        """
import os
try:
    print(os.listdir("/home"))
except Exception as e:
    print("host fs blocked:", type(e).__name__, e)
"""
    )
    print(out(result), end="")

    # Confirm third-party packages are not available yet
    result = sandbox.execute(
        """
for mod in ("numpy", "pandas", "bokeh"):
    try:
        __import__(mod)
        print(mod, "available")
    except ImportError:
        print(mod, "missing")
"""
    )
    print(out(result), end="")


if __name__ == "__main__":
    main()
