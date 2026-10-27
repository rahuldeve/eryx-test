import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("builtin suites", "numpy/pandas test suites (-m slow)")
    group.addoption(
        "--suite-filter", help="only run suite test files whose path contains this"
    )
    group.addoption(
        "--suite-jobs", type=int, default=4, help="sandboxes in parallel (default 4)"
    )
    group.addoption(
        "--update-known-failures",
        action="store_true",
        help="rewrite tests/known_failures/<suite>.txt from this run",
    )
