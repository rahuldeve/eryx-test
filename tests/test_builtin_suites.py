"""numpy's and pandas' own test suites, run inside eryx sandboxes.

Slow (numpy ~4 min, pandas ~25 min), so deselected by default:

    uv run poe test-suites                          # both suites
    uv run poe test-suites -k numpy
    uv run poe test-suites -k pandas --suite-filter groupby

Both suites have failures that come from WASI itself (no subprocesses, 32-bit,
wasm traps). They are listed in known_failures/<suite>.txt, and only failures
that are not listed there fail the test. After fixing or accepting failures,
rerun the whole suite with --update-known-failures.
"""

import pathlib

import pytest
from builtin_suites import SUITES, run_suite

KNOWN_FAILURES_DIR = pathlib.Path(__file__).parent / "known_failures"


def outcomes(results: list[dict]) -> set[str]:
    """Failures, errors and crashed files as "FAILED <node id>" style lines."""
    found = set()
    for r in results:
        if r["crash"]:
            found.add(f"CRASH {r['file']}")
        for line in r["failures"]:
            # "FAILED ../site-packages/<node id> - <message>"
            line = line.split(" - ", 1)[0].replace("../site-packages/", "", 1)
            found.add(line)
    return found


def in_files(line: str, files: set[str]) -> bool:
    return line.split(" ", 1)[1].split("::", 1)[0] in files


@pytest.mark.slow
@pytest.mark.parametrize("suite", SUITES)
def test_suite(suite: str, request: pytest.FixtureRequest) -> None:
    file_filter = request.config.option.suite_filter
    jobs = request.config.option.suite_jobs
    results = run_suite(suite, file_filter, jobs=jobs)
    got = outcomes(results)

    known_path = KNOWN_FAILURES_DIR / f"{suite}.txt"
    if request.config.option.update_known_failures:
        if file_filter:
            pytest.fail("--update-known-failures needs a full run, not --suite-filter")
        known_path.write_text("".join(f"{line}\n" for line in sorted(got)))
        return

    ran = {r["file"] for r in results}
    known = {
        line
        for line in known_path.read_text().splitlines()
        if line and in_files(line, ran)
    }
    fixed = sorted(known - got)
    if fixed:
        print(f"{len(fixed)} known failures now pass:", *fixed, sep="\n  ")
    new = sorted(got - known)
    assert not new, f"{len(new)} new failures:\n" + "\n".join(new)
