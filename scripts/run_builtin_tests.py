"""Run numpy's and pandas' own test suites inside eryx sandboxes.

Each test file runs in a fresh sandbox, so a test that traps the wasm instance
only takes its own file down. Uses the same marker selection as `numpy.test()`
and `pandas.test()`.

    uv run poe test numpy
    uv run poe test pandas -k groupby   # files matching "groupby"
"""

import argparse
import concurrent.futures
import hashlib
import os
import pathlib
import re
import shutil
import subprocess
import tempfile
import time

import eryx
from sandbox_env import (
    NUMPY_PACKAGES,
    PACKAGES_DIR,
    PANDAS_PACKAGES,
    cache_paths,
    extract,
    load_factory,
)

ROOT = PACKAGES_DIR.parent
RESULTS_DIR = ROOT / "test-results"

# Newer hypothesis releases need a native extension, so pin the last pure-Python one
TEST_DEPS = ["pytest==9.1.1", "hypothesis==6.155.7"]
# installs are cached per TEST_DEPS, so changing a pin takes effect
TEST_DEPS_KEY = hashlib.sha256(" ".join(TEST_DEPS).encode()).hexdigest()[:8]

SUITES = {
    "numpy": {
        "packages": NUMPY_PACKAGES,
        "imports": ["numpy"],
        "tests": [],
        # numpy.test(): label="fast" plus its warning filters. numpy has no
        # installed pytest config, so point rootdir at site-packages to pick up
        # numpy/conftest.py.
        "args": [
            "-m",
            "not slow",
            "--rootdir=/site-packages",
            "-W",
            "ignore:Not importing directory",
            "-W",
            "ignore:numpy.dtype size changed",
            "-W",
            "ignore:numpy.ufunc size changed",
            "-W",
            "ignore::UserWarning:cpuinfo",
            "-W",
            "ignore:the matrix subclass is not",
            "-W",
            "ignore:Importing from numpy.matlib is",
        ],
    },
    "pandas": {
        "packages": PANDAS_PACKAGES,
        "imports": ["numpy", "pandas"],
        "tests": ["pandas-tests.tar.gz"],
        # pandas.test(); the rest of the config comes from pandas/pyproject.toml.
        # Its addopts also writes a junit file, which fails because os.makedirs
        # can't stat "/" in the sandbox, so keep just the strictness flags.
        "args": [
            "-m",
            "not slow and not network and not db",
            "-o",
            "addopts=--strict-markers --strict-config",
            # sdists and wheels don't ship the test data files; skip tests that
            # need them, as pandas' own wheel tests do
            "--no-strict-data-files",
        ],
    },
}

# WASI has no os.dup (fd capture) or faulthandler. The cache plugin stays on
# (pandas' conftest reads its --lf option) but writes to /tmp, as site-packages
# is read-only.
COMMON_ARGS = [
    "-p",
    "no:faulthandler",
    "-o",
    "cache_dir=/tmp/.pytest_cache",
    "--capture=sys",
    "-q",
    "--no-header",
    "-rfE",
    "--tb=short",
]

PRELUDE = """
import os, sys
os.chdir("/tmp")
os.environ["NO_COLOR"] = "1"  # stdout has no fileno() for _colorize to check
# eryx's stdout writer lacks isatty(), which pytest's terminal reporter calls
for _stream in (sys.stdout, sys.stderr):
    _stream.isatty = lambda: False
# WASI has no users, but pytest's tmp_path fixture names its directory after
# os.getuid() and checks the owner against it (files are owned by uid 0 here)
if not hasattr(os, "getuid"):
    os.getuid = lambda: 0
# ...nor permission bits, but pytest's cache plugin sets a umask
if not hasattr(os, "umask"):
    os.umask = lambda mask: 0o022
import pytest


class _NoXdist:
    # pandas' S3/fsspec/parquet fixtures take pytest-xdist's worker_id, which
    # is "master" when not running distributed; xdist itself needs processes
    @pytest.fixture
    def worker_id(self):
        return "master"


_rc = pytest.main({args!r}, plugins=[_NoXdist()])
print(f"\\n[exit code {{int(_rc)}}]")
"""

SUMMARY_RE = re.compile(r"^=* ?(?:\d+ \w+(?:, )?)+ in [\d.]+s", re.M)
COUNT_RE = re.compile(
    r"(\d+) (passed|failed|skipped|xfailed|xpassed|errors?|deselected)"
)


def install_test_deps() -> pathlib.Path:
    dest = PACKAGES_DIR / f"test-deps-{TEST_DEPS_KEY}"
    if not dest.exists():
        tmp = dest.with_name(dest.name + ".tmp")
        shutil.rmtree(tmp, ignore_errors=True)
        subprocess.run(
            [
                "uv",
                "pip",
                "install",
                "-q",
                "--target",
                tmp,
                "--python-version",
                "3.14",
                "--python-platform",
                "linux",
                "--only-binary",
                ":all:",
                *TEST_DEPS,
            ],
            check=True,
        )
        shutil.rmtree(tmp / "bin", ignore_errors=True)
        native = list(tmp.rglob("*.so"))
        if native:
            raise SystemExit(f"test deps contain native modules: {native}")
        tmp.rename(dest)
    return dest


def build_test_site_packages(suite: str) -> pathlib.Path:
    """The suite's extracted site-packages, plus pytest/hypothesis and test files."""
    cfg = SUITES[suite]
    _, site_packages = cache_paths(suite, cfg["packages"])
    dest = site_packages.with_name(
        site_packages.name.replace(
            "-site-packages", f"-test-{TEST_DEPS_KEY}-site-packages"
        )
    )
    if dest.exists():
        return dest
    tmp = dest.with_name(dest.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    # hard links: the sandbox mounts site-packages read-only
    shutil.copytree(site_packages, tmp, copy_function=os.link)
    shutil.copytree(install_test_deps(), tmp, copy_function=os.link, dirs_exist_ok=True)
    if cfg["tests"]:
        extra = tmp.with_name(tmp.name + "-extra")
        shutil.rmtree(extra, ignore_errors=True)
        extract([PACKAGES_DIR / t for t in cfg["tests"]], extra)
        shutil.copytree(extra, tmp, dirs_exist_ok=True)
        shutil.rmtree(extra)
    tmp.rename(dest)
    return dest


def run_file(factory, site_packages, rel, args, log_path, limits) -> dict:
    with tempfile.TemporaryDirectory(prefix="eryx-test-") as tmp:
        tmp = pathlib.Path(tmp)
        (tmp / "tmp").mkdir()
        (tmp / "dev").mkdir()
        (tmp / "dev" / "null").touch()  # pytest's logging plugin opens /dev/null
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w") as log:

            def write(b: bytes) -> None:
                log.write(b.decode(errors="replace"))
                log.flush()

            sandbox = factory.create_sandbox(
                site_packages=str(site_packages),
                resource_limits=limits,
                volumes=[
                    (str(tmp / "tmp"), "/tmp", False),
                    (str(tmp / "dev"), "/dev", False),
                ],
                on_stdout=write,
                on_stderr=write,
            )
            start = time.perf_counter()
            crash = None
            try:
                sandbox.execute(
                    PRELUDE.format(args=[*COMMON_ARGS, *args, f"/site-packages/{rel}"])
                )
            except eryx.EryxError as e:
                lines = str(e).strip().splitlines()
                crash = f"{type(e).__name__}: {lines[-1] if lines else ''}"
                log.write(f"\n[sandbox {crash}]\n")
            elapsed = time.perf_counter() - start

    text = log_path.read_text()
    counts: dict[str, int] = {}
    summary = SUMMARY_RE.findall(text)
    if summary:
        for n, kind in COUNT_RE.findall(summary[-1]):
            counts[kind.rstrip("s") if kind.startswith("error") else kind] = int(n)
    failures = [l for l in text.splitlines() if l.startswith(("FAILED ", "ERROR "))]
    exit_code = re.search(r"\[exit code (\S+)\]", text)
    return {
        "file": rel,
        "counts": counts,
        "failures": failures,
        "crash": crash,
        "exit": exit_code.group(1) if exit_code else None,
        "seconds": elapsed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("suite", choices=SUITES)
    parser.add_argument(
        "-k", "--filter", help="only run test files whose path contains this"
    )
    parser.add_argument(
        "-j", "--jobs", type=int, default=4, help="sandboxes in parallel (default 4)"
    )
    parser.add_argument(
        "--timeout", type=int, default=1800, help="per-file timeout, seconds"
    )
    parser.add_argument(
        "--memory-mb", type=int, default=2048, help="per-sandbox memory limit"
    )
    parser.add_argument("pytest_args", nargs="*", help="extra pytest args (after --)")
    opts = parser.parse_args()

    cfg = SUITES[opts.suite]
    factory = load_factory(opts.suite, cfg["packages"], imports=cfg["imports"])
    site_packages = build_test_site_packages(opts.suite)

    files = sorted(
        str(p.relative_to(site_packages))
        for p in (site_packages / opts.suite).rglob("test_*.py")
        if "/tests/" in str(p) and (not opts.filter or opts.filter in str(p))
    )
    if not files:
        raise SystemExit("no test files matched")

    out_dir = RESULTS_DIR / opts.suite
    shutil.rmtree(out_dir, ignore_errors=True)
    limits = eryx.ResourceLimits(
        execution_timeout_ms=opts.timeout * 1000,
        max_memory_bytes=opts.memory_mb * 1024 * 1024,
    )
    args = [*cfg["args"], *opts.pytest_args]
    print(
        f"running {len(files)} {opts.suite} test files, {opts.jobs} at a time; logs in {out_dir}"
    )

    results = []
    start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(opts.jobs) as pool:
        futures = [
            pool.submit(
                run_file,
                factory,
                site_packages,
                rel,
                args,
                out_dir / (rel.removesuffix(".py") + ".log"),
                limits,
            )
            for rel in files
        ]
        for fut in concurrent.futures.as_completed(futures):
            r = fut.result()
            results.append(r)
            status = (
                r["crash"]
                or ", ".join(f"{n} {k}" for k, n in r["counts"].items())
                or f"exit {r['exit']}"
            )
            print(
                f"[{len(results):4}/{len(files)}] {r['file']}: {status} ({r['seconds']:.0f}s)",
                flush=True,
            )

    totals: dict[str, int] = {}
    for r in results:
        for k, n in r["counts"].items():
            totals[k] = totals.get(k, 0) + n
    results.sort(key=lambda r: r["file"])
    crashed = [r for r in results if r["crash"]]
    failures = [f for r in results for f in r["failures"]]

    report = [f"# {opts.suite} test suite in eryx", ""]
    report.append("totals: " + ", ".join(f"{n} {k}" for k, n in sorted(totals.items())))
    report.append(f"files: {len(files)}, crashed sandboxes: {len(crashed)}")
    report.append(f"wall time: {time.perf_counter() - start:.0f}s")
    if crashed:
        report += [
            "",
            "## crashed / timed out",
            *(f"{r['file']}: {r['crash']}" for r in crashed),
        ]
    if failures:
        report += ["", "## failures and errors", *failures]
    (out_dir / "summary.txt").write_text("\n".join(report) + "\n")
    print("\n" + "\n".join(report[:5]))
    print(f"full summary: {out_dir / 'summary.txt'}")


if __name__ == "__main__":
    main()
