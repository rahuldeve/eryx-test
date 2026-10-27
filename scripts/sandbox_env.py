"""Build (or load cached) eryx sandbox factories with WASI packages."""

import hashlib
import os
import pathlib
import shutil
import tarfile
import time
import zipfile

# The numpy/pandas component is too large for eryx's default pooling allocator
# ("component instance size does not fit in pooling allocator requirements").
# Must be set before eryx is imported.
os.environ.setdefault("ERYX_ALLOCATOR", "on-demand")

import eryx  # noqa: E402

PACKAGES_DIR = pathlib.Path(__file__).resolve().parent.parent / "packages"

# Package sets, as file names in PACKAGES_DIR (see fetch_packages.py and
# wasi-build/build_packages.sh)
NUMPY_PACKAGES = ["numpy-wasi.tar.gz"]
PANDAS_PACKAGES = [
    *NUMPY_PACKAGES,
    "pandas-wasi.tar.gz",
    "python_dateutil-2.9.0.post0-py2.py3-none-any.whl",
    "six-1.17.0-py2.py3-none-any.whl",
    "pytz-2026.5-py2.py3-none-any.whl",
    "tzdata-2026.5-py2.py3-none-any.whl",
]


def out(text: bytes | str) -> str:
    # pyeryx 0.11 returns stdout as bytes
    return text.decode() if isinstance(text, bytes) else text


def extract(paths: list[pathlib.Path], dest: pathlib.Path) -> None:
    tmp = dest.with_name(dest.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir()
    for p in paths:
        if p.suffix == ".whl":
            with zipfile.ZipFile(p) as zf:
                zf.extractall(tmp)
        else:
            with tarfile.open(p) as tf:
                tf.extractall(tmp, filter="data")
    tmp.rename(dest)


def cache_paths(name: str, packages: list[str]) -> tuple[pathlib.Path, pathlib.Path]:
    """Return (factory cache file, extracted site-packages dir) for a package set."""
    paths = [PACKAGES_DIR / p for p in packages]
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        raise SystemExit(
            f"missing packages: {missing}\n"
            "run: uv run scripts/fetch_packages.py && bash wasi-build/build_packages.sh"
        )

    digest = hashlib.sha256()
    for p in paths:
        digest.update(p.read_bytes())
    key = f"{name}-{eryx.__version__}-{digest.hexdigest()[:12]}"
    return PACKAGES_DIR / f"{key}.bin", PACKAGES_DIR / f"{key}-site-packages"


def load_factory(name: str, packages: list[str], imports: list[str]) -> eryx.SandboxFactory:
    """Return a factory with `packages` installed and `imports` pre-imported.

    Building takes 15-25s, so the factory is cached per eryx version and package
    set. A saved factory holds the memory snapshot but not the package files, so
    they are also kept extracted and handed back to `load()` as site-packages;
    otherwise any module not pre-imported fails to import.
    """
    paths = [PACKAGES_DIR / p for p in packages]
    cache, site_packages = cache_paths(name, packages)
    start = time.perf_counter()
    if not site_packages.exists():
        extract(paths, site_packages)
    if cache.exists():
        print("factory loading from cache")
    else:
        built = eryx.SandboxFactory(packages=[str(p) for p in paths], imports=imports)
        built.save(str(cache))
        print(f"factory built in {time.perf_counter() - start:.1f}s")
    # Always use the persistent site-packages: a freshly built factory deletes its
    # temporary extraction when garbage-collected, even while sandboxes created
    # from it are still in use.
    return eryx.SandboxFactory.load(str(cache), site_packages=str(site_packages))
