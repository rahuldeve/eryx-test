"""Download the pure-Python packages for the sandbox into ./packages."""

import hashlib
import pathlib
import urllib.request

PACKAGES_DIR = pathlib.Path(__file__).resolve().parent.parent / "packages"

# name -> (url, sha256)
PACKAGES = {
    # Pure-Python pandas dependencies
    "python_dateutil-2.9.0.post0-py2.py3-none-any.whl": (
        "https://files.pythonhosted.org/packages/ec/57/56b9bcc3c9c6a792fcbaf139543cee77261f3651ca9da0c93f5c1221264b/python_dateutil-2.9.0.post0-py2.py3-none-any.whl",
        "a8b2bc7bffae282281c8140a97d3aa9c14da0b136dfe83f850eea9a5f7470427",
    ),
    "six-1.17.0-py2.py3-none-any.whl": (
        "https://files.pythonhosted.org/packages/b7/ce/149a00dd41f10bc29e5921b496af8b574d8413afcd5e30dfa0ed46c2cc5e/six-1.17.0-py2.py3-none-any.whl",
        "4721f391ed90541fddacab5acf947aa0d3dc7d27b2e1e8eda2be8970586c3274",
    ),
    "pytz-2026.5-py2.py3-none-any.whl": (
        "https://files.pythonhosted.org/packages/4f/ef/c66110d46fb800dda0bf33164182dfadabe26a90e4476844d502a23dca8e/pytz-2026.5-py2.py3-none-any.whl",
        "e658af3757f9e26a9d25dd2aff38335acd92bc9104f890a894b2c1ba28311b03",
    ),
    "tzdata-2026.5-py2.py3-none-any.whl": (
        "https://files.pythonhosted.org/packages/94/21/1e5995a1c920cce14e4bffae20c665ec10e7ed03ab25e006cd741092b718/tzdata-2026.5-py2.py3-none-any.whl",
        "b683bd1b6659ddcd810ff02ad09ba821d4bf1065072805063eb35c49617905ac",
    ),
}
# WASI-compiled native packages, built locally by wasi-build/build_packages.sh
LOCAL_PACKAGES = ["numpy-wasi.tar.gz", "pandas-wasi.tar.gz"]


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    PACKAGES_DIR.mkdir(exist_ok=True)
    for name, (url, digest) in PACKAGES.items():
        dest = PACKAGES_DIR / name
        if dest.exists() and sha256(dest) == digest:
            print(f"ok       {name}")
            continue
        print(f"fetching {name}")
        urllib.request.urlretrieve(url, dest)
        actual = sha256(dest)
        if actual != digest:
            dest.unlink()
            raise SystemExit(f"checksum mismatch for {name}: {actual}")
    for name in LOCAL_PACKAGES:
        if not (PACKAGES_DIR / name).exists():
            print(f"missing  {name} (run: bash wasi-build/build_packages.sh)")
    print(f"packages in {PACKAGES_DIR}")


if __name__ == "__main__":
    main()
