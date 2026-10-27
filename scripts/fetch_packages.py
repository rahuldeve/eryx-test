"""Download the pure-Python packages for the sandbox into ./packages."""

import hashlib
import io
import pathlib
import tarfile
import tempfile
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
    # bokeh and its pure-Python dependencies. tornado (bokeh server and notebook
    # support) and pillow (PNG/SVG export) are left out.
    "bokeh-3.10.1-py3-none-any.whl": (
        "https://files.pythonhosted.org/packages/4e/9c/aaf2f11e31a75f0fffc79659ad7f0be01f39bb018bbc83c2a8a02657589e/bokeh-3.10.1-py3-none-any.whl",
        "d73412b6317f70df0ffe6796956af609617523dfdfbc624d4f7f496cd09e8101",
    ),
    "jinja2-3.1.6-py3-none-any.whl": (
        "https://files.pythonhosted.org/packages/62/a1/3d680cbfd5f4b8f15abc1d571870c5fc3e594bb582bc3b64ea099db13e56/jinja2-3.1.6-py3-none-any.whl",
        "85ece4451f492d0c13c5dd7c13a64681a86afae63a5f347908daf103ce6d2f67",
    ),
    "narwhals-2.27.1-py3-none-any.whl": (
        "https://files.pythonhosted.org/packages/d1/89/5d4c86da1130d9059681e5b6cd7645df5c10279a6a079c5c37dcb2cc6f3f/narwhals-2.27.1-py3-none-any.whl",
        "d057df13f5852b8e157596e82eb5e955fad267425df5e420e0ee9863da483b31",
    ),
    "packaging-26.3-py3-none-any.whl": (
        "https://files.pythonhosted.org/packages/63/34/ba1c580383c9eada3711951fef0795c80b829a078d72188184bcab9dd527/packaging-26.3-py3-none-any.whl",
        "d7193f7c8e4e93f444fde0262bf90af30e16fa0ad0ad44cb553c87339b23cd1c",
    ),
    "xyzservices-2026.9.1-py3-none-any.whl": (
        "https://files.pythonhosted.org/packages/a8/71/9219fe65ef32bddc0b5d14e1d98dfefe9d2c9c1b1d1e93efca1f29fcbd4d/xyzservices-2026.9.1-py3-none-any.whl",
        "af4fddac0f1fa5f7951834e2c58e10109e722be548cb1fe4f1a8e8b18cef4c09",
    ),
}
# Packages that only publish native wheels but fall back to pure Python without
# their C speedups: the sdist's package directory is repacked as a tarball.
# name -> (url, sha256, package dir inside the sdist)
SDIST_PACKAGES = {
    "markupsafe-3.0.4-pure.tar.gz": (
        "https://files.pythonhosted.org/packages/38/9b/e422a865e1d5d57d0e509b4e0bf1c1a70a7f6382c29a5aa428df994c8bc8/markupsafe-3.0.4.tar.gz",
        "2e9ad7dd851bf45fab9f75cbff4cb493fee9979e8d8c7c9c3ee119022518edd6",
        "src/markupsafe",
    ),
    "pyyaml-6.0.3-pure.tar.gz": (
        "https://files.pythonhosted.org/packages/05/8e/961c0007c59b8dd7729d542c61a4d537767a59645b82a0b521206e1e25c2/pyyaml-6.0.3.tar.gz",
        "d76623373421df22fb4cf8817020cbb7ef15c725b9d5e45f17e189bfc384190f",
        "lib/yaml",
    ),
}
# WASI-compiled native packages, built locally by wasi-build/build_packages.sh
LOCAL_PACKAGES = ["numpy-wasi.tar.gz", "pandas-wasi.tar.gz"]


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download(url: str, digest: str, dest: pathlib.Path) -> None:
    urllib.request.urlretrieve(url, dest)
    actual = sha256(dest)
    if actual != digest:
        dest.unlink()
        raise SystemExit(f"checksum mismatch for {url}: {actual}")


def repack_sdist(sdist: pathlib.Path, pkg_dir: str, dest: pathlib.Path) -> None:
    """Write the sdist's pure-Python package dir, plus its metadata, to dest."""
    with tarfile.open(sdist) as src, tarfile.open(dest, "w:gz") as out:
        top = src.getnames()[0].split("/")[0]
        prefix = f"{top}/{pkg_dir}/"
        for member in src.getmembers():
            if member.isfile() and member.name.startswith(prefix) and not member.name.endswith(".c"):
                rel = member.name.removeprefix(prefix)
                member_out = tarfile.TarInfo(f"{pkg_dir.rsplit('/', 1)[-1]}/{rel}")
                member_out.size, member_out.mtime, member_out.mode = member.size, member.mtime, 0o644
                out.addfile(member_out, src.extractfile(member))
        # PKG-INFO doubles as dist-info METADATA, so importlib.metadata finds it
        metadata = src.extractfile(f"{top}/PKG-INFO").read()
        info = tarfile.TarInfo(f"{top}.dist-info/METADATA")
        info.size = len(metadata)
        out.addfile(info, io.BytesIO(metadata))


def main() -> None:
    PACKAGES_DIR.mkdir(exist_ok=True)
    for name, (url, digest) in PACKAGES.items():
        dest = PACKAGES_DIR / name
        if dest.exists() and sha256(dest) == digest:
            print(f"ok       {name}")
            continue
        print(f"fetching {name}")
        download(url, digest, dest)
    for name, (url, digest, pkg_dir) in SDIST_PACKAGES.items():
        dest = PACKAGES_DIR / name
        if dest.exists():
            print(f"ok       {name}")
            continue
        print(f"fetching {name}")
        with tempfile.TemporaryDirectory() as tmp:
            sdist = pathlib.Path(tmp, "sdist.tar.gz")
            download(url, digest, sdist)
            repack_sdist(sdist, pkg_dir, dest.with_name(dest.name + ".tmp"))
        dest.with_name(dest.name + ".tmp").rename(dest)
    for name in LOCAL_PACKAGES:
        if not (PACKAGES_DIR / name).exists():
            print(f"missing  {name} (run: bash wasi-build/build_packages.sh)")
    print(f"packages in {PACKAGES_DIR}")


if __name__ == "__main__":
    main()
