#!/usr/bin/env bash
# Cross-compile numpy and pandas for CPython 3.14 / wasm32-wasip2 so eryx can
# late-link them.
#
# The toolchain comes from dicej/wasi-wheels (wasi-sdk, dicej's CPython 3.14
# fork). No published numpy or pandas WASI build targets CPython 3.14
# (wasi-wheels only has numpy 1.26 and a 3.12 pandas), so both are built here
# with meson, pandas against this numpy.
#
# Output: ../packages/numpy-wasi.tar.gz, ../packages/pandas-wasi.tar.gz and
# ../packages/pandas-tests.tar.gz (pandas' test suite, for run_builtin_tests.py)
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
WORK=$HERE/work
OUT=$HERE/../packages
JOBS=${JOBS:-$(nproc)}

WASI_WHEELS_REV=b28844f
NUMPY_VERSION=2.4.6
NUMPY_SHA256=f3a3570c4a2a16746ac2c31a7c7c7b0c186b95ce902e33db6f28094ed7387dda
PANDAS_VERSION=2.3.3
PANDAS_SHA256=e05e1af93b977f7eafa636d043f9f94c7ee3ac81af99c13508215942e64c993b
# The Cython release current when pandas 2.3.3 shipped (2025-09-29). pandas
# only asks for Cython<4, but 3.3 changed the auto-generated pickle state
# (an empty instance __dict__ is left out), which pandas' NDArrayBacked
# __setstate__ rejects: Period/categorical/string/tz arrays can't be unpickled.
CYTHON_VERSION=3.1.4

WHEELS=$WORK/wasi-wheels
WASI_SDK=$WHEELS/build/wasi-sdk
CPYTHON=$WHEELS/cpython/builddir/wasi/install
SYSCONFIG_DIR=$WHEELS/cpython/builddir/wasi/build/lib.wasi-wasm32-3.14
NUMPY_DIR=$WORK/numpy
NUMPY_SRC=$NUMPY_DIR/numpy-$NUMPY_VERSION
# numpy's install stage, which pandas builds against
NUMPY_SITE=$NUMPY_DIR/stage/usr/local/lib/python3.14/site-packages
NUMPY_INC=$NUMPY_SITE/numpy/_core/include
PANDAS_SRC=$WORK/pandas-$PANDAS_VERSION
PANDAS_STAGE=$WORK/stage
PANDAS_SITE=$PANDAS_STAGE/usr/local/lib/python3.14/site-packages
TOOLCHAIN=$WORK/toolchain
VENV=$WORK/venv

mkdir -p "$WORK" "$OUT"

# Host CPython 3.14 (from uv), exposed as python3.14 for wasi-wheels' scripts
HOST_PY=$(uv python find 3.14)
mkdir -p "$WORK/bin"
ln -sf "$HOST_PY" "$WORK/bin/python3.14"
export PATH=$WORK/bin:$PATH

echo "==> wasi-wheels sources"
if [ ! -d "$WHEELS" ]; then
  git clone https://github.com/dicej/wasi-wheels.git "$WHEELS"
fi
git -C "$WHEELS" checkout -q "$WASI_WHEELS_REV"
git -C "$WHEELS" submodule update --init --depth 1 cpython

echo "==> wasi-sdk + CPython 3.14 for WASI"
MAKEFLAGS=-j$JOBS make -C "$WHEELS" "$WASI_SDK" "$CPYTHON"

echo "==> build tools"
[ -d "$VENV" ] || uv venv -q -p "$HOST_PY" "$VENV"
uv pip install -q -p "$VENV" "meson>=1.2.1,<2" ninja "Cython==$CYTHON_VERSION" versioneer
export PATH=$VENV/bin:$PATH

# meson introspects this "python" for EXT_SUFFIX, include dirs, etc. The
# sysconfigdata override fixes the config vars, but get_paths() still derives
# include dirs from the host interpreter's prefix, so patch those too. It runs
# the build venv's python because numpy's and pandas' code generators import
# Cython, and keeps PYTHONPATH, which meson sets for numpy's generators.
mkdir -p "$WORK/wasi-site"
cat > "$WORK/wasi-site/sitecustomize.py" <<'EOF'
import sysconfig

_get_paths = sysconfig.get_paths


def get_paths(*args, **kwargs):
    paths = _get_paths(*args, **kwargs)
    paths["include"] = paths["platinclude"] = sysconfig.get_config_var("INCLUDEPY")
    return paths


sysconfig.get_paths = get_paths
EOF
cat > "$WORK/python-wasi" <<EOF
#!/bin/sh
export _PYTHON_SYSCONFIGDATA_NAME=_sysconfigdata__wasi_wasm32-wasi
export PYTHONPATH=$SYSCONFIG_DIR:$WORK/wasi-site\${PYTHONPATH:+:\$PYTHONPATH}
exec "$VENV/bin/python" "\$@"
EOF
chmod +x "$WORK/python-wasi"


echo "==> cross toolchain"
# Compiler wrappers, since meson can't express these with cross-file args:
# - link extension modules against libpython, but keep meson's configure-time
#   executable checks free of it (static links can't take a shared library)
# - drop --start-group/--end-group, which wasm-ld doesn't know (and lld
#   doesn't need)
# - resolve C++ throws to abort() (see tools/cxa_stub.c)
mkdir -p "$TOOLCHAIN"
"$WASI_SDK/bin/clang" --target=wasm32-wasip2 -fPIC -O2 \
  -c "$HERE/tools/cxa_stub.c" -o "$TOOLCHAIN/cxa_stub.o"
rm -f "$TOOLCHAIN/libcxa_stub.a"
"$WASI_SDK/bin/llvm-ar" rcs "$TOOLCHAIN/libcxa_stub.a" "$TOOLCHAIN/cxa_stub.o"
for tool in clang clang++; do
  cat > "$TOOLCHAIN/$tool" <<EOF
#!/bin/sh
shared=
for arg; do
  shift
  case \$arg in
    -Wl,--start-group|-Wl,--end-group) continue ;;
    -shared) shared=1 ;;
  esac
  set -- "\$@" "\$arg"
done
[ -n "\$shared" ] && set -- "\$@" "$CPYTHON/lib/libpython3.14.so" "$TOOLCHAIN/libcxa_stub.a"
exec "$WASI_SDK/bin/$tool" "\$@"
EOF
  chmod +x "$TOOLCHAIN/$tool"
done

cat > "$TOOLCHAIN/wasi.ini" <<EOF
[constants]
args = ['--target=wasm32-wasip2', '-fPIC']

[binaries]
c = '$TOOLCHAIN/clang'
cpp = '$TOOLCHAIN/clang++'
# wasip2's default linker (wasm-component-ld) isn't recognised by meson; for
# -shared it only forwards to wasm-ld anyway
c_ld = 'lld'
cpp_ld = 'lld'
ar = '$WASI_SDK/bin/llvm-ar'
strip = '$WASI_SDK/bin/llvm-strip'
python = '$WORK/python-wasi'
cython = '$VENV/bin/cython'

[built-in options]
c_args = args
cpp_args = args
c_link_args = ['--target=wasm32-wasip2']
cpp_link_args = ['--target=wasm32-wasip2']

[properties]
needs_exe_wrapper = true
# can't be detected by running a program when cross compiling
longdouble_format = 'IEEE_QUAD_LE'
# the cython sanity check resolves python3 via the host interpreter running
# meson (x86_64 headers); the build itself uses the cross-file python above
skip_sanity_check = true

[host_machine]
system = 'wasi'
cpu_family = 'wasm32'
cpu = 'wasm32'
endian = 'little'
EOF

# pandas' additions, layered over wasi.ini
cat > "$TOOLCHAIN/pandas.ini" <<'EOF'
[built-in options]
c_args = args + ['-DNPY_NO_SIGNAL']
# No C++ exceptions on WASI: compile Cython's try/catch wrappers around
# libcpp calls (only in window/aggregations) down to the try body
cpp_args = args + ['-DNPY_NO_SIGNAL', '-fno-exceptions',
                   '-Dtry=if (true)', '-Dcatch(...)=else', '-D__Pyx_CppExn2PyErr()=((void)0)',
                   '-Wno-keyword-macro']
EOF

TOOLCHAIN_KEY=$(cat "$TOOLCHAIN/clang" "$TOOLCHAIN/clang++" "$TOOLCHAIN/wasi.ini" \
  "$TOOLCHAIN/pandas.ini" "$HERE/tools/cxa_stub.c" | sha256sum | cut -c1-12)

# meson_build MESON SRC KEY STAGE SETUP_ARGS...
# Configure SRC/build-wasi from scratch whenever KEY changes (ninja doesn't
# track the toolchain, the Cython version or numpy's headers and .pxd files as
# inputs of the generated C), then build and install into STAGE.
meson_build() {
  local meson=$1 src=$2 key=$3 stage=$4
  shift 4
  if [ "$(cat "$src/build-wasi/.build-key" 2>/dev/null)" != "$key" ]; then
    rm -rf "$src/build-wasi"
    $meson setup "$src/build-wasi" "$src" -Dbuildtype=release -Db_ndebug=if-release "$@"
    echo "$key" > "$src/build-wasi/.build-key"
  fi
  $meson compile -C "$src/build-wasi" -j "$JOBS"
  rm -rf "$stage"
  $meson install -C "$src/build-wasi" --destdir "$stage" --quiet
}

echo "==> numpy $NUMPY_VERSION sources"
if [ ! -d "$NUMPY_SRC" ]; then
  mkdir -p "$NUMPY_DIR"
  curl -sSL --fail -o "$NUMPY_DIR/numpy.tar.gz" \
    "https://files.pythonhosted.org/packages/source/n/numpy/numpy-$NUMPY_VERSION.tar.gz"
  echo "$NUMPY_SHA256  $NUMPY_DIR/numpy.tar.gz" | sha256sum -c
  tar -xzf "$NUMPY_DIR/numpy.tar.gz" -C "$NUMPY_DIR"
fi

echo "==> numpy build"
# numpy needs its vendored meson fork
numpy_meson() { "$VENV/bin/python" "$NUMPY_SRC/vendored-meson/meson/meson.py" "$@"; }
# No BLAS (lapack_lite instead), no threads on wasip2, and no SIMD: wasm32 has
# no dispatchable CPU features
meson_build numpy_meson "$NUMPY_SRC" "toolchain-$TOOLCHAIN_KEY cython-$CYTHON_VERSION" "$NUMPY_DIR/stage" \
  --cross-file "$TOOLCHAIN/wasi.ini" \
  -Dallow-noblas=true -Dblas=none -Dlapack=none -Ddisable-threading=true \
  -Ddisable-highway=true -Ddisable-svml=true -Ddisable-intel-sort=true \
  -Dcpu-baseline=none -Dcpu-dispatch=none
[ -d "$NUMPY_SITE/numpy" ] || { echo "numpy not found in install stage"; exit 1; }
grep -q 'NPY_SIZEOF_PY_INTPTR_T 4' "$NUMPY_INC/numpy/_numpyconfig.h" \
  || { echo "numpy headers are not wasm32"; exit 1; }
tar -czf "$OUT/numpy-wasi.tar.gz" -C "$NUMPY_SITE" --exclude=__pycache__ numpy
echo "==> wrote $OUT/numpy-wasi.tar.gz"

echo "==> pandas $PANDAS_VERSION sources"
# Patched once, when extracted: delete $PANDAS_SRC to apply changed patches.
# - pandas asks the build python for numpy's include dir; point it at the
#   wasm32 headers, and let Cython find numpy's .pxd files (meson doesn't pass
#   the cross file's cython_args to the transpile step).
# - WASI CPython has no ctypes or mmap. pandas only needs them for Windows
#   clipboard errors, the raw-pointer interchange protocol and memory_map=True,
#   but imports them at module level on the `import pandas` path, so make those
#   imports optional. mmap.mmap is also used in isinstance checks, so it gets a
#   placeholder class that nothing is an instance of.
if [ ! -d "$PANDAS_SRC" ]; then
  curl -sSL --fail -o "$WORK/pandas.tar.gz" \
    "https://files.pythonhosted.org/packages/source/p/pandas/pandas-$PANDAS_VERSION.tar.gz"
  echo "$PANDAS_SHA256  $WORK/pandas.tar.gz" | sha256sum -c
  rm -rf "$PANDAS_SRC.tmp"
  mkdir "$PANDAS_SRC.tmp"
  tar -xzf "$WORK/pandas.tar.gz" -C "$PANDAS_SRC.tmp" --strip-components=1
  python3.14 - "$PANDAS_SRC.tmp" "$NUMPY_INC" "$NUMPY_SITE" <<'EOF'
import pathlib, re, sys

root, inc, site = sys.argv[1:]


def patch(rel, pattern, repl, flags=0):
    path = pathlib.Path(root, rel)
    src, n = re.subn(pattern, lambda m: repl.replace("\\0", m[0]), path.read_text(), flags=flags)
    assert n, f"could not patch {rel}"
    path.write_text(src)


patch("pandas/meson.build", r"incdir_numpy = run_command\(.*?\)\.stdout\(\)\.strip\(\)",
      f"incdir_numpy = '{inc}'", flags=re.S)
for rel in ["pandas/_libs/meson.build", "pandas/_libs/tslibs/meson.build", "pandas/_libs/window/meson.build"]:
    patch(rel, r"'-X always_allow_keywords=true'", f"\\0, '--include-dir', '{site}'")

FALLBACKS = {
    "ctypes": "    ctypes = None\n",
    "mmap": (
        "    import types\n\n"
        "    class _NoMmap:\n"
        "        def __init__(self, *args, **kwargs):\n"
        '            raise OSError("mmap is not available on WASI")\n\n'
        "    mmap = types.SimpleNamespace(mmap=_NoMmap, ACCESS_READ=1)\n"
    ),
}
for mod, rel in [
    ("ctypes", "pandas/errors/__init__.py"),
    ("ctypes", "pandas/core/interchange/from_dataframe.py"),
    ("mmap", "pandas/io/common.py"),
    ("mmap", "pandas/io/excel/_openpyxl.py"),
]:
    patch(rel, rf"^import {mod}\n",
          f"try:\n    import {mod}\nexcept ImportError:  # no {mod} on WASI\n{FALLBACKS[mod]}", flags=re.M)
EOF
  mv "$PANDAS_SRC.tmp" "$PANDAS_SRC"
fi

echo "==> pandas build"
meson_build meson "$PANDAS_SRC" "toolchain-$TOOLCHAIN_KEY cython-$CYTHON_VERSION numpy-$NUMPY_VERSION" "$PANDAS_STAGE" \
  --cross-file "$TOOLCHAIN/wasi.ini" --cross-file "$TOOLCHAIN/pandas.ini"
[ -d "$PANDAS_SITE/pandas" ] || { echo "pandas not found in install stage"; exit 1; }
# pandas/tests is large and unused at runtime...
tar -czf "$OUT/pandas-wasi.tar.gz" -C "$PANDAS_SITE" --exclude='pandas/tests' pandas
echo "==> wrote $OUT/pandas-wasi.tar.gz"
# ...but kept separately for scripts/run_builtin_tests.py
tar -czf "$OUT/pandas-tests.tar.gz" -C "$PANDAS_SITE" pandas/tests
echo "==> wrote $OUT/pandas-tests.tar.gz"
