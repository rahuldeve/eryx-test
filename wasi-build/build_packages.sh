#!/usr/bin/env bash
# Cross-compile numpy for CPython 3.14 / wasm32-wasip2 so eryx can late-link it.
#
# Toolchain and numpy recipe come from dicej/wasi-wheels (wasi-sdk 27, dicej's
# CPython 3.14 fork, upstream numpy 1.26.0b1).
#
# Output: ../packages/numpy-wasi.tar.gz
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
WORK=$HERE/work
OUT=$HERE/../packages
JOBS=${JOBS:-$(nproc)}

WASI_WHEELS_REV=b28844f

WHEELS=$WORK/wasi-wheels
WASI_SDK=$WHEELS/build/wasi-sdk
CPYTHON=$WHEELS/cpython/builddir/wasi/install

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
git -C "$WHEELS" submodule update --init --depth 1 cpython numpy/src
git -C "$WHEELS/numpy/src" submodule update --init --depth 1 --recursive

echo "==> wasi-sdk + CPython 3.14 for WASI"
MAKEFLAGS=-j$JOBS make -C "$WHEELS" "$WASI_SDK" "$CPYTHON"

echo "==> numpy"
if ! ls "$WHEELS"/numpy/src/build/lib.*/numpy/core/_multiarray_umath.*.so >/dev/null 2>&1; then
  (cd "$WHEELS/numpy" && CROSS_PREFIX=$CPYTHON WASI_SDK_PATH=$WASI_SDK bash build.sh)
fi
NUMPY_PKG=$(ls -d "$WHEELS"/numpy/src/build/lib.*/numpy)
tar -czf "$OUT/numpy-wasi.tar.gz" -C "$(dirname "$NUMPY_PKG")" numpy
echo "==> wrote $OUT/numpy-wasi.tar.gz"
