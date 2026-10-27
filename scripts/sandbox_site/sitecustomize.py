"""Runs at interpreter startup inside every sandbox (packed by sandbox_env.py)."""

import os

# WASI has no HOME and no pwd module, so Path.home() raises. bokeh calls it
# at import time; /tmp is where hosts conventionally mount a writable dir.
os.environ.setdefault("HOME", "/tmp")
