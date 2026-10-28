"""numpy, pandas and bokeh in eryx sandboxes.

- `eryx_sandbox.factory`: cached sandbox factories with the WASI packages
  installed, and host DataFrames shared into sandboxes.
"""

import os

# The numpy/pandas component is too large for eryx's default pooling allocator
# ("component instance size does not fit in pooling allocator requirements").
# Set here so it is in place before any submodule creates a sandbox.
os.environ.setdefault("ERYX_ALLOCATOR", "on-demand")
