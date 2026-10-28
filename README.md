# Eryx sandbox with numpy, pandas and bokeh

This repo is a proof of concept for running numpy, pandas and bokeh inside Eryx sandbox. Since WASI wheels does not exist for numpy and pandas, they had to be compiled from source. WASI does not offer a 1:1 execution enviorment to a native one, leading to some missing features and quirks that needs to be known when writing numpy or pandas coe in this sandbox. refer to [sandbox-limitations.md](sandbox-limitations.md) for a non exhaustive list of this


## Usage

Needs [uv](https://docs.astral.sh/uv/). Every command is a
[poe](https://poethepoet.natn.io/) task (`uv run poe` lists them).

```sh
uv sync                       # host dependencies
uv run poe packages           # download the pure-Python packages, then cross-compile
                              # numpy and pandas for WASI (slow; JOBS=8 limits parallelism)
uv run poe dataframe          # demo: plot a host DataFrame with bokeh in a sandbox,
                              # written to out/dataframe-bokeh.html
uv run poe session            # demo: snapshot a session with the DataFrame loaded,
                              # restore it and plot cell by cell, REPL style,
                              # written to out/session-bokeh.html
uv run poe test               # smoke and end-to-end tests
uv run poe test-suites        # numpy's and pandas' own test suites (~30 min)
uv run poe test-suites -k pandas --suite-filter groupby
```

The code lives in the `eryx_sandbox` package (`src/eryx_sandbox/`), which
`uv sync` installs in editable mode; `scripts/` holds the demos and the package
build tooling. To run your own code, build a sandbox from a cached factory and
pass host DataFrames in as pickles on a read-only mount (see
`scripts/run_dataframe_bokeh.py`):

```python
from eryx_sandbox.factory import BOKEH_PACKAGES, load_factory, shared_frames

factory = load_factory("bokeh", BOKEH_PACKAGES, imports=["numpy", "pandas", "bokeh.plotting"])
with shared_frames({"sales": df}) as volumes:
    sandbox = factory.create_sandbox(volumes=volumes)
    result = sandbox.execute("""
import pandas as pd
df = pd.read_pickle("/host/sales.pkl")
result = df.describe().to_dict()
""")
print(result.result)
```

The first `load_factory` call for a package set takes 15-25s; later ones load
from the cache in `packages/` (set `ERYX_PACKAGES_DIR` to use another directory).

All codebase inside this repo was authored by Claude