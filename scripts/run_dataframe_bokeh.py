"""Plot a host-built DataFrame with bokeh inside an eryx sandbox.

The frame is preloaded as the sandbox global `df`: the host pickles it into a
temporary directory mounted read-only, and a short preamble unpickles it before
the plotting code runs (`run_with_variables`). The sandbox returns bokeh's
`json_item` for the plot, which the host embeds in an HTML page with BokehJS.
"""

import json
import pathlib
import pickle
import shutil
import tempfile
import time
from typing import Any

import eryx
import numpy as np
import pandas as pd

from eryx_sandbox.factory import BOKEH_PACKAGES, load_factory, out

# Where the sandbox finds the pickled variables
VARIABLES_MOUNT = "/eryx-variables"
# Prepended to the code: a throwaway function keeps the loader's names out of the
# sandbox's globals
LOAD_VARIABLES = f"""
def _eryx_load():
    import pickle
    with open("{VARIABLES_MOUNT}/variables.pkl", "rb") as f:
        return pickle.load(f)
globals().update(_eryx_load())
del _eryx_load
"""

CODE = """
import pandas as pd
from bokeh.embed import json_item
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.plotting import figure
from bokeh.transform import factor_cmap

print(df.dtypes)
# pickle keeps the exact dtypes, unlike CSV
assert str(df["day"].dtype) == "datetime64[ns, Asia/Kolkata]", df.dtypes
assert isinstance(df["city"].dtype, pd.CategoricalDtype), df.dtypes
assert str(df["rain_mm"].dtype) == "int64", df.dtypes

cities = [str(c) for c in df["city"].cat.categories]
source = ColumnDataSource(df)
p = figure(title="Daily temperature", x_axis_type="datetime", height=300, width=700)
p.scatter("day", "temp", source=source, size=7, legend_field="city",
          color=factor_cmap("city", "Category10_3", cities))
p.add_tools(HoverTool(tooltips=[("city", "@city"), ("temp", "@temp{0.0}")]))

result = json_item(p, "plot")
"""

PAGE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>eryx dataframe bokeh demo</title>
<script src="https://cdn.bokeh.org/bokeh/release/bokeh-{version}.min.js"></script>
</head>
<body>
<div id="plot"></div>
<script>Bokeh.embed.embed_item({item});</script>
</body>
</html>
"""


def make_weather() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 60
    return pd.DataFrame(
        {
            "day": pd.date_range("2026-10-01", periods=n, freq="D", tz="Asia/Kolkata"),
            "city": pd.Categorical(rng.choice(["Delhi", "Mumbai", "Pune"], n)),
            "temp": (30 + 5 * np.sin(np.linspace(0, 3 * np.pi, n))).round(2),
            "rain_mm": rng.integers(0, 40, n),
        }
    )


def run_with_variables(
    factory: eryx.SandboxFactory, code: str, variables: dict[str, Any], **options: Any
) -> eryx.ExecuteResult:
    """Run `code` in a new sandbox with `variables` defined as its globals.

    The variables are pickled into a temporary directory, mounted read-only at
    VARIABLES_MOUNT for the run and removed here afterwards, whether the code
    succeeded or not. Unpickling in the sandbox is safe for the host, and needs the
    sandbox's pandas and numpy to match the host's (pinned in pyproject.toml).
    `options` go to `factory.create_sandbox`.
    """
    tmp = tempfile.mkdtemp(prefix="eryx-variables-")
    try:
        pathlib.Path(tmp, "variables.pkl").write_bytes(pickle.dumps(variables))
        start = time.perf_counter()
        sandbox = factory.create_sandbox(
            volumes=[(tmp, VARIABLES_MOUNT, True)], **options
        )
        print(f"sandbox created in {(time.perf_counter() - start) * 1000:.1f}ms")
        return sandbox.execute(LOAD_VARIABLES + code)
    finally:
        shutil.rmtree(tmp)


def main() -> None:
    factory = load_factory(
        "bokeh", BOKEH_PACKAGES, imports=["numpy", "pandas", "bokeh.plotting"]
    )
    weather = make_weather()
    print(weather.head(), end="\n\n")

    try:
        result = run_with_variables(
            factory,
            CODE,
            {"df": weather},
            resource_limits=eryx.ResourceLimits(max_memory_bytes=512 * 1024 * 1024),
        )
    except eryx.ExecutionError as e:
        raise SystemExit(f"sandbox error: {e}")
    print(out(result.stdout), end="")
    if result.stderr:
        print(out(result.stderr), end="")
    print(f"took {result.duration_ms:.1f}ms")

    item = result.result
    print("json_item keys:", sorted(item), "/ bokeh", item["version"])
    dest = (
        pathlib.Path(__file__).resolve().parent.parent / "out" / "dataframe-bokeh.html"
    )
    dest.parent.mkdir(exist_ok=True)
    dest.write_text(PAGE.format(version=item["version"], item=json.dumps(item)))
    print(f"plot written to {dest}")


if __name__ == "__main__":
    main()
