"""Plot a host-built DataFrame with bokeh inside an eryx sandbox.

The frame reaches the sandbox as a pickle on a read-only volume
(`eryx_sandbox.factory.shared_frames`). The sandbox returns bokeh's `json_item` for the
plot, which the host embeds in an HTML page with BokehJS.
"""

import json
import pathlib
import time

import eryx
import numpy as np
import pandas as pd

from eryx_sandbox.factory import BOKEH_PACKAGES, load_factory, out, shared_frames

CODE = """
import pandas as pd
from bokeh.embed import json_item
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.plotting import figure
from bokeh.transform import factor_cmap

df = pd.read_pickle("/host/weather.pkl")
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


def main() -> None:
    factory = load_factory(
        "bokeh", BOKEH_PACKAGES, imports=["numpy", "pandas", "bokeh.plotting"]
    )
    weather = make_weather()
    print(weather.head(), end="\n\n")

    with shared_frames({"weather": weather}) as volumes:
        start = time.perf_counter()
        sandbox = factory.create_sandbox(
            resource_limits=eryx.ResourceLimits(max_memory_bytes=512 * 1024 * 1024),
            volumes=volumes,
        )
        print(f"sandbox created in {(time.perf_counter() - start) * 1000:.1f}ms")

        try:
            result = sandbox.execute(CODE)
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
