"""Build a bokeh plot inside an eryx sandbox and save it as standalone HTML."""

import pathlib
import tempfile
import time

import eryx
from sandbox_env import BOKEH_PACKAGES, load_factory, out

CODE = """
import bokeh
import numpy as np
import pandas as pd
from bokeh.embed import file_html, json_item
from bokeh.layouts import column
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.plotting import figure
from bokeh.resources import CDN

print("bokeh", bokeh.__version__, "/ pandas", pd.__version__, "/ numpy", np.__version__)

df = pd.DataFrame({
    "day": pd.date_range("2026-10-01", periods=30, freq="D"),
    "temp": 30 + 5 * np.sin(np.linspace(0, 3 * np.pi, 30)),
    "rain_mm": np.random.default_rng(0).gamma(1.5, 4.0, 30).round(1),
})
source = ColumnDataSource(df)

p1 = figure(title="Temperature", x_axis_type="datetime", height=300, width=700)
p1.line("day", "temp", source=source, line_width=2)
p1.scatter("day", "temp", source=source, size=6)
p1.add_tools(HoverTool(tooltips=[("day", "@day{%F}"), ("temp", "@temp{0.0}")],
                       formatters={"@day": "datetime"}))

p2 = figure(title="Rain", x_axis_type="datetime", height=200, width=700, x_range=p1.x_range)
p2.vbar(x="day", top="rain_mm", source=source, width=0.8 * 86_400_000)

layout = column(p1, p2)
html = file_html(layout, CDN, "eryx bokeh demo")
with open("/out/plot.html", "w") as f:
    f.write(html)
print("html bytes:", len(html))

item = json_item(p1)
result = {"html_bytes": len(html), "json_item_keys": sorted(item)}
"""


def main() -> None:
    factory = load_factory(
        "bokeh", BOKEH_PACKAGES, imports=["numpy", "pandas", "bokeh.plotting"]
    )

    with tempfile.TemporaryDirectory() as tmp:
        start = time.perf_counter()
        sandbox = factory.create_sandbox(
            resource_limits=eryx.ResourceLimits(max_memory_bytes=512 * 1024 * 1024),
            volumes=[(tmp, "/out", False)],
        )
        print(f"sandbox created in {(time.perf_counter() - start) * 1000:.1f}ms")

        try:
            result = sandbox.execute(CODE)
        except eryx.ExecutionError as e:
            raise SystemExit(f"sandbox error: {e}")
        print(out(result.stdout), end="")
        if result.stderr:
            print(out(result.stderr), end="")
        print("result:", result.result)
        print(f"took {result.duration_ms:.1f}ms")

        dest = (
            pathlib.Path(__file__).resolve().parent.parent / "out" / "bokeh-demo.html"
        )
        dest.parent.mkdir(exist_ok=True)
        dest.write_text(pathlib.Path(tmp, "plot.html").read_text())
        print(f"plot written to {dest}")


if __name__ == "__main__":
    main()
