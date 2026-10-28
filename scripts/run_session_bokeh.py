"""Plot a host-built DataFrame with bokeh from a snapshotted eryx session.

A preamble session loads the frame from its pickle on a read-only volume
(`eryx_sandbox.factory.shared_frames`) and its state is saved with `snapshot_state()`.
A fresh session restores that snapshot, so the frame is already loaded and the
volume is no longer needed, then runs code cell by cell, REPL style. The last
cell returns bokeh's `json_item` for the plot, which the host embeds in an HTML
page with BokehJS.

The snapshot is pickled by and for the sandbox: the host only stores the bytes
and never unpickles them.
"""

import json
import pathlib
import time

import eryx
from run_dataframe_bokeh import PAGE, make_weather

from eryx_sandbox.factory import BOKEH_PACKAGES, load_factory, out, shared_frames

PREAMBLE = """
import pandas as pd

df = pd.read_pickle("/host/weather.pkl")
"""

CELLS = [
    "df.dtypes",
    """
df["temp_7d"] = df.groupby("city", observed=True)["temp"].transform(
    lambda s: s.rolling(7, min_periods=1).mean()
)
""",
    """
summary = df.groupby("city", observed=True).agg(
    days=("day", "size"), mean_temp=("temp", "mean"), rain_mm=("rain_mm", "sum")
)
summary
""",
    """
from bokeh.embed import json_item
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.palettes import Category10_3
from bokeh.plotting import figure

p = figure(title="Daily temperature, 7-day mean per city", x_axis_type="datetime",
           height=300, width=700)
for (city, group), color in zip(df.groupby("city", observed=True), Category10_3):
    label = f"{city} ({summary.loc[city, 'rain_mm']} mm rain)"
    source = ColumnDataSource(group)
    p.scatter("day", "temp", source=source, size=6, alpha=0.4, color=color)
    p.line("day", "temp_7d", source=source, line_width=2, color=color,
           legend_label=label)
p.add_tools(HoverTool(tooltips=[("city", "@city"), ("day", "@day{%F}"),
                                ("temp", "@temp{0.0}")],
                      formatters={"@day": "datetime"}))

result = json_item(p, "plot")
""",
]

LIMITS = eryx.ResourceLimits(max_memory_bytes=512 * 1024 * 1024)
OUT_DIR = pathlib.Path(__file__).resolve().parent.parent / "out"


def run_cell(session: eryx.Session, code: str) -> eryx.ExecuteResult:
    """Run `code`, echoing it and its output like a REPL.

    A bare expression on the last line is printed, as a REPL would.
    """
    lines = code.strip().splitlines()
    for i, line in enumerate(lines):
        print(("... " if i else ">>> ") + line)
    try:
        compile(lines[-1], "<cell>", "eval")
    except SyntaxError:
        pass
    else:
        lines[-1] = f"print(repr({lines[-1]}))"
    try:
        result = session.execute("\n".join(lines))
    except eryx.ExecutionError as e:
        raise SystemExit(f"sandbox error: {e}")
    print(out(result.stdout), end="")
    if result.stderr:
        print(out(result.stderr), end="")
    print(f"[{result.duration_ms:.1f}ms]\n")
    return result


def make_snapshot(factory: eryx.SandboxFactory) -> bytes:
    """Run PREAMBLE in a session with the frame mounted and snapshot its state."""
    with shared_frames({"weather": make_weather()}) as volumes:
        session = factory.create_session(resource_limits=LIMITS, volumes=volumes)
        session.execute(PREAMBLE)
    return session.snapshot_state()


def main() -> None:
    factory = load_factory(
        "bokeh", BOKEH_PACKAGES, imports=["numpy", "pandas", "bokeh.plotting"]
    )

    start = time.perf_counter()
    snapshot = make_snapshot(factory)
    OUT_DIR.mkdir(exist_ok=True)
    snapshot_path = OUT_DIR / "weather-session.snapshot"
    snapshot_path.write_bytes(snapshot)
    print(
        f"preamble snapshot ({len(snapshot)} bytes) saved to {snapshot_path} "
        f"in {(time.perf_counter() - start) * 1000:.1f}ms"
    )

    # No volumes: the frame comes from the snapshot, not the pickle on the host
    start = time.perf_counter()
    session = factory.create_session(resource_limits=LIMITS)
    session.restore_state(snapshot_path.read_bytes())
    print(f"session restored in {(time.perf_counter() - start) * 1000:.1f}ms\n")

    for code in CELLS:
        result = run_cell(session, code)

    item = result.result
    print("json_item keys:", sorted(item), "/ bokeh", item["version"])
    dest = OUT_DIR / "session-bokeh.html"
    dest.write_text(PAGE.format(version=item["version"], item=json.dumps(item)))
    print(f"plot written to {dest}")


if __name__ == "__main__":
    main()
