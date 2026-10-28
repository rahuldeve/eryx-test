"""End to end: a session snapshot carries a host DataFrame into a new session.

Runs the code from scripts/run_session_bokeh.py: the preamble session loads the
frame from its pickle and is snapshotted, then a session without the volume
restores it and runs the cells, the last of which returns a bokeh `json_item`.
"""

import eryx
import pytest
from run_dataframe_bokeh import make_weather
from run_session_bokeh import CELLS, LIMITS, make_snapshot
from sandbox_env import BOKEH_PACKAGES, load_factory
from test_dataframe_bokeh import find_objects


@pytest.fixture(scope="module")
def factory() -> eryx.SandboxFactory:
    return load_factory(
        "bokeh", BOKEH_PACKAGES, imports=["numpy", "pandas", "bokeh.plotting"]
    )


@pytest.fixture(scope="module")
def snapshot(factory: eryx.SandboxFactory) -> bytes:
    return make_snapshot(factory)


def restore(factory: eryx.SandboxFactory, snapshot: bytes) -> eryx.Session:
    session = factory.create_session(resource_limits=LIMITS)
    session.restore_state(snapshot)
    return session


def test_snapshot_restores_frame_without_volume(
    factory: eryx.SandboxFactory, snapshot: bytes
) -> None:
    session = restore(factory, snapshot)
    result = session.execute("""
import os
result = {
    "mounted": os.path.exists("/host"),
    "dtypes": {k: str(v) for k, v in df.dtypes.items()},
    "rows": len(df),
}
""").result
    assert result == {
        "mounted": False,
        "dtypes": {
            "day": "datetime64[ns, Asia/Kolkata]",
            "city": "category",
            "temp": "float64",
            "rain_mm": "int64",
        },
        "rows": 60,
    }


def test_cells_build_plot_from_restored_session(
    factory: eryx.SandboxFactory, snapshot: bytes
) -> None:
    session = restore(factory, snapshot)
    for code in CELLS:
        result = session.execute(code)
    item = result.result

    assert item["version"].startswith("3.10.")
    (root,) = item["doc"]["roots"]
    assert root["name"] == "Figure"
    assert len(find_objects(item, "ColumnDataSource")) == 3
    # the legend shows the per-city rain total computed in an earlier cell
    rain = make_weather().groupby("city", observed=True)["rain_mm"].sum()
    labels = [li["attributes"]["label"] for li in find_objects(item, "LegendItem")]
    assert labels == [
        {"type": "value", "value": f"{city} ({mm} mm rain)"}
        for city, mm in rain.items()
    ]
