"""End to end: a DataFrame built on the host is plotted with bokeh in a sandbox.

Runs the code from scripts/run_dataframe_bokeh.py: the frame reaches the
sandbox as a pickle on a read-only volume (`sandbox_env.shared_frames`), and the
sandbox returns bokeh's `json_item` for the plot, which is checked against the
original frame.
"""

import base64
import gzip

import eryx
import numpy as np
import pandas as pd
import pytest
from run_dataframe_bokeh import CODE, make_weather
from sandbox_env import BOKEH_PACKAGES, load_factory, shared_frames


@pytest.fixture(scope="module")
def factory() -> eryx.SandboxFactory:
    return load_factory(
        "bokeh", BOKEH_PACKAGES, imports=["numpy", "pandas", "bokeh.plotting"]
    )


@pytest.fixture
def weather() -> pd.DataFrame:
    return make_weather()


def find_objects(node: object, name: str) -> list[dict]:
    """Return every serialized bokeh model called `name` in a json_item tree."""
    found = []
    if isinstance(node, dict):
        if node.get("type") == "object" and node.get("name") == name:
            found.append(node)
        for value in node.values():
            found.extend(find_objects(value, name))
    elif isinstance(node, list):
        for value in node:
            found.extend(find_objects(value, name))
    return found


def decode_column(value: object) -> np.ndarray:
    """Decode a ColumnDataSource column from bokeh's JSON serialization."""
    if isinstance(value, dict) and value.get("type") == "ndarray":
        array = value["array"]
        if isinstance(array, list):
            return np.asarray(array)
        raw = base64.b64decode(array["data"])
        if raw[:2] == b"\x1f\x8b":
            raw = gzip.decompress(raw)
        return np.frombuffer(raw, dtype=value["dtype"]).reshape(value["shape"])
    return np.asarray(value)


def test_bokeh_plot_from_host_dataframe(
    factory: eryx.SandboxFactory, weather: pd.DataFrame
) -> None:
    with shared_frames({"weather": weather}) as volumes:
        sandbox = factory.create_sandbox(
            resource_limits=eryx.ResourceLimits(max_memory_bytes=512 * 1024 * 1024),
            volumes=volumes,
        )
        item = sandbox.execute(CODE).result

    assert item["target_id"] == "plot"
    assert item["version"].startswith("3.10.")
    (root,) = item["doc"]["roots"]
    assert root["name"] == "Figure"
    assert root["id"] == item["root_id"]

    (source,) = find_objects(item, "ColumnDataSource")
    data = {k: decode_column(v) for k, v in source["attributes"]["data"]["entries"]}
    # bokeh sends datetimes as float milliseconds since the epoch (UTC)
    expected_ms = weather["day"].dt.tz_convert("UTC").astype("int64") // 10**6
    np.testing.assert_array_equal(data["day"], expected_ms.to_numpy(float))
    np.testing.assert_array_equal(data["temp"], weather["temp"])
    np.testing.assert_array_equal(data["rain_mm"], weather["rain_mm"])
    assert list(data["city"]) == list(weather["city"].astype(str))
    assert list(data["index"]) == list(weather.index)

    (cmap,) = find_objects(item, "CategoricalColorMapper")
    assert cmap["attributes"]["factors"] == ["Delhi", "Mumbai", "Pune"]
