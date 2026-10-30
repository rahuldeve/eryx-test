"""The read-only variable store, on the host and in a bare `eryx.Session`."""

import gc
import hashlib
import json
import pathlib
import pickle

import eryx
import pytest

from eryx_sandbox.host_accessor import Host
from eryx_sandbox.variables import STORE_MOUNT, VariableStore


def shared(store: VariableStore) -> pathlib.Path:
    return pathlib.Path(store.volume[0])


def test_layout() -> None:
    with VariableStore({"x": 1, "data": {"a": [1, 2]}}) as store:
        root = shared(store)
        view = json.loads((root / "view.json").read_text())
        assert view["format"] == 1
        assert list(view["entries"]) == ["x", "data"]
        x = view["entries"]["x"]
        data = pickle.dumps(1)
        assert x == {
            "name": "x",
            "version": 1,
            "codec": "pickle",
            "blob": hashlib.sha256(data).hexdigest(),
            "type": "int",
            "size": len(data),
        }
        assert (root / "blobs" / x["blob"]).read_bytes() == data
        assert (root / "host.py").exists()
        assert store.volume == (str(root), STORE_MOUNT, True)
        assert not list(root.rglob(".*.tmp"))


def test_passed_dict_is_copied() -> None:
    variables: dict[str, object] = {"x": 1}
    store = VariableStore(variables)
    variables["x"] = [1, 2, 3]
    variables["y"] = "added later"
    assert store.names() == ["x"]
    assert store.info("x").type == "int"
    assert Host(str(shared(store))).x == 1


@pytest.mark.parametrize("name", ["not-a-name", "class", 1])
def test_bad_names(name: object) -> None:
    with pytest.raises(ValueError, match="Python identifiers"):
        VariableStore({name: 1})  # type: ignore[dict-item]


def test_unpicklable_value() -> None:
    with pytest.raises((pickle.PicklingError, AttributeError)):
        VariableStore({"f": lambda: 0})


def test_close_and_context_manager_remove_the_directory() -> None:
    store = VariableStore({"x": 1})
    root = shared(store).parent
    store.close()
    store.close()
    assert not root.exists()
    with VariableStore({"x": 1}) as store:
        root = shared(store).parent
    assert not root.exists()


def test_garbage_collection_removes_the_directory() -> None:
    store = VariableStore({"x": 1})
    root = shared(store).parent
    del store
    gc.collect()
    assert not root.exists()


def test_identical_values_share_a_blob() -> None:
    with VariableStore({"a": [1, 2], "b": [1, 2]}) as store:
        assert store.info("a").blob == store.info("b").blob
        assert len(list((shared(store) / "blobs").iterdir())) == 1


def test_empty_store() -> None:
    with VariableStore() as store:
        assert store.names() == []
        host = Host(str(shared(store)))
        assert list(host) == []
        assert repr(host) == "host variables: none"


def test_read_is_a_fresh_copy() -> None:
    with VariableStore({"data": [1, 2]}) as store:
        host = Host(str(shared(store)))
        host.data.append(3)
        host["data"].append(4)
        assert host.data == [1, 2]


def test_host_access() -> None:
    with VariableStore({"x": 1, "data": {"a": 1}}) as store:
        host = Host(str(shared(store)))
        assert "x" in host
        assert "y" not in host
        assert list(host) == ["x", "data"]
        assert len(host) == 2
        assert {"x", "data", "info"} <= set(dir(host))
        assert host.info("x")["type"] == "int"
        assert repr(host) == "host variables: x (int), data (dict)"


def test_missing_name_lists_available() -> None:
    with VariableStore({"x": 1}) as store:
        host = Host(str(shared(store)))
        with pytest.raises(AttributeError, match="no host variable 'y'; available: x"):
            _ = host.y
        with pytest.raises(KeyError, match="available: x"):
            _ = host["y"]
        with pytest.raises(AttributeError):
            _ = host._private


def test_host_is_read_only() -> None:
    with VariableStore({"x": 1}) as store:
        host = Host(str(shared(store)))
        with pytest.raises(AttributeError, match="read-only"):
            host.x = 2
        with pytest.raises(AttributeError, match="read-only"):
            del host.x
        with pytest.raises(TypeError, match="read-only"):
            host["x"] = 2


def test_in_a_session() -> None:
    with VariableStore({"x": 1, "data": {"a": [1, 2]}}) as store:
        session = eryx.Session(volumes=[store.volume])
        names = "result = sorted(globals())"
        before = session.execute(names).result
        session.execute(store.setup_code)
        after = session.execute(names).result
        result = session.execute(
            "result = {'x': host.x, 'data': host['data'], 'names': list(host)}"
        ).result
    assert result == {"x": 1, "data": {"a": [1, 2]}, "names": ["x", "data"]}
    # setup leaves only `host` behind
    assert set(after) - set(before) == {"host"}


def test_traceback_points_at_host_py() -> None:
    with VariableStore() as store:
        session = eryx.Session(volumes=[store.volume])
        session.execute(store.setup_code)
        with pytest.raises(eryx.ExecutionError, match=f"{STORE_MOUNT}/host.py"):
            session.execute("host.missing")
