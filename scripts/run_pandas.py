"""Run pandas code inside an eryx sandbox."""

import time

import eryx
from sandbox_env import PANDAS_PACKAGES, load_factory, out

CODE = """
import pandas as pd
import numpy as np

print("pandas", pd.__version__, "/ numpy", np.__version__)

df = pd.DataFrame({
    "city": ["Pune", "Delhi", "Pune", "Mumbai", "Delhi", "Mumbai"],
    "temp": [31.5, 38.2, 29.8, 33.1, 40.0, 32.4],
    "rain_mm": [0, 2, 12, 30, 0, 18],
})
print(df)
print()
print(df.groupby("city").agg(mean_temp=("temp", "mean"), total_rain=("rain_mm", "sum")))
print()

ts = pd.Series(np.arange(6.0), index=pd.date_range("2026-01-01", periods=6, freq="D"))
print(ts.rolling(3).mean())
print()
print(pd.Timestamp("2026-10-27 12:00", tz="Asia/Kolkata").tz_convert("UTC"))

print(df.describe().loc[["mean", "max"]])

result = df.groupby("city")["temp"].max().to_dict()
"""


def main() -> None:
    factory = load_factory("pandas", PANDAS_PACKAGES, imports=["numpy", "pandas"])

    start = time.perf_counter()
    sandbox = factory.create_sandbox(
        resource_limits=eryx.ResourceLimits(max_memory_bytes=512 * 1024 * 1024)
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


if __name__ == "__main__":
    main()
