"""Run numpy code inside an eryx sandbox."""

import time

import eryx

from sandbox_env import load_factory, out

CODE = """
import numpy as np

print("numpy", np.__version__)

a = np.arange(12, dtype=np.float64).reshape(3, 4)
print(a)
print("sum:", a.sum(), "mean:", a.mean())
print("a @ a.T =")
print(a @ a.T)

rng = np.random.default_rng(42)
m = rng.standard_normal((4, 4))
print("det:", round(float(np.linalg.det(m)), 6))
print("inv ok:", np.allclose(m @ np.linalg.inv(m), np.eye(4)))
print("fft:", np.round(np.fft.fft([1, 0, -1, 0]).real, 3))

result = {"shape": list(a.shape), "sum": float(a.sum())}
"""


def main() -> None:
    factory = load_factory("numpy", ["numpy-wasi.tar.gz"], imports=["numpy"])

    start = time.perf_counter()
    sandbox = factory.create_sandbox()
    print(f"sandbox created in {(time.perf_counter() - start) * 1000:.1f}ms")

    result = sandbox.execute(CODE)
    print(out(result.stdout), end="")
    print("result:", result.result)
    print(f"took {result.duration_ms:.1f}ms")


if __name__ == "__main__":
    main()
