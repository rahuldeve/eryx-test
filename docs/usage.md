# Using the eryx sandbox: Python, numpy, pandas and bokeh

Notes for agents writing code that runs inside an eryx sandbox built by this
repo: CPython 3.14 on wasm32-wasip2 via pyeryx 0.11, with numpy 2.4.6 and
pandas 2.3.3 compiled by `wasi-build/build_packages.sh`, and bokeh 3.10.1 from
PyPI (`scripts/fetch_packages.py`). Every item below was checked in the
sandbox. Full test-suite results come from
`tests/test_builtin_suites.py` (`uv run poe test-suites`; results in
`test-results/<suite>/summary.txt`).

- [1. Python](#1-python-general-usage-and-limitations): applies to all code
- [2. numpy](#2-numpy)
- [3. pandas](#3-pandas): pandas runs on numpy, so section 2 applies too
- [4. bokeh](#4-bokeh): the bokeh sandbox includes numpy and pandas

---

## 1. Python: general usage and limitations

### 1.1 Crashes are not exceptions

Some bugs in compiled code abort the WebAssembly instance (a "wasm trap")
instead of raising. `try/except` can't catch them: the host's `execute()` call
fails with `ExecutionError: wasm trap: ...`, and all output from that run is
lost. Known triggers are listed under numpy (2.3). Keep separate steps in
separate `execute()` calls if you need partial output to survive.

### 1.2 It's a 32-bit platform

`sys.platform` is `"wasi"`, `platform.machine()` is `"wasm32"`,
`sys.maxsize` is `2**31 - 1`, and `os.cpu_count()` is 1. Python ints are
unaffected, but anything sized by the C pointer width (numpy's default integer,
`Py_ssize_t` limits, address space) is 32-bit. The whole process can never use
more than 4 GiB.

### 1.3 The filesystem is mostly absent

- By default there is **no writable directory at all**, not even `/tmp`
  (`FileNotFoundError`). Files can only be written where the host has mounted a
  volume (`create_sandbox(volumes=[(host_dir, "/tmp", False)])`). Without one,
  keep data in memory (`io.BytesIO`, `io.StringIO`, strings/bytes).
- The working directory is `/`, but `/` itself can't be stat'ed or listed
  (`FileNotFoundError: '/'`). `os.chdir` into a mounted directory first.
- `os.makedirs(path, exist_ok=True)` works. `pathlib.Path.mkdir(parents=True)`
  fails with `OSError: [Errno 29] I/O error` when parents are missing.
- `tempfile` works once `/tmp` is mounted.
- `HOME` is `/tmp`. WASI has no home directory, and without `HOME` (or the
  `pwd` module, 1.4) `Path.home()` raises, so this repo's sandboxes set it at
  startup (`scripts/sandbox_site/sitecustomize.py`). Like the rest of `/tmp`,
  `~` is only writable if the host mounts it.
- Python packages live read-only under `/site-packages` and the stdlib under
  `/python-stdlib`.

### 1.4 Missing standard-library modules

These don't exist: `ctypes`, `mmap`, `bz2`, `lzma`, `fcntl`, `termios`,
`resource`, `pwd`, `grp`, `readline`, `curses`, `tkinter`. Consequences:

- **Compression:** only gzip/zlib and zip work. `.bz2`, `.xz` and `.tar.xz`
  can't be read or written.
- **No `ctypes`:** nothing can call C through ctypes, and any library that
  imports it at module level fails to import.
- **No `mmap`:** no memory-mapped files.

These import but are limited:

- `sqlite3` works **in memory only**. `sqlite3.connect(":memory:")` is fine; a
  file path gives `unable to open database file`, even on a mounted volume.
- `socket`, `ssl`, `urllib` import, but networking is off unless the host
  passes a `NetConfig` (`URLError: networking not enabled for this sandbox`).
- `threading` imports, but threads can't start (next section).

### 1.5 No processes, threads or signals

- `threading.Thread(...).start()` raises `RuntimeError: can't start new thread`,
  so nothing that uses a thread or process pool works (`concurrent.futures`
  executors, `multiprocessing`).
- `subprocess` and `os.system` raise `OSError: wasi does not support
  processes.`. There is no `os.fork` or `os.kill`.
- `signal.alarm` doesn't exist. Use the host's execution timeout instead.
- `time.sleep` works.

### 1.6 Other missing OS functions

- `os.dup` exists but raises `OSError: [Errno 58] Not supported`. This breaks
  anything that duplicates file descriptors, e.g. numpy's `tofile`/`fromfile`
  (section 2.4).
- `os.getuid`, `os.getgid`, `os.umask`, `os.getppid` don't exist. Code that
  calls them, e.g. pytest's `tmp_path`, needs a stub (`os.getuid = lambda: 0`).
- Error numbers are WASI's, not Linux's: a missing file is `[Errno 44]`, not
  `[Errno 2]`. Catch exception types (`FileNotFoundError`, ...), never errno
  numbers or message text.

### 1.7 stdout/stderr are not real files

`sys.stdout` and `sys.stderr` have no `isatty()` or `fileno()`. Libraries that
probe the terminal (progress bars, color output, pytest, `argparse` color in
3.14) raise `AttributeError`. Workarounds: set `os.environ["NO_COLOR"] = "1"`
and add `sys.stdout.isatty = lambda: False`. Output is captured and returned to
the host when the run ends.

### 1.8 Time zones

- The local time zone is UTC (`time.tzname == ('UTC', 'UTC')`).
- There is no system time-zone database. `zoneinfo.ZoneInfo("Asia/Kolkata")`
  only works if the `tzdata` package is installed in the sandbox. The pandas
  and bokeh sandboxes have it; the numpy-only sandbox doesn't.

### 1.9 Resource limits set by the host

The host chooses these when it creates the sandbox. When one is hit, the run
fails with an error rather than slowing down:

- **Memory:** eryx defaults to 128 MiB. This repo's scripts use 512 MiB to
  2 GiB, and wasm32 can never go past 4 GiB.
- **Time:** eryx defaults to a 30 s timeout per `execute()` call.
- Single core, no SIMD, no threads: CPU-heavy code is slower than native.

---

## 2. numpy

Most numpy code just works: array math, broadcasting, indexing, `linalg`,
`fft`, `random`, datetimes, `np.strings`, text I/O, pickling. The problems are
the ones below, roughly in order of how likely they are to bite.

### 2.1 The default integer is 32-bit

Because wasm32 is a 32-bit platform (1.2), `np.intp`, and with it numpy's
default integer, is `int32`:

```python
np.array([1]).dtype  # int32  (int64 on a normal 64-bit machine)
np.arange(3).dtype  # int32
np.array([2**31 - 1]) + 1  # array([-2147483648]), no error or warning
np.full(3, 2**30).sum()  # -1073741824, overflowed silently
```

Integer sums, products, `cumsum` and counters overflow at 2**31 with no error.
Pass `dtype=np.int64` when values can get large (`np.arange(n, dtype=np.int64)`,
`a.sum(dtype=np.int64)`), or use floats. Python ints that don't fit in 32 bits
do give an `int64` array (`np.array([2**40]).dtype` is `int64`).

### 2.2 Floating-point errors are silent

wasm has no floating-point exception flags, so numpy never sees errors:

```python
np.array([1.0]) / 0  # [inf]  no RuntimeWarning
np.sqrt(np.array([-1.0]))  # [nan]  no RuntimeWarning
with np.errstate(all="raise"):
    np.exp(np.array([1000.0]))  # [inf]  does NOT raise FloatingPointError
```

Don't rely on warnings or `np.errstate` to detect bad values. Check the results
explicitly with `np.isfinite`, `np.isnan` or `np.isinf`.

### 2.3 `copy.copy()` of a numpy scalar crashes the sandbox

```python
copy.copy(np.float64(1.5))  # wasm trap: the whole execution dies (see 1.1)
```

This affects every numpy scalar type (`np.float64`, `np.int32`, `np.bool`,
`np.str_`, `np.datetime64`, ...) and is a numpy C bug: a mismatched function
signature that only wasm detects. It also hits scalars taken out of pandas
objects (section 3.5).

These alternatives work: `copy.deepcopy(x)`, `x.copy()`, `x.item()` (to get a
Python number), or just using the scalar as is, since scalars are immutable.
`copy.copy()` of arrays is fine. Watch out for libraries that call `copy.copy`
on values you pass in.

### 2.4 Binary file I/O on real paths fails

numpy's fast binary file I/O duplicates the file descriptor (1.6), so these
raise `OSError: [Errno 58] Not supported`:

| Fails on a file path                         | Works                                                      |
|----------------------------------------------|------------------------------------------------------------|
| `np.save("x.npy", a)`                        | `np.savez` / `np.savez_compressed`, then `np.load("x.npz")` |
| `np.load("x.npy")`                           | `np.load(io.BytesIO(open("x.npy", "rb").read()))`          |
| `a.tofile(path)`                             | `open(path, "wb").write(a.tobytes())`                      |
| `np.fromfile(path, dtype)`                   | `np.frombuffer(open(path, "rb").read(), dtype)`            |
| `np.memmap(...)` (no `mmap`, 1.4)            | read the file into memory                                  |

Text I/O (`np.savetxt`, `np.loadtxt`, `np.genfromtxt`) works, and so does
anything that goes through an in-memory `io.BytesIO` buffer.

### 2.5 Performance

- **No BLAS/LAPACK library.** `linalg` uses numpy's bundled `lapack_lite`, and
  matrix multiplication isn't BLAS-accelerated. Rough timings: a 500x500
  `a @ a` takes ~0.6 s, and `inv` + `eigvals` of a 200x200 matrix ~0.1 s.
  Large dense linear algebra gets slow quickly.
- **No SIMD.** Elementwise math runs as scalar code.
- **`np.longdouble` is 128-bit IEEE quad**, emulated in software: precise but
  slow. It isn't the 80-bit x87 extended type you get on x86 Linux, so
  `np.finfo(np.longdouble)` differs from native.
- **Memory:** a float64 array of 25 million elements is 200 MB. Check sizes
  against the host's memory limit (1.9) before allocating.

### 2.6 Smaller things

- `arr.ctypes` and `numpy.ctypeslib` import, but CPython's `ctypes` module is
  missing (1.4), so anything that actually calls C through ctypes fails.
- `numpy.f2py` can't run (no compiler, no subprocesses), and
  `numpy.distutils` doesn't exist (removed in Python 3.12+).
- numpy's own test suite: 26,542 passed. Every failure traces back to an item
  on this list (fd duplication, `mmap`, `ctypes`, processes, floating-point
  flags, and the scalar-copy crash), apart from one more signature crash in
  a private function only the tests call. A few test modules (`test_multiarray`,
  `test_dtype`, `test_ufunc`, `lib/test_io`) can't be collected at all because
  they import `ctypes`, so those areas have less test coverage here.

---

## 3. pandas

Core pandas works: DataFrame/Series construction, indexing, groupby, merge,
pivot, rolling/resample, datetimes and time zones, categoricals, string
methods, CSV/JSON/pickle I/O, and pickling of any DataFrame. In pandas' own
test suite, 171,360 tests pass. Every failure traces back to an item in
sections 1 to 3 or to something unrelated to the sandbox (3.7). The numpy items that
matter most for pandas code are silent floating-point errors (2.2), the
`copy.copy()` crash (2.3) and the missing filesystem (1.3).

### 3.1 Only CSV, JSON and pickle file formats

Only pure-Python I/O and pandas' own C parser are available. None of pandas'
optional I/O libraries are installed (pyarrow, fastparquet, openpyxl, lxml,
tabulate, matplotlib, ...).

| Works                                                       | Fails                                                       |
|-------------------------------------------------------------|-------------------------------------------------------------|
| `to_csv` / `read_csv` (C and python engines)                | `read_csv(engine="pyarrow")`                                |
| `to_json` / `read_json(io.StringIO(s))`                     | `to_parquet` / `read_parquet`, `to_feather`, `to_orc`       |
| `to_pickle` / `read_pickle`                                 | `to_excel` / `read_excel`                                   |
| `to_html` (writing)                                         | `read_html` (needs lxml)                                    |
| `to_xml` / `read_xml` **with `parser="etree"`**             | `to_xml()` / `read_xml()` with the default lxml parser      |
| `to_sql` / `read_sql` on `sqlite3.connect(":memory:")`      | sqlite on a file path (1.4)                                 |
|                                                             | `to_markdown` (no tabulate), `.plot()` (no matplotlib)      |
|                                                             | `to_clipboard` / `read_clipboard`                           |

Data can also be returned to the caller as strings/bytes (`df.to_csv()` with no
path, `df.to_json()`, `pickle.dumps(df)`) instead of files.

### 3.2 Compression: gzip and zip only

Because `bz2` and `lzma` are missing (1.4) and `zstandard` isn't installed:

- works: `compression="gzip"` / `.gz`, `compression="zip"` / `.zip`
- fails: `.bz2`, `.xz`, `.zst` and `.tar.xz`, with errors like `RuntimeError:
  bz2 module not available`. This includes compression inferred from the file
  extension, so `df.to_csv("out.csv.bz2")` fails too.

### 3.3 No memory-mapped I/O

`read_csv(..., memory_map=True)` raises `OSError: mmap is not available on WASI`.
Drop the argument.

### 3.4 No dataframe interchange protocol

`pd.api.interchange.from_dataframe(...)` and `df.__dataframe__()` consumers
raise `ImportError: ctypes is not available.` The Arrow PyCapsule interface
(`df.__arrow_c_stream__()`) needs pyarrow, which isn't installed.

### 3.5 Behaviour that differs from a normal machine

- **Integer dtypes:** pandas itself defaults to `int64` (`pd.Series([1, 2])`,
  `pd.Index`, `RangeIndex` are all int64), but anything you build with raw
  numpy defaults to `int32` (2.1). Converting between the two can overflow
  silently.
- **No floating-point warnings:** `pd.Series([1.0]) / 0` gives `inf` and
  `np.log` of negative values gives `nan` with no `RuntimeWarning` (2.2).
  Check for bad values with `.isna()` and `np.isfinite`.
- **`copy.copy()` of a numpy scalar taken from a DataFrame crashes the
  sandbox**, e.g. `copy.copy(df["a"].iloc[0])` when the column is
  float/int/bool (2.3). pandas' own scalars (`Timestamp`, `Timedelta`,
  `Period`) are fine. Use `.item()` or `copy.deepcopy()` instead.
- **Time zones:** "local" times and `tz="dateutil/local"` are UTC (1.8). Name
  zones explicitly (`tz="Asia/Kolkata"`). Named zones work because the pandas
  sandbox ships the `tzdata` package.

### 3.6 Performance and memory

Single-threaded and without SIMD, but fine for moderate data: a groupby-mean
over 1,000,000 rows takes ~0.1 s. A DataFrame with one int64 and one float64
column of 1M rows is 12 MB, and operations make temporary copies, so leave
headroom below the host's memory limit (1.9; this repo's scripts give pandas
512 MiB).

### 3.7 Test-suite failures not caused by the sandbox

For anyone reading `test-results/pandas/summary.txt`:

- **20 strict XPASS:** pandas expects some timezone/float tests to fail "on
  32-bit platforms". wasm32 is 32-bit but has a 64-bit `time_t`, so they pass,
  and the strict xfail reports that as a failure.
- **`test_array_inference[data7]`:** `datetime(..., tzinfo=pytz.timezone("CET"))`
  is 42 minutes off. Since tzdata 2024b, `CET` is an alias for
  `Europe/Brussels`, and passing a pytz zone via `tzinfo=` picks up its local
  mean time. This fails natively with pytz 2026.5 too. (Never pass pytz zones
  via `tzinfo=`; use `tz_localize` or `zoneinfo`.)
- **`test_unique_bad_unicode[index]`:** `pd.Index(["\ud83d"] * 2).unique()`
  keeps both copies of a lone UTF-16 surrogate (`Series.unique` is fine). The
  cause hasn't been investigated; it only matters for invalid unicode strings.
- **12 `worker_id` errors:** pytest-xdist isn't installed in the sandbox. The
  test runner now supplies the fixture (these tests then pass or skip), but
  the last full run predates that fix.

---

## 4. bokeh

bokeh is pure Python. The sandbox has bokeh 3.10.1 with its dependencies
jinja2, MarkupSafe, PyYAML, narwhals, packaging and xyzservices, on top of the
pandas sandbox. MarkupSafe and PyYAML are their pure-Python builds, without C
speedups. tornado (bokeh server) and Pillow (image export) are left out.

Building documents works: figures and glyphs (`line`, `scatter`, `vbar`,
`image`, `image_rgba`, `hexbin`, ...), datetime and categorical axes,
`ColumnDataSource` from DataFrames and groupbys, layouts, widgets with
`CustomJS` callbacks, themes (including YAML theme files), map tiles via
xyzservices, and the WebGL backend. numpy scalars as property values (e.g.
`Span(location=df["a"].max())`) are fine; they don't hit the numpy scalar-copy
crash (2.3).

### 4.1 Getting a plot out

There is no browser, display or server, so render the document to HTML or JSON
and hand that to the host:

| Call                                           | Result                                                    |
|------------------------------------------------|-----------------------------------------------------------|
| `file_html(obj, CDN, "title")`                 | a complete HTML page as a string                          |
| `json_item(obj)`                               | a dict for `Bokeh.embed.embed_item` in the host's page    |
| `components(obj)`                              | a `<script>` and a `<div>` to place in an existing page   |
| `save(obj, "/out/plot.html", resources=CDN)`   | the HTML page as a file, on a mounted volume only (1.3)   |

- Return the string in `result` or print it, rather than writing a file, when
  no volume is mounted.
- `show(obj)` writes the file but can't open a browser; it doesn't raise.
- **Resources:** with `CDN`, the page is small (about 15 KB for a simple
  plot) but the viewer's browser loads BokehJS from cdn.bokeh.org. With
  `INLINE`, BokehJS is embedded and the page works offline, but each page is
  about 1.4 MB. Map tiles are fetched by the viewer's browser too, so neither
  needs network access in the sandbox.

### 4.2 Not available

- **bokeh server:** `bokeh serve`, `bokeh.server`, `curdoc()` callbacks and
  `push_notebook` need tornado (`ModuleNotFoundError: No module named
  'tornado'`), which isn't installed. Python callbacks can't run anyway; use
  `CustomJS` for interactivity.
- **Notebook output:** `output_notebook()` needs IPython, which isn't installed.
- **PNG/SVG export:** `export_png`, `export_svg` and `get_screenshot_as_png`
  need selenium, a browser and Pillow. Export the HTML instead.
- **Contours:** `figure.contour` and `contour_data` need contourpy, a C++
  extension that isn't built (`ModuleNotFoundError: No module named
  'contourpy'`).
- **PIL images as glyph data:** Pillow isn't installed. Pass numpy arrays to
  `image` / `image_rgba`.
- **Custom TypeScript extensions:** compiling them needs node.js, which can't
  run (1.5). Built-in models and `CustomJS` work.
- **`bokeh.sampledata`:** the separate `bokeh_sampledata` package isn't
  installed.

### 4.3 Data size

Column data is embedded in the page, so its size grows with the data: a
200,000-point scatter makes a 4 MB page (generated in ~0.5 s). Aggregate or
downsample large data in pandas before plotting (`hexbin` and `image` are
compact ways to show dense data).
