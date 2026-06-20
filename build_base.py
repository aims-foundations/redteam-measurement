"""Shared base class for every dataset's build.py.

Each `<dataset>/build.py` subclasses `BenchmarkBuild` and implements only the
parts that genuinely differ between benchmarks:

  * `INFO` / `slug` / `name`  — class attributes (metadata, not logic)
  * `download(self)`          — fetch raw upstream files into `self.raw_dir`
                                (optional; may also download lazily inside
                                `build_rows` for API-walk datasets)
  * `build_rows(self, bench_id)` — parse raw files into a list of response-row
                                dicts (the only non-boilerplate code)

Everything shared by all builds — path setup, the download helper, the GitHub
API JSON helper, benchmark registration, the `ensure_unique_trials` →
trace-split → parquet-write tail, and the `main()` orchestration — lives here so
it is written and verified once.

The six-step pipeline documented in CLAUDE.md maps onto this class as:
  1. INFO dict ................ subclass class attribute
  2. download upstream ....... `download()` / lazy `self._download(...)`
  3. build long-form df ...... `build_rows()` returning row dicts
  4. ensure_unique_trials .... `finalize()` (base)
  5. split traces + write .... `finalize()` (base)
  6. registry save ........... `finalize()` (base)

A response row dict has the canonical columns:
    subject_id, item_id, benchmark_id, trial, test_condition,
    response, correct_answer, trace
`(subject_id, item_id, trial, test_condition)` is the primary key.
"""

import json
import sys
import urllib.request
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from _registry import (  # noqa: E402
    ensure_unique_trials,
    get_benchmark_id,
    register_item,
    resolve_subject,
    save as registry_save,
)

# Re-exported so children can `from build_base import register_item, ...`
# instead of reaching into `_registry` directly.
__all__ = [
    "BenchmarkBuild",
    "register_item",
    "resolve_subject",
    "get_benchmark_id",
    "ensure_unique_trials",
]

# The sidecar trace table's columns. `responses.parquet` carries these too but
# with `trace` nulled out.
TRACE_COLS = ["subject_id", "item_id", "benchmark_id", "trial", "test_condition", "trace"]

_UA_HEADERS = {"User-Agent": "Mozilla/5.0"}


class BenchmarkBuild:
    """Base build pipeline. Subclass and set `INFO`, `slug`, `name`, then
    implement `download` and `build_rows`."""

    # --- subclass-provided metadata -------------------------------------
    INFO: dict = {}
    slug: str = ""   # short benchmark id, e.g. "jailbreakbench"
    name: str = ""   # display name, e.g. "JailbreakBench"

    def __init__(self, benchmark_file: str):
        """`benchmark_file` is the child's `__file__`; the dataset directory and
        all output paths are derived from it."""
        self.dir = Path(benchmark_file).resolve().parent
        self.raw_dir = self.dir / "raw"
        self.contrib_dir = self.dir / "_contrib"
        self.responses_path = self.dir / "responses.parquet"
        self.traces_path = self.dir / "traces.parquet"
        self.raw_dir.mkdir(exist_ok=True)

    # --- shared download helpers ----------------------------------------
    def _download(
        self,
        url: str,
        dest: Path,
        min_size: int = 100,
        timeout: int = 60,
        announce_cache: bool = False,
    ) -> Path:
        """Download `url` to `dest` unless a cached file > `min_size` bytes
        already exists. Returns `dest`."""
        if dest.exists() and dest.stat().st_size > min_size:
            return dest
        req = urllib.request.Request(url, headers=_UA_HEADERS)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(resp.read())
        return dest

    def _get_json(self, url: str, timeout: int = 60) -> object:
        """Fetch and parse JSON from `url` (used for GitHub contents-API walks)."""
        req = urllib.request.Request(url, headers=_UA_HEADERS)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.load(resp)

    # --- registration ----------------------------------------------------
    def register_benchmark(self) -> str:
        """Register this benchmark from `INFO` and return its `benchmark_id`."""
        info = self.INFO
        return get_benchmark_id(
            self.slug,
            name=self.name,
            license=info["license"],
            source_url=info["data_source_url"],
            description=info["description"],
            modality=info["modality"],
            domain=info["domain"],
            response_type=info["response_type"],
            response_scale=info["response_scale"],
            categorical=info["categorical"],
            paper_url=info["paper_url"],
            release_date=info["release_date"],
        )

    # --- subclass hooks --------------------------------------------------
    def download(self) -> None:
        """Fetch raw upstream files. Default no-op: datasets that download
        lazily inside `build_rows` (GitHub API walks) need not override this."""

    def build_rows(self, bench_id: str) -> list[dict]:
        """Parse raw files into a list of response-row dicts. Must be
        implemented by every subclass."""
        raise NotImplementedError

    # --- shared finalize / orchestration --------------------------------
    def finalize(self, rows: list[dict]) -> pd.DataFrame:
        """Steps 4–6: unique trials, split traces out of responses, write both
        parquets, flush the registry contributions. Returns the full df."""
        df = pd.DataFrame(rows)
        df = ensure_unique_trials(df)

        traces = df.loc[df["trace"].notna(), TRACE_COLS].copy()

        resp = df.copy()
        resp["trace"] = None
        resp.to_parquet(self.responses_path, index=False)
        registry_save(self.contrib_dir)
        if len(traces) > 0:
            traces.to_parquet(self.traces_path, index=False)
        return df

    def main(self) -> pd.DataFrame:
        self.download()
        bench_id = self.register_benchmark()
        rows = self.build_rows(bench_id)
        df = self.finalize(rows)
        return df
