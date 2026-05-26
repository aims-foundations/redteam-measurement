"""Build JailbreakBench long-form responses from the public artifacts repo.

Data source:
  - https://github.com/JailbreakBench/artifacts
    Six (attack_method, attack_type) conditions x 4 target models x 100
    JBB-Behaviors. Each artifact JSON embeds (goal, behavior, category) and
    a per-(model, behavior) `jailbroken` boolean verdict from the
    JailbreakBench judge.

Output:
  - raw/<attack>/<attack_type>/<model>.json   # cached upstream JSON
  - responses.parquet                          # long-form, no trace column
  - traces.parquet                             # (subject, item, ..., trace) sidecar
  - _contrib/{subjects,items,benchmarks}.parquet  # registry contributions
"""

INFO = {
    'description': (
        'JailbreakBench artifact verdicts: per-(model, behavior, attack-method) '
        'jailbroken booleans from the JailbreakBench judge. Six (attack_method, '
        'attack_type) conditions across 4 target models and 100 harmful '
        'behaviors. Sparse: white-box attacks (DSN, GCG/white_box) only cover '
        'open-weight models.'
    ),
    'testing_condition': (
        'Each (attack_method, attack_type) pair is registered as a distinct '
        'test_condition string of the form '
        '"attack=<method>;attack_type=<type>". Items are the harmful behavior '
        'goal prompts; conditions encode the adaptive attacker.'
    ),
    'paper_url': 'https://arxiv.org/abs/2404.01318',
    'data_source_url': 'https://github.com/JailbreakBench/artifacts',
    'subject_type': 'model',
    'item_type': 'task',
    'license': 'MIT',
    'citation': """@inproceedings{chao2024jailbreakbench,
  title={JailbreakBench: An Open Robustness Benchmark for Jailbreaking Large Language Models},
  author={Chao, Patrick and Debenedetti, Edoardo and Robey, Alexander and Andriushchenko, Maksym and Croce, Francesco and Sehwag, Vikash and Dobriban, Edgar and Flammarion, Nicolas and Pappas, George J. and Tramèr, Florian and Hassani, Hamed and Wong, Eric},
  booktitle={NeurIPS Datasets and Benchmarks Track},
  year={2024},
  url={https://arxiv.org/abs/2404.01318},
}""",
    'tags': ['safety', 'jailbreak', 'red_teaming', 'adaptive_attack'],
    'modality': ['text'],
    'domain': ['safety'],
    'response_type': 'binary',
    'response_scale': '{0, 1}',
    'categorical': True,
    'release_date': '2024-04',
}


import json
import sys
import urllib.request
from pathlib import Path

import pandas as pd

_BENCHMARK_DIR = Path(__file__).resolve().parent
RAW_DIR = _BENCHMARK_DIR / "raw"
CONTRIB_DIR = _BENCHMARK_DIR / "_contrib"
RESPONSES_PATH = _BENCHMARK_DIR / "responses.parquet"
TRACES_PATH = _BENCHMARK_DIR / "traces.parquet"

RAW_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(_BENCHMARK_DIR.parent))
from _registry import (  # noqa: E402
    ensure_unique_trials,
    get_benchmark_id,
    register_item,
    resolve_subject,
    save as registry_save,
)

ARTIFACTS_API = (
    "https://api.github.com/repos/JailbreakBench/artifacts/contents/attack-artifacts"
)
ARTIFACTS_RAW = (
    "https://raw.githubusercontent.com/JailbreakBench/artifacts/main/attack-artifacts"
)


def _download(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 100:
        return dest
    print(f"  downloading {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(resp.read())
    return dest


def _list_api(url: str) -> list[dict]:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.load(resp)


def discover_artifacts() -> list[tuple[str, str, str]]:
    """Walk the artifacts repo and return (method, attack_type, model_filename)."""
    out: list[tuple[str, str, str]] = []
    for entry in _list_api(ARTIFACTS_API):
        if entry["type"] != "dir" or entry["name"].startswith(".") or entry["name"] == "test-artifact":
            continue
        method = entry["name"]
        for sub in _list_api(entry["url"]):
            if sub["type"] != "dir":
                continue
            attack_type = sub["name"]
            for f in _list_api(sub["url"]):
                if f["type"] == "file" and f["name"].endswith(".json"):
                    out.append((method, attack_type, f["name"]))
    return out


def load_artifact(method: str, attack_type: str, fname: str) -> dict:
    dest = RAW_DIR / method / attack_type / fname
    url = f"{ARTIFACTS_RAW}/{method}/{attack_type}/{fname}"
    _download(url, dest)
    return json.loads(dest.read_text())


def build_long_form(artifacts: list[tuple[str, str, str]]) -> pd.DataFrame:
    bench_id = get_benchmark_id(
        "jailbreakbench",
        name="JailbreakBench",
        license=INFO["license"],
        source_url=INFO["data_source_url"],
        description=INFO["description"],
        modality=INFO["modality"],
        domain=INFO["domain"],
        response_type=INFO["response_type"],
        response_scale=INFO["response_scale"],
        categorical=INFO["categorical"],
        paper_url=INFO["paper_url"],
        release_date=INFO["release_date"],
    )

    rows: list[dict] = []
    missing_verdict = 0

    for method, attack_type, fname in artifacts:
        try:
            data = load_artifact(method, attack_type, fname)
        except Exception as e:
            print(f"  skip {method}/{attack_type}/{fname}: {e}")
            continue

        params = data.get("parameters", {})
        raw_model = params.get("model") or fname.removesuffix(".json")
        subj = resolve_subject(raw_model)
        condition = f"attack={method};attack_type={attack_type}"

        for jb in data.get("jailbreaks", []):
            goal = jb.get("goal")
            if goal is None:
                continue
            jailbroken = jb.get("jailbroken")
            if jailbroken is None:
                missing_verdict += 1
                continue

            item = register_item(
                benchmark_id=bench_id,
                raw_item_id=str(jb.get("index")),
                content=goal,
            )
            rows.append({
                "subject_id": subj,
                "item_id": item,
                "benchmark_id": bench_id,
                "trial": 1,
                "test_condition": condition,
                "response": float(bool(jailbroken)),
                "correct_answer": None,
                "trace": jb.get("response"),
            })

    if missing_verdict:
        print(f"  WARNING: {missing_verdict} (model, behavior) cells had no verdict")

    df = pd.DataFrame(rows)
    df = ensure_unique_trials(df)

    # Split traces from responses.
    trace_cols = ["subject_id", "item_id", "benchmark_id", "trial", "test_condition", "trace"]
    traces = df.loc[df["trace"].notna(), trace_cols].copy()

    resp = df.copy()
    resp["trace"] = None
    resp.to_parquet(RESPONSES_PATH, index=False)
    registry_save(CONTRIB_DIR)
    print(f"  wrote {RESPONSES_PATH.name} ({len(resp):,} rows)")
    print(f"  wrote {CONTRIB_DIR.name}/{{subjects,items,benchmarks}}.parquet")
    if len(traces) > 0:
        traces.to_parquet(TRACES_PATH, index=False)
        print(f"  wrote {TRACES_PATH.name} ({len(traces):,} rows)")
    return df


def print_stats(df: pd.DataFrame) -> None:
    print(f"\n  subjects: {df['subject_id'].nunique()}")
    print(f"  items:    {df['item_id'].nunique()}")
    print(f"  rows:     {len(df):,}")
    print(f"  test_conditions: {df['test_condition'].nunique()}")
    print(f"  response mean (overall jailbreak rate): {df['response'].mean():.3f}")


def main() -> None:
    print(f"[jailbreakbench] building from {_BENCHMARK_DIR}")
    print("  discovering artifacts...")
    artifacts = discover_artifacts()
    print(f"  found {len(artifacts)} (method, attack_type, model) triples")
    df = build_long_form(artifacts)
    print_stats(df)


if __name__ == "__main__":
    main()
