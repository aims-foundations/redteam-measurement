"""Build StrongREJECT long-form responses from the paper's released eval logs.

Data source (hosted on OSF, linked from the dsbowen/strong_reject Makefile):
  - https://osf.io/download/vyt73/   evals.json
      JSONL with one record per (model, jailbreak, forbidden_prompt) cell.
      4 target models x 38 jailbreak methods x 313 forbidden prompts = 47,576
      observations. Composite `score` in {0, 1/8, ..., 1} from the
      `strongreject_rubric` evaluator (rubric: refusal x avg(specificity,
      convincingness)).

Unlike JailbreakBench and HarmBench (binary `jailbroken` labels), the
response is a **continuous** rubric score, which makes this dataset the
polytomous/continuous slice useful for fitting non-dichotomous IRT models.

Output:
  - raw/evals.jsonl                                # cached upstream JSONL
  - responses.parquet                              # long-form, no trace column
  - traces.parquet                                 # raw model generations
  - _contrib/{subjects,items,benchmarks}.parquet   # registry contributions
"""

INFO = {
    'description': (
        'StrongREJECT rubric scores across 4 target models x 38 jailbreak '
        'methods x 313 forbidden prompts (47,576 cells, fully dense). '
        'Response is the StrongREJECT composite score: '
        '(1 - refusal) * (specificity + convincingness - 2) / 8 in [0, 1].'
    ),
    'testing_condition': (
        'test_condition is "attack=<jailbreak>" — each of the 38 jailbreak '
        'recipes (pair, base64, translation_zulu, gcg_transfer_harmbench, '
        'auto_payload_splitting, evil_confidant, ...) is a separate condition. '
        'The judge (gpt-4o-mini, strongreject_rubric) is held constant across '
        'rows, so it is not part of the condition string.'
    ),
    'paper_url': 'https://arxiv.org/abs/2402.10260',
    'data_source_url': 'https://github.com/dsbowen/strong_reject',
    'subject_type': 'model',
    'item_type': 'task',
    'license': 'MIT',
    'citation': """@inproceedings{souly2024strongreject,
  title={A StrongREJECT for Empty Jailbreaks},
  author={Souly, Alexandra and Lu, Qingyuan and Bowen, Dillon and Trinh, Tu and Hsieh, Elvis and Pandey, Sana and Abbeel, Pieter and Svegliato, Justin and Emmons, Scott and Watkins, Olivia and Toyer, Sam},
  booktitle={NeurIPS},
  year={2024},
  url={https://arxiv.org/abs/2402.10260},
}""",
    'tags': ['safety', 'jailbreak', 'red_teaming', 'rubric_judge'],
    'modality': ['text'],
    'domain': ['safety'],
    'response_type': 'continuous_bounded',
    'response_scale': '{0, 1/8, 2/8, ..., 1} (StrongREJECT rubric composite)',
    'categorical': False,
    'release_date': '2024-02',
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

EVALS_URL = "https://osf.io/download/vyt73/"
EVALS_PATH = RAW_DIR / "evals.jsonl"


def _download(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 1000:
        print(f"  {dest.name} already cached ({dest.stat().st_size:,} bytes)")
        return dest
    print(f"  downloading {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(resp.read())
    print(f"  saved {dest.name} ({dest.stat().st_size:,} bytes)")
    return dest


def iter_records(path: Path):
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def build_long_form() -> pd.DataFrame:
    bench_id = get_benchmark_id(
        "strongreject",
        name="StrongREJECT",
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
    skipped = 0

    for rec in iter_records(EVALS_PATH):
        prompt = rec.get("forbidden_prompt")
        model = rec.get("model")
        attack = rec.get("jailbreak")
        score = rec.get("score")
        if prompt is None or model is None or attack is None or score is None:
            skipped += 1
            continue

        subj = resolve_subject(model)
        item = register_item(
            benchmark_id=bench_id,
            raw_item_id=f"{rec.get('source','?')}::{prompt[:80]}",
            content=prompt,
        )

        # Encode attack prompt + raw model response into one JSON trace blob.
        trace_payload = {
            "jailbroken_prompt": rec.get("jailbroken_prompt"),
            "response": rec.get("response"),
            "refusal": rec.get("refusal"),
            "convincingness": rec.get("convincingness"),
            "specificity": rec.get("specificity"),
            "judge_model": rec.get("judge_model"),
            "evaluator": rec.get("evaluator"),
            "category": rec.get("category"),
            "source": rec.get("source"),
        }

        rows.append({
            "subject_id": subj,
            "item_id": item,
            "benchmark_id": bench_id,
            "trial": 1,
            "test_condition": f"attack={attack}",
            "response": float(score),
            "correct_answer": None,
            "trace": json.dumps(trace_payload, ensure_ascii=False),
        })

    if skipped:
        print(f"  WARNING: skipped {skipped} records with missing fields")

    df = pd.DataFrame(rows)
    df = ensure_unique_trials(df)

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
    print(f"  response (StrongREJECT score) mean/std: "
          f"{df['response'].mean():.3f} / {df['response'].std():.3f}")


def main() -> None:
    print(f"[strongreject] building from {_BENCHMARK_DIR}")
    _download(EVALS_URL, EVALS_PATH)
    df = build_long_form()
    print_stats(df)


if __name__ == "__main__":
    main()
