"""Build XSTest long-form responses from Röttger et al.'s release.

Data source:
  - https://github.com/paul-rottger/xstest
    The XSTest v2 benchmark of 250 safe + 200 unsafe prompts designed to
    measure exaggerated safety (over-refusal) behaviour in LLMs. For each
    of 5 target models, two CSV files contain the same 450 prompts and
    completions but with different judge labels appended:
      evaluation/automated_evaluation_labels/xstest_v2_completions_<model>_gpteval.csv
      evaluation/automated_evaluation_labels/xstest_v2_completions_<model>_streval.csv

Columns:
  - id, type, prompt, completion
  - annotation_1, annotation_2  : two independent human annotators
  - agreement                   : did annotators agree
  - final_label                 : consensus label
  - gpt4_label / strmatch_label : automated judge labels

Labels are categorical:
    1_full_compliance, 2_full_refusal, 3_partial_refusal (sometimes seen),
    4_other (rare). We emit these as an ordinal `response` ∈ {1, 2, 3, 4}
    (lower = more willing to answer).

This dataset measures the *over-refusal* axis: unlike JBB/HarmBench/SR/DNA
(which ask "did the model comply with harm?"), XSTest asks "did the model
unnecessarily refuse benign-but-sensitive queries?". Item `type` records
whether the prompt is "safe" (e.g. homonyms like "how to kill a Python
process") or "unsafe".

Output:
  - raw/xstest_v2_completions_<model>_<judge>.csv   # cached upstream CSVs
  - responses.parquet                               # long-form, no trace
  - traces.parquet                                  # model completions
  - _contrib/{subjects,items,benchmarks}.parquet    # registry contributions
"""

INFO = {
    'description': (
        'XSTest v2 exaggerated-safety benchmark: 450 prompts (250 safe, 200 '
        'unsafe) across 10 categories x 5 target models (GPT-4, Llama-2 '
        'orig/new system prompts, Mistral instruct/guard), with 4 judge '
        'labels per (model, prompt) cell: two human annotators, a GPT-4 '
        'judge, and a string-match heuristic.'
    ),
    'testing_condition': (
        'test_condition is "judge=<source>" where source is one of: '
        'annotator_1, annotator_2, final (human consensus), gpt4, strmatch. '
        'No attack wrapper is applied. The same model completion is scored '
        'under each judge condition.'
    ),
    'paper_url': 'https://arxiv.org/abs/2308.01263',
    'data_source_url': 'https://github.com/paul-rottger/xstest',
    'subject_type': 'model',
    'item_type': 'task',
    'license': 'CC-BY-4.0',
    'citation': """@inproceedings{rottger2024xstest,
  title={{XSTest}: A Test Suite for Identifying Exaggerated Safety Behaviours in Large Language Models},
  author={Röttger, Paul and Kirk, Hannah Rose and Vidgen, Bertie and Attanasio, Giuseppe and Bianchi, Federico and Hovy, Dirk},
  booktitle={NAACL},
  year={2024},
  url={https://arxiv.org/abs/2308.01263},
}""",
    'tags': ['safety', 'over_refusal', 'exaggerated_safety', 'no_attack'],
    'modality': ['text'],
    'domain': ['safety'],
    'response_type': 'ordinal',
    'response_scale': '{1=full_compliance, 2=full_refusal, 3=partial_refusal, 4=other}',
    'categorical': True,
    'release_date': '2023-08',
}


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

BASE = (
    "https://raw.githubusercontent.com/paul-rottger/xstest/main/"
    "evaluation/automated_evaluation_labels"
)

# CSV-suffix → (display model name, judge suffix → judge condition key)
MODEL_KEYS = {
    "gpt4": "gpt-4",
    "llama2new": "llama-2-7b-chat (new system prompt)",
    "llama2orig": "llama-2-7b-chat (original system prompt)",
    "mistralguard": "mistral-7b-instruct-v0.1 (with-guard system prompt)",
    "mistralinstruct": "mistral-7b-instruct-v0.1",
}

# Map label strings to ordinal codes.
LABEL_MAP = {
    "1_full_compliance": 1,
    "2_full_refusal": 2,
    "3_partial_refusal": 3,
    "4_other": 4,
}


def _download(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 500:
        return dest
    print(f"  downloading {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(resp.read())
    return dest


def _ordinal(value):
    if pd.isna(value):
        return None
    key = str(value).strip()
    if key in LABEL_MAP:
        return float(LABEL_MAP[key])
    # Some labels are bare keys like "1" or unexpected strings; try numeric.
    try:
        return float(int(key.split("_")[0]))
    except (ValueError, IndexError):
        return None


def build_long_form() -> pd.DataFrame:
    bench_id = get_benchmark_id(
        "xstest",
        name="XSTest",
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

    for model_key, display in MODEL_KEYS.items():
        subj = resolve_subject(display)

        # GPT-4 judge CSV (also carries human annotations and the consensus label).
        gpt_csv = RAW_DIR / f"xstest_v2_completions_{model_key}_gpteval.csv"
        _download(f"{BASE}/xstest_v2_completions_{model_key}_gpteval.csv", gpt_csv)
        df_gpt = pd.read_csv(gpt_csv)
        # String-match judge CSV (same rows, different machine label column).
        str_csv = RAW_DIR / f"xstest_v2_completions_{model_key}_streval.csv"
        _download(f"{BASE}/xstest_v2_completions_{model_key}_streval.csv", str_csv)
        df_str = pd.read_csv(str_csv)

        # Sanity: same id ordering — merge on id.
        df = df_gpt.merge(
            df_str[["id", "strmatch_label"]],
            on="id",
            how="left",
            suffixes=("", "_str"),
        )

        for _, r in df.iterrows():
            prompt = str(r["prompt"])
            item = register_item(
                benchmark_id=bench_id,
                raw_item_id=str(r["id"]),
                content=prompt,
            )
            completion = r.get("completion")
            trace = (
                str(completion) if isinstance(completion, str) else None
            )

            judges = {
                "annotator_1": r.get("annotation_1"),
                "annotator_2": r.get("annotation_2"),
                "final":       r.get("final_label"),
                "gpt4":        r.get("gpt4_label"),
                "strmatch":    r.get("strmatch_label"),
            }
            for judge, raw_label in judges.items():
                ordinal = _ordinal(raw_label)
                if ordinal is None:
                    skipped += 1
                    continue
                rows.append({
                    "subject_id": subj,
                    "item_id": item,
                    "benchmark_id": bench_id,
                    "trial": 1,
                    "test_condition": f"judge={judge}",
                    "response": ordinal,
                    "correct_answer": None,
                    "trace": trace if judge == "final" else None,  # avoid 5x duplication
                    "item_type": str(r.get("type", "")),
                })

    if skipped:
        print(f"  WARNING: skipped {skipped} cells with missing/unparseable labels")

    df_out = pd.DataFrame(rows).drop(columns=["item_type"])
    df_out = ensure_unique_trials(df_out)

    trace_cols = ["subject_id", "item_id", "benchmark_id", "trial", "test_condition", "trace"]
    traces = df_out.loc[df_out["trace"].notna(), trace_cols].copy()

    resp = df_out.copy()
    resp["trace"] = None
    resp.to_parquet(RESPONSES_PATH, index=False)
    registry_save(CONTRIB_DIR)
    print(f"  wrote {RESPONSES_PATH.name} ({len(resp):,} rows)")
    print(f"  wrote {CONTRIB_DIR.name}/{{subjects,items,benchmarks}}.parquet")
    if len(traces) > 0:
        traces.to_parquet(TRACES_PATH, index=False)
        print(f"  wrote {TRACES_PATH.name} ({len(traces):,} rows)")
    return df_out


def print_stats(df: pd.DataFrame) -> None:
    print(f"\n  subjects: {df['subject_id'].nunique()}")
    print(f"  items:    {df['item_id'].nunique()}")
    print(f"  rows:     {len(df):,}")
    print(f"  test_conditions (judges): {df['test_condition'].nunique()}")
    final = df[df["test_condition"] == "judge=final"]
    if len(final):
        comp = (final["response"] == 1).mean()
        refuse = (final["response"] == 2).mean()
        print(f"  consensus full-compliance rate: {comp:.1%}")
        print(f"  consensus full-refusal rate:    {refuse:.1%}")


def main() -> None:
    print(f"[xstest] building from {_BENCHMARK_DIR}")
    df = build_long_form()
    print_stats(df)


if __name__ == "__main__":
    main()
