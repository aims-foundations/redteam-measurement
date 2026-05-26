"""Build MHJ long-form responses from the Scale-AI multi-turn jailbreak release.

Data source:
  - https://huggingface.co/datasets/tom-gibbs/multi-turn_jailbreak_attack_datasets
    "Complete Harmful Dataset.csv" — 382 (model, goal, cipher) cells from the
    "How Johnny Can Persuade LLMs to Jailbreak Them" line of work (the Scale
    AI MHJ benchmark for multi-turn jailbreak attacks based on word-mapping
    ciphers).

Each row records:
  - Goal (target harmful instruction) and Goal ID
  - Prompt (the cipher-decoded explanation) and Multi-turn conversation transcript
  - Single-turn conversation (the same goal asked in one turn for ablation)
  - Decoded responses (the model's outputs decoded from the cipher)
  - Model identifier (one of 8 frontier LLMs)
  - Input-cipher: word_mapping_random or word_mapping_perp_filter
  - Jailbroken: dict-string {'Multi-turn': X, 'Single-turn': Y} with ordinal 0/1/2
  - UTQ:        dict-string {'Multi-turn': X, 'Single-turn': Y} where UTQ stands
                for "Understanding The Question" — whether the model decoded the
                cipher correctly

We expand each row into long-form (Jailbroken/UTQ × Multi-turn/Single-turn)
test conditions, so each (model, goal, cipher) cell contributes 4 condition
rows per judge.

Output:
  - raw/Complete_Harmful_Dataset.csv                # cached upstream CSV
  - responses.parquet                               # long-form, no trace
  - traces.parquet                                  # decoded model responses
  - _contrib/{subjects,items,benchmarks}.parquet    # registry contributions
"""

INFO = {
    'description': (
        'MHJ (Multi-turn Jailbreak) cipher-attack evaluation: 8 frontier '
        'LLMs (GPT-3.5/4/4o, Claude 3 Opus/Sonnet/Haiku, Llama 3 8B/70B) '
        'evaluated on 23 harmful goals under 2 word-mapping ciphers, with '
        'both multi-turn and single-turn variants and two judge labels '
        '(Jailbroken severity, Understanding The Question).'
    ),
    'testing_condition': (
        'test_condition is "cipher=<input_cipher>;turn=<multi|single>;'
        'judge=<jailbroken|utq>". Each (model, goal, cipher) cell produces '
        'up to 8 rows.'
    ),
    'paper_url': 'https://arxiv.org/abs/2408.15221',
    'data_source_url': (
        'https://huggingface.co/datasets/tom-gibbs/multi-turn_jailbreak_attack_datasets'
    ),
    'subject_type': 'model',
    'item_type': 'task',
    'license': 'cc-by-4.0',
    'citation': """@inproceedings{gibbs2024mhj,
  title={Multi-Turn Jailbreaks Are More Effective and Stealthier Than Single-Turn Attacks},
  author={Gibbs, Tom and Kosak-Hine, Ethan and Ingebretsen, George and Zhang, Jason and Broomfield, Julian and Pieri, Sara and Iranmanesh, Reihaneh and Rabbany, Reihaneh and Pelrine, Kellin},
  year={2024},
  eprint={2408.15221},
  archivePrefix={arXiv},
  url={https://arxiv.org/abs/2408.15221},
}""",
    'tags': ['safety', 'multi_turn', 'jailbreak', 'cipher_attack', 'red_teaming'],
    'modality': ['text'],
    'domain': ['safety'],
    'response_type': 'ordinal',
    'response_scale': '{0=refusal, 1=partial, 2=full jailbreak}',
    'categorical': True,
    'release_date': '2024-08',
}


import ast
import re
import sys
import urllib.parse
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

CSV_URL = (
    "https://huggingface.co/datasets/tom-gibbs/multi-turn_jailbreak_attack_datasets/"
    "resolve/main/" + urllib.parse.quote("Complete Harmful Dataset.csv")
)
CSV_PATH = RAW_DIR / "Complete_Harmful_Dataset.csv"


def _download(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 1000:
        print(f"  {dest.name} already cached ({dest.stat().st_size:,} bytes)")
        return dest
    print(f"  downloading {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(resp.read())
    print(f"  saved {dest.name} ({dest.stat().st_size:,} bytes)")
    return dest


def _parse_dict_label(value):
    """Parse strings like \"{'Multi-turn': 1, 'Single-turn': 2}\" or with no
    space after colon. Returns (multi, single) ordinal floats, with None for
    unparseable parts."""
    if pd.isna(value):
        return (None, None)
    s = str(value)
    # Normalize "{'Multi-turn':1, 'Single-turn': 1}" → standard literal-eval.
    s_norm = re.sub(r":\s*", ": ", s)
    try:
        d = ast.literal_eval(s_norm)
        if not isinstance(d, dict):
            return (None, None)
        m = d.get("Multi-turn")
        si = d.get("Single-turn")
        m = float(m) if m is not None else None
        si = float(si) if si is not None else None
        return (m, si)
    except (ValueError, SyntaxError):
        return (None, None)


def build_long_form() -> pd.DataFrame:
    bench_id = get_benchmark_id(
        "mhj",
        name="MHJ (Multi-turn Human Jailbreaks)",
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

    _download(CSV_URL, CSV_PATH)
    df = pd.read_csv(CSV_PATH)

    rows: list[dict] = []
    skipped = 0
    for _, r in df.iterrows():
        goal = str(r["Goal"])
        item = register_item(
            benchmark_id=bench_id,
            raw_item_id=str(r["Goal ID"]),
            content=goal,
        )
        subj = resolve_subject(str(r["Model"]))
        cipher = str(r.get("Input-cipher", "unknown"))
        decoded = r.get("Decoded responses")
        trace_text = str(decoded) if isinstance(decoded, str) else None

        jb_multi, jb_single = _parse_dict_label(r.get("Jailbroken"))
        utq_multi, utq_single = _parse_dict_label(r.get("UTQ"))

        cells = [
            ("multi", "jailbroken", jb_multi),
            ("single", "jailbroken", jb_single),
            ("multi", "utq",        utq_multi),
            ("single", "utq",       utq_single),
        ]
        for turn, judge, value in cells:
            if value is None:
                skipped += 1
                continue
            rows.append({
                "subject_id": subj,
                "item_id": item,
                "benchmark_id": bench_id,
                "trial": 1,
                "test_condition": f"cipher={cipher};turn={turn};judge={judge}",
                "response": value,
                "correct_answer": None,
                # Only stash the trace once per (subject, item, cipher) to avoid
                # 4x duplication; pin it to the jailbroken-multi row.
                "trace": (trace_text if (turn == "multi" and judge == "jailbroken") else None),
            })

    if skipped:
        print(f"  WARNING: skipped {skipped} unparseable label cells")

    df_out = pd.DataFrame(rows)
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
    print(f"  test_conditions: {df['test_condition'].nunique()}")
    jb = df[df["test_condition"].str.contains("judge=jailbroken")]
    print(f"  mean jailbroken score (0=refusal, 2=full): {jb['response'].mean():.3f}")


def main() -> None:
    print(f"[mhj] building from {_BENCHMARK_DIR}")
    df = build_long_form()
    print_stats(df)


if __name__ == "__main__":
    main()
