"""Build HarmBench long-form responses from the HarmBench website's data repo.

Data sources (the same files the playground at https://www.harmbench.org/ fetches):
  - https://raw.githubusercontent.com/justinphan3110cais/harmbench_website/data/
        playground_data/metadata_text.json       # methods + models manifest
        playground_data/text_behaviors.json      # behavior text + categories
        playground_data/standard/<METHOD>.json   # per-attack results
        playground_data/contextual/<METHOD>.json # contextual-attack results

Layout of each <METHOD>.json:
    {behavior_id: {model_id: {test_case, generation, label, advbench_label}}}

`label` is the HarmBench classifier verdict (Llama-2-13B fine-tuned judge).
`advbench_label` is the AdvBench refusal-string heuristic. We emit both as
separate `test_condition`s so the paper's measurement model can compare
judges as well as attackers.

Output:
  - raw/<category>/<METHOD>.json, raw/metadata_text.json, raw/text_behaviors.json
  - responses.parquet                              # long-form, no trace
  - traces.parquet                                 # generations + adversarial test_cases
  - _contrib/{subjects,items,benchmarks}.parquet   # registry contributions
"""

INFO = {
    'description': (
        'HarmBench standardized red-teaming results: ~29 target models x 140 '
        'behaviors x 16 attack methods, with the HarmBench Llama-2-13B '
        'classifier verdict and the AdvBench refusal-string heuristic as two '
        'separate judges. Covers the standard and contextual behavior '
        'categories from the HarmBench website playground.'
    ),
    'testing_condition': (
        'test_condition is "attack=<method>;category=<standard|contextual>;'
        'judge=<harmbench|advbench>". Each (model, behavior) cell appears under '
        'multiple attacks and multiple judges.'
    ),
    'paper_url': 'https://arxiv.org/abs/2402.04249',
    'data_source_url': 'https://github.com/centerforaisafety/HarmBench',
    'subject_type': 'model',
    'item_type': 'task',
    'license': 'MIT',
    'citation': """@article{mazeika2024harmbench,
  title={HarmBench: A Standardized Evaluation Framework for Automated Red Teaming and Robust Refusal},
  author={Mazeika, Mantas and Phan, Long and Yin, Xuwang and Zou, Andy and Wang, Zifan and Mu, Norman and Sakhaee, Elham and Li, Nathaniel and Basart, Steven and Li, Bo and Forsyth, David and Hendrycks, Dan},
  journal={arXiv preprint arXiv:2402.04249},
  year={2024},
}""",
    'tags': ['safety', 'jailbreak', 'red_teaming', 'adaptive_attack'],
    'modality': ['text'],
    'domain': ['safety'],
    'response_type': 'binary',
    'response_scale': '{0, 1}',
    'categorical': True,
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

DATA_REPO = "justinphan3110cais/harmbench_website"
DATA_BRANCH = "data"
DATA_BASE = (
    f"https://raw.githubusercontent.com/{DATA_REPO}/{DATA_BRANCH}/playground_data"
)
API_BASE = f"https://api.github.com/repos/{DATA_REPO}/contents/playground_data"

CATEGORIES = ("standard", "contextual")  # multimodal/copyright skipped; text-only for now


def _download(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 100:
        return dest
    print(f"  downloading {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(resp.read())
    return dest


def _list_api(path: str) -> list[dict]:
    url = f"{API_BASE}/{path}?ref={DATA_BRANCH}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.load(resp)


def load_json(rel: str) -> object:
    dest = RAW_DIR / rel
    _download(f"{DATA_BASE}/{rel}", dest)
    return json.loads(dest.read_text())


def discover_method_files() -> list[tuple[str, str]]:
    """Return list of (category, method_name) for every <METHOD>.json file."""
    out: list[tuple[str, str]] = []
    for category in CATEGORIES:
        try:
            entries = _list_api(category)
        except Exception as e:
            print(f"  could not list {category}/: {e}")
            continue
        for entry in entries:
            if entry["type"] != "file" or not entry["name"].endswith(".json"):
                continue
            method = entry["name"].removesuffix(".json")
            out.append((category, method))
    return out


def build_long_form(method_files: list[tuple[str, str]]) -> pd.DataFrame:
    bench_id = get_benchmark_id(
        "harmbench",
        name="HarmBench",
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

    # Load behavior content for the items registry.
    behaviors = load_json("text_behaviors.json")
    behavior_text: dict[str, dict] = {
        b["BehaviorID"]: b for b in behaviors if "BehaviorID" in b
    }
    # Pre-register so item_ids are stable even for behaviors with no responses.
    for bid, b in behavior_text.items():
        register_item(
            benchmark_id=bench_id,
            raw_item_id=bid,
            content=b.get("Behavior") or bid,
        )

    rows: list[dict] = []
    skipped_no_label = 0

    for category, method in method_files:
        rel = f"{category}/{method}.json"
        try:
            data = load_json(rel)
        except Exception as e:
            print(f"  skip {rel}: {e}")
            continue

        for behavior_id, by_model in data.items():
            if not isinstance(by_model, dict):
                continue
            content = (
                behavior_text.get(behavior_id, {}).get("Behavior") or behavior_id
            )
            item = register_item(
                benchmark_id=bench_id,
                raw_item_id=behavior_id,
                content=content,
            )

            for model_id, cell in by_model.items():
                if not isinstance(cell, dict):
                    continue
                subj = resolve_subject(model_id)
                test_case = cell.get("test_case")
                generation = cell.get("generation")

                for judge, key in (("harmbench", "label"), ("advbench", "advbench_label")):
                    verdict = cell.get(key)
                    if verdict is None:
                        skipped_no_label += 1
                        continue
                    try:
                        response = float(int(verdict))
                    except (TypeError, ValueError):
                        skipped_no_label += 1
                        continue

                    condition = (
                        f"attack={method};category={category};judge={judge}"
                    )
                    # Encode the attack prompt (which differs per (behavior, model,
                    # attack)) and the model's generation into a single trace blob.
                    if isinstance(test_case, list):
                        test_case_text = "\n".join(str(x) for x in test_case)
                    else:
                        test_case_text = test_case
                    trace = None
                    if generation or test_case_text:
                        trace = json.dumps(
                            {"test_case": test_case_text, "generation": generation},
                            ensure_ascii=False,
                        )

                    rows.append({
                        "subject_id": subj,
                        "item_id": item,
                        "benchmark_id": bench_id,
                        "trial": 1,
                        "test_condition": condition,
                        "response": response,
                        "correct_answer": None,
                        "trace": trace,
                    })

    if skipped_no_label:
        print(f"  WARNING: {skipped_no_label} cells had no usable label")

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
    print(f"  response mean (HarmBench judge only): "
          f"{df.loc[df['test_condition'].str.contains('judge=harmbench'), 'response'].mean():.3f}")


def main() -> None:
    print(f"[harmbench] building from {_BENCHMARK_DIR}")
    print("  discovering method files...")
    method_files = discover_method_files()
    print(f"  found {len(method_files)} (category, method) result files")
    df = build_long_form(method_files)
    print_stats(df)


if __name__ == "__main__":
    main()
