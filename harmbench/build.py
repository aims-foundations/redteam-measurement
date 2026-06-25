"""Build HarmBench long-form responses from the HarmBench website's data repo.

Data sources (the same files the playground at https://www.harmbench.org/ fetches):
  - https://raw.githubusercontent.com/justinphan3110cais/harmbench_website/data/
        playground_data/metadata_text.json       # methods + models manifest
        playground_data/text_behaviors.json      # behavior text + categories
        playground_data/standard/<METHOD>.json   # per-attack results

Layout of each <METHOD>.json:
    {behavior_id: {model_id: {test_case, generation, label, advbench_label}}}

NOTE ON THE `standard/`, `contextual/`, `copyright/` DIRECTORIES: in the upstream
playground repo these are **byte-identical copies** of the same full result set —
each <METHOD>.json contains *all* 140 behaviors regardless of folder (verified:
standard/PAIR.json holds 57 standard + 48 copyright + 35 contextual behaviors).
The folder name does NOT encode a behavior category; the website filters
client-side on each behavior's `FunctionalCategory` field. So we read a single
directory (`SOURCE_CATEGORY`) and take each behavior's true category from
`text_behaviors.json`, attaching it to the item — NOT to `test_condition`.

`label` is the HarmBench classifier verdict (Llama-2-13B fine-tuned judge).
`advbench_label` is the AdvBench refusal-string heuristic. We emit both as
separate `test_condition`s so the paper's measurement model can compare
judges as well as attackers.

Coverage caveat: this is the 140-behavior website subset of HarmBench's 400
textual behaviors (no multimodal); the AdvBench judge is absent for TAP-T.

Output:
  - raw/<category>/<METHOD>.json, raw/metadata_text.json, raw/text_behaviors.json
  - responses.parquet                              # long-form, no trace
  - traces.parquet                                 # generations + adversarial test_cases
  - _contrib/{subjects,items,benchmarks}.parquet   # registry contributions
"""

INFO = {
    'description': (
        'HarmBench standardized red-teaming results: 29 target models x 140 '
        'behaviors x 16 attack methods, with the HarmBench Llama-2-13B '
        'classifier verdict and the AdvBench refusal-string heuristic as two '
        'separate judges. The 140 behaviors are the website-playground subset '
        'of HarmBench\'s 400 textual behaviors and span all three functional '
        'categories (standard, contextual, copyright); each behavior\'s '
        'FunctionalCategory is recorded on the item.'
    ),
    'testing_condition': (
        'test_condition is "attack=<method>;judge=<harmbench|advbench>". Each '
        '(model, behavior) cell appears under multiple attacks and both judges. '
        'A behavior\'s functional category (standard/contextual/copyright) is an '
        'item property, not a condition, and lives on the items registry.'
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
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from build_base import BenchmarkBuild, register_item, resolve_subject  # noqa: E402

DATA_REPO = "justinphan3110cais/harmbench_website"
DATA_BRANCH = "data"
DATA_BASE = (
    f"https://raw.githubusercontent.com/{DATA_REPO}/{DATA_BRANCH}/playground_data"
)
API_BASE = f"https://api.github.com/repos/{DATA_REPO}/contents/playground_data"

# The playground ships standard/, contextual/, copyright/ as identical copies of
# the full result set (see module docstring). We read exactly one of them; the
# behavior's real category comes from text_behaviors.json's FunctionalCategory.
SOURCE_CATEGORY = "standard"


class HarmBench(BenchmarkBuild):
    INFO = INFO
    slug = "harmbench"
    name = "HarmBench"

    def _load_json(self, rel: str) -> object:
        """Download playground_data/<rel> into raw/ (cached) and parse it."""
        dest = self.raw_dir / rel
        self._download(f"{DATA_BASE}/{rel}", dest)
        return json.loads(dest.read_text())

    def _discover_method_files(self) -> list[str]:
        """Return the method name for every <METHOD>.json in SOURCE_CATEGORY."""
        out: list[str] = []
        try:
            entries = self._get_json(f"{API_BASE}/{SOURCE_CATEGORY}?ref={DATA_BRANCH}")
        except Exception:
            return out
        for entry in entries:
            if entry["type"] != "file" or not entry["name"].endswith(".json"):
                continue
            out.append(entry["name"].removesuffix(".json"))
        return out

    def build_rows(self, bench_id: str) -> list[dict]:
        # Load behavior content for the items registry.
        behaviors = self._load_json("text_behaviors.json")
        behavior_text: dict[str, dict] = {
            b["BehaviorID"]: b for b in behaviors if "BehaviorID" in b
        }
        # Pre-register so item_ids are stable even for behaviors with no responses.
        # The behavior's true FunctionalCategory (standard/contextual/copyright)
        # is an intrinsic item property and is attached here.
        for bid, b in behavior_text.items():
            register_item(
                benchmark_id=bench_id,
                raw_item_id=bid,
                content=b.get("Behavior") or bid,
                category=b.get("FunctionalCategory"),
            )

        rows: list[dict] = []

        for method in self._discover_method_files():
            rel = f"{SOURCE_CATEGORY}/{method}.json"
            try:
                data = self._load_json(rel)
            except Exception:
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
                    category=behavior_text.get(behavior_id, {}).get(
                        "FunctionalCategory"
                    ),
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
                            continue
                        try:
                            response = float(int(verdict))
                        except (TypeError, ValueError):
                            continue

                        condition = f"attack={method};judge={judge}"
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
        return rows


if __name__ == "__main__":
    HarmBench(__file__).main()
