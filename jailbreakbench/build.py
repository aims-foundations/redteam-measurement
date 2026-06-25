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

This is the reference implementation of a `BenchmarkBuild` subclass: all the
shared plumbing (paths, download helper, registry registration, the
unique-trials / trace-split / parquet-write tail, and `main()` orchestration)
lives in ../build_base.py. This file supplies only what is unique to
JailbreakBench: the `INFO` metadata, the artifact discovery/download URLs, and
the `build_rows` parsing logic.
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
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from build_base import BenchmarkBuild, register_item, resolve_subject  # noqa: E402

ARTIFACTS_API = (
    "https://api.github.com/repos/JailbreakBench/artifacts/contents/attack-artifacts"
)
ARTIFACTS_RAW = (
    "https://raw.githubusercontent.com/JailbreakBench/artifacts/main/attack-artifacts"
)


class JailbreakBench(BenchmarkBuild):
    INFO = INFO
    slug = "jailbreakbench"
    name = "JailbreakBench"

    def build_rows(self, bench_id: str) -> list[dict]:
        # Walk the artifacts repo for every (method, attack_type, model) JSON file.
        artifacts: list[tuple[str, str, str]] = []
        for entry in self._get_json(ARTIFACTS_API):
            if (
                entry["type"] != "dir"
                or entry["name"].startswith(".")
                or entry["name"] == "test-artifact"
            ):
                continue
            method = entry["name"]
            for sub in self._get_json(entry["url"]):
                if sub["type"] != "dir":
                    continue
                attack_type = sub["name"]
                for f in self._get_json(sub["url"]):
                    if f["type"] == "file" and f["name"].endswith(".json"):
                        artifacts.append((method, attack_type, f["name"]))

        rows: list[dict] = []
        missing_verdict = 0

        for method, attack_type, fname in artifacts:
            dest = self.raw_dir / method / attack_type / fname
            try:
                self._download(f"{ARTIFACTS_RAW}/{method}/{attack_type}/{fname}", dest)
                data = json.loads(dest.read_text())
            except Exception:
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
        return rows

if __name__ == "__main__":
    JailbreakBench(__file__).main()
