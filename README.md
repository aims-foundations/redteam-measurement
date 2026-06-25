# redteam-measurement

Long-form response matrices for adaptive AI red-teaming benchmarks,
formatted to match [aims-foundations/measurement-db][mdb] (the public sync of
`measurement-db-private`). Each dataset folder ships only a `build.py` that
downloads upstream data and emits parquet outputs locally — parquets are
gitignored, identical to the upstream policy.

[mdb]: https://github.com/aims-foundations/measurement-db

## Layout

```
.
├── _registry/
│   └── __init__.py            # registry helpers (resolve_subject, register_item, ...)
├── manifest.yaml              # dataset → {status, domain, response_type}
└── <dataset>/
    ├── build.py               # INFO dict + idempotent ingest pipeline
    ├── raw/                   # cached upstream files (gitignored)
    ├── responses.parquet      # long-form, trace stripped (gitignored)
    ├── traces.parquet         # (subject, item, ..., trace) sidecar (gitignored)
    └── _contrib/              # per-dataset registry contributions (gitignored)
        ├── subjects.parquet
        ├── items.parquet
        └── benchmarks.parquet
```

`responses.parquet` is the canonical table — one row per
`(subject_id, item_id, trial, test_condition)`. `test_condition` is used
here to encode the **adaptive attacker** that generated the prompt (e.g.
`attack=PAIR;attack_type=black_box`), so the same target model evaluated
under different attackers shows up as separate rows on the same item.

Parquet outputs and `raw/` caches are gitignored to mirror the upstream
repo's policy of shipping `build.py` only. Run `python <dataset>/build.py`
to regenerate them.

## Datasets

| dataset | subjects | items | conditions | rows | response |
|---|---|---|---|---|---|
| jailbreakbench | 4 | 105 | 6 attack methods | 1,800 | binary (jailbroken 0/1) |
| harmbench | 29 | 140 | 16 attacks × 2 judges − 1 = 31 | 104,194 | binary (label 0/1) |

### jailbreakbench

100 harmful behaviors from JBB-Behaviors evaluated against four target models
(GPT-3.5, GPT-4, Llama-2-7B-chat, Vicuna-13B-v1.5) under six adaptive-attack
conditions:

- `attack=PAIR;attack_type=black_box`
- `attack=GCG;attack_type=white_box`
- `attack=GCG;attack_type=transfer`
- `attack=DSN;attack_type=white_box`
- `attack=JBC;attack_type=manual`
- `attack=prompt_with_random_search;attack_type=black_box`

The matrix is intentionally sparse: white-box attacks (GCG, DSN) only have
artifacts for open-weight models, since proprietary APIs don't expose
gradients.

Source: <https://github.com/JailbreakBench/artifacts> · Paper:
[Chao et al., 2024](https://arxiv.org/abs/2404.01318).

### harmbench

140 harmful behaviors (semantic categories: chemical/biological, cyber, harmful,
illegal, misinformation_disinformation) evaluated against 29 target models
(Llama-2 7/13/70B, Vicuna 7/13B, Baichuan2 7/13B, Qwen 7/14/72B, Koala 7/13B,
Orca-2 7/13B, Solar-10.7B, Mistral-7B-v2, Mixtral-8×7B, OpenChat-3.5,
Starling-7B, Zephyr-7B {±robust6}, GPT-3.5-{0613,1106}, GPT-4-{0613,1106},
Claude-{instant-1, 2, 2.1}, Gemini) under 16 attack methods × 2 judges = 31 test
conditions (TAP-T has no AdvBench judge, so 32 − 1).

`test_condition` is `attack=<method>;judge=<harmbench|advbench>`. Attack methods:
AutoDAN, AutoPrompt, DirectRequest, FewShot, GBDA, GCG, GCG-M, GCG-T,
HumanJailbreaks, PAIR, PAP, PEZ, TAP, TAP-T, UAT, ZeroShot.

Each cell carries two judge verdicts encoded as separate `test_condition`s:

- `judge=harmbench` — HarmBench's Llama-2-13B fine-tuned classifier (canonical)
- `judge=advbench` — AdvBench's refusal-string heuristic

**Functional category is an item property, not a condition.** Each behavior has
exactly one `FunctionalCategory` — 57 `standard`, 48 `copyright`, 35
`contextual` — stored in the `category` column of the items registry, not in
`test_condition`. (The upstream playground ships `standard/`, `contextual/`,
`copyright/` as byte-identical copies of the same 140-behavior result set, so
the download folder is *not* a category; we read one folder and take each
behavior's category from `text_behaviors.json`.)

**Coverage caveat:** these 140 behaviors are the website-playground subset of
HarmBench's full 400 textual behaviors (200 standard / 100 copyright / 100
contextual); multimodal behaviors are not included.

Source: <https://github.com/centerforaisafety/HarmBench>; data files pulled
from the same JSON dumps the [HarmBench website playground](https://www.harmbench.org/explore)
fetches, hosted at `justinphan3110cais/harmbench_website` (`data` branch). Paper:
[Mazeika et al., 2024](https://arxiv.org/abs/2402.04249).

## Rebuilding

```bash
python jailbreakbench/build.py
python harmbench/build.py
```

Re-running is safe: `raw/` is reused as a cache and parquet outputs are
overwritten.

## Mapping to the paper's notation

For algorithm $A$ with attack method $a \in \{$PAIR, GCG, ...$\}$, filter
`responses.parquet` to `test_condition` containing `attack={a}`. The result
is a sparse $Y_A$ where rows are `subject_id` and columns are `item_id`,
with `response` ∈ {0, 1}.

```python
import pandas as pd
r = pd.read_parquet("jailbreakbench/responses.parquet")
Y_pair = r[r.test_condition.str.contains("attack=PAIR")] \
    .pivot(index="subject_id", columns="item_id", values="response")
```
