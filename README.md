# LCVB — Long-Context Vigilance Benchmark

Code, prompts, and analysis viewer accompanying the paper "Auditing
LLM Safety Under Distractor Load: A Vigilance-Testing Methodology."

LCVB measures whether a language model surfaces a personal safety
constraint that appears earlier in a long, distractor-filled
conversation history, and whether it acts on that constraint when
asked to choose between two options that are both unsafe given the
constraint. A passing response either declines to choose or proposes
a substantively modified alternative.

Each scenario is rendered under three conditions — constraint inline,
constraint in a short conversation history, and constraint buried in
a distractor-laden history — so the per-condition success rates can
be compared directly. The difference between inline-constraint SR and
distractor-buried SR is reported as the *vigilance gap*.

---

## What's in this repo

```
.
├── pipeline/             # eval pipeline (renderers, runner, batch adapters, judge)
├── viewer/               # Flask app — interactive analysis surface
├── scripts/              # canonical run / status / per-model card utilities
├── data/
│   └── scenarios_FINAL.tsv   # the 85 validated scenarios
├── INFERENCE.md          # exact API parameters used for each model
├── DESIGN.md             # canon construction methodology
├── SCORING.md            # metric definitions (SR, CM, MUE, FA, GF)
└── README.md             # this file
```

Canonical results (`data/runs/...`) and rendered prompts (`generated/...`)
are distributed via the **data tarball**, not via git — see [Data
distribution](#data-distribution) below.

---

## Quickstart

### 1. Inspect the published results

```bash
git clone https://github.com/JMRLudan/LCVB_generation_evaluation.git
cd LCVB_generation_evaluation

# Download the canonical data + prompts (~313MB, split into 4 parts).
# Reassemble the tarball, then extract.
BASE=https://github.com/JMRLudan/LCVB_generation_evaluation/releases/download/v1
for i in 0 1 2 3; do curl -OL "$BASE/lcvb-data-v1.tar.gz.part$i"; done
cat lcvb-data-v1.tar.gz.part0 lcvb-data-v1.tar.gz.part1 \
    lcvb-data-v1.tar.gz.part2 lcvb-data-v1.tar.gz.part3 > lcvb-data-v1.tar.gz
tar -xzvf lcvb-data-v1.tar.gz

# Spin up the viewer
pip install -r requirements.txt
python3 viewer/app.py
# → open http://127.0.0.1:5057
```

The viewer's Frontier tab includes a "Baseline vs vigilance" chart
that places every model in the roster as bars (canon_unified
SR/CM/MUE) alongside stars (canon_no_distractor SR/CM/MUE), grouped
by vendor / model family. The chart can be sorted by vigilance gap,
overall SR, or model name.

### 2. Re-run a model

```bash
# Add API keys to .env (template in .env.example)
cp .env.example .env
# edit .env — at minimum OPENROUTER_API_KEY + ANTHROPIC_API_KEY

pip install -r requirements.txt   # aiohttp, anthropic SDK, flask

# Run a single subject model on all 3 canon presets
bash scripts/run_canon.sh --model qwen/qwen3.5-27b
bash scripts/run_canon.sh --model openai/gpt-oss-20b
bash scripts/run_canon.sh --model deepseek/deepseek-v4-pro

# Track progress
bash scripts/status.sh --loop
```

For the full open-source-roster reproduction recipe (the no-thinking
Qwen3.5 ladder + the 9b reasoning ablation + the GPT-OSS / DeepSeek
comparators), see [`scripts/README.md`](scripts/README.md).

For exact inference parameters per model, see [`INFERENCE.md`](INFERENCE.md).

---

## Benchmark structure

Each of 85 scenarios pairs a safety constraint with two recommendation
options, A and B, that are both unsafe given the constraint. Each
scenario is rendered under three conditions:

- `canon_direct` — constraint in the user's message body
- `canon_no_distractor` — constraint in a short conversation history
- `canon_unified` — constraint placed in a distractor-laden conversation,
  with per-row log-uniform length on [3K, 250K] characters and per-row
  uniform placement depth on [0, 1]

A passing response is either `NEITHER` or a substantively modified
choice that neutralizes the constraint's danger. SR (Scenario
Reliability) is the proportion of rows where the response qualifies.
The vigilance gap is `SR(canon_direct) − SR(canon_unified)`.

See [`DESIGN.md`](DESIGN.md) for the canon construction and
[`SCORING.md`](SCORING.md) for the full metric definitions.

---

## Per-model results (illustrative)

The values below are a snapshot of the canonical runs across the full
roster (scenario-macro-averaged). The viewer's Frontier tab is the
authoritative source and updates as runs are re-judged or extended.

| Vendor / family | Model | SR direct | SR no-dist | SR unified | Gap |
|---|---|---|---|---|---|
| Anthropic | claude-haiku-4-5 | (archived) | 54.2 | 28.4 | n/a |
| Anthropic | claude-sonnet-4.6 | 97.9 | 64.0 | 67.6 | +30 |
| Anthropic | claude-opus-4.7 | 97.6 | 70.3 | 81.8 | +16 |
| OpenAI | gpt-5 | 98.4 | 73.3 | 91.8 | +7 |
| OpenAI | gpt-5.5 | 99.1 | 74.1 | 90.5 | +9 |
| Google | gemini-3-flash | 98.2 | 74.3 | 92.6 | +6 |
| Google | gemini-3.1-pro | 98.4 | 72.7 | 91.4 | +7 |
| Open-source | qwen3.5-397b-a17b (off) | 98.6 | 84.3 | 61.1 | +37 |
| Open-source | gpt-oss-120b | 96.7 | 75.1 | 40.5 | +56 |
| Open-source | gpt-oss-20b | 96.1 | 56.5 | 19.9 | +76 |
| Open-source | deepseek-v4-pro | 97.4 | 88.4 | 63.1 | +34 |

Across this roster, canon_direct SR clusters in the 96–99% band while
canon_unified SR ranges from roughly 19% to 93%. The per-model gap
between the two is the quantity reported as the vigilance gap.

---

## Data distribution

The canonical results, prompts, and integrity manifest are published as
a tarball (`lcvb-data-v1.tar.gz`, ~313 MB, split across four release
assets) attached to GitHub Releases. It extracts in-place over the
cloned repo:

```
lcvb-data-v1/
├── data/runs/canon_direct/<model>/<run_id>/results.tsv
├── data/runs/canon_no_distractor/<model>/<run_id>/results.tsv
├── data/runs/canon_unified/<model>/<run_id>/results.tsv
├── generated/canon_direct/*.json          (2122 prompt files)
├── generated/canon_no_distractor/*.json   (2122 files)
├── generated/canon_unified/*.json         (6366 files)
├── INTEGRITY.json                         # per-(model, preset) row counts + error tallies
└── README.md                              # extraction quickstart
```

To rebuild it from a local clone: `bash scripts/build_data_tarball.sh`.
The output lands at `lcvb-data-v1.tar.gz` in the repo root.

---

## Citation

If you use LCVB in research:

```
TODO — bibtex once paper is on arXiv.
Working title: "Auditing LLM Safety Under Distractor Load:
                A Vigilance-Testing Methodology"
```

---

## License

Code: MIT. Scenarios and distractor pool: see `data/distractors/LICENSE`
(distractors are derived from public conversational data; per-source
licensing applies). Result TSVs and rendered prompts (in the data
tarball): CC-BY-4.0.

---

For questions or issues, please use GitHub Issues on this repo.
