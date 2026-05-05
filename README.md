# LCVB — Long-Context Vigilance Benchmark

Code, prompts, and analysis viewer for the paper **"Auditing LLM Safety
Under Distractor Load: A Vigilance-Testing Methodology."**

LCVB tests whether language models can detect personal safety
constraints buried in a long, distractor-filled conversation history.
A vigilant model must surface the constraint and either abstain from
the dangerous-by-default options or propose a heavily-modified safe
alternative — even when the constraint hasn't been mentioned for
many turns.

The **headline finding** is a 30+ percentage-point spread in *vigilance
gap* (SR with constraint inline minus SR with constraint buried in
distractors) across frontier and open-source models. The viewer's
Frontier tab reproduces this in a single grouped-bar chart across
~17 models in five launch stages.

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

# Download + extract the canonical data + prompts (~300MB)
curl -OL https://github.com/JMRLudan/LCVB_generation_evaluation/releases/download/v1/lcvb-data-v1.tar.gz
tar -xzvf lcvb-data-v1.tar.gz

# Spin up the viewer
pip install -r requirements.txt
python3 viewer/app.py
# → open http://127.0.0.1:5057
```

The viewer's **Frontier tab → "Baseline vs vigilance" chart** is the
paper's headline figure: every model in the roster as bars (canon_unified
SR/CM/MUE) plus stars (canon_no_distractor SR/CM/MUE), grouped by
launch stage. Sort by vigilance gap to see the methodology's value
immediately.

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

For the full Stage-6 reproduction recipe (the no-thinking ladder + the
9b reasoning ablation + the 3 comparators), see [`scripts/README.md`](scripts/README.md).

For exact inference parameters per model, see [`INFERENCE.md`](INFERENCE.md).

---

## The benchmark in one paragraph

Each of 85 scenarios pairs a **safety constraint** with two choices A
and B that are both dangerous given that constraint. Three rendering
conditions stress-test the model's ability to surface the constraint:

- **canon_direct** — constraint in the user's message body (ceiling test)
- **canon_no_distractor** — constraint in a short conversation history
- **canon_unified** — constraint buried in a distractor-laden conversation
  with random length (3K–250K chars) and random placement depth (0–1)

A vigilant response either picks `NEITHER` or substantively modifies a
choice to be safe; **SR (Scenario Reliability)** = proportion of rows
where the response did so. A buried-constraint SR substantially below
the inline-constraint SR is the *vigilance gap* — the methodology's
core measurement.

See [`DESIGN.md`](DESIGN.md) for the canon construction details and
[`SCORING.md`](SCORING.md) for the full metric derivation.

---

## Headline numbers

From the canonical Stage 1–6 runs (17 models × 3 presets, all
scenario-macro-averaged):

| Stage | Model | SR direct | SR no-dist | SR unified | Gap |
|---|---|---|---|---|---|
| 1 | claude-haiku-4-5 | (archived) | 54.2 | 28.4 | n/a |
| 2 | claude-sonnet-4.6 | 97.9 | 64.0 | 67.6 | +30 |
| 2 | claude-opus-4.7 | 97.6 | 70.3 | 81.8 | +16 |
| 3 | gpt-5 | 98.4 | 73.3 | 91.8 | +7 |
| 3 | gpt-5.5 | 99.1 | 74.1 | 90.5 | +9 |
| 4 | gemini-3-flash | 98.2 | 74.3 | 92.6 | +6 |
| 4 | gemini-3.1-pro | 98.4 | 72.7 | 91.4 | +7 |
| 6 | qwen3.5-397b-a17b (off) | 98.6 | 84.3 | 61.1 | +37 |
| 6 | gpt-oss-120b | 96.7 | 75.1 | 40.5 | +56 |
| 6 | gpt-oss-20b | 96.1 | 56.5 | 19.9 | +76 |
| 6 | deepseek-v4-pro | 97.4 | 88.4 | 63.1 | +34 |

(Numbers above are illustrative top-of-funnel — the viewer's Frontier
tab is authoritative and updates as new runs land.)

The **vigilance gap (direct − unified)** is the headline insight:
canon_direct numbers cluster at 96–99% across the entire roster, but
unified SR ranges 19–93%. **A standard "constraint-in-prompt" benchmark
sees almost none of this spread.**

---

## Data distribution

The canonical results, prompts, and integrity manifest are published as
a single tarball (`lcvb-data-v1.tar.gz`, ~300MB) attached to GitHub
Releases. It extracts in-place over the cloned repo:

```
lcvb-data-v1/
├── data/runs/canon_direct/<model>/<run_id>/results.tsv
├── data/runs/canon_no_distractor/<model>/<run_id>/results.tsv
├── data/runs/canon_unified/<model>/<run_id>/results.tsv
├── generated/canon_direct/*.json    (2122 prompt files)
├── generated/canon_no_distractor/*.json (2122 files)
├── generated/canon_unified/*.json    (6366 files)
├── INTEGRITY.json   # per-(model, preset) row counts and error tallies
└── README.md        # this exact UX
```

To rebuild it from a local research environment: `bash
scripts/build_data_tarball.sh`. Output lands at `lcvb-data-v1.tar.gz`
in the repo root.

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
