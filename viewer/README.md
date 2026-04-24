# LCVB viewer

A small local web app for browsing and generating LCVB prompts.

Two features, no API calls:

1. **Browse** the `generated/{condition}/` tree. Pick a file, see the
   conversation history rendered as turn cards, the triggering user message,
   and the full metadata (scenario id, evidence variant, placement depth,
   distractor hash, etc.).
2. **Generate** new prompt sets. Pick a renderer (`with_constraint`,
   `fixed_locations`, or `continuous_random`), tweak the knobs in the modal,
   and the server runs the renderer as a subprocess. Progress is polled and
   the prompt list refreshes when the job finishes.

Nothing in this viewer calls any LLM API — eval runs are deliberately not
supported from the UI. Use the `pipeline/run.py` CLI for those.

## Quickstart

```bash
pip install flask
python viewer/app.py
# open http://127.0.0.1:5057/
```

Or install the optional `viewer` extra:

```bash
pip install -e '.[viewer]'
python viewer/app.py --port 8080
```

The server picks `127.0.0.1:5057` by default. Pass `--host 0.0.0.0` to bind
all interfaces, `--port N` for a different port, `--debug` for Flask debug
mode.

## What you'll see

The three columns of the layout:

- **Left sidebar** — list of conditions with counts, scenario + variant
  filters, prompt file list, recent-jobs tracker.
- **Main pane** — turn-by-turn rendered conversation history (user turns
  on one tint, assistant turns on another, evidence pairs highlighted),
  followed by the final user message (the triggering query).
- **Right sidebar** — scenario metadata: constraint text, query options,
  depth, placement_frac, distractor hash, input char length, and so on.

The "Generate new…" button opens a modal with:

- A renderer selector.
- An output directory (must start with `generated/` — the server refuses
  anything else).
- Knobs for each renderer's CLI arguments.
- A live preview of the exact `python …` command that will be run.
- A submit button that kicks off the subprocess and starts polling.

## Safety rails

- File-name params are restricted to simple names (letters, digits,
  `._+-`); no `..`, no slashes.
- Generate always writes under `generated/` (enforced server-side,
  resolved with `Path.resolve()`).
- Renderers themselves refuse to overwrite a non-empty target directory —
  that error surfaces in the job-details modal. If you want to re-generate,
  pick a new directory name (e.g. `generated/fixed_locations_v2/`).

## Routes

| Route | What it does |
|---|---|
| `GET /` | serves the SPA |
| `GET /api/conditions` | renderer registry + file counts |
| `GET /api/prompts?condition=…` | list JSON files (filterable by scenario/variant) |
| `GET /api/prompt?condition=…&file=…` | load one prompt (parsed + raw) |
| `GET /api/scenarios` | scenario id list (from `data/scenarios_FINAL.tsv`) |
| `POST /api/render` | kick off a renderer subprocess; returns `job_id` |
| `GET /api/render/status?job_id=…` | poll job (status, stdout/stderr tails) |
| `GET /api/render/jobs` | list recent jobs (in-memory) |

Jobs are tracked in-memory — they vanish when the server restarts.
