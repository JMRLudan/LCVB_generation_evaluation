#!/usr/bin/env python3
"""
viewer/app.py — Local Flask app for browsing and generating LCVB prompts.
==========================================================================

Two features:
  1. Browse prompts under `generated/{condition}/` — pick a file, see the
     formatted conversation history, metadata, and the final user message.
  2. Trigger a renderer subprocess with knob values from the UI. Jobs run
     in the background; the UI polls status and re-lists when they finish.

No API calls, no cost. Eval runs are NOT available from this UI by design.

Quickstart:
    pip install flask
    python viewer/app.py                  # http://127.0.0.1:5057
    python viewer/app.py --port 8080      # different port

Safety:
  * File-name params are restricted to the `generated/{condition}/` tree.
    No `..`, no absolute paths, no leading `/`.
  * Renderers will refuse to overwrite a non-empty `out_dir` on their own —
    the UI surfaces that error instead of force-deleting.
  * Nothing here touches `data/` other than reading scenarios + pool.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional

try:
    from flask import Flask, jsonify, request, send_from_directory, abort
except ImportError as e:
    print(
        "Flask is required. Install with:\n"
        "    pip install flask\n"
        "or:\n"
        "    pip install -e '.[viewer]'",
        file=sys.stderr,
    )
    raise


# ──────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────
VIEWER_DIR = Path(__file__).resolve().parent
REPO_ROOT = VIEWER_DIR.parent
GENERATED_DIR = REPO_ROOT / "generated"
DATA_DIR = REPO_ROOT / "data"
PIPELINE_DIR = REPO_ROOT / "pipeline"
SCENARIOS_TSV = DATA_DIR / "scenarios_FINAL.tsv"


# ──────────────────────────────────────────────────────────────────────
# Renderer registry
# ──────────────────────────────────────────────────────────────────────
# Each entry describes the renderer script, its default `out_dir` under
# `generated/`, and the knobs the UI can set on it. Knob types: int, float,
# bool, choice, multi_choice, list_float.
RENDERERS: Dict[str, Dict] = {
    "with_constraint": {
        "label": "with_constraint",
        "script": "pipeline/renderers/render_with_constraint.py",
        "default_out_dir": "generated/with_constraint",
        "description": (
            "Direct ablation — evidence seeds and the constraint description "
            "are placed inline in the user message, with no prior "
            "conversation history. Ceiling test."
        ),
        "knobs": [
            {
                "name": "include-no-c-variants",
                "label": "Include A and B (no-constraint) variants",
                "type": "bool",
                "default": False,
            },
        ],
    },
    "no_distractor": {
        "label": "no_distractor",
        "script": "pipeline/renderers/render_no_distractor.py",
        "default_out_dir": "generated/no_distractor",
        "description": (
            "Primary condition — evidence seeds placed in a short timestamped "
            "conversation history in the system prompt; query asked as a "
            "fresh user message. No distractor turns interleaved."
        ),
        "knobs": [
            {
                "name": "c-only",
                "label": "C-only variants (skip A and B baselines)",
                "type": "bool",
                "default": False,
            },
        ],
    },
    "fixed_locations": {
        "label": "fixed_locations",
        "script": "pipeline/renderers/render_fixed_locations.py",
        "default_out_dir": "generated/fixed_locations",
        "description": (
            "Distractor-grid sweep — constraint inserted at fixed relative "
            "depths across both short and long haystacks."
        ),
        "knobs": [
            {
                "name": "num-distractor-draws",
                "label": "Number of distractor draws",
                "type": "int",
                "default": 1,
                "min": 1,
                "max": 5,
            },
            {
                "name": "depths",
                "label": "Depths (fraction — 0.0=top of history, 1.0=just before query)",
                "type": "list_float",
                "default": "0.0,0.25,0.5,0.75,1.0",
            },
            {
                "name": "c-only",
                "label": "C-only variants (skip A and B baselines — cheaper)",
                "type": "bool",
                "default": False,
            },
        ],
    },
    "continuous_random": {
        "label": "continuous_random",
        "script": "pipeline/renderers/render_continuous_random.py",
        "default_out_dir": "generated/continuous_random",
        "description": (
            "Uniform-placement sweep — one stratified random depth per item, "
            "balanced across [0, 1]. Single char budget."
        ),
        "knobs": [
            {
                "name": "num-distractor-draws",
                "label": "Number of distractor re-renders",
                "type": "int",
                "default": 1,
                "min": 1,
                "max": 5,
            },
            {
                "name": "char-budget",
                "label": "Char budget (conversation history length)",
                "type": "int",
                "default": 224000,
                "min": 1000,
                "max": 500000,
            },
            {
                "name": "c-only",
                "label": "C-only variants (skip A and B baselines)",
                "type": "bool",
                "default": False,
            },
        ],
    },
    # ──────────────────────────────────────────────────────────────
    # Canon presets — one per row of the "full table" in the headline
    # results. These are the renderer invocations used for the paper /
    # paper-equivalent runs, tuned to roughly match the main repo's
    # char budgets (~7 chars/tok for the distractor pool content).
    # Use the "Generate canon" button to fire all four in one click.
    # ──────────────────────────────────────────────────────────────
    "canon_direct": {
        "label": "canon · direct",
        "script": "pipeline/renderers/mixer.py",
        "default_out_dir": "generated/canon_direct",
        "canon": True,
        "description": (
            "CANON · ceiling test. Evidence + constraint inline in the "
            "user message, no conversation history. C-present variants only."
        ),
        "knobs": [
            {"name": "n-distractor-draws", "type": "int", "default": 0, "min": 0, "max": 0, "label": "(fixed) n_d=0"},
            {"name": "n-placements", "type": "int", "default": 0, "min": 0, "max": 0, "label": "(fixed) n_p=0"},
            {"name": "n-lengths", "type": "int", "default": 0, "min": 0, "max": 0, "label": "(fixed) n_l=0"},
            {"name": "include-constraint-inline", "type": "bool", "default": True, "label": "(fixed) inline constraint"},
            {"name": "c-only", "type": "bool", "default": True, "label": "C-present variants only"},
            {"name": "condition-label", "type": "str", "default": "canon_direct", "label": "(fixed)"},
        ],
    },
    "canon_no_distractor": {
        "label": "canon · no-distractor",
        "script": "pipeline/renderers/mixer.py",
        "default_out_dir": "generated/canon_no_distractor",
        "canon": True,
        "description": (
            "CANON · primary personalization condition. Short timestamped "
            "evidence history in the system prompt, bare query in the user "
            "message. C-present variants only."
        ),
        "knobs": [
            {"name": "n-distractor-draws", "type": "int", "default": 0, "min": 0, "max": 0, "label": "(fixed) n_d=0"},
            {"name": "n-placements", "type": "int", "default": 0, "min": 0, "max": 0, "label": "(fixed) n_p=0"},
            {"name": "n-lengths", "type": "int", "default": 0, "min": 0, "max": 0, "label": "(fixed) n_l=0"},
            {"name": "include-constraint-inline", "type": "bool", "default": False, "label": "(fixed) short history"},
            {"name": "c-only", "type": "bool", "default": True, "label": "C-present variants only"},
            {"name": "condition-label", "type": "str", "default": "canon_no_distractor", "label": "(fixed)"},
        ],
    },
    "canon_fixed_grid": {
        "label": "canon · fixed grid short/medium/long",
        "script": "pipeline/renderers/mixer.py",
        "default_out_dir": "generated/canon_fixed_grid",
        "canon": True,
        "description": (
            "CANON · distractor grid sweep. Evidence placed at 5 fixed "
            "depths (0, 0.25, 0.5, 0.75, 1.0) across 3 haystacks: "
            "short (10K budget → 8,364 chars mean), "
            "medium (100K budget → 97,080 chars mean), "
            "long (250K budget → 239,954 chars mean; capped by the "
            "~200–230K-char distractor pool). C-present variants only."
        ),
        "knobs": [
            {"name": "n-distractor-draws", "type": "int", "default": 1, "min": 1, "max": 5, "label": "Distractor draws (full grid re-renders)"},
            {"name": "n-placements", "type": "int", "default": 5, "min": 5, "max": 5, "label": "(fixed) 5 depths"},
            {"name": "n-lengths", "type": "int", "default": 3, "min": 3, "max": 3, "label": "(fixed) 3 haystacks"},
            {"name": "placement-mode", "type": "choice", "choices": ["fixed"], "default": "fixed", "label": "(fixed)"},
            {"name": "placements", "type": "list_float", "default": "0.0,0.25,0.5,0.75,1.0", "label": "Depths"},
            {"name": "lengths", "type": "list_int", "default": "10000,100000,250000", "label": "Char budgets"},
            {"name": "length-names", "type": "str", "default": "short,medium,long", "label": "Length names"},
            {"name": "c-only", "type": "bool", "default": True, "label": "C-present variants only"},
            {"name": "condition-label", "type": "str", "default": "canon_fixed_grid", "label": "(fixed)"},
        ],
    },
    "canon_uniform_long": {
        "label": "canon · uniform long (250K budget → 239,960 chars mean)",
        "script": "pipeline/renderers/mixer.py",
        "default_out_dir": "generated/canon_uniform_long",
        "canon": True,
        "description": (
            "CANON · uniform sweep. One stratified-random depth per item "
            "in a 250K-char budget haystack — actual mean 239,960 chars "
            "(capped by the ~200–230K-char distractor pool). Collectively "
            "covers [0, 1] uniformly across items. C-present variants only."
        ),
        "knobs": [
            {"name": "n-distractor-draws", "type": "int", "default": 1, "min": 1, "max": 5, "label": "Distractor draws (re-renders)"},
            {"name": "n-placements", "type": "int", "default": 1, "min": 1, "max": 1, "label": "(fixed) 1 placement"},
            {"name": "n-lengths", "type": "int", "default": 1, "min": 1, "max": 1, "label": "(fixed) 1 length"},
            {"name": "placement-mode", "type": "choice", "choices": ["uniform"], "default": "uniform", "label": "(fixed)"},
            {"name": "lengths", "type": "list_int", "default": "250000", "label": "Char budget"},
            {"name": "length-names", "type": "str", "default": "long", "label": "Length names"},
            {"name": "c-only", "type": "bool", "default": True, "label": "C-present variants only"},
            {"name": "condition-label", "type": "str", "default": "canon_uniform_long", "label": "(fixed)"},
        ],
    },
    "mix_custom": {
        "label": "mix (custom)",
        "script": "pipeline/renderers/mixer.py",
        "default_out_dir": "generated/mix_custom",
        "description": (
            "Direct access to the unified mixer. Set any combination of "
            "n_distractor_draws × n_placements × n_lengths — the four named "
            "conditions above are just presets over this same function."
        ),
        "knobs": [
            {
                "name": "n-distractor-draws",
                "label": "n_distractor_draws (0 = no distractor)",
                "type": "int",
                "default": 1,
                "min": 0,
                "max": 10,
            },
            {
                "name": "n-placements",
                "label": "n_placements (0 = no placement axis)",
                "type": "int",
                "default": 1,
                "min": 0,
                "max": 20,
            },
            {
                "name": "n-lengths",
                "label": "n_lengths (0 = no length axis)",
                "type": "int",
                "default": 1,
                "min": 0,
                "max": 10,
            },
            {
                "name": "placement-mode",
                "label": "placement_mode",
                "type": "choice",
                "choices": ["uniform", "fixed"],
                "default": "uniform",
            },
            {
                "name": "placements",
                "label": "Placements (comma-separated, fixed mode only)",
                "type": "list_float",
                "default": "0.0,0.25,0.5,0.75,1.0",
            },
            {
                "name": "lengths",
                "label": "Char budgets (comma-separated ints)",
                "type": "list_int",
                "default": "24000,224000",
            },
            {
                "name": "length-names",
                "label": "Length names (comma-separated, optional)",
                "type": "str",
                "default": "short,long",
            },
            {
                "name": "include-constraint-inline",
                "label": "Inline the constraint in the user message (with_constraint-style)",
                "type": "bool",
                "default": False,
            },
            {
                "name": "c-only",
                "label": "C-only variants",
                "type": "bool",
                "default": False,
            },
            {
                "name": "condition-label",
                "label": "Label written into metadata.condition",
                "type": "str",
                "default": "mix_custom",
            },
        ],
    },
}


# ──────────────────────────────────────────────────────────────────────
# Job tracking (in-memory; dies with the server)
# ──────────────────────────────────────────────────────────────────────
JOBS: Dict[str, Dict] = {}
JOBS_LOCK = threading.Lock()


def _run_subprocess_job(job_id: str, cmd: List[str], env: Dict[str, str]):
    """Run a renderer subprocess and record stdout/stderr + status."""
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        with JOBS_LOCK:
            JOBS[job_id]["pid"] = proc.pid
            JOBS[job_id]["status"] = "running"
        stdout, stderr = proc.communicate()
        with JOBS_LOCK:
            JOBS[job_id]["status"] = "done" if proc.returncode == 0 else "error"
            JOBS[job_id]["returncode"] = proc.returncode
            JOBS[job_id]["stdout"] = stdout[-8000:]  # tail only
            JOBS[job_id]["stderr"] = stderr[-8000:]
            JOBS[job_id]["finished_at"] = time.time()
    except Exception as e:
        with JOBS_LOCK:
            JOBS[job_id]["status"] = "error"
            JOBS[job_id]["stderr"] = f"Launcher exception: {e}"
            JOBS[job_id]["finished_at"] = time.time()


# ──────────────────────────────────────────────────────────────────────
# Path safety
# ──────────────────────────────────────────────────────────────────────
_SAFE_NAME = re.compile(r"^[A-Za-z0-9._+\-]+$")
_SAFE_NAME_WITH_UNDERSCORE = re.compile(r"^[A-Za-z0-9._+\-_]+$")


def _safe_filename(name: str) -> str:
    if not name or name in (".", "..") or "/" in name or "\\" in name:
        abort(400, description=f"unsafe filename: {name!r}")
    if not _SAFE_NAME_WITH_UNDERSCORE.match(name):
        abort(400, description=f"filename has disallowed characters: {name!r}")
    return name


def _safe_condition(cond: str) -> str:
    if cond not in RENDERERS:
        abort(400, description=f"unknown condition: {cond!r}")
    return cond


# ──────────────────────────────────────────────────────────────────────
# Flask app
# ──────────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder=str(VIEWER_DIR / "static"))


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


_GENERATED_ROOT = REPO_ROOT / "generated"


def _set_dir_count(p: Path) -> int:
    if not p.is_dir():
        return 0
    return sum(
        1 for f in p.iterdir()
        if f.is_file() and f.suffix == ".json" and f.name != "manifest.json"
    )


def _infer_set_condition(set_dir: Path) -> Optional[str]:
    """Return the condition a prompt-set directory belongs to.

    Priority:
      1. manifest.json's ``condition`` field, if it matches a registered name.
      2. Longest registered condition name that is a prefix of the dir name.
      3. None.
    """
    mf = set_dir / "manifest.json"
    if mf.is_file():
        try:
            data = json.loads(mf.read_text())
            cond = data.get("condition")
            if cond in RENDERERS:
                return cond
        except Exception:
            pass
    # Prefix match — prefer the longest-matching registered name
    name = set_dir.name
    candidates = sorted(RENDERERS.keys(), key=len, reverse=True)
    for c in candidates:
        if name == c or name.startswith(c + "_"):
            return c
    return None


def _list_all_sets() -> List[Dict]:
    """Every directory under generated/, each attributed to a condition
    when possible. Order: condition registry order, then alphabetical."""
    if not _GENERATED_ROOT.is_dir():
        return []
    sets = []
    for p in sorted(_GENERATED_ROOT.iterdir()):
        if not p.is_dir() or p.name.startswith("."):
            continue
        cond = _infer_set_condition(p)
        sets.append({
            "name": p.name,
            "path": f"generated/{p.name}",
            "condition": cond,
            "is_default": cond is not None and p.name == cond,
            "file_count": _set_dir_count(p),
        })
    # Sort: by condition (registry order, unknown last), then name
    cond_order = {c: i for i, c in enumerate(RENDERERS.keys())}
    sets.sort(key=lambda s: (
        cond_order.get(s["condition"], len(cond_order)),
        s["name"],
    ))
    return sets


def _resolve_set_dir(set_name: str) -> Path:
    """Safely resolve a set name to an absolute path under generated/."""
    if not set_name or "/" in set_name or "\\" in set_name or ".." in set_name:
        abort(400, description=f"unsafe set name: {set_name!r}")
    p = (_GENERATED_ROOT / set_name).resolve()
    gen_resolved = _GENERATED_ROOT.resolve()
    if not str(p).startswith(str(gen_resolved)):
        abort(400, description="set must stay within generated/")
    if not p.is_dir():
        abort(404, description=f"set not found: {set_name}")
    return p


@app.route("/api/conditions")
def api_conditions():
    """List registered conditions (presets for the Generate modal),
    plus the prompt-sets that currently exist on disk for each.

    The ``file_count`` per condition is the SUM across all sets that
    map to that condition (default dir plus any ``generated/{cond}_*``
    variants). Sets not attributable to any condition appear under
    ``unknown`` in ``/api/sets``.
    """
    all_sets = _list_all_sets()
    sets_by_cond: Dict[str, List[Dict]] = {}
    for s in all_sets:
        if s["condition"]:
            sets_by_cond.setdefault(s["condition"], []).append(s)

    out = []
    for name, spec in RENDERERS.items():
        sets = sets_by_cond.get(name, [])
        out.append({
            "name": name,
            "label": spec["label"],
            "description": spec["description"],
            "default_out_dir": spec["default_out_dir"],
            "file_count": sum(s["file_count"] for s in sets),
            "sets": sets,
            "knobs": spec["knobs"],
        })
    return jsonify(out)


@app.route("/api/sets")
def api_sets():
    """Flat list of every prompt-set directory on disk, attributed to
    a condition where possible. Useful for a set picker that doesn't
    hide sets whose dir name doesn't match a canonical condition."""
    return jsonify({"sets": _list_all_sets()})


@app.route("/api/prompts")
def api_prompts():
    """List prompt files in a set.

    Query params:
      set: directory name under generated/ (preferred). If omitted,
        falls back to the condition's default_out_dir.
      condition: registered renderer name. Required when ``set`` is
        not given.
      scenario: optional scenario_id prefix filter (e.g. "AG-01")
      variant: optional evidence_variant filter
      limit: max entries returned (default 2000)
    """
    set_name = request.args.get("set", "").strip()
    cond_arg = request.args.get("condition", "").strip()
    scenario = request.args.get("scenario", "").strip()
    variant = request.args.get("variant", "").strip()
    limit = int(request.args.get("limit", 2000))

    if set_name:
        out_dir = _resolve_set_dir(set_name)
        cond = _infer_set_condition(out_dir) or ""
    elif cond_arg:
        cond = _safe_condition(cond_arg)
        out_dir = REPO_ROOT / RENDERERS[cond]["default_out_dir"]
    else:
        abort(400, description="must supply either 'set' or 'condition'")
    if not out_dir.is_dir():
        return jsonify({"files": [], "total": 0, "out_dir": str(out_dir.relative_to(REPO_ROOT))})

    files = []
    for p in sorted(out_dir.iterdir()):
        if not p.is_file() or p.suffix != ".json" or p.name == "manifest.json":
            continue
        name = p.name
        if scenario and not name.startswith(scenario):
            continue
        # Filename layout: {sid}_{variant}_{perm}...  — variant is 2nd underscore-field
        parts = name[:-5].split("_")
        file_variant = parts[1] if len(parts) >= 2 else ""
        if variant and file_variant != variant:
            continue
        files.append({
            "file": name,
            "scenario_id": parts[0] if parts else "",
            "variant": file_variant,
            "size_bytes": p.stat().st_size,
        })

    total = len(files)
    return jsonify({
        "files": files[:limit],
        "total": total,
        "truncated": total > limit,
        "out_dir": str(out_dir.relative_to(REPO_ROOT)),
        "set": out_dir.name,
        "condition": cond,
    })


@app.route("/api/prompt")
def api_prompt():
    """Load a single prompt JSON. Accepts either ``set`` (preferred) or
    ``condition`` (falls back to the condition's default dir)."""
    set_name = request.args.get("set", "").strip()
    cond_arg = request.args.get("condition", "").strip()
    fname = _safe_filename(request.args.get("file", ""))
    if set_name:
        out_dir = _resolve_set_dir(set_name)
    elif cond_arg:
        cond = _safe_condition(cond_arg)
        out_dir = REPO_ROOT / RENDERERS[cond]["default_out_dir"]
    else:
        abort(400, description="must supply either 'set' or 'condition'")
    path = out_dir / fname
    if not path.is_file():
        abort(404, description=f"not found: {out_dir.name}/{fname}")

    data = json.loads(path.read_text())
    # Parse the system prompt back into turns if it's a conversation-history
    # style prompt. The renderers use lines like `[YYYY-MM-DD HH:MM:SS] Role: content`.
    turns = _parse_system_prompt_turns(data.get("system_prompt", ""))
    return jsonify({
        "system_prompt": data.get("system_prompt", ""),
        "user_message": data.get("user_message", ""),
        "metadata": data.get("metadata", {}),
        "parsed_turns": turns,
    })


_TURN_RE = re.compile(
    r"^\[(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\] (?P<role>User|Assistant): (?P<content>.*)$"
)


def _parse_system_prompt_turns(system_prompt: str) -> List[Dict]:
    """Heuristic: extract `[ts] Role: content` lines from the system prompt.

    Multi-line turns are supported: text on subsequent lines that doesn't
    match the turn-header regex is appended to the previous turn's content.
    Returns [] if the system prompt isn't in the expected format.
    """
    turns: List[Dict] = []
    if "Below is the conversation history" not in system_prompt:
        return turns
    for line in system_prompt.splitlines():
        m = _TURN_RE.match(line)
        if m:
            turns.append({
                "timestamp": m.group("ts"),
                "role": m.group("role").lower(),
                "content": m.group("content"),
            })
        else:
            if turns and line.strip():
                turns[-1]["content"] += "\n" + line
    return turns


@app.route("/api/scenarios")
def api_scenarios():
    """List scenario ids (for the filter dropdown)."""
    if not SCENARIOS_TSV.is_file():
        return jsonify({"scenarios": []})
    try:
        # Minimal TSV parse — first column = scenario_id.
        lines = SCENARIOS_TSV.read_text().splitlines()
        if not lines:
            return jsonify({"scenarios": []})
        header = lines[0].split("\t")
        sid_col = header.index("scenario_id") if "scenario_id" in header else 0
        status_col = header.index("status") if "status" in header else None
        check_col = header.index("check_personalization") if "check_personalization" in header else None
        sids = []
        for row in lines[1:]:
            cells = row.split("\t")
            if len(cells) <= sid_col:
                continue
            if status_col is not None and len(cells) > status_col and cells[status_col] == "reject":
                continue
            if check_col is not None and len(cells) > check_col and cells[check_col].upper() != "TRUE":
                continue
            sids.append(cells[sid_col])
        return jsonify({"scenarios": sorted(set(sids))})
    except Exception as e:
        return jsonify({"scenarios": [], "error": str(e)})


@app.route("/api/render", methods=["POST"])
def api_render():
    """Kick off a renderer subprocess. Returns job_id.

    Body JSON:
      {
        "condition": "fixed_locations",
        "out_dir": "generated/fixed_locations_test",   # optional
        "knobs": { knob_name: value, ... }
      }
    """
    body = request.get_json(force=True, silent=True) or {}
    cond = _safe_condition(body.get("condition", ""))
    spec = RENDERERS[cond]

    # Resolve out_dir — must live under generated/
    raw_out = body.get("out_dir") or spec["default_out_dir"]
    if raw_out.startswith("/") or ".." in Path(raw_out).parts:
        abort(400, description=f"unsafe out_dir: {raw_out!r}")
    if not raw_out.startswith("generated/"):
        abort(400, description="out_dir must start with 'generated/'")
    out_dir_abs = (REPO_ROOT / raw_out).resolve()
    if not str(out_dir_abs).startswith(str((REPO_ROOT / "generated").resolve())):
        abort(400, description="out_dir must stay within generated/")

    # Build CLI args from knobs
    script_path = REPO_ROOT / spec["script"]
    if not script_path.is_file():
        abort(500, description=f"renderer script missing: {spec['script']}")

    cmd: List[str] = [sys.executable, str(script_path), "--out-dir", str(out_dir_abs)]
    knobs_in = body.get("knobs", {}) or {}
    for knob in spec["knobs"]:
        name = knob["name"]  # already dash-form, matches argparse flag
        flag = "--" + name
        if name not in knobs_in:
            continue
        val = knobs_in[name]
        if knob["type"] == "bool":
            if bool(val):
                cmd.append(flag)
            # Else: omit — argparse store_true default is False.
        elif knob["type"] == "list_float":
            if isinstance(val, list):
                s = ",".join(str(x) for x in val)
            else:
                s = str(val)
            cmd += [flag, s]
        elif knob["type"] == "multi_choice":
            if isinstance(val, list):
                cmd += [flag, ",".join(val)]
            else:
                cmd += [flag, str(val)]
        else:
            cmd += [flag, str(val)]

    job_id = uuid.uuid4().hex[:12]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    with JOBS_LOCK:
        JOBS[job_id] = {
            "id": job_id,
            "condition": cond,
            "cmd": cmd,
            "out_dir": raw_out,
            "status": "queued",
            "started_at": time.time(),
            "stdout": "",
            "stderr": "",
        }
    t = threading.Thread(target=_run_subprocess_job, args=(job_id, cmd, env), daemon=True)
    t.start()
    return jsonify({"job_id": job_id, "cmd": cmd})


@app.route("/api/canon", methods=["POST"])
def api_canon():
    """Kick off every registered canon preset in parallel. Each preset
    writes to its own ``generated/canon_*`` dir and is tracked as its
    own job. Returns a list of job_ids the UI can poll collectively."""
    canon_names = [n for n, s in RENDERERS.items() if s.get("canon")]
    job_ids = []
    for name in canon_names:
        spec = RENDERERS[name]
        raw_out = spec["default_out_dir"]
        out_dir_abs = (REPO_ROOT / raw_out).resolve()
        script_path = REPO_ROOT / spec["script"]
        cmd: List[str] = [sys.executable, str(script_path), "--out-dir", str(out_dir_abs)]
        for knob in spec["knobs"]:
            kn = knob["name"]
            flag = "--" + kn
            val = knob.get("default")
            t = knob.get("type")
            if t == "bool":
                if val:
                    cmd.append(flag)
            elif t == "list_float":
                if isinstance(val, list):
                    cmd += [flag, ",".join(str(x) for x in val)]
                elif val:
                    cmd += [flag, str(val)]
            elif t == "multi_choice":
                if isinstance(val, list):
                    cmd += [flag, ",".join(val)]
                elif val:
                    cmd += [flag, str(val)]
            elif t in ("int",) and val == 0:
                # n_d=0 / n_p=0 / n_l=0 need to be passed explicitly so
                # mixer.py gets them instead of argparse defaults.
                cmd += [flag, "0"]
            else:
                if val is None or val == "":
                    continue
                cmd += [flag, str(val)]
        # Skip if target dir already non-empty — surface a clear error
        # instead of silently refusing later.
        if out_dir_abs.is_dir():
            existing = [p for p in out_dir_abs.iterdir() if not p.name.startswith(".")]
            if existing:
                with JOBS_LOCK:
                    jid = uuid.uuid4().hex[:12]
                    JOBS[jid] = {
                        "id": jid, "condition": name, "cmd": cmd,
                        "out_dir": raw_out, "status": "error",
                        "returncode": 1,
                        "started_at": time.time(),
                        "finished_at": time.time(),
                        "stdout": "",
                        "stderr": (
                            f"{raw_out} already has {len(existing)} files. "
                            "Delete or rename that dir before re-running canon."
                        ),
                    }
                    job_ids.append(jid)
                continue

        job_id = uuid.uuid4().hex[:12]
        env = os.environ.copy()
        env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        with JOBS_LOCK:
            JOBS[job_id] = {
                "id": job_id, "condition": name, "cmd": cmd,
                "out_dir": raw_out, "status": "queued",
                "started_at": time.time(),
                "stdout": "", "stderr": "",
            }
        t = threading.Thread(target=_run_subprocess_job, args=(job_id, cmd, env), daemon=True)
        t.start()
        job_ids.append(job_id)
    return jsonify({"job_ids": job_ids, "canon_names": canon_names})


@app.route("/api/render/status")
def api_render_status():
    job_id = request.args.get("job_id", "")
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            abort(404, description=f"unknown job_id: {job_id}")
        # Return a copy to avoid lock issues
        return jsonify(dict(job))


@app.route("/api/render/jobs")
def api_render_jobs():
    """List recent jobs (most recent first)."""
    with JOBS_LOCK:
        jobs = sorted(
            JOBS.values(),
            key=lambda j: j.get("started_at", 0),
            reverse=True,
        )
        # Trim verbose fields in list view
        summary = [
            {
                "id": j["id"],
                "condition": j["condition"],
                "status": j["status"],
                "out_dir": j.get("out_dir"),
                "started_at": j.get("started_at"),
                "finished_at": j.get("finished_at"),
                "returncode": j.get("returncode"),
            }
            for j in jobs
        ]
    return jsonify({"jobs": summary})


# ──────────────────────────────────────────────────────────────────────
# Entrypoint
# ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5057)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    print(f"LCVB viewer running at http://{args.host}:{args.port}/")
    print(f"  repo root:     {REPO_ROOT}")
    print(f"  generated dir: {GENERATED_DIR}")
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
