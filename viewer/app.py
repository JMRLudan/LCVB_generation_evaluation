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
            "Direct ablation — constraint stated inline in the user message, "
            "no conversation-history distractor."
        ),
        "knobs": [
            {
                "name": "num-distractor-draws",
                "label": "Item replicates per (scenario, variant, perm)",
                "type": "int",
                "default": 1,
                "min": 1,
                "max": 5,
            },
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
                "name": "num-distractor-draws",
                "label": "Item replicates per (scenario, variant, perm)",
                "type": "int",
                "default": 1,
                "min": 1,
                "max": 5,
            },
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
            "Uniform-placement sweep — single deterministic placement_frac per "
            "item, sampled uniformly from [0, 1] via sha256 hash of item id."
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


@app.route("/api/conditions")
def api_conditions():
    """List available conditions, with counts of existing generated files."""
    out = []
    for name, spec in RENDERERS.items():
        out_dir = REPO_ROOT / spec["default_out_dir"]
        count = 0
        if out_dir.is_dir():
            count = sum(
                1 for p in out_dir.iterdir()
                if p.is_file() and p.suffix == ".json" and p.name != "manifest.json"
            )
        out.append({
            "name": name,
            "label": spec["label"],
            "description": spec["description"],
            "default_out_dir": spec["default_out_dir"],
            "file_count": count,
            "knobs": spec["knobs"],
        })
    return jsonify(out)


@app.route("/api/prompts")
def api_prompts():
    """List prompt files under a condition's default out_dir.

    Query params:
      condition: one of the registered renderer names
      scenario: optional scenario_id prefix filter (e.g. "AG-01")
      variant: optional evidence_variant filter
      limit: max entries returned (default 2000)
    """
    cond = _safe_condition(request.args.get("condition", ""))
    scenario = request.args.get("scenario", "").strip()
    variant = request.args.get("variant", "").strip()
    limit = int(request.args.get("limit", 2000))

    out_dir = REPO_ROOT / RENDERERS[cond]["default_out_dir"]
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
    })


@app.route("/api/prompt")
def api_prompt():
    """Load a single prompt JSON."""
    cond = _safe_condition(request.args.get("condition", ""))
    fname = _safe_filename(request.args.get("file", ""))
    path = REPO_ROOT / RENDERERS[cond]["default_out_dir"] / fname
    if not path.is_file():
        abort(404, description=f"not found: {cond}/{fname}")

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
