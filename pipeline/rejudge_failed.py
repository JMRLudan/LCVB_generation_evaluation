#!/usr/bin/env python3
"""
rejudge_failed.py — re-run ONLY the judge on rows whose subject response
succeeded but whose judge call failed (parse_error=1 + raw_response does not
start with ``ERROR:``).

Why: under sustained parallel load the Haiku judge exhausted its 3-retry
rate-limit budget on ~5% of items across the canon run. The subject responses
are valid and expensive — we don't want to re-issue them. This script makes
only judge calls, updates the rows in place, and preserves subject outputs.

Flags:
    --run-id YYYYMMDD_HHMMSS   (required) shared run id across all 5 presets
    --model claude-haiku-...   judge model (default matches pipeline JUDGE_MODEL)
    --concurrency N            judge semaphore (default 10)
    --max-retries N            judge retry budget (default 6, up from 3)
    --dry-run                  report what would be re-judged, skip API calls

Writes: rewrites each preset's results.tsv in place; backs up the original
to ``results.tsv.pre_rejudge_<ts>`` first.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

csv.field_size_limit(sys.maxsize)

from eval_pipeline import load_scenarios, SCENARIOS_TSV, score_result, get_seeds_by_indices, enumerate_permutations  # noqa: E402
import eval_pipeline  # noqa: E402  (writeable JUDGE_MODE attribute)
from openrouter_client import OpenRouterClient  # noqa: E402
from multi_model_runner import judge_response, JUDGE_MODEL  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent.parent
RUNS_DIR = BASE_DIR / "data" / "runs"

CANON_PRESETS = [
    "canon_direct",
    "canon_no_distractor",
    "canon_uniform_short",
    "canon_uniform_medium",
    "canon_uniform_long",
]


def build_query_with_options(scenario: dict) -> str:
    q = scenario.get("benign_triggering_query", "")
    a = scenario.get("choice_a_label", "")
    b = scenario.get("choice_b_label", "")
    return f"{q}\n\nOption A: {a}\nOption B: {b}"


def is_failed_judge(row: dict) -> bool:
    """True if subject succeeded but judge failed."""
    pe = (row.get("parse_error") or "").strip()
    raw = row.get("raw_response") or ""
    if pe != "1":
        return False
    if raw.startswith("ERROR:"):
        return False
    # Belt-and-suspenders: explanation tends to be exactly "(judge error)".
    exp = row.get("explanation") or ""
    return exp == "(judge error)" or pe == "1"


def apply_judge_result(row: dict, parsed: dict, expected: str) -> None:
    """Mutate row in place with fresh judge output + scoring."""
    scores = score_result(parsed, expected)
    row["recommendation"] = parsed.get("recommendation") or ""
    row["flagged"] = parsed.get("flagged") or ""
    row["constraint_mentioned"] = parsed.get("constraint_mentioned") or ""
    row["heavily_modified"] = parsed.get("heavily_modified") or ""
    # Only present in with_analysis mode; left blank in pure_eval.
    if "cited_user_info" in row or parsed.get("cited_user_info") is not None:
        row["cited_user_info"] = parsed.get("cited_user_info") or ""
    row["explanation"] = (parsed.get("explanation") or "").replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
    pe = parsed.get("parse_error", False)
    row["parse_error"] = "1" if pe else "0"
    # Scores are bool|None; normalize to TSV form
    for k in ("vigilance", "general_flag", "false_alarm", "choice_correct", "abstained"):
        v = scores.get(k)
        if v is None:
            row[k] = ""
        elif isinstance(v, bool):
            row[k] = "1" if v else "0"
        else:
            row[k] = str(v)
    row["judge_input_tokens"] = str(parsed.get("judge_input_tokens", 0) or 0)
    row["judge_output_tokens"] = str(parsed.get("judge_output_tokens", 0) or 0)


async def rejudge_preset(
    preset: str, run_id: str, model: str, scenarios: dict,
    client: OpenRouterClient, sem: asyncio.Semaphore, max_retries: int,
    dry_run: bool, include_all: bool = False,
) -> dict:
    """Re-judge failed (or all) rows for one preset. Returns summary dict.

    Args:
        include_all: If True, re-judge every row whose subject succeeded
            (raw_response not starting with ERROR). Default False keeps the
            v1 behavior of touching only parse_error=1 rows.
    """
    # Find the run dir. Model slug is stored on disk with '/' → '_'.
    model_fs = model.replace("/", "_")
    run_dir = RUNS_DIR / preset / model_fs / run_id
    tsv_path = run_dir / "results.tsv"
    if not tsv_path.exists():
        return {"preset": preset, "status": "missing", "path": str(tsv_path)}

    with open(tsv_path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    # Auto-add the cited_user_info column when running in with_analysis mode
    # so DictWriter doesn't drop the new field. Insert it after heavily_modified
    # to match the EvalResult dataclass ordering.
    if eval_pipeline.JUDGE_MODE == "with_analysis" and "cited_user_info" not in fieldnames:
        try:
            insert_at = fieldnames.index("heavily_modified") + 1
        except ValueError:
            insert_at = len(fieldnames)
        fieldnames = list(fieldnames)
        fieldnames.insert(insert_at, "cited_user_info")
        for r in rows:
            r.setdefault("cited_user_info", "")

    if include_all:
        failed = [r for r in rows
                  if not (r.get("raw_response", "") or "").startswith(("ERROR", '"ERROR'))]
    else:
        failed = [r for r in rows if is_failed_judge(r)]
    if not failed:
        return {"preset": preset, "total_rows": len(rows), "to_rejudge": 0}

    summary = {
        "preset": preset, "total_rows": len(rows),
        "to_rejudge": len(failed), "succeeded": 0,
        "still_failing": 0, "errors": 0,
    }

    if dry_run:
        summary["status"] = "dry-run"
        return summary

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    shutil.copy2(tsv_path, tsv_path.with_suffix(f".tsv.pre_rejudge_{ts}"))

    async def process(row: dict) -> None:
        sid = row["scenario_id"]
        variant = row["evidence_variant"]
        scenario = scenarios.get(sid)
        if not scenario:
            summary["errors"] += 1
            return
        query_with_options = build_query_with_options(scenario)
        constraint_desc = scenario.get("constraint_description", "")
        # Re-derive evidence seeds for this row from scenario + permutation.
        # Needed by the with_analysis judge mode; harmless if the active mode
        # is pure_eval (the seeds are simply ignored downstream).
        perm_label = row.get("permutation", "")
        seed_indices: dict = {}
        for perm_l, idx in enumerate_permutations(scenario, variant):
            if perm_l == perm_label:
                seed_indices = idx
                break
        evidence_seeds = get_seeds_by_indices(scenario, variant, seed_indices) if seed_indices else []

        # Unescape raw_response (run.py escapes \n \r \t for TSV safety; judge prompt
        # should see the real response).
        raw = (row.get("raw_response") or "").replace("\\n", "\n").replace("\\r", "\r").replace("\\t", "\t")
        expected = row.get("expected_answer") or ("C" if "C" in variant else variant)
        async with sem:
            parsed = await judge_response(
                client, raw, query_with_options, constraint_desc,
                evidence_seeds=evidence_seeds,
                max_retries=max_retries,
            )
        apply_judge_result(row, parsed, expected)
        if parsed.get("parse_error"):
            summary["still_failing"] += 1
        else:
            summary["succeeded"] += 1

    t0 = time.monotonic()
    tasks = [process(r) for r in failed]
    done = 0
    for coro in asyncio.as_completed(tasks):
        await coro
        done += 1
        if done % 25 == 0 or done == len(tasks):
            elapsed = time.monotonic() - t0
            print(
                f"  [{preset}] {done}/{len(tasks)}  "
                f"ok={summary['succeeded']}  still_failing={summary['still_failing']}  "
                f"elapsed={elapsed:.1f}s  cost={client.total_cost():.4f}",
                flush=True,
            )

    # Rewrite TSV with updated rows (rows was modified in place; failed rows
    # point at the same dicts).
    with open(tsv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    summary["elapsed_s"] = round(time.monotonic() - t0, 1)
    return summary


async def main_async(args: argparse.Namespace) -> int:
    scenarios = load_scenarios(SCENARIOS_TSV, validated_only=True)
    print(f"Loaded {len(scenarios)} scenarios")

    sem = asyncio.Semaphore(args.concurrency)
    total_cost_start = 0.0
    async with OpenRouterClient(
        run_id=f"rejudge_{args.run_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    ) as client:
        if not args.dry_run:
            client.validate_pricing()
        total_cost_start = client.total_cost()

        summaries = []
        for preset in CANON_PRESETS:
            print(f"\n=== {preset} ===")
            s = await rejudge_preset(
                preset, args.run_id, args.model, scenarios,
                client, sem, args.max_retries, args.dry_run,
                include_all=args.all,
            )
            print(f"  summary: {s}")
            summaries.append(s)

        total_cost = client.total_cost() - total_cost_start

    print("\n=== GRAND TOTAL ===")
    total_to = sum(s.get("to_rejudge", 0) for s in summaries)
    total_ok = sum(s.get("succeeded", 0) for s in summaries)
    total_sf = sum(s.get("still_failing", 0) for s in summaries)
    total_er = sum(s.get("errors", 0) for s in summaries)
    print(f"  to_rejudge={total_to}  succeeded={total_ok}  still_failing={total_sf}  errors={total_er}")
    print(f"  rejudge cost: ${total_cost:.4f}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--model", default=JUDGE_MODEL)
    ap.add_argument("--concurrency", type=int, default=10)
    ap.add_argument("--max-retries", type=int, default=6)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--all", action="store_true",
        help=(
            "Re-judge every row whose subject succeeded (instead of only "
            "parse_error=1 rows). Use with --judge-mode with_analysis to "
            "populate cited_user_info on a previously pure_eval'd run."
        ),
    )
    ap.add_argument(
        "--judge-mode",
        choices=["pure_eval", "with_analysis"],
        default=None,
        help="Override the judge prompt variant. See run.py --help.",
    )
    args = ap.parse_args()
    if args.judge_mode:
        eval_pipeline.JUDGE_MODE = args.judge_mode
        print(f"Judge mode: {args.judge_mode}")
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
