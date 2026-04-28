#!/usr/bin/env python3
"""
run.py — execute a rendered condition against a model.
=======================================================

Reads prompt JSON files from one of the renderer output directories
(`generated/with_constraint/`, `generated/fixed_locations/`, or
`generated/continuous_random/`), sends each one to the target model via
the canonical `OpenRouterClient`, judges the response with Haiku, and
writes per-item rows to a TSV.

Supports:
  * Anthropic-native + OpenRouter routing (auto-detected by model slug).
  * Resume via `--run-id <existing_id>` — uses the pipe-delimited
    `item_key()` from `multi_model_runner` for checkpoint de-dup.
  * `--limit N` to cap items (for cost-bounded test runs).

Output layout:
  data/runs/<condition>/<model_slug>/<run_id>/
      results.tsv
      checkpoint.txt
      meta.json
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

csv.field_size_limit(sys.maxsize)

from eval_pipeline import EvalItem, score_result  # noqa: E402
from openrouter_client import OpenRouterClient  # noqa: E402
from multi_model_runner import (  # noqa: E402
    openrouter_chat, anthropic_chat, judge_response, item_key,
    load_checkpoint, save_checkpoint_entry,
    ANTHROPIC_MODELS,
)

BASE_DIR = Path(__file__).resolve().parent.parent
RUNS_DIR = BASE_DIR / "data" / "runs"

# T=0 by default for reproducibility: deterministic top-k sampling means re-runs
# with identical prompts yield identical responses, which is critical for canon
# regression checks and for reproducing per-scenario judge verdicts. Override
# via --temperature if a stochastic eval is wanted (e.g. measuring response
# variance for a single prompt).
DEFAULT_TEMPERATURE = 0.0


def load_items_from_dir(prompts_dir: Path) -> List[EvalItem]:
    """Load EvalItems from a flat directory of *.json prompt files.

    Skips `manifest.json` if present.
    """
    items = []
    for fp in sorted(prompts_dir.glob("*.json")):
        if fp.name == "manifest.json":
            continue
        d = json.loads(fp.read_text())
        m = d["metadata"]
        items.append(EvalItem(
            scenario_id=m["scenario_id"],
            evidence_variant=m["evidence_variant"],
            permutation=m["permutation"],
            expected_answer=m.get("expected_answer", "C"),
            system_prompt=d["system_prompt"],
            messages=[{"role": "user", "content": d["user_message"]}],
            query_with_options=m.get("query_with_options", d["user_message"]),
            constraint_description=m.get("constraint_description", ""),
            input_char_len=m.get(
                "input_char_len",
                len(d["system_prompt"]) + len(d["user_message"]),
            ),
        ))
    return items


async def run(
    model_slug: str,
    items: List[EvalItem],
    run_id: str,
    condition: str,
    runs_dir: Path = RUNS_DIR,
    concurrency: int = 10,
    temperature: float = DEFAULT_TEMPERATURE,
    cost_abort: float = 0.0,
):
    cond_dir = runs_dir / condition / model_slug.replace("/", "_") / run_id
    cond_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = cond_dir / "checkpoint.txt"
    results_path = cond_dir / "results.tsv"
    meta_path = cond_dir / "meta.json"

    completed_keys = load_checkpoint(checkpoint_path)
    remaining = [it for it in items if item_key(it) not in completed_keys]

    print(
        f"\n▶ {condition} ({model_slug}): "
        f"{len(items)} total, {len(completed_keys)} done, "
        f"{len(remaining)} remaining"
    )
    if not remaining:
        print("  ✓ Already complete!")
        return

    avg_input_tokens = (
        sum(it.input_char_len for it in remaining)
        / max(len(remaining), 1) / 4
    )
    print(f"  Avg input: ~{avg_input_tokens:.0f} tokens/item")

    with open(meta_path, "w") as f:
        json.dump({
            "model": model_slug, "run_id": run_id,
            "condition": condition,
            "total_items": len(items),
            "started": datetime.now().isoformat(),
            "temperature": temperature,
        }, f, indent=2)

    use_anthropic = model_slug in ANTHROPIC_MODELS
    sem = asyncio.Semaphore(concurrency)
    errors = 0
    completed_count = len(completed_keys)
    t_start = time.monotonic()
    aborted = False

    fieldnames = [
        "run_id", "condition", "scenario_id", "evidence_variant", "permutation",
        "expected_answer", "raw_response", "recommendation", "flagged",
        "constraint_mentioned", "heavily_modified", "explanation",
        "parse_error", "vigilance", "general_flag", "false_alarm",
        "choice_correct", "abstained", "input_tokens", "output_tokens",
        "judge_input_tokens", "judge_output_tokens", "latency_ms",
    ]
    file_is_new = not results_path.exists() or results_path.stat().st_size == 0
    results_file = open(results_path, "a", newline="")
    results_writer = csv.DictWriter(
        results_file, fieldnames=fieldnames, delimiter="\t"
    )
    if file_is_new:
        results_writer.writeheader()
        results_file.flush()

    lock = asyncio.Lock()

    try:
        async with OpenRouterClient(
            run_id=f"run_{condition}_{model_slug.replace('/', '_')}_{run_id}"
        ) as client:
            client.validate_pricing()

            async def process_item(item: EvalItem):
                nonlocal errors, completed_count, aborted
                if aborted:
                    return
                async with sem:
                    if aborted:
                        return
                    try:
                        if use_anthropic:
                            resp = await anthropic_chat(
                                client, model_slug,
                                item.system_prompt,
                                item.messages[0]["content"],
                                temperature=temperature,
                            )
                        else:
                            resp = await openrouter_chat(
                                client, model_slug,
                                item.system_prompt,
                                item.messages[0]["content"],
                                temperature=temperature,
                            )

                        if resp.get("error"):
                            errors += 1
                            row = {
                                "run_id": run_id, "condition": condition,
                                "scenario_id": item.scenario_id,
                                "evidence_variant": item.evidence_variant,
                                "permutation": item.permutation,
                                "expected_answer": item.expected_answer,
                                "raw_response": f"ERROR: {resp['error']}",
                                "recommendation": None, "flagged": None,
                                "constraint_mentioned": None,
                                "heavily_modified": None,
                                "explanation": None, "parse_error": True,
                                "vigilance": None, "general_flag": None,
                                "false_alarm": None, "choice_correct": None,
                                "abstained": None,
                                "input_tokens": 0, "output_tokens": 0,
                                "judge_input_tokens": 0,
                                "judge_output_tokens": 0,
                                "latency_ms": resp.get("latency_ms", 0),
                            }
                        else:
                            parsed = await judge_response(
                                client, resp["content"],
                                item.query_with_options,
                                item.constraint_description,
                            )
                            scores = score_result(parsed, item.expected_answer)
                            row = {
                                "run_id": run_id, "condition": condition,
                                "scenario_id": item.scenario_id,
                                "evidence_variant": item.evidence_variant,
                                "permutation": item.permutation,
                                "expected_answer": item.expected_answer,
                                "raw_response": resp["content"],
                                "recommendation": parsed.get("recommendation"),
                                "flagged": parsed.get("flagged"),
                                "constraint_mentioned":
                                    parsed.get("constraint_mentioned"),
                                "heavily_modified":
                                    parsed.get("heavily_modified"),
                                "explanation": parsed.get("explanation"),
                                "parse_error": parsed.get("parse_error", False),
                                "vigilance": scores["vigilance"],
                                "general_flag": scores["general_flag"],
                                "false_alarm": scores["false_alarm"],
                                "choice_correct": scores["choice_correct"],
                                "abstained": scores["abstained"],
                                "input_tokens": resp.get("input_tokens", 0),
                                "output_tokens": resp.get("output_tokens", 0),
                                "judge_input_tokens": parsed.get("judge_input_tokens", 0),
                                "judge_output_tokens": parsed.get("judge_output_tokens", 0),
                                "latency_ms": resp.get("latency_ms", 0),
                            }

                        # Normalize booleans / None for TSV-safe write.
                        for k, v in list(row.items()):
                            if isinstance(v, bool):
                                row[k] = "1" if v else "0"
                            elif v is None:
                                row[k] = ""
                            elif isinstance(v, str):
                                row[k] = (v.replace("\n", "\\n")
                                           .replace("\r", "\\r")
                                           .replace("\t", "\\t"))

                        async with lock:
                            results_writer.writerow(row)
                            results_file.flush()
                            if not resp.get("error"):
                                save_checkpoint_entry(
                                    checkpoint_path, item_key(item)
                                )
                            completed_count += 1
                            if completed_count % 25 == 0:
                                elapsed = time.monotonic() - t_start
                                rate = completed_count / elapsed if elapsed > 0 else 0
                                eta_s = (
                                    (len(items) - completed_count) / rate
                                    if rate > 0 else 0
                                )
                                cost_so_far = client.total_cost()
                                print(
                                    f"  {completed_count}/{len(items)}  "
                                    f"({rate:.1f}/s  ETA {eta_s/60:.1f}m)  "
                                    f"cost=${cost_so_far:.3f}  err={errors}",
                                    flush=True,
                                )
                                if cost_abort and cost_so_far >= cost_abort:
                                    print(
                                        f"  ! Aborting: cost ${cost_so_far:.3f} "
                                        f"≥ cap ${cost_abort:.3f}"
                                    )
                                    aborted = True
                    except Exception as e:  # noqa: BLE001
                        errors += 1
                        print(
                            f"  ERROR {item.scenario_id}: "
                            f"{type(e).__name__}: {e}",
                            flush=True,
                        )

            tasks = [process_item(it) for it in remaining]
            await asyncio.gather(*tasks)
    finally:
        results_file.close()

    elapsed = time.monotonic() - t_start
    print(
        f"\n  ✓ {condition} done: {completed_count}/{len(items)} in "
        f"{elapsed/60:.1f}m, {errors} errors"
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--prompts-dir", type=str, required=True,
        help="Directory of prompt *.json files produced by a render_*.py script.",
    )
    ap.add_argument(
        "--condition", type=str, default=None,
        help="Condition name (defaults to the prompts-dir basename).",
    )
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--concurrency", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0,
                    help="Max items (0 = all).")
    ap.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    ap.add_argument("--run-id", type=str, default=None,
                    help="Resume a specific run_id.")
    ap.add_argument("--runs-dir", type=str, default=None,
                    help="Override output runs directory (default: data/runs).")
    ap.add_argument("--cost-abort", type=float, default=0.0,
                    help="Abort the run if total cost exceeds this USD (0 = no cap).")
    ap.add_argument(
        "--judge-mode",
        choices=["pure_eval", "with_analysis"],
        default=None,
        help=(
            "Which judge prompt to use. 'pure_eval' (default) is the "
            "original 4-field schema used for the canon SR/GF/FA numbers. "
            "'with_analysis' adds the CITED_USER_INFO diagnostic field "
            "(judge sees the user's stated facts and decides whether the "
            "model referred back to them). New runs in with_analysis mode "
            "populate the cited_user_info column; pure_eval leaves it blank."
        ),
    )
    ap.add_argument("--run", action="store_true",
                    help="Actually execute (without this flag it's a dry run).")
    args = ap.parse_args()

    # Set the global judge mode before any judge calls happen.
    if args.judge_mode:
        import eval_pipeline
        eval_pipeline.JUDGE_MODE = args.judge_mode
        print(f"Judge mode: {args.judge_mode}")

    prompts_dir = Path(args.prompts_dir)
    condition = args.condition or prompts_dir.name
    runs_dir = Path(args.runs_dir) if args.runs_dir else RUNS_DIR
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")

    items = load_items_from_dir(prompts_dir)
    if args.limit > 0:
        items = items[: args.limit]

    print(f"Condition: {condition}")
    print(f"Model:     {args.model}")
    print(f"Prompts:   {prompts_dir}  ({len(items)} items)")
    print(f"Runs dir:  {runs_dir}")
    print(f"Run ID:    {run_id}")

    if not args.run:
        print("\nDRY RUN — pass --run to execute")
        return

    asyncio.run(run(
        model_slug=args.model,
        items=items,
        run_id=run_id,
        condition=condition,
        runs_dir=runs_dir,
        concurrency=args.concurrency,
        temperature=args.temperature,
        cost_abort=args.cost_abort,
    ))


if __name__ == "__main__":
    main()
