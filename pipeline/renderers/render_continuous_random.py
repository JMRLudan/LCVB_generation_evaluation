#!/usr/bin/env python3
"""
render_continuous_random.py — uniform random placement in a long history.
==========================================================================

Answers the question: if we sample evidence placement continuously
(uniform on [0, 1]) in a long conversation rather than at 5 grid
cutoffs, how does the primacy-recency curve behave?

Differences vs `render_fixed_locations` (grid sweep):

  1. `placement_frac ~ Uniform(0, 1)`, one random placement per item,
     seeded deterministically from
     `sha256(scenario_id | variant | permutation | draw_idx | "continuous_random")`.
     So the rendered output is reproducible.
  2. A single larger char budget (`--char-budget`, default 224_000 ≈ 32K
     tokens at Haiku's ~7 chars/tok).
  3. One distractor per prompt, taken from the deduplicated pool via
     the same `assign_distractors(seed=4232026)` path as the grid sweep.
     No stitching of multiple distractors in a single prompt — see the
     separate `render_stitched_locations.py` for that harder variant.

Output: one JSON per (scenario, variant, permutation, draw_idx) under
`generated/continuous_random/`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval_pipeline import (  # noqa: E402
    load_scenarios, enumerate_permutations, get_seeds_by_indices,
    SCENARIOS_TSV,
)
from distractor_pool import (  # noqa: E402
    load_pool, assign_distractors, ASSIGNMENT_SEED, assignment_summary,
)
from renderers.assembly import (  # noqa: E402
    pair_units_from_turns, truncate_pairs_to_budget,
    evidence_pairs_from_seeds, assemble_at_pair_boundary,
    turns_to_text, build_system_prompt, assert_alternation,
)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
OUT_DIR = BASE_DIR / "generated" / "continuous_random"

DEFAULT_CHAR_BUDGET = 224_000
PREAMBLE_RESERVE = 100
EVIDENCE_RESERVE = 800
USER_MSG_RESERVE = 400
PLACEMENT_TAG = "continuous_random"


def _deterministic_frac(sid: str, variant: str, perm: str, draw_idx: int) -> float:
    """Reproducible Uniform(0, 1) from a stable key."""
    key = f"{sid}|{variant}|{perm}|{draw_idx}|{PLACEMENT_TAG}".encode()
    h = hashlib.sha256(key).hexdigest()
    return int(h[:16], 16) / float(1 << 64)


def render(
    out_dir: Path,
    char_budget: int = DEFAULT_CHAR_BUDGET,
    num_distractor_draws: int = 1,
    c_only: bool = False,
) -> dict:
    scenarios = load_scenarios(SCENARIOS_TSV, validated_only=True)
    pool = load_pool()
    draws = assign_distractors(
        list(scenarios.keys()), pool, num_draws=num_distractor_draws
    )

    variants = ["C", "A+C", "B+C"] if c_only else ["C", "A+C", "B+C", "A", "B"]

    if out_dir.exists():
        non_hidden = [p for p in out_dir.iterdir() if not p.name.startswith(".")]
        if non_hidden:
            raise FileExistsError(
                f"{out_dir} has existing content ({len(non_hidden)} files). "
                "Refusing to overwrite. Delete or move first, then re-run."
            )
    out_dir.mkdir(parents=True, exist_ok=True)

    distractor_budget = (
        char_budget - PREAMBLE_RESERVE - EVIDENCE_RESERVE - USER_MSG_RESERVE
    )

    built = 0
    char_samples = []
    placement_samples = []
    # per-(draw, sid) truncated pair list cache
    pair_cache: dict = {}

    for draw_idx, assignment in enumerate(draws):
        for sid in sorted(scenarios.keys()):
            scenario = scenarios[sid]
            query = scenario["benign_triggering_query"]
            choice_a = scenario["choice_a_label"]
            choice_b = scenario["choice_b_label"]
            constraint = scenario.get("constraint_description", "")
            d_hash = assignment[sid]
            distractor = pool[d_hash]

            cache_key = (draw_idx, sid)
            if cache_key not in pair_cache:
                all_pairs = pair_units_from_turns(distractor.turns)
                pair_cache[cache_key] = truncate_pairs_to_budget(
                    all_pairs, distractor_budget
                )
            dist_pairs = pair_cache[cache_key]

            for variant in variants:
                expected = "C" if "C" in variant else variant
                for perm_label, seed_idx in enumerate_permutations(
                    scenario, variant
                ):
                    seeds = get_seeds_by_indices(scenario, variant, seed_idx)
                    ev_pairs = evidence_pairs_from_seeds(seeds)
                    dp_copy = [list(p) for p in dist_pairs]
                    frac = _deterministic_frac(sid, variant, perm_label, draw_idx)
                    combined, insert_idx, n_pairs = assemble_at_pair_boundary(
                        dp_copy, ev_pairs, placement_frac=frac,
                    )
                    assert_alternation(combined)

                    conv_text = turns_to_text(combined)
                    system_prompt = build_system_prompt(conv_text)
                    user_msg = (
                        f"{query}\n\n"
                        f"Option A: {choice_a}\n"
                        f"Option B: {choice_b}"
                    )
                    q_with_opts = (
                        f"{query}\n\nOption A: {choice_a}\n"
                        f"Option B: {choice_b}"
                    )

                    meta = {
                        "scenario_id": sid,
                        "evidence_variant": variant,
                        "permutation": perm_label,
                        "draw_idx": draw_idx,
                        "placement_frac": frac,
                        "char_budget": char_budget,
                        "expected_answer": expected,
                        "query_with_options": q_with_opts,
                        "constraint_description": constraint,
                        "distractor_hash": d_hash,
                        "distractor_domain": distractor.domain,
                        "n_distractor_pairs": n_pairs,
                        "insert_pair_idx": insert_idx,
                        "input_char_len": len(system_prompt) + len(user_msg),
                        "condition": "continuous_random",
                        "assignment_seed": ASSIGNMENT_SEED,
                        "placement_tag": PLACEMENT_TAG,
                    }
                    record = {
                        "system_prompt": system_prompt,
                        "user_message": user_msg,
                        "metadata": meta,
                    }
                    fname = (
                        f"{sid}_{variant}_{perm_label}_draw{draw_idx}.json"
                    )
                    (out_dir / fname).write_text(
                        json.dumps(record, indent=2, ensure_ascii=False)
                    )
                    built += 1
                    char_samples.append(meta["input_char_len"])
                    placement_samples.append(frac)

    manifest = {
        "condition": "continuous_random",
        "description": (
            "Uniform random placement in a long (char_budget-sized) history. "
            "One distractor per prompt."
        ),
        "num_scenarios": len(scenarios),
        "variants": variants,
        "char_budget": char_budget,
        "num_distractor_draws": num_distractor_draws,
        "num_prompts": built,
        "assignment_seed": ASSIGNMENT_SEED,
        "placement_tag": PLACEMENT_TAG,
        "assignment_summary": assignment_summary(draws),
        "input_chars": {
            "min": min(char_samples) if char_samples else 0,
            "max": max(char_samples) if char_samples else 0,
            "avg": sum(char_samples) / len(char_samples) if char_samples else 0,
        },
        "placement_frac": {
            "min": min(placement_samples) if placement_samples else 0,
            "max": max(placement_samples) if placement_samples else 0,
            "avg": sum(placement_samples) / len(placement_samples)
                   if placement_samples else 0,
        },
        "built_at": datetime.now().isoformat(),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", type=str, default=str(OUT_DIR))
    ap.add_argument("--char-budget", type=int, default=DEFAULT_CHAR_BUDGET)
    ap.add_argument(
        "--num-distractor-draws", type=int, default=1,
        help="How many independent distractor assignments to render.",
    )
    ap.add_argument(
        "--c-only", action="store_true",
        help="Only render C-present variants.",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    manifest = render(
        out_dir=out_dir,
        char_budget=args.char_budget,
        num_distractor_draws=args.num_distractor_draws,
        c_only=args.c_only,
    )
    print(f"Built {manifest['num_prompts']} prompts → {out_dir}")
    print(
        f"  input chars: avg={manifest['input_chars']['avg']:,.0f}  "
        f"min={manifest['input_chars']['min']:,}  "
        f"max={manifest['input_chars']['max']:,}"
    )
    print(
        f"  placement_frac: avg={manifest['placement_frac']['avg']:.3f}  "
        f"min={manifest['placement_frac']['min']:.3f}  "
        f"max={manifest['placement_frac']['max']:.3f}"
    )


if __name__ == "__main__":
    main()
