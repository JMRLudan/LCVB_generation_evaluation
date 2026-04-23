#!/usr/bin/env python3
"""
render_fixed_locations.py — the grid-sweep condition.
======================================================

Bury the constraint evidence at a small number of fixed relative depths in
a conversation history filled with one topically-unrelated distractor
conversation. Sweep over:

  depths        : {0.0, 0.25, 0.5, 0.75, 1.0}
                  (where in the distractor pair sequence to insert the
                   evidence. 0.0 = evidence at the very top of the
                   history; 1.0 = evidence at the very end, immediately
                   before the user's new query. Convention matches
                   `renderers/assembly.assemble_at_pair_boundary`.)
  haystack sizes: {short, long}  — controlled by two char budgets

Construction invariants enforced by this renderer (and shared with the
other renderers in this package; see `renderers/assembly.py` and the
README "Generation and stitching rules" section):

  * **One distractor per prompt.** No stitching of multiple distractor
    conversations here — see `render_stitched_locations.py` for the
    harder multi-source variant.
  * **Keep-beginning truncation.** Walk the distractor conversation
    pair-by-pair from the start; stop when adding the next pair would
    exceed the budget. Drop everything after. Never cut from the middle.
  * **Evidence is inserted at a pair boundary.** User→assistant pair
    units never split. The resulting turn list starts user, ends
    assistant, and strictly alternates — by construction, not by
    post-hoc fix-up.
  * **Distractor assignment is deterministic.** `assign_distractors`
    uses seed 4232026 and guarantees no two scenarios share the same
    distractor within a single draw.

Output:
  One JSON per (scenario, variant, seed_permutation, draw_idx, depth,
  haystack) under `generated/fixed_locations/`. Schema:

      { "system_prompt": "...",
        "user_message":  "<query>\\n\\nOption A: ...\\nOption B: ...",
        "metadata":      { ...sid, variant, permutation, draw_idx,
                           depth, haystack_size, distractor_hash, ... } }
"""

from __future__ import annotations

import argparse
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
OUT_DIR = BASE_DIR / "generated" / "fixed_locations"

# Depth grid — fraction of distractor pairs BEFORE evidence
# (0.0 = evidence at the very bottom / just before the query;
#  1.0 = evidence at the very top of the history)
DEFAULT_DEPTHS = (0.0, 0.25, 0.5, 0.75, 1.0)

# Two haystack sizes (char budgets for the full prompt, including
# preamble + evidence + user message overheads). Tuned against Haiku's
# ~7 chars/tok ratio for "short ~ 3-4K tok input" and "long ~ 32K tok".
DEFAULT_HAYSTACKS = {
    "short": 24_000,    # ~3-4K tokens
    "long":  224_000,   # ~32K tokens
}

# Overheads reserved off the top of the char budget
PREAMBLE_RESERVE = 100
EVIDENCE_RESERVE = 800
USER_MSG_RESERVE = 400


def _distractor_budget(char_budget: int) -> int:
    return char_budget - PREAMBLE_RESERVE - EVIDENCE_RESERVE - USER_MSG_RESERVE


def render(
    out_dir: Path,
    num_distractor_draws: int = 1,
    depths=DEFAULT_DEPTHS,
    haystacks=DEFAULT_HAYSTACKS,
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

    built = 0
    char_samples = []
    # Cache per-scenario truncated distractor pair lists — same within a draw
    # and haystack — so we don't re-truncate for each variant/permutation.
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

            # Truncate once per (draw, sid, haystack)
            pair_cache_key_src = (draw_idx, sid)
            if pair_cache_key_src not in pair_cache:
                pair_cache[pair_cache_key_src] = pair_units_from_turns(
                    distractor.turns
                )
            full_pairs = pair_cache[pair_cache_key_src]

            for haystack_name, char_budget in haystacks.items():
                dbudget = _distractor_budget(char_budget)
                trunc_key = (draw_idx, sid, haystack_name)
                if trunc_key not in pair_cache:
                    pair_cache[trunc_key] = truncate_pairs_to_budget(
                        full_pairs, dbudget
                    )
                dist_pairs = pair_cache[trunc_key]

                for variant in variants:
                    expected = "C" if "C" in variant else variant
                    for perm_label, seed_idx in enumerate_permutations(
                        scenario, variant
                    ):
                        seeds = get_seeds_by_indices(
                            scenario, variant, seed_idx
                        )
                        for depth in depths:
                            ev_pairs = evidence_pairs_from_seeds(seeds)
                            # Fresh pair copies per item so shared refs
                            # don't get their timestamps overwritten.
                            dp_copy = [list(p) for p in dist_pairs]
                            combined, insert_idx, n_pairs = \
                                assemble_at_pair_boundary(
                                    dp_copy, ev_pairs, placement_frac=depth,
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
                                "depth": depth,
                                "haystack_size": haystack_name,
                                "char_budget": char_budget,
                                "expected_answer": expected,
                                "query_with_options": q_with_opts,
                                "constraint_description": constraint,
                                "distractor_hash": d_hash,
                                "distractor_domain": distractor.domain,
                                "n_distractor_pairs": n_pairs,
                                "insert_pair_idx": insert_idx,
                                "input_char_len": len(system_prompt) + len(user_msg),
                                "condition": "fixed_locations",
                                "assignment_seed": ASSIGNMENT_SEED,
                            }
                            record = {
                                "system_prompt": system_prompt,
                                "user_message": user_msg,
                                "metadata": meta,
                            }
                            depth_tag = f"d{int(round(depth * 100)):03d}"
                            fname = (
                                f"{sid}_{variant}_{perm_label}"
                                f"_draw{draw_idx}_{haystack_name}_{depth_tag}.json"
                            )
                            (out_dir / fname).write_text(
                                json.dumps(record, indent=2, ensure_ascii=False)
                            )
                            built += 1
                            char_samples.append(meta["input_char_len"])

    manifest = {
        "condition": "fixed_locations",
        "description": (
            "Grid sweep: evidence placed at {depths} × {haystack sizes}, "
            "buried in one topically-unrelated distractor conversation."
        ),
        "num_scenarios": len(scenarios),
        "variants": variants,
        "depths": list(depths),
        "haystacks": haystacks,
        "num_distractor_draws": num_distractor_draws,
        "num_prompts": built,
        "assignment_seed": ASSIGNMENT_SEED,
        "assignment_summary": assignment_summary(draws),
        "input_chars": {
            "min": min(char_samples) if char_samples else 0,
            "max": max(char_samples) if char_samples else 0,
            "avg": sum(char_samples) / len(char_samples) if char_samples else 0,
        },
        "built_at": datetime.now().isoformat(),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", type=str, default=str(OUT_DIR))
    ap.add_argument(
        "--num-distractor-draws", type=int, default=1,
        help="How many independent distractor assignments to render. "
             "Each draw produces a full grid of prompts. Use >1 to add "
             "variance across distractor choices.",
    )
    ap.add_argument(
        "--depths", type=str, default=",".join(str(d) for d in DEFAULT_DEPTHS),
        help="Comma-separated depth values (fractions in [0, 1]).",
    )
    ap.add_argument(
        "--c-only", action="store_true",
        help="Only render C-present variants (cheaper; tests SR only).",
    )
    args = ap.parse_args()

    depths = tuple(float(x) for x in args.depths.split(","))
    out_dir = Path(args.out_dir)
    manifest = render(
        out_dir=out_dir,
        num_distractor_draws=args.num_distractor_draws,
        depths=depths,
        c_only=args.c_only,
    )
    print(f"Built {manifest['num_prompts']} prompts → {out_dir}")
    print(
        f"  input chars: avg={manifest['input_chars']['avg']:,.0f}  "
        f"min={manifest['input_chars']['min']:,}  "
        f"max={manifest['input_chars']['max']:,}"
    )
    asum = manifest["assignment_summary"]
    print(
        f"  distractor usage across {asum['num_draws']} draw(s): "
        f"{asum['distractors_used']} unique  "
        f"min_uses={asum['min_uses']}  max_uses={asum['max_uses']}"
    )


if __name__ == "__main__":
    main()
