#!/usr/bin/env python3
"""
render_stitched_locations.py — multi-distractor stitched variant.
==================================================================

Now a thin wrapper over ``mixer.mix()`` at ``n_distractors_per_prompt >= 2``.
This file used to be a stub (NotImplementedError) reserving this slot
for the harder multi-source condition; that work is finished — the
general mixer handles it via the ``n_distractors_per_prompt`` axis.

Each prompt here is built by:

  1. Picking N distinct distractor chats (via the mixer's per-slot
     pool shuffle — balanced usage across the full pool, no same-item
     collisions).
  2. Merging them end-to-end with a ``merge_gap_days``-day calendar
     gap between the last turn of one chat and the first of the next.
     Intra-chat timestamp deltas are preserved.
  3. Running the usual keep-beginning truncation and evidence
     insertion against the merged pair sequence.

Construction invariants (unchanged, now enforced by ``mix()``):

  * **Keep-beginning truncation.** Pair-by-pair from the start of the
    merged sequence.
  * **Strict alternation + assistant-final.** Preserved by pair-unit
    assembly.
  * **Deterministic source order.** Slot order (0…N-1) is the merge
    order; seed + draw_idx + slot determines which hash lands in
    which slot.
  * **No stitching in the default-canon renderers.** Single-distractor
    conditions (with_constraint, no_distractor, canon_*) never call
    this path — they use ``n_distractors_per_prompt=1``.

Defaults: N=2, 1-day gap, 5 fixed depths × single 500K-char budget.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from renderers.mixer import mix  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent.parent.parent
OUT_DIR = BASE_DIR / "generated" / "stitched_locations"

DEFAULT_DEPTHS = (0.0, 0.25, 0.5, 0.75, 1.0)
DEFAULT_CHAR_BUDGET = 500_000


def render(
    out_dir: Path,
    n_distractors_per_prompt: int = 2,
    num_distractor_draws: int = 1,
    depths=DEFAULT_DEPTHS,
    char_budget: int = DEFAULT_CHAR_BUDGET,
    merge_gap_days: int = 1,
    c_only: bool = False,
) -> dict:
    if n_distractors_per_prompt < 2:
        raise ValueError(
            f"render_stitched_locations requires n_distractors_per_prompt >=2, "
            f"got {n_distractors_per_prompt}. For n=1, use render_fixed_locations."
        )
    depths = tuple(depths)
    return mix(
        out_dir=out_dir,
        n_distractor_draws=num_distractor_draws,
        n_distractors_per_prompt=n_distractors_per_prompt,
        n_placements=len(depths),
        n_lengths=1,
        placement_mode="fixed",
        placements_list=depths,
        lengths_named={"stitched": char_budget},
        include_constraint_inline=False,
        c_only=c_only,
        condition_label="stitched_locations",
        merge_gap_days=merge_gap_days,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--out-dir", type=str, default=str(OUT_DIR))
    ap.add_argument(
        "--n-distractors-per-prompt", type=int, default=2,
        help="How many distractor conversations to stitch (>=2).",
    )
    ap.add_argument(
        "--num-distractor-draws", type=int, default=1,
        help="How many full re-renders, each with fresh distractor "
             "picks for every item.",
    )
    ap.add_argument(
        "--depths", type=str, default=",".join(str(d) for d in DEFAULT_DEPTHS),
        help="Comma-separated depth values (fractions in [0, 1]).",
    )
    ap.add_argument(
        "--char-budget", type=int, default=DEFAULT_CHAR_BUDGET,
        help="Total char budget for the merged conversation history.",
    )
    ap.add_argument(
        "--merge-gap-days", type=int, default=1,
        help="Calendar-day gap between consecutive stitched distractors.",
    )
    ap.add_argument(
        "--c-only", action="store_true",
        help="Only render C-present variants.",
    )
    args = ap.parse_args()

    depths = tuple(float(x) for x in args.depths.split(","))
    out_dir = Path(args.out_dir)
    manifest = render(
        out_dir=out_dir,
        n_distractors_per_prompt=args.n_distractors_per_prompt,
        num_distractor_draws=args.num_distractor_draws,
        depths=depths,
        char_budget=args.char_budget,
        merge_gap_days=args.merge_gap_days,
        c_only=args.c_only,
    )
    print(f"Built {manifest['num_prompts']} prompts → {out_dir}")
    ic = manifest["input_chars"]
    print(f"  input chars: avg={ic['avg']:,.0f}  min={ic['min']:,}  max={ic['max']:,}")
    if "distractor_usage" in manifest:
        du = manifest["distractor_usage"]
        print(
            f"  distractor usage: {du['distractors_used']} unique  "
            f"min={du['min_uses']}  max={du['max_uses']}  mean={du['mean_uses']:.1f}"
        )


if __name__ == "__main__":
    main()
