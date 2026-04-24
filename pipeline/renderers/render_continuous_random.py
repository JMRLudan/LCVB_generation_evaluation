#!/usr/bin/env python3
"""
render_continuous_random.py — uniform random placement, single length.
=======================================================================

Evidence is placed at a single stratified-random depth inside a
long (default 224 K chars) distractor conversation. Each item is
placed deterministically (hash jitter → uniform in [0, 1]), so the
set of placements covers the full depth range without the grid
artifacts of fixed_locations.

Thin wrapper over ``mixer.mix()`` with:

    n_distractor_draws = num_distractor_draws
    n_placements       = 1
    n_lengths          = 1
    placement_mode     = "uniform"
    lengths_named      = {"long": char_budget}
    condition_label    = "continuous_random"

With ``num_distractor_draws > 1`` each item is re-rendered once per
draw, each draw using a different distractor and a different
stratified-random depth for that item.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from renderers.mixer import mix  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent.parent.parent
OUT_DIR = BASE_DIR / "generated" / "continuous_random"
DEFAULT_CHAR_BUDGET = 224_000


def render(
    out_dir: Path,
    char_budget: int = DEFAULT_CHAR_BUDGET,
    num_distractor_draws: int = 1,
    n_distractors_per_prompt: int = 1,
    c_only: bool = False,
) -> dict:
    return mix(
        out_dir=out_dir,
        n_distractor_draws=num_distractor_draws,
        n_distractors_per_prompt=n_distractors_per_prompt,
        n_placements=1,
        n_lengths=1,
        placement_mode="uniform",
        lengths_named={"long": char_budget},
        include_constraint_inline=False,
        c_only=c_only,
        condition_label="continuous_random",
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--out-dir", type=str, default=str(OUT_DIR))
    ap.add_argument("--char-budget", type=int, default=DEFAULT_CHAR_BUDGET)
    ap.add_argument(
        "--num-distractor-draws", type=int, default=1,
        help="How many full re-renders of the scenario set, each with "
             "a different distractor and fresh stratified placement "
             "per item.",
    )
    ap.add_argument(
        "--n-distractors-per-prompt", type=int, default=1,
        help="How many distractor conversations to merge per prompt "
             "(default 1). >1 stitches multiple distractors end-to-end "
             "with a 1-day gap before placement.",
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
        n_distractors_per_prompt=args.n_distractors_per_prompt,
        c_only=args.c_only,
    )
    print(f"Built {manifest['num_prompts']} prompts → {out_dir}")
    ic = manifest["input_chars"]
    print(f"  input chars: avg={ic['avg']:,.0f}  min={ic['min']:,}  max={ic['max']:,}")
    if "placement_frac" in manifest:
        pf = manifest["placement_frac"]
        print(f"  placement_frac: avg={pf['avg']:.3f}  min={pf['min']:.3f}  max={pf['max']:.3f}")


if __name__ == "__main__":
    main()
