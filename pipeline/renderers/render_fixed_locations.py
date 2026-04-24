#!/usr/bin/env python3
"""
render_fixed_locations.py — grid sweep at fixed depths × haystacks.
====================================================================

Evidence is placed at one of N fixed depths within a distractor
conversation, under one or more named char budgets ("short" ≈ 24 K
chars ≈ 3–4 K tokens, "long" ≈ 224 K chars ≈ 32 K tokens by default).

Thin wrapper over ``mixer.mix()`` with:

    n_distractor_draws = num_distractor_draws
    n_placements       = len(depths)
    n_lengths          = len(haystacks)
    placement_mode     = "fixed"
    placements_list    = depths
    lengths_named      = haystacks      # ordered dict
    condition_label    = "fixed_locations"

With ``num_distractor_draws > 1`` the full (depth × haystack) grid is
re-rendered once per draw, each draw using a different distractor
for each item — per the "re-render" semantic of the axis.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from renderers.mixer import mix  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent.parent.parent
OUT_DIR = BASE_DIR / "generated" / "fixed_locations"

DEFAULT_DEPTHS = (0.0, 0.25, 0.5, 0.75, 1.0)
DEFAULT_HAYSTACKS = {
    "short": 24_000,    # ~3-4 K tokens
    "long":  224_000,   # ~32 K tokens
}


def render(
    out_dir: Path,
    num_distractor_draws: int = 1,
    n_distractors_per_prompt: int = 1,
    depths=DEFAULT_DEPTHS,
    haystacks=None,
    c_only: bool = False,
) -> dict:
    if haystacks is None:
        haystacks = DEFAULT_HAYSTACKS
    depths = tuple(depths)
    return mix(
        out_dir=out_dir,
        n_distractor_draws=num_distractor_draws,
        n_distractors_per_prompt=n_distractors_per_prompt,
        n_placements=len(depths),
        n_lengths=len(haystacks),
        placement_mode="fixed",
        placements_list=depths,
        lengths_named=dict(haystacks),
        include_constraint_inline=False,
        c_only=c_only,
        condition_label="fixed_locations",
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--out-dir", type=str, default=str(OUT_DIR))
    ap.add_argument(
        "--num-distractor-draws", type=int, default=1,
        help="How many full (depth × haystack) grids to render, each "
             "with a different distractor per item.",
    )
    ap.add_argument(
        "--n-distractors-per-prompt", type=int, default=1,
        help="How many distractor conversations to merge per prompt "
             "(default 1 — classic single-distractor). >1 stitches "
             "multiple distractors end-to-end with a 1-day gap.",
    )
    ap.add_argument(
        "--depths", type=str, default=",".join(str(d) for d in DEFAULT_DEPTHS),
        help="Comma-separated depth values (fractions in [0, 1]).",
    )
    ap.add_argument(
        "--c-only", action="store_true",
        help="Only render C-present variants (cheaper).",
    )
    args = ap.parse_args()

    depths = tuple(float(x) for x in args.depths.split(","))
    out_dir = Path(args.out_dir)
    manifest = render(
        out_dir=out_dir,
        num_distractor_draws=args.num_distractor_draws,
        n_distractors_per_prompt=args.n_distractors_per_prompt,
        depths=depths,
        c_only=args.c_only,
    )
    print(f"Built {manifest['num_prompts']} prompts → {out_dir}")
    ic = manifest["input_chars"]
    print(f"  input chars: avg={ic['avg']:,.0f}  min={ic['min']:,}  max={ic['max']:,}")
    if "distractor_usage" in manifest:
        du = manifest["distractor_usage"]
        print(
            f"  distractor usage across {manifest['n_distractor_draws']} draw(s): "
            f"{du['distractors_used']} unique  "
            f"min={du['min_uses']}  max={du['max_uses']}"
        )


if __name__ == "__main__":
    main()
