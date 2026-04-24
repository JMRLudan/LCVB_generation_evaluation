#!/usr/bin/env python3
"""
render_no_distractor.py — short-history condition, no distractor turns.
========================================================================

Primary personalization condition. Evidence seeds are rendered as a
short timestamped conversation history in the system prompt; the
triggering query goes in a fresh user message. No distractor
material interleaved — tests pure history integration, not long-
context attention.

Thin wrapper over ``mixer.mix()`` with:

    n_distractor_draws = 0
    n_placements       = 0
    n_lengths          = 0
    include_constraint_inline = False
    condition_label    = "no_distractor"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from renderers.mixer import mix  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent.parent.parent
OUT_DIR = BASE_DIR / "generated" / "no_distractor"


def render(
    out_dir: Path,
    c_only: bool = False,
) -> dict:
    return mix(
        out_dir=out_dir,
        n_distractor_draws=0,
        n_placements=0,
        n_lengths=0,
        include_constraint_inline=False,
        c_only=c_only,
        condition_label="no_distractor",
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--out-dir", type=str, default=str(OUT_DIR))
    ap.add_argument(
        "--c-only", action="store_true",
        help="Only render C-present variants (skip A and B baselines).",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    manifest = render(out_dir=out_dir, c_only=args.c_only)
    print(f"Built {manifest['num_prompts']} prompts → {out_dir}")
    ic = manifest["input_chars"]
    print(f"  input chars: avg={ic['avg']:,.0f}  min={ic['min']:,}  max={ic['max']:,}")


if __name__ == "__main__":
    main()
