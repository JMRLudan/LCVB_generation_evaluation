#!/usr/bin/env python3
"""
render_with_constraint.py — the "constraint stated explicitly" condition.
==========================================================================

Control condition. Evidence seeds AND the constraint description are
placed directly in the user message; no prior conversation history.
Ceiling test for history integration — failure here is a domain-
knowledge failure, not a long-context or attention failure.

Thin wrapper over ``mixer.mix()`` with:

    n_distractor_draws = 0
    n_placements       = 0
    n_lengths          = 0
    include_constraint_inline = True
    condition_label    = "with_constraint"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from renderers.mixer import mix  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent.parent.parent
OUT_DIR = BASE_DIR / "generated" / "with_constraint"


def render(
    out_dir: Path,
    c_only: bool = True,
) -> dict:
    return mix(
        out_dir=out_dir,
        n_distractor_draws=0,
        n_placements=0,
        n_lengths=0,
        include_constraint_inline=True,
        c_only=c_only,
        condition_label="with_constraint",
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--out-dir", type=str, default=str(OUT_DIR))
    ap.add_argument(
        "--include-no-c-variants", action="store_true",
        help="Also include A and B (no-constraint) variants. Default: only C-present.",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    manifest = render(
        out_dir=out_dir,
        c_only=not args.include_no_c_variants,
    )
    print(f"Built {manifest['num_prompts']} prompts → {out_dir}")
    ic = manifest["input_chars"]
    print(f"  input chars: avg={ic['avg']:,.0f}  min={ic['min']:,}  max={ic['max']:,}")


if __name__ == "__main__":
    main()
