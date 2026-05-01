#!/usr/bin/env python3
"""
render_with_constraint.py — the "constraint stated explicitly" condition.
==========================================================================

Control condition. For C-bearing variants the evidence seeds AND the
constraint description are placed directly in the user message; no
prior conversation history. Ceiling test for history integration —
failure here is a domain-knowledge failure, not a long-context or
attention failure.

Since 2026-05-01 this preset enumerates all 5 variants (C / A+C / B+C
/ A / B). For A and B no-C variants the constraint description is
still rendered in the user message but the constraint-grounding seed
is absent — measures spurious-flag behaviour when the user fact
doesn't actually trigger the constraint.

Thin wrapper over ``mixer.mix()`` with:

    n_distractor_draws = 0
    n_placements       = 0
    n_lengths          = 0
    include_constraint_inline = True
    c_only             = False  (full 5-variant by default)
    condition_label    = "canon_direct"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from renderers.mixer import mix  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent.parent.parent
OUT_DIR = BASE_DIR / "generated" / "canon_direct"


def render(
    out_dir: Path,
    c_only: bool = False,
) -> dict:
    return mix(
        out_dir=out_dir,
        n_distractor_draws=0,
        n_placements=0,
        n_lengths=0,
        include_constraint_inline=True,
        c_only=c_only,
        condition_label="canon_direct",
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--out-dir", type=str, default=str(OUT_DIR))
    ap.add_argument(
        "--c-only", action="store_true",
        help="Restrict to C-bearing variants (skip A and B no-C). Off by "
             "default — canon_direct includes all 5 variants.",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    manifest = render(out_dir=out_dir, c_only=args.c_only)
    print(f"Built {manifest['num_prompts']} prompts → {out_dir}")
    ic = manifest["input_chars"]
    print(f"  input chars: avg={ic['avg']:,.0f}  min={ic['min']:,}  max={ic['max']:,}")


if __name__ == "__main__":
    main()
