#!/usr/bin/env python3
"""
render_xl.py — extended-context experiment for canon-style prompts.
====================================================================

Two new presets sized to probe long-context vigilance:

    canon_xl_200k  → fixed ~700K-char haystack (~200K input tokens at 3.5 chars/tok)
    canon_xl_500k  → fixed ~1.75M-char haystack (~500K input tokens)

Each preset emits 85 scenarios × 3 C-bearing variants × 1 rep = 255
prompts. Same (scenario, variant, depth, evidence-seed,
distractor-seed) tuples are used at both bands so paired-difference
statistics are well-defined: only the char budget changes between the
two preset directories.

Mechanically a thin wrapper over ``mixer.mix()`` with:

    n_distractor_draws        = 1
    n_distractors_per_prompt  = 3      (same as canon_unified)
    n_placements              = 1
    n_lengths                 = 1
    placement_mode            = "uniform_stratified"
    length_mode               = "fixed"
    lengths_named             = {label: target_chars}
    c_only                    = True   (skip A and B no-C variants)
    condition_label           = preset name

Usage:

    python3 -m pipeline.renderers.render_xl --band 200k
    python3 -m pipeline.renderers.render_xl --band 500k

The seed is the project-wide ASSIGNMENT_SEED (4232026), so prompts
are byte-identical across re-runs given the same scenarios + pool.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from renderers.mixer import mix  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Token-to-char conversion. Empirical from gemini-3-flash canon_unified
# data: median ~3.5 chars/token for our prompt mix. Slight overshoot is
# safe since assembly truncates to fit the budget.
CHARS_PER_TOKEN = 3.5

BAND_CONFIGS = {
    "200k": {
        "target_tokens": 200_000,
        "target_chars": int(200_000 * CHARS_PER_TOKEN),  # 700,000
        "out_dir": BASE_DIR / "generated" / "canon_xl_200k",
        "condition_label": "canon_xl_200k",
        # 700K chars / ~250K avg distractor → 3 fits comfortably
        "n_distractors_per_prompt": 3,
    },
    "500k": {
        "target_tokens": 500_000,
        "target_chars": int(500_000 * CHARS_PER_TOKEN),  # 1,750,000
        "out_dir": BASE_DIR / "generated" / "canon_xl_500k",
        "condition_label": "canon_xl_500k",
        # 1.75M chars / ~250K avg distractor → need 7-8; pad to 10 for headroom
        "n_distractors_per_prompt": 10,
    },
}


def render(band: str) -> dict:
    cfg = BAND_CONFIGS[band]
    return mix(
        out_dir=cfg["out_dir"],
        n_distractor_draws=1,
        n_distractors_per_prompt=cfg["n_distractors_per_prompt"],
        n_placements=1,
        n_lengths=1,
        placement_mode="uniform_stratified",
        length_mode="fixed",
        lengths_named={cfg["condition_label"]: cfg["target_chars"]},
        c_only=True,
        single_perm_per_variant=True,
        condition_label=cfg["condition_label"],
        merge_gap_days=1,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument(
        "--band", choices=list(BAND_CONFIGS), required=True,
        help="Which length band to render."
    )
    args = ap.parse_args()

    cfg = BAND_CONFIGS[args.band]
    print(f"Rendering {cfg['condition_label']} → {cfg['out_dir']}")
    print(f"  target: ~{cfg['target_tokens']:,} tokens (~{cfg['target_chars']:,} chars)")

    manifest = render(args.band)

    print(f"  built {manifest['num_prompts']} prompts")
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
