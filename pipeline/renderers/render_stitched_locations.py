#!/usr/bin/env python3
"""
render_stitched_locations.py — harder variant (NOT YET IMPLEMENTED).
=====================================================================

Reserved for the harder multi-source variant where each prompt stitches
MORE THAN ONE distractor conversation together, end-to-end, with the
evidence inserted at a pair boundary of the full assembled sequence.

Construction invariants for when this is implemented (per project memory
"Distractor conversation construction rules"):

  * **Keep-beginning truncation per source.** Each source is independently
    truncated using the `keep-beginning` rule — same as every other
    renderer in this repo.
  * **Assembly preserves strict alternation and endpoints.** The stitched
    sequence must still satisfy the user→assistant alternation invariant
    end-to-end, start on a user turn, and end on an assistant turn.
  * **Deterministic source order.** Concatenate sources in a deterministic
    order; insert evidence at a pair boundary of the full sequence.

Default renderers (`render_with_constraint`, `render_fixed_locations`,
`render_continuous_random`) are always SINGLE-distractor. Multi-source
stitching lives here, and only here, so the baseline condition is never
silently contaminated with the harder variant.
"""

from __future__ import annotations

import argparse
import sys


def render(*args, **kwargs):
    raise NotImplementedError(
        "render_stitched_locations is the harder multi-source variant and "
        "has not been implemented yet. See the docstring for the "
        "construction invariants that any implementation must satisfy."
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.parse_args()
    print(
        "render_stitched_locations: NOT YET IMPLEMENTED.\n"
        "This renderer is reserved for the harder multi-distractor "
        "stitched variant. Default renderers in this repo are single-"
        "distractor by design — see the README for construction rules.",
        file=sys.stderr,
    )
    sys.exit(2)


if __name__ == "__main__":
    main()
