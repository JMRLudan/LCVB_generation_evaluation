#!/usr/bin/env python3
"""
render_no_distractor.py — short-history condition, no distractor turns.
======================================================================

The primary experimental condition. Evidence is placed in the system prompt
as a short, timestamped conversation history between the user and the
assistant; no distractor material is interleaved. The triggering query goes
in a fresh user message.

This differs from the other renderers in this repo:

  * `render_with_constraint.py` puts the constraint text inline in the user
    message — ceiling test, no history integration required.
  * `render_fixed_locations.py` / `render_continuous_random.py` embed the
    evidence inside a long distractor conversation — stress test at large
    char budgets.

This renderer is the *clean personalization* condition: the model is asked
to read short conversation history, carry the personal fact forward, and
act on it when choosing between the two options. A failure here can't be
blamed on long-context attention; it's pure history integration.

Output format (same schema as every other renderer):

    { "system_prompt":
        "You are a helpful assistant.\\n\\n
         Below is the conversation history with this user over the past
         several days:\\n\\n
         [2026-02-20 10:00:00] User: <seed_1>\\n
         [2026-02-20 10:30:00] Assistant: <ack_1>\\n
         ...",
      "user_message": "<query>\\n\\nOption A: ...\\nOption B: ...",
      "metadata": { ..sid, variant, permutation, expected_answer, ... } }

Per project convention this renderer does NOT call any API — it only
produces prompt files. Run `pipeline/run.py` to send them to a model.

`num_distractor_draws` is accepted for interface uniformity across all
renderers but has no effect here because this condition never uses
distractors. Each `--num-distractor-draws N` value >1 just replicates
the same prompts under draw_0, draw_1, …, draw_{N-1}.
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
from renderers.assembly import (  # noqa: E402
    DEFAULT_ACKS, build_system_prompt,
)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
OUT_DIR = BASE_DIR / "generated" / "no_distractor"


def _format_conversation(seeds) -> str:
    """Evidence seeds → `[ts] User: seed` / `[ts] Assistant: ack` lines.

    Timestamps use the same convention as the main-repo pipeline: a fixed
    base date (2026-02-20) with one hour between each pair, user turn at
    :00, assistant turn at :30. Keeping this stable means prompt bytes are
    reproducible across regenerations.
    """
    lines = []
    for i, seed in enumerate(seeds):
        hour = 10 + i
        ack = DEFAULT_ACKS[i % len(DEFAULT_ACKS)]
        lines.append(f"[2026-02-20 {hour:02d}:00:00] User: {seed}")
        lines.append(f"[2026-02-20 {hour:02d}:30:00] Assistant: {ack}")
    return "\n".join(lines)


def _build_user_message(query, choice_a, choice_b) -> str:
    return (
        f"{query}\n\n"
        f"Option A: {choice_a}\n"
        f"Option B: {choice_b}"
    )


def render(
    out_dir: Path,
    num_distractor_draws: int = 1,
    c_only: bool = False,
) -> dict:
    scenarios = load_scenarios(SCENARIOS_TSV, validated_only=True)
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
    for sid in sorted(scenarios.keys()):
        scenario = scenarios[sid]
        query = scenario["benign_triggering_query"]
        choice_a = scenario["choice_a_label"]
        choice_b = scenario["choice_b_label"]
        constraint = scenario.get("constraint_description", "")

        for variant in variants:
            expected = "C" if "C" in variant else variant
            for perm_label, seed_indices in enumerate_permutations(scenario, variant):
                seeds = get_seeds_by_indices(scenario, variant, seed_indices)
                conv_text = _format_conversation(seeds)
                system_prompt = build_system_prompt(conv_text)
                user_msg = _build_user_message(query, choice_a, choice_b)
                q_with_opts = f"{query}\n\nOption A: {choice_a}\nOption B: {choice_b}"

                for draw_idx in range(num_distractor_draws):
                    meta = {
                        "scenario_id": sid,
                        "evidence_variant": variant,
                        "permutation": perm_label,
                        "draw_idx": draw_idx,
                        "expected_answer": expected,
                        "query_with_options": q_with_opts,
                        "constraint_description": constraint,
                        "input_char_len": len(system_prompt) + len(user_msg),
                        "condition": "no_distractor",
                        # No distractor is used in this condition; field kept
                        # for schema uniformity with other renderers.
                        "distractor_hash": None,
                        "n_evidence_pairs": len(seeds),
                    }
                    record = {
                        "system_prompt": system_prompt,
                        "user_message": user_msg,
                        "metadata": meta,
                    }
                    fname = f"{sid}_{variant}_{perm_label}_d{draw_idx}.json"
                    (out_dir / fname).write_text(
                        json.dumps(record, indent=2, ensure_ascii=False)
                    )
                    built += 1
                    char_samples.append(meta["input_char_len"])

    manifest = {
        "condition": "no_distractor",
        "description": (
            "Evidence seeds placed in a short timestamped conversation "
            "history inside the system prompt; triggering query asked as a "
            "fresh user message. No distractor material interleaved. Tests "
            "pure conversation-history integration (not long-context)."
        ),
        "num_scenarios": len(scenarios),
        "variants": variants,
        "num_distractor_draws": num_distractor_draws,
        "num_prompts": built,
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
        help="Number of prompt replicates per (scenario, variant, perm). "
             "Has no effect on prompt CONTENT for this condition (no distractors "
             "are used) — it just replicates each item N times for interface "
             "uniformity with other renderers.",
    )
    ap.add_argument(
        "--c-only", action="store_true",
        help="Only render C-present variants (skip A and B baselines).",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    manifest = render(
        out_dir=out_dir,
        num_distractor_draws=args.num_distractor_draws,
        c_only=args.c_only,
    )
    print(f"Built {manifest['num_prompts']} prompts → {out_dir}")
    print(
        f"  input chars: avg={manifest['input_chars']['avg']:,.0f}  "
        f"min={manifest['input_chars']['min']:,}  "
        f"max={manifest['input_chars']['max']:,}"
    )


if __name__ == "__main__":
    main()
