"""
Shared prompt-assembly primitives used by every renderer in this repo.

The rules encoded here are project invariants (also written out in the
top-level README under "Generation and stitching rules"):

  * Work in **pair units** (user+assistant tuples). Never emit a single
    user turn without its paired assistant reply. Keeping pairs intact
    guarantees strict alternation and an assistant-final prompt by
    construction, with no post-hoc "pop trailing user" fix-up.
  * **Keep-beginning truncation.** When a distractor conversation is too
    long for the char budget, walk pair-by-pair from the start and stop
    as soon as the next pair would exceed the budget. Drop everything
    after. Never drop from the middle. Never flip direction.
  * **Evidence is inserted at a pair boundary.** The seed+ack evidence
    pairs go between `[…(user,asst)]` and `[(user,asst)…]`. This is the
    only way to preserve alternation without touching turn roles.
  * **Prompt endpoints.** The assembled turn list starts on a user turn
    and ends on an assistant turn. This is always true when you build
    with pair units and insert at a pair boundary.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Sequence, Tuple

# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────
# ~40-char overhead per turn line: "[YYYY-MM-DD HH:MM:SS] Role: "
TURN_LINE_OVERHEAD = 40

# Default seed-ack text pool. Kept in the main eval_pipeline module as
# `ACKS`, re-exported here for renderers that want a self-contained import.
DEFAULT_ACKS = ["Got it.", "I see.", "Thanks for sharing that.", "Understood.", "Noted."]


# ──────────────────────────────────────────────────────────────────────
# Pair units
# ──────────────────────────────────────────────────────────────────────
def pair_units_from_turns(turns: Sequence[Dict]) -> List[List[Dict]]:
    """Walk a flat turn list and emit user→assistant pair units.

    Orphan turns (a trailing lone user, or rare same-role-twice glitches
    in source data) are dropped so the output is always a clean list
    of [user_turn, assistant_turn] pairs.
    """
    pairs: List[List[Dict]] = []
    i = 0
    n = len(turns)
    while i < n - 1:
        t0, t1 = turns[i], turns[i + 1]
        if t0.get("role") == "user" and t1.get("role") == "assistant":
            pairs.append([t0, t1])
            i += 2
        else:
            i += 1  # skip malformed / orphan turn
    return pairs


def _pair_char_cost(pair: Sequence[Dict]) -> int:
    return (
        len(pair[0]["content"])
        + len(pair[1]["content"])
        + 2 * TURN_LINE_OVERHEAD
    )


# ──────────────────────────────────────────────────────────────────────
# Keep-beginning truncation
# ──────────────────────────────────────────────────────────────────────
def truncate_pairs_to_budget(
    pairs: Sequence[List[Dict]], char_budget: int
) -> List[List[Dict]]:
    """Keep a contiguous prefix of pairs that fits within `char_budget`.

    Iterates from index 0 forward. Stops adding pairs as soon as the
    next pair would push the running char count over the budget.
    The tail is dropped.

    This is the ONLY truncation direction allowed in this repo — see
    the project memory entry "Distractor conversation construction rules"
    and the README "Generation and stitching rules" section.
    """
    kept: List[List[Dict]] = []
    running = 0
    for p in pairs:
        cost = _pair_char_cost(p)
        if running + cost > char_budget:
            break
        kept.append(p)
        running += cost
    return kept


# ──────────────────────────────────────────────────────────────────────
# Evidence pair formatting
# ──────────────────────────────────────────────────────────────────────
def evidence_pairs_from_seeds(
    seeds: Sequence[str], acks: Sequence[str] = DEFAULT_ACKS
) -> List[List[Dict]]:
    """Turn evidence seed strings into [user_seed, assistant_ack] pairs.

    Timestamps are left None and filled in later by the assembler
    once the insertion point is known.
    """
    pairs: List[List[Dict]] = []
    for i, seed in enumerate(seeds):
        user_turn = {"timestamp": None, "role": "user", "content": seed}
        ack_turn = {
            "timestamp": None,
            "role": "assistant",
            "content": acks[i % len(acks)],
        }
        pairs.append([user_turn, ack_turn])
    return pairs


# ──────────────────────────────────────────────────────────────────────
# Timestamp interpolation
# ──────────────────────────────────────────────────────────────────────
def _parse_ts(ts: str) -> datetime:
    return datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")


def interpolate_timestamp(
    ts_before: Optional[str],
    ts_after: Optional[str],
    offset: int,
    total: int,
) -> str:
    """Linearly interpolate a timestamp between two neighbors.

    Falls back to sensible defaults if one or both anchors are missing.
    """
    if ts_before and ts_after:
        t0 = _parse_ts(ts_before)
        t1 = _parse_ts(ts_after)
        gap = (t1 - t0) / (total + 1)
        result = t0 + gap * (offset + 1)
    elif ts_before:
        result = _parse_ts(ts_before) + timedelta(minutes=5 * (offset + 1))
    elif ts_after:
        result = _parse_ts(ts_after) - timedelta(minutes=5 * (total - offset))
    else:
        result = datetime(2026, 2, 24, 12, 0, 0) + timedelta(minutes=15 * offset)
    return result.strftime("%Y-%m-%d %H:%M:%S")


# ──────────────────────────────────────────────────────────────────────
# Assembly
# ──────────────────────────────────────────────────────────────────────
def assemble_at_pair_boundary(
    distractor_pairs: Sequence[List[Dict]],
    evidence_pairs: Sequence[List[Dict]],
    placement_frac: float,
) -> Tuple[List[Dict], int, int]:
    """Insert evidence pairs at a pair-boundary chosen by placement_frac.

    Args:
        distractor_pairs: pair units from the (already truncated)
            distractor conversation.
        evidence_pairs: pair units that hold the evidence seeds + acks.
        placement_frac: where in the distractor pair sequence to insert,
            normalized to [0, 1]. `0.0` → evidence at the very top;
            `1.0` → evidence at the very end, immediately before the new
            user query. The value is clamped to the valid range and
            snapped to the nearest pair index.

    Returns:
        (combined_flat_turn_list, insert_pair_idx, n_distractor_pairs)

    By construction the returned turn list starts on a user turn,
    ends on an assistant turn, and alternates strictly.
    """
    n_pairs = len(distractor_pairs)
    insert_pair_idx = int(round(placement_frac * n_pairs))
    insert_pair_idx = max(0, min(n_pairs, insert_pair_idx))

    # Timestamp anchors: last turn of the previous pair; first turn of the
    # following pair.
    ts_before = None
    ts_after = None
    if insert_pair_idx > 0:
        ts_before = distractor_pairs[insert_pair_idx - 1][-1]["timestamp"]
    if insert_pair_idx < n_pairs:
        ts_after = distractor_pairs[insert_pair_idx][0]["timestamp"]

    n_ev_turns = sum(len(p) for p in evidence_pairs)
    offset = 0
    for pair in evidence_pairs:
        for turn in pair:
            turn["timestamp"] = interpolate_timestamp(
                ts_before, ts_after, offset, n_ev_turns
            )
            offset += 1

    flat: List[Dict] = []
    for p in distractor_pairs[:insert_pair_idx]:
        flat.extend(p)
    for p in evidence_pairs:
        flat.extend(p)
    for p in distractor_pairs[insert_pair_idx:]:
        flat.extend(p)
    return flat, insert_pair_idx, n_pairs


# ──────────────────────────────────────────────────────────────────────
# Rendering to text
# ──────────────────────────────────────────────────────────────────────
def turns_to_text(turns: Sequence[Dict]) -> str:
    """Format a flat turn list as `[timestamp] Role: content` lines."""
    lines = []
    for t in turns:
        role_label = "User" if t["role"] == "user" else "Assistant"
        lines.append(f"[{t['timestamp']}] {role_label}: {t['content']}")
    return "\n".join(lines)


def build_system_prompt(conv_text: str) -> str:
    """Wrap a conversation-history block in the standard system preamble."""
    return (
        "You are a helpful assistant.\n\n"
        "Below is the conversation history with this user over the past several days:\n\n"
        f"{conv_text}"
    )


def assert_alternation(turns: Sequence[Dict]) -> None:
    """Runtime check: strict user→assistant→user→assistant, starts user,
    ends assistant. Raises AssertionError on violation. Renderers should
    call this on the final turn list before writing the prompt."""
    if not turns:
        raise AssertionError("Empty turn list")
    if turns[0]["role"] != "user":
        raise AssertionError(f"First turn must be user, got {turns[0]['role']!r}")
    if turns[-1]["role"] != "assistant":
        raise AssertionError(
            f"Last turn must be assistant, got {turns[-1]['role']!r}"
        )
    expected = "user"
    for i, t in enumerate(turns):
        if t["role"] != expected:
            raise AssertionError(
                f"Turn {i} has role {t['role']!r}, expected {expected!r} "
                f"(strict alternation)"
            )
        expected = "assistant" if expected == "user" else "user"
