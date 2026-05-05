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

import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Sequence, Tuple

# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────
# ~40-char overhead per turn line: "[YYYY-MM-DD HH:MM:SS] Role: "
TURN_LINE_OVERHEAD = 40

# Default seed-ack text pool. Kept deliberately neutral — very short or
# overly-casual acks ("Got it.", "I see.") were dropped because they can
# read as dismissive of the evidence content.
DEFAULT_ACKS = [
    "Thanks for sharing that.",
    "Understood.",
    "Noted.",
    "Got it.",
    "OK, makes sense.",
    "Mm, alright.",
]

# Picked by sha256-derived row key in evidence_pairs_from_seeds.
EVIDENCE_PREFIXES = [
    "Just wanted to randomly say this, a short acknowledgement would suffice — ",
    "Another random note, just acknowledge and move on — ",
    "Quick aside about me, no need to elaborate — ",
    "Just sharing this for context, brief acknowledgement is fine — ",
]

# Picked by sha256-derived row key in assemble_at_pair_boundary.
RESUMPTION_PREFIXES = [
    "Going back to earlier — ",
    "Resuming the earlier topic — ",
    "Anyway, picking back up — ",
    "Back to what we were discussing — ",
]

# NOTE: We deliberately do NOT emit any chat-boundary marker between
# stitched distractor chunks. The pool entries are themselves
# pre-assembled multi-topic conversations, so inserting a divider only
# at our concat seams while leaving the pool's internal stitch points
# unmarked would assert a topology we don't have. The reader sees the
# merged stream as one ongoing log, which matches what we actually know
# about it.


def _stable_index(parts: Sequence[str], n: int) -> int:
    """Deterministic index in [0, n) from sha256 of joined parts.
    Same key → same index, across processes and machines."""
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()[:16]
    return int(h, 16) % n


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
    """Keep a contiguous prefix of pairs whose total chars meets or just
    exceeds ``char_budget`` (budget = floor, not ceiling). Direction is
    keep-beginning; the pair that crosses the threshold is the last one
    kept. If the whole list fits under the budget, all pairs are kept.
    """
    kept: List[List[Dict]] = []
    running = 0
    for p in pairs:
        kept.append(p)
        running += _pair_char_cost(p)
        if running >= char_budget:
            break
    return kept


# ──────────────────────────────────────────────────────────────────────
# Multi-distractor stitching
# ──────────────────────────────────────────────────────────────────────
def merge_distractor_turn_lists(
    turn_lists: Sequence[Sequence[Dict]],
    gap_days: int = 1,
) -> List[Dict]:
    """Concatenate N distractor turn lists end-to-end, shifting later
    distractors' timestamps so each begins exactly ``gap_days`` after
    the previous ended.

    Args:
        turn_lists: ordered list of turn lists, one per distractor.
            Each turn is ``{"timestamp": "YYYY-MM-DD HH:MM:SS",
            "role": ..., "content": ...}``.
        gap_days: calendar-day gap between consecutive distractors.
            Default 1. First turn of distractor k+1 lands exactly
            ``gap_days`` after the last turn of distractor k.

    Returns:
        A flat turn list with strictly increasing timestamps (given
        each input is monotonic). Turns are deep-copied so callers and
        downstream assemblers can mutate safely.

    Construction notes:
      * Per-distractor intra-turn deltas are preserved — we apply a
        single constant offset per distractor, not per turn.
      * Offset is computed against ``max(ts)`` / ``min(ts)`` rather
        than positional first/last, so the function is robust to the
        rare source row with out-of-order timestamps.
    """
    if not turn_lists:
        return []
    merged: List[Dict] = []
    prev_end: Optional[datetime] = None
    for chat_idx, tl in enumerate(turn_lists):
        copied = [dict(t) for t in tl]
        if not copied:
            continue
        if prev_end is None:
            merged.extend(copied)
            prev_end = max(_parse_ts(t["timestamp"]) for t in copied)
            continue
        first_ts = min(_parse_ts(t["timestamp"]) for t in copied)
        target_start = prev_end + timedelta(days=gap_days)
        offset = target_start - first_ts
        for t in copied:
            shifted = _parse_ts(t["timestamp"]) + offset
            t["timestamp"] = shifted.strftime("%Y-%m-%d %H:%M:%S")
        merged.extend(copied)
        prev_end = max(_parse_ts(t["timestamp"]) for t in copied)
    return merged


# ──────────────────────────────────────────────────────────────────────
# Evidence pair formatting
# ──────────────────────────────────────────────────────────────────────
def evidence_pairs_from_seeds(
    seeds: Sequence[str],
    acks: Sequence[str] = DEFAULT_ACKS,
    row_key: Sequence[str] = (),
) -> List[List[Dict]]:
    """Turn evidence seeds into [user_seed, assistant_ack] pairs.

    Each user turn gets an EVIDENCE_PREFIXES prefix; ack turn gets one
    from `acks`. With `row_key`, both selections are sha256-rotated per
    (row, slot) so different rows surface different combinations, and
    every slot in the row gets a distinct prefix (independent hash with
    collision-bump → full N×(N-1) ordered-pair coverage when len ≥ N).
    Empty `row_key` falls back to position-based cycling.

    Timestamps are filled in by the assembler once placement is known.
    """
    n_pre = len(EVIDENCE_PREFIXES)
    n_ack = len(acks)
    pairs: List[List[Dict]] = []
    used_prefixes: set = set()
    for i, seed in enumerate(seeds):
        if row_key:
            idx = _stable_index(("evidence_prefix",) + tuple(row_key) + (str(i),), n_pre)
            while idx in used_prefixes:
                idx = (idx + 1) % n_pre
            used_prefixes.add(idx)
            ack_idx = _stable_index(("ack",) + tuple(row_key) + (str(i),), n_ack)
        else:
            idx = i % n_pre
            ack_idx = i % n_ack
        user_turn = {
            "timestamp": None,
            "role": "user",
            "content": EVIDENCE_PREFIXES[idx] + seed,
        }
        ack_turn = {
            "timestamp": None,
            "role": "assistant",
            "content": acks[ack_idx],
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
    placement_fracs: Sequence[float],
    row_key: Sequence[str] = (),
) -> Tuple[List[Dict], List[int], int]:
    """Insert each evidence pair at its own pair-boundary position.

    Args:
        distractor_pairs: pair units from the (already truncated) distractor.
        evidence_pairs: pair units holding the evidence seeds + acks.
        placement_fracs: per-pair insertion depths in [0, 1] — same length as
            evidence_pairs. Each frac is clamped and snapped to the nearest
            pair boundary independently. Pairs at the same snapped index
            merge into one block (preserving evidence_pairs order).

    Returns:
        (flat_turn_list, insert_pair_idxs, n_distractor_pairs) where
        insert_pair_idxs lines up with evidence_pairs (one snapped index
        per evidence pair).

    By construction the returned turn list starts on a user turn, ends on
    an assistant turn, and alternates strictly.
    """
    if len(placement_fracs) != len(evidence_pairs):
        raise ValueError(
            f"placement_fracs has {len(placement_fracs)} entries but "
            f"evidence_pairs has {len(evidence_pairs)}"
        )

    n_pairs = len(distractor_pairs)
    insert_pair_idxs = [
        max(0, min(n_pairs, int(round(p * n_pairs)))) for p in placement_fracs
    ]

    # Group evidence pairs by their snapped insert index so colliding pairs
    # (same index) merge into one block. Stable sort preserves the order
    # of evidence_pairs within a tie.
    blocks_at: Dict[int, List[List[Dict]]] = {}
    sorted_ev = sorted(
        range(len(evidence_pairs)),
        key=lambda i: (insert_pair_idxs[i], i),
    )
    for ev_i in sorted_ev:
        blocks_at.setdefault(insert_pair_idxs[ev_i], []).append(evidence_pairs[ev_i])

    n_resume = len(RESUMPTION_PREFIXES)
    flat: List[Dict] = []
    used_resumes: set = set()
    block_seq = 0
    for i in range(n_pairs + 1):
        if i in blocks_at:
            block_pairs = blocks_at[i]
            ts_before = distractor_pairs[i - 1][-1]["timestamp"] if i > 0 else None
            ts_after = distractor_pairs[i][0]["timestamp"] if i < n_pairs else None
            n_block_turns = sum(len(p) for p in block_pairs)
            offset = 0
            for ev_pair in block_pairs:
                for turn in ev_pair:
                    turn["timestamp"] = interpolate_timestamp(
                        ts_before, ts_after, offset, n_block_turns
                    )
                    offset += 1
                flat.extend(ev_pair)
            # Resumption prefix on the first user turn after this block.
            # Skipped at the very end of the haystack (no following turn)
            # AND at the very start (i == 0 means the "following" turn is
            # the conversation's first user message — there's no earlier
            # topic to go back to).
            if 0 < i < n_pairs:
                following = distractor_pairs[i]
                if following and following[0].get("role") == "user":
                    if row_key:
                        idx = _stable_index(
                            ("resumption_prefix",) + tuple(row_key) + (str(block_seq),),
                            n_resume,
                        )
                        while idx in used_resumes:
                            idx = (idx + 1) % n_resume
                        used_resumes.add(idx)
                    else:
                        idx = block_seq % n_resume
                    following[0]["content"] = (
                        RESUMPTION_PREFIXES[idx]
                        + str(following[0].get("content", ""))
                    )
            block_seq += 1
        if i < n_pairs:
            flat.extend(distractor_pairs[i])

    return flat, insert_pair_idxs, n_pairs


# ──────────────────────────────────────────────────────────────────────
# Rendering to text
# ──────────────────────────────────────────────────────────────────────
def turns_to_text(turns: Sequence[Dict]) -> str:
    """Format a flat turn list as `[timestamp] Role: content` lines.
    No chat-boundary divider is emitted between stitched chunks — see
    the module-level comment near the prefix tables."""
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
