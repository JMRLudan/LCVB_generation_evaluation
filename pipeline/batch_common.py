"""
batch_common.py — shared types + helpers for provider batch adapters.
=====================================================================

Common request/response shapes used by the per-provider batch
adapters (`batch_anthropic.py`, `batch_openai.py`, `batch_gemini.py`).
The adapters consume `BatchRequest` lists, translate them to each
provider's batch-job format, submit, poll, fetch, and translate
responses back to a uniform `BatchResult` with the same dict shape
that `openrouter_client._do_*_sync` already returns. This means
existing scoring / parsing in `multi_model_runner.py` keeps working
unchanged on batch results.

Why batch instead of real-time:
  * Frontier providers offer ~50% discount on 24h-window batches.
    For the canon sweep (~10 610 prompts × N models) this is the
    difference between $700 and $1400.
  * Easier to launch and walk away — no async retry plumbing during
    the run, no rate-limit-management code paths.

Conventions:
  * `custom_id` is `f"{run_id}::{scenario_id}::{evidence_variant}::{permutation}"`
    so we can correlate results back to scenarios deterministically.
    Pipe-delimited everywhere else in the repo, but batch-API
    custom_ids forbid `|` on some providers (OpenAI does, Anthropic
    constrains charset more strictly), so we use `::`.
  * All batch jobs run with `temperature=1.0` and `max_tokens=10000`
    by default — same defaults as the real-time path.
  * Cost accounting still goes through `OpenRouterClient`'s cost log;
    adapters call `client._append_cost(...)` directly with synthesized
    rows once the batch completes.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ──────────────────────────────────────────────────────────────────────
# Request / response types
# ──────────────────────────────────────────────────────────────────────
# Anthropic enforces ``^[a-zA-Z0-9_-]{1,64}$`` on batch custom_ids —
# the strictest of the three providers. We target it as the lowest
# common denominator so the same custom_id round-trips everywhere.
_CUSTOM_ID_RE = re.compile(r"^[A-Za-z0-9_\-]+$")
_CUSTOM_ID_MAX = 64

# Compact variant codes — `A+C` becomes `AC`, etc., to (a) avoid the
# forbidden `+` and (b) stay under the 64-char id cap.
_VARIANT_TO_SHORT = {"C": "C", "A": "A", "B": "B", "A+C": "AC", "B+C": "BC"}
_SHORT_TO_VARIANT = {v: k for k, v in _VARIANT_TO_SHORT.items()}


def make_custom_id(run_id: str, scenario_id: str, variant: str, perm: str | int) -> str:
    """Build a deterministic custom_id that round-trips through every
    provider's allowed-charset filter.

    Format: ``<run_id>__<scenario_id>__<variant_short>__<perm>``

    Separator ``__`` (double underscore) avoids collision with the
    inputs (none of run_id / scenario_id / variant / perm naturally
    contains ``__``).

    Variants are encoded compactly — A+C → AC, B+C → BC — so the id
    fits the 64-char Anthropic cap.
    """
    short = _VARIANT_TO_SHORT.get(variant)
    if short is None:
        # Unknown variant — strip just '+' as a fallback.
        short = variant.replace("+", "")
    cid = f"{run_id}__{scenario_id}__{short}__{perm}"
    if not _CUSTOM_ID_RE.match(cid):
        raise ValueError(
            f"custom_id {cid!r} contains characters outside [A-Za-z0-9_-]. "
            f"Sanitize run_id / scenario_id / perm before calling make_custom_id."
        )
    if len(cid) > _CUSTOM_ID_MAX:
        raise ValueError(
            f"custom_id {cid!r} is {len(cid)} chars (>{_CUSTOM_ID_MAX}). "
            f"Shorten run_id (currently {len(run_id)}) — Anthropic caps "
            f"at {_CUSTOM_ID_MAX}."
        )
    return cid


def parse_custom_id(custom_id: str) -> dict[str, str]:
    """Inverse of `make_custom_id`. Returns a dict with run_id,
    scenario_id, variant, perm."""
    parts = custom_id.split("__")
    if len(parts) != 4:
        raise ValueError(
            f"custom_id {custom_id!r} doesn't have 4 __-separated parts"
        )
    run_id, sid, short, perm = parts
    return {
        "run_id": run_id,
        "scenario_id": sid,
        "variant": _SHORT_TO_VARIANT.get(short, short),
        "perm": perm,
    }


@dataclass
class BatchRequest:
    """One eval call to submit as part of a batch.

    `messages` is the OpenAI/OpenRouter chat-completions message list;
    each adapter rewrites this into the provider's native format.
    """
    custom_id: str
    model: str
    messages: list[dict[str, str]]
    max_tokens: int = 10000
    temperature: float = 1.0
    extra_params: dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchResult:
    """One result back from a batch.

    `response` matches the OpenRouter-shaped dict that
    `OpenRouterClient.complete()` returns — i.e., `{choices: [{message:
    {content}}], usage: {prompt_tokens, completion_tokens}, ...}`.
    """
    custom_id: str
    status: str   # "ok" | "error" | "cancelled" | "expired"
    response: dict[str, Any] | None = None
    error: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0


# ──────────────────────────────────────────────────────────────────────
# JSONL helpers
# ──────────────────────────────────────────────────────────────────────
def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    """Write a list of dicts as JSON-lines. UTF-8, no trailing newline."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for i, rec in enumerate(records):
            if i > 0:
                f.write("\n")
            f.write(json.dumps(rec, ensure_ascii=False))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Inverse of write_jsonl. Tolerates trailing newlines."""
    out: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


# ──────────────────────────────────────────────────────────────────────
# Provider dispatch
# ──────────────────────────────────────────────────────────────────────
def infer_provider(model: str) -> str:
    """
    Map a model id to its batch-API provider.

      * Bare anthropic id (e.g. ``claude-haiku-4-5-20251001``) → "anthropic"
      * Slug starting with ``openai/`` or any bare GPT id → "openai"
      * Slug starting with ``google/`` or ``gemini-`` → "gemini"

    Uses a "/" presence check first (matches the real-time path's
    convention from openrouter_client._infer_provider), then narrows
    by prefix.
    """
    m = model.lower()
    if "claude" in m and "/" not in model:
        return "anthropic"
    if m.startswith("openai/") or m.startswith("gpt-"):
        return "openai"
    if m.startswith("google/") or m.startswith("gemini-"):
        return "gemini"
    if "/" in model:
        # Default for OpenRouter-style slugs we haven't classified yet.
        # Caller can override via explicit `provider=` arg.
        raise ValueError(
            f"Cannot infer batch provider from model {model!r}. "
            f"Pass provider= explicitly to the adapter."
        )
    raise ValueError(f"Unknown model {model!r} for batch dispatch.")


# ──────────────────────────────────────────────────────────────────────
# Cost-log integration
# ──────────────────────────────────────────────────────────────────────
def synthesize_cost_rows(
    results: list[BatchResult],
    *,
    run_id: str,
    model: str,
    provider: str,
    or_pricing_id: str,
    prompt_p_per_mtok: float,
    completion_p_per_mtok: float,
    batch_discount: float = 0.5,
) -> list[dict[str, Any]]:
    """Build cost-log rows for a finished batch.

    Pricing applies the batch discount (default 50%). Latency is left
    blank — batches don't have meaningful per-call latency. Status is
    "ok" if the result has a usable response, else the underlying
    error string."""
    rows = []
    pp = prompt_p_per_mtok * (1.0 - batch_discount)
    cp = completion_p_per_mtok * (1.0 - batch_discount)
    now = datetime.now(timezone.utc).isoformat()
    for r in results:
        in_cost = (r.input_tokens / 1_000_000.0) * pp if r.status == "ok" else 0.0
        out_cost = (r.output_tokens / 1_000_000.0) * cp if r.status == "ok" else 0.0
        rows.append({
            "timestamp": now,
            "run_id": run_id,
            "model": model,
            "provider": f"{provider}_batch",
            "input_tokens": r.input_tokens,
            "output_tokens": r.output_tokens,
            "prompt_price_per_mtok": pp,
            "completion_price_per_mtok": cp,
            "input_cost_usd": in_cost,
            "output_cost_usd": out_cost,
            "total_cost_usd": in_cost + out_cost,
            "latency_ms": "",
            "status": r.status,
            "error": r.error or "",
            "pricing_fetched_at": now,
        })
    return rows


# ──────────────────────────────────────────────────────────────────────
# Status helper
# ──────────────────────────────────────────────────────────────────────
@dataclass
class BatchStatus:
    """Provider-agnostic status snapshot returned by adapter.poll()."""
    state: str          # "in_progress" | "ended" | "errored" | "cancelled"
    n_total: int
    n_succeeded: int = 0
    n_failed: int = 0
    n_pending: int = 0
    raw: dict[str, Any] = field(default_factory=dict)

    def is_done(self) -> bool:
        return self.state in ("ended", "errored", "cancelled")


# ──────────────────────────────────────────────────────────────────────
# Chunking
# ──────────────────────────────────────────────────────────────────────
def chunk_requests(
    requests: list[BatchRequest],
    *,
    max_count: int,
    max_bytes: int,
    bytes_per_request_fn=None,
) -> list[list[BatchRequest]]:
    """Split a request list into chunks that each satisfy a per-batch
    count and byte cap.

    Provider caps (mid-2026):
      anthropic: 100 000 requests, 256 MB
      openai:     50 000 requests, 200 MB
      gemini:     50 000 requests, 100 MB

    The byte size used is the JSON-serialized provider-specific record
    size — pass `bytes_per_request_fn(req) -> int` from the relevant
    adapter so this stays accurate. If omitted, falls back to a rough
    estimate based on raw message-content length × 1.1.

    Greedy bin-packing in submission order — preserves custom_id
    determinism within each chunk.
    """
    if bytes_per_request_fn is None:
        def bytes_per_request_fn(r: BatchRequest) -> int:
            n = sum(len(m.get("content", "")) for m in r.messages)
            return int(n * 1.15) + 256  # +overhead for JSON wrapping

    chunks: list[list[BatchRequest]] = []
    cur: list[BatchRequest] = []
    cur_bytes = 0
    for r in requests:
        b = bytes_per_request_fn(r)
        if b > max_bytes:
            raise ValueError(
                f"Single request {r.custom_id!r} estimated at {b:,} bytes — "
                f"exceeds per-batch cap of {max_bytes:,}. Cannot batch."
            )
        # Would adding this request push us over either cap?
        if cur and (len(cur) + 1 > max_count or cur_bytes + b > max_bytes):
            chunks.append(cur)
            cur = []
            cur_bytes = 0
        cur.append(r)
        cur_bytes += b
    if cur:
        chunks.append(cur)
    return chunks
