"""
Canonical API wrapper for the Safety Vigilance Benchmark project.

ALL pipeline scripts that make API calls MUST use this client. Do NOT
hand-roll requests or instantiate the Anthropic SDK in individual scripts —
the cost log and raw I/O log are centralized here and are the project's
single source of truth for spending and reproducibility.

Backends:
    - "openrouter" (default for ids containing "/")
        Sync:  urllib   ·  Async: aiohttp
    - "anthropic" (default for bare ids like "claude-haiku-4-5-...")
        Sync:  anthropic.Anthropic   ·  Async: anthropic.AsyncAnthropic

Logging:
    pipeline/api_logs/costs.csv       — append-only, every call, every run
    pipeline/api_logs/raw_io.csv      — full request/response, gzip-rotates
                                        every 100 rows to raw_io.NNN.csv.gz

Pricing:
    Live from OpenRouter `/api/v1/models`. NO HARDCODED FALLBACK PRICES.
    For Anthropic-native calls, pricing is looked up via a small
    Anthropic-id → OpenRouter-slug mapping (override per-call with
    `or_pricing_id=...` or globally via the constructor). Adding a new
    Anthropic model requires extending the mapping or passing the
    override — never silently guessing.

Environment:
    OPENROUTER_API_KEY  — required for OpenRouter backend
    ANTHROPIC_API_KEY   — required for Anthropic backend
    Both may live in <project_root>/.env (auto-loaded).

Usage (sync):
    from openrouter_client import OpenRouterClient
    c = OpenRouterClient(run_id="haiku45_sweepd_v3")
    c.validate_pricing()
    resp = c.complete(
        model="anthropic/claude-haiku-4.5",
        messages=[...],
        model_params={"temperature": 0.0, "max_tokens": 1024},
    )

Usage (async, with aiohttp lifecycle):
    async with OpenRouterClient(run_id="...") as c:
        c.validate_pricing()
        resp = await c.complete_async(model="...", messages=[...])
"""

from __future__ import annotations

import asyncio
import csv
import gzip
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Optional deps — required only when the matching backend is used.
try:
    import aiohttp  # noqa: F401
except Exception:  # noqa: BLE001
    aiohttp = None  # type: ignore[assignment]

try:
    import anthropic  # noqa: F401
except Exception:  # noqa: BLE001
    anthropic = None  # type: ignore[assignment]


# ──────────────────────────────────────────────────────────────────────
# Paths & constants
# ──────────────────────────────────────────────────────────────────────
_DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parent.parent

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"

ROTATE_EVERY = 100  # rotate raw_io.csv after this many data rows

# Anthropic native id → OpenRouter slug used for pricing lookup.
# This is a translation table only; PRICES still come live from OpenRouter.
# Add models here as the project uses them, or pass `or_pricing_id=` per call.
ANTHROPIC_TO_OR_PRICING_DEFAULT: dict[str, str] = {
    "claude-3-5-haiku-20241022":   "anthropic/claude-3.5-haiku",
    "claude-haiku-4-5-20251001":   "anthropic/claude-haiku-4.5",
    "claude-sonnet-4-5":           "anthropic/claude-sonnet-4.5",
    "claude-sonnet-4-5-20250929":  "anthropic/claude-sonnet-4.5",
    "claude-sonnet-4-5-20250514":  "anthropic/claude-sonnet-4.5",  # legacy alias
    # Anthropic frontier roster. Aliases resolve to current versions on
    # the API; OR pricing slugs are kept up to date.
    "claude-sonnet-4-6":           "anthropic/claude-sonnet-4.6",
    "claude-opus-4-6":             "anthropic/claude-opus-4.6",
    "claude-opus-4-7":             "anthropic/claude-opus-4.7",
}

# OpenAI / Gemini bare-slug → OpenRouter pricing-id maps. We use OR
# only as a pricing oracle (real OpenAI/Gemini batch calls go to those
# providers' native APIs); OR mirrors the upstream rates.
OPENAI_TO_OR_PRICING_DEFAULT: dict[str, str] = {
    # Verified 2026-05-01 against openai.com/api/pricing.
    "gpt-5.5":        "openai/gpt-5.5",
    "gpt-5.5-pro":    "openai/gpt-5.5-pro",
    "gpt-5.4":        "openai/gpt-5.4",
    "gpt-5.4-mini":   "openai/gpt-5.4-mini",
    "gpt-5.4-nano":   "openai/gpt-5.4-nano",
    "gpt-5.2":        "openai/gpt-5.2",
    "gpt-5.1":        "openai/gpt-5.1",
    "gpt-5":          "openai/gpt-5",
    "gpt-5-mini":     "openai/gpt-5-mini",
    "gpt-5-nano":     "openai/gpt-5-nano",
    "gpt-5-pro":      "openai/gpt-5-pro",
}

GEMINI_TO_OR_PRICING_DEFAULT: dict[str, str] = {
    # Verified 2026-05-02 against ai.google.dev/gemini-api/docs/pricing.
    # All Gemini 3.x are currently in -preview state.
    "gemini-3-flash-preview":        "google/gemini-3-flash-preview",
    "gemini-3.1-pro-preview":        "google/gemini-3.1-pro-preview",
    "gemini-3.1-flash-lite-preview": "google/gemini-3.1-flash-lite-preview",
    # Older 2.5 family — kept for back-compat with archived runs
    "gemini-2.5-pro":                "google/gemini-2.5-pro",
    "gemini-2.5-flash":              "google/gemini-2.5-flash",
}

COST_COLS = [
    "timestamp", "run_id", "model", "provider",
    "input_tokens", "output_tokens",
    "prompt_price_per_mtok", "completion_price_per_mtok",
    "input_cost_usd", "output_cost_usd", "total_cost_usd",
    "latency_ms", "status", "error",
    "pricing_fetched_at",
]

RAW_IO_COLS = [
    "timestamp", "run_id", "call_idx", "model", "provider",
    "model_params_json",
    "request_messages_json",
    "response_json",
    "error",
]


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────
def _load_dotenv(project_root: Path) -> None:
    """Tiny zero-dependency .env loader. Only handles KEY=VALUE lines and
    optional surrounding quotes — not the full bash spec.

    Behavior: a non-empty value already in os.environ wins. An empty string
    in os.environ is treated as unset and the .env value takes over (some
    shells export blank ANTHROPIC_API_KEY=, which would otherwise mask the
    real key)."""
    env_path = project_root / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k, v = k.strip(), v.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in ("'", '"'):
            v = v[1:-1]
        if not os.environ.get(k):  # treats "" same as unset
            os.environ[k] = v


def _infer_provider(model: str) -> str:
    """OpenRouter slugs always contain '/'. Bare Anthropic ids don't."""
    return "openrouter" if "/" in model else "anthropic"


# ──────────────────────────────────────────────────────────────────────
# Exceptions
# ──────────────────────────────────────────────────────────────────────
class PricingFetchError(RuntimeError):
    """Live pricing could not be obtained. We never fall back to stale
    or hardcoded prices because cost accuracy is critical."""


class OpenRouterAPIError(RuntimeError):
    """An OpenRouter chat-completions call failed."""


class AnthropicAPIError(RuntimeError):
    """An Anthropic SDK call failed."""


# ──────────────────────────────────────────────────────────────────────
# Client
# ──────────────────────────────────────────────────────────────────────
class OpenRouterClient:
    """Single canonical wrapper. Instantiate one per run.

    Despite the name (kept for backward compatibility), this client also
    routes Anthropic-native calls. All calls flow through the same
    cost log and raw I/O log.

    Args:
        run_id:                 free-form string tagged onto every log row.
        project_root:           override project root.
        anthropic_pricing_map:  override/extend the default Anthropic→OR
                                pricing-id mapping.
    """

    # Class-level lock so concurrent clients in one process serialize
    # their CSV writes against each other.
    _file_lock = threading.Lock()

    def __init__(
        self,
        run_id: str,
        project_root: Path | None = None,
        anthropic_pricing_map: dict[str, str] | None = None,
    ):
        if not run_id or not isinstance(run_id, str):
            raise ValueError("run_id must be a non-empty string")
        self.run_id = run_id

        root = Path(project_root) if project_root else _DEFAULT_PROJECT_ROOT
        _load_dotenv(root)

        # API keys are read from env (.env auto-loaded above). They are
        # NOT required at construction — required when the matching
        # backend is actually used. This lets tests instantiate freely.
        self.api_key = os.environ.get("OPENROUTER_API_KEY")
        self.anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")

        self.log_dir = root / "pipeline" / "api_logs"
        self.cost_csv = self.log_dir / "costs.csv"
        self.raw_io_csv = self.log_dir / "raw_io.csv"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self._init_csv(self.cost_csv, COST_COLS)
        self._init_csv(self.raw_io_csv, RAW_IO_COLS)

        self.anthropic_pricing_map = {
            **ANTHROPIC_TO_OR_PRICING_DEFAULT,
            **(anthropic_pricing_map or {}),
        }

        self.pricing: dict[str, dict[str, float]] | None = None
        self.pricing_fetched_at: str = ""

        self._cost_total: float = 0.0
        self._call_count: int = 0

        # Lazy backends
        self._anthropic_sync: Any = None
        self._anthropic_async: Any = None
        self._aiohttp_session: Any = None

    # ───── CSV bootstrap ──────────────────────────────────────────────
    @staticmethod
    def _init_csv(path: Path, cols: list[str]) -> None:
        if path.exists():
            return
        with path.open("w", newline="") as f:
            csv.writer(f).writerow(cols)

    # ───── Async lifecycle (for aiohttp session) ──────────────────────
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.aclose()

    async def aclose(self) -> None:
        """Close the aiohttp session if one was opened."""
        if self._aiohttp_session is not None:
            await self._aiohttp_session.close()
            self._aiohttp_session = None

    # ───── Pricing (live, no fallback) ────────────────────────────────
    def validate_pricing(self) -> None:
        """Eager pricing fetch. Run scripts SHOULD call this immediately
        after construction so a broken pricing endpoint is detected
        before the run starts."""
        self._ensure_pricing()

    def _ensure_pricing(self) -> None:
        if self.pricing is not None:
            return
        if not self.api_key:
            raise PricingFetchError(
                "OPENROUTER_API_KEY not set — required for live pricing fetch.\n"
                "Add it to <project>/.env or export it in your shell."
            )
        try:
            req = urllib.request.Request(
                OPENROUTER_MODELS_URL,
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read())
        except Exception as e:  # noqa: BLE001
            raise PricingFetchError(
                f"Could not fetch live pricing from {OPENROUTER_MODELS_URL}: {e}\n"
                "Cost accuracy requires live pricing — refusing to proceed.\n"
                "Check network / OpenRouter status and retry."
            ) from e

        pricing: dict[str, dict[str, float]] = {}
        for m in data.get("data", []):
            mid = m.get("id")
            p = m.get("pricing", {}) or {}
            try:
                prompt = float(p.get("prompt", 0))
                completion = float(p.get("completion", 0))
            except (TypeError, ValueError):
                continue
            if mid:
                pricing[mid] = {"prompt": prompt, "completion": completion}

        if not pricing:
            raise PricingFetchError("OpenRouter pricing response was empty.")

        self.pricing = pricing
        self.pricing_fetched_at = datetime.now(timezone.utc).isoformat()

    def _resolve_pricing_id(
        self, model: str, provider: str, or_pricing_id: str | None
    ) -> str:
        """Returns the OpenRouter slug used to look up live prices.

        For native OpenAI/Gemini batches we still use OR as a pricing
        oracle — OR mirrors the upstream rates and saves us from
        scraping multiple pricing endpoints. Calls actually go to the
        native APIs; only the price lookup is via OR.
        """
        if or_pricing_id:
            return or_pricing_id
        if provider in ("openrouter", "anthropic_batch"):
            return model
        if provider == "openrouter_batch":
            return model
        # Native-provider models — translate to OR slug.
        if provider in ("anthropic",):
            mapped = self.anthropic_pricing_map.get(model)
            label = "Anthropic"
            map_var = "ANTHROPIC_TO_OR_PRICING_DEFAULT"
        elif provider in ("openai",):
            mapped = OPENAI_TO_OR_PRICING_DEFAULT.get(model)
            label = "OpenAI"
            map_var = "OPENAI_TO_OR_PRICING_DEFAULT"
        elif provider in ("gemini",):
            mapped = GEMINI_TO_OR_PRICING_DEFAULT.get(model)
            label = "Gemini"
            map_var = "GEMINI_TO_OR_PRICING_DEFAULT"
        else:
            raise PricingFetchError(
                f"Unknown provider '{provider}' for pricing lookup."
            )
        if not mapped:
            raise PricingFetchError(
                f"No OpenRouter pricing mapping for {label} model '{model}'. "
                f"Either add it to {map_var} (in this file) "
                f"or pass `or_pricing_id='<vendor>/...'` to complete()."
            )
        return mapped

    def _model_pricing(self, or_id: str) -> tuple[float, float]:
        """Returns (prompt_per_token_usd, completion_per_token_usd)."""
        self._ensure_pricing()
        assert self.pricing is not None
        if or_id not in self.pricing:
            raise PricingFetchError(
                f"OR slug '{or_id}' not in OpenRouter pricing "
                f"({len(self.pricing)} models available). "
                f"Verify the slug at https://openrouter.ai/models."
            )
        p = self.pricing[or_id]
        return p["prompt"], p["completion"]

    # ───── Public: pre-run estimate ───────────────────────────────────
    def pre_run_estimate(
        self,
        model: str,
        n_calls: int,
        avg_in_tokens: int,
        avg_out_tokens: int,
        provider: str | None = None,
        or_pricing_id: str | None = None,
    ) -> float:
        """Estimate total USD cost for a planned run."""
        provider = provider or _infer_provider(model)
        or_id = self._resolve_pricing_id(model, provider, or_pricing_id)
        prompt_p, completion_p = self._model_pricing(or_id)
        per_call = avg_in_tokens * prompt_p + avg_out_tokens * completion_p
        return per_call * n_calls

    # ───── Log rotation ───────────────────────────────────────────────
    def _maybe_rotate_raw_io(self) -> None:
        """If raw_io.csv has > ROTATE_EVERY data rows, gzip it to the
        next sequential archive and start a fresh file with just the
        header. Caller must hold the file lock."""
        if not self.raw_io_csv.exists():
            return
        with self.raw_io_csv.open("rb") as f:
            n_data_rows = sum(1 for _ in f) - 1
        if n_data_rows < ROTATE_EVERY:
            return

        existing = sorted(self.log_dir.glob("raw_io.*.csv.gz"))
        idx = len(existing)
        archive = self.log_dir / f"raw_io.{idx:03d}.csv.gz"

        with self.raw_io_csv.open("rb") as src, gzip.open(archive, "wb") as dst:
            dst.write(src.read())
        with self.raw_io_csv.open("w", newline="") as f:
            csv.writer(f).writerow(RAW_IO_COLS)

    def _append_cost(self, row: dict[str, Any]) -> None:
        with self._file_lock:
            with self.cost_csv.open("a", newline="") as f:
                csv.writer(f).writerow([row.get(c, "") for c in COST_COLS])

    def _append_raw_io(self, row: dict[str, Any]) -> None:
        with self._file_lock:
            self._maybe_rotate_raw_io()
            with self.raw_io_csv.open("a", newline="") as f:
                csv.writer(f).writerow([row.get(c, "") for c in RAW_IO_COLS])

    # ───── Unified post-call logging ──────────────────────────────────
    def _log_call(
        self,
        *,
        ts: str,
        call_idx: int,
        model: str,
        provider: str,
        model_params: dict[str, Any],
        messages: list[dict[str, str]],
        response_data: dict[str, Any],
        in_tok: int,
        out_tok: int,
        prompt_p_tok: float,
        completion_p_tok: float,
        latency_ms: int,
        status: str,
        error: str,
    ) -> None:
        """Single logging path used by every backend (sync/async, OR/Anthropic)."""
        in_cost = in_tok * prompt_p_tok
        out_cost = out_tok * completion_p_tok
        total_cost = in_cost + out_cost

        self._append_cost({
            "timestamp": ts,
            "run_id": self.run_id,
            "model": model,
            "provider": provider,
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "prompt_price_per_mtok": f"{prompt_p_tok * 1_000_000:.6f}",
            "completion_price_per_mtok": f"{completion_p_tok * 1_000_000:.6f}",
            "input_cost_usd": f"{in_cost:.6f}",
            "output_cost_usd": f"{out_cost:.6f}",
            "total_cost_usd": f"{total_cost:.6f}",
            "latency_ms": latency_ms,
            "status": status,
            "error": error,
            "pricing_fetched_at": self.pricing_fetched_at,
        })
        self._append_raw_io({
            "timestamp": ts,
            "run_id": self.run_id,
            "call_idx": call_idx,
            "model": model,
            "provider": provider,
            "model_params_json": json.dumps(model_params, sort_keys=True),
            "request_messages_json": json.dumps(messages),
            "response_json": json.dumps(response_data),
            "error": error,
        })

    # ───── Anthropic helpers ──────────────────────────────────────────
    def _split_anthropic_messages(
        self, messages: list[dict[str, str]]
    ) -> tuple[str | None, list[dict[str, str]]]:
        """Anthropic SDK takes `system` separately from the message list."""
        system = None
        out: list[dict[str, str]] = []
        for m in messages:
            if m.get("role") == "system":
                system = m.get("content", "")
            else:
                out.append(m)
        return system, out

    def _anthropic_response_to_dict(self, resp: Any) -> dict[str, Any]:
        """Normalize Anthropic SDK response to an OpenRouter-shaped dict
        so downstream code that expects choices[0].message.content works."""
        try:
            text = "".join(
                block.text for block in resp.content if getattr(block, "type", None) == "text"
            )
        except Exception:  # noqa: BLE001
            text = ""
        return {
            "id": getattr(resp, "id", ""),
            "model": getattr(resp, "model", ""),
            "choices": [{"message": {"role": "assistant", "content": text}, "finish_reason": getattr(resp, "stop_reason", "")}],
            "usage": {
                "prompt_tokens": getattr(resp.usage, "input_tokens", 0) if getattr(resp, "usage", None) else 0,
                "completion_tokens": getattr(resp.usage, "output_tokens", 0) if getattr(resp, "usage", None) else 0,
            },
        }

    # ───── Public: sync chat completion ───────────────────────────────
    def complete(
        self,
        model: str,
        messages: list[dict[str, str]],
        model_params: dict[str, Any] | None = None,
        provider: str | None = None,
        timeout: float = 120.0,
        or_pricing_id: str | None = None,
    ) -> dict[str, Any]:
        """Sync chat completion. See module docstring for usage."""
        provider = provider or _infer_provider(model)
        or_id = self._resolve_pricing_id(model, provider, or_pricing_id)
        # Validate pricing exists BEFORE we burn an API call we can't price
        self._model_pricing(or_id)

        model_params = dict(model_params or {})
        ts = datetime.now(timezone.utc).isoformat()
        self._call_count += 1
        call_idx = self._call_count

        if provider == "openrouter":
            response_data, in_tok, out_tok, latency_ms, status, error = \
                self._do_openrouter_sync(model, messages, model_params, timeout)
        elif provider == "anthropic":
            response_data, in_tok, out_tok, latency_ms, status, error = \
                self._do_anthropic_sync(model, messages, model_params, timeout)
        else:
            raise ValueError(f"Unknown provider: {provider!r}")

        prompt_p, completion_p = self._model_pricing(or_id)
        if status == "ok":
            self._cost_total += in_tok * prompt_p + out_tok * completion_p

        self._log_call(
            ts=ts, call_idx=call_idx, model=model, provider=provider,
            model_params=model_params, messages=messages,
            response_data=response_data, in_tok=in_tok, out_tok=out_tok,
            prompt_p_tok=prompt_p, completion_p_tok=completion_p,
            latency_ms=latency_ms, status=status, error=error,
        )

        if status != "ok":
            cls = OpenRouterAPIError if provider == "openrouter" else AnthropicAPIError
            raise cls(f"{provider} call failed [{status}]: {error}")
        return response_data

    # ───── Public: async chat completion ──────────────────────────────
    async def complete_async(
        self,
        model: str,
        messages: list[dict[str, str]],
        model_params: dict[str, Any] | None = None,
        provider: str | None = None,
        timeout: float = 120.0,
        or_pricing_id: str | None = None,
    ) -> dict[str, Any]:
        """Async chat completion. Use under `async with OpenRouterClient(...)`
        so the aiohttp session is cleaned up properly."""
        provider = provider or _infer_provider(model)
        or_id = self._resolve_pricing_id(model, provider, or_pricing_id)
        self._model_pricing(or_id)

        model_params = dict(model_params or {})
        ts = datetime.now(timezone.utc).isoformat()
        self._call_count += 1
        call_idx = self._call_count

        if provider == "openrouter":
            response_data, in_tok, out_tok, latency_ms, status, error = \
                await self._do_openrouter_async(model, messages, model_params, timeout)
        elif provider == "anthropic":
            response_data, in_tok, out_tok, latency_ms, status, error = \
                await self._do_anthropic_async(model, messages, model_params, timeout)
        else:
            raise ValueError(f"Unknown provider: {provider!r}")

        prompt_p, completion_p = self._model_pricing(or_id)
        if status == "ok":
            self._cost_total += in_tok * prompt_p + out_tok * completion_p

        # CSV writes use a thread lock; safe to call from the event loop.
        self._log_call(
            ts=ts, call_idx=call_idx, model=model, provider=provider,
            model_params=model_params, messages=messages,
            response_data=response_data, in_tok=in_tok, out_tok=out_tok,
            prompt_p_tok=prompt_p, completion_p_tok=completion_p,
            latency_ms=latency_ms, status=status, error=error,
        )

        if status != "ok":
            cls = OpenRouterAPIError if provider == "openrouter" else AnthropicAPIError
            raise cls(f"{provider} call failed [{status}]: {error}")
        return response_data

    # ───── Backend: OpenRouter sync ───────────────────────────────────
    def _do_openrouter_sync(self, model, messages, model_params, timeout):
        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY not set")
        params = {"max_tokens": 10000, **model_params}  # caller can override
        body = {"model": model, "messages": messages, **params}
        start = time.time()
        error, status, response_data = "", "ok", None
        try:
            req = urllib.request.Request(
                OPENROUTER_API_URL,
                data=json.dumps(body).encode(),
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://localhost/svb",
                    "X-Title": "Safety Vigilance Benchmark",
                },
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                response_data = json.loads(resp.read())
        except urllib.error.HTTPError as e:
            try:
                body_text = e.read().decode(errors="replace")[:500]
            except Exception:  # noqa: BLE001
                body_text = "<unreadable>"
            error, status = f"HTTP {e.code}: {body_text}", "http_error"
            response_data = {"error": error}
        except Exception as e:  # noqa: BLE001
            error, status = f"{type(e).__name__}: {e}", "error"
            response_data = {"error": error}
        latency_ms = int((time.time() - start) * 1000)
        usage = (response_data or {}).get("usage") or {}
        in_tok = int(usage.get("prompt_tokens", 0) or 0)
        out_tok = int(usage.get("completion_tokens", 0) or 0)
        return response_data, in_tok, out_tok, latency_ms, status, error

    # ───── Backend: OpenRouter async ──────────────────────────────────
    async def _do_openrouter_async(self, model, messages, model_params, timeout):
        if aiohttp is None:
            raise RuntimeError("aiohttp not installed; pip install aiohttp")
        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY not set")
        if self._aiohttp_session is None or self._aiohttp_session.closed:
            self._aiohttp_session = aiohttp.ClientSession()
        params = {"max_tokens": 10000, **model_params}  # caller can override
        body = {"model": model, "messages": messages, **params}
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://localhost/svb",
            "X-Title": "Safety Vigilance Benchmark",
        }
        start = time.time()
        error, status, response_data = "", "ok", None
        try:
            async with self._aiohttp_session.post(
                OPENROUTER_API_URL, json=body, headers=headers,
                timeout=aiohttp.ClientTimeout(total=timeout),
            ) as resp:
                if resp.status >= 400:
                    body_text = (await resp.text())[:500]
                    error, status = f"HTTP {resp.status}: {body_text}", "http_error"
                    response_data = {"error": error}
                else:
                    response_data = await resp.json()
        except Exception as e:  # noqa: BLE001
            error, status = f"{type(e).__name__}: {e}", "error"
            response_data = {"error": error}
        latency_ms = int((time.time() - start) * 1000)
        usage = (response_data or {}).get("usage") or {}
        in_tok = int(usage.get("prompt_tokens", 0) or 0)
        out_tok = int(usage.get("completion_tokens", 0) or 0)
        return response_data, in_tok, out_tok, latency_ms, status, error

    # ───── Backend: Anthropic sync ────────────────────────────────────
    def _do_anthropic_sync(self, model, messages, model_params, timeout):
        if anthropic is None:
            raise RuntimeError("anthropic SDK not installed; pip install anthropic")
        if not self.anthropic_api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        if self._anthropic_sync is None:
            self._anthropic_sync = anthropic.Anthropic(api_key=self.anthropic_api_key)
        system, msgs = self._split_anthropic_messages(messages)
        # Anthropic SDK requires max_tokens; default to 10000 if absent.
        params = dict(model_params)
        params.setdefault("max_tokens", 10000)
        kwargs: dict[str, Any] = {"model": model, "messages": msgs, **params}
        if system is not None:
            kwargs["system"] = system
        start = time.time()
        error, status, response_data = "", "ok", None
        in_tok = out_tok = 0
        try:
            resp = self._anthropic_sync.messages.create(timeout=timeout, **kwargs)
            response_data = self._anthropic_response_to_dict(resp)
            in_tok = response_data["usage"]["prompt_tokens"]
            out_tok = response_data["usage"]["completion_tokens"]
        except Exception as e:  # noqa: BLE001
            error, status = f"{type(e).__name__}: {e}", "error"
            response_data = {"error": error}
        latency_ms = int((time.time() - start) * 1000)
        return response_data, in_tok, out_tok, latency_ms, status, error

    # ───── Backend: Anthropic async ───────────────────────────────────
    async def _do_anthropic_async(self, model, messages, model_params, timeout):
        if anthropic is None:
            raise RuntimeError("anthropic SDK not installed; pip install anthropic")
        if not self.anthropic_api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        if self._anthropic_async is None:
            self._anthropic_async = anthropic.AsyncAnthropic(api_key=self.anthropic_api_key)
        system, msgs = self._split_anthropic_messages(messages)
        # Anthropic SDK requires max_tokens; default to 10000 if absent.
        params = dict(model_params)
        params.setdefault("max_tokens", 10000)
        kwargs: dict[str, Any] = {"model": model, "messages": msgs, **params}
        if system is not None:
            kwargs["system"] = system
        start = time.time()
        error, status, response_data = "", "ok", None
        in_tok = out_tok = 0
        try:
            resp = await self._anthropic_async.messages.create(timeout=timeout, **kwargs)
            response_data = self._anthropic_response_to_dict(resp)
            in_tok = response_data["usage"]["prompt_tokens"]
            out_tok = response_data["usage"]["completion_tokens"]
        except Exception as e:  # noqa: BLE001
            error, status = f"{type(e).__name__}: {e}", "error"
            response_data = {"error": error}
        latency_ms = int((time.time() - start) * 1000)
        return response_data, in_tok, out_tok, latency_ms, status, error

    # ───── Convenience accessors ──────────────────────────────────────
    # ───── Public: log a batch of pre-completed results ──────────────
    def log_batch_results(
        self,
        results: list,                 # list[BatchResult]
        *,
        model: str,
        provider: str,
        or_pricing_id: str | None = None,
        batch_discount: float = 0.5,
    ) -> float:
        """Append cost + raw I/O rows for a batch of completed results.

        Used by `pipeline.batch_runner` after a provider batch job
        finishes. Cost rows are tagged with the synthesized provider
        string ``f"{provider}_batch"`` so they're distinguishable from
        real-time calls in `costs.csv`. Returns the total billed USD.

        `batch_discount` defaults to 0.5 (the published rate for all
        three currently-supported providers as of 2026-05).
        Pass 0.0 to log at full real-time pricing.
        """
        # Lazy: only resolve pricing once we know we have results.
        if not results:
            return 0.0
        or_id = self._resolve_pricing_id(model, provider, or_pricing_id)
        prompt_p_tok, completion_p_tok = self._model_pricing(or_id)
        prompt_p_tok *= (1.0 - batch_discount)
        completion_p_tok *= (1.0 - batch_discount)

        total = 0.0
        provider_tag = f"{provider}_batch"
        for r in results:
            ts = datetime.now(timezone.utc).isoformat()
            self._call_count += 1
            in_tok = int(r.input_tokens or 0)
            out_tok = int(r.output_tokens or 0)
            in_cost = in_tok * prompt_p_tok if r.status == "ok" else 0.0
            out_cost = out_tok * completion_p_tok if r.status == "ok" else 0.0
            row_total = in_cost + out_cost
            total += row_total
            if r.status == "ok":
                self._cost_total += row_total
            self._append_cost({
                "timestamp": ts,
                "run_id": self.run_id,
                "model": model,
                "provider": provider_tag,
                "input_tokens": in_tok,
                "output_tokens": out_tok,
                "prompt_price_per_mtok": f"{prompt_p_tok * 1_000_000:.6f}",
                "completion_price_per_mtok": f"{completion_p_tok * 1_000_000:.6f}",
                "input_cost_usd": f"{in_cost:.6f}",
                "output_cost_usd": f"{out_cost:.6f}",
                "total_cost_usd": f"{row_total:.6f}",
                "latency_ms": "",
                "status": r.status,
                "error": (r.error or "")[:500],
                "pricing_fetched_at": self.pricing_fetched_at,
            })
            # Raw I/O parity with real-time path. Request messages aren't
            # available here (they're in the source prompt files), so we
            # log only the response + the custom_id for traceback.
            self._append_raw_io({
                "timestamp": ts,
                "run_id": self.run_id,
                "call_idx": self._call_count,
                "model": model,
                "provider": provider_tag,
                "model_params_json": json.dumps({"batch_discount": batch_discount}),
                "request_messages_json": json.dumps({"custom_id": r.custom_id}),
                "response_json": json.dumps(r.response or {}),
                "error": r.error or "",
            })
        return total

    def total_cost(self) -> float:
        return self._cost_total

    def call_count(self) -> int:
        return self._call_count

    def print_progress(self, i: int, n: int, prefix: str = "") -> None:
        avg = self._cost_total / max(self._call_count, 1)
        eta_total = avg * n
        sys.stderr.write(
            f"\r{prefix}[{self.run_id}] {i}/{n} "
            f"· ${self._cost_total:.4f} so far · ~${eta_total:.2f} ETA total"
        )
        sys.stderr.flush()
        if i >= n:
            sys.stderr.write("\n")
