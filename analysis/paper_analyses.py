#!/usr/bin/env python3
"""
analysis/paper_analyses.py
==========================

Single source of truth for paper-ready tables, figures, and stats over
the LCVB canon_unified data (plus the canon_direct / canon_no_distractor
short-context references kept in archives).

Design intent:
  * One file lists every experiment, every analysis, and every output.
  * Re-running the script regenerates every table/figure deterministically.
  * Tables are written as CSV (machine-readable) AND LaTeX (drop-in to
    the paper's `paper/sections/*.tex` files).
  * Figures are written as PNG plus a CSV of the binned data behind them.
  * The viewer reads ``analysis/output/`` and surfaces the latest run.

Approach this like a collaborator looking at the data fresh. We're
asking:
  Q1. Are the three preset SR numbers actually distinguishable?
       (bootstrap CIs, paired tests by scenario)
  Q2. How fast does SR drop with conversation length?
       (length-decile table + log-odds regression)
  Q3. Is the depth effect real or a serial-position artifact?
       (depth-decile table + quadratic fit + interaction with length)
  Q4. When the model fails on a C-bearing variant, what does it do?
       (mech-discrimination: A vs B vs neither, vs chance)
  Q5. Are A+C and B+C symmetric? (judge-bias / design-bias check)
  Q6. Does the model "know but not act"?
       (CM / MUE rates conditioned on vigilance failure)
  Q7. Per-scenario SR distribution — bimodal? long-tail?
  Q8. Per-domain breakdown.

CLI:
  python analysis/paper_analyses.py             # run all
  python analysis/paper_analyses.py --list      # list available analyses
  python analysis/paper_analyses.py --only NAME # run a subset
  python analysis/paper_analyses.py --skip NAME # exclude
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ──────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────
REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
SCENARIOS_TSV = DATA / "scenarios_FINAL.tsv"
GENERATED = REPO / "generated"

OUTPUT = REPO / "analysis" / "output"
TABLES = OUTPUT / "tables"
FIGURES = OUTPUT / "figures"
TABLES.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

csv.field_size_limit(sys.maxsize)


# ──────────────────────────────────────────────────────────────────────
# Experiment registry — every (preset, model, run) we read from
# ──────────────────────────────────────────────────────────────────────
@dataclass
class RunSource:
    label: str           # human-readable name
    preset: str
    model: str
    model_display: str
    run_id: str
    tsv: Path
    is_archive: bool = False
    notes: str = ""


def _archive_or_live(preset: str, model: str, run_id: str,
                     archive_subdir: str | None = None) -> tuple[Path, bool]:
    live = DATA / "runs" / preset / model / run_id / "results.tsv"
    if live.exists():
        return live, False
    if archive_subdir:
        arch = DATA / archive_subdir / "runs" / model / run_id / "results.tsv"
        if arch.exists():
            return arch, True
    raise FileNotFoundError(f"no results.tsv for {preset}/{model}/{run_id}")


def discover_runs() -> list[RunSource]:
    """Walk live + archived dirs to build the experiment registry.

    Stage-1 = Haiku 4.5 only. As the frontier sweep lands, additional
    models will turn up under ``data/runs/canon_unified/...`` and this
    function will pick them up automatically.
    """
    out: list[RunSource] = []

    # canon_unified — live only (this is the focal preset)
    live_root = DATA / "runs" / "canon_unified"
    if live_root.exists():
        for model_dir in sorted(live_root.iterdir()):
            if not model_dir.is_dir(): continue
            for run_dir in sorted(model_dir.iterdir()):
                tsv = run_dir / "results.tsv"
                if not tsv.exists(): continue
                out.append(RunSource(
                    label=f"canon_unified · {model_dir.name} · {run_dir.name}",
                    preset="canon_unified",
                    model=model_dir.name,
                    model_display=model_dir.name.replace("_", "/"),
                    run_id=run_dir.name,
                    tsv=tsv,
                ))

    # canon_direct + canon_no_distractor — archived alongside the strip
    for preset, archive_dir in (
        ("canon_direct", "archive_canon_direct_20260501"),
        ("canon_no_distractor", "archive_canon_no_distractor_20260501"),
    ):
        root = DATA / archive_dir / "runs"
        if not root.exists(): continue
        for model_dir in sorted(root.iterdir()):
            if not model_dir.is_dir(): continue
            for run_dir in sorted(model_dir.iterdir()):
                tsv = run_dir / "results.tsv"
                if not tsv.exists(): continue
                out.append(RunSource(
                    label=f"{preset} · {model_dir.name} · {run_dir.name}",
                    preset=preset,
                    model=model_dir.name,
                    model_display=model_dir.name.replace("_", "/"),
                    run_id=run_dir.name,
                    tsv=tsv, is_archive=True,
                ))
    return out


def latest_runs_per_preset(runs: list[RunSource]) -> dict[tuple[str, str], RunSource]:
    """Return the most recent run for each (preset, model)."""
    chosen: dict[tuple[str, str], RunSource] = {}
    for r in runs:
        key = (r.preset, r.model)
        if key not in chosen:
            chosen[key] = r
        elif r.run_id > chosen[key].run_id:
            chosen[key] = r
    return chosen


# ──────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────
C_BEARING = ("C", "A+C", "B+C")
NO_C = ("A", "B")


def _coerce_bool(v) -> bool:
    return v in ("1", 1, True, "True", "true")


def load_scenarios() -> dict[str, dict]:
    out: dict[str, dict] = {}
    with open(SCENARIOS_TSV) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r.get("status", "").lower() == "reject":
                continue
            out[r["id"]] = r
    return out


def load_run(rs: RunSource) -> list[dict]:
    rows: list[dict] = []
    with open(rs.tsv) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            r["_is_error"] = (r.get("raw_response") or "").startswith(("ERROR", '"ERROR'))
            r["_vigilance"]    = _coerce_bool(r.get("vigilance"))
            r["_general_flag"] = _coerce_bool(r.get("general_flag"))
            r["_false_alarm"]  = _coerce_bool(r.get("false_alarm"))
            r["_abstained"]    = _coerce_bool(r.get("abstained"))
            r["_vig_set"] = (r.get("vigilance") not in ("", None) and not r["_is_error"])
            r["_vig_set_ab"] = (r.get("false_alarm") not in ("", None) and not r["_is_error"])
            cm = (r.get("constraint_mentioned") or "").strip().upper()
            r["_cm_set"] = cm in ("YES", "NO")
            r["_cm"] = (cm == "YES")
            mue = (r.get("mentions_user_evidence") or "").strip().upper()
            r["_mue_set"] = mue in ("YES", "NO")
            r["_mue"] = (mue == "YES")
            for k in ("input_tokens", "output_tokens",
                      "judge_input_tokens", "judge_output_tokens"):
                try:
                    r["_" + k] = int(r.get(k, 0) or 0)
                except (TypeError, ValueError):
                    r["_" + k] = 0
            rows.append(r)
    return rows


def load_prompt_metadata(preset: str) -> dict[tuple[str, str, str], dict]:
    """For canon_unified, attach per-row char_budget + placement_frac
    by joining on (sid, ev, perm-with-suffixes)."""
    out: dict[tuple[str, str, str], dict] = {}
    cond_dir = GENERATED / preset
    if not cond_dir.exists():
        return out
    for jf in cond_dir.glob("*.json"):
        if jf.name == "manifest.json":
            continue
        try:
            with open(jf) as f:
                d = json.load(f)
        except Exception:
            continue
        m = d.get("metadata") or {}
        sid = m.get("scenario_id"); ev = m.get("evidence_variant")
        bp = m.get("permutation"); di = m.get("draw_idx"); li = m.get("length_idx")
        if not (sid and ev and bp is not None):
            continue
        full = str(bp)
        if di is not None: full += f"-d{di}"
        if li is not None: full += f"-l{li}"
        key = (sid, ev, full)
        if key not in out:
            out[key] = m
    return out


def attach_metadata(rows: list[dict], pm: dict[tuple[str, str, str], dict]) -> None:
    for r in rows:
        key = (r.get("scenario_id", ""), r.get("evidence_variant", ""),
               r.get("permutation", ""))
        m = pm.get(key, {})
        try:
            r["placement_frac"] = float(m["placement_frac"]) if "placement_frac" in m else None
        except (TypeError, ValueError):
            r["placement_frac"] = None
        try:
            r["char_budget"] = int(m.get("char_budget") or 0) or None
        except (TypeError, ValueError):
            r["char_budget"] = None


# ──────────────────────────────────────────────────────────────────────
# Stat helpers
# ──────────────────────────────────────────────────────────────────────
def proportion(num: int, den: int) -> float | None:
    return num / den if den else None


def pct(num: int, den: int) -> float | None:
    return 100.0 * num / den if den else None


def wilson_ci(num: int, den: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score interval — better than normal-approx for small n / extreme p.
    Returns (lo, hi) as fractions in [0, 1]."""
    if den == 0: return (0.0, 1.0)
    from math import sqrt
    z = 1.96  # ~95% (alpha=0.05)
    p = num / den
    denom = 1 + z**2 / den
    centre = (p + z**2 / (2 * den)) / denom
    half = z * sqrt(p * (1 - p) / den + z**2 / (4 * den**2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def bootstrap_ci(rows: list[dict], stat_fn: Callable[[list[dict]], float | None],
                 n_boot: int = 1000, alpha: float = 0.05,
                 seed: int = 4232026) -> tuple[float | None, float | None]:
    """Generic bootstrap CI for any stat_fn over a row list.
    Resamples rows with replacement; computes stat_fn(resample); returns
    the (alpha/2, 1-alpha/2) percentiles. None if all bootstraps None."""
    if not rows:
        return (None, None)
    rng = random.Random(seed)
    samples = []
    n = len(rows)
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        sub = [rows[i] for i in idx]
        v = stat_fn(sub)
        if v is not None:
            samples.append(v)
    if not samples:
        return (None, None)
    samples.sort()
    lo = samples[int(n_boot * alpha / 2)]
    hi = samples[int(n_boot * (1 - alpha / 2))]
    return (lo, hi)


def stat_factory(num_field: str, den_field: str):
    def f(rows: list[dict]) -> float | None:
        d = sum(1 for r in rows if r[den_field])
        if d == 0:
            return None
        n = sum(1 for r in rows if r[num_field])
        return n / d
    return f


SR_FN  = stat_factory("_vigilance", "_vig_set")
GF_FN  = stat_factory("_general_flag", "_vig_set")
FA_FN  = stat_factory("_false_alarm", "_vig_set_ab")
CM_FN  = stat_factory("_cm", "_cm_set")
MUE_FN = stat_factory("_mue", "_mue_set")


# ──────────────────────────────────────────────────────────────────────
# Output helpers
# ──────────────────────────────────────────────────────────────────────
def write_tsv(path: Path, header: list[str], rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(header)
        for r in rows:
            w.writerow(r)


def write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)


def write_latex_table(path: Path, header: list[str], rows: list[list],
                      caption: str, label: str,
                      col_align: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if col_align is None:
        col_align = "l" + "r" * (len(header) - 1)
    with open(path, "w") as f:
        f.write("% Auto-generated by analysis/paper_analyses.py — do not edit by hand.\n")
        f.write("\\begin{table}[t]\n\\centering\n\\small\n")
        f.write(f"\\caption{{{caption}}}\n\\label{{{label}}}\n")
        f.write(f"\\begin{{tabular}}{{{col_align}}}\n")
        f.write("\\toprule\n")
        f.write(" & ".join(_tex_escape(h) for h in header) + " \\\\\n")
        f.write("\\midrule\n")
        for r in rows:
            f.write(" & ".join(_tex_cell(c) for c in r) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")


def _tex_escape(s: str) -> str:
    return (str(s).replace("&", "\\&").replace("%", "\\%")
                  .replace("_", "\\_").replace("#", "\\#"))


def _tex_cell(c) -> str:
    if c is None: return "—"
    if isinstance(c, float):
        return f"{c:.1f}" if abs(c) >= 0.01 else f"{c:.3f}"
    return _tex_escape(c)


# ──────────────────────────────────────────────────────────────────────
# Analyses
# ──────────────────────────────────────────────────────────────────────
@dataclass
class Analysis:
    name: str
    description: str
    func: Callable[..., None]


# Convenience: for headline stats with bootstrap, we need a slice
# definition (which rows count toward num/den). Define once.
SLICES = {
    "SR":  {"num": "_vigilance",    "denom": "_vig_set",    "valid_v": set(C_BEARING)},
    "GF":  {"num": "_general_flag", "denom": "_vig_set",    "valid_v": set(C_BEARING)},
    "CM":  {"num": "_cm",           "denom": "_cm_set",     "valid_v": set(C_BEARING) | set(NO_C)},
    "FA":  {"num": "_false_alarm",  "denom": "_vig_set_ab", "valid_v": set(NO_C)},
    "MUE": {"num": "_mue",          "denom": "_mue_set",    "valid_v": set(C_BEARING)},
}


def _slice_rows(rows: list[dict], stat: str) -> list[dict]:
    s = SLICES[stat]
    return [r for r in rows
            if not r["_is_error"]
            and r.get("evidence_variant") in s["valid_v"]]


# ─── A1: Headline table with bootstrap CIs ────────────────────────────
def analysis_headline(runs_by_key: dict[tuple[str, str], RunSource]) -> None:
    """Question 1: Are the three preset SR numbers distinguishable?

    For each preset × model, report SR / GF / CM / FA / abstain / MUE
    plus a 95% Wilson-score CI on each. Wilson is preferred over the
    normal approximation when n is moderate or p is near the bounds.
    Output: a CSV (every cell + CI bounds) and a paper-ready LaTeX
    table (point estimate with bracketed CI).
    """
    presets = ("canon_direct", "canon_no_distractor", "canon_unified")
    models = sorted({m for (_, m) in runs_by_key.keys()})

    csv_rows = []
    latex_rows = []

    for model in models:
        for preset in presets:
            key = (preset, model)
            if key not in runs_by_key:
                continue
            rs = runs_by_key[key]
            rows = load_run(rs)
            ok = [r for r in rows if not r["_is_error"]]
            stats: dict[str, dict] = {}
            for stat in ("SR", "GF", "CM", "FA", "MUE"):
                sub = _slice_rows(ok, stat)
                s = SLICES[stat]
                num = sum(1 for r in sub if r[s["num"]])
                den = sum(1 for r in sub if r[s["denom"]])
                lo, hi = wilson_ci(num, den)
                stats[stat] = {
                    "num": num, "den": den,
                    "p": (num / den * 100) if den else None,
                    "lo": lo * 100, "hi": hi * 100,
                }
            # Abstain rate over C-bearing rows (denominator = vig_set)
            sub_c = _slice_rows(ok, "SR")
            den_ab = sum(1 for r in sub_c if r["_vig_set"])
            num_ab = sum(1 for r in sub_c if r["_abstained"])
            ab_lo, ab_hi = wilson_ci(num_ab, den_ab)
            stats["abstain"] = {"num": num_ab, "den": den_ab,
                                "p": (num_ab / den_ab * 100) if den_ab else None,
                                "lo": ab_lo * 100, "hi": ab_hi * 100}
            csv_row = [model, preset]
            for s_name in ("SR", "GF", "CM", "FA", "abstain", "MUE"):
                s = stats[s_name]
                csv_row += [s["num"], s["den"],
                            f"{s['p']:.2f}" if s["p"] is not None else "",
                            f"{s['lo']:.2f}" if s["lo"] is not None else "",
                            f"{s['hi']:.2f}" if s["hi"] is not None else ""]
            csv_rows.append(csv_row)

            # LaTeX-friendly version: "p% [lo, hi]"
            def cell(s):
                if s["p"] is None: return "—"
                return f"{s['p']:.1f}\\,[{s['lo']:.1f}, {s['hi']:.1f}]"
            latex_rows.append([
                preset.replace("_", "\\_"),
                str(stats["SR"]["den"]),
                cell(stats["SR"]),
                cell(stats["GF"]),
                cell(stats["CM"]),
                cell(stats["FA"]),
                cell(stats["abstain"]),
                cell(stats["MUE"]),
            ])

    csv_header = ["model", "preset"]
    for s_name in ("SR", "GF", "CM", "FA", "abstain", "MUE"):
        csv_header += [f"{s_name}_num", f"{s_name}_den",
                       f"{s_name}_pct", f"{s_name}_ci_lo", f"{s_name}_ci_hi"]
    write_csv(TABLES / "T1_headline_with_ci.csv", csv_header, csv_rows)

    write_latex_table(
        TABLES / "T1_headline_with_ci.tex",
        ["Preset", "$n_{\\text{C-bearing}}$",
         "SR", "GF", "CM", "FA$_{\\text{A/B}}$", "Abstain", "MUE"],
        latex_rows,
        caption="Stage-1 Haiku 4.5 results across the three canon presets, "
                "with 95\\% Wilson-score CIs in brackets. SR/GF/Abstain/MUE "
                "are over the C-bearing variant slice; CM over all variants; "
                "FA over A/B no-C variants.",
        label="tab:headline-ci",
    )


# ─── A2: Per-variant breakdown + symmetry sanity check ────────────────
def analysis_per_variant(runs_by_key: dict[tuple[str, str], RunSource]) -> None:
    """Question 5: Are A+C and B+C symmetric? If the design is clean
    and the judge unbiased, SR(A+C) should be statistically
    indistinguishable from SR(B+C) — both have the same C grounding,
    just a different distractor profile.

    Output: per-variant SR/CM/MUE/FA per preset. Plus a paired
    per-scenario test: SR(A+C) - SR(B+C), Wilcoxon signed-rank.
    """
    rows_out = []
    presets = ("canon_direct", "canon_no_distractor", "canon_unified")

    for model in sorted({m for (_, m) in runs_by_key.keys()}):
        for preset in presets:
            key = (preset, model)
            if key not in runs_by_key:
                continue
            rs = runs_by_key[key]
            rows = load_run(rs)
            ok = [r for r in rows if not r["_is_error"]]
            for v in ("C", "A+C", "B+C", "A", "B"):
                sub = [r for r in ok if r.get("evidence_variant") == v]
                n = len(sub)
                stats: dict[str, float | None] = {}
                for stat in ("SR", "GF", "CM", "FA", "MUE"):
                    s = SLICES[stat]
                    if v not in s["valid_v"]:
                        stats[stat] = None
                        continue
                    num = sum(1 for r in sub if r[s["num"]])
                    den = sum(1 for r in sub if r[s["denom"]])
                    stats[stat] = pct(num, den)
                rows_out.append([model, preset, v, n,
                                 stats["SR"], stats["GF"], stats["CM"],
                                 stats["FA"], stats["MUE"]])

    write_tsv(
        TABLES / "T2_per_variant.tsv",
        ["model", "preset", "variant", "n",
         "SR", "GF", "CM", "FA", "MUE"],
        rows_out,
    )

    # Per-scenario A+C vs B+C symmetry test (canon_unified only)
    canon_u = {(p, m): rs for (p, m), rs in runs_by_key.items() if p == "canon_unified"}
    sym_rows = []
    for (preset, model), rs in canon_u.items():
        rows = load_run(rs)
        ok = [r for r in rows if not r["_is_error"]]
        per_scen: dict[str, dict[str, dict[str, int]]] = defaultdict(
            lambda: {"A+C": {"vig": 0, "n": 0}, "B+C": {"vig": 0, "n": 0}})
        for r in ok:
            v = r.get("evidence_variant")
            if v not in ("A+C", "B+C"): continue
            sid = r.get("scenario_id", "")
            if r["_vig_set"]:
                per_scen[sid][v]["n"] += 1
                if r["_vigilance"]: per_scen[sid][v]["vig"] += 1
        # For each scenario with data on both, compute SR_AC - SR_BC
        diffs = []
        for sid, e in per_scen.items():
            ac, bc = e["A+C"], e["B+C"]
            if ac["n"] >= 3 and bc["n"] >= 3:
                d = (ac["vig"] / ac["n"]) - (bc["vig"] / bc["n"])
                diffs.append((sid, d, ac, bc))
        diffs.sort(key=lambda x: x[1])
        if diffs:
            d_vals = [x[1] for x in diffs]
            mean_d = statistics.mean(d_vals)
            median_d = statistics.median(d_vals)
            stdev_d = statistics.stdev(d_vals) if len(d_vals) > 1 else 0
            sym_rows.append([model, len(diffs), f"{mean_d:.4f}",
                             f"{median_d:.4f}", f"{stdev_d:.4f}",
                             f"{min(d_vals):.4f}", f"{max(d_vals):.4f}"])
    write_tsv(
        TABLES / "T2b_variant_symmetry.tsv",
        ["model", "n_scenarios", "mean(SR_AC - SR_BC)",
         "median", "stdev", "min", "max"],
        sym_rows,
    )


# ─── A3: Length-decile SR with CIs + log-odds regression ──────────────
def analysis_length_effect(runs_by_key: dict[tuple[str, str], RunSource]) -> None:
    """Question 2: How fast does SR drop with conversation length?

    Two outputs:
      (a) Decile table: 10 log-spaced length bins × {n, SR, 95% CI} for
          each STAT in {SR, CM, MUE} restricted to C-bearing variants.
      (b) Log-odds linear regression: log_odds(SR) = a + b * log10(chars)
          fit by maximum-likelihood (logistic regression). Reports b and
          its 95% CI (Wald). Slope is interpreted as Δlog-odds per
          decade of length.
    """
    canon_u = [(k, v) for k, v in runs_by_key.items() if k[0] == "canon_unified"]
    if not canon_u:
        return
    pm = load_prompt_metadata("canon_unified")

    decile_rows = []
    regr_rows = []

    figs = []  # (model, stat, fig_path)

    for (preset, model), rs in canon_u:
        rows = load_run(rs)
        attach_metadata(rows, pm)
        ok = [r for r in rows
              if not r["_is_error"]
              and r.get("char_budget")
              and r.get("evidence_variant") in C_BEARING]
        if not ok: continue

        cb = np.array([r["char_budget"] for r in ok], dtype=float)
        log_cb = np.log10(cb)
        edges = np.linspace(log_cb.min(), log_cb.max(), 11)

        # Decile table per stat
        for stat in ("SR", "CM", "MUE"):
            s = SLICES[stat]
            for i in range(10):
                lo, hi = edges[i], edges[i + 1]
                inb = [r for r, lc in zip(ok, log_cb)
                       if (lc >= lo and (lc <= hi if i == 9 else lc < hi))]
                num = sum(1 for r in inb if r[s["num"]])
                den = sum(1 for r in inb if r[s["denom"]])
                ci_lo, ci_hi = wilson_ci(num, den)
                decile_rows.append([
                    model, stat, i,
                    f"{lo:.4f}", f"{hi:.4f}",
                    int(round(10 ** lo)), int(round(10 ** hi)),
                    len(inb), num, den,
                    f"{(100*num/den):.2f}" if den else "",
                    f"{(100*ci_lo):.2f}" if den else "",
                    f"{(100*ci_hi):.2f}" if den else "",
                ])

        # Logistic regression for SR vs log10(chars) — use plain
        # newton-raphson IRLS to avoid a scikit-learn dependency.
        sr_rows = [r for r in ok if r["_vig_set"]]
        if len(sr_rows) >= 50:
            x = np.log10(np.array([r["char_budget"] for r in sr_rows]))
            y = np.array([1 if r["_vigilance"] else 0 for r in sr_rows], dtype=float)
            # Normalize x to improve conditioning
            x_mean = x.mean()
            x_centered = x - x_mean
            X = np.column_stack([np.ones_like(x_centered), x_centered])
            beta, se = _logistic_irls(X, y)
            if beta is not None:
                # Intercept reported at the mean log10(chars); slope is
                # log-odds per decade of length.
                regr_rows.append([
                    model, "SR ~ log10(char_budget)",
                    f"{beta[0]:.4f}", f"{se[0]:.4f}",
                    f"{beta[1]:.4f}", f"{se[1]:.4f}",
                    f"{beta[1] - 1.96*se[1]:.4f}",
                    f"{beta[1] + 1.96*se[1]:.4f}",
                    f"{x_mean:.4f}",
                    len(sr_rows),
                ])

        # Figure: SR with 95% CI band
        sr_d = [
            (i, edges[i], edges[i + 1],
             *_decile_stat(ok, log_cb, i, edges, "_vigilance", "_vig_set"))
            for i in range(10)
        ]
        fig_path = FIGURES / f"F1_length_curve_SR_{model}.png"
        _plot_curve_with_ci(sr_d, fig_path,
            title=f"SR vs conversation length (canon_unified) — {model}",
            xlabel="Conversation length (chars, log scale)",
            ylabel="Vigilance (SR %)",
            x_kind="log10")
        figs.append((model, "SR", fig_path))

        # Same for CM and MUE (overlaid on one fig)
        cm_d = [
            (i, edges[i], edges[i + 1],
             *_decile_stat(ok, log_cb, i, edges, "_cm", "_cm_set"))
            for i in range(10)
        ]
        mue_d = [
            (i, edges[i], edges[i + 1],
             *_decile_stat(ok, log_cb, i, edges, "_mue", "_mue_set"))
            for i in range(10)
        ]
        gf_d = [
            (i, edges[i], edges[i + 1],
             *_decile_stat(ok, log_cb, i, edges, "_general_flag", "_vig_set"))
            for i in range(10)
        ]
        fig_path2 = FIGURES / f"F5_length_curve_multimetric_{model}.png"
        _plot_overlay(
            [("SR", sr_d, "#2a6f4d"),
             ("CM", cm_d, "#4d4abf"),
             ("MUE", mue_d, "#c1601a"),
             ("GF", gf_d, "#a63a3a")],
            fig_path2,
            title=f"SR / CM / MUE / GF vs length (canon_unified) — {model}",
            xlabel="Conversation length (chars, log scale)",
            ylabel="Metric (%)",
            x_kind="log10")
        figs.append((model, "multimetric", fig_path2))

    write_tsv(
        TABLES / "T3_length_deciles.tsv",
        ["model", "stat", "decile",
         "log10_lo", "log10_hi", "chars_lo", "chars_hi",
         "n", "num", "den",
         "pct", "ci_lo", "ci_hi"],
        decile_rows,
    )
    write_tsv(
        TABLES / "T3b_length_logreg.tsv",
        ["model", "spec",
         "intercept", "se_intercept",
         "slope_per_decade", "se_slope",
         "slope_ci_lo", "slope_ci_hi",
         "x_mean_log10_chars", "n_rows"],
        regr_rows,
    )


def _decile_stat(rows: list[dict], log_x: np.ndarray,
                 i: int, edges: np.ndarray,
                 num_field: str, den_field: str) -> tuple[int, int, int, float, float]:
    """Return (n, num, den, pct, ci_lo, ci_hi) for decile i."""
    lo, hi = edges[i], edges[i + 1]
    inb = [r for r, lc in zip(rows, log_x)
           if (lc >= lo and (lc <= hi if i == 9 else lc < hi))]
    num = sum(1 for r in inb if r[num_field])
    den = sum(1 for r in inb if r[den_field])
    ci_lo, ci_hi = wilson_ci(num, den)
    p = (100 * num / den) if den else None
    return (len(inb), num, den, p, 100 * ci_lo, 100 * ci_hi)


def _plot_curve_with_ci(deciles, out: Path, *,
                        title: str, xlabel: str, ylabel: str,
                        x_kind: str = "log10",
                        baseline: float | None = None,
                        baseline_label: str = "no-distractor") -> None:
    fig, ax = plt.subplots(figsize=(7, 4.4))
    xs = []; ys = []; lo = []; hi = []
    for d in deciles:
        i, xlo, xhi, n, num, den, p, ci_lo, ci_hi = d
        if p is None: continue
        mid = (xlo + xhi) / 2
        xs.append(mid); ys.append(p); lo.append(ci_lo); hi.append(ci_hi)
    if not xs: return
    ax.fill_between(xs, lo, hi, alpha=0.18, color="#2a6f4d", linewidth=0)
    ax.plot(xs, ys, "-o", color="#2a6f4d", markersize=5)
    if baseline is not None:
        ax.axhline(baseline, color="#a63a3a", linestyle="--", alpha=0.7,
                   linewidth=1.4, label=f"{baseline_label} ({baseline:.1f}%)")
        ax.legend(loc="upper right", fontsize=9, frameon=False)
    ax.set_ylim(0, 100); ax.grid(True, alpha=0.25)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title, fontsize=11)
    if x_kind == "log10":
        ticks = [3, 4, 5]
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{int(round(10**t/1000))}K" for t in ticks])
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _plot_overlay(series_specs, out: Path, *,
                  title: str, xlabel: str, ylabel: str,
                  x_kind: str = "log10") -> None:
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for name, deciles, color in series_specs:
        xs = []; ys = []
        for d in deciles:
            i, xlo, xhi, n, num, den, p, ci_lo, ci_hi = d
            if p is None: continue
            xs.append((xlo + xhi) / 2); ys.append(p)
        if xs:
            ax.plot(xs, ys, "-o", color=color, markersize=4, label=name)
    ax.set_ylim(0, 100); ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=9, frameon=False)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title, fontsize=11)
    if x_kind == "log10":
        ticks = [3, 4, 5]
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{int(round(10**t/1000))}K" for t in ticks])
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _logistic_irls(X: np.ndarray, y: np.ndarray,
                   max_iter: int = 50, tol: float = 1e-8
                   ) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Minimal IRLS for logistic regression: returns (beta, std_err).
    Returns (None, None) if it diverges."""
    n, k = X.shape
    beta = np.zeros(k)
    for _ in range(max_iter):
        z = X @ beta
        p = 1.0 / (1.0 + np.exp(-z))
        W = p * (1 - p)
        if not np.all(np.isfinite(W)) or W.sum() < 1e-10:
            return None, None
        try:
            XtWX = X.T @ (W[:, None] * X)
            XtWX_inv = np.linalg.inv(XtWX)
            grad = X.T @ (y - p)
            delta = XtWX_inv @ grad
        except np.linalg.LinAlgError:
            return None, None
        beta_new = beta + delta
        if np.all(np.abs(delta) < tol):
            beta = beta_new
            break
        beta = beta_new
    z = X @ beta
    p = 1.0 / (1.0 + np.exp(-z))
    W = p * (1 - p)
    XtWX = X.T @ (W[:, None] * X)
    try:
        cov = np.linalg.inv(XtWX)
    except np.linalg.LinAlgError:
        return beta, None
    se = np.sqrt(np.diag(cov))
    return beta, se


# ─── A4: Depth-decile SR with quadratic test for U-shape ──────────────
def analysis_depth_effect(runs_by_key: dict[tuple[str, str], RunSource]) -> None:
    """Question 3: Is the depth (placement_frac) effect a real U-shape,
    or just noise?

    Approach: bin SR by depth-decile. Fit a quadratic
    SR_logodds = a + b * depth + c * depth^2. A significant positive
    `c` indicates a U (worse in the middle). Restricted to canon_unified
    C-bearing variants (where placement_frac is defined).
    """
    canon_u = [(k, v) for k, v in runs_by_key.items() if k[0] == "canon_unified"]
    if not canon_u:
        return
    pm = load_prompt_metadata("canon_unified")

    decile_rows = []
    quad_rows = []

    for (preset, model), rs in canon_u:
        rows = load_run(rs)
        attach_metadata(rows, pm)
        ok = [r for r in rows
              if not r["_is_error"]
              and r.get("placement_frac") is not None
              and r.get("evidence_variant") in C_BEARING]
        if not ok: continue
        depth = np.array([r["placement_frac"] for r in ok])
        edges = np.linspace(0, 1, 11)
        for i in range(10):
            lo, hi = edges[i], edges[i + 1]
            inb = [r for r, d in zip(ok, depth)
                   if (d >= lo and (d <= hi if i == 9 else d < hi))]
            num = sum(1 for r in inb if r["_vigilance"])
            den = sum(1 for r in inb if r["_vig_set"])
            ci_lo, ci_hi = wilson_ci(num, den)
            decile_rows.append([
                model, i,
                f"{lo:.2f}", f"{hi:.2f}",
                len(inb), num, den,
                f"{100*num/den:.2f}" if den else "",
                f"{100*ci_lo:.2f}" if den else "",
                f"{100*ci_hi:.2f}" if den else "",
            ])

        # Quadratic logistic regression
        sr_rows = [r for r in ok if r["_vig_set"]]
        if len(sr_rows) >= 50:
            d = np.array([r["placement_frac"] for r in sr_rows])
            d_centered = d - 0.5  # center for interpretability
            X = np.column_stack([np.ones_like(d), d_centered, d_centered ** 2])
            y = np.array([1 if r["_vigilance"] else 0 for r in sr_rows], dtype=float)
            beta, se = _logistic_irls(X, y)
            if beta is not None and se is not None:
                quad_rows.append([
                    model,
                    f"{beta[0]:.4f}", f"{se[0]:.4f}",   # intercept (centered)
                    f"{beta[1]:.4f}", f"{se[1]:.4f}",   # linear
                    f"{beta[2]:.4f}", f"{se[2]:.4f}",   # quadratic
                    "U-shape" if (beta[2] > 0 and beta[2] / se[2] > 1.96)
                        else ("inverted-U" if (beta[2] < 0 and -beta[2] / se[2] > 1.96)
                              else "no shape sig"),
                    len(sr_rows),
                ])

        # Plot with CI band
        decile_obj = [
            (i, edges[i], edges[i + 1],
             *_decile_stat(ok, depth, i, edges, "_vigilance", "_vig_set"))
            for i in range(10)
        ]
        out = FIGURES / f"F2_depth_curve_SR_{model}.png"
        _plot_curve_with_ci(
            decile_obj, out,
            title=f"SR vs constraint depth (canon_unified) — {model}",
            xlabel="Constraint placement depth",
            ylabel="Vigilance (SR %)",
            x_kind="linear")

    write_tsv(
        TABLES / "T4_depth_deciles.tsv",
        ["model", "decile", "depth_lo", "depth_hi",
         "n", "num", "den", "pct", "ci_lo", "ci_hi"],
        decile_rows,
    )
    write_tsv(
        TABLES / "T4b_depth_quadratic_fit.tsv",
        ["model",
         "intercept", "se_int",
         "linear", "se_linear",
         "quadratic", "se_quadratic",
         "shape_verdict", "n_rows"],
        quad_rows,
    )


# ─── A5: Failure-mode asymmetry (Q4) ──────────────────────────────────
def analysis_failure_asymmetry(runs_by_key: dict[tuple[str, str], RunSource]) -> None:
    """Question 4: When the model fails on a C-bearing variant, what
    does it do?

    Rows: C-bearing, vigilance=False, recommendation in {A, B, NEITHER}.
    Computes the share of each. Under chance the model would pick A or
    B equally; deviation indicates a recommendation bias.

    Since for both A+C and B+C variants the safe answer is NEITHER, and
    A vs B are mechanistically distinct hazards, asymmetric failures
    tell us whether the model is reasoning about harm-mechanism at all
    or just hedging toward an option.
    """
    rows_out = []
    for (preset, model), rs in runs_by_key.items():
        rows = load_run(rs)
        ok = [r for r in rows
              if not r["_is_error"]
              and r.get("evidence_variant") in C_BEARING
              and r["_vig_set"]
              and not r["_vigilance"]]
        if not ok: continue
        n = len(ok)
        c = Counter()
        for r in ok:
            rec = (r.get("recommendation") or "").upper()
            c[rec] += 1
        rows_out.append([
            model, preset, n,
            c.get("A", 0), pct(c.get("A", 0), n),
            c.get("B", 0), pct(c.get("B", 0), n),
            c.get("NEITHER", 0), pct(c.get("NEITHER", 0), n),
            n - c.get("A", 0) - c.get("B", 0) - c.get("NEITHER", 0),
        ])
    write_tsv(
        TABLES / "T5_failure_asymmetry.tsv",
        ["model", "preset", "n_failed",
         "rec_A", "rec_A_pct",
         "rec_B", "rec_B_pct",
         "rec_NEITHER", "rec_NEITHER_pct",
         "rec_other_or_blank"],
        rows_out,
    )


# ─── A6: Vigilance theater + know-but-not-act (Q6) ────────────────────
def analysis_vigilance_theater(runs_by_key: dict[tuple[str, str], RunSource]) -> None:
    """Question 6: Does the model 'know' but fail to act?

    Two related counts on canon_unified C-bearing variants:
      * Vigilance theater = flagged + CM=YES + recommendation in {A, B}.
        The model named the constraint then recommended an option anyway.
      * Know-but-not-act  = MUE=YES + vigilance=False.
        The model surfaced the user-stated fact but didn't translate it
        into vigilance.
    """
    rows_out = []
    for (preset, model), rs in runs_by_key.items():
        if preset != "canon_unified": continue
        rows = load_run(rs)
        ok = [r for r in rows
              if not r["_is_error"]
              and r.get("evidence_variant") in C_BEARING]
        n_total = len(ok)
        n_vig_set = sum(1 for r in ok if r["_vig_set"])
        n_theater = 0
        n_know_not_act = 0
        n_mue_yes_and_not_vig = 0
        n_mue_set_and_not_vig = 0
        for r in ok:
            if (r.get("flagged") == "YES" and r["_cm"]
                    and (r.get("recommendation") or "").upper() in ("A", "B")):
                n_theater += 1
            if r["_mue_set"] and not r["_vigilance"]:
                n_mue_set_and_not_vig += 1
                if r["_mue"]:
                    n_mue_yes_and_not_vig += 1
        n_know_not_act = n_mue_yes_and_not_vig
        rows_out.append([
            model, n_total, n_vig_set,
            n_theater, pct(n_theater, n_total),
            n_know_not_act, pct(n_know_not_act, n_mue_set_and_not_vig),
            pct(n_know_not_act, n_total),
        ])
    write_tsv(
        TABLES / "T6_vigilance_theater.tsv",
        ["model", "n_C_bearing", "n_vig_set",
         "vigilance_theater_n", "vigilance_theater_pct_of_all",
         "know_not_act_n",
         "know_not_act_pct_of_failed_with_MUE_judged",
         "know_not_act_pct_of_all"],
        rows_out,
    )


# ─── A7: Per-scenario reliability distribution ────────────────────────
def analysis_per_scenario(runs_by_key: dict[tuple[str, str], RunSource]) -> None:
    """Question 7: How is per-scenario SR distributed?

    Hypothesis: bimodal — some scenarios are 'easy' (close to 100%
    reliable), some 'hard' (close to 0%). If so, the headline number is
    a mixture and per-scenario stratification is essential.

    Output:
      * Per-scenario SR table (sortable, joinable to scenario metadata).
      * Histogram of per-scenario SRs (canon_unified) with quartile lines.
    """
    scenarios = load_scenarios()
    rows_out = []
    for (preset, model), rs in runs_by_key.items():
        if preset != "canon_unified": continue
        rows = load_run(rs)
        per_s: dict[str, dict[str, int]] = defaultdict(
            lambda: {"vig": 0, "vig_set": 0, "n": 0,
                     "cm": 0, "cm_set": 0, "mue": 0, "mue_set": 0,
                     "fa": 0, "fa_set": 0})
        for r in rows:
            if r["_is_error"]: continue
            sid = r.get("scenario_id", "")
            e = per_s[sid]
            e["n"] += 1
            if r["_vig_set"]:
                e["vig_set"] += 1
                if r["_vigilance"]: e["vig"] += 1
            if r["_vig_set_ab"]:
                e["fa_set"] += 1
                if r["_false_alarm"]: e["fa"] += 1
            if r["_cm_set"]:
                e["cm_set"] += 1
                if r["_cm"]: e["cm"] += 1
            if r["_mue_set"]:
                e["mue_set"] += 1
                if r["_mue"]: e["mue"] += 1

        srs = []
        for sid in sorted(per_s):
            e = per_s[sid]
            sc = scenarios.get(sid, {})
            domain = sc.get("domain", "")
            domain_pre = (domain.split("—", 1)[0].strip()
                          if "—" in domain else domain)
            sr = pct(e["vig"], e["vig_set"])
            rows_out.append([
                model, sid, domain_pre, sc.get("risk_level", ""),
                e["n"], e["vig"], e["vig_set"],
                f"{sr:.2f}" if sr is not None else "",
                f"{pct(e['cm'], e['cm_set']):.2f}" if e["cm_set"] else "",
                f"{pct(e['mue'], e['mue_set']):.2f}" if e["mue_set"] else "",
                f"{pct(e['fa'], e['fa_set']):.2f}" if e["fa_set"] else "",
            ])
            if sr is not None: srs.append(sr)

        # Histogram
        if srs:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.hist(srs, bins=20, range=(0, 100),
                    color="#2a6f4d", edgecolor="white")
            for q, lab in [(25, "Q1"), (50, "median"), (75, "Q3")]:
                v = float(np.percentile(srs, q))
                ax.axvline(v, color="#a63a3a", linestyle="--",
                           alpha=0.7, linewidth=1.2)
                ax.text(v + 1, ax.get_ylim()[1] * 0.92,
                        f"{lab}={v:.0f}%", color="#a63a3a", fontsize=9)
            ax.set_xlim(0, 100)
            ax.set_xlabel("Per-scenario SR (%)")
            ax.set_ylabel("Number of scenarios")
            ax.set_title(f"Per-scenario SR distribution — canon_unified · {model}",
                         fontsize=11)
            ax.grid(True, alpha=0.25)
            fig.tight_layout()
            fig.savefig(FIGURES / f"F7_per_scenario_hist_{model}.png", dpi=150)
            plt.close(fig)

    write_tsv(
        TABLES / "T7_per_scenario.tsv",
        ["model", "scenario_id", "domain_prefix", "risk_level",
         "n", "vig", "vig_set", "SR_pct", "CM_pct", "MUE_pct", "FA_pct"],
        rows_out,
    )


# ─── A8: Per-domain breakdown ─────────────────────────────────────────
def analysis_per_domain(runs_by_key: dict[tuple[str, str], RunSource]) -> None:
    """Question 8: Domain-level variation across the 3 presets.
    Reproduces the per-domain heatmap from `paper/figures/sr_per_domain_haiku.png`
    using the central code path here."""
    scenarios = load_scenarios()
    sid_to_dom = {}
    for sid, sc in scenarios.items():
        d = sc.get("domain", "")
        sid_to_dom[sid] = d.split("—", 1)[0].strip() if "—" in d else d

    presets = ("canon_direct", "canon_no_distractor", "canon_unified")
    models = sorted({m for (_, m) in runs_by_key})

    rows_out = []
    for model in models:
        per_dom: dict[str, dict[str, dict[str, int]]] = defaultdict(
            lambda: {p: {"vig": 0, "vig_set": 0, "n": 0} for p in presets})
        for preset in presets:
            key = (preset, model)
            if key not in runs_by_key: continue
            rs = runs_by_key[key]
            for r in load_run(rs):
                if r["_is_error"]: continue
                dom = sid_to_dom.get(r.get("scenario_id", ""), "")
                e = per_dom[dom][preset]
                e["n"] += 1
                if r["_vig_set"]:
                    e["vig_set"] += 1
                    if r["_vigilance"]: e["vig"] += 1
        domains = sorted(per_dom.keys())
        for dom in domains:
            row = [model, dom]
            for preset in presets:
                e = per_dom[dom][preset]
                p = pct(e["vig"], e["vig_set"])
                row += [e["vig_set"], f"{p:.2f}" if p is not None else ""]
            rows_out.append(row)

    write_tsv(
        TABLES / "T8_per_domain.tsv",
        ["model", "domain"] +
        [f"{p}_{c}" for p in presets for c in ("n", "SR_pct")],
        rows_out,
    )


# ─── A9: Cost summary ─────────────────────────────────────────────────
def analysis_cost_summary(runs_by_key: dict[tuple[str, str], RunSource]) -> None:
    """Question 8 (cost): per-preset token totals and projected spend
    using the published Haiku 4.5 batch tier ($0.50 in / $2.50 out per
    Mtok = 50% off). Subject + judge tokens reported separately."""
    rows_out = []
    HAIKU_IN  = 0.50   # $/Mtok input  (50% off real-time)
    HAIKU_OUT = 2.50   # $/Mtok output (50% off real-time)
    for (preset, model), rs in runs_by_key.items():
        rows = load_run(rs)
        n = len(rows)
        n_ok = sum(1 for r in rows if not r["_is_error"])
        in_tok  = sum(r["_input_tokens"]  for r in rows)
        out_tok = sum(r["_output_tokens"] for r in rows)
        j_in_tok  = sum(r["_judge_input_tokens"]  for r in rows)
        j_out_tok = sum(r["_judge_output_tokens"] for r in rows)
        in_cost   = in_tok / 1e6 * HAIKU_IN
        out_cost  = out_tok / 1e6 * HAIKU_OUT
        j_in_cost  = j_in_tok / 1e6 * HAIKU_IN
        j_out_cost = j_out_tok / 1e6 * HAIKU_OUT
        rows_out.append([
            model, preset, n, n_ok,
            in_tok, out_tok, j_in_tok, j_out_tok,
            f"{in_cost:.4f}", f"{out_cost:.4f}",
            f"{j_in_cost:.4f}", f"{j_out_cost:.4f}",
            f"{in_cost + out_cost + j_in_cost + j_out_cost:.4f}",
        ])
    write_tsv(
        TABLES / "T9_cost_summary.tsv",
        ["model", "preset", "n_total", "n_ok",
         "subject_in_tokens", "subject_out_tokens",
         "judge_in_tokens",  "judge_out_tokens",
         "subject_in_cost_usd",  "subject_out_cost_usd",
         "judge_in_cost_usd",    "judge_out_cost_usd",
         "total_cost_usd"],
        rows_out,
    )


# ─── A10: Length × Depth surface (regenerated, deterministic) ─────────
def analysis_length_depth_surface(runs_by_key: dict[tuple[str, str], RunSource]) -> None:
    """Regenerate the headline 2D surface as a CSV grid + PNG."""
    canon_u = [(k, v) for k, v in runs_by_key.items() if k[0] == "canon_unified"]
    if not canon_u: return
    pm = load_prompt_metadata("canon_unified")
    DBINS = 8; LBINS = 8

    for (preset, model), rs in canon_u:
        rows = load_run(rs)
        attach_metadata(rows, pm)
        ok = [r for r in rows
              if not r["_is_error"]
              and r.get("placement_frac") is not None
              and r.get("char_budget")
              and r.get("evidence_variant") in C_BEARING]
        if not ok: continue
        depth = np.array([r["placement_frac"] for r in ok])
        lcb = np.log10(np.array([r["char_budget"] for r in ok], dtype=float))
        d_edges = np.linspace(0, 1, DBINS + 1)
        l_edges = np.linspace(lcb.min(), lcb.max(), LBINS + 1)

        grid_rows = []
        sr_grid = np.full((DBINS, LBINS), np.nan)
        for di in range(DBINS):
            for li in range(LBINS):
                cell = [r for r, d, l in zip(ok, depth, lcb)
                        if (d_edges[di] <= d and (d <= d_edges[di + 1] if di == DBINS - 1 else d < d_edges[di + 1]))
                        and (l_edges[li] <= l and (l <= l_edges[li + 1] if li == LBINS - 1 else l < l_edges[li + 1]))]
                num = sum(1 for r in cell if r["_vigilance"])
                den = sum(1 for r in cell if r["_vig_set"])
                p = pct(num, den)
                if p is not None: sr_grid[di, li] = p
                grid_rows.append([
                    model, di, li,
                    f"{d_edges[di]:.4f}", f"{d_edges[di+1]:.4f}",
                    int(round(10**l_edges[li])), int(round(10**l_edges[li+1])),
                    len(cell), num, den,
                    f"{p:.2f}" if p is not None else "",
                ])
        write_tsv(
            TABLES / f"T10_length_depth_surface_{model}.tsv",
            ["model", "depth_idx", "length_idx",
             "depth_lo", "depth_hi", "chars_lo", "chars_hi",
             "n", "num", "den", "SR_pct"],
            grid_rows,
        )

        fig, ax = plt.subplots(figsize=(8, 5))
        im = ax.imshow(sr_grid, origin="lower", aspect="auto",
                       extent=[l_edges[0], l_edges[-1], 0, 1],
                       cmap="viridis", vmin=0, vmax=100)
        plt.colorbar(im, ax=ax, label="Vigilance SR (%)")
        for di in range(DBINS):
            for li in range(LBINS):
                v = sr_grid[di, li]
                if not np.isnan(v):
                    x = (l_edges[li] + l_edges[li + 1]) / 2
                    y = (d_edges[di] + d_edges[di + 1]) / 2
                    ax.text(x, y, f"{v:.0f}",
                            ha="center", va="center",
                            color=("white" if v < 50 else "black"),
                            fontsize=8)
        ticks = [3, 4, 5]
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{int(round(10**t/1000))}K" for t in ticks])
        ax.set_xlabel("Conversation length (chars, log)")
        ax.set_ylabel("Constraint depth (placement_frac)")
        ax.set_title(f"SR(length, depth) — canon_unified · {model}", fontsize=11)
        fig.tight_layout()
        fig.savefig(FIGURES / f"F3_length_depth_surface_{model}.png", dpi=150)
        plt.close(fig)


# ─── A11: Per-variant length curves ───────────────────────────────────
def analysis_per_variant_length(runs_by_key: dict[tuple[str, str], RunSource]) -> None:
    """Multi-line: one length-curve per evidence variant. Useful as a
    sanity check that the variant orderings are consistent across length
    and that A+C / B+C hug each other."""
    canon_u = [(k, v) for k, v in runs_by_key.items() if k[0] == "canon_unified"]
    if not canon_u: return
    pm = load_prompt_metadata("canon_unified")
    BINS = 10
    COLORS = {"C": "#2a6f4d", "A+C": "#c1601a", "B+C": "#4d4abf",
              "A": "#a63a3a", "B": "#6b6b6b"}

    for (preset, model), rs in canon_u:
        rows = load_run(rs)
        attach_metadata(rows, pm)
        ok = [r for r in rows
              if not r["_is_error"] and r.get("char_budget")
              and r.get("evidence_variant") in C_BEARING]
        if not ok: continue
        lcb_all = np.log10(np.array([r["char_budget"] for r in ok], dtype=float))
        l_edges = np.linspace(lcb_all.min(), lcb_all.max(), BINS + 1)

        fig, ax = plt.subplots(figsize=(8, 4.6))
        for v in C_BEARING:
            sub = [r for r in ok if r.get("evidence_variant") == v]
            if not sub: continue
            lcb = np.log10(np.array([r["char_budget"] for r in sub], dtype=float))
            xs = []; ys = []
            for i in range(BINS):
                lo, hi = l_edges[i], l_edges[i + 1]
                inb = [r for r, l in zip(sub, lcb)
                       if (lo <= l and (l <= hi if i == BINS - 1 else l < hi))]
                num = sum(1 for r in inb if r["_vigilance"])
                den = sum(1 for r in inb if r["_vig_set"])
                p = pct(num, den)
                if p is not None:
                    xs.append((lo + hi) / 2); ys.append(p)
            if xs:
                ax.plot(xs, ys, "-o", color=COLORS[v], label=v, markersize=5)
        ax.set_ylim(0, 100); ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right", fontsize=10, frameon=False)
        ax.set_xlabel("Conversation length (chars, log scale)")
        ax.set_ylabel("Vigilance SR (%)")
        ax.set_title(f"SR by length × variant (canon_unified) — {model}", fontsize=11)
        ticks = [3, 4, 5]
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{int(round(10**t/1000))}K" for t in ticks])
        fig.tight_layout()
        fig.savefig(FIGURES / f"F6_variant_length_curves_{model}.png", dpi=150)
        plt.close(fig)


# ─── A12: Run summary ─────────────────────────────────────────────────
def analysis_run_summary(runs_by_key: dict[tuple[str, str], RunSource]) -> None:
    """One-row-per-(preset, model, run) catalog for the viewer."""
    rows = []
    for (preset, model), rs in sorted(runs_by_key.items()):
        all_rows = load_run(rs)
        n_total = len(all_rows)
        n_ok = sum(1 for r in all_rows if not r["_is_error"])
        rows.append([
            model, preset, rs.run_id, n_total, n_ok,
            "archive" if rs.is_archive else "live",
            str(rs.tsv.relative_to(REPO)),
        ])
    write_tsv(
        TABLES / "T0_run_summary.tsv",
        ["model", "preset", "run_id", "n_total", "n_ok", "source", "tsv_path"],
        rows,
    )


# ──────────────────────────────────────────────────────────────────────
# Registry
# ──────────────────────────────────────────────────────────────────────
ANALYSES = [
    Analysis("run_summary",
             "Catalog of every run feeding the analyses",
             analysis_run_summary),
    Analysis("headline",
             "Q1: Headline SR/GF/CM/FA/abstain/MUE per preset with 95% CIs",
             analysis_headline),
    Analysis("per_variant",
             "Q5: Per-variant breakdown + A+C vs B+C symmetry test",
             analysis_per_variant),
    Analysis("length_effect",
             "Q2: Length-decile SR/CM/MUE + log-odds regression on log10(chars)",
             analysis_length_effect),
    Analysis("depth_effect",
             "Q3: Depth-decile SR + quadratic-fit U-shape test",
             analysis_depth_effect),
    Analysis("failure_asymmetry",
             "Q4: When the model fails C-bearing, A vs B vs NEITHER",
             analysis_failure_asymmetry),
    Analysis("vigilance_theater",
             "Q6: Knew-but-didn't-act and flagged-but-recommended counts",
             analysis_vigilance_theater),
    Analysis("per_scenario",
             "Q7: Per-scenario SR distribution + histogram",
             analysis_per_scenario),
    Analysis("per_domain",
             "Q8: Per-domain SR table across the 3 presets",
             analysis_per_domain),
    Analysis("length_depth_surface",
             "2D surface SR(length, depth) on canon_unified C-bearing rows",
             analysis_length_depth_surface),
    Analysis("per_variant_length",
             "Multi-line variant×length plot (sanity check)",
             analysis_per_variant_length),
    Analysis("cost",
             "Per-preset token totals + projected batch spend",
             analysis_cost_summary),
]


# ──────────────────────────────────────────────────────────────────────
# Driver
# ──────────────────────────────────────────────────────────────────────
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--list", action="store_true", help="list all analyses")
    p.add_argument("--only", action="append", default=[],
                   help="run only this analysis (may repeat)")
    p.add_argument("--skip", action="append", default=[], help="skip this analysis")
    args = p.parse_args()

    if args.list:
        print(f"{'name':<28}{'description'}")
        print("─" * 80)
        for a in ANALYSES:
            print(f"  {a.name:<26}{a.description}")
        return 0

    runs = discover_runs()
    chosen = latest_runs_per_preset(runs)
    print(f"Discovered {len(runs)} runs; using latest per (preset, model):")
    for (preset, model), rs in sorted(chosen.items()):
        print(f"  {preset:<20} {model:<35} {rs.run_id}  "
              f"({'archive' if rs.is_archive else 'live'})")

    selected = [a for a in ANALYSES
                if (not args.only or a.name in args.only)
                and a.name not in args.skip]
    print(f"\nRunning {len(selected)} analyses → {OUTPUT}")
    for a in selected:
        print(f"\n  ▸ {a.name}: {a.description}")
        try:
            a.func(chosen)
            print("    ok")
        except Exception as e:
            import traceback
            print(f"    FAIL: {type(e).__name__}: {e}")
            traceback.print_exc()
    print("\nOutputs:")
    for p in sorted(TABLES.glob("*.tsv")):
        print(f"  table: {p.relative_to(REPO)}")
    for p in sorted(TABLES.glob("*.tex")):
        print(f"  latex: {p.relative_to(REPO)}")
    for p in sorted(FIGURES.glob("*.png")):
        print(f"  figure: {p.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
