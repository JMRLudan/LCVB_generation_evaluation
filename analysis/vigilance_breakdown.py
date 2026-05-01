#!/usr/bin/env python3
"""
Vigilance (SR) breakdown for LCVB canon run.
Presets: canon_direct, canon_no_distractor, canon_unified.
Outputs: markdown tables to analysis/vigilance_breakdown.md.
"""
from __future__ import annotations
import json
import os
import re
import sys
from io import StringIO
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE = Path(__file__).resolve().parent.parent  # LCVB_generation_evaluation/
DATA = BASE / "data"
RUNS = DATA / "runs"
GEN = BASE / "generated"
MODEL = "claude-haiku-4-5-20251001"
RID = os.environ.get("LCVB_RID", "20260424_182110")

PRESETS = ["canon_direct", "canon_no_distractor", "canon_unified"]
UNIFORM_PRESETS = ["canon_unified"]  # with-distractor; sliced for depth-curve / 2D-surface analyses.

OUT_MD = Path(__file__).resolve().parent / "vigilance_breakdown.md"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fmt_pct(x):
    if pd.isna(x):
        return "-"
    return f"{100*x:5.1f}%"


def agg(df, grouper):
    """Return aggregation table with SR, GF, FA, Abstention, n per level."""
    g = df.groupby(grouper, dropna=False)
    out = g.agg(
        n=("vigilance", "size"),
        SR=("vigilance", "mean"),
        GF=("general_flag", "mean"),
        FA=("false_alarm", "mean"),
        Abstained=("abstained", "mean"),
    ).reset_index()
    return out


def md_table(df, cols=None, floats=None):
    """Render a small DataFrame as a pipe-separated markdown table."""
    if cols is None:
        cols = list(df.columns)
    floats = set(floats or [])
    header = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"
    lines = [header, sep]
    for _, row in df[cols].iterrows():
        vals = []
        for c in cols:
            v = row[c]
            if c in floats:
                vals.append(fmt_pct(v))
            elif isinstance(v, float):
                if pd.isna(v):
                    vals.append("-")
                elif abs(v) < 1 and v != 0:
                    vals.append(f"{v:.3f}")
                else:
                    vals.append(f"{v:.1f}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Load scenarios
# ---------------------------------------------------------------------------
scn = pd.read_csv(DATA / "scenarios_FINAL.tsv", sep="\t")
# 86 rows; keep rows with check_personalization TRUE and status != reject
scn["check_personalization"] = scn["check_personalization"].astype(str).str.lower() == "true"
scn["status_clean"] = scn["status"].astype(str).str.lower().str.strip()
scn_valid = scn[scn["check_personalization"] & (scn["status_clean"] != "reject")].copy()

# Derive a short domain label (before the em dash)
def short_domain(d):
    if pd.isna(d):
        return "Unknown"
    return str(d).split("—")[0].strip()

scn_valid["domain_short"] = scn_valid["domain"].apply(short_domain)

# Risk level: strip the descriptor after " — "
def risk_tier(r):
    if pd.isna(r):
        return "Unknown"
    s = str(r).strip()
    # values like "1 — Direct risk, High probability" → we want just the leading tier number label
    # but semantically the user wants low/medium/high/catastrophic buckets.
    # The TSV uses a numeric-first encoding; keep leading token.
    return s.split("—")[0].strip() or s

scn_valid["risk_tier"] = scn_valid["risk_level"].apply(risk_tier)

scn_meta = scn_valid[["id", "domain_short", "risk_tier", "constraint_description"]].rename(
    columns={"id": "scenario_id"}
)

# ---------------------------------------------------------------------------
# Load results
# ---------------------------------------------------------------------------
frames = []
excluded_counts = {}
dup_stats = {}
for preset in PRESETS:
    path = RUNS / preset / MODEL / RID / "results.tsv"
    df = pd.read_csv(path, sep="\t")
    df["preset"] = preset

    # Dedup by (scenario_id, evidence_variant, permutation); prefer parse_error=0
    before = len(df)
    df = df.sort_values("parse_error").drop_duplicates(
        subset=["scenario_id", "evidence_variant", "permutation"], keep="first"
    )
    dup_stats[preset] = before - len(df)

    # Count parse errors
    n_pe = int(df["parse_error"].fillna(0).astype(int).sum())
    excluded_counts[preset] = n_pe

    # Exclude parse errors from rate computations
    df = df[df["parse_error"].fillna(0).astype(int) == 0].copy()
    frames.append(df)

res = pd.concat(frames, ignore_index=True)
# Join scenario metadata
res = res.merge(scn_meta, on="scenario_id", how="left")

# Cast rate fields to numeric (0/1) just in case
for col in ["vigilance", "general_flag", "false_alarm", "abstained", "choice_correct"]:
    if col in res.columns:
        res[col] = pd.to_numeric(res[col], errors="coerce").fillna(0).astype(int)

# ---------------------------------------------------------------------------
# Load uniform-preset prompt metadata (placement_frac, char_budget, etc.)
# ---------------------------------------------------------------------------
def load_prompt_meta(preset):
    rows = []
    pdir = GEN / preset
    if not pdir.exists():
        return pd.DataFrame()
    for fp in pdir.glob("*.json"):
        try:
            with open(fp) as f:
                j = json.load(f)
        except Exception:
            continue
        m = j.get("metadata", {})
        rows.append({
            "preset": preset,
            "scenario_id": m.get("scenario_id"),
            "evidence_variant": m.get("evidence_variant"),
            "permutation": m.get("permutation"),
            "placement_frac": m.get("placement_frac"),
            "char_budget": m.get("char_budget"),
            "input_char_len": m.get("input_char_len"),
            "n_distractor_pairs": m.get("n_distractor_pairs"),
        })
    return pd.DataFrame(rows)

prompt_meta_frames = []
for p in UNIFORM_PRESETS:
    prompt_meta_frames.append(load_prompt_meta(p))
prompt_meta = pd.concat(prompt_meta_frames, ignore_index=True) if prompt_meta_frames else pd.DataFrame()

# Join onto results only for uniform presets
res_uniform = res[res["preset"].isin(UNIFORM_PRESETS)].merge(
    prompt_meta,
    on=["preset", "scenario_id", "evidence_variant", "permutation"],
    how="left",
)

# ---------------------------------------------------------------------------
# Preset summary + parse-error caveat
# ---------------------------------------------------------------------------
preset_rows = []
for p in PRESETS:
    sub = res[res["preset"] == p]
    preset_rows.append({
        "preset": p,
        "n_scored": len(sub),
        "excluded_parse_errors": excluded_counts[p],
        "duplicates_removed": dup_stats[p],
        "SR": sub["vigilance"].mean() if len(sub) else float("nan"),
        "GF": sub["general_flag"].mean() if len(sub) else float("nan"),
        "FA": sub["false_alarm"].mean() if len(sub) else float("nan"),
        "Abstained": sub["abstained"].mean() if len(sub) else float("nan"),
    })
preset_summary = pd.DataFrame(preset_rows)

# ---------------------------------------------------------------------------
# (1) By domain
# ---------------------------------------------------------------------------
dom_tab = (
    res.groupby(["domain_short", "preset"])
    ["vigilance"].agg(["mean", "size"]).unstack("preset")
)
# Multi-level columns — flatten
dom_flat = pd.DataFrame(index=dom_tab.index)
for p in PRESETS:
    if ("mean", p) in dom_tab.columns:
        dom_flat[f"SR_{p}"] = dom_tab[("mean", p)]
        dom_flat[f"n_{p}"] = dom_tab[("size", p)]
dom_flat = dom_flat.reset_index()

# SR drop direct → with-distractor per domain.
ceil_col = "SR_canon_direct"
distractor_col = "SR_canon_unified"
n_ceil_high = int((dom_flat[ceil_col] > 0.80).sum()) if ceil_col in dom_flat else 0
n_long_low = int((dom_flat[distractor_col] < 0.20).sum()) if distractor_col in dom_flat else 0
if ceil_col in dom_flat and distractor_col in dom_flat:
    dom_flat["drop_direct_to_distractor"] = dom_flat[ceil_col] - dom_flat[distractor_col]
    sort_col = "drop_direct_to_distractor"
else:
    sort_col = ceil_col if ceil_col in dom_flat else dom_flat.columns[0]
dom_flat_sorted = dom_flat.sort_values(sort_col, ascending=False)

# ---------------------------------------------------------------------------
# (2) By risk tier
# ---------------------------------------------------------------------------
risk_tab = (
    res.groupby(["risk_tier", "preset"])["vigilance"]
    .agg(["mean", "size"]).unstack("preset")
)
risk_flat = pd.DataFrame(index=risk_tab.index)
for p in PRESETS:
    if ("mean", p) in risk_tab.columns:
        risk_flat[f"SR_{p}"] = risk_tab[("mean", p)]
        risk_flat[f"n_{p}"] = risk_tab[("size", p)]
risk_flat = risk_flat.reset_index()

# ---------------------------------------------------------------------------
# (3) By evidence_variant
# ---------------------------------------------------------------------------
ev_tab = (
    res.groupby(["evidence_variant", "preset"])["vigilance"]
    .agg(["mean", "size"]).unstack("preset")
)
ev_flat = pd.DataFrame(index=ev_tab.index)
for p in PRESETS:
    if ("mean", p) in ev_tab.columns:
        ev_flat[f"SR_{p}"] = ev_tab[("mean", p)]
        ev_flat[f"n_{p}"] = ev_tab[("size", p)]
ev_flat = ev_flat.reset_index()

# ---------------------------------------------------------------------------
# (4) By placement_frac (uniform only)
# ---------------------------------------------------------------------------
bins = [0.0, 0.25, 0.5, 0.75, 1.0001]
labels = ["[0.00,0.25)", "[0.25,0.50)", "[0.50,0.75)", "[0.75,1.00]"]
res_uniform = res_uniform.copy()
res_uniform["placement_bin"] = pd.cut(
    res_uniform["placement_frac"], bins=bins, labels=labels, include_lowest=True, right=False
)
plc_tab = (
    res_uniform.groupby(["placement_bin", "preset"], observed=False)["vigilance"]
    .agg(["mean", "size"]).unstack("preset")
)
plc_flat = pd.DataFrame(index=plc_tab.index)
for p in UNIFORM_PRESETS:
    if ("mean", p) in plc_tab.columns:
        plc_flat[f"SR_{p}"] = plc_tab[("mean", p)]
        plc_flat[f"n_{p}"] = plc_tab[("size", p)]
plc_flat = plc_flat.reset_index()

# ---------------------------------------------------------------------------
# (5) By input_tokens quartile (within each uniform preset)
# ---------------------------------------------------------------------------
quart_rows = []
for p in UNIFORM_PRESETS:
    sub = res_uniform[res_uniform["preset"] == p].copy()
    sub = sub.dropna(subset=["input_tokens"])
    if len(sub) == 0:
        continue
    try:
        sub["tok_q"] = pd.qcut(sub["input_tokens"], q=4, labels=["Q1", "Q2", "Q3", "Q4"])
    except Exception:
        sub["tok_q"] = pd.cut(sub["input_tokens"], bins=4, labels=["Q1", "Q2", "Q3", "Q4"])
    for q in ["Q1", "Q2", "Q3", "Q4"]:
        s = sub[sub["tok_q"] == q]
        quart_rows.append({
            "preset": p,
            "tok_quartile": q,
            "n": len(s),
            "tok_min": int(s["input_tokens"].min()) if len(s) else None,
            "tok_max": int(s["input_tokens"].max()) if len(s) else None,
            "SR": s["vigilance"].mean() if len(s) else float("nan"),
        })
tok_quart = pd.DataFrame(quart_rows)

# ---------------------------------------------------------------------------
# (6) Per-scenario SR across presets
# ---------------------------------------------------------------------------
scen_tab = (
    res.groupby(["scenario_id", "preset"])["vigilance"]
    .mean().unstack("preset").reset_index()
)
for p in PRESETS:
    if p not in scen_tab.columns:
        scen_tab[p] = float("nan")

scen_tab = scen_tab.merge(scn_meta[["scenario_id", "domain_short", "risk_tier"]], on="scenario_id", how="left")
scen_tab["min_SR"] = scen_tab[PRESETS].min(axis=1)
scen_tab["max_SR"] = scen_tab[PRESETS].max(axis=1)
scen_tab["SR_range"] = scen_tab["max_SR"] - scen_tab["min_SR"]
scen_tab["SR_mean"] = scen_tab[PRESETS].mean(axis=1)

robust = scen_tab[scen_tab["min_SR"] >= 0.80].sort_values("min_SR", ascending=False)
brittle = scen_tab[
    (scen_tab.get("canon_direct", 0) >= 0.66)
    & (scen_tab.get("canon_uniform_long", 0) <= 0.10)
].sort_values("SR_range", ascending=False)
universal_miss = scen_tab[scen_tab["max_SR"] <= 0.10].sort_values("max_SR")

# ---------------------------------------------------------------------------
# (7) Correlations / marginal effects
# ---------------------------------------------------------------------------
# Per-item-level Pearson correlations
corr_rows = []
for p in PRESETS:
    sub = res[res["preset"] == p]
    if len(sub) < 5:
        continue
    corr_rows.append({
        "preset": p,
        "n": len(sub),
        "corr_SR_abstained": sub["vigilance"].corr(sub["abstained"]),
        "corr_SR_generalflag": sub["vigilance"].corr(sub["general_flag"]),
        "corr_SR_falsealarm": sub["vigilance"].corr(sub["false_alarm"]),
        "corr_GF_abstained": sub["general_flag"].corr(sub["abstained"]),
    })
corr_tab = pd.DataFrame(corr_rows)

# Per-scenario: does GF rising predict SR falling?
scen_gf = res.groupby(["scenario_id", "preset"])["general_flag"].mean().unstack("preset")
scen_sr = res.groupby(["scenario_id", "preset"])["vigilance"].mean().unstack("preset")
sc_corr_rows = []
for p in PRESETS:
    if p in scen_gf.columns and p in scen_sr.columns:
        x = scen_gf[p]
        y = scen_sr[p]
        mask = x.notna() & y.notna()
        if mask.sum() >= 5:
            sc_corr_rows.append({
                "preset": p,
                "n_scenarios": int(mask.sum()),
                "scn_corr_GF_SR": x[mask].corr(y[mask]),
            })
scen_corr_tab = pd.DataFrame(sc_corr_rows)

# Also: within-item delta — when SR=0 (miss), does GF tend to be 1?
gf_given_sr_rows = []
for p in PRESETS:
    sub = res[res["preset"] == p]
    if len(sub) == 0:
        continue
    sr0 = sub[sub["vigilance"] == 0]
    sr1 = sub[sub["vigilance"] == 1]
    gf_given_sr_rows.append({
        "preset": p,
        "P(GF=1 | SR=0)": sr0["general_flag"].mean() if len(sr0) else float("nan"),
        "P(GF=1 | SR=1)": sr1["general_flag"].mean() if len(sr1) else float("nan"),
        "P(abst | SR=0)": sr0["abstained"].mean() if len(sr0) else float("nan"),
        "P(abst | SR=1)": sr1["abstained"].mean() if len(sr1) else float("nan"),
    })
gf_given_sr = pd.DataFrame(gf_given_sr_rows)


# ---------------------------------------------------------------------------
# Render markdown
# ---------------------------------------------------------------------------
buf = StringIO()
w = buf.write

w(f"# LCVB canon run — vigilance breakdown (Haiku 4.5)\n\n")
w(f"- Run ID: `{RID}`\n")
w(f"- Model: `{MODEL}`\n")
w(f"- Presets: {', '.join(PRESETS)}\n")
w(f"- Scenarios used: {len(scn_meta)} (check_personalization=TRUE, status!=reject)\n\n")

w("## Preset summary\n\n")
cols = ["preset", "n_scored", "excluded_parse_errors", "duplicates_removed", "SR", "GF", "FA", "Abstained"]
w(md_table(preset_summary, cols=cols, floats={"SR", "GF", "FA", "Abstained"}))
w("\n\n")

w("## 1. Vigilance by domain\n\n")
w("Columns are SR (and n) per preset.\n\n")
# Build compact columns: SR per preset + n_direct
compact_cols = ["domain_short"] + [f"SR_{p}" for p in PRESETS] + ["drop_direct_to_long"]
compact_cols = [c for c in compact_cols if c in dom_flat_sorted.columns]
float_cols = {c for c in compact_cols if c.startswith("SR_") or c == "drop_direct_to_long"}
w(md_table(dom_flat_sorted, cols=compact_cols, floats=float_cols))
w("\n\n")
# Full table with n
full_cols = ["domain_short"] + [c for p in PRESETS for c in (f"SR_{p}", f"n_{p}") if c in dom_flat.columns]
w("### Domain table with n per preset\n\n")
w(md_table(dom_flat.sort_values("domain_short"), cols=full_cols,
           floats={c for c in full_cols if c.startswith("SR_")}))
w("\n\n")
w(f"- Domains with SR > 80% on `{ceiling_preset}`: **{n_ceil_high} / {len(dom_flat)}**\n")
w(f"- Domains with SR < 20% on `{long_preset}`: **{n_long_low} / {len(dom_flat)}**\n\n")

w("## 2. Vigilance by risk tier\n\n")
rt_cols = ["risk_tier"] + [c for p in PRESETS for c in (f"SR_{p}", f"n_{p}") if c in risk_flat.columns]
w(md_table(risk_flat, cols=rt_cols, floats={c for c in rt_cols if c.startswith("SR_")}))
w("\n\n")

w("## 3. Vigilance by evidence_variant\n\n")
w("`C` = constraint alone; `A+C` / `B+C` add counter-preference seeds toward the dangerous option.\n\n")
ev_cols = ["evidence_variant"] + [c for p in PRESETS for c in (f"SR_{p}", f"n_{p}") if c in ev_flat.columns]
w(md_table(ev_flat, cols=ev_cols, floats={c for c in ev_cols if c.startswith("SR_")}))
w("\n\n")

w("## 4. Vigilance by placement_frac bin (uniform presets only)\n\n")
plc_cols = ["placement_bin"] + [c for p in UNIFORM_PRESETS for c in (f"SR_{p}", f"n_{p}") if c in plc_flat.columns]
w(md_table(plc_flat, cols=plc_cols, floats={c for c in plc_cols if c.startswith("SR_")}))
w("\n\n")

w("## 5. Vigilance by input-tokens quartile (within each uniform preset)\n\n")
w(md_table(tok_quart, cols=["preset", "tok_quartile", "n", "tok_min", "tok_max", "SR"],
           floats={"SR"}))
w("\n\n")

w("## 6. Per-scenario SR across presets\n\n")
w(f"- Scenarios scored (any preset): {len(scen_tab)}\n")
w(f"- Scenarios robust (min SR >= 0.80 across all 5 presets): **{len(robust)}**\n")
w(f"- Scenarios brittle (direct >= 0.66 but uniform_long <= 0.10): **{len(brittle)}**\n")
w(f"- Scenarios universal miss (max SR <= 0.10 in every preset): **{len(universal_miss)}**\n\n")

if len(robust):
    w("### Robust scenarios\n\n")
    w(md_table(robust.head(25), cols=["scenario_id", "domain_short", "risk_tier"] + PRESETS + ["min_SR"],
               floats=set(PRESETS) | {"min_SR"}))
    w("\n\n")
if len(brittle):
    w("### Brittle scenarios (sharp transition — most publishable cases)\n\n")
    w(md_table(brittle.head(30), cols=["scenario_id", "domain_short", "risk_tier"] + PRESETS + ["SR_range"],
               floats=set(PRESETS) | {"SR_range"}))
    w("\n\n")
if len(universal_miss):
    w("### Universal-miss scenarios\n\n")
    w(md_table(universal_miss, cols=["scenario_id", "domain_short", "risk_tier"] + PRESETS + ["max_SR"],
               floats=set(PRESETS) | {"max_SR"}))
    w("\n\n")

w("## 7. Correlations / marginal effects\n\n")
w("### Item-level Pearson correlations\n\n")
cols = ["preset", "n", "corr_SR_abstained", "corr_SR_generalflag", "corr_SR_falsealarm", "corr_GF_abstained"]
w(md_table(corr_tab, cols=cols))
w("\n\n")

w("### Scenario-level correlation of GF and SR (per preset)\n\n")
w(md_table(scen_corr_tab, cols=["preset", "n_scenarios", "scn_corr_GF_SR"]))
w("\n\n")

w("### Conditional probabilities\n\n")
w(md_table(gf_given_sr,
           cols=["preset", "P(GF=1 | SR=0)", "P(GF=1 | SR=1)", "P(abst | SR=0)", "P(abst | SR=1)"],
           floats={"P(GF=1 | SR=0)", "P(GF=1 | SR=1)", "P(abst | SR=0)", "P(abst | SR=1)"}))
w("\n\n")

# ---------------------------------------------------------------------------
# Headline patterns
# ---------------------------------------------------------------------------
w("## Headline patterns\n\n")

# Compute a few key numbers for the write-up
direct_sr = preset_summary.set_index("preset").loc["canon_direct", "SR"]
nd_sr = preset_summary.set_index("preset").loc["canon_no_distractor", "SR"]
s_sr = preset_summary.set_index("preset").loc["canon_uniform_short", "SR"]
m_sr = preset_summary.set_index("preset").loc["canon_uniform_medium", "SR"]
l_sr = preset_summary.set_index("preset").loc["canon_uniform_long", "SR"]

w(f"- **Context length is the dominant axis.** Overall SR collapses from {fmt_pct(direct_sr)} (direct) "
  f"and {fmt_pct(nd_sr)} (no-distractor) to {fmt_pct(s_sr)} / {fmt_pct(m_sr)} / {fmt_pct(l_sr)} on "
  f"uniform short/medium/long — a monotonic decay with increasing distractor context.\n")

# Evidence variant headline: look at uniform_long row
ev_long = ev_flat.set_index("evidence_variant")
if "SR_canon_uniform_long" in ev_long.columns:
    try:
        c_long = ev_long.loc["C", "SR_canon_uniform_long"]
        a_long = ev_long.loc["A+C", "SR_canon_uniform_long"]
        b_long = ev_long.loc["B+C", "SR_canon_uniform_long"]
        # Note: direction is opposite the pre-registered hypothesis — A+C/B+C actually HELP.
        w(f"- **Evidence-variant effect runs opposite to the pre-registered hypothesis.** On "
          f"`canon_uniform_long`, SR is *lowest* for C-only at {fmt_pct(c_long)}, vs "
          f"{fmt_pct(a_long)} (A+C) / {fmt_pct(b_long)} (B+C). The extra A- or B-seeded turns "
          f"appear to scaffold the constraint (more related-topic surface area), not distract from "
          f"it. Worth re-examining whether A+/B+ seeds inadvertently cue the constraint domain.\n")
    except KeyError:
        pass

# Placement: compare first vs last bin on long
if "SR_canon_uniform_long" in plc_flat.columns:
    try:
        plc_ix = plc_flat.set_index("placement_bin")
        first_bin = plc_ix.iloc[0]["SR_canon_uniform_long"]
        last_bin = plc_ix.iloc[-1]["SR_canon_uniform_long"]
        w(f"- **Recency dominates on long context.** Placement-bin SR on uniform_long: "
          f"{fmt_pct(first_bin)} at [0.00,0.25) vs {fmt_pct(last_bin)} at [0.75,1.00]. Constraints "
          f"near the end of the history are roughly {last_bin/first_bin:.1f}x more likely to be caught "
          f"than ones near the beginning.\n")
    except Exception:
        pass

# Brittleness stat
w(f"- **{len(brittle)} of {len(scen_tab)} scenarios** show a sharp transition (high SR on direct, near-zero on long) "
  f"— these are the most informative items for the paper; {len(robust)} are robust across all 5 conditions and "
  f"{len(universal_miss)} are essentially unsolved anywhere.\n")

# Abstention headline
if not gf_given_sr.empty:
    # average across presets
    mean_abst_given_sr1 = gf_given_sr["P(abst | SR=1)"].mean()
    mean_abst_given_sr0 = gf_given_sr["P(abst | SR=0)"].mean()
    w(f"- **Abstention is diagnostic of vigilance.** Averaged across presets, P(abstain | SR=1) ≈ "
      f"{fmt_pct(mean_abst_given_sr1)} vs P(abstain | SR=0) ≈ {fmt_pct(mean_abst_given_sr0)}; "
      f"heavy-modify/abstain is the primary expression of a caught constraint.\n")

# Parse error note
tot_pe = sum(excluded_counts.values())
w(f"\n*Caveats: excluded {tot_pe} parse-error rows across the 5 presets "
  f"({', '.join(f'{k}={v}' for k,v in excluded_counts.items())}); "
  f"{sum(dup_stats.values())} duplicate row(s) removed (keeping non-parse-error).*\n")

# ---------------------------------------------------------------------------
# Write output
# ---------------------------------------------------------------------------
OUT_MD.parent.mkdir(parents=True, exist_ok=True)
OUT_MD.write_text(buf.getvalue())
print(f"Wrote {OUT_MD}")
print(f"Total scored rows: {len(res)}")
print(f"Parse errors excluded: {excluded_counts}")
print(f"Duplicates removed: {dup_stats}")
