# LCVB — Design Notes for the Paper

This is framing material for the methods section. The code is what it is;
this document is about how to *describe* it.

---

## The mixer as a parameterized generative function

`pipeline/renderers/mixer.py` exposes a single function `mix()` that maps
a configuration point to a deterministic set of prompts. Every renderer
in this repository is a thin wrapper that supplies a specific config.

The configuration space is a 6–7-dimensional hypercube:

| Axis | Type | What it controls |
|---|---|---|
| `n_distractor_draws` | `int ≥ 0` | Number of full re-renders of the scenario set. `0` → no distractor path at all. |
| `n_distractors_per_prompt` | `int ≥ 1` | How many distractor conversations are merged end-to-end into each prompt's history. `1` → classic single-distractor; `≥ 2` → the stitched variant. |
| `n_placements` | `int ≥ 0` | How many evidence insertion points per `(item, draw, length)` cell. |
| `placement_mode` | `{fixed, uniform}` | Whether `placements_list` is supplied explicitly or stratified-sampled from `[0, 1]`. |
| `n_lengths` | `int ≥ 0` | How many char budgets per cell. |
| `lengths_named` / `lengths_list` | `dict[str,int]` / `list[int]` | The actual budgets. |
| `include_constraint_inline` | `bool` | Whether the constraint description is folded into the user message (ceiling-test condition) vs inserted into history. |
| `c_only` | `bool` | Whether to include the `A`- and `B`-only variants. |
| `merge_gap_days` | `int ≥ 0` | Inter-chat timestamp gap when `n_distractors_per_prompt ≥ 2`. |

Plus a fixed `seed` (`4232026`) and a scenario TSV. Given these, `mix()`
is a pure function: same config → same prompts, byte-for-byte.

### The conditions in the paper are points in this space

The five canonical conditions map cleanly onto specific config points:

| Condition | `n_d_draws` | `n_d_per_prompt` | `n_placements` | `n_lengths` | `placement_mode` | `include_constraint_inline` |
|---|---|---|---|---|---|---|
| **Direct / ceiling** (`canon_direct`) | 0 | 1 | 0 | 0 | — | `True` |
| **No-distractor / primary** (`canon_no_distractor`) | 0 | 1 | 0 | 0 | — | `False` |
| **Uniform sweep, short** (`canon_uniform_short`) | 1 | 3 | 1 | 1 | `uniform_stratified` | `False` |
| **Uniform sweep, medium** (`canon_uniform_medium`) | 1 | 3 | 1 | 1 | `uniform_stratified` | `False` |
| **Uniform sweep, long** (`canon_uniform_long`) | 1 | 3 | 1 | 1 | `uniform_stratified` | `False` |

The three uniform sweeps use the same `placement_mode="uniform_stratified"`
(per-scenario stratified placements; every scenario covers `[0, 1]`
uniformly and has mean placement `0.5`) and differ only in
`lengths_named`: `{"short": 10_000}`, `{"medium": 100_000}`, and
`{"long": 250_000}` respectively. All use `merge_gap_days = 1` and
`c_only = True`.

A `canon_fixed_grid` condition (5 fixed depths × 3 haystacks) was part
of an earlier iteration of the canon and remains callable through
`mix_custom` with `placement_mode="fixed"` if needed for ablations,
but is no longer part of the headline canon.

Any new condition — ablations, rebuttal experiments, robustness checks —
is defined the same way: specify the point in the hypercube, call
`mix()`, commit the call site. Nothing is bespoke.

---

## Two layers of permutation

It is worth distinguishing these explicitly in the paper, because
"permutation" gets overloaded otherwise.

**Layer 1 — scenario-level seed permutations.** Each scenario in
`scenarios_FINAL.tsv` supplies up to three C-seeds (evidence sentences
supporting the safety constraint), three A-seeds (supporting option A),
and three B-seeds (supporting option B). The `A+C` variant enumerates
the Cartesian product of the relevant seed indices — `c0_a0`, `c0_a1`, …,
`c2_a2` — producing up to 9 permutations per `(scenario, variant)` pair.
After applying `c_only` and the validated filter, 85 scenarios × 3
C-present variants × ~7 average perms ≈ **1,646 items**. This layer is
*upstream* of the mixer and identical across every condition.

**Layer 2 — mixer-level permutations.** Given an `(scenario_id,
variant, scenario_perm)` item, the mixer decides:

- Which distractor hash(es) to pull from the 99-pool (per-draw,
  per-slot shuffle; balanced usage).
- At what normalized depth(s) to insert evidence (fixed list or
  stratified sample).
- Against which char budget(s) to truncate.

Every point in the config hypercube specifies how the mixer permutes
these — typically producing multiple prompts per item (e.g. the grid
sweep emits 5 depths × 3 haystacks = 15 prompts per item).

The two layers compose independently. The paper should refer to them by
distinct terms (we suggest "scenario permutations" vs "placement /
distractor configuration") to avoid confusion.

---

## Reproducibility claim

For any `(scenario_id, evidence_variant, scenario_perm, mix_config,
seed)`, the produced prompt is byte-identical on re-run. Concretely:

- Scenario loading is deterministic (TSV row order → sorted
  enumeration).
- Seed-permutation enumeration is deterministic (Cartesian product in
  fixed axis order).
- Distractor assignment is deterministic: per-slot
  `Random(seed + draw_idx + slot × stride).sample(pool, |pool|)`,
  then round-robin over items in sorted-triple order, with a
  reject-sample pass on same-item collisions.
- Stratified placements are derived from
  `sha256(scenario_id|variant|perm|draw_idx|length_idx|bin_idx)`.
- Timestamp shifts (for `n_distractors_per_prompt ≥ 2`) are computed
  from each distractor's own `min` / `max` timestamps — no wall-clock
  dependency.

No component touches a random generator that isn't seeded from these
inputs. This is verifiable by diffing two separate renders.

---

## What this design *does not* control

- **The distractor pool content.** The 99 distractor conversations in
  `data/distractors/` are fixed artifacts — synthesized upstream and
  deduplicated. The mixer picks from the pool but does not generate
  new chat content. Consequence: phrases like "Got it." or "I see."
  that appear organically inside distractors show up in rendered
  prompts, regardless of what `DEFAULT_ACKS` is set to. Evidence-ack
  text is the only dialogue the mixer itself authors.
- **The scoring semantics.** Rules for SR, General Flag, False Alarm,
  Choice Correct live in `SCORING.md` and `pipeline/eval_pipeline.py`'s
  judge prompt; they are orthogonal to prompt construction.
- **The judge.** Held fixed at Haiku 4.5 per project convention. A
  different judge is a different benchmark.

---

## Suggested phrasing for the methods section

> We define prompt generation as a deterministic function
> `mix(config, seed) → {prompts}` over a six-axis configuration space.
> The four reported conditions are specific points in this space (see
> Table N). Every prompt is fully specified by its `(scenario_id,
> evidence_variant, scenario_perm, mix_config, seed)` tuple; all code
> and artifacts needed to reproduce the reported prompt bytes are
> released at [repo URL].

The canonical config points are in `viewer/app.py`'s `RENDERERS`
registry under the `canon` flag, and as direct CLI invocations in the
`render_*.py` wrappers. A table in the appendix listing the exact
`mix()` argument values per reported row is probably worth including.
