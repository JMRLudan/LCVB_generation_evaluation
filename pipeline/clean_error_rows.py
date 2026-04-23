"""
Remove error rows + dedupe results.tsv for sweep_d runs so `--resume`
retries only missing items.

For each condition dir under data/runs_sweep_d/<model>/<run_id>/<cond>:
  1. Read all rows in results.tsv.
  2. Group by (scenario_id, evidence_variant, permutation).
  3. For each group: keep one row, preferring non-ERROR rows.
     If only ERROR rows exist, drop the key entirely.
  4. Rewrite results.tsv with kept rows only.
  5. Rewrite checkpoint.txt as PIPE-delimited keys (matches runner's
     item_key() format in multi_model_runner.py).
  6. Back up both files + save dropped errors to results.tsv.errors_<ts>.

IMPORTANT: checkpoint key format is 'scenario_id|evidence_variant|permutation'
(pipe). An earlier version used '/' which did NOT match the runner's format
and caused entire conditions to be re-processed on resume.

Usage:
  python3 clean_error_rows.py --model <model_slug_fs> --run-id <run_id>
  python3 clean_error_rows.py --model <model_slug_fs> --run-id <run_id> --dry-run

Multiple models can be cleaned at once via --model repeated.
"""

import argparse
import csv
import shutil
import sys
from datetime import datetime
from pathlib import Path

csv.field_size_limit(sys.maxsize)

RUNS_DIR = Path(__file__).parent.parent / "data" / "runs_sweep_d"


def _key(row: dict) -> str:
    """Pipe-delimited key — matches multi_model_runner.item_key()."""
    return f"{row['scenario_id']}|{row['evidence_variant']}|{row['permutation']}"


def clean_condition(cond_dir: Path, dry_run: bool = False) -> tuple[int, int, int]:
    """Return (n_kept_good, n_errors_dropped, n_duplicates_dropped)."""
    results_path = cond_dir / "results.tsv"
    checkpoint_path = cond_dir / "checkpoint.txt"
    if not results_path.exists():
        return (0, 0, 0)

    with open(results_path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    # Group by key; for each key keep one: prefer non-ERROR (last-seen wins to
    # pick most recent successful retry).
    by_key: dict[str, dict] = {}
    errors_dropped = 0
    duplicates_dropped = 0
    for row in rows:
        k = _key(row)
        is_err = (row.get("raw_response") or "").startswith("ERROR:")
        if k not in by_key:
            by_key[k] = row
            continue
        # Key already seen — decide replace vs drop
        prev = by_key[k]
        prev_err = (prev.get("raw_response") or "").startswith("ERROR:")
        if prev_err and not is_err:
            by_key[k] = row  # upgrade from error to good
            errors_dropped += 1
        elif not prev_err and not is_err:
            duplicates_dropped += 1  # keep first good; drop this dup
        elif not prev_err and is_err:
            errors_dropped += 1  # keep first good; drop later error
        else:
            duplicates_dropped += 1  # both errors; drop this one

    # Drop entries that are still errors (no good row ever seen)
    kept: list[dict] = []
    dropped_errors_only: list[dict] = []
    for k, row in by_key.items():
        if (row.get("raw_response") or "").startswith("ERROR:"):
            dropped_errors_only.append(row)
            errors_dropped += 1
        else:
            kept.append(row)

    total_original = len(rows)
    n_good = len(kept)
    total_errors_or_dups = total_original - n_good
    # breakdown helpful but not needed for return
    # (errors_dropped + duplicates_dropped should equal total_errors_or_dups)

    if dry_run:
        return (n_good, errors_dropped, duplicates_dropped)

    if total_errors_or_dups == 0 and len(rows) > 0:
        # Still ensure checkpoint.txt is correctly pipe-delimited, because prior
        # buggy cleanup may have written slash-delimited keys even for clean data.
        keys = {_key(r) for r in kept}
        with open(checkpoint_path, "w") as f:
            for k in sorted(keys):
                f.write(k + "\n")
        return (n_good, 0, 0)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    if dropped_errors_only:
        errors_path = cond_dir / f"results.tsv.errors_{ts}"
        with open(errors_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
            writer.writeheader()
            writer.writerows(dropped_errors_only)

    shutil.copy2(results_path, cond_dir / f"results.tsv.bak_{ts}")
    if checkpoint_path.exists():
        shutil.copy2(checkpoint_path, cond_dir / f"checkpoint.txt.bak_{ts}")

    with open(results_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(kept)

    keys = {_key(r) for r in kept}
    with open(checkpoint_path, "w") as f:
        for k in sorted(keys):
            f.write(k + "\n")

    return (n_good, errors_dropped, duplicates_dropped)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        help="Model dir name under data/runs_sweep_d/ (e.g. 'google_gemini-2.5-flash'). Can repeat.",
    )
    parser.add_argument(
        "--run-id",
        required=True,
        help="Run id directory name (shared across models)."
        " If the exact id does not exist for a given model, the only run_dir under that model is used.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    grand_good = 0
    grand_bad = 0
    for model in args.model:
        model_dir = RUNS_DIR / model
        if not model_dir.exists():
            print(f"[SKIP] {model}: no dir at {model_dir}")
            continue
        run_dir = model_dir / args.run_id
        if not run_dir.exists():
            runs = sorted([p for p in model_dir.iterdir() if p.is_dir()])
            if len(runs) != 1:
                print(f"[SKIP] {model}: run_id {args.run_id} missing; {len(runs)} runs found, need explicit run-id")
                continue
            run_dir = runs[0]
            print(f"[INFO] {model}: using only run {run_dir.name}")

        model_good = 0
        model_errs = 0
        model_dups = 0
        for cond_dir in sorted(p for p in run_dir.iterdir() if p.is_dir()):
            good, errs, dups = clean_condition(cond_dir, dry_run=args.dry_run)
            model_good += good
            model_errs += errs
            model_dups += dups
            if errs > 0 or dups > 0:
                tag = "[DRY]" if args.dry_run else "[CLEAN]"
                print(f"  {tag} {model}/{cond_dir.name}: kept {good} good, dropped {errs} errors, {dups} dup good-rows")

        print(f"{model} TOTAL: kept {model_good} good, dropped {model_errs} errors, {model_dups} dup good-rows")
        grand_good += model_good
        grand_bad += model_errs + model_dups

    print()
    print(f"GRAND TOTAL: kept {grand_good} good rows, dropped {grand_bad} rows (errors + dups)")
    if args.dry_run:
        print("(dry run — no files modified)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
