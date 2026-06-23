#!/usr/bin/env python3
"""Analyze predictions_after.json from a repetition-only training run.

Produces a per-word CSV and a Markdown report breaking down the gap between
exact argmax accuracy and paper-like all-units-within-radius accuracy.

No model imports required — reads only JSON/CSV files from the run directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

CSV_COLUMNS = [
    "item_index",
    "word_or_label",
    "T",
    "argmax_exact_match",
    "paper_like_match",
    "strict_trial_match",
    "input_phase_all_silent",
    "output_phase_all_within_radius",
    "n_output_ticks",
    "n_argmax_errors",
    "n_positive_failures",
    "n_negative_failures",
    "n_input_silence_failures",
    "min_positive_activation",
    "mean_positive_activation",
    "max_negative_activation",
    "mean_negative_activation",
    "mean_argmax_margin",
    "min_argmax_margin",
    "failure_type",
]

# Failure types in display order
_FAILURE_TYPE_ORDER = [
    "paper_like_success",
    "positive_under_threshold_only",
    "negative_over_threshold_only",
    "positive_and_negative_failures",
    "input_silence_failure_only",
    "mixed_failure",
    "argmax_error",
]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_predictions(run_dir: Path) -> list[dict]:
    path = run_dir / "predictions_after.json"
    if not path.exists():
        print(f"ERROR: predictions_after.json not found at {path}", file=sys.stderr)
        sys.exit(1)
    with open(path) as f:
        return json.load(f)


def load_run_config(run_dir: Path) -> dict:
    path = run_dir / "run_config.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def load_final_metrics(run_dir: Path) -> dict | None:
    path = run_dir / "metrics.csv"
    if not path.exists():
        return None
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return rows[-1] if rows else None


# ---------------------------------------------------------------------------
# Core per-word analysis
# ---------------------------------------------------------------------------


def classify_failure_type(p: dict) -> str:
    """Classify why a word fails the paper-like criterion (exact for eval_radius=0.1).

    The stored threshold accuracies use thresholds 0.9 (positive) and 0.1 (negative),
    which are exactly 1-radius and radius for eval_radius=0.1. So the classification
    is exact up to floating-point rounding, not an approximation.
    """
    if p["output_all_units_within_radius"]:
        return "paper_like_success"
    if not p["exact_match"]:
        return "argmax_error"
    pos_ok = abs(p["output_positive_threshold_acc"] - 1.0) < 1e-9
    neg_ok = abs(p["output_negative_threshold_acc"] - 1.0) < 1e-9
    input_ok = p["input_all_silent_within_radius"]
    if pos_ok and neg_ok and not input_ok:
        return "input_silence_failure_only"
    if not pos_ok and neg_ok and input_ok:
        return "positive_under_threshold_only"
    if pos_ok and not neg_ok and input_ok:
        return "negative_over_threshold_only"
    if not pos_ok and not neg_ok:
        return "positive_and_negative_failures"
    return "mixed_failure"


def analyze_one(p: dict, index: int, motor_size: int = 39,
                sound_input_size: int = 39) -> dict:
    """Return one CSV row for a single word prediction entry.

    Counts are reconstructed from stored threshold accuracies and are exact up to
    floating-point rounding, because the accuracies were computed from discrete
    threshold counts.
    """
    T = p["T"]
    n_neg_per_tick = motor_size - 1

    n_argmax_errors = round((1.0 - p["phoneme_accuracy"]) * T)
    n_positive_failures = round((1.0 - p["output_positive_threshold_acc"]) * T)
    n_negative_failures = round((1.0 - p["output_negative_threshold_acc"]) * T * n_neg_per_tick)
    n_input_silence_failures = round((1.0 - p["input_silence_threshold_acc"]) * T * sound_input_size)

    return {
        "item_index": index,
        "word_or_label": p["label"],
        "T": T,
        "argmax_exact_match": p["exact_match"],
        "paper_like_match": p["output_all_units_within_radius"],
        "strict_trial_match": p["trial_all_supervised_units_within_radius"],
        "input_phase_all_silent": p["input_all_silent_within_radius"],
        "output_phase_all_within_radius": p["output_all_units_within_radius"],
        "n_output_ticks": T,
        "n_argmax_errors": n_argmax_errors,
        "n_positive_failures": n_positive_failures,
        "n_negative_failures": n_negative_failures,
        "n_input_silence_failures": n_input_silence_failures,
        "min_positive_activation": "N/A",
        "mean_positive_activation": round(p["mean_positive_output"], 4),
        "max_negative_activation": "N/A",
        "mean_negative_activation": round(p["mean_negative_output"], 4),
        "mean_argmax_margin": "N/A",
        "min_argmax_margin": "N/A",
        "failure_type": classify_failure_type(p),
    }


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------


def _pct(n: int, total: int) -> str:
    return f"{100.0 * n / total:.1f}%" if total > 0 else "0.0%"


def _fmean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else float("nan")


def generate_markdown(
    rows: list[dict],
    predictions: list[dict],
    run_config: dict,
    run_dir: Path,
    eval_radius: float,
    csv_path: Path,
    md_path: Path,
) -> str:
    n = len(rows)
    motor_size = run_config.get("motor_output_size", 39)
    n_neg_per_tick = motor_size - 1

    n_exact = sum(1 for r in rows if r["argmax_exact_match"])
    n_paper = sum(1 for r in rows if r["paper_like_match"])
    n_strict = sum(1 for r in rows if r["strict_trial_match"])
    n_exact_not_paper = sum(
        1 for r in rows if r["argmax_exact_match"] and not r["paper_like_match"]
    )
    n_argmax_fail = n - n_exact

    type_counts: Counter[str] = Counter(r["failure_type"] for r in rows)

    n_words_any_pos_fail = sum(1 for r in rows if r["n_positive_failures"] > 0)
    n_words_any_neg_fail = sum(1 for r in rows if r["n_negative_failures"] > 0)
    n_words_both_fail = sum(
        1 for r in rows
        if r["n_positive_failures"] > 0 and r["n_negative_failures"] > 0
    )

    # Aggregate tick/unit rates (distinct from per-word rates)
    exact_rows = [r for r in rows if r["argmax_exact_match"]]
    total_pos_ticks = sum(r["T"] for r in exact_rows)
    total_pos_fail_ticks = sum(r["n_positive_failures"] for r in exact_rows)
    pct_pos_ticks = 100.0 * total_pos_fail_ticks / total_pos_ticks if total_pos_ticks else 0.0

    total_neg_pairs = sum(r["T"] * n_neg_per_tick for r in rows)
    total_neg_fail_pairs = sum(r["n_negative_failures"] for r in rows)
    pct_neg_pairs = 100.0 * total_neg_fail_pairs / total_neg_pairs if total_neg_pairs else 0.0

    by_T: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        by_T[r["T"]].append(r)

    mean_pos_acts = [r["mean_positive_activation"] for r in exact_rows
                     if isinstance(r["mean_positive_activation"], float)]
    mean_neg_acts = [r["mean_negative_activation"] for r in rows
                     if isinstance(r["mean_negative_activation"], float)]

    paper_rows = [r for r in rows if r["paper_like_match"]]
    pos_under_rows = sorted(
        [r for r in rows if r["failure_type"] == "positive_under_threshold_only"],
        key=lambda r: r["n_positive_failures"],
    )[:5]
    neg_over_rows = sorted(
        [r for r in rows if r["failure_type"] == "negative_over_threshold_only"],
        key=lambda r: r["n_negative_failures"],
    )[:5]
    argmax_fail_rows = [r for r in rows if not r["argmax_exact_match"]]
    near_miss_rows = sorted(
        [r for r in rows if r["argmax_exact_match"] and not r["paper_like_match"]],
        key=lambda r: r["n_positive_failures"] + r["n_negative_failures"],
    )[:5]

    dominant = type_counts.most_common(1)[0][0] if type_counts else "unknown"
    n_input_fail_words = sum(1 for r in rows if not r["input_phase_all_silent"])

    lines: list[str] = []
    a = lines.append

    # Header
    a("# Repetition Prediction Error Analysis")
    a("")
    a(f"Run directory: `{run_dir}`  ")
    a(f"CSV output: `{csv_path}`  ")
    a(f"Eval radius: `{eval_radius}`")
    a("")
    a("> **paper_like_match** = `output_all_units_within_radius`: ALL output-phase motor units"
      " within radius (positive units > 1−r, negative units < r).  ")
    a("> **strict_trial_match** = `trial_all_supervised_units_within_radius`: ALL supervised"
      " units across BOTH input and output phases within radius.")
    a("")
    a("> **Count reconstruction note:** `n_positive_failures`, `n_negative_failures`, and"
      " `n_input_silence_failures` are reconstructed from stored threshold accuracies and"
      " are exact up to floating-point rounding, because the accuracies were computed from"
      " discrete threshold counts.")
    a("")

    # §1
    a("## §1 Run metadata")
    a("")
    if run_config:
        cfg_keys = [
            "epochs", "lr", "sound_proj_size", "output_positive_weight",
            "eval_radius", "zero_error_radius", "seed", "motor_output_size",
            "loss_variant", "loss_reduction",
        ]
        a("| Key | Value |")
        a("|---|---|")
        for k in cfg_keys:
            if k in run_config:
                a(f"| `{k}` | `{run_config[k]}` |")
        if "model_config" in run_config:
            for k, v in run_config["model_config"].items():
                a(f"| `model.{k}` | `{v}` |")
    else:
        a("*run_config.json not found.*")
    a("")

    # §2
    a("## §2 Summary counts")
    a("")
    a(f"| Metric | Count | % of {n} |")
    a("|---|---|---|")
    a(f"| Total items | {n} | 100.0% |")
    a(f"| Exact argmax match | {n_exact} | {_pct(n_exact, n)} |")
    a(f"| Paper-like match (output phase all-units within radius) | {n_paper} | {_pct(n_paper, n)} |")
    a(f"| Strict trial match (input + output all-units within radius) | {n_strict} | {_pct(n_strict, n)} |")
    a(f"| Exact match but not paper-like | {n_exact_not_paper} | {_pct(n_exact_not_paper, n)} |")
    a(f"| Argmax failure | {n_argmax_fail} | {_pct(n_argmax_fail, n)} |")
    a("")
    a(f"*Among the {n_exact} exact-match words:*")
    a("")
    a(f"| | Count | % of {n_exact} exact-match words |")
    a("|---|---|---|")
    a(f"| Words with ≥1 positive output-tick failure (positive unit ≤ {1.0 - eval_radius:.1f})"
      f" | {n_words_any_pos_fail} | {_pct(n_words_any_pos_fail, n_exact)} |")
    a(f"| Words with ≥1 negative output-unit failure (negative unit ≥ {eval_radius:.1f})"
      f" | {n_words_any_neg_fail} | {_pct(n_words_any_neg_fail, n_exact)} |")
    a(f"| Words with both positive and negative failures"
      f" | {n_words_both_fail} | {_pct(n_words_both_fail, n_exact)} |")
    a("")
    a("*Aggregate failure rates across ticks/units (not per-word rates):*")
    a("")
    a("| | Failing | Total | Rate |")
    a("|---|---|---|---|")
    a(f"| Positive output ticks (exact-match words only)"
      f" | {total_pos_fail_ticks} | {total_pos_ticks} | {pct_pos_ticks:.1f}% |")
    a(f"| Negative (tick, unit) pairs (all words)"
      f" | {total_neg_fail_pairs} | {total_neg_pairs} | {pct_neg_pairs:.1f}% |")
    a("")

    # §3
    a("## §3 Failure mode table")
    a("")
    a("| failure_type | count | % of total |")
    a("|---|---|---|")
    for ft in _FAILURE_TYPE_ORDER:
        c = type_counts.get(ft, 0)
        if c > 0:
            a(f"| `{ft}` | {c} | {_pct(c, n)} |")
    a("")

    # §4
    a("## §4 Length analysis")
    a("")
    a("| T | Count | Exact match | Paper-like | Mean pos failures/word | Mean neg failures/word |")
    a("|---|---|---|---|---|---|")
    for t in sorted(by_T.keys()):
        grp = by_T[t]
        ng = len(grp)
        ne = sum(1 for r in grp if r["argmax_exact_match"])
        np_ = sum(1 for r in grp if r["paper_like_match"])
        mean_pf = sum(r["n_positive_failures"] for r in grp) / ng
        mean_nf = sum(r["n_negative_failures"] for r in grp) / ng
        a(f"| {t} | {ng} | {ne}/{ng} ({_pct(ne, ng)}) | {np_}/{ng} ({_pct(np_, ng)})"
          f" | {mean_pf:.2f} | {mean_nf:.2f} |")
    a("")

    # §5
    a("## §5 Activation threshold analysis")
    a("")
    a("### §5a Positive activation (exact-match words only)")
    a("")
    a("Based on `mean_positive_output` (mean activation of the target-positive unit"
      " across output ticks per word).")
    a("")
    if mean_pos_acts:
        n_pos_no_fail = sum(1 for r in exact_rows if r["n_positive_failures"] == 0)
        a(f"- Mean of per-word mean positive activation: **{_fmean(mean_pos_acts):.4f}**")
        a(f"- Min: {min(mean_pos_acts):.4f} | Max: {max(mean_pos_acts):.4f}")
        a(f"- Words with zero positive failures (all positive units > {1.0 - eval_radius:.1f})"
          f": **{n_pos_no_fail}/{len(exact_rows)}**")
    a("")
    a("### §5b Negative activation (all words)")
    a("")
    a("Based on `mean_negative_output` (mean activation across all target-negative units"
      " × output ticks per word).")
    a("")
    if mean_neg_acts:
        n_neg_no_fail = sum(1 for r in rows if r["n_negative_failures"] == 0)
        a(f"- Mean of per-word mean negative activation: **{_fmean(mean_neg_acts):.4f}**")
        a(f"- Min: {min(mean_neg_acts):.4f} | Max: {max(mean_neg_acts):.4f}")
        a(f"- Words with zero negative failures (all negative units < {eval_radius:.1f})"
          f": **{n_neg_no_fail}/{n}**")
    a("")
    a("### §5c Radius sensitivity (documented limitation)")
    a("")
    a(f"Threshold classification is only possible at `eval_radius={eval_radius}` for"
      " this run because only summary statistics (means, booleans, and threshold counts)"
      " are stored per word. To compute passage rates at other radii, per-tick per-unit"
      " motor activations would need to be saved during evaluation.")
    a("")

    # §6
    a("## §6 Examples")
    a("")
    a("### §6a Paper-like successes")
    a("")
    if paper_rows:
        for r in paper_rows:
            pred = predictions[r["item_index"]]
            a(f"- `{r['word_or_label']}` (T={r['T']}): "
              f"target={pred['target_phonemes']}, predicted={pred['predicted_phonemes']}  ")
            a(f"  mean_positive={r['mean_positive_activation']:.4f},"
              f" mean_negative={r['mean_negative_activation']:.4f}")
    else:
        a("*None.*")
    a("")

    a("### §6b Exact-match words failing only because positives under threshold")
    a("")
    a("*(sorted by n_positive_failures ascending — nearest miss first)*")
    a("")
    if pos_under_rows:
        for r in pos_under_rows:
            pred = predictions[r["item_index"]]
            a(f"- `{r['word_or_label']}` (T={r['T']}, {r['n_positive_failures']} positive"
              f" failure(s)): target={pred['target_phonemes']},"
              f" mean_positive={r['mean_positive_activation']:.4f}")
    else:
        a("*None.*")
    a("")

    a("### §6c Exact-match words failing only because negatives over threshold")
    a("")
    a("*(sorted by n_negative_failures ascending — nearest miss first)*")
    a("")
    if neg_over_rows:
        for r in neg_over_rows:
            pred = predictions[r["item_index"]]
            a(f"- `{r['word_or_label']}` (T={r['T']}, {r['n_negative_failures']} negative"
              f" failure(s)): target={pred['target_phonemes']},"
              f" mean_negative={r['mean_negative_activation']:.4f}")
    else:
        a("*None.*")
    a("")

    a("### §6d Argmax failures")
    a("")
    if argmax_fail_rows:
        for r in argmax_fail_rows:
            pred = predictions[r["item_index"]]
            a(f"- `{r['word_or_label']}` (T={r['T']}, {r['n_argmax_errors']} error(s)):"
              f" target={pred['target_phonemes']},"
              f" predicted={pred['predicted_phonemes']}")
    else:
        a("*None.*")
    a("")

    a("### §6e Near-misses (exact match, fewest total threshold failures)")
    a("")
    a("*(sorted by n_positive_failures + n_negative_failures ascending)*")
    a("")
    if near_miss_rows:
        for r in near_miss_rows:
            a(f"- `{r['word_or_label']}` (T={r['T']}): {r['n_positive_failures']} pos failure(s),"
              f" {r['n_negative_failures']} neg failure(s) — `{r['failure_type']}`")
    else:
        a("*None.*")
    a("")

    # §7
    a("## §7 Interpretation")
    a("")
    a(f"Of the {n_exact} exact-match words, {n_exact_not_paper}"
      f" ({_pct(n_exact_not_paper, n_exact)} of exact matches) fail the paper-like"
      " all-units-within-radius criterion.")
    a("")
    a(f"**Positive tick failure rate:** {pct_pos_ticks:.1f}% of output positive ticks fail"
      f" the `>{1.0 - eval_radius:.1f}` threshold (across exact-match words)."
      " This is a per-tick rate, not a per-word rate: "
      f"**{n_words_any_pos_fail}/{n_exact}** words"
      f" ({_pct(n_words_any_pos_fail, n_exact)}) have at least one positive failure.")
    a("")
    a(f"**Negative unit failure rate:** {pct_neg_pairs:.1f}% of (output tick, negative unit)"
      " pairs fail the `<" + f"{eval_radius:.1f}` threshold. There are {n_neg_per_tick}"
      " negative units per output tick, so even a small per-pair rate can cause many words"
      f" to have at least one failing unit. Indeed, **{n_words_any_neg_fail}/{n}** words"
      f" ({_pct(n_words_any_neg_fail, n)}) have at least one negative failure.")
    a("")
    a(f"The dominant failure mode is **`{dominant}`**"
      f" ({type_counts[dominant]} words, {_pct(type_counts[dominant], n)} of total).")
    a("")
    a(f"Input silence is near-perfect: only {n_input_fail_words} word(s) fail the"
      " input-silence criterion.")
    a("")

    # §8
    a("## §8 Next-step implications")
    a("")
    a("| Signal | Implication |")
    a("|---|---|")
    a(f"| Positive tick failure rate = {pct_pos_ticks:.0f}%,"
      f" {n_words_any_pos_fail}/{n_exact} words | Primary bottleneck: positive units not"
      " reaching threshold. Continue training or increase `output_positive_weight` |")
    a(f"| Negative failure rate = {pct_neg_pairs:.1f}%,"
      f" {n_words_any_neg_fail}/{n} words | Secondary issue: check whether"
      " `output_positive_weight=3` is under-penalising negatives |")
    a(f"| {n_exact}/{n} exact argmax matches | Model has learned phoneme ordering;"
      " calibration (activation magnitude) is the remaining gap |")
    a(f"| {n_paper}/{n} paper-like vs {n_strict}/{n} strict | Difference ="
      f" {n_paper - n_strict} words where the input phase creates the gap |")
    a("")
    a("Options to consider:")
    a("")
    a("1. **Continue to 500 epochs** — if loss is still decreasing at epoch 300,"
      " more training may push positive units above the threshold.")
    a("2. **Lower lr** — sharper positive activations often emerge from finer convergence.")
    a("3. **Increase output_positive_weight further** — if positive failures dominate"
      " and negatives are already well-suppressed.")
    a("4. **Widen eval_radius** — if the strict 0.1 criterion is an overly demanding"
      " target and mean activations are already close to 0.9/0.1.")
    a("")

    # §9
    a("## §9 Limitations")
    a("")
    a("The following fields are **not computable** from `predictions_after.json` because"
      " only per-word summary statistics are stored:")
    a("")
    a("- `min_positive_activation` / `max_negative_activation` — per-tick extremes"
      " require per-tick motor tensors")
    a("- `mean_argmax_margin` / `min_argmax_margin` — margin between argmax and"
      " runner-up requires full per-tick motor tensors")
    a("- Radius sensitivity: threshold-passage rates at eval_radius ≠"
      f" {eval_radius} require raw activations, not means")
    a("- Which specific output tick failed within a word — count is available"
      " (`n_positive_failures`), but tick identity is not")
    a("")
    a("These fields appear as `N/A` in the CSV.")
    a("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Analyze predictions_after.json from a repetition-only run."
    )
    p.add_argument(
        "--run-dir", required=True, type=Path, metavar="PATH",
        help="Directory containing predictions_after.json",
    )
    p.add_argument(
        "--eval-radius", type=float, default=0.1,
        help="Eval radius used in the training run (default 0.1)",
    )
    p.add_argument(
        "--output-md", required=True, type=Path, metavar="PATH",
        help="Markdown report output path",
    )
    p.add_argument(
        "--output-csv", required=True, type=Path, metavar="PATH",
        help="CSV output path",
    )
    args = p.parse_args(argv)

    run_dir = args.run_dir.resolve()
    predictions = load_predictions(run_dir)
    run_config = load_run_config(run_dir)
    load_final_metrics(run_dir)  # loaded for completeness; not currently used in body

    motor_size = run_config.get("motor_output_size", 39)
    sound_input_size = run_config.get("raw_sound_input_size", motor_size)
    rows = [
        analyze_one(pred, i, motor_size=motor_size, sound_input_size=sound_input_size)
        for i, pred in enumerate(predictions)
    ]

    csv_path = args.output_csv
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    md_path = args.output_md
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md = generate_markdown(
        rows, predictions, run_config, run_dir,
        args.eval_radius, csv_path, md_path,
    )
    with open(md_path, "w") as f:
        f.write(md)

    n = len(rows)
    counts = Counter(r["failure_type"] for r in rows)
    print(f"Analyzed {n} words.")
    print(f"CSV:      {csv_path}")
    print(f"Markdown: {md_path}")
    print("")
    print("Failure type summary:")
    for ft in _FAILURE_TYPE_ORDER:
        c = counts.get(ft, 0)
        if c > 0:
            print(f"  {ft}: {c} ({100.0 * c / n:.1f}%)")


if __name__ == "__main__":
    main()
