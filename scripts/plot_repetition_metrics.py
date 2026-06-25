#!/usr/bin/env python3
"""Regenerate loss-curve plots from an existing repetition-only run directory.

Reads metrics.csv (and optionally run_config.json) and saves plots into the
run directory (or a specified output directory):

  loss_curve.png              — avg loss + rolling mean (always)
  loss_curve_range_clipped.png — avg/min/max with clipped y-axis
                                 (skipped if min_loss or max_loss columns absent)
  loss_decomposition_curve.png — output-pos, output-neg, input BCE
                                 (skipped if avg_eval_* columns absent)

No model imports required — reads only CSV/JSON files.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lichtheim2.metrics import rolling_mean as _rolling_mean  # noqa: E402


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_metrics(csv_path: Path) -> list[dict[str, float]]:
    """Read metrics.csv; coerce numeric columns to float where possible."""
    if not csv_path.exists():
        print(f"ERROR: metrics.csv not found at {csv_path}", file=sys.stderr)
        sys.exit(1)
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            coerced: dict[str, float | str] = {}
            for k, v in row.items():
                try:
                    coerced[k] = float(v)
                except (ValueError, TypeError):
                    coerced[k] = v
            rows.append(coerced)
    return rows


# ---------------------------------------------------------------------------
# Plot functions
# ---------------------------------------------------------------------------


def plot_avg_loss(
    epochs: list[float],
    avg_losses: list[float],
    rolling_window: int,
    out_path: Path,
    plt,
) -> None:
    """Save loss_curve.png: avg loss + rolling mean."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, avg_losses, linewidth=1.5, label="avg loss")
    if len(epochs) >= rolling_window:
        smooth = _rolling_mean(avg_losses, rolling_window)
        ax.plot(epochs, smooth, linewidth=2, label=f"rolling mean (w={rolling_window})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("BCE loss (sum)")
    ax.set_title("Repetition-only training — average loss per epoch")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_range_clipped(
    epochs: list[float],
    avg_losses: list[float],
    min_losses: list[float],
    max_losses: list[float],
    clip_percentile: float,
    out_path: Path,
    plt,
) -> None:
    """Save loss_curve_range_clipped.png.

    Y-axis upper bound is clipped to clip_percentile of avg_loss values, not
    max_loss, so early max_loss spikes do not compress the average trajectory.
    """
    try:
        import numpy as np
        y_top = float(np.percentile(avg_losses, clip_percentile))
    except ImportError:
        # Fallback: sort-based percentile without numpy
        sorted_vals = sorted(avg_losses)
        idx = max(0, int(len(sorted_vals) * clip_percentile / 100) - 1)
        y_top = sorted_vals[idx]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.fill_between(epochs, min_losses, max_losses, alpha=0.15, label="min/max range")
    ax.plot(epochs, avg_losses, linewidth=1.5, label="avg loss")
    ax.set_ylim(bottom=0, top=y_top * 1.05)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("BCE loss (sum)")
    ax.set_title(
        f"Repetition-only training — avg/min/max loss\n"
        f"(y-axis clipped at {clip_percentile:.0f}th percentile of avg_loss)"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_decomposition(
    epochs: list[float],
    pos_bce: list[float],
    neg_bce: list[float],
    input_bce: list[float],
    out_path: Path,
    plt,
) -> None:
    """Save loss_decomposition_curve.png."""
    fig, ax = plt.subplots(figsize=(8, 4))
    for vals, label in [
        (pos_bce,   "output-positive BCE"),
        (neg_bce,   "output-negative BCE"),
        (input_bce, "input BCE"),
    ]:
        ax.plot(epochs, vals, linewidth=1.5, label=label)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("BCE loss (sum, unweighted eval)")
    ax.set_title("Loss decomposition — eval pass per epoch")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv=None) -> None:
    p = argparse.ArgumentParser(
        description="Regenerate loss-curve plots from an existing repetition-only run."
    )
    p.add_argument("--run-dir", required=True, type=Path, metavar="PATH",
                   help="Directory containing metrics.csv")
    p.add_argument("--output-dir", type=Path, default=None, metavar="PATH",
                   help="Directory to write plot files (default: same as --run-dir)")
    p.add_argument("--rolling-window", type=int, default=20, metavar="N",
                   help="Rolling-mean window size (default: 20)")
    p.add_argument("--clip-percentile", type=float, default=95.0, metavar="PCT",
                   help="Y-axis clip percentile of avg_loss for range plot (default: 95)")
    args = p.parse_args(argv)

    run_dir = args.run_dir.resolve()
    out_dir = (args.output_dir or run_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_metrics(run_dir / "metrics.csv")
    if not rows:
        print("ERROR: metrics.csv is empty.", file=sys.stderr)
        sys.exit(1)

    # Require at minimum: epoch and avg_loss
    if "epoch" not in rows[0] or "avg_loss" not in rows[0]:
        print("ERROR: metrics.csv must contain 'epoch' and 'avg_loss' columns.",
              file=sys.stderr)
        sys.exit(1)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("ERROR: matplotlib not available — cannot generate plots.", file=sys.stderr)
        sys.exit(1)

    epochs = [r["epoch"] for r in rows]
    avg_losses = [r["avg_loss"] for r in rows]
    saved: list[str] = []

    # Always: avg-only main plot
    plot_avg_loss(epochs, avg_losses, args.rolling_window,
                  out_dir / "loss_curve.png", plt)
    saved.append("loss_curve.png")

    # Optional: range-clipped plot
    if "min_loss" in rows[0] and "max_loss" in rows[0]:
        plot_range_clipped(
            epochs, avg_losses,
            [r["min_loss"] for r in rows],
            [r["max_loss"] for r in rows],
            args.clip_percentile,
            out_dir / "loss_curve_range_clipped.png",
            plt,
        )
        saved.append("loss_curve_range_clipped.png")
    else:
        print("  (min_loss/max_loss columns absent — skipping loss_curve_range_clipped.png)")

    # Optional: decomposition plot
    decomp_keys = ("avg_eval_output_pos_bce", "avg_eval_output_neg_bce", "avg_eval_input_bce")
    if all(k in rows[0] for k in decomp_keys):
        plot_decomposition(
            epochs,
            [r["avg_eval_output_pos_bce"] for r in rows],
            [r["avg_eval_output_neg_bce"] for r in rows],
            [r["avg_eval_input_bce"]      for r in rows],
            out_dir / "loss_decomposition_curve.png",
            plt,
        )
        saved.append("loss_decomposition_curve.png")
    else:
        print("  (decomposition columns absent — skipping loss_decomposition_curve.png)")

    print(f"  Saved: {', '.join(saved)}")
    if out_dir != run_dir:
        print(f"  Output directory: {out_dir}")


if __name__ == "__main__":
    main()
