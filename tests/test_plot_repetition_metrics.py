"""Lightweight tests for scripts/plot_repetition_metrics.py.

Uses synthetic metrics CSVs only — no training runs, no lichtheim2 imports.
All plots are rendered to a non-interactive Agg backend in tmp_path.
"""
from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ---------------------------------------------------------------------------
# Script loader
# ---------------------------------------------------------------------------


def _load_script():
    script_path = (
        Path(__file__).parent.parent / "scripts" / "plot_repetition_metrics.py"
    )
    spec = importlib.util.spec_from_file_location("plot_repetition_metrics", script_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Synthetic CSV builders
# ---------------------------------------------------------------------------

_FULL_COLS = [
    "epoch", "avg_loss", "min_loss", "max_loss", "n_trials", "all_finite",
    "avg_eval_input_bce", "avg_eval_output_bce", "avg_eval_motor_bce",
    "avg_eval_output_pos_bce", "avg_eval_output_neg_bce",
    "avg_eval_n_active_input", "avg_eval_n_active_output_pos",
    "avg_eval_n_active_output_neg",
]


def _write_metrics_csv(
    path: Path,
    n_epochs: int = 50,
    with_max_outlier: bool = False,
    include_min_max: bool = True,
    include_decomp: bool = True,
) -> None:
    """Write a synthetic metrics.csv with controllable column sets."""
    cols = ["epoch", "avg_loss"]
    if include_min_max:
        cols += ["min_loss", "max_loss"]
    cols += ["n_trials", "all_finite"]
    if include_decomp:
        cols += [
            "avg_eval_input_bce",
            "avg_eval_output_pos_bce",
            "avg_eval_output_neg_bce",
        ]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for e in range(1, n_epochs + 1):
            avg = max(0.1, 100.0 / e)
            row: dict[str, float | int | str] = {
                "epoch": e,
                "avg_loss": avg,
                "n_trials": 200,
                "all_finite": "True",
            }
            if include_min_max:
                row["min_loss"] = avg * 0.5
                # Spike max_loss at epoch 1 if requested
                row["max_loss"] = avg * 1000.0 if (with_max_outlier and e == 1) else avg * 2.0
            if include_decomp:
                row["avg_eval_input_bce"]      = avg * 0.05
                row["avg_eval_output_pos_bce"] = avg * 0.6
                row["avg_eval_output_neg_bce"] = avg * 0.35
            writer.writerow(row)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_avg_only_plot_does_not_crash_with_max_outliers(tmp_path):
    """Large max_loss spike at epoch 1 must not prevent loss_curve.png from saving."""
    _write_metrics_csv(tmp_path / "metrics.csv", n_epochs=50, with_max_outlier=True)
    mod = _load_script()
    mod.main(["--run-dir", str(tmp_path)])
    assert (tmp_path / "loss_curve.png").exists()
    assert (tmp_path / "loss_curve.png").stat().st_size > 0


def test_missing_min_max_skips_range_clipped(tmp_path):
    """Without min_loss/max_loss columns, range-clipped plot is skipped gracefully."""
    _write_metrics_csv(tmp_path / "metrics.csv", include_min_max=False)
    mod = _load_script()
    mod.main(["--run-dir", str(tmp_path)])
    assert (tmp_path / "loss_curve.png").exists()
    assert not (tmp_path / "loss_curve_range_clipped.png").exists()


def test_missing_decomp_columns_skips_decomp_plot(tmp_path):
    """Without avg_eval_* columns, decomposition plot is skipped gracefully."""
    _write_metrics_csv(tmp_path / "metrics.csv", include_decomp=False)
    mod = _load_script()
    mod.main(["--run-dir", str(tmp_path)])
    assert (tmp_path / "loss_curve.png").exists()
    assert not (tmp_path / "loss_decomposition_curve.png").exists()


def test_full_csv_generates_all_three_plots(tmp_path):
    """Full CSV with all columns generates all three plot files."""
    _write_metrics_csv(tmp_path / "metrics.csv", n_epochs=50)
    mod = _load_script()
    mod.main(["--run-dir", str(tmp_path)])

    for name in ["loss_curve.png", "loss_curve_range_clipped.png",
                 "loss_decomposition_curve.png"]:
        p = tmp_path / name
        assert p.exists(), f"Missing plot: {name}"
        assert p.stat().st_size > 0, f"Empty plot file: {name}"


def test_output_dir_option(tmp_path):
    """--output-dir saves plots to a different directory than --run-dir."""
    run_dir = tmp_path / "run"
    out_dir = tmp_path / "plots"
    run_dir.mkdir()
    _write_metrics_csv(run_dir / "metrics.csv")
    mod = _load_script()
    mod.main(["--run-dir", str(run_dir), "--output-dir", str(out_dir)])
    assert (out_dir / "loss_curve.png").exists()
    assert not (run_dir / "loss_curve.png").exists()


def test_rolling_mean_helper():
    """_rolling_mean produces the correct trailing window average."""
    mod = _load_script()
    values = [10.0, 8.0, 6.0, 4.0, 2.0]
    result = mod._rolling_mean(values, window=3)
    # i=0: mean([10]) = 10.0
    # i=1: mean([10,8]) = 9.0
    # i=2: mean([10,8,6]) = 8.0
    # i=3: mean([8,6,4]) = 6.0
    # i=4: mean([6,4,2]) = 4.0
    assert result == pytest.approx([10.0, 9.0, 8.0, 6.0, 4.0])
