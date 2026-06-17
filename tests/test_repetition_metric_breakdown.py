"""Always-run unit tests for _compute_repetition_metric_breakdown.

Tests use synthetic tensors only — no private CSV data required.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from train_repetition_only import _compute_repetition_metric_breakdown


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _one_hot(T: int, N: int, indices: list[int]) -> torch.Tensor:
    """Create (T, N) one-hot float tensor with 1.0 at given column indices."""
    t = torch.zeros(T, N)
    t[torch.arange(T), torch.tensor(indices)] = 1.0
    return t


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_all_suppressed():
    """All outputs near 0 → negative acc high, positive acc 0, positive mean low."""
    T, N = 4, 39
    motor_input   = torch.full((T, N), 0.05)
    motor_output  = torch.full((T, N), 0.05)
    motor_targets = _one_hot(T, N, [0, 1, 2, 3])

    m = _compute_repetition_metric_breakdown(motor_input, motor_output, motor_targets)

    assert m["input_silence_threshold_acc"]   == 1.0   # 0.05 < 0.1 for all input units
    assert m["output_negative_threshold_acc"] == 1.0   # 0.05 < 0.1 for all negative units
    assert m["output_positive_threshold_acc"] == 0.0   # 0.05 not > 0.9
    assert m["mean_positive_output"]          < 0.1    # near 0
    assert m["mean_negative_output"]          < 0.1    # near 0


def test_perfect_production():
    """Positive unit = 1.0, all negatives = 0.0 → both threshold accs = 1.0."""
    T, N   = 3, 39
    pos_idx = [5, 10, 20]
    motor_input   = torch.full((T, N), 0.02)
    motor_output  = torch.zeros(T, N)
    motor_targets = _one_hot(T, N, pos_idx)
    motor_output[torch.arange(T), torch.tensor(pos_idx)] = 1.0

    m = _compute_repetition_metric_breakdown(motor_input, motor_output, motor_targets)

    assert m["output_positive_threshold_acc"] == 1.0
    assert m["output_negative_threshold_acc"] == 1.0
    assert abs(m["mean_positive_output"] - 1.0) < 1e-4
    assert abs(m["mean_negative_output"] - 0.0) < 1e-4
    assert m["input_silence_threshold_acc"]   == 1.0


def test_input_not_suppressed():
    """Input motor activations = 0.5 → input_silence_threshold_acc = 0."""
    T, N = 2, 5
    motor_input   = torch.full((T, N), 0.5)   # 0.5 >= 0.1, not suppressed
    motor_output  = torch.zeros(T, N)
    motor_targets = _one_hot(T, N, [0, 1])

    m = _compute_repetition_metric_breakdown(motor_input, motor_output, motor_targets)

    assert m["input_silence_threshold_acc"] == 0.0


def test_output_positive_threshold_strict_boundary():
    """Positive unit at exactly 0.9 must NOT count (strict >)."""
    T, N = 1, 10
    motor_input   = torch.zeros(T, N)
    motor_output  = torch.zeros(T, N)
    motor_targets = _one_hot(T, N, [2])
    motor_output[0, 2] = 0.9   # exactly at boundary

    m = _compute_repetition_metric_breakdown(motor_input, motor_output, motor_targets)

    assert m["output_positive_threshold_acc"] == 0.0   # strict >, not >=


def test_mean_max_motor_output():
    """mean_max_motor_output is the mean of per-tick max activations."""
    T, N = 3, 5
    motor_input   = torch.zeros(T, N)
    motor_output  = torch.zeros(T, N)
    motor_targets = _one_hot(T, N, [0, 1, 2])

    # Set known max per tick: tick 0→0.3, tick 1→0.6, tick 2→0.9
    motor_output[0, 3] = 0.3
    motor_output[1, 3] = 0.6
    motor_output[2, 3] = 0.9

    m = _compute_repetition_metric_breakdown(motor_input, motor_output, motor_targets)

    expected_mean_max = (0.3 + 0.6 + 0.9) / 3
    assert abs(m["mean_max_motor_output"] - expected_mean_max) < 1e-4


def test_all_keys_present():
    """Return dict always contains exactly the 6 expected keys."""
    T, N = 2, 5
    m = _compute_repetition_metric_breakdown(
        torch.zeros(T, N),
        torch.zeros(T, N),
        _one_hot(T, N, [0, 1]),
    )
    expected_keys = {
        "input_silence_threshold_acc",
        "output_negative_threshold_acc",
        "output_positive_threshold_acc",
        "mean_positive_output",
        "mean_negative_output",
        "mean_max_motor_output",
    }
    assert set(m.keys()) == expected_keys


def test_all_values_finite():
    """All returned values are finite floats."""
    T, N = 4, 39
    m = _compute_repetition_metric_breakdown(
        torch.rand(T, N),
        torch.rand(T, N),
        _one_hot(T, N, [0, 5, 10, 20]),
    )
    for key, val in m.items():
        assert math.isfinite(val), f"Non-finite value for key {key!r}: {val}"


def test_device_safe_indexing_cpu():
    """Indexing works correctly on CPU (basic sanity for device-safe arange)."""
    T, N = 2, 5
    motor_output  = torch.zeros(T, N)
    motor_targets = _one_hot(T, N, [1, 3])
    motor_output[0, 1] = 0.95   # positive unit tick 0 → above 0.9
    motor_output[1, 3] = 0.80   # positive unit tick 1 → below 0.9

    m = _compute_repetition_metric_breakdown(
        torch.zeros(T, N), motor_output, motor_targets
    )

    # Only tick 0 positive unit exceeds 0.9 → 1/2 = 0.5
    assert abs(m["output_positive_threshold_acc"] - 0.5) < 1e-4
    assert abs(m["mean_positive_output"] - (0.95 + 0.80) / 2) < 1e-4
