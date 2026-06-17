"""Always-run unit tests for _compute_repetition_word_accuracy.

Tests use synthetic tensors only — no private CSV data required.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from train_repetition_only import _compute_repetition_word_accuracy


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


def test_perfect_output_phase():
    """All output units within radius of target → output metric True."""
    T, N  = 3, 39
    pos   = [0, 5, 10]
    # Motor output: positive units near 1, negatives near 0; all within 0.05 of target.
    motor_output  = torch.full((T, N), 0.05)
    motor_targets = _one_hot(T, N, pos)
    motor_output[torch.arange(T), torch.tensor(pos)] = 0.96  # > 0.9 from target=1
    motor_input   = torch.full((T, N), 0.02)

    m = _compute_repetition_word_accuracy(motor_input, motor_output, motor_targets, radius=0.1)

    assert m["output_all_units_within_radius"] is True


def test_one_negative_unit_too_high():
    """One negative unit above radius → output metric False."""
    T, N  = 2, 5
    motor_output  = torch.zeros(T, N)
    motor_targets = _one_hot(T, N, [0, 1])
    motor_output[0, 2] = 0.15   # negative unit (target=0) at 0.15 > radius 0.1
    motor_output[torch.arange(T), torch.tensor([0, 1])] = 0.95
    motor_input   = torch.zeros(T, N)

    m = _compute_repetition_word_accuracy(motor_input, motor_output, motor_targets, radius=0.1)

    assert m["output_all_units_within_radius"] is False


def test_positive_unit_too_low():
    """Positive unit far below target=1 → |output−target|=1.0 > radius → output metric False."""
    T, N  = 2, 5
    motor_output  = torch.zeros(T, N)   # all outputs = 0; positive targets = 1 → diff = 1.0
    motor_targets = _one_hot(T, N, [0, 1])
    motor_input   = torch.zeros(T, N)

    m = _compute_repetition_word_accuracy(motor_input, motor_output, motor_targets, radius=0.1)

    assert m["output_all_units_within_radius"] is False


def test_input_not_silent_output_can_be_correct():
    """Input phase above radius → input metric False; output metric is independent."""
    T, N  = 2, 5
    motor_input   = torch.full((T, N), 0.5)   # not silent
    motor_output  = torch.zeros(T, N)
    motor_targets = _one_hot(T, N, [0, 1])
    # Make output phase perfect (positive near 1, negatives near 0)
    motor_output[torch.arange(T), torch.tensor([0, 1])] = 0.95

    m = _compute_repetition_word_accuracy(motor_input, motor_output, motor_targets, radius=0.1)

    assert m["input_all_silent_within_radius"] is False
    assert m["output_all_units_within_radius"] is True


def test_input_silent():
    """All input units below radius → input metric True."""
    T, N  = 2, 5
    motor_input   = torch.full((T, N), 0.05)   # 0.05 < 0.1
    motor_output  = torch.zeros(T, N)
    motor_targets = _one_hot(T, N, [0, 1])

    m = _compute_repetition_word_accuracy(motor_input, motor_output, motor_targets, radius=0.1)

    assert m["input_all_silent_within_radius"] is True


def test_strict_requires_both_phases():
    """Strict metric is False if only one phase is correct."""
    T, N  = 2, 5
    motor_output  = torch.zeros(T, N)
    motor_targets = _one_hot(T, N, [0, 1])
    motor_output[torch.arange(T), torch.tensor([0, 1])] = 0.95  # output correct

    # Input not silent
    motor_input_noisy = torch.full((T, N), 0.5)
    m1 = _compute_repetition_word_accuracy(motor_input_noisy, motor_output, motor_targets)
    assert m1["output_all_units_within_radius"]          is True
    assert m1["input_all_silent_within_radius"]           is False
    assert m1["trial_all_supervised_units_within_radius"] is False

    # Both phases correct → strict True
    motor_input_silent = torch.full((T, N), 0.02)
    m2 = _compute_repetition_word_accuracy(motor_input_silent, motor_output, motor_targets)
    assert m2["output_all_units_within_radius"]          is True
    assert m2["input_all_silent_within_radius"]           is True
    assert m2["trial_all_supervised_units_within_radius"] is True


def test_custom_radius_respected():
    """A unit at 0.15 fails radius=0.1 but passes radius=0.2."""
    T, N  = 1, 5
    motor_output  = torch.zeros(T, N)
    motor_targets = _one_hot(T, N, [0])
    motor_output[0, 0] = 0.85   # positive unit: |0.85 − 1.0| = 0.15
    motor_input   = torch.zeros(T, N)

    m_strict = _compute_repetition_word_accuracy(motor_input, motor_output, motor_targets, radius=0.1)
    m_loose  = _compute_repetition_word_accuracy(motor_input, motor_output, motor_targets, radius=0.2)

    assert m_strict["output_all_units_within_radius"] is False   # 0.15 >= 0.1
    assert m_loose["output_all_units_within_radius"]  is True    # 0.15 < 0.2


def test_all_keys_present():
    """Return dict always contains exactly the 3 expected keys."""
    T, N = 2, 5
    m = _compute_repetition_word_accuracy(
        torch.zeros(T, N), torch.zeros(T, N), _one_hot(T, N, [0, 1])
    )
    expected = {
        "output_all_units_within_radius",
        "input_all_silent_within_radius",
        "trial_all_supervised_units_within_radius",
    }
    assert set(m.keys()) == expected


def test_all_values_are_bool():
    """All returned values are Python bools."""
    T, N = 2, 5
    m = _compute_repetition_word_accuracy(
        torch.rand(T, N), torch.rand(T, N), _one_hot(T, N, [0, 1])
    )
    for key, val in m.items():
        assert isinstance(val, bool), f"Key {key!r} has type {type(val)}, expected bool"
