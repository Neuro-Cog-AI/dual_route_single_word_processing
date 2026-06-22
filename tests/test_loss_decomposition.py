"""Always-run unit tests for _compute_trial_loss_decomposition (Phase 3i).

Tests the per-epoch diagnostic loss decomposition helper introduced in
train_repetition_only.py. No CSV data required — all tests use synthetic
ModelState / TickResult objects and make_repetition_trial().
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

# Add src/ so lichtheim2 imports work (also done inside the script itself).
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lichtheim2.layers import ModelState, TickResult
from lichtheim2.losses import compute_trial_loss_breakdown
from lichtheim2.tasks import Task
from lichtheim2.trials import SupervisedTrial, make_repetition_trial


# ---------------------------------------------------------------------------
# Load _compute_trial_loss_decomposition from the training script
# ---------------------------------------------------------------------------

def _load_train_script():
    script_path = (
        Path(__file__).parent.parent / "scripts" / "train_repetition_only.py"
    )
    spec = importlib.util.spec_from_file_location("train_repetition_only", script_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_train_mod = _load_train_script()
_compute_trial_loss_decomposition = _train_mod._compute_trial_loss_decomposition


# ---------------------------------------------------------------------------
# Synthetic helpers
# ---------------------------------------------------------------------------

_DUMMY_VATL_SIZE = 3
_DUMMY_ISMG_SIZE = 4
_DUMMY_MSTG_SIZE = 6
_DUMMY_ASTG_SIZE = 8
_DUMMY_TRI_SIZE  = 4


def _make_state(motor: torch.Tensor) -> ModelState:
    """ModelState with the given motor activation; all other fields zeroed/small."""
    return ModelState(
        iSMG=torch.zeros(_DUMMY_ISMG_SIZE),
        iSMG_context=torch.zeros(_DUMMY_ISMG_SIZE),
        motor=motor,
        motor_context=motor.clone(),
        mSTG=torch.zeros(_DUMMY_MSTG_SIZE),
        aSTG=torch.zeros(_DUMMY_ASTG_SIZE),
        vATL_out=torch.zeros(_DUMMY_VATL_SIZE),
        vATL_context=torch.zeros(_DUMMY_VATL_SIZE),
        triangularis=torch.zeros(_DUMMY_TRI_SIZE),
    )


def _make_tick_results(motor_outputs: torch.Tensor) -> list[TickResult]:
    """Build TickResult list from (2T, motor_size) motor output tensor."""
    motor_size = motor_outputs.shape[1]
    results = []
    for t in range(motor_outputs.shape[0]):
        results.append(TickResult(
            tick_index=t,
            task=Task.REPETITION,
            sound_input=torch.zeros(motor_size),
            vATL_input_used=torch.zeros(_DUMMY_VATL_SIZE),
            state=_make_state(motor_outputs[t]),
        ))
    return results


def _make_trial(T: int, motor_size: int) -> SupervisedTrial:
    """Synthetic repetition trial: one-hot phoneme per tick, cycling over units."""
    phon = torch.zeros(T, motor_size)
    for t in range(T):
        phon[t, t % motor_size] = 1.0
    return make_repetition_trial(phon, motor_size, item_id=0, label="synth")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_all_keys_present():
    """_compute_trial_loss_decomposition returns exactly the expected 8 keys."""
    T, motor_size = 2, 5
    trial = _make_trial(T, motor_size)
    motor_outputs = torch.rand(2 * T, motor_size)
    tick_results  = _make_tick_results(motor_outputs)
    d = _compute_trial_loss_decomposition(tick_results, trial, 0.0)
    expected = {
        "input_bce", "output_bce", "motor_bce",
        "output_pos_bce", "output_neg_bce",
        "n_active_input", "n_active_output_pos", "n_active_output_neg",
    }
    assert set(d.keys()) == expected


def test_input_plus_output_equals_total():
    """input_bce + output_bce == motor_bce (by construction, no dead zone)."""
    T, motor_size = 3, 5
    trial = _make_trial(T, motor_size)
    motor_outputs = torch.sigmoid(torch.randn(2 * T, motor_size))
    tick_results  = _make_tick_results(motor_outputs)
    d = _compute_trial_loss_decomposition(tick_results, trial, 0.0)
    assert abs(d["input_bce"] + d["output_bce"] - d["motor_bce"]) < 1e-5


def test_pos_plus_neg_equals_output():
    """output_pos_bce + output_neg_bce == output_bce (pos/neg partition)."""
    T, motor_size = 3, 5
    trial = _make_trial(T, motor_size)
    motor_outputs = torch.sigmoid(torch.randn(2 * T, motor_size))
    tick_results  = _make_tick_results(motor_outputs)
    d = _compute_trial_loss_decomposition(tick_results, trial, 0.0)
    assert abs(d["output_pos_bce"] + d["output_neg_bce"] - d["output_bce"]) < 1e-5


def test_all_units_active_no_dead_zone():
    """With zero_error_radius=0, all (tick, unit) pairs are counted as active."""
    T, motor_size = 2, 5
    trial = _make_trial(T, motor_size)
    motor_outputs = torch.full((2 * T, motor_size), 0.5)
    tick_results  = _make_tick_results(motor_outputs)
    d = _compute_trial_loss_decomposition(tick_results, trial, 0.0)
    assert d["n_active_input"] == T * motor_size
    assert d["n_active_output_pos"] == T                   # one positive unit per output tick
    assert d["n_active_output_neg"] == T * (motor_size - 1)


def test_dead_zone_excludes_units():
    """When motor_outputs == motor_targets everywhere, all units fall in the dead zone."""
    T, motor_size = 2, 5
    trial = _make_trial(T, motor_size)
    # Perfect predictions: |output - target| = 0 < radius
    motor_outputs = trial.motor_targets.clone()
    tick_results  = _make_tick_results(motor_outputs)
    d = _compute_trial_loss_decomposition(tick_results, trial, zero_error_radius=0.1)
    assert d["n_active_input"] == 0
    assert d["n_active_output_pos"] == 0
    assert d["n_active_output_neg"] == 0


def test_positive_unit_bce_when_suppressed():
    """Motor ≈ 0 while target = 1 → large output_pos_bce."""
    T, motor_size = 2, 5
    trial = _make_trial(T, motor_size)
    motor_outputs = torch.full((2 * T, motor_size), 1e-6)
    tick_results  = _make_tick_results(motor_outputs)
    d = _compute_trial_loss_decomposition(tick_results, trial, 0.0)
    # BCE(1e-6, 1) = -log(1e-6) ≈ 13.8 per positive unit
    assert d["output_pos_bce"] > 10.0 * T


def test_negative_bce_near_zero_when_suppressed():
    """Motor ≈ 0 while target = 0 → output_neg_bce ≈ 0."""
    T, motor_size = 2, 5
    trial = _make_trial(T, motor_size)
    motor_outputs = torch.full((2 * T, motor_size), 1e-6)
    tick_results  = _make_tick_results(motor_outputs)
    d = _compute_trial_loss_decomposition(tick_results, trial, 0.0)
    # BCE(1e-6, 0) = -log(1 - 1e-6) ≈ 1e-6 per unit; total is tiny
    assert d["output_neg_bce"] < 0.001


def test_decomp_consistent_with_losses_py():
    """input_bce + output_bce ≈ compute_trial_loss_breakdown().motor_loss."""
    T, motor_size = 3, 5
    trial = _make_trial(T, motor_size)
    torch.manual_seed(42)
    motor_outputs = torch.sigmoid(torch.randn(2 * T, motor_size))
    tick_results  = _make_tick_results(motor_outputs)

    d = _compute_trial_loss_decomposition(tick_results, trial, 0.0)
    with torch.no_grad():
        breakdown = compute_trial_loss_breakdown(tick_results, trial, 0.0)

    expected = breakdown.motor_loss.item()
    actual   = d["input_bce"] + d["output_bce"]
    assert abs(actual - expected) < 1e-4, (
        f"decomp total ({actual:.6f}) != breakdown.motor_loss ({expected:.6f})"
    )
