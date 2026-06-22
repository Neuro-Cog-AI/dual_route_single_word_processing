"""Always-run unit tests for the output-positive loss weighting diagnostic (Phase 3k).

Tests that:
  - weight=1.0 gives identical loss to the unweighted baseline (exact baseline safety);
  - weight > 1 increases loss when positive units are suppressed;
  - weight has no effect on input-phase ticks (all targets = 0 there);
  - weight has no effect on negative output-phase units (target < 0.5);
  - dead zone excludes positive units before weighting applies;
  - train_step propagates the weight correctly;
  - validate_args rejects non-positive weights;
  - weight has no effect on a SPEAKING trial (task scope check, user-required).

No CSV data required — all tests use synthetic configs and tensors.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lichtheim2.layers import ModelState, TickResult
from lichtheim2.losses import compute_trial_loss_breakdown
from lichtheim2.tasks import Task
from lichtheim2.trials import SupervisedTrial, make_repetition_trial, make_speaking_trial


# ---------------------------------------------------------------------------
# Synthetic helpers
# ---------------------------------------------------------------------------

_DUMMY_VATL_SIZE = 3
_DUMMY_ISMG_SIZE = 4
_DUMMY_MSTG_SIZE = 6
_DUMMY_ASTG_SIZE = 8
_DUMMY_TRI_SIZE  = 4


def _make_state(motor: torch.Tensor) -> ModelState:
    """ModelState with the given motor activation; all other fields zeroed."""
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


def _make_tick_results(
    motor_outputs: torch.Tensor,
    task: Task = Task.REPETITION,
) -> list[TickResult]:
    """Build TickResult list from (n_ticks, motor_size) motor output tensor."""
    motor_size = motor_outputs.shape[1]
    return [
        TickResult(
            tick_index=t,
            task=task,
            sound_input=torch.zeros(motor_size),
            vATL_input_used=torch.zeros(_DUMMY_VATL_SIZE),
            state=_make_state(motor_outputs[t]),
        )
        for t in range(motor_outputs.shape[0])
    ]


def _make_trial(T: int, motor_size: int) -> SupervisedTrial:
    """Synthetic REPETITION trial: one-hot phoneme per tick, cycling over units."""
    phon = torch.zeros(T, motor_size)
    for t in range(T):
        phon[t, t % motor_size] = 1.0
    return make_repetition_trial(phon, motor_size, item_id=0, label="synth")


def _load_train_script():
    script_path = (
        Path(__file__).parent.parent / "scripts" / "train_repetition_only.py"
    )
    spec = importlib.util.spec_from_file_location("train_repetition_only", script_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Test 1: weight=1.0 is identical to default (unweighted) call
# ---------------------------------------------------------------------------


def test_weight_1_0_identical_to_standard_bce():
    """output_positive_weight=1.0 gives exactly the same loss as the default call."""
    T, motor_size = 3, 5
    trial = _make_trial(T, motor_size)
    torch.manual_seed(0)
    motor_outputs = torch.sigmoid(torch.randn(2 * T, motor_size))
    tick_results = _make_tick_results(motor_outputs)

    with torch.no_grad():
        loss_default = compute_trial_loss_breakdown(tick_results, trial, 0.0).total_loss.item()
        loss_w1 = compute_trial_loss_breakdown(
            tick_results, trial, 0.0, output_positive_weight=1.0
        ).total_loss.item()

    assert abs(loss_default - loss_w1) < 1e-6


# ---------------------------------------------------------------------------
# Test 2: weight > 1 increases loss when positive units are suppressed
# ---------------------------------------------------------------------------


def test_higher_weight_increases_loss_on_positive_failure():
    """With suppressed motor (≈0) and target=1, output_positive_weight > 1 raises loss."""
    T, motor_size = 3, 5
    trial = _make_trial(T, motor_size)
    # Suppress all outputs → positive units wrong (target=1, output≈0)
    motor_outputs = torch.full((2 * T, motor_size), 1e-6)
    tick_results = _make_tick_results(motor_outputs)

    with torch.no_grad():
        loss_w1 = compute_trial_loss_breakdown(
            tick_results, trial, 0.0, output_positive_weight=1.0
        ).total_loss.item()
        loss_w5 = compute_trial_loss_breakdown(
            tick_results, trial, 0.0, output_positive_weight=5.0
        ).total_loss.item()

    assert loss_w5 > loss_w1


# ---------------------------------------------------------------------------
# Test 3: weight has no effect when positive units are in the dead zone
# ---------------------------------------------------------------------------


def test_weight_no_effect_when_positives_correct():
    """When all outputs match targets (dead zone), weight has no effect."""
    T, motor_size = 2, 5
    trial = _make_trial(T, motor_size)
    # Perfect predictions → all units dead-zoned with radius=0.1
    motor_outputs = trial.motor_targets.clone()
    tick_results = _make_tick_results(motor_outputs)

    with torch.no_grad():
        loss_w1 = compute_trial_loss_breakdown(
            tick_results, trial, 0.1, output_positive_weight=1.0
        ).total_loss.item()
        loss_w5 = compute_trial_loss_breakdown(
            tick_results, trial, 0.1, output_positive_weight=5.0
        ).total_loss.item()

    # All dead-zoned → weight irrelevant
    assert abs(loss_w1 - loss_w5) < 1e-6


# ---------------------------------------------------------------------------
# Test 4: weight has no effect on input-phase ticks
# ---------------------------------------------------------------------------


def test_weight_does_not_affect_input_phase():
    """Weight scoped to output phase; input-phase loss is unchanged by weight > 1."""
    T, motor_size = 3, 5
    trial = _make_trial(T, motor_size)
    # Input phase: 0.5 everywhere (wrong; input targets = 0) → active loss
    # Output phase: exactly match targets → no output loss
    motor_outputs = torch.zeros(2 * T, motor_size)
    motor_outputs[:T] = 0.5
    motor_outputs[T:] = trial.motor_targets[T:].clone()
    tick_results = _make_tick_results(motor_outputs)

    with torch.no_grad():
        loss_w1 = compute_trial_loss_breakdown(
            tick_results, trial, 0.0, output_positive_weight=1.0
        ).total_loss.item()
        loss_w5 = compute_trial_loss_breakdown(
            tick_results, trial, 0.0, output_positive_weight=5.0
        ).total_loss.item()

    assert loss_w1 > 0.0  # meaningful test: input phase loss is non-zero
    assert abs(loss_w1 - loss_w5) < 1e-6


# ---------------------------------------------------------------------------
# Test 5: weight does not affect negative output-phase units
# ---------------------------------------------------------------------------


def test_weight_does_not_affect_negative_units():
    """Weight only applies to units with target >= 0.5; negative units are unaffected."""
    T, motor_size = 3, 5
    trial = _make_trial(T, motor_size)
    # Positive units: perfect (match targets → trivial BCE)
    # Negative units in output phase: wrong (0.5 when target=0 → active loss)
    motor_outputs = trial.motor_targets.clone().float()
    for t in range(T):
        for unit in range(motor_size):
            if trial.motor_targets[T + t, unit] < 0.5:
                motor_outputs[T + t, unit] = 0.5
    tick_results = _make_tick_results(motor_outputs)

    with torch.no_grad():
        loss_w1 = compute_trial_loss_breakdown(
            tick_results, trial, 0.0, output_positive_weight=1.0
        ).total_loss.item()
        loss_w5 = compute_trial_loss_breakdown(
            tick_results, trial, 0.0, output_positive_weight=5.0
        ).total_loss.item()

    assert loss_w1 > 0.0  # meaningful: negative unit error exists
    assert abs(loss_w1 - loss_w5) < 1e-6


# ---------------------------------------------------------------------------
# Test 6: dead zone excludes units before weighting applies
# ---------------------------------------------------------------------------


def test_dead_zone_excludes_before_weighting():
    """Dead zone applied before weighting: large weight cannot resuscitate dead units."""
    T, motor_size = 2, 5
    trial = _make_trial(T, motor_size)
    # Perfect predictions → all units dead-zoned with radius=0.1
    motor_outputs = trial.motor_targets.clone()
    tick_results = _make_tick_results(motor_outputs)

    with torch.no_grad():
        bd_w100 = compute_trial_loss_breakdown(
            tick_results, trial, 0.1, output_positive_weight=100.0
        )

    # All dead-zoned → total loss ≈ 0 even with weight=100
    assert bd_w100.total_loss.item() < 1e-6


# ---------------------------------------------------------------------------
# Test 7: train_step propagates output_positive_weight to the loss
# ---------------------------------------------------------------------------


def test_train_step_weight_affects_gradient():
    """train_step with output_positive_weight=5 gives a different loss than weight=1."""
    from lichtheim2.config import ModelConfig
    from lichtheim2.model import Lichtheim2Model
    from lichtheim2.trainer import train_step

    cfg = ModelConfig(
        sound_input_size=5,
        motor_output_size=5,
        iSMG_hidden_size=4,
        mSTG_hidden_size=6,
        aSTG_hidden_size=8,
        triangularis_hidden_size=4,
        vATL_size=3,
        repetition_ticks=4,
        comprehension_ticks=2,
        speaking_ticks=2,
    )
    T, motor_size = 3, 5
    trial = _make_trial(T, motor_size)

    torch.manual_seed(42)
    model_w1 = Lichtheim2Model(cfg)
    opt_w1 = torch.optim.SGD(model_w1.parameters(), lr=0.01)
    loss_w1 = train_step(model_w1, trial, opt_w1, cfg, output_positive_weight=1.0)

    torch.manual_seed(42)
    model_w5 = Lichtheim2Model(cfg)
    opt_w5 = torch.optim.SGD(model_w5.parameters(), lr=0.01)
    loss_w5 = train_step(model_w5, trial, opt_w5, cfg, output_positive_weight=5.0)

    assert loss_w5 != loss_w1


# ---------------------------------------------------------------------------
# Test 8: validate_args rejects non-positive weights
# ---------------------------------------------------------------------------


def test_validate_args_rejects_nonpositive_weight():
    """validate_args returns an error string for --output-positive-weight <= 0."""
    mod = _load_train_script()

    args_zero = mod.parse_args(["--data-dir", ".", "--output-positive-weight", "0.0"])
    args_neg  = mod.parse_args(["--data-dir", ".", "--output-positive-weight", "-1.0"])

    assert mod.validate_args(args_zero) is not None, "weight=0.0 should be rejected"
    assert mod.validate_args(args_neg)  is not None, "weight=-1.0 should be rejected"


# ---------------------------------------------------------------------------
# Test 9: weight has no effect on a SPEAKING trial (task scope check)
# ---------------------------------------------------------------------------


def test_weight_no_effect_on_speaking_trial():
    """output_positive_weight > 1 has no effect on a SPEAKING trial.

    The weighting mask is scoped to task==REPETITION AND output phase T..2T-1.
    A SPEAKING trial (task != REPETITION) must give identical loss regardless of weight.
    """
    T, motor_size = 3, 5
    phon = torch.zeros(T, motor_size)
    for t in range(T):
        phon[t, t % motor_size] = 1.0   # one positive unit per tick
    sem = torch.zeros(_DUMMY_VATL_SIZE)
    trial = make_speaking_trial(phon, sem, motor_size, item_id=0, label="synth_speaking")

    # Suppress all motor outputs → positive units wrong → non-trivial loss
    # (if weighting were incorrectly applied to SPEAKING, w5 would differ from w1)
    motor_outputs = torch.full((T, motor_size), 1e-6)
    tick_results = _make_tick_results(motor_outputs, task=Task.SPEAKING)

    with torch.no_grad():
        loss_w1 = compute_trial_loss_breakdown(
            tick_results, trial, 0.0, output_positive_weight=1.0
        ).total_loss.item()
        loss_w5 = compute_trial_loss_breakdown(
            tick_results, trial, 0.0, output_positive_weight=5.0
        ).total_loss.item()

    # Loss must be non-zero (ensures the test case is meaningful)
    assert loss_w1 > 1.0, f"Expected non-trivial loss, got {loss_w1}"
    # Weight must have NO effect on SPEAKING task
    assert abs(loss_w1 - loss_w5) < 1e-6, (
        f"weight=5.0 changed SPEAKING loss: w1={loss_w1:.6f}, w5={loss_w5:.6f}"
    )
