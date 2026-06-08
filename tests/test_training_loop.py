"""Tests for the training loss and online training step.

All tests are always-run, CPU-only, using synthetic tensors.
No CSV files or private data required.
"""
from pathlib import Path

import torch
import torch.optim as optim

from lichtheim2.config import load_config
from lichtheim2.losses import compute_trial_loss
from lichtheim2.model import Lichtheim2Model
from lichtheim2.trainer import move_trial_to_device, train_step
from lichtheim2.trials import (
    make_comprehension_trial,
    make_repetition_trial,
    make_speaking_trial,
)

# ---------------------------------------------------------------------------
# Shared setup — toy config, small synthetic tensors
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent.parent
cfg = load_config(REPO_ROOT / "configs" / "toy.yaml")

T          = 3
SOUND_SIZE = cfg.sound_input_size
MOTOR_SIZE = cfg.motor_output_size
VATL_SIZE  = cfg.vATL_size

torch.manual_seed(0)
PHON = torch.rand(T, SOUND_SIZE)
SEM  = torch.rand(VATL_SIZE)


def _fresh_model() -> Lichtheim2Model:
    torch.manual_seed(0)
    return Lichtheim2Model(cfg)


# ---------------------------------------------------------------------------
# compute_trial_loss: basic properties
# ---------------------------------------------------------------------------

def test_compute_loss_is_scalar():
    model = _fresh_model()
    trial = make_repetition_trial(PHON, MOTOR_SIZE)
    results = model.run_trial(trial.task, trial.phon_tensor, torch.zeros(VATL_SIZE), cfg)
    loss = compute_trial_loss(results, trial)
    assert loss.shape == (), "loss must be a 0-dim tensor"


def test_compute_loss_positive():
    model = _fresh_model()
    trial = make_repetition_trial(PHON, MOTOR_SIZE)
    results = model.run_trial(trial.task, trial.phon_tensor, torch.zeros(VATL_SIZE), cfg)
    loss = compute_trial_loss(results, trial)
    assert loss.item() > 0.0


def test_compute_motor_loss_only_repetition():
    """Repetition trial has no semantic targets; semantic part must not raise."""
    model = _fresh_model()
    trial = make_repetition_trial(PHON, MOTOR_SIZE)
    assert trial.semantic_targets is None
    results = model.run_trial(trial.task, trial.phon_tensor, torch.zeros(VATL_SIZE), cfg)
    loss = compute_trial_loss(results, trial)
    assert loss.item() > 0.0


def test_compute_loss_includes_semantic_for_comprehension():
    """Comprehension trial has both motor and semantic loss."""
    model = _fresh_model()
    trial = make_comprehension_trial(PHON, SEM, MOTOR_SIZE)
    assert trial.semantic_targets is not None
    results = model.run_trial(trial.task, trial.phon_tensor, torch.zeros(VATL_SIZE), cfg)
    loss_with_sem = compute_trial_loss(results, trial)
    assert loss_with_sem.item() > 0.0


def test_compute_loss_masks_inactive_ticks():
    """Setting half the motor loss mask to False should reduce loss."""
    model = _fresh_model()
    trial_full = make_repetition_trial(PHON, MOTOR_SIZE)
    results = model.run_trial(trial_full.task, trial_full.phon_tensor, torch.zeros(VATL_SIZE), cfg)

    from dataclasses import replace as dc_replace
    # Zero-out the mask for the output phase (ticks T..2T-1)
    half_mask = trial_full.motor_loss_mask.clone()
    half_mask[T:] = False
    trial_half = dc_replace(trial_full, motor_loss_mask=half_mask)

    loss_full = compute_trial_loss(results, trial_full)
    loss_half = compute_trial_loss(results, trial_half)
    assert loss_half.item() < loss_full.item(), "masking out ticks must reduce loss"


def test_compute_loss_requires_grad():
    model = _fresh_model()
    trial = make_repetition_trial(PHON, MOTOR_SIZE)
    results = model.run_trial(trial.task, trial.phon_tensor, torch.zeros(VATL_SIZE), cfg)
    loss = compute_trial_loss(results, trial)
    assert loss.requires_grad, "loss must have requires_grad=True"


# ---------------------------------------------------------------------------
# Zero-error radius
# ---------------------------------------------------------------------------

def test_zero_error_radius_reduces_loss():
    """With radius=0.1, units close to their target contribute zero loss."""
    model = _fresh_model()
    # All-zero targets: model output near 0.5, so most units are outside the dead zone
    # but some may be within 0.1 of 0 by chance
    trial = make_repetition_trial(PHON, MOTOR_SIZE)
    results = model.run_trial(trial.task, trial.phon_tensor, torch.zeros(VATL_SIZE), cfg)

    loss_no_radius = compute_trial_loss(results, trial, zero_error_radius=0.0)
    loss_with_radius = compute_trial_loss(results, trial, zero_error_radius=0.1)
    # Loss with dead zone must be <= standard loss (dead zone can only remove terms)
    assert loss_with_radius.item() <= loss_no_radius.item()


def test_zero_error_radius_semantic():
    """Zero-error radius applies to semantic loss too."""
    model = _fresh_model()
    trial = make_comprehension_trial(PHON, SEM, MOTOR_SIZE)
    results = model.run_trial(trial.task, trial.phon_tensor, torch.zeros(VATL_SIZE), cfg)

    loss_no_r = compute_trial_loss(results, trial, zero_error_radius=0.0)
    loss_with_r = compute_trial_loss(results, trial, zero_error_radius=0.1)
    assert loss_with_r.item() <= loss_no_r.item()


# ---------------------------------------------------------------------------
# train_step
# ---------------------------------------------------------------------------

def test_train_step_returns_float():
    model = _fresh_model()
    trial = make_repetition_trial(PHON, MOTOR_SIZE)
    opt = optim.SGD(model.parameters(), lr=0.1)
    result = train_step(model, trial, opt, cfg)
    assert isinstance(result, float)
    assert torch.isfinite(torch.tensor(result))


def test_train_step_parameters_change():
    """Model parameters must change after one gradient step."""
    model = _fresh_model()
    before = [p.clone() for p in model.parameters()]
    trial = make_repetition_trial(PHON, MOTOR_SIZE)
    opt = optim.SGD(model.parameters(), lr=0.5)
    train_step(model, trial, opt, cfg)
    after = list(model.parameters())
    changed = any(not torch.equal(b, a) for b, a in zip(before, after))
    assert changed, "at least one parameter must change after a training step"


def test_train_step_no_nan():
    """Loss must remain finite over several steps."""
    torch.manual_seed(0)
    model = _fresh_model()
    trial = make_repetition_trial(PHON, MOTOR_SIZE)
    opt = optim.SGD(model.parameters(), lr=0.1)
    for _ in range(10):
        loss = train_step(model, trial, opt, cfg)
        assert torch.isfinite(torch.tensor(loss)), f"NaN/Inf loss encountered: {loss}"


def test_loss_decreases_with_training():
    """Final loss must be lower than initial loss after several steps.

    Uses a fixed seed and a simple all-zero repetition target (motor output
    trained toward 0). The model starts with outputs near 0.5, so there is a
    clear gradient signal. Monotonic per-step decrease is not required.
    """
    torch.manual_seed(0)
    model = Lichtheim2Model(cfg)  # fresh model with seed 0

    # Synthetic repetition trial — target: zeros in output phase
    phon = torch.rand(T, SOUND_SIZE)
    trial = make_repetition_trial(phon, MOTOR_SIZE)

    opt = optim.SGD(model.parameters(), lr=0.1)

    initial_loss = train_step(model, trial, opt, cfg)
    for _ in range(18):
        train_step(model, trial, opt, cfg)
    final_loss = train_step(model, trial, opt, cfg)

    assert final_loss < initial_loss, (
        f"Expected loss to decrease: initial={initial_loss:.4f}, final={final_loss:.4f}"
    )


# ---------------------------------------------------------------------------
# move_trial_to_device
# ---------------------------------------------------------------------------

def test_move_trial_to_cpu():
    trial = make_repetition_trial(PHON, MOTOR_SIZE)
    moved = move_trial_to_device(trial, torch.device("cpu"))
    assert moved.phon_tensor.device.type == "cpu"
    assert moved.motor_targets.device.type == "cpu"
    assert moved.motor_loss_mask.device.type == "cpu"


def test_move_trial_preserves_values():
    trial = make_repetition_trial(PHON, MOTOR_SIZE)
    moved = move_trial_to_device(trial, "cpu")
    assert torch.allclose(moved.phon_tensor, trial.phon_tensor)
    assert torch.allclose(moved.motor_targets, trial.motor_targets)
