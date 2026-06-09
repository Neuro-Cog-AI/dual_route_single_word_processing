"""Phase 3c-5 tests: loss normalization and task-balance audit.

Structure:
- Always-run (no CSV required): toy config + synthetic tensors.
- CSV-dependent: skip gracefully per specific file.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lichtheim2.config import load_config
from lichtheim2.losses import LossBreakdown, compute_trial_loss, compute_trial_loss_breakdown

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from audit_loss_scaling import validate_args
from lichtheim2.model import Lichtheim2Model
from lichtheim2.tasks import Task
from lichtheim2.trials import (
    make_comprehension_trial,
    make_repetition_trial,
    make_speaking_trial,
)

# ---------------------------------------------------------------------------
# Shared setup — toy config, small synthetic tensors
# ---------------------------------------------------------------------------

REPO_ROOT    = Path(__file__).parent.parent
DATA_DIR     = REPO_ROOT / "data" / "raw" / "nwr_swp"
PHONEMES_CSV = DATA_DIR / "phonemes.csv"
WFE_CSV      = DATA_DIR / "wfe.csv"

cfg        = load_config(REPO_ROOT / "configs" / "toy.yaml")  # sound=5, motor=5, vATL=6
T          = 3
SOUND_SIZE = cfg.sound_input_size   # 5
MOTOR_SIZE = cfg.motor_output_size  # 5
VATL_SIZE  = cfg.vATL_size          # 6

torch.manual_seed(0)
PHON = torch.rand(T, SOUND_SIZE)
SEM  = torch.rand(VATL_SIZE)


def _fresh_model() -> Lichtheim2Model:
    torch.manual_seed(0)
    return Lichtheim2Model(cfg)


def _run(model: Lichtheim2Model, trial):
    """Run model.run_trial() matching the trainer's sem_for_run logic."""
    sem = trial.sem_input if trial.sem_input is not None else torch.zeros(VATL_SIZE)
    return model.run_trial(trial.task, trial.phon_tensor, sem, cfg)


# ---------------------------------------------------------------------------
# Parity: compute_trial_loss_breakdown().total_loss == compute_trial_loss()
# Tests that the refactor does not change training behavior.
# ---------------------------------------------------------------------------


def test_parity_repetition():
    model   = _fresh_model()
    trial   = make_repetition_trial(PHON, MOTOR_SIZE)
    results = _run(model, trial)
    assert torch.isclose(
        compute_trial_loss_breakdown(results, trial).total_loss,
        compute_trial_loss(results, trial),
    )


def test_parity_comprehension():
    model   = _fresh_model()
    trial   = make_comprehension_trial(PHON, SEM, MOTOR_SIZE)
    results = _run(model, trial)
    assert torch.isclose(
        compute_trial_loss_breakdown(results, trial).total_loss,
        compute_trial_loss(results, trial),
    )


def test_parity_speaking():
    model   = _fresh_model()
    trial   = make_speaking_trial(PHON, SEM, MOTOR_SIZE)
    results = _run(model, trial)
    assert torch.isclose(
        compute_trial_loss_breakdown(results, trial).total_loss,
        compute_trial_loss(results, trial),
    )


def test_parity_with_zero_error_radius():
    """Parity must hold with a non-zero dead zone (zero_error_radius > 0)."""
    model   = _fresh_model()
    trial   = make_comprehension_trial(PHON, SEM, MOTOR_SIZE)
    results = _run(model, trial)
    assert torch.isclose(
        compute_trial_loss_breakdown(results, trial, zero_error_radius=0.1).total_loss,
        compute_trial_loss(results, trial, zero_error_radius=0.1),
    )


# ---------------------------------------------------------------------------
# Structural consistency
# ---------------------------------------------------------------------------


def test_motor_plus_semantic_equals_total():
    """motor_loss + semantic_loss == total_loss (comprehension)."""
    model   = _fresh_model()
    trial   = make_comprehension_trial(PHON, SEM, MOTOR_SIZE)
    results = _run(model, trial)
    bd      = compute_trial_loss_breakdown(results, trial)
    assert torch.isclose(bd.motor_loss + bd.semantic_loss, bd.total_loss)


# ---------------------------------------------------------------------------
# Active element counts (no dead zone)
# ---------------------------------------------------------------------------


def test_n_active_motor_no_radius_repetition():
    """Repetition: all 2T ticks masked × motor_size units = n_active_motor."""
    model   = _fresh_model()
    trial   = make_repetition_trial(PHON, MOTOR_SIZE)
    results = _run(model, trial)
    bd      = compute_trial_loss_breakdown(results, trial, zero_error_radius=0.0)
    assert bd.n_active_motor   == 2 * T * MOTOR_SIZE
    assert bd.n_active_semantic == 0


def test_n_active_semantic_no_radius_comprehension():
    """Comprehension: all T ticks masked × vATL_size units = n_active_semantic."""
    model   = _fresh_model()
    trial   = make_comprehension_trial(PHON, SEM, MOTOR_SIZE)
    results = _run(model, trial)
    bd      = compute_trial_loss_breakdown(results, trial, zero_error_radius=0.0)
    assert bd.n_active_semantic == T * VATL_SIZE
    assert bd.n_active_motor    == T * MOTOR_SIZE


# ---------------------------------------------------------------------------
# No semantic contribution for REP / SPK
# ---------------------------------------------------------------------------


def test_no_semantic_repetition():
    model   = _fresh_model()
    trial   = make_repetition_trial(PHON, MOTOR_SIZE)
    results = _run(model, trial)
    bd      = compute_trial_loss_breakdown(results, trial)
    assert bd.n_active_semantic == 0
    assert bd.semantic_loss.item() == 0.0
    assert math.isnan(bd.semantic_loss_per_unit)


def test_no_semantic_speaking():
    model   = _fresh_model()
    trial   = make_speaking_trial(PHON, SEM, MOTOR_SIZE)
    results = _run(model, trial)
    bd      = compute_trial_loss_breakdown(results, trial)
    assert bd.n_active_semantic == 0
    assert bd.semantic_loss.item() == 0.0


# ---------------------------------------------------------------------------
# Dead zone reduces active count and total loss
# ---------------------------------------------------------------------------


def test_radius_reduces_active_motor_count():
    """n_active_motor with dead zone <= n_active_motor without dead zone."""
    model   = _fresh_model()
    trial   = make_repetition_trial(PHON, MOTOR_SIZE)
    results = _run(model, trial)
    bd_no   = compute_trial_loss_breakdown(results, trial, zero_error_radius=0.0)
    bd_yes  = compute_trial_loss_breakdown(results, trial, zero_error_radius=0.1)
    assert bd_yes.n_active_motor <= bd_no.n_active_motor


def test_radius_reduces_total_loss():
    """total_loss with dead zone <= total_loss without dead zone."""
    model   = _fresh_model()
    trial   = make_repetition_trial(PHON, MOTOR_SIZE)
    results = _run(model, trial)
    bd_no   = compute_trial_loss_breakdown(results, trial, zero_error_radius=0.0)
    bd_yes  = compute_trial_loss_breakdown(results, trial, zero_error_radius=0.1)
    assert bd_yes.total_loss.item() <= bd_no.total_loss.item()


# ---------------------------------------------------------------------------
# Normalized per-unit losses
# ---------------------------------------------------------------------------


def test_normalized_motor_finite_repetition():
    model   = _fresh_model()
    trial   = make_repetition_trial(PHON, MOTOR_SIZE)
    results = _run(model, trial)
    bd      = compute_trial_loss_breakdown(results, trial)
    assert math.isfinite(bd.motor_loss_per_unit)


def test_normalized_semantic_finite_comprehension():
    model   = _fresh_model()
    trial   = make_comprehension_trial(PHON, SEM, MOTOR_SIZE)
    results = _run(model, trial)
    bd      = compute_trial_loss_breakdown(results, trial)
    assert math.isfinite(bd.semantic_loss_per_unit)


# ---------------------------------------------------------------------------
# validate_args
# ---------------------------------------------------------------------------


def _make_audit_args(**overrides):
    import argparse
    base = argparse.Namespace(max_words=5, zero_error_radius=0.0)
    for k, v in overrides.items():
        setattr(base, k, v)
    return base


def test_validate_args_valid():
    assert validate_args(_make_audit_args()) is None


@pytest.mark.parametrize("field,value", [
    ("max_words",        0),
    ("max_words",       -1),
    ("zero_error_radius", -0.1),
])
def test_validate_args_invalid(field, value):
    err = validate_args(_make_audit_args(**{field: value}))
    assert err is not None
    assert isinstance(err, str)


# ---------------------------------------------------------------------------
# CSV-dependent integration test
# ---------------------------------------------------------------------------

_wfe_available = pytest.mark.skipif(
    not (PHONEMES_CSV.exists() and WFE_CSV.exists()),
    reason="private CSV data (phonemes.csv + wfe.csv) not present",
)


@_wfe_available
def test_csv_breakdown_by_task_finite():
    """3 real words × 3 tasks: all breakdown fields finite; comp has semantic units."""
    from lichtheim2.config import load_config as _lc
    from lichtheim2.data import load_word_items
    from lichtheim2.encoding import load_phoneme_inventory
    from lichtheim2.semantics import assign_artificial_semantics

    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from train_multitask_real_data import build_word_trials

    eng_cfg   = _lc(REPO_ROOT / "configs" / "english_nwr.yaml")
    inventory = load_phoneme_inventory(PHONEMES_CSV)
    words_all = load_word_items(WFE_CSV, inventory)
    sem_map   = assign_artificial_semantics(words_all, vATL_size=eng_cfg.vATL_size, seed=0)
    words     = words_all[:3]

    trials    = build_word_trials(words, sem_map, eng_cfg.motor_output_size)
    assert len(trials) == 9  # 3 words × 3 tasks

    eng_model = Lichtheim2Model(eng_cfg)

    for trial in trials:
        sem     = trial.sem_input if trial.sem_input is not None else torch.zeros(eng_cfg.vATL_size)
        results = eng_model.run_trial(trial.task, trial.phon_tensor, sem, eng_cfg)
        bd      = compute_trial_loss_breakdown(results, trial)
        assert math.isfinite(bd.total_loss.item())
        assert math.isfinite(bd.motor_loss_per_unit)
        if trial.task == Task.COMPREHENSION:
            assert bd.n_active_semantic > 0
            assert math.isfinite(bd.semantic_loss_per_unit)
