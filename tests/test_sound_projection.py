"""Always-run unit and shape tests for the optional sound input projection.

Phase 3h: tests that the optional dense projection in Lichtheim2Model
behaves correctly for both the baseline (no projection) and projection cases.
No private CSV data required — all tests use synthetic configs and tensors.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lichtheim2.config import ModelConfig
from lichtheim2.layers import init_state
from lichtheim2.model import Lichtheim2Model
from lichtheim2.tasks import Task


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_cfg(sound_proj_size: int | None = None) -> ModelConfig:
    """Tiny synthetic config for fast testing."""
    return ModelConfig(
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
        sound_proj_size=sound_proj_size,
    )


# ---------------------------------------------------------------------------
# Tests: baseline (no projection)
# ---------------------------------------------------------------------------


def test_no_projection_sound_proj_is_none():
    """Baseline: model.sound_proj is None when sound_proj_size=None."""
    model = Lichtheim2Model(_make_cfg(sound_proj_size=None))
    assert model.sound_proj is None


def test_no_projection_iSMG_takes_raw_sound():
    """Baseline: sound_to_iSMG in_features == sound_input_size."""
    cfg = _make_cfg(sound_proj_size=None)
    model = Lichtheim2Model(cfg)
    assert model.sound_to_iSMG.in_features == cfg.sound_input_size


def test_no_projection_mSTG_takes_raw_sound():
    """Baseline: sound_to_mSTG in_features == sound_input_size."""
    cfg = _make_cfg(sound_proj_size=None)
    model = Lichtheim2Model(cfg)
    assert model.sound_to_mSTG.in_features == cfg.sound_input_size


# ---------------------------------------------------------------------------
# Tests: projection enabled
# ---------------------------------------------------------------------------


def test_projection_reduces_effective_input_iSMG():
    """With projection: sound_to_iSMG in_features == sound_proj_size."""
    cfg = _make_cfg(sound_proj_size=3)
    model = Lichtheim2Model(cfg)
    assert model.sound_to_iSMG.in_features == 3


def test_projection_reduces_effective_input_mSTG():
    """With projection: sound_to_mSTG in_features == sound_proj_size."""
    cfg = _make_cfg(sound_proj_size=3)
    model = Lichtheim2Model(cfg)
    assert model.sound_to_mSTG.in_features == 3


def test_projection_bias_false():
    """sound_proj must have no bias (bias=False)."""
    model = Lichtheim2Model(_make_cfg(sound_proj_size=3))
    assert model.sound_proj is not None
    assert model.sound_proj.bias is None


def test_raw_sound_size_stored_not_proj_size():
    """model._sound_input_size is always the raw phoneme dim, not the projected dim."""
    cfg = _make_cfg(sound_proj_size=3)
    model = Lichtheim2Model(cfg)
    assert model._sound_input_size == cfg.sound_input_size  # 5, not 3


def test_zero_sound_zero_projection():
    """Zero phoneme input → zero projected output (bias=False invariant)."""
    cfg = _make_cfg(sound_proj_size=3)
    model = Lichtheim2Model(cfg)
    zero_sound = torch.zeros(cfg.sound_input_size)
    projected = model.sound_proj(zero_sound)
    assert (projected == 0.0).all()


def test_projection_weight_init_in_range():
    """sound_proj.weight initialised with uniform(−1, 1) [feedforward convention]."""
    cfg = _make_cfg(sound_proj_size=3)
    model = Lichtheim2Model(cfg)
    w = model.sound_proj.weight
    assert w.min().item() >= -1.0 and w.max().item() <= 1.0


# ---------------------------------------------------------------------------
# Tests: forward_tick shapes
# ---------------------------------------------------------------------------


def test_forward_tick_motor_shape_no_projection():
    """Without projection: motor output shape == (motor_output_size,)."""
    cfg = _make_cfg(sound_proj_size=None)
    model = Lichtheim2Model(cfg)
    state = init_state(cfg)
    sound = torch.zeros(cfg.sound_input_size)
    new_state, _ = model.forward_tick(state, sound)
    assert new_state.motor.shape == (cfg.motor_output_size,)


def test_forward_tick_motor_shape_with_projection():
    """With projection: motor output is still (motor_output_size,), not (sound_proj_size,)."""
    cfg = _make_cfg(sound_proj_size=3)
    model = Lichtheim2Model(cfg)
    state = init_state(cfg)
    sound = torch.zeros(cfg.sound_input_size)
    new_state, _ = model.forward_tick(state, sound)
    assert new_state.motor.shape == (cfg.motor_output_size,)


def test_forward_tick_shape_guard_wrong_dim():
    """forward_tick raises ValueError if sound has wrong last dimension."""
    cfg = _make_cfg(sound_proj_size=3)
    model = Lichtheim2Model(cfg)
    state = init_state(cfg)
    wrong_sound = torch.zeros(3)   # projected dim, not raw dim
    with pytest.raises(ValueError, match="sound_input_size"):
        model.forward_tick(state, wrong_sound)


# ---------------------------------------------------------------------------
# Tests: run_trial end-to-end
# ---------------------------------------------------------------------------


def test_run_trial_with_projection_tick_count():
    """run_trial with projection returns 2T TickResults for REPETITION."""
    cfg = _make_cfg(sound_proj_size=3)
    model = Lichtheim2Model(cfg)
    T = 2
    phon = torch.zeros(T, cfg.sound_input_size)
    sem  = torch.zeros(cfg.vATL_size)
    results = model.run_trial(Task.REPETITION, phon, sem, cfg)
    assert len(results) == 2 * T


def test_run_trial_no_projection_baseline_tick_count():
    """run_trial without projection returns 2T TickResults for REPETITION."""
    cfg = _make_cfg(sound_proj_size=None)
    model = Lichtheim2Model(cfg)
    T = 3
    phon = torch.zeros(T, cfg.sound_input_size)
    sem  = torch.zeros(cfg.vATL_size)
    results = model.run_trial(Task.REPETITION, phon, sem, cfg)
    assert len(results) == 2 * T


def test_tick_result_sound_input_is_raw_dim():
    """TickResult.sound_input always records the raw 39D phoneme, not the projected dim."""
    cfg = _make_cfg(sound_proj_size=3)
    model = Lichtheim2Model(cfg)
    phon = torch.zeros(2, cfg.sound_input_size)
    sem  = torch.zeros(cfg.vATL_size)
    results = model.run_trial(Task.REPETITION, phon, sem, cfg)
    for r in results:
        assert r.sound_input.shape == (cfg.sound_input_size,), (
            f"tick {r.tick_index}: expected ({cfg.sound_input_size},), "
            f"got {r.sound_input.shape}"
        )


# ---------------------------------------------------------------------------
# Tests: config loading backward compatibility
# ---------------------------------------------------------------------------


def test_config_load_missing_sound_proj_size_defaults_none(tmp_path):
    """Existing YAML files without sound_proj_size load as sound_proj_size=None."""
    import yaml
    from lichtheim2.config import load_config

    yaml_content = {
        "model": {
            "sound_input_size": 5,
            "motor_output_size": 5,
            "iSMG_hidden_size": 4,
            "mSTG_hidden_size": 6,
            "aSTG_hidden_size": 8,
            "triangularis_hidden_size": 4,
            "vATL_size": 3,
        },
        "tasks": {
            "repetition_ticks": 4,
            "comprehension_ticks": 2,
            "speaking_ticks": 2,
        },
    }
    p = tmp_path / "test.yaml"
    p.write_text(yaml.dump(yaml_content))
    cfg = load_config(p)
    assert cfg.sound_proj_size is None


def test_config_load_sound_proj_size_from_yaml(tmp_path):
    """YAML with sound_proj_size: 20 is correctly loaded."""
    import yaml
    from lichtheim2.config import load_config

    yaml_content = {
        "model": {
            "sound_input_size": 5,
            "motor_output_size": 5,
            "iSMG_hidden_size": 4,
            "mSTG_hidden_size": 6,
            "aSTG_hidden_size": 8,
            "triangularis_hidden_size": 4,
            "vATL_size": 3,
            "sound_proj_size": 20,
        },
        "tasks": {
            "repetition_ticks": 4,
            "comprehension_ticks": 2,
            "speaking_ticks": 2,
        },
    }
    p = tmp_path / "test_proj.yaml"
    p.write_text(yaml.dump(yaml_content))
    cfg = load_config(p)
    assert cfg.sound_proj_size == 20
