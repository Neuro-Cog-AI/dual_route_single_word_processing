"""Always-run unit tests for the dorsal-motor-only diagnostic flag (Phase 3j).

Tests that:
  - the default model is behaviorally unchanged when dorsal_motor_only=False;
  - triangularis_to_motor is excluded from motor_net when dorsal_motor_only=True;
  - output shapes are preserved;
  - the flag combines correctly with the dense sound projection;
  - config loading defaults correctly when the field is absent from YAML.

No CSV data required — all tests use synthetic configs and tensors.
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


def _make_cfg(
    sound_proj_size: int | None = None,
    dorsal_motor_only: bool = False,
) -> ModelConfig:
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
        dorsal_motor_only=dorsal_motor_only,
    )


# ---------------------------------------------------------------------------
# Tests: config and model flag storage
# ---------------------------------------------------------------------------


def test_default_dorsal_motor_only_false():
    """ModelConfig defaults to dorsal_motor_only=False when not specified."""
    cfg = _make_cfg()
    assert cfg.dorsal_motor_only is False


def test_model_stores_flag_false():
    """Lichtheim2Model stores dorsal_motor_only=False from default config."""
    model = Lichtheim2Model(_make_cfg(dorsal_motor_only=False))
    assert model.dorsal_motor_only is False


def test_model_stores_flag_true():
    """Lichtheim2Model stores dorsal_motor_only=True when set in config."""
    model = Lichtheim2Model(_make_cfg(dorsal_motor_only=True))
    assert model.dorsal_motor_only is True


# ---------------------------------------------------------------------------
# Tests: motor readout behavior (controlled weights)
# ---------------------------------------------------------------------------


def test_full_model_triangularis_contributes():
    """With zeroed iSMG→motor weights and large triangularis weights, full model
    motor output is clearly different from sigmoid(0) = 0.5."""
    cfg = _make_cfg(dorsal_motor_only=False)
    model = Lichtheim2Model(cfg)

    with torch.no_grad():
        model.iSMG_to_motor.weight.zero_()
        model.iSMG_to_motor.bias.zero_()
        model.triangularis_to_motor.weight.fill_(1.0)

    state = init_state(cfg)
    sound = torch.zeros(cfg.sound_input_size)
    new_state, _ = model.forward_tick(state, sound)

    baseline_0_5 = torch.full((cfg.motor_output_size,), 0.5)
    # triangularis contributes → motor_net ≠ 0 → output ≠ 0.5
    assert not torch.allclose(new_state.motor, baseline_0_5, atol=1e-6)


def test_dorsal_motor_only_triangularis_ignored():
    """With zeroed iSMG→motor weights and large triangularis weights, dorsal-only
    model motor output is exactly sigmoid(0) = 0.5 (triangularis has no effect)."""
    cfg = _make_cfg(dorsal_motor_only=True)
    model = Lichtheim2Model(cfg)

    with torch.no_grad():
        model.iSMG_to_motor.weight.zero_()
        model.iSMG_to_motor.bias.zero_()
        model.triangularis_to_motor.weight.fill_(1.0)

    state = init_state(cfg)
    sound = torch.zeros(cfg.sound_input_size)
    new_state, _ = model.forward_tick(state, sound)

    expected = torch.full((cfg.motor_output_size,), 0.5)
    assert torch.allclose(new_state.motor, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# Tests: output shape preserved
# ---------------------------------------------------------------------------


def test_motor_output_shape_unchanged():
    """Motor output shape is (motor_output_size,) with dorsal_motor_only=True."""
    cfg = _make_cfg(dorsal_motor_only=True)
    model = Lichtheim2Model(cfg)
    state = init_state(cfg)
    sound = torch.zeros(cfg.sound_input_size)
    new_state, _ = model.forward_tick(state, sound)
    assert new_state.motor.shape == (cfg.motor_output_size,)


# ---------------------------------------------------------------------------
# Tests: compatibility with dense projection
# ---------------------------------------------------------------------------


def test_dense_proj_and_dorsal_motor_only_compatible():
    """sound_proj_size=3 and dorsal_motor_only=True can be combined; motor shape correct."""
    cfg = _make_cfg(sound_proj_size=3, dorsal_motor_only=True)
    model = Lichtheim2Model(cfg)
    state = init_state(cfg)
    sound = torch.zeros(cfg.sound_input_size)
    new_state, _ = model.forward_tick(state, sound)
    assert new_state.motor.shape == (cfg.motor_output_size,)


def test_run_trial_tick_count_dorsal_motor_only():
    """run_trial returns 2T TickResults for REPETITION with dorsal_motor_only=True."""
    cfg = _make_cfg(dorsal_motor_only=True)
    model = Lichtheim2Model(cfg)
    T = 3
    phon = torch.zeros(T, cfg.sound_input_size)
    sem  = torch.zeros(cfg.vATL_size)
    results = model.run_trial(Task.REPETITION, phon, sem, cfg)
    assert len(results) == 2 * T


# ---------------------------------------------------------------------------
# Tests: config loading backward compatibility
# ---------------------------------------------------------------------------


def test_config_load_missing_field_defaults_false(tmp_path):
    """YAML without dorsal_motor_only loads as dorsal_motor_only=False (backward compat)."""
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
    p = tmp_path / "test_no_dorsal_flag.yaml"
    p.write_text(yaml.dump(yaml_content))
    cfg = load_config(p)
    assert cfg.dorsal_motor_only is False
