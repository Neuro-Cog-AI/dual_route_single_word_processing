"""Phase 1 tests for toy tick-by-tick dynamics.

Uses configs/toy.yaml throughout. A module-level cfg and model are
constructed once; individual tests are stateless (no shared mutable state).
"""
from dataclasses import replace as dc_replace
from pathlib import Path

import torch

from lichtheim2.config import load_config
from lichtheim2.layers import ModelState, TickResult, init_state
from lichtheim2.model import Lichtheim2Model
from lichtheim2.tasks import Task

# ---------------------------------------------------------------------------
# Module-level setup
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent.parent
cfg = load_config(REPO_ROOT / "configs" / "toy.yaml")
model = Lichtheim2Model(cfg)
model.eval()

# Deterministic synthetic patterns for all tests
torch.manual_seed(0)
phon = torch.rand(3, cfg.sound_input_size)   # (n_morae, sound_size)
sem  = torch.rand(cfg.vATL_size)             # (vATL_size,)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def test_config_loads():
    c = load_config(REPO_ROOT / "configs" / "toy.yaml")
    for field in (
        "sound_input_size", "motor_output_size", "iSMG_hidden_size",
        "mSTG_hidden_size", "aSTG_hidden_size", "triangularis_hidden_size",
        "vATL_size",
    ):
        assert getattr(c, field) > 0, f"{field} must be a positive integer"


# ---------------------------------------------------------------------------
# ModelState initialisation
# ---------------------------------------------------------------------------

def test_init_state_shapes():
    s = init_state(cfg)
    assert s.iSMG.shape          == (cfg.iSMG_hidden_size,)
    assert s.iSMG_context.shape  == (cfg.iSMG_hidden_size,)
    assert s.motor.shape         == (cfg.motor_output_size,)
    assert s.motor_context.shape == (cfg.motor_output_size,)
    assert s.mSTG.shape          == (cfg.mSTG_hidden_size,)
    assert s.aSTG.shape          == (cfg.aSTG_hidden_size,)
    assert s.vATL_out.shape      == (cfg.vATL_size,)
    assert s.vATL_context.shape  == (cfg.vATL_size,)
    assert s.triangularis.shape  == (cfg.triangularis_hidden_size,)


def test_init_state_values():
    s = init_state(cfg)
    half_fields = ("iSMG", "iSMG_context", "mSTG", "aSTG",
                   "vATL_out", "vATL_context", "triangularis")
    for name in half_fields:
        t = getattr(s, name)
        assert torch.allclose(t, torch.full_like(t, 0.5)), \
            f"{name} should initialise to 0.5 [Paper]"
    assert torch.allclose(s.motor, torch.zeros_like(s.motor)), \
        "motor should initialise to 0 [Paper]"
    assert torch.allclose(s.motor_context, torch.zeros_like(s.motor_context)), \
        "motor_context should initialise to 0 [Paper]"


# ---------------------------------------------------------------------------
# forward_tick — return type and shapes
# ---------------------------------------------------------------------------

def test_forward_tick_returns_state_and_vATL_used():
    state = init_state(cfg)
    result = model.forward_tick(state, sound=None, clamp_vATL_in=None)
    assert isinstance(result, tuple) and len(result) == 2
    new_state, vATL_used = result
    assert isinstance(new_state, ModelState)
    assert isinstance(vATL_used, torch.Tensor)


def test_forward_tick_shapes():
    state = init_state(cfg)
    new_state, vATL_used = model.forward_tick(state, sound=phon[0])
    assert new_state.iSMG.shape          == (cfg.iSMG_hidden_size,)
    assert new_state.motor.shape         == (cfg.motor_output_size,)
    assert new_state.mSTG.shape          == (cfg.mSTG_hidden_size,)
    assert new_state.aSTG.shape          == (cfg.aSTG_hidden_size,)
    assert new_state.vATL_out.shape      == (cfg.vATL_size,)
    assert new_state.vATL_context.shape  == (cfg.vATL_size,)
    assert new_state.triangularis.shape  == (cfg.triangularis_hidden_size,)
    assert vATL_used.shape               == (cfg.vATL_size,)


def test_activations_in_range():
    state = init_state(cfg)
    new_state, _ = model.forward_tick(state, sound=phon[0])
    for name in ("iSMG", "motor", "mSTG", "aSTG", "vATL_out", "triangularis"):
        t = getattr(new_state, name)
        assert t.min() >= 0.0 and t.max() <= 1.0, \
            f"{name} activation outside [0, 1]"


# ---------------------------------------------------------------------------
# Copy-back mechanics
# ---------------------------------------------------------------------------

def test_iSMG_elman_updates():
    state = init_state(cfg)
    new_state, _ = model.forward_tick(state, sound=phon[0])
    assert torch.allclose(new_state.iSMG_context, new_state.iSMG), \
        "iSMG_context must equal iSMG at end of tick (Elman copy)"


def test_motor_copy_updates():
    state = init_state(cfg)
    new_state, _ = model.forward_tick(state, sound=phon[0])
    assert torch.allclose(new_state.motor_context, new_state.motor), \
        "motor_context must equal motor at end of tick"


def test_vATL_context_updated():
    state = init_state(cfg)
    new_state, _ = model.forward_tick(state, sound=phon[0], clamp_vATL_in=None)
    assert torch.allclose(new_state.vATL_context, new_state.vATL_out), \
        "vATL_context must equal vATL_out at end of tick (prepared for next tick)"


def test_vATL_context_used_as_input():
    state = init_state(cfg)
    _, vATL_used = model.forward_tick(state, sound=None, clamp_vATL_in=None)
    assert torch.allclose(vATL_used, state.vATL_context), \
        "When no clamp given, vATL_input_used must equal state.vATL_context"


def test_clamp_overrides_vATL_context():
    state = init_state(cfg)
    # Use zeros as clamp — clearly distinct from the 0.5 initial vATL_context
    clamp = torch.zeros(cfg.vATL_size)
    _, vATL_used = model.forward_tick(state, sound=None, clamp_vATL_in=clamp)
    assert torch.allclose(vATL_used, clamp), \
        "vATL_input_used must equal the clamped tensor"
    assert not torch.allclose(vATL_used, state.vATL_context), \
        "Clamped value must differ from vATL_context (zeros vs 0.5 init)"


# ---------------------------------------------------------------------------
# run_trial — trial length and TickResult contents
# ---------------------------------------------------------------------------

def test_repetition_returns_6_results():
    results = model.run_trial(Task.REPETITION, phon, sem, cfg)
    assert len(results) == 6
    assert all(isinstance(r, TickResult) for r in results)


def test_comprehension_returns_3_results():
    results = model.run_trial(Task.COMPREHENSION, phon, sem, cfg)
    assert len(results) == 3
    assert all(isinstance(r, TickResult) for r in results)


def test_speaking_returns_3_results():
    results = model.run_trial(Task.SPEAKING, phon, sem, cfg)
    assert len(results) == 3
    assert all(isinstance(r, TickResult) for r in results)


def test_speaking_vATL_input_is_clamped():
    results = model.run_trial(Task.SPEAKING, phon, sem, cfg)
    for r in results:
        assert torch.allclose(r.vATL_input_used, sem), \
            f"Tick {r.tick_index}: vATL_input_used must equal sem_pattern in speaking"


def test_repetition_sound_zeros_at_output_ticks():
    results = model.run_trial(Task.REPETITION, phon, sem, cfg)
    zero = torch.zeros(cfg.sound_input_size)
    for r in results[3:]:  # ticks 3-5 (output phase)
        assert torch.allclose(r.sound_input, zero), \
            f"Tick {r.tick_index}: sound_input must be zero during output phase"


def test_tick_indices_correct():
    for task in (Task.REPETITION, Task.COMPREHENSION, Task.SPEAKING):
        results = model.run_trial(task, phon, sem, cfg)
        for expected, r in enumerate(results):
            assert r.tick_index == expected, \
                f"{task.name}: expected tick_index {expected}, got {r.tick_index}"


def test_carry_chain_influences_computation():
    """iSMG_context must actually influence next-tick iSMG via the Elman connection.

    Two states that are identical in every field except iSMG_context must
    produce different iSMG activations when given the same sound input.
    This is deterministic: state fields and sound are explicitly set (no random data).
    """
    base = init_state(cfg)
    # state_a: iSMG_context = 0.5 (from init_state)
    state_a = base
    # state_b: iSMG_context = 0.0 — only this field differs
    state_b = dc_replace(base, iSMG_context=torch.zeros(cfg.iSMG_hidden_size))

    sound = torch.full((cfg.sound_input_size,), 0.3)  # fixed non-trivial input

    result_a, _ = model.forward_tick(state_a, sound=sound)
    result_b, _ = model.forward_tick(state_b, sound=sound)

    assert not torch.allclose(result_a.iSMG, result_b.iSMG), (
        "iSMG_context must influence iSMG computation via the Elman connection; "
        "states differing only in iSMG_context must produce different iSMG outputs"
    )
