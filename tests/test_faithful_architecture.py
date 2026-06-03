"""Phase 2a tests for the faithful Lichtheim 2 architecture skeleton.

Validates that Lichtheim2Model instantiates with the real layer sizes from
Ueno et al. (2011) and that forward_tick / run_trial produce correct shapes
and dynamics at faithful scale.

Does NOT test: faithful weight initialisation, loss, training, data encoding,
or lesioning. Those are later phases.

Existing toy tests (tests/test_tick_dynamics.py) run unchanged.
"""
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
faithful_cfg = load_config(REPO_ROOT / "configs" / "lichtheim2.yaml")
faithful_model = Lichtheim2Model(faithful_cfg)
faithful_model.eval()

# Dummy patterns sized to faithful dimensions (not linguistically meaningful)
torch.manual_seed(1)
phon = torch.rand(3, faithful_cfg.sound_input_size)  # (3, 21)
sem  = torch.rand(faithful_cfg.vATL_size)            # (50,)


# ---------------------------------------------------------------------------
# Config: verify exact paper values
# ---------------------------------------------------------------------------

def test_faithful_config_loads():
    cfg = load_config(REPO_ROOT / "configs" / "lichtheim2.yaml")
    assert cfg.sound_input_size          == 21,  "sound_input_size must be 21 [Paper]"
    assert cfg.motor_output_size         == 21,  "motor_output_size must be 21 [Paper]"
    assert cfg.iSMG_hidden_size          == 50,  "iSMG_hidden_size must be 50 [Paper]"
    assert cfg.mSTG_hidden_size          == 200, "mSTG_hidden_size must be 200 [Paper]"
    assert cfg.aSTG_hidden_size          == 650, "aSTG_hidden_size must be 650 [Paper]"
    assert cfg.triangularis_hidden_size  == 200, "triangularis_hidden_size must be 200 [Paper]"
    assert cfg.vATL_size                 == 50,  "vATL_size must be 50 [Paper]"
    assert cfg.repetition_ticks          == 6
    assert cfg.comprehension_ticks       == 3
    assert cfg.speaking_ticks            == 3


# ---------------------------------------------------------------------------
# Instantiation
# ---------------------------------------------------------------------------

def test_faithful_model_instantiates():
    model = Lichtheim2Model(faithful_cfg)
    assert isinstance(model, Lichtheim2Model)


# ---------------------------------------------------------------------------
# ModelState initialisation
# ---------------------------------------------------------------------------

def test_faithful_init_state_shapes():
    s = init_state(faithful_cfg)
    assert s.iSMG.shape          == (50,)
    assert s.iSMG_context.shape  == (50,)
    assert s.motor.shape         == (21,)
    assert s.motor_context.shape == (21,)
    assert s.mSTG.shape          == (200,)
    assert s.aSTG.shape          == (650,)
    assert s.vATL_out.shape      == (50,)
    assert s.vATL_context.shape  == (50,)
    assert s.triangularis.shape  == (200,)


# ---------------------------------------------------------------------------
# forward_tick
# ---------------------------------------------------------------------------

def test_faithful_forward_tick_shapes():
    state = init_state(faithful_cfg)
    new_state, vATL_used = faithful_model.forward_tick(state, sound=phon[0])
    assert new_state.iSMG.shape          == (50,)
    assert new_state.motor.shape         == (21,)
    assert new_state.mSTG.shape          == (200,)
    assert new_state.aSTG.shape          == (650,)
    assert new_state.vATL_out.shape      == (50,)
    assert new_state.vATL_context.shape  == (50,)
    assert new_state.triangularis.shape  == (200,)
    assert vATL_used.shape               == (50,)


def test_faithful_activations_in_range():
    state = init_state(faithful_cfg)
    new_state, _ = faithful_model.forward_tick(state, sound=phon[0])
    for name in ("iSMG", "motor", "mSTG", "aSTG", "vATL_out", "triangularis"):
        t = getattr(new_state, name)
        assert t.min() >= 0.0 and t.max() <= 1.0, \
            f"{name} outside [0, 1] with faithful config"


# ---------------------------------------------------------------------------
# Copy-back mechanics
# ---------------------------------------------------------------------------

def test_faithful_iSMG_elman_updates():
    state = init_state(faithful_cfg)
    new_state, _ = faithful_model.forward_tick(state, sound=phon[0])
    assert torch.allclose(new_state.iSMG_context, new_state.iSMG), \
        "iSMG_context must equal iSMG at end of tick (Elman copy)"


def test_faithful_vATL_context_updated():
    state = init_state(faithful_cfg)
    new_state, _ = faithful_model.forward_tick(state, sound=phon[0])
    assert torch.allclose(new_state.vATL_context, new_state.vATL_out), \
        "vATL_context must equal vATL_out at end of tick"


# ---------------------------------------------------------------------------
# run_trial — tick counts and TickResult types
# ---------------------------------------------------------------------------

def test_faithful_repetition_returns_6_results():
    results = faithful_model.run_trial(Task.REPETITION, phon, sem, faithful_cfg)
    assert len(results) == 6
    assert all(isinstance(r, TickResult) for r in results)


def test_faithful_comprehension_returns_3_results():
    results = faithful_model.run_trial(Task.COMPREHENSION, phon, sem, faithful_cfg)
    assert len(results) == 3
    assert all(isinstance(r, TickResult) for r in results)


def test_faithful_speaking_returns_3_results():
    results = faithful_model.run_trial(Task.SPEAKING, phon, sem, faithful_cfg)
    assert len(results) == 3
    assert all(isinstance(r, TickResult) for r in results)


def test_faithful_speaking_vATL_clamped():
    results = faithful_model.run_trial(Task.SPEAKING, phon, sem, faithful_cfg)
    for r in results:
        assert torch.allclose(r.vATL_input_used, sem), \
            f"Tick {r.tick_index}: vATL_input_used must equal sem_pattern in speaking"
