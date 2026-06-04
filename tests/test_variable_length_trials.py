"""Phase 2c tests for unbatched variable-length trial support.

Verifies that build_trial_inputs and run_trial correctly infer T from
phon_pattern.shape[0] and produce the right number of TickResults for
T = 1, 2, 3, and 5.

Does NOT test: padding, masking, EOS, batching, training, or real data.
All existing toy tests continue to pass (they use T=3 and are unaffected).
"""
from pathlib import Path

import torch

from lichtheim2.config import load_config
from lichtheim2.layers import TickResult, init_state
from lichtheim2.model import Lichtheim2Model
from lichtheim2.tasks import Task

# ---------------------------------------------------------------------------
# Module-level setup — toy config only (variable-length is config-agnostic)
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent.parent
cfg = load_config(REPO_ROOT / "configs" / "toy.yaml")
model = Lichtheim2Model(cfg)
model.eval()

torch.manual_seed(42)
sem = torch.rand(cfg.vATL_size)


def _phon(T: int) -> torch.Tensor:
    """Return a deterministic (T, sound_input_size) phoneme pattern."""
    torch.manual_seed(T)
    return torch.rand(T, cfg.sound_input_size)


# ---------------------------------------------------------------------------
# Repetition tick counts
# ---------------------------------------------------------------------------

def test_repetition_T2_returns_4_results():
    results = model.run_trial(Task.REPETITION, _phon(2), sem, cfg)
    assert len(results) == 4, "Repetition with T=2 must return 2T=4 TickResults"
    assert all(isinstance(r, TickResult) for r in results)


def test_repetition_T5_returns_10_results():
    results = model.run_trial(Task.REPETITION, _phon(5), sem, cfg)
    assert len(results) == 10, "Repetition with T=5 must return 2T=10 TickResults"


# ---------------------------------------------------------------------------
# Comprehension tick counts
# ---------------------------------------------------------------------------

def test_comprehension_T5_returns_5_results():
    results = model.run_trial(Task.COMPREHENSION, _phon(5), sem, cfg)
    assert len(results) == 5, "Comprehension with T=5 must return T=5 TickResults"


# ---------------------------------------------------------------------------
# Speaking tick counts
# ---------------------------------------------------------------------------

def test_speaking_T5_returns_5_results():
    results = model.run_trial(Task.SPEAKING, _phon(5), sem, cfg)
    assert len(results) == 5, "Speaking with T=5 must return T=5 TickResults"


# ---------------------------------------------------------------------------
# Repetition: input vs output phase content
# ---------------------------------------------------------------------------

def test_repetition_output_ticks_silent():
    """Output phase ticks (T..2T-1) must have zero sound input."""
    T = 5
    phon = _phon(T)
    results = model.run_trial(Task.REPETITION, phon, sem, cfg)
    zero = torch.zeros(cfg.sound_input_size)
    for r in results[T:]:
        assert torch.allclose(r.sound_input, zero), \
            f"Tick {r.tick_index}: output-phase sound_input must be zero"


def test_repetition_input_ticks_match_phonemes():
    """Input phase ticks (0..T-1) must carry the correct phoneme pattern."""
    T = 3
    phon = _phon(T)
    results = model.run_trial(Task.REPETITION, phon, sem, cfg)
    for i in range(T):
        assert torch.allclose(results[i].sound_input, phon[i]), \
            f"Tick {i}: sound_input must equal phon[{i}]"


# ---------------------------------------------------------------------------
# Speaking: vATL clamping for all ticks
# ---------------------------------------------------------------------------

def test_speaking_all_ticks_clamped():
    """Every speaking tick must use sem as vATL_input_used."""
    T = 5
    results = model.run_trial(Task.SPEAKING, _phon(T), sem, cfg)
    for r in results:
        assert torch.allclose(r.vATL_input_used, sem), \
            f"Tick {r.tick_index}: vATL_input_used must equal sem_pattern in speaking"


# ---------------------------------------------------------------------------
# Tick indices
# ---------------------------------------------------------------------------

def test_tick_indices_sequential():
    """TickResult.tick_index must be 0, 1, ..., len-1 for each task and T."""
    for T in (2, 5):
        phon = _phon(T)
        for task, expected_len in [
            (Task.REPETITION, 2 * T),
            (Task.COMPREHENSION, T),
            (Task.SPEAKING, T),
        ]:
            results = model.run_trial(task, phon, sem, cfg)
            assert len(results) == expected_len
            for i, r in enumerate(results):
                assert r.tick_index == i, \
                    f"{task.name} T={T}: expected tick_index {i}, got {r.tick_index}"


# ---------------------------------------------------------------------------
# Activations remain in [0, 1] for variable T
# ---------------------------------------------------------------------------

def test_variable_T_activations_in_range():
    """All activations must remain in [0, 1] for T=5 repetition."""
    results = model.run_trial(Task.REPETITION, _phon(5), sem, cfg)
    for r in results:
        for name in ("iSMG", "motor", "mSTG", "aSTG", "vATL_out", "triangularis"):
            t = getattr(r.state, name)
            assert t.min() >= 0.0 and t.max() <= 1.0, \
                f"Tick {r.tick_index}, {name}: activation outside [0, 1]"


# ---------------------------------------------------------------------------
# 1D fallback: phon_pattern of shape (sound_size,) is treated as T=1
# ---------------------------------------------------------------------------

def test_1d_phon_treated_as_T1():
    """A 1-D phon_pattern (sound_size,) must be treated as T=1.

    Repetition: 2 TickResults (1 input + 1 output).
    Comprehension: 1 TickResult.
    Speaking: 1 TickResult.
    """
    phon_1d = torch.rand(cfg.sound_input_size)   # shape (sound_size,), not (T, sound_size)
    rep  = model.run_trial(Task.REPETITION,   phon_1d, sem, cfg)
    comp = model.run_trial(Task.COMPREHENSION, phon_1d, sem, cfg)
    spk  = model.run_trial(Task.SPEAKING,     phon_1d, sem, cfg)
    assert len(rep)  == 2, "1-D phon → T=1 → repetition must return 2 TickResults"
    assert len(comp) == 1, "1-D phon → T=1 → comprehension must return 1 TickResult"
    assert len(spk)  == 1, "1-D phon → T=1 → speaking must return 1 TickResult"
