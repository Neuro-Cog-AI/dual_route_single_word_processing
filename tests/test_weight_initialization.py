"""Phase 2b tests for weight initialisation and connectivity conventions.

Tests assert the initialisation scheme implemented in Lichtheim2Model._init_weights().
Source tags indicate the basis for each convention:
  [Paper]    — directly stated in Ueno et al. (2011) Experimental Procedures
  [Inferred] — implementation assumption; not directly stated by the paper

All tests run on both the toy config and the faithful config to ensure both
instantiate correctly after the init change.
"""
from pathlib import Path

import torch
import torch.nn as nn

from lichtheim2.config import load_config
from lichtheim2.model import Lichtheim2Model

# ---------------------------------------------------------------------------
# Module-level setup
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent.parent
toy_model     = Lichtheim2Model(load_config(REPO_ROOT / "configs" / "toy.yaml"))
faithful_model = Lichtheim2Model(load_config(REPO_ROOT / "configs" / "lichtheim2.yaml"))


# ---------------------------------------------------------------------------
# Standard feedforward weights: [−1, 1]  [Paper]
# ---------------------------------------------------------------------------

def test_standard_weights_in_range():
    """Standard feedforward weight tensors must lie in [−1, 1] [Paper]."""
    std_attrs = (
        "sound_to_iSMG", "iSMG_to_motor",
        "sound_to_mSTG", "mSTG_to_aSTG",
        "aSTG_to_vATL", "aSTG_to_triangularis",
        "triangularis_to_motor",
    )
    for model in (toy_model, faithful_model):
        for name in std_attrs:
            w = getattr(model, name).weight
            assert w.min() >= -1.0 and w.max() <= 1.0, \
                f"{name}.weight outside [−1, 1] [Paper]"


# ---------------------------------------------------------------------------
# Elman weight: [−0.5, 0.5]  [Paper]
# ---------------------------------------------------------------------------

def test_elman_weight_in_range():
    """iSMG Elman weight must lie in [−0.5, 0.5] [Paper]."""
    for model in (toy_model, faithful_model):
        w = model.iSMG_elman.weight
        assert w.min() >= -0.5 and w.max() <= 0.5, \
            "iSMG_elman.weight outside [−0.5, 0.5] [Paper]"


# ---------------------------------------------------------------------------
# Copy-back/context weights: [−0.5, 0.5]  [Inferred]
# ---------------------------------------------------------------------------

def test_copy_back_weights_in_range():
    """Copy-back/context weights must lie in [−0.5, 0.5].

    [Inferred] — the paper states 'recurrent connections' use [−0.5, 0.5] but
    does not explicitly define whether copy-back connections (motor_copy_to_iSMG,
    vATL_in_to_aSTG) qualify. We treat them as recurrent by assumption.
    """
    ctx_attrs = ("motor_copy_to_iSMG", "vATL_in_to_aSTG")
    for model in (toy_model, faithful_model):
        for name in ctx_attrs:
            w = getattr(model, name).weight
            assert w.min() >= -0.5 and w.max() <= 0.5, \
                f"{name}.weight outside [−0.5, 0.5] [Inferred]"


# ---------------------------------------------------------------------------
# Additive bias: −1.0  [Inferred]
# ---------------------------------------------------------------------------

def test_bias_initialized_to_minus_one():
    """All nn.Linear.bias values that are not None must equal −1.0.

    [Inferred] — this is our PyTorch approximation of the LENS bias-link
    convention. The paper states LENS bias-link weights are initialised to
    suppress early hidden-unit activation, but the mapping to nn.Linear.bias
    is an implementation assumption, not a direct paper quote.
    """
    for model in (toy_model, faithful_model):
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and module.bias is not None:
                assert torch.all(module.bias == -1.0), \
                    f"{name}.bias should be −1.0 [Inferred PyTorch convention]"


# ---------------------------------------------------------------------------
# No bias on copy and Elman layers  [Paper]
# ---------------------------------------------------------------------------

def test_no_bias_on_context_layers():
    """Elman and copy layers must have no trainable bias link [Paper]."""
    no_bias_attrs = ("iSMG_elman", "motor_copy_to_iSMG", "vATL_in_to_aSTG")
    for model in (toy_model, faithful_model):
        for name in no_bias_attrs:
            module = getattr(model, name)
            assert module.bias is None, \
                f"{name} must have bias=None [Paper]"


# ---------------------------------------------------------------------------
# Connectivity completeness  [Supp Fig S1]
# ---------------------------------------------------------------------------

def test_connectivity_completeness():
    """All 10 expected weight matrices must exist on the model [Supp Fig S1]."""
    expected = (
        "sound_to_iSMG", "iSMG_elman", "motor_copy_to_iSMG", "iSMG_to_motor",
        "sound_to_mSTG", "mSTG_to_aSTG", "vATL_in_to_aSTG",
        "aSTG_to_vATL", "aSTG_to_triangularis", "triangularis_to_motor",
    )
    for model in (toy_model, faithful_model):
        for attr in expected:
            assert hasattr(model, attr), f"Missing connection: {attr}"
            assert isinstance(getattr(model, attr), nn.Linear), \
                f"{attr} must be an nn.Linear"


# ---------------------------------------------------------------------------
# Both configs still instantiate  (regression)
# ---------------------------------------------------------------------------

def test_both_configs_instantiate():
    """Toy and faithful configs must both instantiate after the init change."""
    from lichtheim2.config import load_config
    toy      = Lichtheim2Model(load_config(REPO_ROOT / "configs" / "toy.yaml"))
    faithful = Lichtheim2Model(load_config(REPO_ROOT / "configs" / "lichtheim2.yaml"))
    assert isinstance(toy, Lichtheim2Model)
    assert isinstance(faithful, Lichtheim2Model)