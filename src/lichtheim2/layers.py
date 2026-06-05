from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from lichtheim2.config import ModelConfig
    from lichtheim2.tasks import Task


@dataclass
class ModelState:
    """Between-tick carry state for the Lichtheim 2 model.

    Holds the per-layer activation tensors and the copy-back context
    fields required as inputs at the next tick. This is NOT a record of
    what inputs were used during the tick that produced this state —
    see TickResult for that.

    [Paper] init: hidden layers to 0.5, insular-motor output to 0.
    """

    iSMG: torch.Tensor           # current iSMG activation  (iSMG_size,)
    iSMG_context: torch.Tensor   # Elman copy; = iSMG at end of tick; input next tick
    motor: torch.Tensor          # current motor activation  (motor_size,)
    motor_context: torch.Tensor  # copy of motor; = motor at end of tick; input next tick
    mSTG: torch.Tensor           # current mSTG activation   (mSTG_size,)
    aSTG: torch.Tensor           # current aSTG activation   (aSTG_size,)
    vATL_out: torch.Tensor       # vATL output computed this tick  (vATL_size,)
    vATL_context: torch.Tensor   # copy of vATL_out; = vATL_out at end of tick;
                                 # used as vATL input at the next tick unless
                                 # the task clamps vATL_in externally (speaking)
    triangularis: torch.Tensor   # current triangularis activation  (tri_size,)


@dataclass
class TickResult:
    """Full record of one tick within a trial.

    Returned by Lichtheim2Model.run_trial(). Every input and every internal
    activation at every tick is accessible here.
    """

    tick_index: int
    task: "Task"
    sound_input: torch.Tensor        # actual sound vector fed in this tick (zeros if silent)
    vATL_input_used: torch.Tensor    # actual vATL input fed into aSTG this tick
                                     # = state.vATL_context OR the externally clamped pattern
    state: ModelState                # layer activations after this tick


def init_state(
    cfg: "ModelConfig",
    device: "torch.device | str" = "cpu",
) -> ModelState:
    """Initialise a fresh trial state on the given device.

    [Paper]: "At the beginning of each trial, activations for all units in the
    hidden layer (including vATL-output layer) were set to 0.5, and for all
    units in the insular-motor output layer to zero."

    Args:
        cfg:    model configuration
        device: device for the state tensors; should match the model device
    """
    dev = torch.device(device) if isinstance(device, str) else device
    return ModelState(
        iSMG=torch.full((cfg.iSMG_hidden_size,), 0.5, device=dev),
        iSMG_context=torch.full((cfg.iSMG_hidden_size,), 0.5, device=dev),
        motor=torch.zeros(cfg.motor_output_size, device=dev),
        motor_context=torch.zeros(cfg.motor_output_size, device=dev),
        mSTG=torch.full((cfg.mSTG_hidden_size,), 0.5, device=dev),
        aSTG=torch.full((cfg.aSTG_hidden_size,), 0.5, device=dev),
        vATL_out=torch.full((cfg.vATL_size,), 0.5, device=dev),
        vATL_context=torch.full((cfg.vATL_size,), 0.5, device=dev),
        triangularis=torch.full((cfg.triangularis_hidden_size,), 0.5, device=dev),
    )
