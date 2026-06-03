from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from lichtheim2.layers import ModelState, TickResult, init_state
from lichtheim2.tasks import Task, build_trial_inputs

if TYPE_CHECKING:
    from lichtheim2.config import ModelConfig


class Lichtheim2Model(nn.Module):
    """Tick-by-tick Lichtheim 2 neurocomputational model.

    Implements the dual dorsal-ventral architecture from Ueno et al. (2011)
    with explicit LayerState objects and manual copy-back connections following
    Supplemental Figure S1.

    Bias convention [Paper]: copy and Elman layers receive no bias; all other
    layers do. Where two pathways converge on the same output layer (motor),
    the bias is carried by one connection only to avoid double-counting
    [Inferred].

    This module is fully differentiable. forward_tick and run_trial do not
    call .detach(), torch.no_grad(), or .backward(). Training will be added
    in Phase 3 without changes to these methods.
    """

    def __init__(self, cfg: "ModelConfig") -> None:
        super().__init__()
        s   = cfg.sound_input_size
        i   = cfg.iSMG_hidden_size
        mo  = cfg.motor_output_size
        ms  = cfg.mSTG_hidden_size
        a   = cfg.aSTG_hidden_size
        v   = cfg.vATL_size
        t   = cfg.triangularis_hidden_size

        # --- Dorsal pathway ---
        self.sound_to_iSMG       = nn.Linear(s,  i,  bias=True)   # carries iSMG bias
        self.iSMG_elman          = nn.Linear(i,  i,  bias=False)   # Elman; no bias [Paper]
        self.motor_copy_to_iSMG  = nn.Linear(mo, i,  bias=False)   # copy;  no bias [Paper]
        self.iSMG_to_motor       = nn.Linear(i,  mo, bias=True)    # carries motor bias

        # --- Ventral pathway ---
        self.sound_to_mSTG       = nn.Linear(s,  ms, bias=True)
        self.mSTG_to_aSTG        = nn.Linear(ms, a,  bias=True)    # carries aSTG bias
        self.vATL_in_to_aSTG     = nn.Linear(v,  a,  bias=False)   # copy;  no bias [Paper]
        self.aSTG_to_vATL        = nn.Linear(a,  v,  bias=True)
        self.aSTG_to_triangularis = nn.Linear(a,  t,  bias=True)
        self.triangularis_to_motor = nn.Linear(t, mo, bias=False)  # motor bias in iSMG_to_motor

    def forward_tick(
        self,
        state: ModelState,
        sound: torch.Tensor | None = None,
        clamp_vATL_in: torch.Tensor | None = None,
    ) -> tuple[ModelState, torch.Tensor]:
        """Advance all layers by one tick.

        Args:
            state:         carry state from the previous tick (or init_state)
            sound:         phonological input (sound_size,); None → zero vector
            clamp_vATL_in: semantic pattern to hard-clamp at vATL input (speaking);
                           None → use state.vATL_context

        Returns:
            (new_state, vATL_input_used)
            vATL_input_used is the actual tensor fed into aSTG this tick —
            either state.vATL_context or the external clamp.
        """
        sound_size = self.sound_to_iSMG.in_features

        # 1. Resolve inputs
        sound_in = sound if sound is not None else torch.zeros(sound_size)
        vATL_input_used = (
            clamp_vATL_in if clamp_vATL_in is not None else state.vATL_context
        )

        # 2. Dorsal: iSMG receives sound + Elman context + motor copy-back [Supp Fig S1]
        iSMG_net = (
            self.sound_to_iSMG(sound_in)
            + self.iSMG_elman(state.iSMG_context)
            + self.motor_copy_to_iSMG(state.motor_context)
        )
        new_iSMG = torch.sigmoid(iSMG_net)

        # 3. Ventral hidden: mSTG then aSTG
        #    aSTG receives mSTG output + vATL_in (copy-back or clamp) [Supp Fig S1]
        new_mSTG = torch.sigmoid(self.sound_to_mSTG(sound_in))
        aSTG_net = (
            self.mSTG_to_aSTG(new_mSTG)
            + self.vATL_in_to_aSTG(vATL_input_used)
        )
        new_aSTG = torch.sigmoid(aSTG_net)

        # 4. Ventral output: vATL and triangularis-opercularis
        new_vATL_out = torch.sigmoid(self.aSTG_to_vATL(new_aSTG))
        new_triangularis = torch.sigmoid(self.aSTG_to_triangularis(new_aSTG))

        # 5. Motor: both pathways converge on insular-motor cortex [Paper Fig 1]
        motor_net = (
            self.iSMG_to_motor(new_iSMG)
            + self.triangularis_to_motor(new_triangularis)
        )
        new_motor = torch.sigmoid(motor_net)

        # 6. Copy-back update — context fields set at end of tick [Supp Fig S1]
        new_state = ModelState(
            iSMG=new_iSMG,
            iSMG_context=new_iSMG,          # Elman: copy current iSMG
            motor=new_motor,
            motor_context=new_motor,         # copy current motor output
            mSTG=new_mSTG,
            aSTG=new_aSTG,
            vATL_out=new_vATL_out,
            vATL_context=new_vATL_out,       # prepared for next tick; may be overridden by clamp
            triangularis=new_triangularis,
        )
        return new_state, vATL_input_used

    def run_trial(
        self,
        task: Task,
        phon_pattern: torch.Tensor,
        sem_pattern: torch.Tensor,
        cfg: "ModelConfig",
    ) -> list[TickResult]:
        """Run a complete trial and return one TickResult per tick.

        All inputs and internal activations at every tick are accessible
        through the returned TickResult list.
        """
        state = init_state(cfg)
        tick_inputs = build_trial_inputs(task, phon_pattern, sem_pattern, cfg)
        zero_sound = torch.zeros(cfg.sound_input_size)
        results: list[TickResult] = []

        for t, (sound, clamp_vATL) in enumerate(tick_inputs):
            new_state, vATL_used = self.forward_tick(state, sound, clamp_vATL)
            results.append(TickResult(
                tick_index=t,
                task=task,
                sound_input=sound if sound is not None else zero_sound,
                vATL_input_used=vATL_used,
                state=new_state,
            ))
            state = new_state

        return results
