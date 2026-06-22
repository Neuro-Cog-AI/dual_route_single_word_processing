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
    with explicit ModelState objects and manual copy-back connections following
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
        p   = cfg.sound_proj_size      # None = no projection (paper pathway)
        i   = cfg.iSMG_hidden_size
        mo  = cfg.motor_output_size
        ms  = cfg.mSTG_hidden_size
        a   = cfg.aSTG_hidden_size
        v   = cfg.vATL_size
        t   = cfg.triangularis_hidden_size

        self._sound_input_size = s   # always the raw phoneme dim; used in forward_tick
        self.dorsal_motor_only = cfg.dorsal_motor_only  # [Phase 3j diagnostic]

        # --- Optional dense input projection [Adapted — not in Ueno et al. 2011] ---
        # Maps the raw one-hot phoneme vector (s,) to a dense (p,) representation
        # shared by both dorsal and ventral input paths. bias=False: downstream
        # layers (sound_to_iSMG, sound_to_mSTG) already carry biases; a projection
        # bias would be redundant. With bias=False each row of sound_proj.weight is
        # the learned embedding for one phoneme. [Open #10]
        self.sound_proj = nn.Linear(s, p, bias=False) if p is not None else None
        s_eff = p if p is not None else s   # effective sound dim fed into iSMG / mSTG

        # --- Dorsal pathway ---
        self.sound_to_iSMG       = nn.Linear(s_eff, i,  bias=True)   # carries iSMG bias
        self.iSMG_elman          = nn.Linear(i,     i,  bias=False)   # Elman; no bias [Paper]
        self.motor_copy_to_iSMG  = nn.Linear(mo,    i,  bias=False)   # copy;  no bias [Paper]
        self.iSMG_to_motor       = nn.Linear(i,     mo, bias=True)    # carries motor bias

        # --- Ventral pathway ---
        self.sound_to_mSTG       = nn.Linear(s_eff, ms, bias=True)
        self.mSTG_to_aSTG        = nn.Linear(ms, a,  bias=True)    # carries aSTG bias
        self.vATL_in_to_aSTG     = nn.Linear(v,  a,  bias=False)   # copy;  no bias [Paper]
        self.aSTG_to_vATL        = nn.Linear(a,  v,  bias=True)
        self.aSTG_to_triangularis = nn.Linear(a,  t,  bias=True)
        self.triangularis_to_motor = nn.Linear(t, mo, bias=False)  # motor bias in iSMG_to_motor

        self._init_weights()

    def _init_weights(self) -> None:
        """Apply faithful weight initialisation from Ueno et al. (2011).

        Standard feedforward weights: uniform [−1, 1] [Paper]
        Elman weights:                uniform [−0.5, 0.5] [Paper]
        Copy-back/context weights:    uniform [−0.5, 0.5] [Inferred — paper says
            "recurrent connections" but does not explicitly define which connections
            qualify beyond the Elman; treating copy-back as recurrent is an assumption]
        Additive bias (nn.Linear.bias): set to −1.0 [Inferred — PyTorch approximation
            of the LENS bias-link convention; the paper states LENS bias links suppress
            early activation, but the translation to nn.Linear.bias is our assumption]
        Copy and Elman layers: no bias by construction [Paper]
        """
        # Standard feedforward weights: [−1, 1]
        feedforward_modules = [
            self.sound_to_iSMG,
            self.iSMG_to_motor,
            self.sound_to_mSTG,
            self.mSTG_to_aSTG,
            self.aSTG_to_vATL,
            self.aSTG_to_triangularis,
            self.triangularis_to_motor,
        ]
        if self.sound_proj is not None:
            feedforward_modules.append(self.sound_proj)  # bias=False; weight-only init
        for module in feedforward_modules:
            nn.init.uniform_(module.weight, -1.0, 1.0)
            if module.bias is not None:
                nn.init.constant_(module.bias, -1.0)

        # Elman and copy-back/context weights: [−0.5, 0.5]
        # iSMG_elman is Elman [Paper]; motor_copy_to_iSMG and vATL_in_to_aSTG
        # are treated as recurrent by assumption [Inferred].
        for module in (
            self.iSMG_elman,
            self.motor_copy_to_iSMG,
            self.vATL_in_to_aSTG,
        ):
            nn.init.uniform_(module.weight, -0.5, 0.5)
            # bias=False on all three by construction; nothing to set

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
        sound_size = self._sound_input_size   # always the raw phoneme dim (e.g. 39)
        device = self.sound_to_iSMG.weight.device

        # 1. Resolve inputs
        if sound is not None and sound.shape[-1] != sound_size:
            raise ValueError(
                f"sound.shape[-1]={sound.shape[-1]} != sound_input_size={sound_size}; "
                "pass the raw phoneme vector, not a pre-projected representation"
            )
        sound_in = sound if sound is not None else torch.zeros(sound_size, device=device)
        vATL_input_used = (
            clamp_vATL_in if clamp_vATL_in is not None else state.vATL_context
        )

        # 1b. Optional dense input projection [Adapted — not in Ueno et al. 2011]
        # Projects the raw phoneme vector to a dense representation before both
        # dorsal and ventral input paths. Zero sound → zero projection (bias=False).
        if self.sound_proj is not None:
            sound_in = self.sound_proj(sound_in)   # (sound_size,) → (sound_proj_size,)

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
        # When dorsal_motor_only=True (diagnostic), triangularis_to_motor is excluded
        # from motor_net. The ventral pathway above is still fully computed each tick.
        # triangularis_to_motor receives no gradient in this mode (disconnected from loss).
        # [Phase 3j diagnostic — not in Ueno et al. 2011]
        if self.dorsal_motor_only:
            motor_net = self.iSMG_to_motor(new_iSMG)
        else:
            motor_net = (
                self.iSMG_to_motor(new_iSMG)
                + self.triangularis_to_motor(new_triangularis)
            )
        new_motor = torch.sigmoid(motor_net)

        # 6. Copy-back update — context fields set at end of tick [Supp Fig S1]
        # vATL_context is always set to vATL_out of this tick, even when the input
        # was externally clamped (speaking task). In speaking, every tick is clamped,
        # so vATL_context is populated but never used; the next tick's clamp overrides
        # it. No mutation of the clamp tensor is needed.
        new_state = ModelState(
            iSMG=new_iSMG,
            iSMG_context=new_iSMG,          # Elman: copy current iSMG
            motor=new_motor,
            motor_context=new_motor,         # copy current motor output
            mSTG=new_mSTG,
            aSTG=new_aSTG,
            vATL_out=new_vATL_out,
            vATL_context=new_vATL_out,       # prepared for next tick
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
        device = next(self.parameters()).device
        state = init_state(cfg, device=device)
        tick_inputs = build_trial_inputs(task, phon_pattern, sem_pattern, cfg)
        zero_sound = torch.zeros(cfg.sound_input_size, device=device)
        results: list[TickResult] = []

        for t, (sound, clamp_vATL) in enumerate(tick_inputs):
            # Move task-generated tensors to model device. build_trial_inputs
            # always creates CPU tensors via torch.zeros / .clone(); moving them
            # here ensures correctness when the model is on CUDA or MPS.
            sound_dev = sound.to(device) if sound is not None else None
            clamp_dev = clamp_vATL.to(device) if clamp_vATL is not None else None
            new_state, vATL_used = self.forward_tick(state, sound_dev, clamp_dev)
            # build_trial_inputs always returns actual tensors for sound (never None);
            # the None branch below is a safety net for hypothetical direct callers.
            results.append(TickResult(
                tick_index=t,
                task=task,
                sound_input=sound_dev if sound_dev is not None else zero_sound,
                vATL_input_used=vATL_used,
                state=new_state,
            ))
            state = new_state

        return results
