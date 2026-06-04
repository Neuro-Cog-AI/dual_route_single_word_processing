from __future__ import annotations

from enum import Enum, auto
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from lichtheim2.config import ModelConfig


class Task(Enum):
    REPETITION = auto()
    COMPREHENSION = auto()
    SPEAKING = auto()


def build_trial_inputs(
    task: Task,
    phon_pattern: torch.Tensor,
    sem_pattern: torch.Tensor,
    cfg: "ModelConfig",
) -> list[tuple[torch.Tensor, torch.Tensor | None]]:
    """Return one (sound_input, clamp_vATL_in) pair per tick.

    Trial length T is inferred from phon_pattern.shape[0]. Config fields such
    as cfg.repetition_ticks, cfg.comprehension_ticks, and cfg.speaking_ticks
    are historical reference values for the original fixed-length (3-mora)
    setup and are NOT used here. Runtime tick counts are always T-derived.

    sound_input is always a tensor (zeros for silent ticks).
    clamp_vATL_in is the semantic pattern when the task hard-clamps vATL_in,
    or None when vATL_in should come from state.vATL_context.

    Args:
        phon_pattern: shape (T, sound_size) — T phonemes, one per input tick.
            When 1-D (shape (sound_size,)), treated as a single phoneme: T=1.
        sem_pattern:  shape (vATL_size,).

    Task schedules:
        Repetition    (2T ticks): T input ticks (one phoneme each), then T
                                  output ticks (zero sound input) [Paper, generalised]
        Comprehension (T ticks):  T input ticks; semantic output at tick T [Paper]
        Speaking      (T ticks):  T output ticks; sound is zero; sem_pattern
                                  clamped to vATL_in at each tick [Paper].
                                  phon_pattern is required even for speaking
                                  because its length T determines the number of
                                  output ticks.
    """
    sound_size = cfg.sound_input_size
    zero_sound = torch.zeros(sound_size)

    # Normalise phon_pattern to shape (T, sound_size).
    # 1-D input is treated as a single phoneme: T=1.
    if phon_pattern.dim() == 1:
        morae = phon_pattern.unsqueeze(0)   # (1, sound_size) → T=1
    else:
        morae = phon_pattern                # (T, sound_size)

    T = morae.shape[0]

    if task == Task.REPETITION:
        inputs: list[tuple[torch.Tensor, torch.Tensor | None]] = []
        for i in range(T):
            inputs.append((morae[i].clone(), None))       # ticks 0..T-1: phoneme input
        for _ in range(T):
            inputs.append((zero_sound.clone(), None))     # ticks T..2T-1: silent output phase
        return inputs

    elif task == Task.COMPREHENSION:
        return [(morae[i].clone(), None) for i in range(T)]

    elif task == Task.SPEAKING:
        # Sound is zero; semantic pattern hard-clamped to vATL_in [Paper].
        # T is still determined by phon_pattern.shape[0].
        return [(zero_sound.clone(), sem_pattern.clone()) for _ in range(T)]

    else:
        raise ValueError(f"Unknown task: {task}")
