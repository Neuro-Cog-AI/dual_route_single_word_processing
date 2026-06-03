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

    sound_input is always a tensor (zeros for silent ticks).
    clamp_vATL_in is the semantic pattern when the task hard-clamps vATL_in,
    or None when vATL_in should come from state.vATL_context.

    Args:
        phon_pattern: shape (n_morae, sound_size) or (sound_size,).
            When 1-D, the same vector is broadcast to all three mora positions.
        sem_pattern:  shape (vATL_size,).

    Task schedules [Paper]:
        Repetition   (6 ticks): morae at ticks 0-2, zeros at ticks 3-5
        Comprehension (3 ticks): morae at ticks 0-2
        Speaking      (3 ticks): zeros for sound; sem_pattern clamped to vATL_in
    """
    sound_size = cfg.sound_input_size
    zero_sound = torch.zeros(sound_size)

    # Normalise phon_pattern to shape (n_morae, sound_size)
    if phon_pattern.dim() == 1:
        morae = phon_pattern.unsqueeze(0).expand(3, -1)
    else:
        morae = phon_pattern  # (n_morae, sound_size)

    if task == Task.REPETITION:
        inputs: list[tuple[torch.Tensor, torch.Tensor | None]] = []
        for i in range(3):
            inputs.append((morae[i].clone(), None))       # ticks 0-2: sound input
        for _ in range(3):
            inputs.append((zero_sound.clone(), None))     # ticks 3-5: silence
        return inputs

    elif task == Task.COMPREHENSION:
        return [(morae[i].clone(), None) for i in range(3)]

    elif task == Task.SPEAKING:
        # Sound is silent; semantic pattern hard-clamped to vATL_in [Paper]
        return [(zero_sound.clone(), sem_pattern.clone()) for _ in range(3)]

    else:
        raise ValueError(f"Unknown task: {task}")
