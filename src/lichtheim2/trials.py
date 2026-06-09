from __future__ import annotations

from dataclasses import dataclass

import torch

from lichtheim2.tasks import Task


@dataclass
class SupervisedTrial:
    """Target tensors and loss masks for one supervised training trial.

    Produced by make_repetition_trial(), make_comprehension_trial(), or
    make_speaking_trial(). Consumed by the training loop (Phase 3c).

    All target tensors use the same dtype as the model activations (float32).
    Loss masks are bool tensors; True indicates ticks where loss is computed.
    """

    task: Task
    n_ticks: int                            # total ticks (2T for repetition, T otherwise)

    # Inputs
    phon_tensor: torch.Tensor               # (T, sound_size) — phoneme input sequence
    sem_input: torch.Tensor | None          # (vATL_size,) — semantic input (speaking only)

    # Motor output targets and mask
    motor_targets: torch.Tensor             # (n_ticks, motor_size)
    motor_loss_mask: torch.Tensor           # (n_ticks,) bool

    # Semantic output targets and mask
    semantic_targets: torch.Tensor | None   # (n_ticks, vATL_size) or None
    semantic_loss_mask: torch.Tensor | None # (n_ticks,) bool or None

    # Optional metadata for debugging / logging
    item_id: int | None = None
    label: str | None = None
    loss_weight: float = 1.0  # scalar multiplier on the loss; 1.0 = unweighted (default)


def make_repetition_trial(
    phon_tensor: torch.Tensor,
    motor_size: int,
    item_id: int | None = None,
    label: str | None = None,
) -> SupervisedTrial:
    """Create a supervised repetition trial.

    Tick structure (T = phon_tensor.shape[0]):
      Input phase  (ticks 0..T-1):   sound clamped; motor target = zeros [Paper]
      Output phase (ticks T..2T-1):  sound = zeros; motor target = phon_tensor

    Motor loss is computed at ALL ticks [Paper: motor "required to be silent"
    during input phase, meaning it has a zero target with gradient].
    Semantic: not evaluated in repetition.

    Works for both real words and pseudowords (no semantic tensor required).

    Args:
        phon_tensor: (T, sound_size) — phonological input sequence
        motor_size:  output dimension of the motor layer;
                     must equal phon_tensor.shape[1] since phon_tensor is the
                     motor target during the output phase

    Raises:
        ValueError: if motor_size != phon_tensor.shape[1]
    """
    T, phon_dim = phon_tensor.shape
    if phon_dim != motor_size:
        raise ValueError(
            f"phon_tensor.shape[1]={phon_dim} != motor_size={motor_size}; "
            "phonological input and motor output dimensions must match for repetition"
        )

    n_ticks = 2 * T
    motor_targets = torch.cat([
        torch.zeros(T, motor_size),  # input phase: motor silent
        phon_tensor.clone(),         # output phase: reproduce phonemes
    ], dim=0)
    motor_loss_mask = torch.ones(n_ticks, dtype=torch.bool)

    return SupervisedTrial(
        task=Task.REPETITION,
        n_ticks=n_ticks,
        phon_tensor=phon_tensor,
        sem_input=None,
        motor_targets=motor_targets,
        motor_loss_mask=motor_loss_mask,
        semantic_targets=None,
        semantic_loss_mask=None,
        item_id=item_id,
        label=label,
    )


def make_comprehension_trial(
    phon_tensor: torch.Tensor,
    sem_tensor: torch.Tensor,
    motor_size: int,
    item_id: int | None = None,
    label: str | None = None,
) -> SupervisedTrial:
    """Create a supervised comprehension trial.

    Tick structure (T = phon_tensor.shape[0]):
      All T ticks: sound clamped to phonemes; vATL output trained to match
                   sem_tensor; motor trained to be silent.

    Semantic loss: applied at ALL ticks [Paper/Supp: "target semantic pattern
    was compared to the output of the vATL layer at every time tick"].
    Motor loss: motor required to be silent throughout [Paper].
    sem_tensor is the TARGET (vATL output), not an input.

    Args:
        phon_tensor: (T, sound_size)
        sem_tensor:  (vATL_size,) — semantic target, applied at every tick
        motor_size:  motor output dimension (for zero-target shape)
    """
    T = phon_tensor.shape[0]

    motor_targets = torch.zeros(T, motor_size)
    motor_loss_mask = torch.ones(T, dtype=torch.bool)

    semantic_targets = sem_tensor.unsqueeze(0).expand(T, -1).clone()  # (T, vATL_size)
    semantic_loss_mask = torch.ones(T, dtype=torch.bool)

    return SupervisedTrial(
        task=Task.COMPREHENSION,
        n_ticks=T,
        phon_tensor=phon_tensor,
        sem_input=None,
        motor_targets=motor_targets,
        motor_loss_mask=motor_loss_mask,
        semantic_targets=semantic_targets,
        semantic_loss_mask=semantic_loss_mask,
        item_id=item_id,
        label=label,
    )


def make_speaking_trial(
    phon_tensor: torch.Tensor,
    sem_tensor: torch.Tensor,
    motor_size: int,
    item_id: int | None = None,
    label: str | None = None,
) -> SupervisedTrial:
    """Create a supervised speaking/naming trial.

    Tick structure (T = phon_tensor.shape[0]):
      All T ticks: sem_tensor clamped to vATL_in (semantic INPUT);
                   motor output trained to produce phon_tensor [Paper].

    sem_tensor is the INPUT here (clamped to vATL_in), not a target.
    Semantic output: not evaluated — vATL is driven by the clamped input.

    Args:
        phon_tensor: (T, sound_size) — target phoneme sequence for motor output
        sem_tensor:  (vATL_size,) — semantic input, clamped at every tick
        motor_size:  output dimension; must equal phon_tensor.shape[1]

    Raises:
        ValueError: if motor_size != phon_tensor.shape[1]
    """
    T, phon_dim = phon_tensor.shape
    if phon_dim != motor_size:
        raise ValueError(
            f"phon_tensor.shape[1]={phon_dim} != motor_size={motor_size}; "
            "phonological output and motor output dimensions must match for speaking"
        )

    motor_targets = phon_tensor.clone()
    motor_loss_mask = torch.ones(T, dtype=torch.bool)

    return SupervisedTrial(
        task=Task.SPEAKING,
        n_ticks=T,
        phon_tensor=phon_tensor,
        sem_input=sem_tensor,
        motor_targets=motor_targets,
        motor_loss_mask=motor_loss_mask,
        semantic_targets=None,
        semantic_loss_mask=None,
        item_id=item_id,
        label=label,
    )
