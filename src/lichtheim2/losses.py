from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from lichtheim2.layers import TickResult
    from lichtheim2.trials import SupervisedTrial


def compute_trial_loss(
    tick_results: list["TickResult"],
    trial: "SupervisedTrial",
    zero_error_radius: float = 0.0,
) -> torch.Tensor:
    """Compute combined motor + semantic binary cross-entropy loss for one trial.

    Loss is computed only at ticks where the corresponding loss mask is True.
    Gradient flows back through tick_results (model outputs from run_trial()).

    Motor loss:
      BCE between model motor activations and trial.motor_targets at masked ticks.
    Semantic loss (if trial.semantic_targets is not None):
      BCE between vATL outputs and trial.semantic_targets at masked ticks.

    Zero-error radius:
      If zero_error_radius > 0, units where |prediction − target| < zero_error_radius
      contribute zero loss (no gradient). Applied consistently to both motor and
      semantic losses. Default 0.0 = standard BCE with no dead zone.
      From the paper: zero_error_radius = 0.1 [Paper].

    Args:
        tick_results:       list of TickResult from model.run_trial()
        trial:              SupervisedTrial with targets and masks
        zero_error_radius:  dead-zone threshold; 0.0 disables it

    Returns:
        Scalar tensor with .requires_grad=True (gradient flows back to model params).
    """
    # -- Motor loss --
    motor_outputs = torch.stack(
        [r.state.motor for r in tick_results], dim=0
    )  # (n_ticks, motor_size)

    raw_motor = F.binary_cross_entropy(
        motor_outputs, trial.motor_targets, reduction="none"
    )  # (n_ticks, motor_size)

    if zero_error_radius > 0.0:
        dead_motor = (motor_outputs.detach() - trial.motor_targets).abs() < zero_error_radius
        raw_motor = raw_motor * (~dead_motor).float()

    motor_loss = (
        raw_motor * trial.motor_loss_mask.unsqueeze(-1).float()
    ).sum()

    # -- Semantic loss (comprehension only; None for repetition / speaking) --
    if trial.semantic_targets is not None and trial.semantic_loss_mask is not None:
        vATL_outputs = torch.stack(
            [r.state.vATL_out for r in tick_results], dim=0
        )  # (n_ticks, vATL_size)

        raw_sem = F.binary_cross_entropy(
            vATL_outputs, trial.semantic_targets, reduction="none"
        )  # (n_ticks, vATL_size)

        if zero_error_radius > 0.0:
            dead_sem = (vATL_outputs.detach() - trial.semantic_targets).abs() < zero_error_radius
            raw_sem = raw_sem * (~dead_sem).float()

        semantic_loss = (
            raw_sem * trial.semantic_loss_mask.unsqueeze(-1).float()
        ).sum()
    else:
        semantic_loss = torch.zeros(1, device=motor_loss.device)[0]

    return motor_loss + semantic_loss
