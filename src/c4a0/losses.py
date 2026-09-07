"""Shared policy/value objectives; eligibility and head coefficients are independent."""

import torch


def policy_value_losses(predictions, batch, value_loss_weight: float = 1.0):
    policy, q_penalty, q_no_penalty = predictions
    _, target, penalty_target, value_target = batch[:4]
    policy_weight = batch[4] if len(batch) > 4 else torch.ones_like(penalty_target)
    value_weight = batch[5] if len(batch) > 5 else torch.ones_like(penalty_target)
    policy_each = (target * (torch.log(target + 1e-8) - policy)).sum(dim=1)
    policy_loss = (policy_each * policy_weight).sum() / policy_weight.sum().clamp_min(
        1e-8
    )
    denominator = value_weight.sum().clamp_min(1e-8)
    penalty_loss = (
        (q_penalty - penalty_target).square() * value_weight
    ).sum() / denominator
    value_loss = (
        (q_no_penalty - value_target).square() * value_weight
    ).sum() / denominator
    return policy_loss, penalty_loss * value_loss_weight, value_loss * value_loss_weight
