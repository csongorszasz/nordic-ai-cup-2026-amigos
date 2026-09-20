"""Bounded hybrid actions with fixed-latent likelihoods for policy updates.

Continuous densities use normalized distance and normalized angles, before the
environment's movement_limit and pi scales. These state-only Jacobian constants
cancel in PPO ratios for positive movement limits. At zero movement limit the
normalized distance remains an auxiliary action, not a density of executed distance.
Entropy is the analytic latent Normal plus eligible Bernoulli surrogate, not the
entropy of the transformed action distribution.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Tensor
from torch.nn import functional as F

from src.benchmarking.policies import validate_actions
from src.policies.features import validate_step
from src.policies.geometry import can_spawn, movement_limit
from src.utils.DTOs import ActionRequest, StepResponse


ACTION_VERSION = "bounded-v1"
VECTOR_ACTION_VERSION = "bc-vector-angles-v2"


def model_action_version(config) -> str:
    return VECTOR_ACTION_VERSION if config.angle_head == "vector_bc" else ACTION_VERSION


class _ActionOutput(Protocol):
    mean: Tensor
    log_std: Tensor
    spawn_logits: Tensor


@dataclass(frozen=True)
class ActionSample:
    actions: list[ActionRequest]
    latent: Tensor
    spawn: Tensor
    eligible: Tensor
    log_prob: Tensor
    entropy: Tensor


def _parameters(output: _ActionOutput, count: int | None = None) -> Tensor:
    mean = output.mean
    if not isinstance(mean, Tensor) or mean.ndim != 2 or mean.shape[1] != 3:
        raise ValueError("Action mean must have shape [N, 3].")
    if count is not None and mean.shape[0] != count:
        raise ValueError("Action output must cover exactly the observed agents.")
    for name, shape in (("mean", mean.shape), ("log_std", mean.shape),
                        ("spawn_logits", (mean.shape[0],))):
        value = getattr(output, name)
        if (
            not isinstance(value, Tensor) or value.shape != shape
            or not value.is_floating_point() or value.device != mean.device
            or value.dtype != mean.dtype
        ):
            raise ValueError(f"Action {name} has an invalid shape, device, or dtype.")
        if not torch.isfinite(value).all().item():
            raise ValueError(f"Action {name} contains non-finite values.")
    std = output.log_std.exp()
    if not (torch.isfinite(std) & (std > 0)).all().item():
        raise ValueError("Action standard deviations must be finite and positive.")
    return std


def evaluate_actions(
    output: _ActionOutput, latent: Tensor, spawn: Tensor, eligible: Tensor,
) -> tuple[Tensor, Tensor]:
    """Evaluate the saved latent and collection-time eligibility without clipping."""
    std = _parameters(output)
    mean = output.mean
    for name, value, shape in (
        ("latent", latent, mean.shape),
        ("spawn", spawn, (mean.shape[0],)),
        ("eligible", eligible, (mean.shape[0],)),
    ):
        if not isinstance(value, Tensor) or value.shape != shape or value.device != mean.device:
            raise ValueError(f"Action {name} has an invalid shape or device.")
    if latent.dtype != mean.dtype or not torch.isfinite(latent).all().item():
        raise ValueError("Action latent must have the output dtype and finite values.")
    if eligible.dtype != torch.bool:
        raise ValueError("Action eligibility must be boolean.")
    if spawn.is_complex() or not ((spawn == 0) | (spawn == 1)).all().item():
        raise ValueError("Action spawn must contain only zero or one.")
    if ((spawn != 0) & ~eligible).any().item():
        raise ValueError("An ineligible action cannot execute reproduction.")

    normal = -0.5 * ((latent - mean) / std).square() - output.log_std - 0.5 * math.log(math.tau)
    distance_jacobian = -F.softplus(-latent[:, 0]) - F.softplus(latent[:, 0])
    angle_magnitude = latent[:, 1:].abs()
    angle_jacobian = 2 * (math.log(2) - angle_magnitude - F.softplus(-2 * angle_magnitude))
    continuous = normal.sum(-1) - distance_jacobian - angle_jacobian.sum(-1)
    binary = -F.binary_cross_entropy_with_logits(
        output.spawn_logits, spawn.to(dtype=mean.dtype), reduction="none",
    )
    log_prob = continuous + torch.where(eligible, binary, torch.zeros_like(binary))

    magnitude = output.spawn_logits.abs()
    binary_entropy = F.softplus(-magnitude) + magnitude * torch.sigmoid(-magnitude)
    entropy = (output.log_std + 0.5 * (1 + math.log(math.tau))).sum(-1)
    entropy = entropy + torch.where(eligible, binary_entropy, torch.zeros_like(binary_entropy))
    if not torch.isfinite(log_prob).all().item() or not torch.isfinite(entropy).all().item():
        raise ValueError("Action likelihood or entropy is non-finite.")
    return log_prob, entropy


def sample_actions(
    output: _ActionOutput, step: StepResponse, *, deterministic: bool = False,
    generator: torch.Generator | None = None,
) -> ActionSample:
    """Draw detached collection latents; deterministic calls do not consume RNG."""
    validate_step(step)
    if getattr(output, "angle_vectors", None) is not None and not deterministic:
        raise ValueError("The vector BC head supports deterministic inference only; stochastic collection is not validated.")
    std = _parameters(output, len(step.agent_status))
    mean = output.mean
    if deterministic:
        latent = mean.detach().clone()
    else:
        noise = torch.randn(mean.shape, dtype=mean.dtype, device=mean.device, generator=generator)
        latent = (mean + std * noise).detach()
    if not torch.isfinite(latent).all().item():
        raise ValueError("Sampled action latents are non-finite.")
    normalized = torch.cat((latent[:, :1].sigmoid(), latent[:, 1:].tanh()), dim=1).cpu().tolist()
    decoded = [
        (movement_limit(agent) * row[0], math.pi * row[1], math.pi * row[2])
        for agent, row in zip(step.agent_status, normalized)
    ]
    eligible = torch.tensor(
        [can_spawn(agent, distance, turn)
         for agent, (distance, _, turn) in zip(step.agent_status, decoded)],
        dtype=torch.bool, device=mean.device,
    )
    if deterministic:
        requested_spawn = output.spawn_logits.detach() >= 0
    else:
        requested_spawn = torch.rand(
            output.spawn_logits.shape, dtype=mean.dtype, device=mean.device, generator=generator,
        ) < output.spawn_logits.detach().sigmoid()
    spawn = (requested_spawn & eligible).to(dtype=mean.dtype)
    actions = [
        ActionRequest(
            agent_id=agent.agent_id, move_distance=distance, move_direction=direction,
            turn_angle=turn, spawn_agent=bool(reproduce),
        )
        for agent, (distance, direction, turn), reproduce
        in zip(step.agent_status, decoded, spawn.cpu().tolist())
    ]
    actions = validate_actions(actions, [agent.agent_id for agent in step.agent_status])
    log_prob, entropy = evaluate_actions(output, latent, spawn, eligible)
    return ActionSample(actions, latent, spawn, eligible, log_prob, entropy)


def imitation_targets(
    step: StepResponse, teacher_actions: Sequence[ActionRequest],
) -> tuple[list[list[float]], list[bool]]:
    """Public-observation targets; eligibility follows the teacher's executed action."""
    validate_step(step)
    teachers = validate_actions(teacher_actions, [agent.agent_id for agent in step.agent_status])
    by_id = {action.agent_id: action for action in teachers}
    targets, eligibility = [], []
    for agent in step.agent_status:
        teacher = by_id[agent.agent_id]
        limit = movement_limit(agent)
        if not 0 <= teacher.move_distance <= limit:
            raise ValueError(f"Agent {agent.agent_id}: teacher distance exceeds movement bounds.")
        eligible = can_spawn(agent, teacher.move_distance, teacher.turn_angle)
        if teacher.spawn_agent and not eligible:
            raise ValueError(f"Agent {agent.agent_id}: teacher requested ineligible reproduction.")
        targets.append([
            teacher.move_distance / limit if limit > 0 else 0.0,
            teacher.move_direction, teacher.turn_angle, float(teacher.spawn_agent),
            float(teacher.move_distance > 0), float(limit > 0),
        ])
        eligibility.append(eligible)
    return targets, eligibility


def imitation_components(output: _ActionOutput, target: Tensor, eligible: Tensor) -> Tensor:
    """Per-agent distance, circular travel/turn, and masked reproduction losses."""
    distance = (output.mean[:, 0].sigmoid() - target[:, 0]).square() * target[:, 5]
    vectors = getattr(output, "angle_vectors", None)
    if vectors is None:
        angles = math.pi * output.mean[:, 1:].tanh()
        travel = (1 - torch.cos(angles[:, 0] - target[:, 1])) * target[:, 4]
        turn = 1 - torch.cos(angles[:, 1] - target[:, 2])
    else:
        units = torch.stack((target[:, 1:3].cos(), target[:, 1:3].sin()), dim=-1)
        angular = 0.5 * (vectors - units).square().sum(-1)
        travel, turn = angular[:, 0] * target[:, 4], angular[:, 1]
    spawn = F.binary_cross_entropy_with_logits(output.spawn_logits, target[:, 3], reduction="none")
    return torch.stack((distance, travel, turn, torch.where(eligible, spawn, torch.zeros_like(spawn))), dim=1)


def imitation_loss(
    output: _ActionOutput, step: StepResponse, teacher_actions: Sequence[ActionRequest],
) -> Tensor:
    """Mean per-agent distance MSE, circular angles, and eligible reproduction BCE."""
    _parameters(output, len(step.agent_status))
    targets, eligibility = imitation_targets(step, teacher_actions)
    if not targets:
        return output.mean.sum() + output.spawn_logits.sum()
    target = output.mean.new_tensor(targets)
    eligible = torch.tensor(eligibility, dtype=torch.bool, device=output.mean.device)
    loss = imitation_components(output, target, eligible).sum(1).mean()
    if not torch.isfinite(loss).item():
        raise ValueError("Imitation loss is non-finite.")
    return loss
