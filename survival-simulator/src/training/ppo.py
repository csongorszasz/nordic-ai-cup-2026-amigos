"""Raw-team-return PPO with token-bounded, truncated recurrent microbatches.

The MAPPO-style actor surrogate sums living agents' clipped losses / 5, then
averages team ticks. The fixed divisor never changes with population. Value loss
is once per team tick; entropy is the analytic latent surrogate averaged over
living agents, then ticks. Sequences accumulate gradients into one epoch step,
so short/token-limited sequences are not overweighted.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from src.policies.actions import evaluate_actions, imitation_loss
from src.policies.config import ExperimentConfig
from src.policies.networks import PolicyNetwork
from src.training.imitation import checked_optimizer_step
from src.training.rollout import (
    Rollout, require_finite, sequence_chunks, unroll_sequence,
)


ACTOR_DIVISOR = 5.0


def generalized_advantage_estimate(
    rewards: Sequence[float] | torch.Tensor,
    values: Sequence[float] | torch.Tensor,
    terminated: Sequence[bool] | torch.Tensor,
    last_value: float,
    gamma: float = 1.0,
    lam: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return CPU float64 (advantages, returns), with only TEAM terminal masks.

    Rollout/sequence boundaries are not terminal. The last raw value bootstraps
    an unfinished team trajectory; a true extinction/horizon ignores it.
    """
    if not math.isfinite(gamma) or not math.isfinite(lam) or not 0 <= gamma <= 1 or not 0 <= lam <= 1:
        raise ValueError("GAE gamma and lambda must lie in [0, 1].")
    if not math.isfinite(last_value):
        raise FloatingPointError("The GAE bootstrap value is not finite.")
    rewards = torch.as_tensor(rewards, dtype=torch.float64, device="cpu").detach()
    values = torch.as_tensor(values, dtype=torch.float64, device="cpu").detach()
    terminated = torch.as_tensor(terminated, device="cpu")
    if (
        rewards.ndim != 1 or not rewards.numel() or values.shape != rewards.shape
        or terminated.shape != rewards.shape or terminated.dtype != torch.bool
    ):
        raise ValueError("GAE requires equally sized, nonempty 1-D rewards/values and boolean terminals.")
    require_finite(rewards, "Raw rewards")
    require_finite(values, "Raw values")
    advantages = torch.empty_like(rewards)
    carry = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        alive = 0.0 if terminated[index].item() else 1.0
        next_value = last_value if index == len(rewards) - 1 else values[index + 1].item()
        delta = rewards[index].item() + gamma * alive * next_value - values[index].item()
        carry = delta + gamma * lam * alive * carry
        advantages[index] = carry
    returns = advantages + values
    require_finite(advantages, "GAE advantages")
    require_finite(returns, "Raw returns")
    return advantages, returns


@dataclass
class RunningReturnStats:
    """Population variance in raw return units; scaling never recenters the critic."""

    count: int = 0
    mean: float = 0.0
    m2: float = 0.0

    @property
    def scale(self) -> float:
        return max(1.0, math.sqrt(self.m2 / self.count)) if self.count else 1.0

    def update(self, returns: Sequence[float] | torch.Tensor) -> None:
        values = torch.as_tensor(returns, dtype=torch.float64, device="cpu").detach()
        if values.ndim != 1 or not values.numel():
            raise ValueError("Return normalization requires nonempty raw team returns.")
        require_finite(values, "Normalizer raw returns")
        count = values.numel()
        mean = float(values.mean().item())
        m2 = float(((values - mean) ** 2).sum().item())
        total = self.count + count
        delta = mean - self.mean
        new_mean = self.mean + delta * count / total
        new_m2 = self.m2 + m2 + delta * delta * self.count * count / total
        if not math.isfinite(new_mean) or not math.isfinite(new_m2):
            raise FloatingPointError("Raw return statistics overflowed.")
        self.count, self.mean, self.m2 = total, new_mean, new_m2

    def state_dict(self) -> dict:
        return {"count": self.count, "mean": self.mean, "m2": self.m2}

    def load_state_dict(self, state: dict) -> None:
        count, mean, m2 = state.get("count"), state.get("mean"), state.get("m2")
        if (
            type(count) is not int or count < 0
            or type(mean) not in (float, int) or type(m2) not in (float, int)
            or not math.isfinite(mean) or not math.isfinite(m2) or m2 < 0
            or (count == 0 and (mean != 0 or m2 != 0))
        ):
            raise ValueError("Invalid saved raw return statistics.")
        self.count, self.mean, self.m2 = count, float(mean), float(m2)


def scaled_value_loss(value: torch.Tensor, target: float | torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """Half squared RAW value error, scaled only by a fixed-for-update std >= 1."""
    if not math.isfinite(scale) or scale < 1:
        raise ValueError("The raw-value loss scale must be finite and at least one.")
    target = torch.as_tensor(target, device=value.device, dtype=value.dtype).detach()
    if target.shape != value.shape:
        raise ValueError("Raw value and return targets must have matching shapes.")
    loss = 0.5 * ((value - target) / scale).square()
    require_finite(loss, "Raw-unit value loss")
    return loss


def clipped_policy_objective(
    log_prob: torch.Tensor,
    old_log_prob: torch.Tensor,
    team_advantage: float | torch.Tensor,
    clip_ratio: float = 0.2,
    actor_divisor: float = ACTOR_DIVISOR,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (fixed-divisor actor loss, per-agent k3 KL estimate, clip fraction)."""
    if (
        log_prob.ndim != 1 or not log_prob.numel() or old_log_prob.shape != log_prob.shape
        or not 0 < clip_ratio < 1 or not math.isfinite(actor_divisor) or actor_divisor <= 0
    ):
        raise ValueError("Invalid per-agent PPO probabilities or clipping/normalization settings.")
    require_finite(log_prob, "Current log probability")
    require_finite(old_log_prob, "Behavior log probability")
    advantage = torch.as_tensor(team_advantage, device=log_prob.device, dtype=log_prob.dtype).detach()
    if advantage.numel() != 1:
        raise ValueError("All living actors in a tick must share one team advantage.")
    require_finite(advantage, "Normalized team advantage")
    log_ratio = log_prob - old_log_prob.detach()
    ratio = log_ratio.exp()
    require_finite(ratio, "PPO importance ratio; reduce the learning rate if it overflows")
    surrogate = torch.minimum(
        ratio * advantage, ratio.clamp(1.0 - clip_ratio, 1.0 + clip_ratio) * advantage,
    )
    loss = -surrogate.sum() / actor_divisor
    approximate_kl = (log_ratio.expm1() - log_ratio).mean()
    clip_fraction = ((ratio - 1).abs() > clip_ratio).to(log_prob.dtype).mean()
    require_finite(loss, "Clipped actor loss")
    require_finite(approximate_kl, "PPO approximate KL")
    return loss, approximate_kl, clip_fraction


def teacher_regularization_weight(config: ExperimentConfig, update: int) -> float:
    if not 0 <= update < config.updates:
        raise ValueError("The PPO update must be below the target total update count.")
    return config.ppo.imitation_weight * (config.updates - 1 - update) / max(1, config.updates - 1)


def train_ppo(
    network: PolicyNetwork,
    optimizer: torch.optim.Optimizer,
    rollout: Rollout,
    config: ExperimentConfig,
    *,
    update: int,
    value_normalizer: RunningReturnStats,
) -> dict:
    if rollout.policy_version != update:
        raise ValueError("PPO requires the frozen rollout from the current update's policy version.")
    if len(rollout.trajectories) != len(rollout.last_values) or not rollout.frame_count:
        raise ValueError("PPO requires a nonempty rollout with a bootstrap value per environment.")
    chunks = sequence_chunks(
        rollout.trajectories, config.ppo.sequence_length, config.resources.max_tokens,
    )
    frames, advantages, returns = [], [], []
    for trajectory, last_value in zip(rollout.trajectories, rollout.last_values, strict=True):
        if not trajectory:
            continue
        previous = None
        for frame in trajectory:
            if (
                frame.teacher_mask.dtype != torch.bool
                or frame.teacher_mask.shape != (len(frame.features.agent_ids),)
            ):
                raise ValueError("PPO requires one boolean teacher-execution mask per living actor.")
            if frame.teacher_mask.any().item():
                raise ValueError("PPO cannot use teacher-executed actions as on-policy learner samples.")
            if previous is not None and (
                frame.env_index != previous.env_index
                or frame.episode_start != previous.terminated
                or (not previous.terminated and (
                    frame.world_seed != previous.world_seed
                    or frame.episode_step != previous.episode_step + 1
                ))
            ):
                raise ValueError("PPO trajectories must be chronological per environment with explicit team resets.")
            previous = frame
        env_advantages, env_returns = generalized_advantage_estimate(
            [frame.reward for frame in trajectory], [frame.value for frame in trajectory],
            [frame.terminated for frame in trajectory], last_value,
            config.ppo.gamma, config.ppo.gae_lambda,
        )
        frames.extend(trajectory)
        advantages.append(env_advantages)
        returns.append(env_returns)
    raw_advantages, raw_returns = torch.cat(advantages), torch.cat(returns)
    advantage_mean = raw_advantages.mean()
    advantage_std = raw_advantages.std(unbiased=False)
    require_finite(advantage_mean, "Raw advantage mean")
    require_finite(advantage_std, "Raw advantage standard deviation")
    normalized_advantages = (raw_advantages - advantage_mean) / advantage_std.clamp_min(1e-8)
    require_finite(normalized_advantages, "Normalized advantages")
    if config.ppo.normalize_value:
        value_normalizer.update(raw_returns)
    value_scale = value_normalizer.scale if config.ppo.normalize_value else 1.0
    targets = {
        id(frame): (float(advantage), float(target))
        for frame, advantage, target in zip(frames, normalized_advantages, raw_returns, strict=True)
    }
    if len(targets) != len(frames):
        raise ValueError("A PPO rollout cannot contain a duplicated frame.")
    teacher_weight = teacher_regularization_weight(config, update)
    was_training = network.training
    network.train()
    optimizer_steps = 0
    early_stopped = False
    stop_kl = 0.0
    last_metrics = {}
    try:
        for _ in range(config.ppo.epochs):
            optimizer.zero_grad(set_to_none=True)
            metrics = dict.fromkeys(
                ("loss", "actor_loss", "value_loss", "entropy", "teacher_loss", "approx_kl", "clip_fraction"),
                0.0,
            )
            for chunk in chunks:
                sequence_losses = []
                for frame, output in unroll_sequence(network, chunk):
                    if output.value.numel() != 1:
                        raise ValueError("PPO's critic must emit one raw team value per tick.")
                    value = output.value.reshape(())
                    device = output.mean.device
                    log_prob, entropy = evaluate_actions(
                        output, frame.latent.to(device), frame.spawn.to(device), frame.eligible.to(device),
                    )
                    if entropy.shape != log_prob.shape:
                        raise ValueError("Action entropy must have one entry per living actor.")
                    require_finite(entropy, "Analytic latent entropy surrogate")
                    advantage, target = targets[id(frame)]
                    actor, kl, clipped = clipped_policy_objective(
                        log_prob, frame.old_log_prob.to(device), advantage, config.ppo.clip_ratio,
                    )
                    critic = scaled_value_loss(value, target, value_scale)
                    entropy_term = entropy.mean()
                    teacher = (
                        imitation_loss(output, frame.step, list(frame.teacher_actions))
                        if teacher_weight else value.new_zeros(())
                    )
                    if teacher.numel() != 1:
                        raise ValueError("Teacher regularization must be scalar per team tick.")
                    require_finite(teacher, "PPO teacher regularization")
                    teacher = teacher.reshape(())
                    loss = (
                        actor + config.ppo.value_weight * critic
                        - config.ppo.entropy_weight * entropy_term + teacher_weight * teacher
                    )
                    require_finite(loss, "PPO team-tick loss")
                    sequence_losses.append(loss)
                    for name, term in (
                        ("loss", loss), ("actor_loss", actor), ("value_loss", critic),
                        ("entropy", entropy_term), ("teacher_loss", teacher),
                        ("approx_kl", kl), ("clip_fraction", clipped),
                    ):
                        metrics[name] += float(term.detach().cpu().item()) / len(frames)
                sequence_loss = torch.stack(sequence_losses).sum() / len(frames)
                require_finite(sequence_loss, "PPO sequence loss")
                sequence_loss.backward()
            if metrics["approx_kl"] > config.ppo.target_kl:
                optimizer.zero_grad(set_to_none=True)
                if optimizer_steps == 0:
                    raise ValueError(
                        "The initial PPO KL already exceeds target_kl; the rollout does not "
                        "match the current policy/recurrent state."
                    )
                early_stopped, stop_kl = True, metrics["approx_kl"]
                break
            metrics["gradient_norm"] = checked_optimizer_step(
                network, optimizer, config.optimizer.max_grad_norm,
            )
            optimizer_steps += 1
            last_metrics = metrics
    finally:
        network.train(was_training)
    return {
        **last_metrics, "optimizer_steps": optimizer_steps, "early_stopped": early_stopped,
        "kl_at_stop": stop_kl, "sequence_chunks": len(chunks), "frames": len(frames),
        "raw_advantage_mean": float(advantage_mean), "raw_advantage_std": float(advantage_std),
        "raw_return_mean": float(raw_returns.mean()), "raw_return_std": float(raw_returns.std(unbiased=False)),
        "raw_reward_sum": sum(frame.reward for frame in frames),
        "value_error_scale": value_scale, "teacher_weight": teacher_weight,
        "actor_divisor": ACTOR_DIVISOR,
    }
