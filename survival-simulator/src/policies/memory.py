"""Detached, identity-keyed state for one policy session."""

from collections.abc import Sequence
from numbers import Integral

import torch
from torch import Tensor

from src.benchmarking.policies import validate_actions
from src.utils.DTOs import ActionRequest


def _agent_ids(agent_ids: Sequence[int]) -> tuple[int, ...]:
    ids = tuple(agent_ids)
    if (
        any(isinstance(value, bool) or not isinstance(value, Integral) or value < 0 for value in ids)
        or len(set(ids)) != len(ids)
    ):
        raise ValueError("Memory agent IDs must be unique nonnegative integers.")
    return ids


class PolicyMemory:
    def __init__(self):
        self.previous_actions: dict[int, ActionRequest] = {}
        self._hidden: dict[int, Tensor] = {}

    def reset(self) -> None:
        self._hidden.clear()
        self.previous_actions.clear()

    def prepare(
        self, agent_ids: Sequence[int], hidden_size: int, device: torch.device | str,
    ) -> Tensor:
        """Return an independent detached initial-state snapshot; prune deaths."""
        ids = _agent_ids(agent_ids)
        if isinstance(hidden_size, bool) or not isinstance(hidden_size, Integral) or hidden_size <= 0:
            raise ValueError("Memory hidden_size must be a positive integer.")
        device = torch.device(device)
        retained = [self._hidden[agent_id] for agent_id in ids if agent_id in self._hidden]
        if any(value.shape != (hidden_size,) for value in retained):
            raise ValueError("Stored hidden state does not match hidden_size.")
        dtype = retained[0].dtype if retained else torch.float32
        rows = [
            self._hidden[agent_id].to(device=device, dtype=dtype)
            if agent_id in self._hidden else torch.zeros(hidden_size, device=device, dtype=dtype)
            for agent_id in ids
        ]
        prepared = torch.stack(rows).detach() if rows else torch.zeros(
            (0, hidden_size), device=device, dtype=dtype,
        )
        states = {agent_id: prepared[index].clone() for index, agent_id in enumerate(ids)}
        previous = {agent_id: self.previous_actions[agent_id]
                    for agent_id in ids if agent_id in self.previous_actions}
        self._hidden, self.previous_actions = states, previous
        return prepared

    def commit(
        self, agent_ids: Sequence[int], next_hidden: Tensor, actions: Sequence[ActionRequest],
    ) -> None:
        ids = _agent_ids(agent_ids)
        if (
            not isinstance(next_hidden, Tensor) or next_hidden.ndim != 2
            or next_hidden.shape[0] != len(ids) or next_hidden.shape[1] == 0
            or not next_hidden.is_floating_point() or not torch.isfinite(next_hidden).all().item()
        ):
            raise ValueError("Next hidden state must be finite with shape [N, hidden_size].")
        if any(
            self._hidden[agent_id].shape != next_hidden.shape[1:]
            for agent_id in ids if agent_id in self._hidden
        ):
            raise ValueError("Next hidden state does not match the prepared hidden_size.")
        validated = validate_actions(actions, ids)
        states = {agent_id: next_hidden[index].detach().clone() for index, agent_id in enumerate(ids)}
        previous = {action.agent_id: action for action in validated}
        self._hidden, self.previous_actions = states, previous
