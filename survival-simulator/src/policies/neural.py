"""Neural inference for a single explicitly resettable game session."""

from numbers import Integral

import torch

from src.policies.actions import sample_actions
from src.policies.features import encode_step, validate_step
from src.policies.memory import PolicyMemory
from src.policies.networks import PolicyNetwork
from src.utils.DTOs import ActionRequest, StepResponse


class NeuralPolicy:
    stateful = True

    def __init__(self, network: PolicyNetwork, seed: int, *, deterministic: bool = True):
        if isinstance(seed, bool) or not isinstance(seed, Integral):
            raise ValueError("Policy seed must be an integer.")
        self.network = network
        self.memory = PolicyMemory()
        self.seed = int(seed)
        self.deterministic = deterministic
        self._generators: dict[torch.device, torch.Generator] = {}
        self._generator(next(network.parameters()).device)

    def _generator(self, device: torch.device) -> torch.Generator:
        if device not in self._generators:
            self._generators[device] = torch.Generator(device=device).manual_seed(self.seed)
        return self._generators[device]

    def reset(self) -> None:
        self.memory.reset()
        for generator in self._generators.values():
            generator.manual_seed(self.seed)

    def act(self, step: StepResponse) -> list[ActionRequest]:
        validate_step(step)
        return self.act_validated(step)

    def act_validated(self, step: StepResponse) -> list[ActionRequest]:
        # Canonical evaluation also removes reduction-order differences across DTO permutations.
        ordered = step.model_copy(update={
            "agent_status": sorted(step.agent_status, key=lambda agent: agent.agent_id),
        })
        batch = encode_step(ordered, self.memory.previous_actions, validate=False)
        if not batch.agent_ids:
            self.reset()
            return []
        device = next(self.network.parameters()).device
        hidden = self.memory.prepare(batch.agent_ids, self.network.config.hidden_size, device)
        self.network.eval()
        with torch.inference_mode():
            output = self.network(batch, hidden)
            sample = sample_actions(
                output, ordered, deterministic=self.deterministic, generator=self._generator(device),
            )
        self.memory.commit(batch.agent_ids, output.next_hidden, sample.actions)
        by_id = {action.agent_id: action for action in sample.actions}
        return [by_id[agent.agent_id] for agent in step.agent_status]
