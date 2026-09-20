"""Ragged, permutation-equivariant actors with explicit recurrent state."""

import math
from dataclasses import dataclass
from numbers import Integral

import numpy as np
import torch
from torch import Tensor, nn

from src.policies.config import ModelConfig
from src.policies.features import (
    ENTITY_DIM, ENTITY_TYPES, SCALAR_DIM, PUBLIC_SCALAR_DIM, FeatureBatch, feature_version,
)


@dataclass(frozen=True)
class PolicyOutput:
    mean: Tensor
    log_std: Tensor
    spawn_logits: Tensor
    value: Tensor
    next_hidden: Tensor
    angle_vectors: Tensor | None = None


class PolicyNetwork(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        if not isinstance(config, ModelConfig):
            raise TypeError("PolicyNetwork requires a validated ModelConfig.")
        self.config = config
        hidden, entity = config.hidden_size, config.entity_size
        self.entity_encoders = nn.ModuleList([
            nn.Sequential(nn.Linear(ENTITY_DIM, entity), nn.SiLU(), nn.Linear(entity, entity), nn.SiLU())
            for _ in ENTITY_TYPES
        ])
        self.scalar_dim = PUBLIC_SCALAR_DIM if config.public_context else SCALAR_DIM
        self.scalar_encoder = nn.Sequential(nn.Linear(self.scalar_dim, hidden), nn.SiLU())
        self.attention_queries = nn.ModuleList(
            [nn.Linear(hidden, entity) for _ in ENTITY_TYPES] if config.encoder == "attention" else []
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden + len(ENTITY_TYPES) * (2 * entity + 1), hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
        )
        self.recurrent = nn.GRUCell(hidden, hidden) if config.memory == "gru" else None
        self.actor = nn.Sequential(
            nn.Linear(hidden * (3 if config.team_context else 1), hidden), nn.SiLU(),
        )
        self.mean_head = nn.Linear(hidden, 3)
        self.spawn_head = nn.Linear(hidden, 1)
        self.angle_vectors = nn.Linear(hidden, 4) if config.angle_head == "vector_bc" else None
        if self.angle_vectors is not None:
            nn.init.zeros_(self.angle_vectors.weight)
            with torch.no_grad():
                self.angle_vectors.bias.copy_(torch.tensor([1.0, 0.0, 1.0, 0.0]))
        fraction = (config.initial_log_std - config.log_std_min) / (config.log_std_max - config.log_std_min)
        self.log_std_parameter = nn.Parameter(torch.full((3,), math.log(fraction / (1 - fraction))))
        self.critic = nn.Sequential(
            nn.Linear(hidden * (2 if config.critic == "team" else 1), hidden), nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def _inputs(self, batch: FeatureBatch, hidden: Tensor | None) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        if not isinstance(batch, FeatureBatch):
            raise TypeError("PolicyNetwork requires a FeatureBatch.")
        ids = batch.agent_ids
        if (
            any(isinstance(value, bool) or not isinstance(value, Integral) or value < 0 for value in ids)
            or len(set(ids)) != len(ids)
        ):
            raise ValueError("Feature agent IDs must be unique nonnegative integers.")
        count = len(ids)
        if not isinstance(batch.entities, np.ndarray) or batch.entities.ndim != 2:
            raise ValueError("Entity features must have shape [M, ENTITY_DIM].")
        entities = batch.entities.shape[0]
        for name, shape, integer in (
            ("scalars", (count, self.scalar_dim), False),
            ("entities", (entities, ENTITY_DIM), False),
            ("entity_types", (entities,), True), ("entity_owners", (entities,), True),
        ):
            array = getattr(batch, name)
            if not isinstance(array, np.ndarray) or array.shape != shape:
                raise ValueError(f"Feature {name} has an invalid shape.")
            if integer:
                if array.dtype != np.int64:
                    raise ValueError(f"Feature {name} must use int64 indices.")
            elif not np.issubdtype(array.dtype, np.floating) or not np.isfinite(array).all():
                raise ValueError(f"Feature {name} must contain finite floating-point values.")
        if batch.feature_version != feature_version(self.config.public_context, self.config.peer_context):
            raise ValueError("Feature schema does not match the model configuration.")
        if ((batch.entity_types < 0) | (batch.entity_types >= len(ENTITY_TYPES))).any():
            raise ValueError("Unknown entity type index.")
        if ((batch.entity_owners < 0) | (batch.entity_owners >= count)).any():
            raise ValueError("Entity owner must index a living agent.")
        reference = self.mean_head.weight
        scalars = torch.as_tensor(
            np.ascontiguousarray(batch.scalars), device=reference.device, dtype=reference.dtype,
        )
        objects = torch.as_tensor(
            np.ascontiguousarray(batch.entities), device=reference.device, dtype=reference.dtype,
        )
        types = torch.as_tensor(np.ascontiguousarray(batch.entity_types), device=reference.device)
        owners = torch.as_tensor(np.ascontiguousarray(batch.entity_owners), device=reference.device)
        if hidden is None:
            hidden = reference.new_zeros((count, self.config.hidden_size))
        elif (
            not isinstance(hidden, Tensor) or hidden.shape != (count, self.config.hidden_size)
            or not hidden.is_floating_point() or not torch.isfinite(hidden).all().item()
        ):
            raise ValueError("Hidden state must be finite with shape [N, hidden_size].")
        hidden = hidden.to(device=reference.device, dtype=reference.dtype)
        if any(not torch.isfinite(value).all().item() for value in (scalars, objects, hidden)):
            raise ValueError("Features or hidden state overflow the model dtype.")
        return scalars, objects, types, owners, hidden

    def _pool_entities(self, scalar: Tensor, entities: Tensor, types: Tensor, owners: Tensor) -> Tensor:
        count, width = scalar.shape[0], self.config.entity_size
        pooled = []
        for kind, encoder in enumerate(self.entity_encoders):
            indices = torch.nonzero(types == kind, as_tuple=False).flatten()
            if indices.numel() == 0:
                pooled.append(scalar.new_zeros((count, 2 * width + 1)))
                continue
            # Chunk MLP work, not observations. Storage is ragged O(M * entity_size).
            encoded = torch.cat([
                encoder(entities.index_select(0, chunk))
                for chunk in indices.split(self.config.entity_chunk_size)
            ], dim=0)
            owner = owners.index_select(0, indices)
            counts = torch.bincount(owner, minlength=count).to(dtype=scalar.dtype).unsqueeze(1)
            maximum = scalar.new_full((count, width), -torch.inf).scatter_reduce(
                0, owner.unsqueeze(1).expand_as(encoded), encoded, reduce="amax",
            )
            maximum = torch.where(counts > 0, maximum, torch.zeros_like(maximum))
            if self.config.encoder == "attention":
                query = self.attention_queries[kind](scalar).index_select(0, owner)
                scores = (query * encoded).sum(1) / math.sqrt(width)
                offset = scalar.new_full((count,), -torch.inf).scatter_reduce(
                    0, owner, scores.detach(), reduce="amax",
                )
                weights = (scores - offset.index_select(0, owner)).exp()
                denominator = scalar.new_zeros(count).index_add(0, owner, weights).unsqueeze(1)
                aggregate = scalar.new_zeros((count, width)).index_add(
                    0, owner, encoded * weights.unsqueeze(1),
                ) / denominator.clamp_min(1)
            else:
                aggregate = scalar.new_zeros((count, width)).index_add(0, owner, encoded)
                aggregate = aggregate / counts.clamp_min(1)
            pooled.append(torch.cat((aggregate, maximum, counts.log1p()), dim=1))
        return torch.cat(pooled, dim=1)

    @staticmethod
    def _team_pool(features: Tensor) -> Tensor:
        if features.shape[0] == 0:
            zero = features.sum(0)
            return torch.cat((zero, zero))
        return torch.cat((features.mean(0), features.amax(0)))

    def encode_agents(self, batch: FeatureBatch, hidden: Tensor | None = None) -> tuple[Tensor, Tensor]:
        scalars, entities, types, owners, hidden = self._inputs(batch, hidden)
        scalar = self.scalar_encoder(scalars)
        objects = self._pool_entities(scalar, entities, types, owners)
        features = self.fusion(torch.cat((scalar, objects), dim=1))
        return features, hidden

    def decode_agents(
        self, features: Tensor, groups: Tensor | None = None, group_count: int = 1,
    ) -> PolicyOutput:
        if groups is None:
            team = self._team_pool(features)
            context = team.unsqueeze(0).expand(features.shape[0], -1)
        else:
            if (groups.shape != (len(features),) or groups.dtype != torch.long
                    or groups.device != features.device or group_count < 1
                    or ((groups < 0) | (groups >= group_count)).any().item()):
                raise ValueError("Invalid independent-team group indices.")
            counts = torch.bincount(groups, minlength=group_count).to(features.dtype).unsqueeze(1)
            if (counts == 0).any().item():
                raise ValueError("Every grouped team must have at least one agent.")
            average = features.new_zeros((group_count, features.shape[1])).index_add(0, groups, features) / counts
            maximum = features.new_full((group_count, features.shape[1]), -torch.inf).scatter_reduce(
                0, groups.unsqueeze(1).expand_as(features), features, reduce="amax",
            )
            team = torch.cat((average, maximum), dim=1)
            context = team.index_select(0, groups)
        actor_features = (
            torch.cat((features, context), dim=1)
            if self.config.team_context else features
        )
        actor = self.actor(actor_features)
        if self.config.critic == "team":
            value = self.critic(team).squeeze(-1)
            if features.shape[0] == 0:
                value = value * 0
        else:
            local_values = self.critic(features).squeeze(-1)
            if groups is None:
                value = local_values.mean() if features.shape[0] else local_values.sum()
            else:
                value = local_values.new_zeros(group_count).index_add(0, groups, local_values) / counts.squeeze(1)
        mean = self.mean_head(actor)
        vectors = self.angle_vectors(actor).reshape(-1, 2, 2) if self.angle_vectors is not None else None
        if vectors is not None:
            # BC trains the Cartesian vectors directly. Runtime retains the existing
            # action DTO, but no gradient passes through the atan2 seam or zero norm.
            angles = torch.atan2(vectors[:, :, 1], vectors[:, :, 0]).detach() / math.pi
            bound = 1 - torch.finfo(angles.dtype).eps
            mean = torch.cat((mean[:, :1], torch.atanh(angles.clamp(-bound, bound))), dim=1)
        output = PolicyOutput(
            mean,
            (self.config.log_std_min + (self.config.log_std_max - self.config.log_std_min)
             * self.log_std_parameter.sigmoid()).expand(features.shape[0], 3),
            self.spawn_head(actor).squeeze(-1), value, features, vectors,
        )
        checked = (
            output.mean, output.log_std, output.spawn_logits, output.value, output.next_hidden,
        ) + ((vectors,) if vectors is not None else ())
        if any(not torch.isfinite(value).all().item() for value in checked):
            raise ValueError("Policy network produced non-finite output.")
        return output

    def forward(self, batch: FeatureBatch, hidden: Tensor | None = None) -> PolicyOutput:
        features, hidden = self.encode_agents(batch, hidden)
        if self.recurrent is not None and features.shape[0]:
            features = self.recurrent(features, hidden)
        return self.decode_agents(features)
