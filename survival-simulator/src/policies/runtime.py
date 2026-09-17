"""The benchmark/serving factory; heuristic use does not import PyTorch."""

from pathlib import Path

from src.benchmarking.policies import Policy
from src.policies.config import RuntimeConfig


def create_policy(seed: int, config: dict) -> Policy:
    settings = RuntimeConfig.model_validate(config)
    if settings.policy == "heuristic":
        from src.policies.heuristic import HeuristicPolicy

        return HeuristicPolicy(seed, settings.heuristic)
    import torch
    from src.policies.neural import NeuralPolicy
    from src.training.artifacts import load_checkpoint

    if settings.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("The configured CUDA inference device is unavailable.")
    torch.set_num_threads(settings.torch_threads)
    checkpoint = load_checkpoint(
        Path(settings.checkpoint), device=settings.device,
        expected_sha256=settings.checkpoint_sha256,
    )
    return NeuralPolicy(checkpoint.network, seed, deterministic=True)
