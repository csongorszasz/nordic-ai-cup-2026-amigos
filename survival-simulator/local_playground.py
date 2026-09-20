"""Render random, heuristic, or trained policies in the local simulator."""

import argparse
import random
import sys
from pathlib import Path

import pygame

from src.benchmarking.config import read_json
from src.benchmarking.policies import RandomPolicy, validate_actions
from src.core import SimulationCore
from src.policies.runtime import create_policy
from src.utils.DTOs import ObservationResponse, StepResponse


def local_simulation(verbose=True, *, config=None, seed=None, fps=60, max_steps=None):
    if seed is None:
        seed = random.randint(0, 2**32 - 1)
    policy = (
        RandomPolicy(seed) if config is None
        else create_policy(seed, read_json(Path(config)))
    )
    print(f"Seed: {seed}")

    pygame.init()
    try:
        sim = SimulationCore(seed=seed)
        screen, clock, font = None, None, None
        if verbose:
            info = pygame.display.Info()
            scale = 0.9 * min(info.current_w / sim.env_width, info.current_h / sim.env_height)
            size = (max(1, int(sim.env_width * scale)), max(1, int(sim.env_height * scale)))
            screen = pygame.display.set_mode(size, pygame.SCALED)
            pygame.display.set_caption(f"Survival Simulator | Seed: {seed}")
            clock = pygame.time.Clock()
            font = pygame.font.SysFont(None, 24)

        actions = []
        ticks = 0
        while True:
            if verbose and any(
                event.type == pygame.QUIT
                or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE)
                for event in pygame.event.get()
            ):
                break

            state = sim.step(actions)
            ticks += 1
            status = (
                f'Score: {state["score"]:.2f} | Agents alive: {state["num_agents"]} | '
                f'Time: {state["sim_time"]:.2f}'
            )
            if verbose:
                sim.env.draw(screen)
                screen.blit(font.render(status, True, (255, 255, 255)), (20, 20))
                pygame.display.flip()
                clock.tick(fps)
            print(status)

            if state["num_agents"] == 0 or state["sim_time"] > 3000:
                print(f"Game over! Final Score: {state['score']}")
                break
            if max_steps is not None and ticks >= max_steps:
                print(f"Step limit reached. Score: {state['score']}")
                break

            step = StepResponse(
                game_status="ok", score=state["score"], sim_time=state["sim_time"],
                n_agents=state["num_agents"],
                agent_status=[ObservationResponse.model_validate(value) for value in state["observations"]],
            )
            proposed = validate_actions(policy.act(step), [agent.agent_id for agent in step.agent_status])
            actions = [(action.agent_id, action) for action in proposed]
    finally:
        pygame.quit()


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--config", type=Path, help="Runtime policy JSON; omit for the random policy.")
    cli.add_argument("--seed", type=int, help="World and policy seed; random when omitted.")
    cli.add_argument("--fps", type=int, default=60, help="Maximum rendered frames per second (default: 60).")
    cli.add_argument("--headless", action="store_true", help="Run without a window or frame-rate cap.")
    cli.add_argument("--max-steps", type=int, help="Stop after this many simulator ticks, including bootstrap.")
    args = cli.parse_args(argv)
    if args.seed is not None and not 0 <= args.seed <= 2**32 - 1:
        cli.error("--seed must be an integer between 0 and 2^32 - 1.")
    if args.fps <= 0:
        cli.error("--fps must be positive.")
    if args.max_steps is not None and args.max_steps <= 0:
        cli.error("--max-steps must be positive.")
    try:
        local_simulation(
            verbose=not args.headless, config=args.config, seed=args.seed,
            fps=args.fps, max_steps=args.max_steps,
        )
    except (OSError, ValueError, ImportError) as exc:
        print(f"Playground error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
