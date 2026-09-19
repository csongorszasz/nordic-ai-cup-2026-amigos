import hashlib
import math
import os
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, Response
from pydantic_core import from_json, to_json

from src.benchmarking.config import read_json
from src.benchmarking.policies import Policy
from src.policies.config import RuntimeConfig
from src.policies.features import validate_step
from src.policies.runtime import create_policy
from src.serving.session import PolicySession, SessionConflict
from src.utils.DTOs import ObservationResponse, StepResponse

HOST = "0.0.0.0"
PORT = 9052

_GAME_STATUSES = {"running": "ok", "ok": "ok", "game_over": "game_over"}
_OBSERVATION_TYPES = {
    name.casefold(): name for name in ("Fruit", "Agent", "Predator", "Tree", "Edge")
}
_AGENT_FIELDS = tuple(ObservationResponse.model_fields)
_AGENT_NUMERIC_FIELDS = (
    "energy", "age", "speed", "sprint_speed", "hearing_radius",
    "vision_angle", "vision_range", "max_energy",
)


def _finite_number(value: object, field: str) -> float:
    if isinstance(value, bool) or type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{field} must be a finite number.")
    return float(value)


def normalize_submission_step(step: StepResponse | dict) -> StepResponse:
    """Translate the public verifier payload into the simulator's internal DTO vocabulary."""
    values = (
        step.model_dump(mode="python", exclude_unset=True)
        if isinstance(step, StepResponse) else step
    )
    if not isinstance(values, dict):
        raise ValueError("Request body must be a JSON object.")
    raw_status = values.get("game_status")
    status = _GAME_STATUSES.get(raw_status.casefold()) if isinstance(raw_status, str) else None
    if status is None:
        raise ValueError(f"Unknown game status: {raw_status!r}")
    values["game_status"] = status
    raw_agents = values.get("agent_status", [])
    if not isinstance(raw_agents, list):
        raise ValueError("agent_status must be a list.")
    count = values.get("n_agents", len(raw_agents))
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError("n_agents must be a nonnegative integer.")
    score = _finite_number(values.get("score"), "score")
    sim_time = _finite_number(values.get("sim_time", 0.0), "sim_time")
    agents = []
    for raw_agent in raw_agents:
        if not isinstance(raw_agent, dict):
            raise ValueError("Each agent_status entry must be an object.")
        missing = [name for name in _AGENT_FIELDS if name not in raw_agent]
        if missing:
            raise ValueError(f"Agent status is missing fields: {', '.join(missing)}.")
        agent_id = raw_agent["agent_id"]
        if isinstance(agent_id, bool) or not isinstance(agent_id, int) or agent_id < 0:
            raise ValueError("agent_id must be a nonnegative integer.")
        if not isinstance(raw_agent["biome"], str):
            raise ValueError("biome must be a string.")
        normalized_agent = dict(raw_agent)
        for field in _AGENT_NUMERIC_FIELDS:
            normalized_agent[field] = _finite_number(raw_agent[field], field)
        observations = raw_agent["observations"]
        if not isinstance(observations, list):
            raise ValueError("Agent observations must be a list.")
        for observation in observations:
            if not isinstance(observation, dict):
                raise ValueError("Each observation must be an object.")
            raw_type = observation.get("type")
            canonical = (
                _OBSERVATION_TYPES.get(raw_type.casefold())
                if isinstance(raw_type, str) else None
            )
            if canonical is None:
                raise ValueError(f"Unknown observation type: {raw_type!r}")
            observation["type"] = canonical
        agents.append(ObservationResponse.model_construct(
            **{name: normalized_agent[name] for name in _AGENT_FIELDS},
            _fields_set=set(_AGENT_FIELDS),
        ))
    provided = set(values)
    return StepResponse.model_construct(
        game_status=status,
        score=score,
        sim_time=sim_time,
        n_agents=count,
        agent_status=agents,
        _fields_set=provided | {"game_status", "n_agents", "agent_status"},
    )


def create_app(
    config_path: Path | None = None, *, policy_factory: Callable[[], Policy] | None = None,
    stateful: bool | None = None, single_stream: bool | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(application: FastAPI):
        factory = policy_factory
        uses_state = stateful
        agreement = os.environ.get("SURVIVAL_SINGLE_STREAM", "0")
        if agreement not in ("0", "1"):
            raise ValueError("SURVIVAL_SINGLE_STREAM must be 0 or 1.")
        allowed = agreement == "1" if single_stream is None else single_stream
        kind = "injected"
        if factory is None:
            path = config_path or Path(os.environ.get(
                "SURVIVAL_POLICY_CONFIG", str(Path(__file__).parent / "configs" / "controller.json"),
            ))
            options = RuntimeConfig.model_validate(read_json(path))
            seed = int(os.environ.get("SURVIVAL_POLICY_SEED", "1"))
            if seed < 0 or seed > 2**32 - 1:
                raise ValueError("SURVIVAL_POLICY_SEED must be a uint32.")
            kind = options.policy
            factory = lambda: create_policy(seed, options.model_dump(mode="json"))
        application.state.session = PolicySession(
            factory, stateful=uses_state, single_stream=allowed,
        )
        application.state.policy_kind = kind
        yield

    application = FastAPI(title="Survival Simulator Agent Endpoint", lifespan=lifespan)

    @application.post("/predict")
    async def predict(request: Request):
        try:
            body = await request.body()
            normalized_step = normalize_submission_step(from_json(
                body, allow_inf_nan=False, cache_strings="keys",
            ))
            validate_step(normalized_step)
            signature = hashlib.blake2b(body, digest_size=16).digest()
            actions = application.state.session.predict(
                normalized_step, validated=True, signature=signature,
            )
        except SessionConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return Response(
            content=to_json({
                "actions": [action.model_dump(mode="json") for action in actions],
            }),
            media_type="application/json",
        )

    @application.get("/")
    def index():
        return {"message": "Agent endpoint running!", "policy": application.state.policy_kind}

    return application


app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=HOST, port=PORT)