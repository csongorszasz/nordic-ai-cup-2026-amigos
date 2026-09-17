import os
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Body, FastAPI, HTTPException

from src.benchmarking.config import read_json
from src.benchmarking.policies import Policy
from src.policies.config import RuntimeConfig
from src.policies.features import validate_step
from src.policies.runtime import create_policy
from src.serving.session import PolicySession, SessionConflict
from src.utils.DTOs import StepResponse

HOST = "0.0.0.0"
PORT = 9052

def create_app(
    config_path: Path | None = None, *, policy_factory: Callable[[], Policy] | None = None,
    stateful: bool = False, single_stream: bool | None = None,
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
            uses_state = options.policy == "neural"
            kind = options.policy
            factory = lambda: create_policy(seed, options.model_dump(mode="json"))
        application.state.session = PolicySession(
            factory, stateful=uses_state, single_stream=allowed,
        )
        application.state.policy_kind = kind
        yield

    application = FastAPI(title="Survival Simulator Agent Endpoint", lifespan=lifespan)

    @application.post("/predict")
    def predict(step: StepResponse = Body(...)):
        try:
            validate_step(step)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        try:
            actions = application.state.session.predict(step)
        except SessionConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {"actions": [action.model_dump(mode="json") for action in actions]}

    @application.get("/")
    def index():
        return {"message": "Agent endpoint running!", "policy": application.state.policy_kind}

    return application


app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=HOST, port=PORT)