from app.engine.generation_config import GenerationConfig
from app.engine.request import GenerationRequest


def test_request_creation():

    config = GenerationConfig()

    request = GenerationRequest(
        request_id="1",
        prompt="Hello InferX",
        config=config,
    )

    assert request.request_id == "1"
    assert request.prompt == "Hello InferX"
    assert request.config is config