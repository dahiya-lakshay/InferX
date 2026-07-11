from app.engine.generation_config import GenerationConfig
from app.engine.request import GenerationRequest
from app.engine.sequence import Sequence
from app.engine.stopping import StoppingCriteria


def test_max_tokens():

    config = GenerationConfig(
        max_new_tokens=2,
    )

    request = GenerationRequest(
        request_id="1",
        prompt="Hello",
        config=config,
    )

    sequence = Sequence(request)

    sequence.append_token(10)
    sequence.append_token(20)

    stopping = StoppingCriteria()

    assert stopping.should_stop(sequence)


def test_eos_token():

    config = GenerationConfig(
        eos_token_id=2,
    )

    request = GenerationRequest(
        request_id="1",
        prompt="Hello",
        config=config,
    )

    sequence = Sequence(request)

    sequence.append_token(2)

    stopping = StoppingCriteria()

    assert stopping.should_stop(sequence)


def test_continue():

    config = GenerationConfig(
        max_new_tokens=10,
        eos_token_id=2,
    )

    request = GenerationRequest(
        request_id="1",
        prompt="Hello",
        config=config,
    )

    sequence = Sequence(request)

    sequence.append_token(5)

    stopping = StoppingCriteria()

    assert stopping.should_stop(sequence) is False