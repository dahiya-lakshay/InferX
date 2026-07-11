from app.engine.generation_config import GenerationConfig
from app.engine.request import GenerationRequest
from app.engine.sequence import Sequence


def test_sequence_creation():

    request = GenerationRequest(
        request_id="1",
        prompt="Hello",
        config=GenerationConfig(),
    )

    sequence = Sequence(
        request=request,
    )

    assert sequence.current_length == 0
    assert sequence.is_finished is False


def test_append_token():

    request = GenerationRequest(
        request_id="1",
        prompt="Hello",
        config=GenerationConfig(),
    )

    sequence = Sequence(
        request=request,
    )

    sequence.append_token(42)

    sequence.append_token(17)

    assert sequence.generated_tokens == [42, 17]

    assert sequence.current_length == 2


def test_all_tokens():

    request = GenerationRequest(
        request_id="1",
        prompt="Hello",
        config=GenerationConfig(),
    )

    sequence = Sequence(
        request=request,
        token_ids=[1, 2, 3],
    )

    sequence.append_token(4)

    sequence.append_token(5)

    assert sequence.all_tokens == [
        1,
        2,
        3,
        4,
        5,
    ]