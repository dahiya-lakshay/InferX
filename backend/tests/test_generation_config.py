from app.engine.generation_config import GenerationConfig


def test_default_values():

    config = GenerationConfig()

    assert config.max_new_tokens == 50
    assert config.temperature == 1.0
    assert config.top_k == 50
    assert config.top_p == 1.0
    assert config.do_sample is False