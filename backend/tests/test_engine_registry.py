import pytest

from app.core.engine_registry import EngineRegistry


class DummyEngine:
    pass


def test_register_engine():

    registry = EngineRegistry()

    engine = DummyEngine()

    registry.register(engine)

    assert registry.get() is engine


def test_unregistered_engine():

    registry = EngineRegistry()

    with pytest.raises(RuntimeError):

        registry.get()