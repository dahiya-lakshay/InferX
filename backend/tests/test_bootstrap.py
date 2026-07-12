from app.core.bootstrap import bootstrap
from app.core.engine_registry import engine_registry


def test_bootstrap():

    bootstrap()

    assert engine_registry.get() is not None