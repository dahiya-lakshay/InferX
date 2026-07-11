import torch

from app.engine.kv_cache import KVCache


def test_empty_cache():

    cache = KVCache()

    keys, values = cache.get()

    assert keys is None

    assert values is None


def test_single_update():

    cache = KVCache()

    k = torch.randn(2, 8, 5, 64)

    v = torch.randn(2, 8, 5, 64)

    keys, values = cache.update(
        k,
        v,
    )

    assert keys.shape == (
        2,
        8,
        5,
        64,
    )

    assert values.shape == (
        2,
        8,
        5,
        64,
    )


def test_multiple_updates():

    cache = KVCache()

    cache.update(
        torch.randn(2, 8, 4, 64),
        torch.randn(2, 8, 4, 64),
    )

    keys, values = cache.update(
        torch.randn(2, 8, 3, 64),
        torch.randn(2, 8, 3, 64),
    )

    assert keys.shape == (
        2,
        8,
        7,
        64,
    )

    assert values.shape == (
        2,
        8,
        7,
        64,
    )


def test_clear():

    cache = KVCache()

    cache.update(
        torch.randn(1, 8, 4, 32),
        torch.randn(1, 8, 4, 32),
    )

    cache.clear()

    keys, values = cache.get()

    assert keys is None

    assert values is None