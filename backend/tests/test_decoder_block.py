import torch

from app.engine.decoder_block import DecoderBlock


def test_decoder_output_shape():

    block = DecoderBlock(
        embedding_dim=512,
        num_heads=8,
        hidden_dim=2048,
    )

    x = torch.randn(2, 16, 512)

    output, weights = block(x)

    assert output.shape == (2, 16, 512)

    assert weights.shape == (2, 8, 16, 16)


def test_gradient_flow():

    block = DecoderBlock(
        embedding_dim=256,
        num_heads=8,
        hidden_dim=1024,
    )

    x = torch.randn(
        2,
        10,
        256,
        requires_grad=True,
    )

    output, _ = block(x)

    loss = output.mean()

    loss.backward()

    assert x.grad is not None