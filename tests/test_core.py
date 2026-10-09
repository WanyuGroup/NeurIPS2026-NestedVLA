import torch

from nested_vla import DeltaTokenizer, NestedVLAGenerator, make_topk_mask


def test_tokenizer_round_trip_and_generator():
    torch.manual_seed(0)
    delta = {
        "layers.0.attn.weight": torch.randn(9, 7),
        "layers.1.mlp.weight": torch.randn(5, 11),
    }
    tokenizer = DeltaTokenizer.from_state_dict(delta, patch_size=4)
    patches = tokenizer.encode(delta)
    reconstructed = tokenizer.decode(patches)
    for name in delta:
        assert torch.equal(delta[name], reconstructed[name])

    mask = make_topk_mask(patches, keep_ratio=0.25)
    scales = tokenizer.patch_scales(patches)
    target = tokenizer.standardize(patches, scales)
    model = NestedVLAGenerator(
        tokenizer.metadata,
        num_tasks=2,
        task_dim=16,
        slot_dim=8,
        hidden_dim=24,
        grid_channels=4,
        grid_size=4,
    )
    prediction = model(torch.tensor([0]))
    assert prediction.shape == (1, *patches.shape)
    loss = model.masked_loss(prediction, target[None], mask[None])
    loss.backward()
    generated = model.generate_delta(0, tokenizer, mask, scales)
    assert generated.keys() == delta.keys()
    assert all(generated[name].shape == delta[name].shape for name in delta)
