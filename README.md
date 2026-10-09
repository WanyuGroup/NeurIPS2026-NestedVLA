# NestedVLA

Minimal algorithm release for **NestedVLA: Learning to Consolidate and
Generate Skills for Vision-Language-Action Models**.

This repository contains the method components needed to:

1. turn action-expert parameter deltas into fixed-size patch tokens;
2. compute a frozen top-(K) support mask and per-patch scales;
3. train a task-conditioned hypernetwork on the tokenized deltas; and
4. generate and reconstruct a task-specific action-expert update.

The release intentionally excludes benchmark datasets, robot simulators,
training checkpoints, and experiment-specific launch files.

## Install

```bash
pip install -e .
```

Only PyTorch is required.

## Minimal example

```python
import torch
from nested_vla import (
    DeltaTokenizer,
    NestedVLAGenerator,
    make_topk_mask,
)

deltas = {
    "transformer.layers.0.attn.weight": torch.randn(24, 16),
    "transformer.layers.0.mlp.weight": torch.randn(32, 24),
}

tokenizer = DeltaTokenizer.from_state_dict(deltas, patch_size=8)
target = tokenizer.encode(deltas)
mask = make_topk_mask(target, keep_ratio=0.10)
scales = tokenizer.patch_scales(target)
standardized = tokenizer.standardize(target, scales)

model = NestedVLAGenerator(
    metadata=tokenizer.metadata,
    num_tasks=3,
    task_dim=64,
    hidden_dim=128,
)
task_id = torch.tensor([0])
prediction = model(task_id)
loss = model.masked_loss(prediction, standardized.unsqueeze(0), mask.unsqueeze(0))
loss.backward()

generated = model.generate_delta(
    task_id=0,
    tokenizer=tokenizer,
    mask=mask,
    scales=scales,
)
```

The example uses synthetic tensors. In a VLA pipeline, `deltas` are the
action-expert differences between independently fine-tuned policies and the
frozen base policy.

## Scope

This initial release focuses on the core algorithm and a stable tensor-level
API. Integration with individual VLA frameworks and released checkpoints will
be added separately.

## Citation

```bibtex
@inproceedings{du2026nestedvla,
  title={NestedVLA: Learning to Consolidate and Generate Skills for Vision-Language-Action Models},
  author={Du, Guodong and Lin, Wanyu},
  booktitle={Advances in Neural Information Processing Systems},
  year={2026}
}
```

## License

MIT
