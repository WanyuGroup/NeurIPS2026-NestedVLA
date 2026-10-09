from __future__ import annotations

import torch
from torch import Tensor


def make_topk_mask(patches: Tensor, keep_ratio: float = 0.10) -> Tensor:
    """Return a per-patch magnitude mask with a fixed keep ratio."""

    if not 0 < keep_ratio <= 1:
        raise ValueError("keep_ratio must be in (0, 1]")
    flat = patches.abs().flatten(start_dim=1)
    keep = max(1, round(flat.shape[1] * keep_ratio))
    indices = flat.topk(keep, dim=1, sorted=False).indices
    mask = torch.zeros_like(flat, dtype=torch.bool)
    mask.scatter_(1, indices, True)
    return mask.view_as(patches)
