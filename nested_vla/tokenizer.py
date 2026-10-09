from __future__ import annotations

from collections import OrderedDict
from typing import Mapping

import torch
from torch import Tensor

from .metadata import ParameterMetadata


class DeltaTokenizer:
    """Convert a parameter delta to and from fixed-size 2D patches."""

    def __init__(self, metadata: ParameterMetadata):
        self.metadata = metadata

    @classmethod
    def from_state_dict(
        cls,
        state_dict: Mapping[str, Tensor],
        patch_size: int = 256,
    ) -> "DeltaTokenizer":
        return cls(ParameterMetadata.from_state_dict(state_dict, patch_size))

    def encode(self, state_dict: Mapping[str, Tensor]) -> Tensor:
        p = self.metadata.patch_size
        patches: list[Tensor] = []
        for slot in self.metadata.slots:
            if slot.name not in state_dict:
                raise KeyError(f"missing parameter: {slot.name}")
            matrix = state_dict[slot.name].reshape(slot.rows, slot.cols)
            padded = matrix.new_zeros(slot.patches_h * p, slot.patches_w * p)
            padded[: slot.rows, : slot.cols] = matrix
            patches.extend(
                padded[i : i + p, j : j + p]
                for i in range(0, padded.shape[0], p)
                for j in range(0, padded.shape[1], p)
            )
        return torch.stack(patches)

    def decode(self, patches: Tensor) -> OrderedDict[str, Tensor]:
        if patches.shape != (
            self.metadata.total_patches,
            self.metadata.patch_size,
            self.metadata.patch_size,
        ):
            raise ValueError("patch tensor does not match tokenizer metadata")

        p = self.metadata.patch_size
        result: OrderedDict[str, Tensor] = OrderedDict()
        for slot in self.metadata.slots:
            canvas = patches.new_zeros(slot.patches_h * p, slot.patches_w * p)
            index = slot.start
            for row in range(slot.patches_h):
                for col in range(slot.patches_w):
                    canvas[row * p : (row + 1) * p, col * p : (col + 1) * p] = patches[index]
                    index += 1
            result[slot.name] = canvas[: slot.rows, : slot.cols].reshape(slot.shape)
        return result

    @staticmethod
    def patch_scales(patches: Tensor, eps: float = 1e-6) -> Tensor:
        return patches.square().mean(dim=(-2, -1)).sqrt().clamp_min(eps)

    @staticmethod
    def standardize(patches: Tensor, scales: Tensor) -> Tensor:
        return patches / scales[:, None, None]

    @staticmethod
    def destandardize(patches: Tensor, scales: Tensor) -> Tensor:
        return patches * scales[:, None, None]
