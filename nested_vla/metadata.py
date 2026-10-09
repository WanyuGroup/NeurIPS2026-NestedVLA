from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Mapping

import torch


@dataclass(frozen=True)
class ParameterSlot:
    """Patch layout for one tensor in a parameter delta."""

    name: str
    shape: tuple[int, ...]
    rows: int
    cols: int
    patches_h: int
    patches_w: int
    start: int
    stop: int
    parameter_type: int
    layer: int


@dataclass(frozen=True)
class ParameterMetadata:
    patch_size: int
    slots: tuple[ParameterSlot, ...]
    parameter_types: tuple[str, ...]
    total_patches: int
    max_layer: int

    @staticmethod
    def _matrix_shape(tensor: torch.Tensor) -> tuple[int, int]:
        if tensor.ndim == 0:
            return 1, 1
        if tensor.ndim == 1:
            return 1, tensor.numel()
        return tensor.shape[0], math.prod(tensor.shape[1:])

    @staticmethod
    def _layer_id(name: str) -> int:
        match = re.search(r"(?:layers?|blocks?)\.(\d+)", name)
        return int(match.group(1)) if match else 0

    @staticmethod
    def _parameter_type(name: str) -> str:
        parts = name.split(".")
        if len(parts) >= 2:
            return ".".join(parts[-2:])
        return parts[-1]

    @classmethod
    def from_state_dict(
        cls,
        state_dict: Mapping[str, torch.Tensor],
        patch_size: int,
    ) -> "ParameterMetadata":
        if patch_size <= 0:
            raise ValueError("patch_size must be positive")
        if not state_dict:
            raise ValueError("state_dict must not be empty")

        type_names = sorted({cls._parameter_type(name) for name in state_dict})
        type_to_id = {name: index for index, name in enumerate(type_names)}
        slots: list[ParameterSlot] = []
        cursor = 0
        max_layer = 0

        for name, tensor in state_dict.items():
            rows, cols = cls._matrix_shape(tensor)
            patches_h = math.ceil(rows / patch_size)
            patches_w = math.ceil(cols / patch_size)
            count = patches_h * patches_w
            layer = cls._layer_id(name)
            max_layer = max(max_layer, layer)
            slots.append(
                ParameterSlot(
                    name=name,
                    shape=tuple(tensor.shape),
                    rows=rows,
                    cols=cols,
                    patches_h=patches_h,
                    patches_w=patches_w,
                    start=cursor,
                    stop=cursor + count,
                    parameter_type=type_to_id[cls._parameter_type(name)],
                    layer=layer,
                )
            )
            cursor += count

        return cls(
            patch_size=patch_size,
            slots=tuple(slots),
            parameter_types=tuple(type_names),
            total_patches=cursor,
            max_layer=max_layer,
        )
