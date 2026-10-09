from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .metadata import ParameterMetadata

if TYPE_CHECKING:
    from .tokenizer import DeltaTokenizer


class HyperConvBlock(nn.Module):
    """Small convolutional encoder for a learned task grid."""

    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, channels * 2, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels * 2, channels * 2, 3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x).flatten(1)


class NestedVLAGenerator(nn.Module):
    """Task-conditioned generator for tokenized action-expert deltas."""

    def __init__(
        self,
        metadata: ParameterMetadata,
        num_tasks: int,
        *,
        task_dim: int = 256,
        slot_dim: int = 64,
        hidden_dim: int = 512,
        grid_channels: int = 32,
        grid_size: int = 8,
        out_of_mask_weight: float = 0.05,
    ):
        super().__init__()
        if num_tasks <= 0:
            raise ValueError("num_tasks must be positive")
        self.metadata = metadata
        self.num_tasks = num_tasks
        self.out_of_mask_weight = out_of_mask_weight
        self.grid_channels = grid_channels
        self.grid_size = grid_size

        self.task_embedding = nn.Embedding(num_tasks, task_dim)
        self.task_to_grid = nn.Linear(task_dim, grid_channels * grid_size * grid_size)
        self.task_encoder = HyperConvBlock(grid_channels)

        self.type_embedding = nn.Embedding(len(metadata.parameter_types), slot_dim)
        self.layer_embedding = nn.Embedding(metadata.max_layer + 1, slot_dim)
        self.row_embedding = nn.Embedding(
            max(slot.patches_h for slot in metadata.slots), slot_dim
        )
        self.col_embedding = nn.Embedding(
            max(slot.patches_w for slot in metadata.slots), slot_dim
        )

        task_features = grid_channels * 2
        self.head = nn.Sequential(
            nn.Linear(task_features + slot_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, metadata.patch_size**2),
        )

        type_ids: list[int] = []
        layer_ids: list[int] = []
        row_ids: list[int] = []
        col_ids: list[int] = []
        for slot in metadata.slots:
            for row in range(slot.patches_h):
                for col in range(slot.patches_w):
                    type_ids.append(slot.parameter_type)
                    layer_ids.append(slot.layer)
                    row_ids.append(row)
                    col_ids.append(col)
        self.register_buffer("type_ids", torch.tensor(type_ids, dtype=torch.long))
        self.register_buffer("layer_ids", torch.tensor(layer_ids, dtype=torch.long))
        self.register_buffer("row_ids", torch.tensor(row_ids, dtype=torch.long))
        self.register_buffer("col_ids", torch.tensor(col_ids, dtype=torch.long))

    def forward(self, task_id: Tensor) -> Tensor:
        if task_id.ndim != 1:
            raise ValueError("task_id must have shape (batch,)")
        task = self.task_embedding(task_id)
        grid = self.task_to_grid(task).view(
            task.shape[0], self.grid_channels, self.grid_size, self.grid_size
        )
        task_features = self.task_encoder(grid)

        slot_features = torch.cat(
            [
                self.type_embedding(self.type_ids),
                self.layer_embedding(self.layer_ids),
                self.row_embedding(self.row_ids),
                self.col_embedding(self.col_ids),
            ],
            dim=-1,
        )
        task_features = task_features[:, None, :].expand(
            -1, self.metadata.total_patches, -1
        )
        slot_features = slot_features[None, :, :].expand(task_id.shape[0], -1, -1)
        patches = self.head(torch.cat([task_features, slot_features], dim=-1))
        p = self.metadata.patch_size
        return patches.view(task_id.shape[0], self.metadata.total_patches, p, p)

    def masked_loss(self, prediction: Tensor, target: Tensor, mask: Tensor) -> Tensor:
        weights = torch.where(
            mask,
            torch.ones((), device=prediction.device, dtype=prediction.dtype),
            torch.full(
                (),
                self.out_of_mask_weight,
                device=prediction.device,
                dtype=prediction.dtype,
            ),
        )
        return (weights * F.mse_loss(prediction, target, reduction="none")).mean()

    @torch.no_grad()
    def generate_delta(
        self,
        task_id: int,
        tokenizer: "DeltaTokenizer",
        mask: Tensor,
        scales: Tensor,
    ) -> dict[str, Tensor]:
        device = next(self.parameters()).device
        prediction = self(torch.tensor([task_id], device=device))[0]
        dense = tokenizer.destandardize(prediction, scales.to(device))
        sparse = dense * mask.to(device)
        return dict(tokenizer.decode(sparse))
