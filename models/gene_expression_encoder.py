from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class GeneExpressionValueEncoder(nn.Module):
    """
    Project per-gene continuous expression values into the model embedding space.

    This mirrors the value-encoder idea used in scGPT: each scalar expression value
    is mapped to one embedding vector and added to the corresponding gene token embedding.
    """

    def __init__(
        self,
        output_dim: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
        max_value: float = 20.0,
        use_output_layernorm: bool = True,
    ) -> None:
        super().__init__()
        hidden_dim = hidden_dim or output_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.max_value = max_value
        self.projector = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
        self.output_norm = nn.LayerNorm(output_dim) if use_output_layernorm else nn.Identity()

    def forward(self, gene_expression: torch.Tensor) -> torch.Tensor:
        if gene_expression.ndim != 2:
            raise ValueError(
                f"gene_expression must have shape (batch, seq_len), got {tuple(gene_expression.shape)}"
            )
        x = gene_expression.unsqueeze(-1)
        x = torch.clamp(x, min=0.0, max=self.max_value)
        x = self.projector(x)
        return self.output_norm(x)
