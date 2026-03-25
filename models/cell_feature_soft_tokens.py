from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class CellFeatureSoftTokenizer(nn.Module):
    """
    Project a dense cell-level feature vector into a small number of soft tokens.

    The output stays continuous in embedding space. It is meant to be concatenated
    with token embeddings and fed through `inputs_embeds`, rather than converted
    into discrete token ids.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_soft_tokens: int = 4,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
        use_input_layernorm: bool = True,
        use_output_layernorm: bool = True,
    ) -> None:
        super().__init__()
        if num_soft_tokens <= 0:
            raise ValueError(f"num_soft_tokens must be positive, got {num_soft_tokens}")

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_soft_tokens = num_soft_tokens
        self.hidden_dim = hidden_dim or max(input_dim, output_dim)

        self.input_norm = nn.LayerNorm(input_dim) if use_input_layernorm else nn.Identity()
        self.projector = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, num_soft_tokens * output_dim),
        )
        self.output_norm = nn.LayerNorm(output_dim) if use_output_layernorm else nn.Identity()

    def forward(self, cell_features: torch.Tensor) -> torch.Tensor:
        if cell_features.ndim != 2:
            raise ValueError(
                f"cell_features must have shape (batch, input_dim), got {tuple(cell_features.shape)}"
            )
        if cell_features.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected cell_features last dim {self.input_dim}, got {cell_features.shape[-1]}"
            )

        hidden = self.input_norm(cell_features)
        soft_tokens = self.projector(hidden)
        soft_tokens = soft_tokens.view(cell_features.shape[0], self.num_soft_tokens, self.output_dim)
        return self.output_norm(soft_tokens)
