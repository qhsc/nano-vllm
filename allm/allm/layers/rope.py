from functools import lru_cache
import torch
from torch import nn


class RotaryEmbedding(nn.Module):
    dtype = torch.float32

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position: int,
        base: float,
    ):
        super().__init__()
        self.max_position = max_position
        assert head_size == rotary_dim
        self.rotary_dim = rotary_dim
        self.base = base

        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, rotary_dim, 2, dtype=self.dtype) / rotary_dim)
        )
        pos = torch.arange(max_position, dtype=self.dtype)
        freqs = pos.unsqueeze_(-1) * inv_freq
        cos = freqs.cos()
        sin = freqs.sin()
        self.register_buffer("cos_sin_cache", torch.cat((cos, sin), dim=-1))

    @torch.compile
    def forward(self, postions: torch.Tensor, query: torch.Tensor, key: torch.Tensor):
        cos, sin = self.cos_sin_cache[postions].chunk(2, dim=-1)  # type: ignore
        query = self.apply_rotary_emb(query, cos, sin)
        key = self.apply_rotary_emb(key, cos, sin)
        return query, key

    def apply_rotary_emb(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        ori_shape = x.shape
        x = x.view(cos.size(0), -1, self.rotary_dim)
        x1, x2 = torch.chunk(x.to(self.dtype), 2, dim=-1)
        cos = cos.unsqueeze(-2)
        sin = sin.unsqueeze(-2)
        y1 = x1 * cos - x2 * sin
        y2 = x2 * cos + x1 * sin
        return torch.cat((y1, y2), dim=-1).to(x.dtype).view(ori_shape)


@lru_cache(1)
def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: dict | None = None,
):
    assert rope_scaling is None
    rotary_emb = RotaryEmbedding(
        head_size=head_size, rotary_dim=rotary_dim, max_position=max_position, base=base
    )
    return rotary_emb
