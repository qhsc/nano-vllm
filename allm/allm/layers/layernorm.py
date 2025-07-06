import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        dtype: torch.dtype | None = None,
        act_dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.dtype = dtype or torch.bfloat16
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=self.dtype))
        self.act_dtype = act_dtype

    @torch.compile
    def rms_forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype

        x = x.to(self.act_dtype)
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))

        x = x.to(orig_dtype).mul_(self.weight)
        return x

    @torch.compile
    def add_rms_forward(
        self, x: torch.Tensor, residual: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        orig_dtype = x.dtype

        x = x.to(self.act_dtype).add_(residual.to(self.act_dtype))
        residual = x.to(orig_dtype)
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))

        x = x.to(orig_dtype).mul_(self.weight)
        return x, residual

    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return self.rms_forward(x)
        else:
            return self.add_rms_forward(x, residual)
