import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
from allm.utils.dist import get_tp_rank, get_tp_size, get_tp_group
from allm.utils.context import get_context


class VocabParallelEmbedding(nn.Module):
    """
    Embedding layer that is parallelized across the vocabulary dimension.
    """

    def __init__(
        self, vocab_size: int, embedding_dim: int, dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.tp_rank = get_tp_rank()
        self.tp_size = get_tp_size()
        self.vocab_size = vocab_size
        self.vocab_size_padded = (
            (self.vocab_size + self.tp_size - 1) // self.tp_size * self.tp_size
        )
        self.vocab_size_per_partition = self.vocab_size_padded // self.tp_size
        self.embedding_dim = embedding_dim
        self.dtype = dtype or torch.bfloat16

        self.vocab_offset = self.vocab_size_per_partition * self.tp_rank
        self.shard_size = self.vocab_size_per_partition + (
            0
            if self.tp_rank != self.tp_size - 1
            else self.vocab_size - self.vocab_size_padded
        )
        self.weight = nn.Parameter(
            torch.empty(
                self.vocab_size_per_partition, self.embedding_dim, dtype=self.dtype
            )
        )
        setattr(self.weight, "weight_loader", self.weight_loader)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param.data.copy_(loaded_weight.narrow(0, self.vocab_offset, self.shard_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.tp_size > 1:
            mask = (x >= self.vocab_offset) & (x < self.vocab_offset + self.shard_size)
            x = mask * (x - self.vocab_offset)
        y = F.embedding(x, self.weight)
        if self.tp_size > 1:
            y = mask.unsqueeze(1) * y
            dist.all_reduce(y, group=get_tp_group())
        return y


class ParallelLMHead(VocabParallelEmbedding):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        dtype: torch.dtype | None = None,
        bias: bool = False,
    ):
        super().__init__(vocab_size, embedding_dim, dtype)
        if bias:
            self.bias = nn.Parameter(torch.empty(self.vocab_size_per_partition))
            setattr(self.bias, "weight_loader", self.weight_loader)
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor | None:
        context = get_context()
        if context.is_prefill:
            last_indices = context.cu_seqlens_q[1:] - 1  # type: ignore
            x = x[last_indices].contiguous()
        logits = F.linear(x, self.weight, self.bias)
        if self.tp_size > 1:
            all_logits = (
                [torch.empty_like(logits) for _ in range(self.tp_size)]
                if self.tp_rank == 0
                else None
            )
            dist.gather(logits, all_logits, 0)
            logits = torch.cat(all_logits, -1) if self.tp_rank == 0 else None
        return logits
