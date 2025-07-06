import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from allm.utils.dist import divide, get_tp_rank, get_tp_size


class LinearBase(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        dtype: torch.dtype | None = None,
        tp_dim: int | None = None,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.dtype = dtype or torch.bfloat16
        self.tp_dim = tp_dim if tp_dim is not None else 0
        self.tp_rank = get_tp_rank()
        self.tp_size = get_tp_size()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class ReplicatedLinear(LinearBase):
    """ReplicatedLinear is a linear layer that is replicated across all GPUs,
    which means the weight is the same across all GPUs and not sharded.

    Args:
        input_size (int): The number of input features.
        output_size (int): The number of output features.
        bias (bool, optional): Whether to use bias. Defaults to False.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        dtype: torch.dtype,
        bias: bool = False,
    ):
        super().__init__(input_size, output_size, dtype)

        self.weight = nn.Parameter(
            torch.empty(output_size, input_size, dtype=self.dtype)
        )
        setattr(self.weight, "weight_loader", self.weight_loader)

        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
            setattr(self.bias, "weight_loader", self.weight_loader)
        else:
            self.register_parameter("bias", None)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # linear perform x*weight^T + bias
        return F.linear(x, self.weight, self.bias)


class ColumnParallelLinear(LinearBase):
    """ColumnParallelLinear is a linear layer which parameters is sharded in the ouput dimension.

    Args:
        input_size (int): The number of input features.
        output_size (int): The number of output features.
        tp_dim (int): The dimension to shard the parameters.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        dtype: None | torch.dtype = None,
        tp_dim: int | None = -2,
        bias: bool = False,
        gather_output: bool = False,
    ):
        super().__init__(input_size, output_size, dtype, tp_dim)

        self.output_size_per_partition = divide(output_size, self.tp_size)
        self.gather_output = gather_output and self.tp_size > 1

        self.weight = nn.Parameter(
            torch.empty(
                self.output_size_per_partition,
                self.input_size,
                dtype=self.dtype,
            )
        )
        setattr(self.weight, "weight_loader", self.weight_loader)

        if bias:
            self.bias = nn.Parameter(
                torch.empty(self.output_size_per_partition, dtype=self.dtype)
            )
            setattr(self.bias, "weight_loader", self.weight_loader)
        else:
            self.register_parameter("bias", None)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        assert self.tp_dim is not None, "tp_dim must be set"
        shard_size = param.data.size(self.tp_dim)
        shard_weight = loaded_weight.narrow(
            self.tp_dim, self.tp_rank * shard_size, shard_size
        )
        param.data.copy_(shard_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.weight, self.bias)
        if self.gather_output:
            output_list = [torch.empty_like(y) for _ in range(self.tp_size)]
            dist.all_gather(output_list, y)
            y = torch.cat(output_list, dim=-1)
        return y


class MergedColumnParallelLinear(ColumnParallelLinear):
    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        dtype: None | torch.dtype = None,
        tp_dim: int | None = -2,
        bias: bool = False,
        gather_output: bool = False,
    ):
        self.output_sizes = output_sizes
        super().__init__(
            input_size, sum(output_sizes), dtype, tp_dim, bias, gather_output
        )

    def weight_loader(
        self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: int
    ):
        shard_size = divide(self.output_sizes[loaded_shard_id], self.tp_size)
        shard_weight = loaded_weight.narrow(
            self.tp_dim, shard_size * self.tp_rank, shard_size
        )

        param_offset = divide(sum(self.output_sizes[0:loaded_shard_id]), self.tp_size)
        param.data.narrow(self.tp_dim, param_offset, shard_size).copy_(shard_weight)


class QKVParallelLinear(ColumnParallelLinear):
    """
    QKVParallelLinear compute qkv in parallel of heads.
    """

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        dtype: None | torch.dtype = None,
        tp_dim: int | None = -2,
        bias: bool = False,
        gather_output: bool = False,
    ):
        self.head_size = head_size
        self.total_num_heads = total_num_heads
        self.total_num_kv_heads = total_num_kv_heads or total_num_heads
        output_size = (
            self.total_num_heads + 2 * self.total_num_kv_heads
        ) * self.head_size
        super().__init__(hidden_size, output_size, dtype, tp_dim, bias, gather_output)

        self.num_heads = divide(self.total_num_heads, self.tp_size)
        self.num_kv_heads = divide(self.total_num_kv_heads, self.tp_size)

    def weight_loader(
        self, param: nn.Parameter, loaded_weight: torch.Tensor, load_shared_id: str
    ):
        assert load_shared_id in ["q", "k", "v"]
        shard_offset = 0
        shard_size = self.num_heads * self.head_size
        if load_shared_id in ["k", "v"]:
            shard_offset += shard_size
            shard_size = self.num_kv_heads * self.head_size
            if load_shared_id == "v":
                shard_offset += shard_size
        param.data.narrow(self.tp_dim, shard_offset, shard_size).copy_(
            loaded_weight.narrow(self.tp_dim, self.tp_rank * shard_size, shard_size)
        )


class RowParallelLinear(LinearBase):
    """RowParallelLinear is a linear layer which parameters is sharded in the input dimension.

    Args:
        input_size (int): The number of input features.
        output_size (int): The number of output features.
        tp_dim (int): The dimension to shard the parameters.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        dtype: None | torch.dtype = None,
        tp_dim: int | None = -1,
        bias: bool = False,
        reduce_output: bool = False,
    ):
        super().__init__(input_size, output_size, dtype, tp_dim)

        self.input_size_per_partition = divide(input_size, self.tp_size)
        self.reduce_output = reduce_output and self.tp_size > 1

        self.weight = nn.Parameter(
            torch.empty(
                self.output_size, self.input_size_per_partition, dtype=self.dtype
            )
        )
        setattr(self.weight, "weight_loader", self.weight_loader)

        if bias:
            self.bias = nn.Parameter(torch.empty(self.output_size, dtype=self.dtype))
            setattr(self.bias, "weight_loader", self.weight_loader)
        else:
            self.register_parameter("bias", None)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param.data.copy_(
            loaded_weight.narrow(
                self.tp_dim,
                self.tp_rank * self.input_size_per_partition,
                self.input_size_per_partition,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)
        if self.reduce_output:
            dist.all_reduce(y, op=dist.ReduceOp.SUM)
        return y
