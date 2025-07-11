import torch
import hashlib
from torch import nn

from transformers.models.qwen3 import Qwen3Config

from allm.layers.linear import (
    QKVParallelLinear,
    RowParallelLinear,
    MergedColumnParallelLinear,
)
from allm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead
from allm.layers.layernorm import RMSNorm
from allm.layers.rope import get_rope
from allm.layers.attention import Attention
from allm.utils.dist import get_tp_size
from allm.layers.act import SiluAndMul


class Qwen3Attention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        max_position: int = 1024 * 32,
        rms_norm_eps: float = 1e-6,
        dtype: torch.dtype | None = None,
        qkv_bias: bool = False,
        rope_theta: float = 10000.0,
        rope_scaling: tuple | None = None,
    ):
        super().__init__()
        self.total_num_heads = num_heads
        self.total_num_kv_heads = num_kv_heads
        self.num_heads = self.total_num_heads // get_tp_size()
        self.num_kv_heads = self.total_num_kv_heads // get_tp_size()
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.attention_scale = self.head_dim**-0.5

        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

        self.qkv_proj = QKVParallelLinear(
            hidden_size=hidden_size,
            head_size=head_dim,
            total_num_heads=self.total_num_heads,
            total_num_kv_heads=self.total_num_kv_heads,
            bias=qkv_bias,
            dtype=dtype,
        )
        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps, dtype=dtype)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps, dtype=dtype)
        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )

        self.attn = Attention(
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            num_kv_heads=self.num_kv_heads,
            scale=self.attention_scale,
        )

        self.o_proj = RowParallelLinear(
            input_size=self.total_num_heads * self.head_dim,
            output_size=hidden_size,
            dtype=dtype,
            tp_dim=-1,
            bias=False,
            reduce_output=True,
        )

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor):
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q_by_head = q.view(-1, self.num_heads, self.head_dim)
        q_by_head = self.q_norm(q_by_head)
        q = q_by_head.view(q.shape)
        k_by_head = k.view(-1, self.num_kv_heads, self.head_dim)
        k_by_head = self.k_norm(k_by_head)
        k = k_by_head.view(k.shape)
        q, k = self.rotary_emb(positions, q, k)
        o = self.attn(q, k, v)
        output = self.o_proj(o)
        return output


class Qwen3MLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        dtype: torch.dtype | None = None,
        hiddent_act: str = "silu",
    ):
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[intermediate_size] * 2,
            bias=False,
            dtype=dtype,
        )
        self.down_proj = RowParallelLinear(
            input_size=intermediate_size,
            output_size=hidden_size,
            bias=False,
            dtype=dtype,
            reduce_output=True,
        )
        assert hiddent_act == "silu"
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x = self.down_proj(x)
        return x


class Qwen3DecoderLayer(nn.Module):

    def __init__(
        self,
        config,
    ) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, dtype=config.torch_dtype
        )
        self.self_attn = Qwen3Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, "attention_bias", False),
            head_dim=getattr(config, "head_dim"),
            rope_theta=getattr(config, "rope_theta", 1000000),
            rope_scaling=getattr(config, "rope_scaling", None),
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, dtype=config.torch_dtype
        )
        self.mlp = Qwen3MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            dtype=config.torch_dtype,
            hiddent_act=config.hidden_act,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)

        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual  # type: ignore


class Qwen3Model(nn.Module):

    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(
            vocab_size=config.vocab_size,
            embedding_dim=config.hidden_size,
            dtype=config.torch_dtype,
        )
        self.layers = nn.ModuleList(
            [Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, dtype=config.torch_dtype
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        i = 0
        for layer in self.layers:
            i += 1
            hidden_states, residual = layer(positions, hidden_states, residual)

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Qwen3ForCausalLLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        config: Qwen3Config,
    ):
        super().__init__()
        self.model = Qwen3Model(config)
        self.lm_head = ParallelLMHead(
            vocab_size=config.vocab_size,
            embedding_dim=config.hidden_size,
            dtype=config.torch_dtype,
        )
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions)
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        logits = self.lm_head(hidden_states)
        return logits
