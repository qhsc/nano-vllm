import pickle
import numpy as np
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from allm.config import Config
from allm.models.qwen3 import Qwen3ForCausalLLM
from allm.layers.sampler import Sampler
from allm.utils.context import set_context, get_context, reset_context
from allm.utils.loader import load_model
from allm.engine.sequence import Sequence
from allm.utils.logger import logger


class Shm:
    def __init__(
        self, name: str, size: int, is_creator: bool, event: Event | list[Event]
    ):
        self.name = name
        self.size = size
        self.is_creator = is_creator
        self.event = event
        if is_creator:
            self.shm = SharedMemory(name=name, create=True, size=size)
            dist.barrier()
        else:
            dist.barrier()
            self.shm = SharedMemory(name=name, create=False)

    def write(self, method_name: str, *args: list):
        data = pickle.dumps([method_name, *args])
        assert self.is_creator and isinstance(self.event, list)
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4 : n + 4] = data
        for event in self.event:
            event.set()

    def read(self) -> tuple[str, list]:
        assert not self.is_creator and isinstance(self.event, Event)
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4 : n + 4])
        self.event.clear()
        return method_name, args

    def close(self):
        self.shm.close()
        dist.barrier()
        if self.is_creator:
            self.shm.unlink()


class ModelRunner:
    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        self.rank = rank
        self.event = event

        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size

        dist.init_process_group(
            "nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank
        )

        if rank == 0:
            logger.info(f"Initializing torch distributed done")
        torch.cuda.set_device(rank)
        torch.set_default_device("cuda")

        self.model = Qwen3ForCausalLLM(config.hf_config)  # type: ignore
        if self.rank == 0:
            logger.info(f"Loading model architecture: {self.model}")
        load_model(self.model, config.model)
        self.sampler = Sampler()

        if rank == 0:
            logger.info(f"Warming up model...")
        self.warmup_model()

        if rank == 0:
            logger.info(f"Allocating kv cache...")
        self.allocate_kv_cache()
        if rank == 0:
            logger.info(
                f"Allocated kv cache with: num_tokens={config.num_kvcache_blocks*self.block_size}"
            )

        torch.set_default_device("cpu")

        if self.world_size > 1:
            self.shm = Shm(
                name="allm_shm",
                size=512 * 1024 * 1024,
                is_creator=self.rank == 0,
                event=event,
            )
            if not self.shm.is_creator:
                self.loop()
        else:
            self.shm = None

    def loop(self):
        while self.shm is not None:
            method_name, args = self.shm.read()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def call(self, method_name, *args):
        method = getattr(self, method_name, None)
        if method is None:
            raise ValueError(f"Method {method_name} not found")
        if self.shm is not None and self.shm.is_creator:
            self.shm.write(method_name, *args)
        return method(*args)

    def exit(self):
        if self.shm is not None:
            self.shm.close()
            self.shm = None
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_seqs = self.config.max_num_batched_tokens // self.config.max_model_len
        max_seqs = min(max_seqs, self.config.max_num_seqs)
        seqs = [Sequence([0] * self.config.max_model_len) for _ in range(max_seqs)]
        self.run(seqs, True)
        torch.cuda.empty_cache()

    def allocate_kv_cache(self, num_kvcache_blocks: int | None = None):
        config = self.config
        hf_config = config.hf_config

        num_kv_heads = hf_config.num_key_value_heads // self.world_size  # type: ignore
        kv_cache_shape = [
            2,
            hf_config.num_hidden_layers,  # type: ignore
            1,  # num_blocks placeholder
            self.block_size,
            num_kv_heads,
            hf_config.head_dim,  # type: ignore
        ]

        if num_kvcache_blocks is None:
            free, total = torch.cuda.mem_get_info()
            used = total - free  # all system level usded memory

            peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
            current = torch.cuda.memory_stats()[
                "allocated_bytes.all.current"
            ]  # pytorch current allocated memory
            max_torch_dynamic_memory = peak - current
            kv_cache_available = (
                total * config.gpu_memory_utilization - used - max_torch_dynamic_memory
            )

            dtype = hf_config.torch_dtype  # type: ignore
            kv_cache_shape[2] = int(
                kv_cache_available // (np.prod(kv_cache_shape) * dtype.itemsize)
            )
            config.num_kvcache_blocks = kv_cache_shape[2]
            assert kv_cache_shape[2] > 0
        else:
            kv_cache_shape[2] = num_kvcache_blocks
            config.num_kvcache_blocks = num_kvcache_blocks

        self.kv_cache = torch.zeros(kv_cache_shape, dtype=dtype, device="cuda")
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence]) -> torch.Tensor:
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [
            seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs
        ]
        block_tables = torch.tensor(
            block_tables, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        for seq in seqs:
            seqlen = len(seq)
            input_ids.extend(seq[seq.num_cached_tokens :])
            positions.extend(list(range(seq.num_cached_tokens, seqlen)))
            seqlen_q = seqlen - seq.num_cached_tokens
            seqlen_k = seqlen
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:
                continue
            for i in range(seq.num_cached_blocks, seq.num_blocks):
                start = seq.block_table[i] * self.block_size
                if i != seq.num_blocks - 1:
                    end = start + self.block_size
                else:
                    end = start + seq.last_block_num_tokens
                slot_mapping.extend(list(range(start, end)))
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:  # prefix cache
            block_tables = self.prepare_block_tables(seqs)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(
            non_blocking=True
        )
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(
            non_blocking=True
        )
        cu_seqlens_q = torch.tensor(
            cu_seqlens_q, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(
            cu_seqlens_k, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        slot_mapping = torch.tensor(
            slot_mapping, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        set_context(
            True,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            slot_mapping,
            None,
            block_tables,
        )
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq))
            context_lens.append(len(seq))
            slot_mapping.append(
                seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1
            )
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(
            non_blocking=True
        )
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(
            non_blocking=True
        )
        slot_mapping = torch.tensor(
            slot_mapping, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        context_lens = torch.tensor(
            context_lens, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        set_context(
            False,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
        )
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = []
        for seq in seqs:
            temperatures.append(seq.temperature)
        temperatures = torch.tensor(
            temperatures, dtype=torch.float32, pin_memory=True
        ).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(
        self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool
    ):
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            bs = input_ids.size(0)
            context = get_context()
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            for k, v in graph_vars.items():
                if k != "outputs":
                    v.zero_()
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"][:bs] = context.slot_mapping  # type: ignore
            graph_vars["context_lens"][:bs] = context.context_lens  # type: ignore
            graph_vars["block_tables"][
                :bs, : context.block_tables.size(1)  # type: ignore
            ] = context.block_tables
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        input_ids, positions = (
            self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        )
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        logits = self.run_model(input_ids, positions, is_prefill)
        token_ids = (
            self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        )
        reset_context()
        return token_ids  # type: ignore

    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)  # type: ignore
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(
                False,
                slot_mapping=slot_mapping[:bs],
                context_lens=context_lens[:bs],
                block_tables=block_tables[:bs],
            )
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])  # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])  # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
