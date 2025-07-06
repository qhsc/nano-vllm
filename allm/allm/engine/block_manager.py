from collections import deque
import xxhash
import numpy as np

from allm.engine.sequence import Sequence


class Block:
    def __init__(self, block_id: int):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:
    def __init__(self, num_blocks: int, block_size: int):
        assert num_blocks > 0

        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]

        # free blocks are blocks that are not in use by sequences.
        # but them may be also keep cached tokens which can be reused.
        self.free_block_ids: deque[int] = deque(range(num_blocks))

        # used blocks are blocks that are currently in use by sequences.
        self.used_block_ids: set[int] = set()

        # maybe_cached_hash_to_block is a dictionary that maps a hash to a block id.
        # it is used to cache blocks that may be reused.
        self.maybe_cached_hash_to_block: dict[int, int] = dict()

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self, block_id: int) -> Block:
        block = self.blocks[block_id]
        assert block.ref_count == 0
        block.reset()
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return self.blocks[block_id]

    def _deallocate_block(self, block_id: int):
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= seq.num_blocks

    def allocate(self, seq: Sequence):
        assert not seq.block_table
        prefix_hash = -1
        cache_miss = False
        for i in range(seq.num_blocks):
            token_ids = seq.block(i)
            prefix_hash = (
                self.compute_hash(token_ids, prefix_hash)
                if len(token_ids) == self.block_size
                else -1
            )

            maybe_cached_block_id = self.maybe_cached_hash_to_block.get(prefix_hash, -1)
            if (
                maybe_cached_block_id == -1
                or self.blocks[maybe_cached_block_id].token_ids != token_ids
            ):
                cache_miss = True

            if cache_miss:
                block_id = self.free_block_ids[0]
                block = self._allocate_block(block_id=block_id)
            else:
                block_id = maybe_cached_block_id
                seq.num_cached_tokens += self.block_size
                if block_id in self.used_block_ids:
                    block = self.blocks[block_id]
                    block.ref_count += 1
                else:
                    block = self._allocate_block(block_id)

            if prefix_hash != -1:
                block.update(prefix_hash, token_ids)
                self.maybe_cached_hash_to_block[prefix_hash] = block_id

            seq.block_table.append(block_id)

    def deallocate(self, seq: Sequence):
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        block_table = seq.block_table
        last_block = self.blocks[block_table[-1]]
        if len(seq) % self.block_size == 1 or self.block_size == 1:
            # just append a new token, which need to be store in a new block.
            assert last_block.hash != -1
            block_id = self.free_block_ids[0]
            self._allocate_block(block_id)
            block_table.append(block_id)
        elif len(seq) % self.block_size == 0:
            # append a new token, which need to be store in the last block.
            assert last_block.hash == -1
            token_ids = seq.block(seq.num_blocks - 1)
            prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
            h = self.compute_hash(token_ids, prefix)
            last_block.update(h, token_ids)
            self.maybe_cached_hash_to_block[h] = last_block.block_id
        else:
            assert last_block.hash == -1
