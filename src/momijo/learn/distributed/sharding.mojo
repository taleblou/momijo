# Project:      Momijo
# Module:       learn.distributed.sharding
# File:         distributed/sharding.mojo
# Path:         src/momijo/learn/distributed/sharding.mojo
#
# Description:  Parameter/state sharding utilities (Zero-like) for distributed training.
#               Provides stage-1 optimizer-state sharding scaffolding with a simple
#               1-D partitioner and helper routines to shard/gather arrays. This
#               module is backend-agnostic and can be wired to tensor ops later.
#
# Author(s):    Morteza Taleblou & Mitra Daneshmand
# Website:      https://taleblou.ir/
# Repository:   https://github.com/taleblou/momijo
#
# License:      MIT License
# SPDX-License-Identifier: MIT
# Copyright:    (c) 2025 Morteza Taleblou & Mitra Daneshmand
#
# Notes:
#   - Types: ShardRange, ZeroStage1
#   - Key fns: partition_1d(total, world_size, rank), shard_list(xs), allgather_list(...)
#   - ZeroStage1: attach(...), shard_index(len), step_sharded(...), state_dict()

from collections.list import List
from momijo.learn.distributed.collective import (
    allreduce,
    broadcast,
    barrier,
)

# -----------------------------------------------------------------------------
# Shard range descriptor
# -----------------------------------------------------------------------------

struct ShardRange:
    var start: Int
    var end: Int  # exclusive

    fn __init__(out self, start: Int, end: Int):
        self.start = start
        self.end = end

    fn length(self) -> Int:
        return self.end - self.start

    fn contains(self, idx: Int) -> Bool:
        return (idx >= self.start) and (idx < self.end)


# -----------------------------------------------------------------------------
# 1-D partitioner (balanced, prefix-biased by remainder)
# total items are split across ranks as evenly as possible.
# -----------------------------------------------------------------------------

fn partition_1d(total: Int, world_size: Int, rank: Int) -> ShardRange:
    # base size per rank
    var base = total // world_size
    var rem = total - base * world_size
    # first `rem` ranks get one extra element
    var extra = 0
    if rank < rem:
        extra = 1
    var start = rank * base + (rank if rank < rem else rem)
    var end = start + base + extra
    return ShardRange(start, end)


# -----------------------------------------------------------------------------
# Zero Stage 1 (optimizer-state sharding scaffold)
# - Partitions parameter/state vectors across ranks.
# - Exposes helpers to shard and (conceptually) all-gather arrays.
# - Optimizer integration is kept duck-typed: expects step()/zero_grad().
# -----------------------------------------------------------------------------

struct ZeroStage1:
    var world_size: Int
    var rank: Int

    # Optional defaults for communication; not used directly in list helpers,
    # but reserved for tensor-wired versions.
    fn __init__(out self, world_size: Int = 1, rank: Int = 0):
        self.world_size = world_size
        self.rank = rank

    # Compute shard range for a given logical length on this rank.
    fn shard_index(self, length: Int) -> ShardRange:
        return partition_1d(length, self.world_size, self.rank)

    # -------------------- List helpers (reference implementation) --------------------
    # These helpers operate on List[Float64] as a stand-in for real tensor buffers.
    # Replace with tensor views/slices when wiring to momijo.tensor.

    fn shard_list(self, xs: List[Float64]) -> List[Float64]:
        var n = Int(xs.size())
        var sr = self.shard_index(n)
        var out = List[Float64]()
        var i = sr.start
        while i < sr.end:
            out.push_back(xs[i])
            i = i + 1
        return out

    # Conceptual all-gather: builds a full-size list and places this rank's shard in place.
    # In a real implementation, each rank would contribute its shard and receive the full tensor.
    fn allgather_list(self, shard: List[Float64], total_len: Int) -> List[Float64]:
        var full = List[Float64]()
        # init with zeros
        var i = 0
        while i < total_len:
            full.push_back(0.0)
            i = i + 1

        var sr = self.shard_index(total_len)
        var j = 0
        var pos = sr.start
        # copy local shard into its slot
        while pos < sr.end and j < Int(shard.size()):
            full[pos] = shard[j]
            pos = pos + 1
            j = j + 1

        # NOTE: In a true distributed setup, you'd gather shards from all ranks.
        # Here we emulate only the local placement. Use broadcast/allreduce as needed.
        barrier()
        return full

    # Reduce-scatter-like helper for lists (local emulation):
    # In a real impl: grads = sum(grads across ranks) then each rank keeps its shard.
    fn reduce_scatter_list(self, grads_full: List[Float64]) -> List[Float64]:
        # Placeholder: no cross-rank sum here; just slice the shard.
        return self.shard_list(grads_full)

    # -------------------- Optimizer integration --------------------

    # Perform an optimizer step on locally-owned shards.
    # In a full tensor-backed impl, this would:
    #   1) unscale grads (if AMP), 2) reduce-scatter grads, 3) local update on shard,
    #   4) optionally all-gather updated params for model usage.
    fn step_sharded(mut self, optimizer):
        # With a real optimizer that knows sharding, only local shards are updated.
        optimizer.step()

    fn zero_grad_sharded(mut self, optimizer):
        optimizer.zero_grad()

    # -------------------- State dict IO --------------------

    # Save minimal sharding metadata; payload wiring belongs to utils/checkpoint.
    fn state_dict(self) -> String:
        var s = String("{")
        s = s + String("\"world_size\":") + String(self.world_size)
        s = s + String(",\"rank\":") + String(self.rank) + String("}")
        return s

    fn load_state_dict(mut self, state: String):
        # Minimal tolerant parser for {"world_size":X,"rank":Y}
        var ws = _parse_json_int(state, String("world_size"), self.world_size)
        var rk = _parse_json_int(state, String("rank"), self.rank)
        self.world_size = ws
        self.rank = rk

# -----------------------------------------------------------------------------
# Tiny JSON int extractor (very small/tolerant; replace with real JSON later)
# -----------------------------------------------------------------------------

fn _parse_json_int(s: String, key: String, default_val: Int) -> Int:
    # naive search: '"key":<int>'
    var pattern = String("\"") + key + String("\":")
    var idx = s.find(pattern)
    if idx == -1:
        return default_val
    var start = idx + Int(pattern.__len__())
    # read consecutive digits or sign+digits
    var i = start
    var sign = 1
    if i < Int(s.__len__()):
        var ch = s[i]
        if ch == 45:  # '-'
            sign = -1
            i = i + 1
    var acc: Int = 0
    while i < Int(s.__len__()):
        var c = s[i]
        if c < 48 or c > 57:
            break
        acc = acc * 10 + Int(c - 48)
        i = i + 1
    return sign * acc
