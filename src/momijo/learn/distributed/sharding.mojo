# MIT License
# SPDX-License-Identifier: MIT
# Project:      Momijo
# Module:       learn.distributed.sharding
# File:         src/momijo/learn/distributed/sharding.mojo
#
# Description:
#   Parameter/state sharding utilities (ZeRO-like) for distributed training.
#   - ShardRange: half-open [start, end) span descriptor.
#   - partition_1d: balanced 1-D partitioner across ranks.
#   - ZeroStage1: scaffold for optimizer-state sharding with simple list-based
#                 reference helpers (shard/allgather/reduce_scatter emulation),
#                 plus minimal state_dict I/O.
#
# Author(s):    Morteza Taleblou & Mitra Daneshmand
# Website:      https://taleblou.ir/
# Repository:   https://github.com/taleblou/momijo

# -----------------------------------------------------------------------------#
# Imports
# -----------------------------------------------------------------------------#
from collections.list import List

# Central tensor import per project standards:
# (Adapters below are no-ops for now; wire them when tensor APIs are finalized)
from momijo.tensor.tensor import Tensor

# Communication primitives are declared here for future wiring.
# Provide real implementations in momijo.learn.distributed.collective,
# or stub them during single-process testing.
from momijo.learn.distributed.collective import (
    allreduce,   # fn allreduce(buf) -> buf
    broadcast,   # fn broadcast(buf, root: Int) -> buf
    barrier      # fn barrier() -> None
)

# -----------------------------------------------------------------------------#
# Shard range descriptor
# -----------------------------------------------------------------------------#

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

    fn clamp_index(self, idx: Int) -> Int:
        var i = idx
        if i < self.start: i = self.start
        if i > self.end: i = self.end
        return i

    fn __str__(self) -> String:
        return String("ShardRange(") + String(self.start) + String(", ") + String(self.end) + String(")")


# -----------------------------------------------------------------------------#
# 1-D partitioner (balanced, prefix-biased by remainder)
# -----------------------------------------------------------------------------#

fn partition_1d(total: Int, world_size: Int, rank: Int) -> ShardRange:
    # Preconditions
    var ws = world_size if world_size > 0 else 1
    var rk = rank
    if rk < 0: rk = 0
    if rk >= ws: rk = ws - 1
    var n = total if total > 0 else 0

    var base = n // ws
    var rem = n - base * ws
    var extra = 0
    if rk < rem:
        extra = 1
    # offset of this rank = rk*base + min(rk, rem)
    var start = rk * base + (rk if rk < rem else rem)
    var end = start + base + extra
    return ShardRange(start, end)


# -----------------------------------------------------------------------------#
# Zero Stage 1 (optimizer-state sharding scaffold)
# -----------------------------------------------------------------------------#

struct ZeroStage1:
    var world_size: Int
    var rank: Int

    fn __init__(out self, world_size: Int = 1, rank: Int = 0):
        self.world_size = (world_size if world_size > 0 else 1)
        self.rank = (rank if rank >= 0 else 0)

    # Compute shard range for a given logical length on this rank.
    fn shard_index(self, length: Int) -> ShardRange:
        return partition_1d(length, self.world_size, self.rank)

    # -------------------- List helpers (reference implementation) --------------------
    # These helpers operate on List[Float32]/List[Int] as stand-ins for tensor buffers.
    # Replace with tensor views/slices when wiring to momijo.tensor.

    fn shard_list(self, xs: List[Float32]) -> List[Float32]:
        var n = len(xs)
        var sr = self.shard_index(n)
        var out = List[Float32]()
        var i = sr.start
        while i < sr.end:
            out.append(xs[i])
            i = i + 1
        return out

    fn shard_list_i32(self, xs: List[Int]) -> List[Int]:
        var n = len(xs)
        var sr = self.shard_index(n)
        var out = List[Int]()
        var i = sr.start
        while i < sr.end:
            out.append(xs[i])
            i = i + 1
        return out

    # Emulated all-gather: returns a full-length buffer with this rank's shard placed.
    # Real impl would gather shards from all ranks; here we only place local shard.
    fn allgather_list(self, shard: List[Float32], total_len: Int) -> List[Float32]:
        var full = _zeros_f64(total_len)
        var sr = self.shard_index(total_len)
        var j = 0
        var pos = sr.start
        while pos < sr.end and j < len(shard):
            full[pos] = shard[j]
            pos = pos + 1
            j = j + 1
        barrier()
        return full

    fn allgather_list_i32(self, shard: List[Int], total_len: Int) -> List[Int]:
        var full = _zeros_i32(total_len)
        var sr = self.shard_index(total_len)
        var j = 0
        var pos = sr.start
        while pos < sr.end and j < len(shard):
            full[pos] = shard[j]
            pos = pos + 1
            j = j + 1
        barrier()
        return full

    # Reduce-scatter-like helper for lists (single-process emulation).
    # Real impl: sum across ranks then keep local shard.
    fn reduce_scatter_list(self, grads_full: List[Float32]) -> List[Float32]:
        # Placeholder: no cross-rank sum here; just slice the shard.
        return self.shard_list(grads_full)

    # -------------------- Optimizer integration (duck-typed) --------------------

    # Perform an optimizer step on locally-owned shards.
    # With a tensor-backed optimizer, this would unscale, reduce-scatter, then update.
    fn step_sharded(mut self, optimizer):
        optimizer.step()

    fn zero_grad_sharded(mut self, optimizer):
        optimizer.zero_grad()

    # -------------------- State dict I/O --------------------

    fn state_dict(self) -> String:
        var s = String("{")
        s = s + String("\"world_size\":") + String(self.world_size)
        s = s + String(",\"rank\":") + String(self.rank) + String("}")
        return s

    fn load_state_dict(mut self, state: String):
        var ws = _parse_json_int(state, String("world_size"), self.world_size)
        var rk = _parse_json_int(state, String("rank"), self.rank)
        if ws <= 0: ws = 1
        if rk < 0: rk = 0
        self.world_size = ws
        self.rank = rk

    fn __str__(self) -> String:
        var s = String("ZeroStage1(world_size=")
        s = s + String(self.world_size) + String(", rank=") + String(self.rank) + String(")")
        return s


# -----------------------------------------------------------------------------#
# Tensor adapters (placeholders for real tensor wiring)
# -----------------------------------------------------------------------------#
# NOTE:
# - We keep these functions minimal and non-intrusive. They forward to the
#   list-based reference helpers above. When your momijo.tensor 1-D APIs
#   are finalized (creation/indexing/slicing), replace the bodies with true
#   tensor-slice logic without intermediate lists.

@always_inline
fn _tensor_to_list_f64(t: Tensor) -> List[Float32]:
    # TODO: Replace with real tensor -> list extraction when API is ready.
    # For now, return empty to avoid accidental reliance in single-process tests.
    var out = List[Float32]()
    return out

@always_inline
fn _list_to_tensor_f64(xs: List[Float32]) -> Tensor:
    # TODO: Replace with real list -> tensor creation (1-D) when API is ready.
    var t = Tensor()  # placeholder zero-alloc; wire to Tensor[Float32] factory later
    return t

@always_inline
fn _tensor_to_list_i32(t: Tensor) -> List[Int]:
    var out = List[Int]()
    return out

@always_inline
fn _list_to_tensor_i32(xs: List[Int]) -> Tensor:
    var t = Tensor()
    return t

# Shard a 1-D Float32 tensor (adapter).
fn shard_tensor_f64(self: ZeroStage1, t: Tensor) -> Tensor:
    var xs = _tensor_to_list_f64(t)
    var shard_xs = self.shard_list(xs)
    return _list_to_tensor_f64(shard_xs)

# All-gather (emulated) for a 1-D Float32 tensor (adapter).
fn allgather_tensor_f64(self: ZeroStage1, shard: Tensor, total_len: Int) -> Tensor:
    var xs = _tensor_to_list_f64(shard)
    var full = self.allgather_list(xs, total_len)
    return _list_to_tensor_f64(full)

# Reduce-scatter (emulated) for a 1-D Float32 tensor (adapter).
fn reduce_scatter_tensor_f64(self: ZeroStage1, grads_full: Tensor) -> Tensor:
    var xs = _tensor_to_list_f64(grads_full)
    var shard_xs = self.reduce_scatter_list(xs)
    return _list_to_tensor_f64(shard_xs)


# -----------------------------------------------------------------------------#
# Tiny helpers
# -----------------------------------------------------------------------------#

@always_inline
fn _zeros_f64(n: Int) -> List[Float32]:
    var out = List[Float32]()
    var i = 0
    while i < n:
        out.append(0.0)
        i = i + 1
    return out

@always_inline
fn _zeros_i32(n: Int) -> List[Int]:
    var out = List[Int]()
    var i = 0
    while i < n:
        out.append(0)
        i = i + 1
    return out

# Tiny JSON int extractor (very small/tolerant; replace with real JSON later)
fn _parse_json_int(s: String, key: String, default_val: Int) -> Int:
    var pattern = String("\"") + key + String("\":")
    var idx = s.find(pattern)
    if idx == -1:
        return default_val
    var start = idx + Int(pattern.__len__())
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
