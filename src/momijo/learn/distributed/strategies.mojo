# MIT License
# SPDX-License-Identifier: MIT
# Project:      Momijo
# Module:       learn.distributed.strategies
# File:         src/momijo/learn/distributed/strategies.mojo
#
# Description:
#   High-level distributed training strategies for Momijo Learn.
#   - DataParallel: single-process, multi-device abstraction (local scatter/gather).
#   - DistributedDataParallel: multi-process training with gradient all-reduce.
#   Backend-agnostic: synchronization primitives are provided by
#   momijo.learn.distributed.collective (allreduce, broadcast, barrier).
#
# Author(s):    Morteza Taleblou & Mitra Daneshmand
# Website:      https://taleblou.ir/
# Repository:   https://github.com/taleblou/momijo
#
# Notes:
#   - Types: DataParallel, DistributedDataParallel
#   - DP expects the caller/backend to handle real device placement & execution.
#   - DDP expects runtime to wire collective ops to tensor backends.
#   - Optimizer duck-typed: must provide step() and zero_grad().
#   - Model duck-typed: must provide state_dict() -> String and load_state_dict(String).
#   - Uses Tensor.slice(...) and cat(List[Tensor[T]], dim) from momijo.tensor.

from collections.list import List
from momijo.learn.distributed.collective import allreduce
from momijo.learn.distributed.collective import broadcast
from momijo.learn.distributed.collective import barrier

# Central tensor facade (minimal, stable import path per project standards)
from momijo.tensor.tensor import Tensor
from momijo.tensor.tensor import cat

# -----------------------------------------------------------------------------
# Local Utilities (pure-Int helpers)
# -----------------------------------------------------------------------------

@always_inline
fn _min_i(a: Int, b: Int) -> Int:
    return a if a < b else b

@always_inline
fn _imax(a: Int, b: Int) -> Int:
    return a if a > b else b

# -----------------------------------------------------------------------------
# DataParallel (single-process, multi-device)
# -----------------------------------------------------------------------------

struct DataParallel:
    var _model
    var _devices: List[Int]
    var _chunks: Int

    fn __init__(out self, model, devices: List[Int] = List[Int](), chunks: Int = 1):
        self._model = model
        self._devices = devices
        self._chunks = _imax(1, chunks)

    fn world_size(self) -> Int:
        var n = len(self._devices)
        if n <= 0:
            return 1
        return n

    fn devices(self) -> List[Int]:
        return self._devices

    fn chunks(self) -> Int:
        return self._chunks

    fn set_chunks(mut self, chunks: Int):
        self._chunks = _imax(1, chunks)

    fn model(self):
        return self._model

    # ------------------------------
    # Backend-agnostic (generic)
    # ------------------------------

    fn scatter(self, batch):
        # Caller/backend can override to implement arbitrary routing.
        return batch

    fn gather(self, outputs):
        # Caller/backend can override to implement arbitrary merging.
        return outputs

    # ------------------------------
    # Tensor-typed conveniences
    # ------------------------------
    # Split along batch dimension=0 into up to min(chunks, world_size) parts.
    # Uses Tensor.slice(ax=0, start, stop, step=1). Final shard absorbs remainder.

    fn scatter_tensor(self, batch: Tensor) -> List[Tensor]:
        var out = List[Tensor]()
        var ndev = self.world_size()
        var parts = _min_i(self._chunks, ndev)
        if parts <= 1:
            out.append(batch)
            return out

        var shp = batch.shape()
        # Expect batch at axis=0; if empty, return single shard.
        if len(shp) == 0:
            out.append(batch)
            return out

        var n = shp[0]
        if n <= 0:
            out.append(batch)
            return out

        # Even split with tail on last shard.
        var base = n // parts
        var rem  = n - base * parts

        var start = 0
        var i = 0
        while i < parts:
            var take = base
            if i == parts - 1:
                take = base + rem
            var stop = start + take
            # Safe-guard against degenerate intervals
            if stop <= start:
                # produce empty-slice semantics: let slice handle it
                out.append(batch.slice(0, start, start, 1))
            else:
                out.append(batch.slice(0, start, stop, 1))
            start = stop
            i += 1
        return out

    # Merge a list of per-device tensors back to a single tensor along batch axis=0.
    fn gather_tensor(self, outputs: List[Tensor]) -> Tensor:
        var k = len(outputs)
        assert(k > 0 and "gather_tensor: empty outputs")
        if k == 1:
            return outputs[0]
        return cat(outputs, 0)

    fn __str__(self) -> String:
        var n = self.world_size()
        var s = String("DataParallel(world_size=") + String(n)
        s = s + String(", chunks=") + String(self._chunks) + String(")")
        return s

# -----------------------------------------------------------------------------
# DistributedDataParallel (multi-process)
# -----------------------------------------------------------------------------

struct DistributedDataParallel:
    var _model
    var _world_size: Int
    var _rank: Int

    fn __init__(out self, model, world_size: Int, rank: Int):
        self._model = model
        self._world_size = _imax(1, world_size)
        self._rank = rank

    fn world_size(self) -> Int:
        return self._world_size

    fn rank(self) -> Int:
        return self._rank

    fn model(self):
        return self._model

    # ------------------------------
    # Parameter sync
    # ------------------------------

    # Broadcast parameters from src (usually rank 0) to all ranks.
    # Expects: model.state_dict() -> String and model.load_state_dict(String).
    fn broadcast_parameters(self, src: Int = 0):
        if self._rank == src:
            var state = self._model.state_dict()
            state = broadcast(state, src)   # rank 0 returns its own state
        else:
            var state = String("")          # receive buffer
            state = broadcast(state, src)
            self._model.load_state_dict(state)
        barrier()  # if all ranks aligned

    # ------------------------------
    # Gradient sync (generic & Tensor)
    # ------------------------------

    # Generic all-reduce for an opaque "grad_blob".
    fn allreduce_gradients(self, grad_blob):
        var reduced = allreduce(grad_blob)
        return reduced

    # Tensor overload: reduce grads across ranks. No averaging here.
    fn allreduce_tensor(self, t: Tensor) -> Tensor:
        return allreduce(t)

    # Reduce and optionally average by world size (when True).
    # Uses scalar division; per your tensor operators, this yields Float tensors.
    fn allreduce_tensor_keep(self, t: Tensor, average: Bool = False) -> Tensor:
        var r = allreduce(t)
        if average and self._world_size > 1:
            # dtype-aware division is handled by Tensor overloads to Float64
            r = r / Float64(self._world_size)
        return r

    # ------------------------------
    # Step sequence
    # ------------------------------
    #  - Reduce gradients (caller may choose average=True).
    #  - Skip optimizer step if AMP overflow detected (found_inf == True).
    #  - Always zero gradients after the step.
    fn step_and_sync(
        mut self,
        optimizer,
        grad_blob,
        found_inf: Bool = False,
        average: Bool = True
    ):
        var reduced = self.allreduce_gradients(grad_blob) 
        #   var reduced_t = self.allreduce_tensor_keep(grad_tensor, average=True)
        if not found_inf:
            optimizer.step()
        optimizer.zero_grad()

    fn __str__(self) -> String:
        var s = String("DDP(world_size=") + String(self._world_size)
        s = s + String(", rank=") + String(self._rank) + String(")")
        return s
