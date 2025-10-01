# Project:      Momijo
# Module:       learn.distributed.strategies
# File:         distributed/strategies.mojo
# Path:         src/momijo/learn/distributed/strategies.mojo
#
# Description:  High-level distributed training strategies for Momijo Learn.
#               Provides two wrappers:
#                 - DataParallel: single-process, multi-device abstraction (local scatter/gather).
#                 - DistributedDataParallel: multi-process training with gradient all-reduce.
#               This file is backend-agnostic and calls into `learn.distributed.collective`
#               for synchronization primitives (allreduce, broadcast, barrier).
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
#   - Types: DataParallel, DistributedDataParallel
#   - Key fns (no-ops until wired with real backend):
#       * DataParallel.scatter(), gather()
#       * DistributedDataParallel.broadcast_parameters(), allreduce_gradients()
#   - Assumes the wrapped `model` provides (duck-typed):
#       state_dict() -> String, load_state_dict(String), (optional) __str__()
#   - Optimizer duck-typed API in step wrappers: step(mut), zero_grad(mut)

from collections.list import List
from momijo.learn.distributed.collective import allreduce
from momijo.learn.distributed.collective import broadcast
from momijo.learn.distributed.collective import barrier

# -----------------------------------------------------------------------------
# DataParallel (single-process, multi-device placeholder)
# -----------------------------------------------------------------------------
# Intent:
#   - Scatter a batch across N local devices, run the model replicas, gather outputs.
#   - This skeleton leaves the actual device handling to future backends.
#   - Useful as a consistent type to plug into the training engine even before
#     real multi-device execution is implemented.

struct DataParallel:
    var _model
    var _devices: List[Int]
    var _chunks: Int

    fn __init__(out self, model, devices: List[Int] = List[Int](), chunks: Int = 1):
        self._model = model
        self._devices = devices
        self._chunks = chunks

    fn world_size(self) -> Int:
        # For single-process DP, we treat world size as number of local devices.
        var n = Int(self._devices.size())
        if n <= 0:
            return 1
        return n

    fn devices(self) -> List[Int]:
        return self._devices

    fn chunks(self) -> Int:
        return self._chunks

    fn set_chunks(mut self, chunks: Int):
        # Number of micro-batches to split input into.
        self._chunks = chunks

    fn model(self):
        # Access the wrapped model (read-only accessor).
        return self._model

    # Placeholder that would split the input into micro-batches per device.
    fn scatter(self, batch):
        # TODO: implement real chunking per device
        return batch

    # Placeholder that would combine per-device outputs back to a single output.
    fn gather(self, outputs):
        # TODO: implement real gather logic
        return outputs

    # Optional: string representation
    fn __str__(self) -> String:
        var n = self.world_size()
        return String("DataParallel(world_size=") + String(n) + String(", chunks=") + String(self._chunks) + String(")")


# -----------------------------------------------------------------------------
# DistributedDataParallel (multi-process)
# -----------------------------------------------------------------------------
# Intent:
#   - Each process (rank) holds one replica of the model.
#   - After backward (or before optimizer.step), gradients are all-reduced across ranks.
#   - Parameters are typically broadcast from rank 0 at startup to ensure consistency.
#   - This skeleton is backend-agnostic; you must wire collective ops to your runtime.

struct DistributedDataParallel:
    var _model
    var _world_size: Int
    var _rank: Int

    fn __init__(out self, model, world_size: Int, rank: Int):
        self._model = model
        self._world_size = world_size
        self._rank = rank

    fn world_size(self) -> Int:
        return self._world_size

    fn rank(self) -> Int:
        return self._rank

    fn model(self):
        return self._model

    # -- Synchronization primitives (placeholders to be wired to real tensor ops) --

    # Broadcast parameters from src (usually rank 0) to all ranks.
    fn broadcast_parameters(self, src: Int = 0):
        # In a real implementation, iterate model parameters and broadcast tensors.
        # Here we serialize the state dict on src and broadcast the string.
        if self._rank == src:
            var state = self._model.state_dict()  # expected: String
            _ = broadcast(state, src)
        else:
            var state = String("")  # placeholder that broadcast would fill
            state = broadcast(state, src)
            self._model.load_state_dict(state)

        barrier()  # ensure all ranks aligned

    # All-reduce gradients across ranks (sum/mean). Here we expose a generic hook.
    # Caller should pass in an abstract "grad_blob" or implement tensor-wise reduce inside model.
    fn allreduce_gradients(self, grad_blob) ->:
        # Return the reduced gradients. With a real backend, choose "sum" or "mean".
        # Here, we simply call the collective façade; no-op if not implemented.
        var reduced = allreduce(grad_blob)
        return reduced

    # Convenience: typical DDP step sequence
    #  - `found_inf`: pass True if AMP overflow detected to skip stepping.
    fn step_and_sync(
        mut self,
        optimizer,
        grad_blob,
        found_inf: Bool = False,
        average: Bool = True
    ):
        # Reduce gradients
        var reduced = self.allreduce_gradients(grad_blob)
        # If averaging is requested, caller should divide by world_size later
        # when wiring real tensor math. Kept here for API clarity.

        # Skip step on overflow (AMP)
        if not found_inf:
            optimizer.step()
        # Zero grads regardless, to match common practice
        optimizer.zero_grad()

    fn __str__(self) -> String:
        return String("DDP(world_size=") + String(self._world_size) + String(", rank=") + String(self._rank) + String(")")
