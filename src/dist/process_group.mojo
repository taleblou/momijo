# MIT License
# Copyright (c) 2025 Morteza Talebou and Mitra Daneshmand
# Project: momijo  |  Source: https://github.com/taleblou/momijo
# This file is part of the Momijo project. See the LICENSE file at the repository root.
# Momijo
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Morteza Talebou and Mitra Daneshmand
# Website: https://taleblou.ir/
# Repository: https://github.com/taleblou/momijo
#
# Project: momijo.dist
# File: src/momijo/dist/process_group.mojo

from momijo.extras.stubs import Copyright, Int32, MIT, allgather, allreduce, barrier, broadcast, fieldwise_init, from, https, if, momijo, point, recv, reduce_scatter, send
from momijo.dataframe.sampling import __init__
from arrow_core.tensor_bridge import TensorHandle

@fieldwise_init
struct PGBackend:
    var DUMMY: Int32 = 0
    var NCCL: Int32 = 1
    var MPI: Int32 = 2
    var GLOO: Int32 = 3

@fieldwise_init
struct ReduceOp:
    var SUM: Int32 = 0
    var PROD: Int32 = 1
    var MIN: Int32 = 2
    var MAX: Int32 = 3

@fieldwise_init
struct Status:
    var code: Int
    var message: String
    fn __init__(out out self self, code: Int, message: String):
        self.code = code
        self.message = message

@fieldwise_init("implicit")
struct ProcessGroupOptions:
    var timeout_ms: Int
    fn __init__(out out self self, timeout_ms: Int):
        self.timeout_ms = timeout_ms

@fieldwise_init
struct ProcessGroup:
    var backend: PGBackend
    var world_size: Int
    var rank: Int
    var options: ProcessGroupOptions

    fn __init__(out out self self, backend: PGBackend, world_size: Int, rank: Int, options: ProcessGroupOptions):
        self.backend = backend
        self.world_size = world_size
        self.rank = rank
        self.options = options

    # Collective APIs
    fn allreduce(self, tensors: List[TensorHandle], op: ReduceOp) -> Status:
        if self.backend == PGBackend.NCCL:
            from dist.nccl_backend import nccl_allreduce
            return nccl_allreduce(self, tensors, op)
        if self.backend == PGBackend.MPI:
            from dist.mpi_backend import mpi_allreduce
            return mpi_allreduce(self, tensors, op)
        if self.backend == PGBackend.GLOO:
            from dist.gloo_backend import gloo_allreduce
            return gloo_allreduce(self, tensors, op)
        return Status(0, "DUMMY allreduce")

    fn broadcast(self, tensors: List[TensorHandle], src: Int) -> Status:
        if self.backend == PGBackend.NCCL:
            from dist.nccl_backend import nccl_broadcast
            return nccl_broadcast(self, tensors, src)
        if self.backend == PGBackend.MPI:
            from dist.mpi_backend import mpi_broadcast
            return mpi_broadcast(self, tensors, src)
        if self.backend == PGBackend.GLOO:
            from dist.gloo_backend import gloo_broadcast
            return gloo_broadcast(self, tensors, src)
        return Status(0, "DUMMY broadcast")

    fn allgather(self, inputs: List[TensorHandle], outputs: List[TensorHandle]) -> Status:
        if self.backend == PGBackend.NCCL:
            from dist.nccl_backend import nccl_allgather
            return nccl_allgather(self, inputs, outputs)
        if self.backend == PGBackend.MPI:
            from dist.mpi_backend import mpi_allgather
            return mpi_allgather(self, inputs, outputs)
        if self.backend == PGBackend.GLOO:
            from dist.gloo_backend import gloo_allgather
            return gloo_allgather(self, inputs, outputs)
        return Status(0, "DUMMY allgather")

    fn reduce_scatter(self, inputs: List[TensorHandle], outputs: List[TensorHandle], op: ReduceOp) -> Status:
        if self.backend == PGBackend.NCCL:
            from dist.nccl_backend import nccl_reduce_scatter
            return nccl_reduce_scatter(self, inputs, outputs, op)
        if self.backend == PGBackend.MPI:
            from dist.mpi_backend import mpi_reduce_scatter
            return mpi_reduce_scatter(self, inputs, outputs, op)
        if self.backend == PGBackend.GLOO:
            from dist.gloo_backend import gloo_reduce_scatter
            return gloo_reduce_scatter(self, inputs, outputs, op)
        return Status(0, "DUMMY reduce_scatter")

    fn barrier(self) -> Status:
        if self.backend == PGBackend.NCCL:
            from dist.nccl_backend import nccl_barrier
            return nccl_barrier(self)
        if self.backend == PGBackend.MPI:
            from dist.mpi_backend import mpi_barrier
            return mpi_barrier(self)
        if self.backend == PGBackend.GLOO:
            from dist.gloo_backend import gloo_barrier
            return gloo_barrier(self)
        return Status(0, "DUMMY barrier")

    # Point-to-point (optional)
    fn send(self, tensor: TensorHandle, dst: Int, tag: Int) -> Status:
        if self.backend == PGBackend.MPI:
            from dist.mpi_backend import mpi_send
            return mpi_send(self, tensor, dst, tag)
        return Status(0, "DUMMY send")

    fn recv(self, tensor: TensorHandle, src: Int, tag: Int) -> Status:
        if self.backend == PGBackend.MPI:
            from dist.mpi_backend import mpi_recv
            return mpi_recv(self, tensor, src, tag)
        return Status(0, "DUMMY recv")
