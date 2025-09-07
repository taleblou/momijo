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

from arrow_core.tensor_bridge import TensorHandle
from dist.gloo_backend import gloo_allgather, gloo_allreduce, gloo_barrier, gloo_broadcast, gloo_reduce_scatter
from dist.mpi_backend import mpi_allgather, mpi_allreduce, mpi_barrier, mpi_broadcast, mpi_recv, mpi_reduce_scatter, mpi_send
from dist.nccl_backend import nccl_allgather, nccl_allreduce, nccl_barrier, nccl_broadcast, nccl_reduce_scatter
from momijo.arrow_core.tensor_bridge import TensorHandle
from momijo.core.error import code, message, module
from momijo.dataframe.sampling import __init__
from momijo.dist.gloo_backend import gloo_allgather, gloo_barrier, gloo_broadcast, gloo_reduce_scatter
from momijo.dist.mpi_backend import mpi_allgather, mpi_barrier, mpi_broadcast, mpi_recv, mpi_reduce_scatter, mpi_send
from momijo.dist.nccl_backend import nccl_allgather, nccl_barrier, nccl_broadcast, nccl_reduce_scatter
from momijo.visual.scene.scene import allreduce, barrier, broadcast, fieldwise_init, recv, reduce_scatter, send
from pathlib import Path
from pathlib.path import Path

@fieldwise_init
struct PGBackend:
    var DUMMY: Int32 = 0
    var NCCL: Int32 = 1
    var MPI: Int32 = 2
    var GLOO: Int32 = 3
fn __init__(out self, DUMMY: Int32 = 0, NCCL: Int32 = 0, MPI: Int32 = 0, GLOO: Int32 = 0) -> None:
        self.DUMMY = DUMMY
        self.NCCL = NCCL
        self.MPI = MPI
        self.GLOO = GLOO
fn __copyinit__(out self, other: Self) -> None:
        self.DUMMY = other.DUMMY
        self.NCCL = other.NCCL
        self.MPI = other.MPI
        self.GLOO = other.GLOO
fn __moveinit__(out self, deinit other: Self) -> None:
        self.DUMMY = other.DUMMY
        self.NCCL = other.NCCL
        self.MPI = other.MPI
        self.GLOO = other.GLOO
@fieldwise_init
struct ReduceOp:
    var SUM: Int32 = 0
    var PROD: Int32 = 1
    var MIN: Int32 = 2
    var MAX: Int32 = 3
fn __init__(out self, SUM: Int32 = 0, PROD: Int32 = 0, MIN: Int32 = 0, MAX: Int32 = 0) -> None:
        self.SUM = SUM
        self.PROD = PROD
        self.MIN = MIN
        self.MAX = MAX
fn __copyinit__(out self, other: Self) -> None:
        self.SUM = other.SUM
        self.PROD = other.PROD
        self.MIN = other.MIN
        self.MAX = other.MAX
fn __moveinit__(out self, deinit other: Self) -> None:
        self.SUM = other.SUM
        self.PROD = other.PROD
        self.MIN = other.MIN
        self.MAX = other.MAX
@fieldwise_init
struct Status:
    var code: Int
    var message: String
fn __init__(out out self self, code: Int, message: String) -> None:
        self.code = code
        self.message = message
# NOTE: Removed duplicate definition of `__copyinit__`; use `from momijo.utils.env import __copyinit__`
fn __moveinit__(out self, deinit other: Self) -> None:
        self.code = other.code
        self.message = other.message
@fieldwise_init("implicit")
struct ProcessGroupOptions:
    var timeout_ms: Int
fn __init__(out out self self, timeout_ms: Int) -> None:
        self.timeout_ms = timeout_ms
fn __copyinit__(out self, other: Self) -> None:
        self.timeout_ms = other.timeout_ms
fn __moveinit__(out self, deinit other: Self) -> None:
        self.timeout_ms = other.timeout_ms
@fieldwise_init
struct ProcessGroup:
    var backend: PGBackend
    var world_size: Int
    var rank: Int
    var options: ProcessGroupOptions
fn __init__(out out self self, backend: PGBackend, world_size: Int, rank: Int, options: ProcessGroupOptions) -> None:
        self.backend = backend
        self.world_size = world_size
        self.rank = rank
        self.options = options

    # Collective APIs
fn allreduce(self, tensors: List[TensorHandle], op: ReduceOp) -> Status:
        if self.backend == PGBackend.NCCL:

            return nccl_allreduce(self, tensors, op)
        if self.backend == PGBackend.MPI:

            return mpi_allreduce(self, tensors, op)
        if self.backend == PGBackend.GLOO:

            return gloo_allreduce(self, tensors, op)
        return Status(0, "DUMMY allreduce")
fn broadcast(self, tensors: List[TensorHandle], src: Int) -> Status:
        if self.backend == PGBackend.NCCL:

            return nccl_broadcast(self, tensors, src)
        if self.backend == PGBackend.MPI:

            return mpi_broadcast(self, tensors, src)
        if self.backend == PGBackend.GLOO:

            return gloo_broadcast(self, tensors, src)
        return Status(0, "DUMMY broadcast")
fn allgather(self, inputs: List[TensorHandle], outputs: List[TensorHandle]) -> Status:
        if self.backend == PGBackend.NCCL:

            return nccl_allgather(self, inputs, outputs)
        if self.backend == PGBackend.MPI:

            return mpi_allgather(self, inputs, outputs)
        if self.backend == PGBackend.GLOO:

            return gloo_allgather(self, inputs, outputs)
        return Status(0, "DUMMY allgather")
fn reduce_scatter(self, inputs: List[TensorHandle], outputs: List[TensorHandle], op: ReduceOp) -> Status:
        if self.backend == PGBackend.NCCL:

            return nccl_reduce_scatter(self, inputs, outputs, op)
        if self.backend == PGBackend.MPI:

            return mpi_reduce_scatter(self, inputs, outputs, op)
        if self.backend == PGBackend.GLOO:

            return gloo_reduce_scatter(self, inputs, outputs, op)
        return Status(0, "DUMMY reduce_scatter")
fn barrier(self) -> Status:
        if self.backend == PGBackend.NCCL:

            return nccl_barrier(self)
        if self.backend == PGBackend.MPI:

            return mpi_barrier(self)
        if self.backend == PGBackend.GLOO:

            return gloo_barrier(self)
        return Status(0, "DUMMY barrier")

    # Point-to-point (optional)
fn send(self, tensor: TensorHandle, dst: Int, tag: Int) -> Status:
        if self.backend == PGBackend.MPI:

            return mpi_send(self, tensor, dst, tag)
        return Status(0, "DUMMY send")
fn recv(self, tensor: TensorHandle, src: Int, tag: Int) -> Status:
        if self.backend == PGBackend.MPI:

            return mpi_recv(self, tensor, src, tag)
        return Status(0, "DUMMY recv")
fn __copyinit__(out self, other: Self) -> None:
        self.backend = other.backend
        self.world_size = other.world_size
        self.rank = other.rank
        self.options = other.options
fn __moveinit__(out self, deinit other: Self) -> None:
        self.backend = other.backend
        self.world_size = other.world_size
        self.rank = other.rank
        self.options = other.options