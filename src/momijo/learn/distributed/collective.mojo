# Project:      Momijo
# Module:       learn.distributed.collective
# File:         distributed/collective.mojo
# Path:         src/momijo/learn/distributed/collective.mojo
#
# Description:  Collective communication primitives for Momijo Learn.
#               This backend-agnostic facade currently implements single-process
#               (no-op) semantics so code can run identically on a single host.
#               The API is designed to be wired later to true multi-process backends
#               (e.g., NCCL/IPC/RDMA). Signatures are stable.
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
#   - Types: ReduceOp
#   - Core fns (single-process safe): init_process_group, get_world_size, get_rank,
#     allreduce(x), allreduce_sum(...), broadcast(...), barrier()
#   - Overloads provided for Int, Float64, and List[...] variants.
#   - No global mutable state; current implementation assumes single rank (0/1).


from collections.list import List

# -----------------------------------------------------------------------------
# Reduce operation tag (for future backends)
# -----------------------------------------------------------------------------
struct ReduceOp:
    var code: Int

    fn __init__(out self, code: Int):
        self.code = code

    @staticmethod
    fn sum() -> ReduceOp:
        return ReduceOp(0)

    @staticmethod
    fn prod() -> ReduceOp:
        return ReduceOp(1)

    @staticmethod
    fn max() -> ReduceOp:
        return ReduceOp(2)

    @staticmethod
    fn min() -> ReduceOp:
        return ReduceOp(3)

    @staticmethod
    fn avg() -> ReduceOp:
        return ReduceOp(4)


# -----------------------------------------------------------------------------
# Process group facade (single-process placeholder)
# -----------------------------------------------------------------------------
fn init_process_group(backend: String = String("single"), world_size: Int = 1, rank: Int = 0):
    # Placeholder: in future, initialize transport/context here.
    # Single-process: nothing to do.
    assert(world_size == 1, "Only single-process is supported in the placeholder implementation.")
    assert(rank == 0, "Only rank 0 is supported in the placeholder implementation.")
    return

fn destroy_process_group():
    # Placeholder to mirror APIs like torch.distributed.destroy_process_group()
    return

fn get_world_size() -> Int:
    # Single-process
    return 1

fn get_rank() -> Int:
    # Single-process
    return 0


# -----------------------------------------------------------------------------
# Helpers (single-process arithmetic)
# -----------------------------------------------------------------------------
# Sum for lists
fn _sum_int(xs: List[Int]) -> Int:
    var s: Int = 0
    var i: Int = 0
    while i < Int(xs.size()):
        s = s + xs[i]
        i = i + 1
    return s

fn _sum_f64(xs: List[Float64]) -> Float64:
    var s: Float64 = 0.0
    var i: Int = 0
    while i < Int(xs.size()):
        s = s + xs[i]
        i = i + 1
    return s

# Elementwise ops for lists (same-length required)
fn _ewise_int(a: List[Int], b: List[Int], op: ReduceOp) -> List[Int]:
    assert(Int(a.size()) == Int(b.size()), "List sizes must match for elementwise reduction.")
    var out = List[Int]()
    out.reserve(a.size())
    var i: Int = 0
    while i < Int(a.size()):
        var v: Int = 0
        if op.code == ReduceOp.sum().code:
            v = a[i] + b[i]
        elif op.code == ReduceOp.prod().code:
            v = a[i] * b[i]
        elif op.code == ReduceOp.max().code:
            v = (a[i] if a[i] > b[i] else b[i])
        elif op.code == ReduceOp.min().code:
            v = (a[i] if a[i] < b[i] else b[i])
        else:
            # avg is not well-defined without count here; caller should divide after sum.
            v = a[i] + b[i]
        out.push_back(v)
        i = i + 1
    return out

fn _ewise_f64(a: List[Float64], b: List[Float64], op: ReduceOp) -> List[Float64]:
    assert(Int(a.size()) == Int(b.size()), "List sizes must match for elementwise reduction.")
    var out = List[Float64]()
    out.reserve(a.size())
    var i: Int = 0
    while i < Int(a.size()):
        var v: Float64 = 0.0
        if op.code == ReduceOp.sum().code:
            v = a[i] + b[i]
        elif op.code == ReduceOp.prod().code:
            v = a[i] * b[i]
        elif op.code == ReduceOp.max().code:
            v = (a[i] if a[i] > b[i] else b[i])
        elif op.code == ReduceOp.min().code:
            v = (a[i] if a[i] < b[i] else b[i])
        else:
            v = a[i] + b[i]
        out.push_back(v)
        i = i + 1
    return out


# -----------------------------------------------------------------------------
# Public collectives — single-process semantics (no-op or deterministic local op)
# -----------------------------------------------------------------------------
# NOTE: We keep the original signatures (x only) and add typed overloads
# to cover common data forms. Future backends can use ReduceOp to select kernels.

# Generic pass-through (kept for backward compatibility with the original stub)
fn allreduce(x):
    # In single-process, allreduce equals identity.
    return x

# Scalars
fn allreduce_sum(x: Int) -> Int:
    # Single-process: just return x
    return x

fn allreduce_sum_f64(x: Float64) -> Float64:
    return x

# Lists (elementwise)
fn allreduce_sum(xs: List[Int]) -> List[Int]:
    # Single-process: return a copy (preserve value semantics)
    var out = List[Int]()
    out.reserve(xs.size())
    var i: Int = 0
    while i < Int(xs.size()):
        out.push_back(xs[i])
        i = i + 1
    return out

fn allreduce_sum_f64(xs: List[Float64]) -> List[Float64]:
    var out = List[Float64]()
    out.reserve(xs.size())
    var i: Int = 0
    while i < Int(xs.size()):
        out.push_back(xs[i])
        i = i + 1
    return out

# Broadcast: returns x if src == 0; asserts single-process otherwise
fn broadcast(x, src: Int = 0):
    assert(src == 0, "Only rank 0 exists in single-process broadcast.")
    return x

# Barrier: no-op in single-process
fn barrier():
    # In a true backend, this would synchronize all ranks.
    return

# Optional extended API: reduce with explicit op (for future wiring)
# (Overloads for Int/List[Int]/Float64/List[Float64])

fn reduce(x: Int, op: ReduceOp = ReduceOp.sum(), dst: Int = 0) -> Int:
    assert(dst == 0, "Only rank 0 exists in single-process reduce.")
    # Single-process reduction equals identity
    return x

fn reduce_f64(x: Float64, op: ReduceOp = ReduceOp.sum(), dst: Int = 0) -> Float64:
    assert(dst == 0, "Only rank 0 exists in single-process reduce.")
    return x

fn reduce_list_int(xs: List[Int], op: ReduceOp = ReduceOp.sum(), dst: Int = 0) -> List[Int]:
    assert(dst == 0, "Only rank 0 exists in single-process reduce.")
    # Identity copy
    var out = List[Int]()
    out.reserve(xs.size())
    var i: Int = 0
    while i < Int(xs.size()):
        out.push_back(xs[i])
        i = i + 1
    return out

fn reduce_list_f64(xs: List[Float64], op: ReduceOp = ReduceOp.sum(), dst: Int = 0) -> List[Float64]:
    assert(dst == 0, "Only rank 0 exists in single-process reduce.")
    var out = List[Float64]()
    out.reserve(xs.size())
    var i: Int = 0
    while i < Int(xs.size()):
        out.push_back(xs[i])
        i = i + 1
    return out
