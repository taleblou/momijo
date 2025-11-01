# MIT License
# SPDX-License-Identifier: MIT
# Project:      Momijo
# Module:       learn.distributed.collective
# File:         src/momijo/learn/distributed/collective.mojo
#
# Description:
#   Collective communication primitives for Momijo Learn.
#   This backend-agnostic facade currently implements single-process semantics
#   (deterministic local / no-op operations) so code can run identically on one host.
#   API is designed to be wired later to true multi-process backends (e.g., NCCL/IPC/RDMA).
#
# Author(s):    Morteza Taleblou & Mitra Daneshmand
# Website:      https://taleblou.ir/
# Repository:   https://github.com/taleblou/momijo
#
# Notes:
#   - Types: ReduceOp
#   - Core fns (single-process safe): init_process_group, destroy_process_group,
#     get_world_size, get_rank, allreduce(x), allreduce_sum(...), broadcast(...), barrier()
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

    fn __str__(self) -> String:
        var s = "ReduceOp("
        if self.code == 0:
            s = s + "sum"
        elif self.code == 1:
            s = s + "prod"
        elif self.code == 2:
            s = s + "max"
        elif self.code == 3:
            s = s + "min"
        else:
            s = s + "avg"
        s = s + ")"
        return s


# -----------------------------------------------------------------------------
# Process group facade (single-process placeholder)
# -----------------------------------------------------------------------------
fn init_process_group(backend: String = String("single"), world_size: Int = 1, rank: Int = 0):
    # Placeholder: initialize transport/context here in a real backend.
    # Single-process: validate and return.
    assert(world_size == 1, "Only single-process is supported in the placeholder implementation.")
    assert(rank == 0, "Only rank 0 is supported in the placeholder implementation.")
    # 'backend' is currently informational; ignored in single-process.

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
# Sums (scalars and lists)
fn _sum_int(xs: List[Int]) -> Int:
    var s: Int = 0
    var i: Int = 0
    while i < len(xs)):
        s = s + xs[i]
        i = i + 1
    return s

fn _sum_f64(xs: List[Float64]) -> Float64:
    var s: Float64 = 0.0
    var i: Int = 0
    while i < len(xs)):
        s = s + xs[i]
        i = i + 1
    return s

# Elementwise ops for lists (same length)
fn _ewise_int(a: List[Int], b: List[Int], op: ReduceOp) -> List[Int]:
    assert(len(a) == len(b), "List sizes must match for elementwise reduction.")
    var out = List[Int]()
    out.reserve(len(a))
    var i: Int = 0
    while i < Int(len(a)):
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
            # avg is ambiguous elementwise without global count; default to sum.
            v = a[i] + b[i]
        out.push_back(v)
        i = i + 1
    return out

fn _ewise_f64(a: List[Float64], b: List[Float64], op: ReduceOp) -> List[Float64]:
    assert(len(a) == len(b), "List sizes must match for elementwise reduction.")
    var out = List[Float64]()
    out.reserve(len(a))
    var i: Int = 0
    while i < Int(len(a)):
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
# Public collectives — single-process semantics (no-op or local deterministic op)
# -----------------------------------------------------------------------------
# Generic pass-through (kept for compatibility with original stub)
fn allreduce(x):
    # In single-process, allreduce equals identity.
    return x

# Scalars
fn allreduce_sum(x: Int) -> Int:
    # Single-process: identity
    return x

fn allreduce_sum_f64(x: Float64) -> Float64:
    return x

# Lists (elementwise identity)
fn allreduce_sum(xs: List[Int]) -> List[Int]:
    var out = List[Int]()
    out.reserve(len(xs))
    var i: Int = 0
    while i < Int(len(xs)):
        out.push_back(xs[i])
        i = i + 1
    return out

fn allreduce_sum_f64(xs: List[Float64]) -> List[Float64]:
    var out = List[Float64]()
    out.reserve(len(xs))
    var i: Int = 0
    while i < Int(len(xs)):
        out.push_back(xs[i])
        i = i + 1
    return out

# Broadcast: return x if src == 0; assert single-process otherwise
fn broadcast(x, src: Int = 0):
    assert(src == 0, "Only rank 0 exists in single-process broadcast.")
    return x

# Barrier: no-op in single-process
fn barrier():
    return

# Optional extended API: reduce with explicit op (for future wiring)
# Overloads for Int/List[Int]/Float64/List[Float64]; all identity in single-process.

fn reduce(x: Int, op: ReduceOp = ReduceOp.sum(), dst: Int = 0) -> Int:
    assert(dst == 0, "Only rank 0 exists in single-process reduce.")
    return x

fn reduce_f64(x: Float64, op: ReduceOp = ReduceOp.sum(), dst: Int = 0) -> Float64:
    assert(dst == 0, "Only rank 0 exists in single-process reduce.")
    return x

fn reduce_list_int(xs: List[Int], op: ReduceOp = ReduceOp.sum(), dst: Int = 0) -> List[Int]:
    assert(dst == 0, "Only rank 0 exists in single-process reduce.")
    var out = List[Int]()
    out.reserve(len(xs))
    var i: Int = 0
    while i < len(xs):
        out.push_back(xs[i])
        i = i + 1
    return out

fn reduce_list_f64(xs: List[Float64], op: ReduceOp = ReduceOp.sum(), dst: Int = 0) -> List[Float64]:
    assert(dst == 0, "Only rank 0 exists in single-process reduce.")
    var out = List[Float64]()
    out.reserve(len(xs))
    var i: Int = 0
    while i < len(xs):
        out.push_back(xs[i])
        i = i + 1
    return out
