# Project:      Momijo
# Module:       src.momijo.tensor.registry
# File:         registry.mojo
# Path:         src/momijo/tensor/registry.mojo
#
# Description:  Core tensor/ndarray components: shapes/strides, broadcasting rules,
#               element-wise ops, and foundational kernels.
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
#   - Structs: Op, Backend
#   - Key functions: __init__, __copyinit__, __moveinit__, __init__, __copyinit__, __moveinit__, op_add, op_sum ...
#   - Uses generic functions/types with explicit trait bounds.


from momijo.tensor.tensor import Tensor

struct Op:
    var name: String
fn __init__(out self, name: String) -> None:
        self.name = name
fn __copyinit__(out self, other: Self) -> None:
        self.name = other.name
fn __moveinit__(out self, deinit other: Self) -> None:
        self.name = other.name
struct Backend:
    var name: String
fn __init__(out self, name: String) -> None:
        self.name = name
fn __copyinit__(out self, other: Self) -> None:
        self.name = other.name
fn __moveinit__(out self, deinit other: Self) -> None:
        self.name = other.name
# -- Accessors (instead of globals) -------------------------------------------
fn op_add()    -> Op:     return Op(String("add"))
fn op_sum()    -> Op:     return Op(String("sum"))
fn op_mean()   -> Op:     return Op(String("mean"))
fn op_matmul() -> Op:     return Op(String("matmul"))
fn backend_ref()  -> Backend: return Backend(String("ref"))
fn backend_simd() -> Backend: return Backend(String("simd"))
fn backend_gpu()  -> Backend: return Backend(String("gpu"))

# Default backend accessor (no mutable global state)
fn DEFAULT_BACKEND() -> Backend:
    return backend_ref()

# No-op setter to preserve API compatibility (no global state is changed).
fn set_default_backend(_b: Backend) -> None:
    # Intentionally a no-op to avoid globals.
    return

# -- Reference kernels (stubs so tests can import) ----------------------------


fn ref_add_f32(a: F32Tensor, b: F32Tensor, dst: F32Tensor) -> None:
    # TODO: implement reference kernel; this is a stub to satisfy imports.
    return
fn ref_add_i32(a: I32Tensor, b: I32Tensor, dst: I32Tensor) -> None:
    # TODO: implement reference kernel; this is a stub to satisfy imports.
    return

# -- Public ops (minimal, type-stable shims) ----------------------------------

# Float64 entry point (used by elementwise.add_f64 wrapper).
fn add(a: Tensor[Float64], b: Tensor[Float64]) -> Tensor[Float64]:
    # TODO: dispatch based on DEFAULT_BACKEND() and device when kernels are wired.
    return a

# Placeholder for a registration API hook
fn register_default_kernels() -> None:
    return