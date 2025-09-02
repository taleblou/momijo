# MIT License
# Project: momijo  |  Source: https://github.com/taleblou/momijo
# SPDX-License-Identifier: MIT
#
# Project: momijo.tensor
# File: momijo/tensor/registry.mojo

from momijo.tensor.tensor import Tensor

# -- Lightweight descriptors ---------------------------------------------------

struct Op:
    var name: String
    fn __init__(out self, name: String):
        self.name = name

struct Backend:
    var name: String
    fn __init__(out self, name: String):
        self.name = name

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
fn set_default_backend(_b: Backend):
    # Intentionally a no-op to avoid globals.
    return

# -- Reference kernels (stubs so tests can import) ----------------------------

alias F32Tensor = Tensor[Float32]
alias I32Tensor = Tensor[Int32]

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
fn register_default_kernels():
    return
