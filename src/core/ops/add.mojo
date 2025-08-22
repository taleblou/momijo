# MIT License
# Copyright (c) 2025 Morteza Talebou and Mitra Daneshmand
# Project: momijo  |  Source: https://github.com/taleblou/momijo
# This file is part of the Momijo project. See the LICENSE file at the repository root.
# Momijo 
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Morteza Taleblou and Mitra Daneshmand
# Website: https://taleblou.ir/
# Repository: https://github.com/taleblou/momijo
#
# Project: momijo.core
# File: momijo/core/ops/add.mojo
#
# This file is part of the Momijo project.
# See the LICENSE file at the repository root for license information. 

from momijo.core.error import MomijoError
from momijo.core.result import Result
from builtin.dtype import DType
from momijo.core.tensor import Tensor

# -------------------------
# Helpers
# -------------------------

@staticmethod
fn _device_check[T, U](a: Tensor[T], b: Tensor[U]) -> Result[None]:
    if a.device() != b.device():
        return Result[None].fail(MomijoError.invalid_argument("device mismatch: " + a.device() + " vs " + b.device(), "momijo.core.ops.add"), None)
    return Result[None].ok(None)

# -------------------------
# Generic add (same element type)
# -------------------------

fn add_same[T](a: Tensor[T], b: Tensor[T]) -> Result[Tensor[T]]:
    var ok = _device_check[T, T](a, b)
    if ok.is_err():
        # fallback: return a copy of `a` to keep a valid Tensor inside Result
        return Result[Tensor[T]].fail(ok.error(), a.copy())
    # Use Tensor.zip_with which supports broadcasting
    return a.zip_with[T, T](b, fn(x: T, y: T) -> T: x + y, a.dtype())

# Backward: ∂(a+b)/∂a = 1 , ∂(a+b)/∂b = 1  (ignoring broadcasting reductions)
# For now, we assume upstream gradient already matches the forward output shape.
fn add_backward_same[T](dy: Tensor[T]) -> (Tensor[T], Tensor[T]):
    # In a full autograd system, if broadcasting was used, we would sum-reduce over broadcasted axes.
    # Here we just pass-through for simplicity.
    return (dy.copy(), dy.copy())

# -------------------------
# Concrete specializations
# -------------------------

fn add_f64(a: Tensor[Float64], b: Tensor[Float64]) -> Result[Tensor[Float64]]:
    return add_same[Float64](a, b)

fn add_i64(a: Tensor[Int64], b: Tensor[Int64]) -> Result[Tensor[Int64]]:
    return add_same[Int64](a, b)

# Scalars
fn add_scalar_f64(a: Tensor[Float64], s: Float64) -> Tensor[Float64]:
    return a.add_scalar(s)

fn add_scalar_i64(a: Tensor[Int64], s: Int64) -> Tensor[Int64]:
    return a.add_scalar(s)

# -------------------------
# Mixed-type add (upcast both to Float64)
# -------------------------
# This is a pragmatic helper when T differs. We cast both to Float64 and add.
# The caller can later cast back if desired.

fn add_auto_f64[A, B](a: Tensor[A], b: Tensor[B]) -> Result[Tensor[Float64]]:
    var ok = _device_check[A, B](a, b)
    if ok.is_err():
        return Result[Tensor[Float64]].fail(ok.error(), Tensor[Float64].full(a.shape(), 0.0, DType.f64(), a.device()))
    # convert functions (best-effort String(...) → not used; rely on explicit convert lambdas)
    var a64 = a.astype[Float64](DType.f64(), fn(x: A) -> Float64: Float64(x))
    var b64 = b.astype[Float64](DType.f64(), fn(y: B) -> Float64: Float64(y))
    return add_same[Float64](a64, b64)

# -------------------------
# UX-friendly alias (Float64 default)
# -------------------------
# If your project mostly uses Float64 tensors, you can import `add` directly.

fn add(a: Tensor[Float64], b: Tensor[Float64]) -> Result[Tensor[Float64]]:
    return add_f64(a, b)

fn add_backward(dy: Tensor[Float64]) -> (Tensor[Float64], Tensor[Float64]):
    return add_backward_same[Float64](dy)