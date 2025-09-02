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
# Project: momijo.core.ops
# File: momijo/core/ops/mul.mojo

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
        return Result[None].fail(MomijoError.invalid_argument("device mismatch: " + a.device() + " vs " + b.device(), "momijo.core.ops.mul"), None)
    return Result[None].ok(None)

# -------------------------
# Generic mul (same element type)
# -------------------------

fn mul_same[T: Copyable & Movable](a: Tensor[T], b: Tensor[T]) -> Result[Tensor[T]]:
    var ok = _device_check[T, T](a, b)
    if ok.is_err():
        # fallback keeps a valid tensor in Result
        return Result[Tensor[T]].fail(ok.error(), a.copy())
    # Use Tensor.zip_with which supports broadcasting
    return a.zip_with[T, T](b, fn(x: T, y: T) -> T: x * y, a.dtype())

# NOTE: [non-ASCII comment removed]
# NOTE: If forward used broadcasting, a fully-correct backward must sum-reduce
# over broadcasted axes back to a/b original shapes. This naive version returns
# tensors in the forward's output shape.
fn mul_backward_same[T: Copyable & Movable](a: Tensor[T], b: Tensor[T], dy: Tensor[T]) -> (Tensor[T], Tensor[T]):
    # da = dy * b
    var da_res = dy.zip_with[T, T](b, fn(gy: T, bv: T) -> T: gy * bv, dy.dtype())
    var da = da_res.is_ok() ? da_res.value() : dy.copy()
    # db = dy * a
    var db_res = dy.zip_with[T, T](a, fn(gy: T, av: T) -> T: gy * av, dy.dtype())
    var db = db_res.is_ok() ? db_res.value() : dy.copy()
    return (da, db)

# -------------------------
# Concrete specializations
# -------------------------

fn mul_f64(a: Tensor[Float64], b: Tensor[Float64]) -> Result[Tensor[Float64]]:
    return mul_same[Float64](a, b)

fn mul_i64(a: Tensor[Int64], b: Tensor[Int64]) -> Result[Tensor[Int64]]:
    return mul_same[Int64](a, b)

# Scalars
fn mul_scalar_f64(a: Tensor[Float64], s: Float64) -> Tensor[Float64]:
    return a.mul_scalar(s)

fn mul_scalar_i64(a: Tensor[Int64], s: Int64) -> Tensor[Int64]:
    return a.mul_scalar(s)

# -------------------------
# Mixed-type mul (upcast to Float64 pragmatically)
# -------------------------

fn mul_auto_f64[A, B](a: Tensor[A], b: Tensor[B]) -> Result[Tensor[Float64]]:
    var ok = _device_check[A, B](a, b)
    if ok.is_err():
        return Result[Tensor[Float64]].fail(ok.error(), Tensor[Float64].full(a.shape(), 0.0, DType.f64(), a.device()))
    var a64 = a.astype[Float64](DType.f64(), fn(x: A) -> Float64: Float64(x))
    var b64 = b.astype[Float64](DType.f64(), fn(y: B) -> Float64: Float64(y))
    return mul_same[Float64](a64, b64)

# -------------------------
# UX-friendly alias (Float64 default)
# -------------------------

fn mul(a: Tensor[Float64], b: Tensor[Float64]) -> Result[Tensor[Float64]]:
    return mul_f64(a, b)

fn mul_backward(a: Tensor[Float64], b: Tensor[Float64], dy: Tensor[Float64]) -> (Tensor[Float64], Tensor[Float64]):
    return mul_backward_same[Float64](a, b, dy)