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
# Project: momijo.tensor
# File: src/momijo/tensor/ufunc.mojo
 from momijo.tensor.tensor_base import Tensor
from momijo.tensor.errors import check_same_shape
from momijo.tensor.strides import shape_product

 
from momijo.core.ndarray import offset
from momijo.tensor.tensor import to_contiguous
 
 
from momijo.tensor.tensor import Tensor

# Elementwise add: supports non-contiguous inputs by materializing contiguous views
fn add(a: Tensor[Float64], b: Tensor[Float64]) -> Tensor[Float64]:
    _require(a.shape == b.shape, String("Shapes must match for elementwise add"))
    var ac = a.to_contiguous()
    var bc = b.to_contiguous()
    var out = Tensor[Float64](shape=ac.shape, fill=0.0)
    var n = ac.size()
    var i = 0
    while i < n:
        out.data[i] = ac.data[ac.offset + i] + bc.data[bc.offset + i]
        i += 1
    return out

# Elementwise multiply
fn mul(a: Tensor[Float64], b: Tensor[Float64]) -> Tensor[Float64]:
    _require(a.shape == b.shape, String("Shapes must match for elementwise mul"))
    var ac = a.to_contiguous()
    var bc = b.to_contiguous()
    var out = Tensor[Float64](shape=ac.shape, fill=0.0)
    var n = ac.size()
    var i = 0
    while i < n:
        out.data[i] = ac.data[ac.offset + i] * bc.data[bc.offset + i]
        i += 1
    return out

# Add scalar to every element
fn scalar_add(a: Tensor[Float64], c: Float64) -> Tensor[Float64]:
    var ac = a.to_contiguous()
    var out = Tensor[Float64](shape=ac.shape, fill=0.0)
    var n = ac.size()
    var i = 0
    while i < n:
        out.data[i] = ac.data[ac.offset + i] + c
        i += 1
    return out


# Helpers
fn _alloc_like[T: Copyable & Movable](a: Tensor[T]) -> Tensor[T]:
    return Tensor[T](a.shape(), a.get_flat(0), a._dtype)

fn add[T: Copyable & Movable](a: Tensor[T], b: Tensor[T]) -> Tensor[T]:
    check_same_shape(a.shape(), b.shape(), "add")
    var out = _alloc_like(a)
    var n = a.size()
    var i = 0
    while i < n:
        out.set_flat(i, a.get_flat(i) + b.get_flat(i))
        i += 1
    return out

fn sub[T: Copyable & Movable](a: Tensor[T], b: Tensor[T]) -> Tensor[T]:
    check_same_shape(a.shape(), b.shape(), "sub")
    var out = _alloc_like(a)
    var n = a.size()
    var i = 0
    while i < n:
        out.set_flat(i, a.get_flat(i) - b.get_flat(i))
        i += 1
    return out

fn mul[T: Copyable & Movable](a: Tensor[T], b: Tensor[T]) -> Tensor[T]:
    check_same_shape(a.shape(), b.shape(), "mul")
    var out = _alloc_like(a)
    var n = a.size()
    var i = 0
    while i < n:
        out.set_flat(i, a.get_flat(i) * b.get_flat(i))
        i += 1
    return out

fn div[T: Copyable & Movable](a: Tensor[T], b: Tensor[T]) -> Tensor[T]:
    check_same_shape(a.shape(), b.shape(), "div")
    var out = _alloc_like(a)
    var n = a.size()
    var i = 0
    while i < n:
        out.set_flat(i, a.get_flat(i) / b.get_flat(i))
        i += 1
    return out

fn pow_[T: Copyable & Movable](a: Tensor[T], b: Tensor[T]) -> Tensor[T]:
    check_same_shape(a.shape(), b.shape(), "pow")
    var out = _alloc_like(a)
    var n = a.size()
    var i = 0
    while i < n:
        out.set_flat(i, a.get_flat(i) ** b.get_flat(i))
        i += 1
    return out

# Comparison operators -> Bool tensor
fn eq[T: Copyable & Movable](a: Tensor[T], b: Tensor[T]) -> Tensor[Bool]:
    check_same_shape(a.shape(), b.shape(), "eq")
    var out = Tensor[Bool](a.shape(), False, a._dtype)
    var n = a.size()
    var i = 0
    while i < n:
        out.set_flat(i, a.get_flat(i) == b.get_flat(i))
        i += 1
    return out

fn ne[T: Copyable & Movable](a: Tensor[T], b: Tensor[T]) -> Tensor[Bool]:
    check_same_shape(a.shape(), b.shape(), "ne")
    var out = Tensor[Bool](a.shape(), False, a._dtype)
    var n = a.size()
    var i = 0
    while i < n:
        out.set_flat(i, a.get_flat(i) != b.get_flat(i))
        i += 1
    return out

fn lt[T: Copyable & Movable](a: Tensor[T], b: Tensor[T]) -> Tensor[Bool]:
    check_same_shape(a.shape(), b.shape(), "lt")
    var out = Tensor[Bool](a.shape(), False, a._dtype)
    var n = a.size()
    var i = 0
    while i < n:
        out.set_flat(i, a.get_flat(i) < b.get_flat(i))
        i += 1
    return out

fn le[T: Copyable & Movable](a: Tensor[T], b: Tensor[T]) -> Tensor[Bool]:
    check_same_shape(a.shape(), b.shape(), "le")
    var out = Tensor[Bool](a.shape(), False, a._dtype)
    var n = a.size()
    var i = 0
    while i < n:
        out.set_flat(i, a.get_flat(i) <= b.get_flat(i))
        i += 1
    return out

fn gt[T: Copyable & Movable](a: Tensor[T], b: Tensor[T]) -> Tensor[Bool]:
    check_same_shape(a.shape(), b.shape(), "gt")
    var out = Tensor[Bool](a.shape(), False, a._dtype)
    var n = a.size()
    var i = 0
    while i < n:
        out.set_flat(i, a.get_flat(i) > b.get_flat(i))
        i += 1
    return out

fn ge[T: Copyable & Movable](a: Tensor[T], b: Tensor[T]) -> Tensor[Bool]:
    check_same_shape(a.shape(), b.shape(), "ge")
    var out = Tensor[Bool](a.shape(), False, a._dtype)
    var n = a.size()
    var i = 0
    while i < n:
        out.set_flat(i, a.get_flat(i) >= b.get_flat(i))
        i += 1
    return out

# Math ufuncs
fn neg[T: Copyable & Movable](a: Tensor[T]) -> Tensor[T]:
    var out = _alloc_like(a)
    var n = a.size()
    var i = 0
    while i < n:
        out.set_flat(i, -a.get_flat(i))
        i += 1
    return out

fn abs_[T: Copyable & Movable](a: Tensor[T]) -> Tensor[T]:
    var out = _alloc_like(a)
    var n = a.size()
    var i = 0
    while i < n:
        var v = a.get_flat(i)
        if v < 0: v = -v
        out.set_flat(i, v)
        i += 1
    return out

fn exp[T: Copyable & Movable](a: Tensor[T]) -> Tensor[T]:
    var out = _alloc_like(a)
    var n = a.size()
    var i = 0
    while i < n:
        out.set_flat(i, builtin.exp(a.get_flat(i)))
        i += 1
    return out

fn log[T: Copyable & Movable](a: Tensor[T]) -> Tensor[T]:
    var out = _alloc_like(a)
    var n = a.size()
    var i = 0
    while i < n:
        out.set_flat(i, builtin.log(a.get_flat(i)))
        i += 1
    return out

fn sin[T: Copyable & Movable](a: Tensor[T]) -> Tensor[T]:
    var out = _alloc_like(a)
    var n = a.size()
    var i = 0
    while i < n:
        out.set_flat(i, builtin.sin(a.get_flat(i)))
        i += 1
    return out

fn cos[T: Copyable & Movable](a: Tensor[T]) -> Tensor[T]:
    var out = _alloc_like(a)
    var n = a.size()
    var i = 0
    while i < n:
        out.set_flat(i, builtin.cos(a.get_flat(i)))
        i += 1
    return out

fn tan[T: Copyable & Movable](a: Tensor[T]) -> Tensor[T]:
    var out = _alloc_like(a)
    var n = a.size()
    var i = 0
    while i < n:
        out.set_flat(i, builtin.tan(a.get_flat(i)))
        i += 1
    return out
