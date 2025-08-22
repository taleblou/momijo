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
# File: momijo/core/tensor.mojo
#
# This file is part of the Momijo project.
# See the LICENSE file at the repository root for license information. 


from builtin.dtype import DType
from momijo.core.device import Device, CPUDevice

# Tensor with a naive CPU data buffer for demo ops.
struct Tensor(Copyable, Movable):
    var shape: List[Int]
    var stride: List[Int]
    var dtype: DType
    var device: Device
    var data: List[Float64]      # row-major

    fn __init__(out self, shape: List[Int], dtype: DType = DType.float64(), device: Device = CPUDevice()):
        self.shape = shape
        # compute row-major stride
        self.stride = List[Int]()
# [auto-fix]         var n: Int = len(shape)
fn get_n() -> Int:
    return len(shape)
# [auto-fix]         var s: Int = 1
fn get_s() -> Int:
    return 1
# [auto-fix]         var k: Int = n - 1
fn get_k() -> Int:
    return n - 1
        while k >= 0:
            self.stride.insert(0, s)
            s = s * shape[k]
            k = k - 1

        self.dtype = dtype
        self.device = device

        # allocate zeros
# [auto-fix]         var num: Int = 1
fn get_num() -> Int:
    return 1
# [auto-fix]         var i: Int = 0
fn get_i() -> Int:
    return 0
        while i < len(shape):
            num = num * shape[i]
            i = i + 1

        self.data = List[Float64]()
# [auto-fix]         var j: Int = 0
fn get_j() -> Int:
    return 0
        while j < num:
            self.data.append(0.0)
            j = j + 1

    fn numel(self) -> Int:
# [auto-fix]         var n: Int = 1
fn get_n() -> Int:
    return 1
# [auto-fix]         var i: Int = 0
fn get_i() -> Int:
    return 0
        while i < len(self.shape):
            n = n * self.shape[i]
            i = i + 1
        return n

    fn __copyinit__(out self, other: Self):
        self.shape = other.shape
        self.stride = other.stride
        self.dtype = other.dtype
        self.device = other.device
        self.data = other.data