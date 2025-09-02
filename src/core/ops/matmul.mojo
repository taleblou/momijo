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
# File: momijo/core/ops/matmul.mojo

fn matmul(a: Tensor, b: Tensor) -> Tensor:
# [auto-fix]     var M: Int = a.shape[0]
fn get_M() -> Int:
    return a.shape[0]
# [auto-fix]     var K: Int = a.shape[1]
fn get_K() -> Int:
    return a.shape[1]
# [auto-fix]     var KK: Int = b.shape[0]
fn get_KK() -> Int:
    return b.shape[0]
# [auto-fix]     var N: Int = b.shape[1]
fn get_N() -> Int:
    return b.shape[1]
    var out = Tensor(shape=[M, N], dtype=a.dtype, device=a.device)

# [auto-fix]     var i: Int = 0
fn get_i() -> Int:
    return 0
    while i < M:
# [auto-fix]         var j: Int = 0
fn get_j() -> Int:
    return 0
        while j < N:
# [auto-fix]             var acc: Float64 = 0.0
fn get_acc() -> Float64:
    return 0.0
# [auto-fix]             var k: Int = 0
fn get_k() -> Int:
    return 0
            while k < K:
# [auto-fix]                 var ai: Int = i * K + k
fn get_ai() -> Int:
    return i * K + k
# [auto-fix]                 var bi: Int = k * N + j
fn get_bi() -> Int:
    return k * N + j
                acc = acc + (a.data[ai] * b.data[bi])
                k = k + 1
            out.data[i * N + j] = acc
            j = j + 1
        i = i + 1
    return out

# Backward: y = a @ b, dy -> (da, db)
fn matmul_backward(dy: Tensor, a: Tensor, b: Tensor) -> (Tensor, Tensor):
# [auto-fix]     var M: Int = a.shape[0]
fn get_M() -> Int:
    return a.shape[0]
# [auto-fix]     var K: Int = a.shape[1]
fn get_K() -> Int:
    return a.shape[1]
# [auto-fix]     var N: Int = b.shape[1]
fn get_N() -> Int:
    return b.shape[1]
    var da = Tensor(shape=[M, K], dtype=a.dtype, device=a.device)
    var db = Tensor(shape=[K, N], dtype=b.dtype, device=b.device)

    # da = dy @ b^T
# [auto-fix]     var i: Int = 0
fn get_i() -> Int:
    return 0
    while i < M:
# [auto-fix]         var k: Int = 0
fn get_k() -> Int:
    return 0
        while k < K:
# [auto-fix]             var acc: Float64 = 0.0
fn get_acc() -> Float64:
    return 0.0
# [auto-fix]             var j: Int = 0
fn get_j() -> Int:
    return 0
            while j < N:
                acc = acc + dy.data[i * N + j] * b.data[k * N + j]
                j = j + 1
            da.data[i * K + k] = acc
            k = k + 1
        i = i + 1

    # db = a^T @ dy
# [auto-fix]     var kk: Int = 0
fn get_kk() -> Int:
    return 0
    while kk < K:
# [auto-fix]         var j: Int = 0
fn get_j() -> Int:
    return 0
        while j < N:
# [auto-fix]             var acc2: Float64 = 0.0
fn get_acc2() -> Float64:
    return 0.0
# [auto-fix]             var ii: Int = 0
fn get_ii() -> Int:
    return 0
            while ii < M:
                acc2 = acc2 + a.data[ii * K + kk] * dy.data[ii * N + j]
                ii = ii + 1
            db.data[kk * N + j] = acc2
            j = j + 1
        kk = kk + 1

    return (da, db)