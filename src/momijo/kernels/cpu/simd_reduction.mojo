# Project:      Momijo
# Module:       src.momijo.kernels.cpu.simd_reduction
# File:         simd_reduction.mojo
# Path:         src/momijo/kernels/cpu/simd_reduction.mojo
#
# Description:  src.momijo.kernels.cpu.simd_reduction — focused Momijo functionality with a stable public API.
#               Composable building blocks intended for reuse.
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
#   - Key functions: simd_sum, simd_max, simd_min, simd_dot, _self_test
#   - Uses generic functions/types with explicit trait bounds.


from stdlib.simd import SIMD

fn simd_sum(xs: List[Float64]) -> Float64:
    var n = len(xs)
    var acc = SIMD[Float64, 4](0.0)
    var i = 0
    while i + 4 <= n:
        var vx = SIMD.load[Float64, 4](xs, i)
        acc += vx
        i += 4
    var result: Float64 = 0.0
    for j in range(4):
        result += acc[j]
    while i < n:
        result += xs[i]
        i += 1
    return result

# SIMD max reduction
fn simd_max(xs: List[Float64]) -> Float64:
    assert len(xs) > 0, "Empty list in simd_max"
    var n = len(xs)
    var vmax = SIMD.load[Float64, 4](xs, 0)
    var i = 4
    while i + 4 <= n:
        var vx = SIMD.load[Float64, 4](xs, i)
        for j in range(4):
            if vx[j] > vmax[j]:
                vmax[j] = vx[j]
        i += 4
    var result = vmax[0]
    for j in range(1, 4):
        if vmax[j] > result:
            result = vmax[j]
    while i < n:
        if xs[i] > result:
            result = xs[i]
        i += 1
    return result

# SIMD min reduction
fn simd_min(xs: List[Float64]) -> Float64:
    assert len(xs) > 0, "Empty list in simd_min"
    var n = len(xs)
    var vmin = SIMD.load[Float64, 4](xs, 0)
    var i = 4
    while i + 4 <= n:
        var vx = SIMD.load[Float64, 4](xs, i)
        for j in range(4):
            if vx[j] < vmin[j]:
                vmin[j] = vx[j]
        i += 4
    var result = vmin[0]
    for j in range(1, 4):
        if vmin[j] < result:
            result = vmin[j]
    while i < n:
        if xs[i] < result:
            result = xs[i]
        i += 1
    return result

# SIMD dot product
fn simd_dot(xs: List[Float64], ys: List[Float64]) -> Float64:
    assert len(xs) == len(ys), "Vector size mismatch in simd_dot"
    var n = len(xs)
    var acc = SIMD[Float64, 4](0.0)
    var i = 0
    while i + 4 <= n:
        var vx = SIMD.load[Float64, 4](xs, i)
        var vy = SIMD.load[Float64, 4](ys, i)
        acc += vx * vy
        i += 4
    var result: Float64 = 0.0
    for j in range(4):
        result += acc[j]
    while i < n:
        result += xs[i] * ys[i]
        i += 1
    return result

# --- Minimal self-test for smoke testing ---
fn _self_test() -> Bool:
    var ok = True
    var xs = [1.0, 2.0, 3.0, 4.0]
    if simd_sum(xs) != 10.0:
        ok = False
    if simd_max(xs) != 4.0:
        ok = False
    if simd_min(xs) != 1.0:
        ok = False
    var ys = [2.0, 2.0, 2.0, 2.0]
    if simd_dot(xs, ys) != 20.0:
        ok = False
    return ok