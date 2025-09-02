# MIT License
# Project: momijo  |  Source: https://github.com/taleblou/momijo
# SPDX-License-Identifier: MIT
#
# Project: momijo.tensor.ops
# File: momijo/tensor/ops/linalg.mojo

from momijo.tensor.tensor import Tensor

# ---------- Small utilities ----------
fn argmax_index(xs: List[Float64]) -> Int:
    if len(xs) == 0:
        return -1
    var best = xs[0]
    var idx = 0
    var i = 1
    while i < len(xs):
        if xs[i] > best:
            best = xs[i]
            idx = i
        i += 1
    return idx

fn argmin_index(xs: List[Float64]) -> Int:
    if len(xs) == 0:
        return -1
    var best = xs[0]
    var idx = 0
    var i = 1
    while i < len(xs):
        if xs[i] < best:
            best = xs[i]
            idx = i
        i += 1
    return idx

fn ensure_not_empty[T: Copyable & Movable](xs: List[T]) -> Bool:
    return len(xs) > 0

fn __module_name__() -> String:
    return String("momijo/tensor/ops/linalg.mojo")

# small Float32 abs
fn abs32(x: Float32) -> Float32:
    if x < Float32(0.0):
        return -x
    return x

fn __self_test__() -> Bool:
    # Basic smoke test for list matmul
    var a = [Float32(1.0), Float32(2.0), Float32(3.0), Float32(4.0)]  # 2x2
    var b = [Float32(5.0), Float32(6.0), Float32(7.0), Float32(8.0)]  # 2x2
    var c = matmul_list_f32(a, b, 2, 2, 2)
    if len(c) != 4: return False
    # Expected [[19,22],[43,50]]
    if abs32(c[0] - Float32(19.0)) > Float32(1e-6): return False
    if abs32(c[1] - Float32(22.0)) > Float32(1e-6): return False
    if abs32(c[2] - Float32(43.0)) > Float32(1e-6): return False
    if abs32(c[3] - Float32(50.0)) > Float32(1e-6): return False
    return True

# ---------- Reference GEMM on flat lists (row-major) ----------
# Shapes: (m, k) x (k, n) -> (m, n)
# Inputs are 1D row-major buffers: A[i,k] at A[i*k + t], B[t,n] at B[t*n + j]
fn matmul_list_f32(A: List[Float32], B: List[Float32], m: Int, k: Int, n: Int) -> List[Float32]:
    var C = List[Float32]()
    # pre-allocate output with zeros
    var z = 0
    while z < m * n:
        C.append(Float32(0.0))
        z += 1

    var i = 0
    while i < m:
        var j = 0
        while j < n:
            var s: Float32 = Float32(0.0)
            var t = 0
            while t < k:
                # row-major offsets
                var ia = i * k + t
                var ib = t * n + j
                s = s + A[ia] * B[ib]
                t += 1
            C[i * n + j] = s
            j += 1
        i += 1
    return C

# ---------- Tensor entry point (Float32) ----------
alias F32Tensor = Tensor[Float32]

# Temporary stub to keep module compiling; replace with real implementation once Tensor APIs are finalized.
fn matmul_f32(a: F32Tensor, b: F32Tensor) -> F32Tensor:
    return a
