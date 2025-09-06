# ============================================================================
#  Project: Momijo
#  File: simd_elemwise.mojo
#  Description: SIMD-accelerated elementwise operations for CPU kernels
#  Authors: Morteza Taleblou, Mitra Daneshmand
#  License: MIT (https://opensource.org/licenses/MIT)
#  Website: https://taleblou.ir/
# ============================================================================

from math import exp
from stdlib.simd import SIMD

# Elementwise add with SIMD (Float64 vectors)
fn simd_add(xs: List[Float64], ys: List[Float64]) -> List[Float64]:
    assert len(xs) == len(ys), "Vector size mismatch in simd_add"
    var n = len(xs)
    var out = List[Float64]()
    out.reserve(n)
    var i = 0
    while i + 4 <= n:
        var vx = SIMD.load[Float64, 4](xs, i)
        var vy = SIMD.load[Float64, 4](ys, i)
        var vz = vx + vy
        for j in range(4):
            out.append(vz[j])
        i += 4
    while i < n:
        out.append(xs[i] + ys[i])
        i += 1
    return out


# Elementwise multiply with SIMD
fn simd_mul(xs: List[Float64], ys: List[Float64]) -> List[Float64]:
    assert len(xs) == len(ys), "Vector size mismatch in simd_mul"
    var n = len(xs)
    var out = List[Float64]()
    out.reserve(n)
    var i = 0
    while i + 4 <= n:
        var vx = SIMD.load[Float64, 4](xs, i)
        var vy = SIMD.load[Float64, 4](ys, i)
        var vz = vx * vy
        for j in range(4):
            out.append(vz[j])
        i += 4
    while i < n:
        out.append(xs[i] * ys[i])
        i += 1
    return out


# Elementwise sigmoid with SIMD
fn simd_sigmoid(xs: List[Float64]) -> List[Float64]:
    var n = len(xs)
    var out = List[Float64]()
    out.reserve(n)
    var i = 0
    while i + 4 <= n:
        var vx = SIMD.load[Float64, 4](xs, i)
        var vz = SIMD[Float64, 4](0.0)
        for j in range(4):
            vz[j] = 1.0 / (1.0 + exp(-vx[j]))
        for j in range(4):
            out.append(vz[j])
        i += 4
    while i < n:
        out.append(1.0 / (1.0 + exp(-xs[i])))
        i += 1
    return out


# --- Minimal self-test for smoke testing ---
fn _self_test() -> Bool:
    var ok = True
    var xs = [1.0, 2.0, 3.0, 4.0]
    var ys = [5.0, 6.0, 7.0, 8.0]
    var zs = simd_add(xs, ys)
    if zs[0] != 6.0 or zs[3] != 12.0:
        ok = False
    var ms = simd_mul(xs, ys)
    if ms[0] != 5.0 or ms[3] != 32.0:
        ok = False
    var sg = simd_sigmoid([0.0])
    if abs(sg[0] - 0.5) > 1e-6:
        ok = False
    return ok

 
