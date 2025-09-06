# ============================================================================
#  Project: Momijo
#  File: vectorization.mojo
#  Description: Vectorization and SIMD helpers for kernel implementations
#  Authors: Morteza Taleblou, Mitra Daneshmand
#  License: MIT (https://opensource.org/licenses/MIT)
#  Website: https://taleblou.ir/
# ============================================================================

from math import sqrt
from stdlib.simd import SIMD

# Apply elementwise operation on two float vectors
fn simd_add(a: SIMD[Float64, 4], b: SIMD[Float64, 4]) -> SIMD[Float64, 4]:
    return a + b

fn simd_sub(a: SIMD[Float64, 4], b: SIMD[Float64, 4]) -> SIMD[Float64, 4]:
    return a - b

fn simd_mul(a: SIMD[Float64, 4], b: SIMD[Float64, 4]) -> SIMD[Float64, 4]:
    return a * b

fn simd_div(a: SIMD[Float64, 4], b: SIMD[Float64, 4]) -> SIMD[Float64, 4]:
    return a / b


# Dot product of two vectors using SIMD
fn simd_dot(a: List[Float64], b: List[Float64]) -> Float64:
    assert len(a) == len(b), "Vector size mismatch in simd_dot"
    var acc = SIMD[Float64, 4](0.0)
    var n = len(a)
    var i = 0
    while i + 4 <= n:
        var va = SIMD.load[Float64, 4](a, i)
        var vb = SIMD.load[Float64, 4](b, i)
        acc += va * vb
        i += 4
    var result: Float64 = 0.0
    for j in range(4):
        result += acc[j]
    # Handle remaining elements
    while i < n:
        result += a[i] * b[i]
        i += 1
    return result


# Square root on SIMD vector
fn simd_sqrt(v: SIMD[Float64, 4]) -> SIMD[Float64, 4]:
    var out = SIMD[Float64, 4](0.0)
    for i in range(4):
        out[i] = sqrt(v[i])
    return out


# --- Minimal self-test for smoke testing ---
fn _self_test() -> Bool:
    var ok = True
    var a = [1.0, 2.0, 3.0, 4.0]
    var b = [2.0, 2.0, 2.0, 2.0]
    var dot_val = simd_dot(a, b)
    if dot_val != 20.0:
        ok = False
    return ok

 