# MIT License
# SPDX-License-Identifier: MIT
# Project:      momijo.tests
# Module:       tests.test_tensor_dtypes
# File:         examples/Tensor/demo_tensor_dtypes.mojo
#
# Description:
#   dtype casting & simple column-like access tests for momijo.tensor. 

from momijo.tensor import tensor
from collections.list import List

# -----------------------------------------------------------------------------
# utilities
# -----------------------------------------------------------------------------
fn check_shape(name: String, got: List[Int], exp: List[Int]) -> None:
    var ok = True
    if len(got) != len(exp):
        ok = False
    else:
        var i = 0
        while i < len(got):
            if got[i] != exp[i]:
                ok = False
                break
            i += 1

    if ok:
        print(name + " PASS: " + got.__str__())
    else:
        print(name + " FAIL: got=" + got.__str__() + " expected=" + exp.__str__())

# -----------------------------------------------------------------------------
# 1) basic 1D casting (Int -> Float64 / Int32)
# -----------------------------------------------------------------------------
fn test_casting_1d() -> None:
    print("\n=== test_casting_1d ===")

    # Basic int tensor (1D)
    var a = tensor.Tensor([1, 2, 3])
    var b = tensor.to_float64(a)   # safe cast to Float64
    var c = tensor.to_int32(a)     # cast/view to Int32 (per implementation)

    print("a (int): " + a.to_string())
    print("b (float64): " + b.to_string())
    print("c (int32): " + c.to_string())

    check_shape("a shape", a.shape(), [3])
    check_shape("b shape", b.shape(), [3])
    check_shape("c shape", c.shape(), [3])

    # Mixed float literals
    var mixed = tensor.Tensor([1.0, 2.5, 1.0])
    var mixed64 = tensor.to_float64(mixed)
    print("mixed (float): " + mixed.to_string())
    print("mixed64: " + mixed64.to_string())
    check_shape("mixed shape", mixed.shape(), [3])
    check_shape("mixed64 shape", mixed64.shape(), [3])

# -----------------------------------------------------------------------------
# 2) 2D casting and column-like access
# -----------------------------------------------------------------------------
fn test_casting_2d_and_cols() -> None:
    print("\n=== test_casting_2d_and_cols ===")

    # Make a 2D int tensor via arange + reshape
    var m_int = tensor.arange(0, 12).reshape([3, 4])
    var m_f64 = tensor.to_float64(m_int)
    var m_i32 = tensor.to_int32(m_int)

    print("m_int: " + m_int.to_string())
    print("m_f64: " + m_f64.to_string())
    print("m_i32: " + m_i32.to_string())

    check_shape("m_int shape", m_int.shape(), [3, 4])
    check_shape("m_f64 shape", m_f64.shape(), [3, 4])
    check_shape("m_i32 shape", m_i32.shape(), [3, 4])

    # Structured-like: two columns [code, age] as Float64
    var rows = List[List[Float64]]()
    rows.append([65.0, 23.0])
    rows.append([66.0, 31.0])
    rows.append([67.0, 28.0])

    var people = tensor.Tensor[Float64](rows=rows)
    print("people: " + people.to_string())
    check_shape("people shape", people.shape(), [3, 2])

    # Column extraction: ages = people[:, 1]
    var ages = people[:, 1]
    print("people ages (col=1): " + ages.to_string())
    check_shape("ages shape", ages.shape(), [3])

    # Example: casting the column for downstream ops
    var ages64 = tensor.to_float64(ages)
    print("ages as float64: " + ages64.to_string())
    check_shape("ages64 shape", ages64.shape(), [3])

# -----------------------------------------------------------------------------
# 3) 3D casting across numeric dtypes
# -----------------------------------------------------------------------------
fn test_casting_3d_various_dtypes() -> None:
    print("\n=== test_casting_3d_various_dtypes ===")

    var t = tensor.arange(0, 24).reshape([2, 3, 4])  # Int default
    var t_f32 = tensor.to_float32(t)
    var t_f64 = tensor.to_float64(t)
    var t_i16 = tensor.to_int16(t)
    var t_i32 = tensor.to_int32(t)
    var t_i64 = tensor.to_int64(t)

    print("t (int): " + t.to_string())
    print("t_f32: " + t_f32.to_string())
    print("t_f64: " + t_f64.to_string())
    print("t_i16: " + t_i16.to_string())
    print("t_i32: " + t_i32.to_string())
    print("t_i64: " + t_i64.to_string())

    check_shape("t shape", t.shape(), [2, 3, 4])
    check_shape("t_f32 shape", t_f32.shape(), [2, 3, 4])
    check_shape("t_f64 shape", t_f64.shape(), [2, 3, 4])
    check_shape("t_i16 shape", t_i16.shape(), [2, 3, 4])
    check_shape("t_i64 shape", t_i64.shape(), [2, 3, 4])

# -----------------------------------------------------------------------------
# 4) astype variants (if your API supports them)
# -----------------------------------------------------------------------------
fn test_astype_variants() -> None:
    print("\n=== test_astype_variants ===")
    var a = tensor.arange(0, 6)  # [6]

    # If factories or type-based overloads exist, prefer those; else use functional casts:
    var b = tensor.to_float64(a)
    var c = tensor.to_int32(a)

    print("a: " + a.to_string())
    print("b via to_float64: " + b.to_string())
    print("c via to_int32: " + c.to_string())

    check_shape("a shape", a.shape(), [6])
    check_shape("b shape", b.shape(), [6])
    check_shape("c shape", c.shape(), [6])

    # 2D variant
    var m = tensor.arange(0, 12).reshape([3, 4])
    var mf64 = tensor.to_float64(m)
    var mi32 = tensor.to_int32(m)
    check_shape("m shape", m.shape(), [3, 4])
    check_shape("mf64 shape", mf64.shape(), [3, 4])
    check_shape("mi32 shape", mi32.shape(), [3, 4])

# -----------------------------------------------------------------------------
# 5) boolean & safe truncation sanity (optional)
# -----------------------------------------------------------------------------
fn test_boolean_and_safe_truncation() -> None:
    print("\n=== test_boolean_and_safe_truncation ===")

    # Boolean tensor (if generic Tensor[T] supports Bool)
    var mask = tensor.Tensor([True, False, True, False])
    check_shape("mask shape", mask.shape(), [4])
    print("mask: " + mask.to_string())

    # Cast numeric -> Bool if provided by the API (example left commented)
    # var a = tensor.arange(0, 4)
    # var a_bool = tensor.to_bool(a)
    # print("a_bool: " + a_bool.to_string())

    # Safe truncation example (depends on cast semantics in your implementation)
    var f = tensor.Tensor([1.9, -2.1, 3.0])
    var f_to_i32 = tensor.to_int32(f)
    print("f (float): " + f.to_string())
    print("f_to_i32 (trunc/saturate per impl): " + f_to_i32.to_string())
    check_shape("f shape", f.shape(), [3])
    check_shape("f_to_i32 shape", f_to_i32.shape(), [3])

# -----------------------------------------------------------------------------
# 6) 4D large tensor quick pass
# -----------------------------------------------------------------------------
fn test_casting_4d_quick() -> None:
    print("\n=== test_casting_4d_quick ===")

    var t4 = tensor.arange(0, 120).reshape([2, 3, 4, 5])
    var t4_f64 = tensor.to_float64(t4)
    var t4_i32 = tensor.to_int32(t4)

    print("t4: " + t4.to_string())
    print("t4_f64: " + t4_f64.to_string())
    print("t4_i32: " + t4_i32.to_string())

    check_shape("t4 shape", t4.shape(), [2, 3, 4, 5])
    check_shape("t4_f64 shape", t4_f64.shape(), [2, 3, 4, 5])
    check_shape("t4_i32 shape", t4_i32.shape(), [2, 3, 4, 5])

# -----------------------------------------------------------------------------
# entrypoint
# -----------------------------------------------------------------------------
fn main() -> None:
    test_casting_1d()
    test_casting_2d_and_cols()
    test_casting_3d_various_dtypes()
    test_astype_variants()
    test_boolean_and_safe_truncation()
    test_casting_4d_quick()
