# MIT License 
# Copyright (c) 2025 Morteza Talebou and Mitra Daneshmand
# Project: momijo  |  Source: https://github.com/taleblou/momijo
# This file is part of the Momijo project. See the LICENSE file at the repository root.
# Momijo
# SPDX-License-Identifier: MIT
# Website: https://taleblou.ir/
# Repository: https://github.com/taleblou/momijo
#
# Project: momijo.arrow_core
# File: momijo/arrow_core/dtype_arrow.mojo

# ---------- Module meta ----------
fn __module_name__() -> String:
    return String("momijo/arrow_core/dtype_arrow.mojo")

fn __self_test__() -> Bool:
    var dt = DataType.int32()
    return dt.tag == ArrowType.INT32()

# ---------- Arrow type-id namespace (no globals) ----------
struct ArrowType:
    @staticmethod
    fn BOOL() -> Int32:     return 1
    @staticmethod
    fn INT8() -> Int32:     return 2
    @staticmethod
    fn INT16() -> Int32:    return 3
    @staticmethod
    fn INT32() -> Int32:    return 4
    @staticmethod
    fn INT64() -> Int32:    return 5
    @staticmethod
    fn UINT8() -> Int32:    return 6
    @staticmethod
    fn UINT16() -> Int32:   return 7
    @staticmethod
    fn UINT32() -> Int32:   return 8
    @staticmethod
    fn UINT64() -> Int32:   return 9
    @staticmethod
    fn FLOAT16() -> Int32:  return 10
    @staticmethod
    fn BF16() -> Int32:     return 11
    @staticmethod
    fn FLOAT32() -> Int32:  return 12
    @staticmethod
    fn FLOAT64() -> Int32:  return 13
    @staticmethod
    fn STRING() -> Int32:   return 20
    @staticmethod
    fn BINARY() -> Int32:   return 21
    @staticmethod
    fn LIST() -> Int32:     return 30
    @staticmethod
    fn STRUCT() -> Int32:   return 31

# ---------------- DataType ----------------
struct DataType(ExplicitlyCopyable, Movable):
    var tag: Int32

    fn __init__(out self, tag: Int32):
        self.tag = tag

    fn __copyinit__(out self, other: Self):
        self.tag = other.tag

    # Static constructors
    @staticmethod
    fn bool_() -> DataType:     return DataType(ArrowType.BOOL())
    @staticmethod
    fn int8() -> DataType:      return DataType(ArrowType.INT8())
    @staticmethod
    fn int16() -> DataType:     return DataType(ArrowType.INT16())
    @staticmethod
    fn int32() -> DataType:     return DataType(ArrowType.INT32())
    @staticmethod
    fn int64() -> DataType:     return DataType(ArrowType.INT64())
    @staticmethod
    fn uint8() -> DataType:     return DataType(ArrowType.UINT8())
    @staticmethod
    fn uint16() -> DataType:    return DataType(ArrowType.UINT16())
    @staticmethod
    fn uint32() -> DataType:    return DataType(ArrowType.UINT32())
    @staticmethod
    fn uint64() -> DataType:    return DataType(ArrowType.UINT64())
    @staticmethod
    fn float16() -> DataType:   return DataType(ArrowType.FLOAT16())
    @staticmethod
    fn bfloat16() -> DataType:  return DataType(ArrowType.BF16())
    @staticmethod
    fn float32() -> DataType:   return DataType(ArrowType.FLOAT32())
    @staticmethod
    fn float64() -> DataType:   return DataType(ArrowType.FLOAT64())
    @staticmethod
    fn string() -> DataType:    return DataType(ArrowType.STRING())
    @staticmethod
    fn binary() -> DataType:    return DataType(ArrowType.BINARY())
    @staticmethod
    fn list_() -> DataType:     return DataType(ArrowType.LIST())
    @staticmethod
    fn struct_() -> DataType:   return DataType(ArrowType.STRUCT())

# ---------------------------
# Helper functions
# ---------------------------
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

fn ensure_not_empty[T: ExplicitlyCopyable & Movable](xs: List[T]) -> Bool:
    return len(xs) > 0

# -------------------------------------------
# Module-level wrappers mirroring ArrowType.*
# -------------------------------------------
fn BOOL() -> Int32:       return ArrowType.BOOL()
fn INT8() -> Int32:       return ArrowType.INT8()
fn INT16() -> Int32:      return ArrowType.INT16()
fn INT32() -> Int32:      return ArrowType.INT32()
fn INT64() -> Int32:      return ArrowType.INT64()
fn UINT8() -> Int32:      return ArrowType.UINT8()
fn UINT16() -> Int32:     return ArrowType.UINT16()
fn UINT32() -> Int32:     return ArrowType.UINT32()
fn UINT64() -> Int32:     return ArrowType.UINT64()
fn FLOAT16() -> Int32:    return ArrowType.FLOAT16()
fn BF16() -> Int32:       return ArrowType.BF16()
fn FLOAT32() -> Int32:    return ArrowType.FLOAT32()
fn FLOAT64() -> Int32:    return ArrowType.FLOAT64()
fn STRING() -> Int32:     return ArrowType.STRING()
fn BINARY() -> Int32:     return ArrowType.BINARY()
fn LIST() -> Int32:       return ArrowType.LIST()
fn STRUCT() -> Int32:     return ArrowType.STRUCT()
