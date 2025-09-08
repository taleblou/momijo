# Project:      Momijo
# Module:       src.momijo.dataframe.column
# File:         column.mojo
# Path:         src/momijo/dataframe/column.mojo
#
# Description:  src.momijo.dataframe.column — focused Momijo functionality with a stable public API.
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
#   - Structs: ColumnTag, Column
#   - Key functions: F64, I64, BOOL, STR, __init__, __copyinit__, __moveinit__, __init__ ...
#   - Static methods present.


from momijo.dataframe.bitmap import Bitmap
from momijo.dataframe.series_bool import SeriesBool
from momijo.dataframe.series_f64 import SeriesF64
from momijo.dataframe.series_i64 import SeriesI64
from momijo.dataframe.series_str import SeriesStr

struct ColumnTag:
    @staticmethod
fn F64() -> Int:  return 1
    @staticmethod
fn I64() -> Int:  return 2
    @staticmethod
fn BOOL() -> Int: return 3
    @staticmethod
fn STR() -> Int:  return 4
fn __init__(out self, ) -> None:
        pass
fn __copyinit__(out self, other: Self) -> None:
        pass
fn __moveinit__(out self, deinit other: Self) -> None:
        pass
# Explicitly copyable/movable so it can live in List[Column]
struct Column(Copyable, Movable):
    var tag: Int
    var f64: SeriesF64
    var i64: SeriesI64
    var b:   SeriesBool
    var s:   SeriesStr

    # Default constructor
fn __init__(out self) -> None:
        self.tag = ColumnTag.F64()
        self.f64 = SeriesF64()
        self.i64 = SeriesI64()
        self.b   = SeriesBool()
        self.s   = SeriesStr()

    # Copy constructor
fn __copyinit__(out self, other: Self) -> None:
        self.tag = other.tag
        self.f64 = other.f64
        self.i64 = other.i64
        self.b   = other.b
        self.s   = other.s

    # ---------- Instance builders (mut self) ----------
fn from_f64(mut self, s: SeriesF64) -> None:
        self.tag = ColumnTag.F64()
        self.f64 = s
        self.i64 = SeriesI64()
        self.b   = SeriesBool()
        self.s   = SeriesStr()
fn from_i64(mut self, s: SeriesI64) -> None:
        self.tag = ColumnTag.I64()
        self.i64 = s
        self.f64 = SeriesF64()
        self.b   = SeriesBool()
        self.s   = SeriesStr()
fn from_bool(mut self, s: SeriesBool) -> None:
        self.tag = ColumnTag.BOOL()
        self.b   = s
        self.f64 = SeriesF64()
        self.i64 = SeriesI64()
        self.s   = SeriesStr()
fn from_str(mut self, s: SeriesStr) -> None:
        self.tag = ColumnTag.STR()
        self.s   = s
        self.f64 = SeriesF64()
        self.i64 = SeriesI64()
        self.b   = SeriesBool()

    # ---------- Introspection ----------
fn dtype(self) -> Int:
        return self.tag
fn dtype_name(self) -> String:
        if self.tag == ColumnTag.F64():
            return String("f64")
        elif self.tag == ColumnTag.I64():
            return String("i64")
        elif self.tag == ColumnTag.BOOL():
            return String("bool")
        else:
            return String("str")
fn is_f64(self)  -> Bool: return self.tag == ColumnTag.F64()
fn is_i64(self)  -> Bool: return self.tag == ColumnTag.I64()
fn is_bool(self) -> Bool: return self.tag == ColumnTag.BOOL()
fn is_str(self)  -> Bool: return self.tag == ColumnTag.STR()
fn name(self) -> String:
        if self.tag == ColumnTag.F64():
            return self.f64.name
        elif self.tag == ColumnTag.I64():
            return self.i64.name
        elif self.tag == ColumnTag.BOOL():
            return self.b.name
        else:
            return self.s.name
fn rename(mut self, new_name: String) -> None:
        if self.tag == ColumnTag.F64():
            self.f64.name = new_name
        elif self.tag == ColumnTag.I64():
            self.i64.name = new_name
        elif self.tag == ColumnTag.BOOL():
            self.b.name = new_name
        else:
            self.s.name = new_name
fn len(self) -> Int:
        if self.tag == ColumnTag.F64():
            return self.f64.len()
        elif self.tag == ColumnTag.I64():
            return self.i64.len()
        elif self.tag == ColumnTag.BOOL():
            return self.b.len()
        else:
            return self.s.len()

    # ---------- Validity / Nulls ----------
fn validity(self) -> Bitmap:
        if self.tag == ColumnTag.F64():
            return self.f64.valid
        elif self.tag == ColumnTag.I64():
            return self.i64.valid
        elif self.tag == ColumnTag.BOOL():
            return self.b.valid
        else:
            return self.s.valid
fn null_count(self) -> Int:
        var n = self.len()
        var v = self.validity().count_true()
        return n - v
fn is_valid(self, i: Int) -> Bool:
        if self.tag == ColumnTag.F64():
            return self.f64.is_valid(i)
        elif self.tag == ColumnTag.I64():
            return self.i64.is_valid(i)
        elif self.tag == ColumnTag.BOOL():
            return self.b.is_valid(i)
        else:
            return self.s.is_valid(i)

    # ---------- Row slicing ----------
fn gather(self, mask: Bitmap) -> Column:
        var out = Column()
        if self.tag == ColumnTag.F64():
            out.from_f64(self.f64.gather(mask))
        elif self.tag == ColumnTag.I64():
            out.from_i64(self.i64.gather(mask))
        elif self.tag == ColumnTag.BOOL():
            out.from_bool(self.b.gather(mask))
        else:
            out.from_str(self.s.gather(mask))
        return out
fn take(self, idxs: List[Int]) -> Column:
        var out = Column()
        if self.tag == ColumnTag.F64():
            out.from_f64(self.f64.take(idxs))
        elif self.tag == ColumnTag.I64():
            out.from_i64(self.i64.take(idxs))
        elif self.tag == ColumnTag.BOOL():
            out.from_bool(self.b.take(idxs))
        else:
            out.from_str(self.s.take(idxs))
        return out

    # ---------- Value access ----------
fn value_str(self, i: Int) -> String:
        if not self.is_valid(i):
            return String("")
        if self.tag == ColumnTag.F64():
            return String(self.f64.get(i))
        elif self.tag == ColumnTag.I64():
            return String(self.i64.get(i))
        elif self.tag == ColumnTag.BOOL():
            if self.b.get(i):
                return String("true")
            else:
                return String("false")
        else:
            return self.s.get(i)
fn get_f64(self, i: Int) -> Float64:
        if not self.is_valid(i):
            return 0.0
        if self.tag == ColumnTag.F64():
            return self.f64.get(i)
        elif self.tag == ColumnTag.I64():
            return Float64(self.i64.get(i))
        else:
            return 0.0
fn get_i64(self, i: Int) -> Int64:
        if not self.is_valid(i):
            return 0
        if self.tag == ColumnTag.I64():
            return self.i64.get(i)
        elif self.tag == ColumnTag.F64():
            return Int64(self.f64.get(i))
        else:
            return 0
fn get_bool(self, i: Int) -> Bool:
        if not self.is_valid(i):
            return False
        if self.tag == ColumnTag.BOOL():
            return self.b.get(i)
        else:
            return False
fn get_str(self, i: Int) -> String:
        if not self.is_valid(i):
            return String("")
        if self.tag == ColumnTag.STR():
            return self.s.get(i)
        else:
            return String("")
fn as_f64_or_nan(self, i: Int) -> Float64:
        if not self.is_valid(i):
            return 0.0
        if self.tag == ColumnTag.F64():
            return self.f64.get(i)
        elif self.tag == ColumnTag.I64():
            return Float64(self.i64.get(i))
        else:
            return 0.0
fn as_i64_or_zero(self, i: Int) -> Int64:
        if not self.is_valid(i):
            return 0
        if self.tag == ColumnTag.I64():
            return self.i64.get(i)
        elif self.tag == ColumnTag.F64():
            return Int64(self.f64.get(i))
        else:
            return 0