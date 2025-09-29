# Project:      Momijo
# Module:       dataframe.column
# File:         column.mojo
# Path:         dataframe/column.mojo
#
# Description:  dataframe.column — Column module for Momijo DataFrame.
#               Implements core data structures, algorithms, and convenience APIs for production use.
#               Designed as a stable, composable building block within the Momijo public API.
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
#   - Key functions: F64, I64, BOOL, STR, __moveinit__, __init__, dtype, dtype_name, is_f64, is_i64, is_bool, is_str, name, rename, len, validity, null_count, is_valid
#   - Static methods present.

from momijo.dataframe.bitmap import Bitmap
from momijo.dataframe.series_bool import SeriesBool
from momijo.dataframe.series_f64 import SeriesF64
from momijo.dataframe.series_i64 import SeriesI64
from momijo.dataframe.series_str import SeriesStr

from momijo.dataframe.compat import as_f64_or_nan as get_f64
from momijo.dataframe.compat import as_i64_or_zero as get_i64

# ---------- ColumnTag ----------
struct ColumnTag:
    @staticmethod
    fn F64() -> Int: return 1
    @staticmethod
    fn I64() -> Int: return 2
    @staticmethod
    fn BOOL() -> Int: return 3
    @staticmethod
    fn STR() -> Int: return 4

    fn __moveinit__(out self, deinit other: Self):
        pass


# ---------- Column ----------
struct Column(Copyable, Movable):
    var tag: Int
    var f64: SeriesF64
    var i64: SeriesI64
    var b: SeriesBool
    var s: SeriesStr

    fn __init__(out self):
        self.tag = ColumnTag.F64()
        self.f64 = SeriesF64()
        self.i64 = SeriesI64()
        self.b = SeriesBool()
        self.s = SeriesStr()

    # dtype info
    fn dtype(self) -> Int:
        return self.tag

    fn dtype_name(self) -> String:
        if self.tag == ColumnTag.F64(): return String("f64")
        elif self.tag == ColumnTag.I64(): return String("i64")
        elif self.tag == ColumnTag.BOOL(): return String("bool")
        else: return String("str")

    # length
    fn len(self) -> Int:
        if self.tag == ColumnTag.F64(): return self.f64.len()
        elif self.tag == ColumnTag.I64(): return self.i64.len()
        elif self.tag == ColumnTag.BOOL(): return self.b.len()
        else: return self.s.len()

    # validity
    fn is_valid(self, i: Int) -> Bool:
        if self.tag == ColumnTag.F64(): return self.f64.valid.is_set(i)
        elif self.tag == ColumnTag.I64(): return self.i64.valid.is_set(i)
        elif self.tag == ColumnTag.BOOL(): return self.b.valid.is_set(i)
        else: return self.s.valid.is_set(i)

    fn null_count(self) -> Int:
        return self.len() - self.validity().count_true()

    fn validity(self) -> Bitmap:
        if self.tag == ColumnTag.F64(): return self.f64.valid
        elif self.tag == ColumnTag.I64(): return self.i64.valid
        elif self.tag == ColumnTag.BOOL(): return self.b.valid
        else: return self.s.valid

    # value access
    fn value_str(self, i: Int) -> String:
        if not self.is_valid(i): return String("")
        if self.tag == ColumnTag.F64(): return String(self.f64.get(i))
        elif self.tag == ColumnTag.I64(): return String(self.i64.get(i))
        elif self.tag == ColumnTag.BOOL(): return String("true") if self.b.get(i) else String("false")
        else: return self.s.get(i)

    fn get_bool(self, i: Int) -> Bool:
        if not self.is_valid(i): return False
        if self.tag == ColumnTag.BOOL(): return self.b.get(i)
        return False

    fn get_str(self, i: Int) -> String:
        if not self.is_valid(i): return String("")
        if self.tag == ColumnTag.STR(): return self.s.get(i)
        return String("")

    fn get_string(self, i: Int) -> String:
        return self.get_str(i)

    # construction from series
    fn from_str(mut self, s: SeriesStr):
        self.tag = ColumnTag.STR()
        self.s = s

    fn from_f64(mut self, s: SeriesF64):
        self.tag = ColumnTag.F64()
        self.f64 = s

    fn from_i64(mut self, s: SeriesI64):
        self.tag = ColumnTag.I64()
        self.i64 = s

    fn from_bool(mut self, s: SeriesBool):
        self.tag = ColumnTag.BOOL()
        self.b = s

    # convert to list
    fn to_list(self) -> List[String]:
        var out = List[String]()
        var i = 0
        while i < self.len():
            out.append(self.get_string(i))
            i += 1
        return out

    # take/gather operations
    fn take(self, idxs: List[Int]) -> Column:
        var out = Column()
        if self.tag == ColumnTag.F64(): out.from_f64(self.f64.take(idxs))
        elif self.tag == ColumnTag.I64(): out.from_i64(self.i64.take(idxs))
        elif self.tag == ColumnTag.BOOL(): out.from_bool(self.b.take(idxs))
        else: out.from_str(self.s.take(idxs))
        return out

    fn gather(self, mask: Bitmap) -> Column:
        var out = Column()
        if self.tag == ColumnTag.F64(): out.from_f64(self.f64.gather(mask))
        elif self.tag == ColumnTag.I64(): out.from_i64(self.i64.gather(mask))
        elif self.tag == ColumnTag.BOOL(): out.from_bool(self.b.gather(mask))
        else: out.from_str(self.s.gather(mask))
        return out

    # rename column
    fn rename(mut self, new_name: String):
        if self.tag == ColumnTag.F64(): self.f64.name = new_name
        elif self.tag == ColumnTag.I64(): self.i64.name = new_name
        elif self.tag == ColumnTag.BOOL(): self.b.name = new_name
        else: self.s.name = new_name

    # item access
    fn __getitem__(self, i: Int) -> String:
        return self.get_str(i)

    # check type helpers
    fn is_f64(self) -> Bool:
        return self.tag == ColumnTag.F64()

    fn is_i64(self) -> Bool:
        return self.tag == ColumnTag.I64()

    fn is_bool(self) -> Bool:
        return self.tag == ColumnTag.BOOL()

    fn is_str(self) -> Bool:
        return self.tag == ColumnTag.STR()
    
    
        # clone method for Column
    fn clone(self) -> Column:
        var out = Column()
        out.tag = self.tag
        if self.tag == ColumnTag.F64():
            out.f64 = self.f64.clone()
        elif self.tag == ColumnTag.I64():
            out.i64 = self.i64.clone()
        elif self.tag == ColumnTag.BOOL():
            out.b = self.b.clone()
        else:
            out.s = self.s.clone()
        return out

    # get_name method for Column
    fn get_name(self) -> String:
        if self.tag == ColumnTag.F64():
            return self.f64.name
        elif self.tag == ColumnTag.I64():
            return self.i64.name
        elif self.tag == ColumnTag.BOOL():
            return self.b.name
        else:
            return self.s.name


fn from_list_int(list: List[Int], name: String) -> Column:
    var series = SeriesI64()
    series.name = name
    series.data = List[Int]()
    var i = 0
    var n = len(list)
    while i < n:
        series.data.append(list[i])
        i += 1
    var col = Column()
    col.from_i64(series)
    return col


# ---- Added helpers (no new imports, non-intrusive) ----

# Decode column values to List[String] (string view)
fn astype(col: Column, target: String) -> List[String]:
    var n = col.len()
    var out = List[String]()
    var i = 0
    while i < n:
        out.append(col.get_str(i))
        i += 1
    return out



# Reorder categories according to new_cats; values not present become -1 in codes
fn reorder(col: Column, new_cats: List[String]) -> (List[String], List[Int]):
    var rev = Dict[String, Int]()
    var i = 0
    while i < len(new_cats):
        rev[new_cats[i]] = i
        i += 1

    var codes = List[Int]()
    var n = col.len()
    var j = 0
    while j < n:
        var v = col.get_str(j)
        if v in rev:
            codes.append(rev[v])
        else:
            codes.append(-1)
        j += 1
    return (new_cats, codes)

# Remove unused categories by scanning actual values
fn remove_unused(col: Column) -> (List[String], List[Int]):
    var cats = List[String]()
    var map_old_new = Dict[String, Int]()
    var codes = List[Int]()

    var n = col.len()
    var i = 0
    while i < n:
        var v = col.get_str(i)
        if v in map_old_new:
            codes.append(map_old_new[v])
        else:
            var nid = len(cats)
            map_old_new[v] = nid
            cats.append(v)
            codes.append(nid)
        i += 1

    return (cats, codes)

