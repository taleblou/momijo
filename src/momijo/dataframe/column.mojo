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
from momijo.dataframe.api import *

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
# A tagged column holding exactly one active Series{F64,I64,Bool,Str}.
# Deep-copy semantics; no recursive clone/copy loops.

struct Column(Copyable, Movable):
    var tag: Int
    var f64: SeriesF64
    var i64: SeriesI64
    var b:   SeriesBool
    var s:   SeriesStr

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------
    fn __init__(out self):
        # Default to empty string column (safer for printing). Change to F64 if you prefer.
        self.tag = ColumnTag.STR()
        self.f64 = SeriesF64()
        self.i64 = SeriesI64()
        self.b   = SeriesBool()
        self.s   = SeriesStr()

    fn __copyinit__(out self, other: Self):
        self.tag = other.tag
        self.s   = SeriesStr()
        self.f64 = SeriesF64()
        self.i64 = SeriesI64()
        self.b   = SeriesBool()

        if other.tag == ColumnTag.STR():
            self.s.set_name(other.s.get_name())
            self.s.data  = other.s.values().copy()
            self.s.valid = other.s.validity().copy()
            var n = len(self.s.data)
            if self.s.valid.len() != n:
                self.s.valid.resize(n, True)
        elif other.tag == ColumnTag.I64():
            self.i64.set_name(other.i64.get_name())
            self.i64.data  = other.i64.values().copy()
            self.i64.valid = other.i64.validity().copy()
            var n = len(self.i64.data)
            if self.i64.valid.len() != n:
                self.i64.valid.resize(n, True)
        elif other.tag == ColumnTag.F64():
            self.f64.set_name(other.f64.get_name())
            self.f64.data  = other.f64.values().copy()
            self.f64.valid = other.f64.validity().copy()
            var n = len(self.f64.data)
            if self.f64.valid.len() != n:
                self.f64.valid.resize(n, True)
        elif other.tag == ColumnTag.BOOL():
            self.b.set_name(other.b.get_name())
            self.b.data  = other.b.values().copy()
            self.b.valid = other.b.validity().copy()
            var n = len(self.b.data)
            if self.b.valid.len() != n:
                self.b.valid.resize(n, True)

 
    fn clone(self) -> Self:
        var out = Column()
        out.tag = self.tag
        if self.tag == ColumnTag.F64():
            out.f64 = self.f64.clone()
            out.i64 = SeriesI64()
            out.b   = SeriesBool()
            out.s   = SeriesStr()
        elif self.tag == ColumnTag.I64():
            out.f64 = SeriesF64()
            out.i64 = self.i64.clone()
            out.b   = SeriesBool()
            out.s   = SeriesStr()
        elif self.tag == ColumnTag.BOOL():
            out.f64 = SeriesF64()
            out.i64 = SeriesI64()
            out.b   = self.b.clone()
            out.s   = SeriesStr()
        else:
            out.f64 = SeriesF64()
            out.i64 = SeriesI64()
            out.b   = SeriesBool()
            out.s   = self.s.clone()
        return out


    # ------------------------------------------------------------------
    # DType info
    # ------------------------------------------------------------------
    fn dtype(self) -> Int:
        return self.tag

    fn dtype_name(self) -> String:
        if self.tag == ColumnTag.F64():   return String("f64")
        elif self.tag == ColumnTag.I64(): return String("i64")
        elif self.tag == ColumnTag.BOOL():return String("bool")
        else:                              return String("str")

    # ------------------------------------------------------------------
    # Length / validity
    # ------------------------------------------------------------------
    fn len(self) -> Int:
        if self.tag == ColumnTag.F64():   return self.f64.len()
        elif self.tag == ColumnTag.I64(): return self.i64.len()
        elif self.tag == ColumnTag.BOOL():return self.b.len()
        else:                              return self.s.len()

    fn is_valid(self, i: Int) -> Bool:
        if i < 0 or i >= self.len():
            return False
        if self.tag == ColumnTag.F64():    return self.f64.valid.get(i)
        elif self.tag == ColumnTag.I64():  return self.i64.valid.get(i)
        elif self.tag == ColumnTag.BOOL(): return self.b.valid.get(i)
        else:                               return self.s.valid.get(i)

    fn validity(self) -> Bitmap:
        if self.tag == ColumnTag.F64():    return self.f64.valid.copy()
        elif self.tag == ColumnTag.I64():  return self.i64.valid.copy()
        elif self.tag == ColumnTag.BOOL(): return self.b.valid.copy()
        else:                               return self.s.valid.copy()

    fn null_count(self) -> Int:
        return self.len() - self.validity().count_true()

    # ------------------------------------------------------------------
    # Value access
    # ------------------------------------------------------------------
    fn value_str(self, i: Int) -> String:
        if not self.is_valid(i): return String("")
        if self.tag == ColumnTag.F64():    return String(self.f64.get(i))
        elif self.tag == ColumnTag.I64():  return String(self.i64.get(i))
        elif self.tag == ColumnTag.BOOL(): return String("true") if self.b.get(i) else String("false")
        else:                               return self.s.get(i)

    fn get_bool(self, i: Int) -> Bool:
        if not self.is_valid(i): return False
        if self.tag == ColumnTag.BOOL(): return self.b.get(i)
        return False

    fn get_str(self, i: Int) -> String:
        if not self.is_valid(i): return String("")
        if self.tag == ColumnTag.STR(): return self.s.get(i)
        # Non-str columns: return textual representation
        return self.value_str(i)

    fn get_string(self, i: Int) -> String:
        return self.get_str(i)

    # ------------------------------------------------------------------
    # Builders from series (deep copy)
    # ------------------------------------------------------------------  
    fn from_str(mut self, src: SeriesStr) -> None:
        self.tag = ColumnTag.STR()
        self.s.set_name(src.get_name())
        self.s.data  = src.values().copy()
        self.s.valid = src.validity().copy()
        var n = len(self.s.data)
        if self.s.valid.len() != n:
            self.s.valid.resize(n, True)
        self.f64 = SeriesF64(); self.i64 = SeriesI64(); self.b = SeriesBool()

    fn from_i64(mut self, src: SeriesI64) -> None:
        self.tag = ColumnTag.I64()
        self.i64.set_name(src.get_name())
        self.i64.data  = src.values().copy()
        self.i64.valid = src.validity().copy()
        var n = len(self.i64.data)
        if self.i64.valid.len() != n:
            self.i64.valid.resize(n, True)
        self.f64 = SeriesF64(); self.b = SeriesBool(); self.s = SeriesStr()

    fn from_f64(mut self, src: SeriesF64) -> None:
        self.tag = ColumnTag.F64()
        self.f64.set_name(src.get_name())
        self.f64.data  = src.values().copy()
        self.f64.valid = src.validity().copy()
        var n = len(self.f64.data)
        if self.f64.valid.len() != n:
            self.f64.valid.resize(n, True)
        self.i64 = SeriesI64(); self.b = SeriesBool(); self.s = SeriesStr()

    fn from_bool(mut self, src: SeriesBool) -> None:
        self.tag = ColumnTag.BOOL()
        self.b.set_name(src.get_name())
        self.b.data  = src.values().copy()
        self.b.valid = src.validity().copy()
        var n = len(self.b.data)
        if self.b.valid.len() != n:
            self.b.valid.resize(n, True)
        self.f64 = SeriesF64(); self.i64 = SeriesI64(); self.s = SeriesStr()




    # ------------------------------------------------------------------
    # Conversions
    # ------------------------------------------------------------------
    fn to_list(self) -> List[String]:
        var out = List[String]()
        var i = 0
        var n = self.len()
        while i < n:
            out.append(self.value_str(i))
            i += 1
        return out

    # ------------------------------------------------------------------
    # Indexing / name ops
    # ------------------------------------------------------------------
    fn __getitem__(self, i: Int) -> String:
        return self.get_str(i)

    fn rename(mut self, new_name: String) -> None:
        if self.tag == ColumnTag.F64():    self.f64.name = String(new_name)
        elif self.tag == ColumnTag.I64():  self.i64.name = String(new_name)
        elif self.tag == ColumnTag.BOOL(): self.b.name   = String(new_name)
        else:                               self.s.name   = String(new_name)

    fn get_name(self) -> String:
        if self.tag == ColumnTag.F64():    return self.f64.name
        elif self.tag == ColumnTag.I64():  return self.i64.name
        elif self.tag == ColumnTag.BOOL(): return self.b.name
        else:                               return self.s.name

    # ------------------------------------------------------------------
    # Type predicates
    # ------------------------------------------------------------------
    fn is_f64(self)  -> Bool: return self.tag == ColumnTag.F64()
    fn is_i64(self)  -> Bool: return self.tag == ColumnTag.I64()
    fn is_bool(self) -> Bool: return self.tag == ColumnTag.BOOL()
    fn is_str(self)  -> Bool: return self.tag == ColumnTag.STR()

    # ------------------------------------------------------------------
    # Take / gather (shape-preserving selectors)
    # ------------------------------------------------------------------
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

    # ---------------- Column setters ---------------- 

    # String
    fn set_string_series(mut self, src: SeriesStr) -> None:
        # tag first
        self.tag = ColumnTag.STR()

        # rebuild in place
        self.s.set_name(src.get_name())

        # deep-copy buffers (no aliasing)
        self.s.data  = src.data.copy()
        self.s.valid = src.valid.copy()

        # reset other variants
        self.f64 = SeriesF64()
        self.i64 = SeriesI64()
        self.b   = SeriesBool()

        # sync lengths
        var n = len(self.s.data)
        if self.s.valid.len() != n:
            self.s.valid.resize(n, True)


    # Bool
    fn set_bool_series(mut self, src: SeriesBool) -> None:
        self.tag = ColumnTag.BOOL()

        self.b.set_name(src.get_name())
        self.b.data  = src.data.copy()
        self.b.valid = src.valid.copy()

        self.s   = SeriesStr()
        self.f64 = SeriesF64()
        self.i64 = SeriesI64()

        var n = len(self.b.data)
        if self.b.valid.len() != n:
            self.b.valid.resize(n, True)


    # Float64
    fn set_f64_series(mut self, src: SeriesF64) -> None:
        self.tag = ColumnTag.F64()

        self.f64.set_name(src.get_name())

        # deep-copy data
        self.f64.data.clear()
        self.f64.data.reserve(src.len())
        var i = 0
        while i < src.len():
            self.f64.data.append(src.data[i])
            i += 1

        # deep-copy validity
        self.f64.valid = src.valid.copy()

        # reset others
        self.s   = SeriesStr()
        self.i64 = SeriesI64()
        self.b   = SeriesBool()

        # sync lengths
        var n = len(self.f64.data)
        if self.f64.valid.len() != n:
            self.f64.valid.resize(n, True)


    # Int64
    fn set_i64_series(mut self, src: SeriesI64) -> None:
        self.tag = ColumnTag.I64()

        self.i64.set_name(src.get_name())
        self.i64.data  = src.data.copy()
        self.i64.valid = src.valid.copy()

        self.s   = SeriesStr()
        self.f64 = SeriesF64()
        self.b   = SeriesBool()

        var n = len(self.i64.data)
        if self.i64.valid.len() != n:
            self.i64.valid.resize(n, True)


 
    # Return a boolean series where True marks null/None cells (validity == False)
    fn null_mask(self) -> SeriesBool:
        var out = SeriesBool()
        # Derive a name
        var cname = String("")
        if self.tag == ColumnTag.STR():
            cname = self.s.get_name()
        elif self.tag == ColumnTag.BOOL():
            cname = self.b.get_name()
        elif self.tag == ColumnTag.F64():
            cname = self.f64.get_name()
        elif self.tag == ColumnTag.I64():
            cname = self.i64.get_name()
        out.set_name(String("is_none(") + cname + String(")"))

        out.data  = List[Bool]()
        out.valid = Bitmap(0, True)

        if self.tag == ColumnTag.STR():
            var n = len(self.s.data)
            var i = 0
            while i < n:
                out.data.append(not self.s.valid.get(i))
                i += 1
            out.valid.resize(len(out.data), True)
            return out

        if self.tag == ColumnTag.BOOL():
            var n = len(self.b.data)
            var i = 0
            while i < n:
                out.data.append(not self.b.valid.get(i))
                i += 1
            out.valid.resize(len(out.data), True)
            return out

        if self.tag == ColumnTag.F64():
            var n = len(self.f64.data)
            var i = 0
            while i < n:
                out.data.append(not self.f64.valid.get(i))
                i += 1
            out.valid.resize(len(out.data), True)
            return out

        if self.tag == ColumnTag.I64():
            var n = len(self.i64.data)
            var i = 0
            while i < n:
                out.data.append(not self.i64.valid.get(i))
                i += 1
            out.valid.resize(len(out.data), True)
            return out

        return out

    # True if at least one row is null/None
    fn has_none(self) -> Bool:
        if self.tag == ColumnTag.STR():
            var n = len(self.s.data)
            var i = 0
            while i < n:
                if not self.s.valid.get(i):
                    return True
                i += 1
            return False

        if self.tag == ColumnTag.BOOL():
            var n = len(self.b.data)
            var i = 0
            while i < n:
                if not self.b.valid.get(i):
                    return True
                i += 1
            return False

        if self.tag == ColumnTag.F64():
            var n = len(self.f64.data)
            var i = 0
            while i < n:
                if not self.f64.valid.get(i):
                    return True
                i += 1
            return False

        if self.tag == ColumnTag.I64():
            var n = len(self.i64.data)
            var i = 0
            while i < n:
                if not self.i64.valid.get(i):
                    return True
                i += 1
            return False

        return False

    # Alias so you can write: if col.is_none(): ...
    fn is_none(self) -> Bool:
        return self.has_none()

    # Optional: True if all rows are null/None (empty → False)
    fn all_none(self) -> Bool:
        if self.tag == ColumnTag.STR():
            var n = len(self.s.data)
            if n == 0: return False
            var i = 0
            while i < n:
                if self.s.valid.get(i):
                    return False
                i += 1
            return True

        if self.tag == ColumnTag.BOOL():
            var n = len(self.b.data)
            if n == 0: return False
            var i = 0
            while i < n:
                if self.b.valid.get(i):
                    return False
                i += 1
            return True

        if self.tag == ColumnTag.F64():
            var n = len(self.f64.data)
            if n == 0: return False
            var i = 0
            while i < n:
                if self.f64.valid.get(i):
                    return False
                i += 1
            return True

        if self.tag == ColumnTag.I64():
            var n = len(self.i64.data)
            if n == 0: return False
            var i = 0
            while i < n:
                if self.i64.valid.get(i):
                    return False
                i += 1
            return True

        return False

    fn value(self) -> Column:
            var out = Column()
            out.__copyinit__(self)   # uses your custom deep-copy logic for all variants
            return out.copy()



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
    return col.copy()


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


 
# ---------------- Value facade (string-backed) ----------------
struct Value(Copyable, Movable):
    var tag: Int       # 1=BOOL, 2=INT32, 3=INT64, 5=FLOAT64
    var i:   Int
    var f:   Float64
    var b:   Bool

    fn __init__(out self):
        self.tag = 0
        self.i = 0
        self.f = 0.0
        self.b = False

    fn as_string(self) -> String:
        if self.tag == 1:
            return String(self.b)
        elif self.tag == 2 or self.tag == 3:
            return String(self.i)
        elif self.tag == 5:
            return String(self.f)
        else:
            return String("")

    @staticmethod
    fn int32(x: Int) -> Value:
        var v = Value()
        v.tag = 2
        v.i = x
        return v.copy()

    @staticmethod
    fn int64(x: Int) -> Value:
        var v = Value()
        v.tag = 3
        v.i = x
        return v.copy()

    @staticmethod
    fn float64(x: Float64) -> Value:
        var v = Value()
        v.tag = 5
        v.f = x
        return v.copy()

    @staticmethod
    fn bool(x: Bool) -> Value:
        var v = Value()
        v.tag = 1
        v.b = x
        return v.copy()

 

# ---------------- CellAccessor ----------------
struct CellAccessor:
    var row_i: Int
    var col_j: Int
    fn __init__(out self, row_i: Int, col_j: Int):
        self.row_i = row_i
        self.col_j = col_j

fn is_valid_cell(df: DataFrame, r: Int, c: Int) -> Bool:
    var nr = df.nrows()
    var nc = df.ncols()
    return (r >= 0 and r < nr) and (c >= 0 and c < nc)


# Internal: rebuild a column from string values
fn rebuild_column_with(vals: List[String], name: String) -> Column:
    return col_from_list(vals, name)


fn find_first(labels: List[String], target: String) -> Int:
    var i = 0
    var n = len(labels)
    while i < n:
        if labels[i] == target:
            return i
        i += 1
    return -1

fn labels_to_indices(index_vals: List[String], labels: List[String]) -> List[Int]:
    var out = List[Int]()
    var i = 0
    var m = len(labels)
    while i < m:
        var pos = find_first(index_vals, labels[i])
        if pos >= 0:
            out.append(pos)
        i += 1
    return out.copy() 


fn col_from_list(values: List[String], name: String) -> Column:
    var s = SeriesStr()
    s.set_name(String(name))
    s.data = values.copy()
    var n = len(s.data)
    if s.valid.len() != n:
        s.valid.resize(n, True)

    var col = Column()
    col.from_str(s)
    return col.copy()

 
# Build a column from a list of String with an explicit dtype tag
fn col_from_list_with_tag(vals: List[String], name: String, tag: Int) -> Column:
    var c = col_from_list(vals, name)   
    c.tag = tag                      
    return c.copy()


# --- Optional parsers returning value ---
fn parse_i64(s: String) -> Optional[Int64]:
    try:
        return Optional[Int64](Int64(s))
    except _:
        return Optional[Int64]()

fn parse_f64(s: String) -> Optional[Float64]:
    try:
        return Optional[Float64](Float64(s))
    except _:
        return Optional[Float64]()


fn coerce_to_i64(v: Value) -> Optional[Int64]:
    var s = v.as_string()
    var oi = parse_i64(s)
    if oi is not None:
        return oi
    var of = parse_f64(s)
    if of is not None:
        var z = Int64(of.value())
        return Optional[Int64](z)
    if s == "true" or s == "True" or s == "1":
        return Optional[Int64](Int64(1))
    if s == "false" or s == "False" or s == "0":
        return Optional[Int64](Int64(0))
    return Optional[Int64]()

fn coerce_to_f64(v: Value) -> Optional[Float64]:
    var s = v.as_string()
    var of = parse_f64(s)
    if of is not None:
        return of
    var oi = parse_i64(s)
    if oi is not None:
        var z = Float64(oi.value())
        return Optional[Float64](z)
    if s == "true" or s == "True" or s == "1":
        return Optional[Float64](1.0)
    if s == "false" or s == "False" or s == "0":
        return Optional[Float64](0.0)
    return Optional[Float64]()

fn coerce_to_bool(v: Value) -> Optional[Bool]:
    var s = v.as_string()
    if s == "true" or s == "True" or s == "1":
        return Optional[Bool](True)
    if s == "false" or s == "False" or s == "0":
        return Optional[Bool](False)
    var oi = parse_i64(s)
    if oi is not None:
        var b = oi.value() != 0
        return Optional[Bool](b)
    var of = parse_f64(s)
    if of is not None:
        var b2 = of.value() != 0.0
        return Optional[Bool](b2)
    return Optional[Bool]()



# --- type-preserving cell write (string storage + tag restore) ---

fn set_cell_preserve_type(mut df: DataFrame, r: Int, j: Int, v: Value) -> Bool:
    var cname = String(df.col_names[j])
    var original_tag = df.cols[j].dtype()   # Int tag
    var nr = df.nrows()

    var xs = List[String]()
    xs.reserve(nr)

    var k = 0
    if original_tag == tag_int64() or original_tag == tag_int32():
        var co = coerce_to_i64(v)
        if co is None: return False
        var repl = String(co.value())       # "999" / "-5"
        while k < nr:
            if k == r: xs.append(repl) else: xs.append(df.cols[j].get_string(k))
            k += 1
        df.cols[j] = col_from_list_with_tag(xs, cname, original_tag)
        return True

    if original_tag == tag_float64() or original_tag == tag_float32():
        var co = coerce_to_f64(v)
        if co is None: return False
        var repl = String(co.value())       # "999.0"
        while k < nr:
            if k == r: xs.append(repl) else: xs.append(df.cols[j].get_string(k))
            k += 1
        df.cols[j] = col_from_list_with_tag(xs, cname, original_tag)
        return True

    if original_tag == tag_bool():
        var co = coerce_to_bool(v)
        if co is None: return False
        var repl = String("")
        if co.value():
            repl = String("true")
        else:
            repl = String("false")
        while k < nr:
            if k == r: xs.append(repl) else: xs.append(df.cols[j].get_string(k))
            k += 1
        df.cols[j] = col_from_list_with_tag(xs, cname, original_tag)
        return True

    if original_tag == tag_string():
        var repl = String(v.as_string())
        while k < nr:
            if k == r: xs.append(repl) else: xs.append(df.cols[j].get_string(k))
            k += 1
        df.cols[j] = col_from_list_with_tag(xs, cname, original_tag)  # tag_string()
        return True

    return False
