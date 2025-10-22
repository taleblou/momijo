# Project:      Momijo
# Module:       dataframe.api
# File:         api.mojo
# Path:         dataframe/api.mojo
#
# Description:  dataframe.api — Public API façade aggregating common surface functions.
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
#   - Structs: —
#   - Key functions: _find_col, df_from_columns, df_from_pairs, df_make, col_str, df_head, df_shape, df_dtypes, df_describe, df_rename, df_set_index, df_reset_index, make_pairs, pairs_append, df_nlargest, df_nsmallest, df_clip, df_fillna_i64

from momijo.dataframe.column import ColumnTag,get_string
from momijo.dataframe.selection import RowRange, ColRange, loc, iloc, select
from momijo.dataframe._groupby_core import groupby_agg, groupby_transform
from momijo.dataframe.helper import align_columns, align_rows
from momijo.dataframe.io_csv import read_csv, to_csv
from momijo.dataframe.io_json import read_json, to_json, read_json_lines, to_json_lines
from momijo.dataframe.utils import infer_dtype, compute_stats,clip_f64, nlargest_f64, nsmallest_f64
from momijo.dataframe.frame import DataFrame, get_column_at, width
from momijo.dataframe.helpers import find_col,__dict_contains
from momijo.dataframe.na import fillna_col_bool, fillna_col_f64, fillna_col_i64 
from momijo.dataframe.series_bool import append
from momijo.dataframe.series_f64 import SeriesF64
from momijo.dataframe.series_str import SeriesStr
from momijo.dataframe.series_i64 import SeriesI64
from momijo.dataframe.series_bool import SeriesBool
from momijo.dataframe.bitmap import *


from momijo.dataframe.series_bool import SeriesBool as SeriesBoolT
from momijo.dataframe.series_str import SeriesStr as SeriesStrT
from momijo.dataframe.series_f64 import SeriesF64 as SeriesF64T
from momijo.dataframe.series_i64 import SeriesI64 as SeriesI64T
from collections.dictionary import Dictionary
 
# Type aliases
#alias ColPair = (String, List[String])
# ColPair: a (name, values) column pair with deep-copy semantics.
# Notes:
# - Not ImplicitlyCopyable because it owns a List[String].
# - Provides explicit __copyinit__ to satisfy Copyable.

from collections.list import List


# -------------------- Minimal DType facade (extend if you have one) --------------------
# MIT License
# SPDX-License-Identifier: MIT
# Project: momijo.dataframe
# File: src/momijo/dataframe/dtype.mojo
# Description: Simple DType with nullability flag.

struct DType:
    var tag: Int
    var nullable: Bool

    fn __init__(out self, tag: Int, nullable: Bool = False):
        self.tag = tag
        self.nullable = nullable

    fn __eq__(self, other: Self) -> Bool:
        # Equal only if both tag and nullability match
        return self.tag == other.tag and self.nullable == other.nullable

    fn __ne__(self, other: Self) -> Bool:
        return not (self == other)

    # Optional helper: produce same tag with different nullability
    fn with_nullable(self, make_nullable: Bool) -> DType:
        return DType(self.tag, make_nullable)

    # ------- Static constructors (nullable-aware) -------
    @staticmethod
    fn BOOL(nullable: Bool = False) -> DType:
        return DType(1, nullable)

    @staticmethod
    fn INT32(nullable: Bool = False) -> DType:
        return DType(2, nullable)

    @staticmethod
    fn INT64(nullable: Bool = False) -> DType:
        return DType(3, nullable)

    @staticmethod
    fn FLOAT32(nullable: Bool = False) -> DType:
        return DType(4, nullable)

    @staticmethod
    fn FLOAT64(nullable: Bool = False) -> DType:
        return DType(5, nullable)

    @staticmethod
    fn STRING(nullable: Bool = False) -> DType:
        return DType(6, nullable)


fn tag_bool()    -> Int: return 1
fn tag_int32()   -> Int: return 2
fn tag_int64()   -> Int: return 3
fn tag_float32() -> Int: return 4
fn tag_float64() -> Int: return 5
fn tag_string()  -> Int: return 6

fn tag_name(t: Int) -> String:
    if t == tag_bool():    return String("BOOL")
    if t == tag_int32():   return String("INT32")
    if t == tag_int64():   return String("INT64")
    if t == tag_float32(): return String("FLOAT32")
    if t == tag_float64(): return String("FLOAT64")
    if t == tag_string():  return String("STRING")
    return String("UNKNOWN(" + String(t) + ")")

fn dtype_name(tag: Int) -> String:
    if tag == tag_bool():    return String("bool")
    if tag == tag_int32():   return String("int32")
    if tag == tag_int64():   return String("int64")
    if tag == tag_float32(): return String("float32")
    if tag == tag_float64(): return String("float64")
    if tag == tag_string():  return String("string")
    return String("unknown(" + String(tag) + ")")


# ---- DType -> tag (single source of truth) ----
fn dtype_to_tag(dt: DType) -> Int:
    if dt == DType.BOOL():    return tag_bool()
    if dt == DType.INT32():   return tag_int32()
    if dt == DType.INT64():   return tag_int64()
    if dt == DType.FLOAT32(): return tag_float32()
    if dt == DType.FLOAT64(): return tag_float64()
    if dt == DType.STRING():  return tag_string()
    return tag_string()  # safe default
# ---- Value dtype tag ----
fn value_tag(v: Value) -> Int:
    # Use whichever your Value exposes: get_dtype() or dtype
    return v.get_dtype().tag   # change to `v.dtype.tag` if yours is a field

fn bool(nullable: Bool = False) -> DType:
    return DType.BOOL(nullable=nullable)

fn int32(nullable: Bool = False) -> DType:
    return DType.INT32(nullable=nullable)

fn int64(nullable: Bool = False) -> DType:
    return DType.INT64(nullable=nullable)

fn float32(nullable: Bool = False) -> DType:
    return DType.FLOAT32(nullable=nullable)

fn float64(nullable: Bool = False) -> DType:
    return DType.FLOAT64(nullable=nullable)

fn string(nullable: Bool = False) -> DType:
    return DType.STRING(nullable=nullable)


# ColumnTag-aware helpers for Int-based dtype()
@always_inline
fn is_f64_tag(tag: Int) -> Bool:  return tag == ColumnTag.F64()
@always_inline
fn is_i64_tag(tag: Int) -> Bool:  return tag == ColumnTag.I64()
@always_inline
fn is_str_tag(tag: Int) -> Bool:  return tag == ColumnTag.STR()
@always_inline
fn is_bool_tag(tag: Int) -> Bool: return tag == ColumnTag.BOOL()

@always_inline
fn is_f64_col(c: Column) -> Bool:  return is_f64_tag(c.dtype())
@always_inline
fn is_i64_col(c: Column) -> Bool:  return is_i64_tag(c.dtype())
@always_inline
fn is_str_col(c: Column) -> Bool:  return is_str_tag(c.dtype())
@always_inline
fn is_bool_col(c: Column) -> Bool: return is_bool_tag(c.dtype())



# --- Float64 helpers (NaN) ---
# ---------- Float64 / NaN helpers ----------
@always_inline
fn f64_nan() -> Float64:
    var z = 0.0
    return z / z  # NaN

@always_inline
fn is_nan(x: Float64) -> Bool:
    return x != x
 
# Returns (value, ok). ok=False ⇒ value=NaN.
fn try_parse_f64(s: String) -> (Float64, Bool):
    # empty-like sentinels
    if s == "" or s == "nan" or s == "NaN" or s == "None":
        return (f64_nan(), False)

    var i = 0
    var n = len(s)
    var neg = False

    # optional sign
    if i < n and (s[i] == '-' or s[i] == '+'):
        neg = (s[i] == '-')
        i += 1

    # integer part
    var have_int = False
    var int_part: Int = 0
    while i < n:
        var ch = s[i]
        var is_digit = (ch == '0' or ch == '1' or ch == '2' or ch == '3' or ch == '4' or
                        ch == '5' or ch == '6' or ch == '7' or ch == '8' or ch == '9')
        if not is_digit:
            break
        have_int = True
        var d: Int = 0
        if ch == '1':
            d = 1
        elif ch == '2':
            d = 2
        elif ch == '3':
            d = 3
        elif ch == '4':
            d = 4
        elif ch == '5':
            d = 5
        elif ch == '6':
            d = 6
        elif ch == '7':
            d = 7
        elif ch == '8':
            d = 8
        elif ch == '9':
            d = 9
        else:
            d = 0
        int_part = int_part * 10 + d
        i += 1

    # fractional part
    var have_frac = False
    var frac_part: Int = 0
    var frac_base: Int = 1
    if i < n and s[i] == '.':
        i += 1
        while i < n:
            var ch2 = s[i]
            var is_digit2 = (ch2 == '0' or ch2 == '1' or ch2 == '2' or ch2 == '3' or ch2 == '4' or
                             ch2 == '5' or ch2 == '6' or ch2 == '7' or ch2 == '8' or ch2 == '9')
            if not is_digit2:
                break
            have_frac = True
            var d2: Int = 0
            if ch2 == '1':
                d2 = 1
            elif ch2 == '2':
                d2 = 2
            elif ch2 == '3':
                d2 = 3
            elif ch2 == '4':
                d2 = 4
            elif ch2 == '5':
                d2 = 5
            elif ch2 == '6':
                d2 = 6
            elif ch2 == '7':
                d2 = 7
            elif ch2 == '8':
                d2 = 8
            elif ch2 == '9':
                d2 = 9
            else:
                d2 = 0
            frac_part = frac_part * 10 + d2
            frac_base = frac_base * 10
            i += 1

    # reject trailing garbage or missing digits
    if i != n or (not have_int and not have_frac):
        return (f64_nan(), False)

    var v = Float64(int_part)
    if have_frac:
        v = v + Float64(frac_part) / Float64(frac_base)
    if neg:
        v = -v
    return (v, True)

fn parse_f64_or_nan(s: String) -> Float64:
    var (v, ok) = try_parse_f64(s)
    if ok:
        return v
    return f64_nan()

# -------------------- Series facades (return Column) --------------------
 

# Single generic facade: covers Bool, Int, Float64, String (and similar),
# and returns List[String] to feed your existing DataFrame(columns, data, index, ...) 

# ---------------- Int (no nulls) ----------------
fn Series(values: List[Int], dtype: DType) -> List[String]:
    var out = List[String]()
    var i = 0
    var n = len(values)
    while i < n:
        out.append(String(values[i]))
        i += 1
    return out.copy()

# ---------------- String (no nulls) ----------------
fn Series(values: List[String], dtype: DType) -> List[String]:
    return values.copy()

# ===== Float64, Bool, and nullable variants use distinct names =====

# # ---------------- Float64 (no nulls) ----------------
# fn Series(values: List[Float64], dtype: DType) -> List[String]:
#     var out = List[String]()
#     var i = 0
#     var n = len(values)
#     while i < n:
#         out.append(String(values[i]))
#         i += 1
#     return out.copy()

# # ---------------- Float64 (nullable) ----------------
# fn Series(values: List[Optional[Float64]], dtype: DType) -> List[String]:
#     var out = List[String]()
#     var i = 0
#     var n = len(values)
#     while i < n:
#         var v = values[i]
#         if v is None:
#             out.append(String(""))
#         else:
#             out.append(String(v.value()))
#         i += 1
#     return out.copy()

# # ---------------- Bool (no nulls) ----------------
# fn Series(values: List[Bool], dtype: DType) -> List[String]:
#     var out = List[String]()
#     var i = 0
#     var n = len(values)
#     while i < n:
#         if values[i]:
#             out.append(String("True"))
#         else:
#             out.append(String("False"))
#         i += 1
#     return out.copy()

# # ---------------- Bool (nullable) ----------------
# fn Series(values: List[Optional[Bool]], dtype: DType) -> List[String]:
#     var out = List[String]()
#     var i = 0
#     var n = len(values)
#     while i < n:
#         var v = values[i]
#         if v is None:
#             out.append(String(""))
#         else:
#             if v.value():
#                 out.append(String("True"))
#             else:
#                 out.append(String("False"))
#         i += 1
#     return out.copy()

# # ---------------- String (nullable) ----------------
# fn Series(values: List[Optional[String]], dtype: DType) -> List[String]:
#     var out = List[String]()
#     var i = 0
#     var n = len(values)
#     while i < n:
#         var s = values[i]
#         if s is None:
#             out.append(String(""))
#         else:
#             out.append(s.value())
#         i += 1
#     return out.copy()


 


# -------------------- Index & DataFrame facades --------------------
fn ToDataFrame(mapping: Dict[String, List[String]], index: List[String]) -> DataFrame:
    var names = List[String]()
    for k in mapping.keys():
        names.append(String(k))

    var data = List[List[String]]()
    var i = 0
    while i < len(names):
        var name = names[i]
        var opt_vals = mapping.get(name)
        if opt_vals is not None:
            var vals = opt_vals.value().copy()     # Optional.value() per your project
            data.append(vals.copy())
        i += 1

    # delegate to your existing ctor: DataFrame(columns, data, index, index_name="")
    return DataFrame(names, data, index, String(""))

 


# ---------- helpers: convert typed lists to List[String] ----------
fn to_string_list_i(xs: List[Int]) -> List[String]:
    var out = List[String]()
    var i = 0
    while i < len(xs):
        out.append(String(xs[i]))
        i += 1
    return out

fn to_string_list_f(xs: List[Float64]) -> List[String]:
    var out = List[String]()
    var i = 0
    while i < len(xs):
        out.append(String(xs[i]))
        i += 1
    return out

fn to_string_list_s(xs: List[String]) -> List[String]:
    return xs.copy()

 

# ---------- ToDataFrame without index (auto 0..n-1) ----------
fn ToDataFrame(mapping: Dict[String, List[String]]) -> DataFrame:
    # Collect names first
    var names = List[String]()
    for k in mapping.keys():
        names.append(String(k))

    # Determine nrows from the first present column
    var nrows = 0
    var j = 0
    while j < len(names):
        var opt_vals = mapping.get(names[j])
        if opt_vals is not None:
            nrows = len(opt_vals.value())
            break
        j += 1

    # Build default string index: "0","1",...,"nrows-1"
    var index = List[String]()
    var r = 0
    while r < nrows:
        index.append(String(r))
        r += 1

    # Reuse the explicit-index builder
    return ToDataFrame(mapping, index)


fn Index(labels: List[String]) -> List[String]:
    return labels.copy()

struct ColKind(Copyable, Movable):
    var tag: Int
    fn __init__(out self):
        self.tag = 0   

    @staticmethod
    fn STR() -> ColKind:
        var k = ColKind()
        k.tag = 0
        return k.copy()
    @staticmethod
    fn I64() -> ColKind:
        var k = ColKind()
        k.tag = 1
        return k.copy()
    @staticmethod
    fn F64() -> ColKind:
        var k = ColKind()
        k.tag = 2
        return k.copy()


struct ColPair(Copyable, Movable):
    var name: String
    var kind: ColKind
    var s: List[String]
    var i: List[Int]
    var f: List[Float64]

    fn __init__(out self):
        self.name = String("")
        self.kind = ColKind.STR()
        self.s = List[String]()
        self.i = List[Int]()
        self.f = List[Float64]() 

    fn __copyinit__(out self, other: Self):
        self.name = String(other.name)
        self.kind = other.kind.copy()
        self.s = other.s.copy()
        self.i = other.i.copy()
        self.f = other.f.copy()
 
    @staticmethod
    fn of_str(name: String, values: List[String]) -> Self:
        var cp = ColPair()
        cp.name = String(name)
        cp.kind = ColKind.STR()
        cp.s = values.copy()
        return cp.copy()

    @staticmethod
    fn of_i64(name: String, values: List[Int]) -> Self:
        var cp = ColPair()
        cp.name = String(name)
        cp.kind = ColKind.I64()
        cp.i = values.copy()
        return cp.copy()

    @staticmethod
    fn of_f64(name: String, values: List[Float64]) -> Self:
        var cp = ColPair()
        cp.name = String(name)
        cp.kind = ColKind.F64()
        cp.f = values.copy()
        return cp.copy()
  
 
    # -------- Convenience factories --------
    @staticmethod
    fn of(name: String, values: List[String]) -> Self:
        var out = ColPair(name, values)
        return out

    @staticmethod
    fn empty(name: String) -> Self:
        var out_vals = List[String]()
        var out = ColPair(name, out_vals)
        return out

    @staticmethod
    fn single(name: String, value: String) -> Self:
        var out_vals = List[String]()
        out_vals.append(String(value))
        var out = ColPair(name, out_vals)
        return out

    # -------- Basic utilities --------
    fn len(self) -> Int:
        return len(self.values)

    fn is_empty(self) -> Bool:
        return len(self.values) == 0

    # Safe access: returns a copy of the value at index i; clamps to valid range.
    fn get(self, i: Int) -> String:
        var n = len(self.values)
        if n == 0:
            return String("")
        var ii = i
        if ii < 0:
            ii = 0
        if ii >= n:
            ii = n - 1
        return String(self.values[ii])

    # Push one value (owned copy)
    fn push(mut self, value: String) -> None:
        self.values.append(String(value))

    # Extend by another list (deep-copy each element)
    fn extend(mut self, more: List[String]) -> None:
        var j = 0
        var m = len(more)
        while j < m:
            self.values.append(String(more[j]))
            j += 1

    # Clear all values
    fn clear(mut self) -> None:
        # Replace with a fresh empty list to drop capacity if desired
        var empty_list = List[String]()
        self.values = empty_list

    # Return a printable summary
    fn __str__(self) -> String:
        var b = String("ColPair(name=") + self.name + String(", n=") + String(len(self.values)) + String(", values=[")
        var n = len(self.values)
        var i = 0
        while i < n:
            b = b + self.values[i]
            if i + 1 < n:
                b = b + String(", ")
            i += 1
        b = b + String("])")
        return b



fn make_pairs() -> List[ColPair]:
    return List[ColPair]()
# For index-based pairs
struct IndexPair(ExplicitlyCopyable, Movable):
    var i: Int
    var j: Int

    fn __init__(out self, i: Int, j: Int):
        self.i = i
        self.j = j

    # Explicit copy initializer (required by ExplicitlyCopyable)
    fn __copyinit__(out self, other: Self):
        self.i = other.i
        self.j = other.j

    # Explicit copy() method (required by ExplicitlyCopyable)
    fn copy(self) -> Self:
        return Self(self)
 


# Internal helper: find column index by name (returns -1 if not found)
fn _find_col(df: DataFrame, name: String) -> Int:
    var i = 0
    while i < df.ncols():
        if df.col_names[i] == name:
            return i
        i += 1
    return -1

# Constructors
fn df_from_columns(columns: List[String], data: List[List[String]]) -> DataFrame:
# Assumes: len(columns) == len(data), and each data[i] has same length.
    var idx = List[String]()
    return DataFrame(columns, data, idx, String(""))

 
 

# ColPair: struct with fields .name: String and .values: List[String]

# Ensure names are non-empty and unique by adding suffixes when needed. 
# ---------- Helpers (keep English-only comments) ----------
fn _ensure_unique_names(raw: List[String]) -> List[String]:
    var out = List[String]()
    var i: Int = 0
    var n: Int = len(raw)
    while i < n:
        var base = raw[i]
        if len(base) == 0:
            base = String("_col_") + String(i)

        var name = base
        var suffix: Int = 1
        var k: Int = 0
        while k < len(out):
            if out[k] == name:
                name = base + String("_") + String(suffix)
                suffix += 1
                k = 0
                continue
            k += 1

        out.append(name)
        i += 1
    return out.copy()

fn _columns_to_rows(cols_values: List[List[String]]) -> List[List[String]]:
    var ncols: Int = len(cols_values)
    if ncols == 0:
        return List[List[String]]()
    var nrows: Int = len(cols_values[0])

    var rows = List[List[String]]()
    var r: Int = 0
    while r < nrows:
        var row = List[String]()
        var c: Int = 0
        while c < ncols:
            row.append(cols_values[c][r])
            c += 1
        rows.append(row.copy())
        r += 1
    return rows.copy()


 
fn _to_strings_i64(xs: List[Int], limit: Int) -> List[String]:
    var out = List[String]()
    var j = 0
    var n = len(xs)
    var m = limit
    while j < m and j < n:
        out.append(String(xs[j]))
        j += 1
    while j < m:
        out.append(String(""))
        j += 1
    return out.copy()

fn _to_strings_f64(xs: List[Float64], limit: Int) -> List[String]:
    var out = List[String]()
    var j = 0
    var n = len(xs)
    var m = limit
    while j < m and j < n:
        out.append(String(xs[j]))
        j += 1
    while j < m:
        out.append(String(""))
        j += 1
    return out.copy()
fn df_from_pairs(pairs: List[ColPair]) -> DataFrame:
    var n = len(pairs)
    if n == 0:
        return df_from_columns(List[String](), List[List[String]]())

    var headers = List[String]()
    var min_len: Int = -1
    var k = 0
    while k < n:
        headers.append(pairs[k].name)
        var vlen = 0
        var kind = pairs[k].kind.tag
        if kind == ColKind.STR().tag:
            vlen = len(pairs[k].s)
        elif kind == ColKind.I64().tag:
            vlen = len(pairs[k].i)
        else:
            vlen = len(pairs[k].f)
        if min_len < 0 or vlen < min_len:
            min_len = vlen
        k += 1

    if min_len <= 0:
        return df_from_columns(headers, List[List[String]]())

    var data_s = List[List[String]]()
    k = 0
    while k < n:
        var kind2 = pairs[k].kind.tag
        if kind2 == ColKind.STR().tag:
            var src = pairs[k].s.copy()
            var col = List[String]()
            var j = 0
            var cap = len(src)
            while j < min_len and j < cap:
                col.append(src[j])
                j += 1
            while j < min_len:
                col.append(String(""))
                j += 1
            data_s.append(col.copy())
        elif kind2 == ColKind.I64().tag:
            var tmp_i = _to_strings_i64(pairs[k].i, min_len)
            data_s.append(tmp_i.copy())
        else:
            var tmp_f = _to_strings_f64(pairs[k].f, min_len)
            data_s.append(tmp_f.copy())
        k += 1

    var frame = df_from_columns(headers, data_s)

    k = 0
    while k < n:
        var nm = headers[k]
        var kind3 = pairs[k].kind.tag
        if kind3 == ColKind.I64().tag:
            frame = coerce_str_to_i64(frame, nm)
        elif kind3 == ColKind.F64().tag:
            frame = coerce_str_to_f64(frame, nm)
        # STR: no coercion
        k += 1

    return frame


fn coerce_str_to_i64(frame: DataFrame, col: String) -> DataFrame:
    var out = copy(frame)

    # locate column index
    var idx: Int = -1
    var c = 0
    while c < out.ncols():
        if out.col_names[c] == col:
            idx = c
            break
        c += 1
    if idx < 0:
        return out

    # already int → no-op
    if out.cols[idx].tag == ColumnTag.I64():
        return out

    # if string → coerce to f64 first
    if out.cols[idx].tag == ColumnTag.STR():
        out = coerce_str_to_f64(out, col)

    # expect f64 now
    if out.cols[idx].tag != ColumnTag.F64():
        return out

    # build Int values by truncating f64
    var n = out.nrows()
    var vals = List[Int]()
    vals.reserve(n)

    var i = 0
    while i < n:
        vals.append(Int(out.cols[idx].f64.get(i)))
        i += 1

    # construct SeriesI64 using (name, values)
    var s_i64 = SeriesI64(col, vals.copy())

    # replace the column in place
    out.cols[idx].set_i64_series(s_i64)

    return out



 

# Minimal df_make: build DataFrame from column names and values
fn df_make(col_names: List[String], cols: List[List[String]]) -> DataFrame:
    var idx = List[String]()
    return DataFrame(col_names, cols, idx, String(""))

 

# Introspection
fn df_head(df: DataFrame, n: Int) -> DataFrame:
# Return first n rows (or all rows if n >= nrows)
    var rows = df.nrows()
    var k = n
    if k > rows:
        k = rows

    var out = DataFrame()
    out.index_name = df.index_name
    out.index_vals = List[String]()

# Copy index values if they exist and lengths match
    if len(df.index_vals) == rows:
        var r = 0
        while r < k:
            out.index_vals.append(String(df.index_vals[r]))
            r += 1

# Copy columns
    var c = 0
    while c < df.ncols():
        out.col_names.append(String(df.col_names[c]))
        var col = List[String]()
        var r2 = 0
        while r2 < k:
            col.append(String(df.cols[c][r2]))
            r2 += 1
        out.cols.append(col)
        c += 1
    return out

fn df_shape(df: DataFrame) -> String:
    var r = df.nrows()
    var c = df.ncols()
    return String("(") + String(r) + String(", ") + String(c) + String(")")

fn df_dtypes(frame: DataFrame) -> String:
    var out = String("")
    var i = 0
    while i < frame.ncols():
        var c = frame.get_column_by_index(i)  # or get_column(frame.col_names[i])
        var tag = c.dtype()
        var name = frame.col_names[i]
        var s = String("")
        if tag == ColumnTag.STR():
            s = "string"
        elif tag == ColumnTag.F64():
            s = "float"
        elif tag == ColumnTag.I64():
            s = "int"
        elif tag == ColumnTag.BOOL():
            s = "bool"
        else:
            s = "unknown"
        out += name + ": " + s + "\n"
        i += 1
    return out

fn df_describe(df: DataFrame) -> String:
    var s = String("")
    var wrote = 0
    var i = 0
    while i < df.ncols():
        var dt = infer_dtype(df.cols[i])
        if dt == String("int") or dt == String("float"):
            var st = compute_stats(df.cols[i])
            if wrote == 1:
                s = s + String("\n")
            s = s + String("[") + df.col_names[i] + String("] ")
            s = s + String("count=") + String(st.count)
            s = s + String(" mean=") + String(st.mean)
            s = s + String(" std=") + String(st.std)
            s = s + String(" min=") + String(st.min)
            s = s + String(" max=") + String(st.max)
            wrote = 1
        i += 1
    if wrote == 0:
        s = String("(no numeric columns)")
    return s

# Rename / index helpers
# NOTE: 'Dict' is assumed available in project context; adjust import if needed.
fn df_rename(df: DataFrame, mapping: Dict[String, String], axis_name: String) -> DataFrame:
    var out = df.copy()
    var i = 0
    while i < out.ncols():
        var old_name = out.col_names[i]
        if old_name in mapping:
            out.col_names[i] = String(mapping[old_name])
        i += 1
    out.index_name = String(axis_name)
    return out

fn df_set_index(df: DataFrame, col_name: String) -> DataFrame:
    var idx = _find_col(df, col_name)
    if idx < 0:
        return df.copy()

    var out = DataFrame()
    out.index_name = String(col_name)
    out.index_vals = List[String]()

    var rows = df.nrows()
    var r = 0
    while r < rows:
        out.index_vals.append(String(df.cols[idx][r]))
        r += 1

    var c = 0
    while c < df.ncols():
        if c != idx:
            out.col_names.append(String(df.col_names[c]))
            var cc = List[String]()
            var rr = 0
            while rr < rows:
                cc.append(String(df.cols[c][rr]))
                rr += 1
            out.cols.append(cc)
        c += 1

    return out

fn df_reset_index(df: DataFrame) -> DataFrame:
    if not (df.index_name != String("") and len(df.index_vals) == df.nrows()):
        return df.copy()

    var out = DataFrame()
    out.col_names.append(String(df.index_name))

    var ic = List[String]()
    var i = 0
    while i < len(df.index_vals):
        ic.append(String(df.index_vals[i]))
        i += 1
    out.cols.append(ic)

    var c = 0
    while c < df.ncols():
        out.col_names.append(String(df.col_names[c]))
        var cc = List[String]()
        var r = 0
        while r < df.nrows():
            cc.append(String(df.cols[c][r]))
            r += 1
        out.cols.append(cc)
        c += 1

    out.index_name = String("")
    out.index_vals = List[String]()
    return out


# -------------------------------------------------------------------
# Function: make_Index
# Returns an empty list of IndexPair.
# Useful as a placeholder or default return value.
# -------------------------------------------------------------------
fn make_Index() -> List[IndexPair]:
    return List[IndexPair]()

# -------------------------------------------------------------------
# Function: make_Index
# Overloaded version.
# For a given number of columns n, generate all index pairs (i, j)
# such that 0 <= i < j < n.
# Example: n = 3 -> [(0,1), (0,2), (1,2)]
# -------------------------------------------------------------------
fn make_Index(n: Int) -> List[IndexPair]:
    var pairs = List[IndexPair]()
    var i = 0
    while i < n:
        var j = i + 1
        while j < n:
            pairs.append(IndexPair(i, j))
            j += 1
        i += 1
    return pairs


fn df_nlargest(df: DataFrame, col: String, n: Int) -> DataFrame
    var idx = find_col(df, col)
    if idx < 0:
        return df
    var scores = List[Float64]()
    var i = 0
    while i < df.nrows():
        scores.append(df.cols[idx].get_f64(i))
        i += 1
    var top_idx = nlargest_f64(scores, n)
    var col_names = df.col_names
    var cols = List[List[String]]()
    var c = 0
    while c < df.ncols():
        var vals = List[String]()
        var j = 0
        while j < len(top_idx):
            vals.append(df.cols[c][top_idx[j]])
            j += 1
        cols.append(Column.from_str(SeriesStr(col_names[c], vals)))
        c += 1
    return DataFrame(col_names, cols)
fn df_nsmallest(df: DataFrame, col: String, n: Int) -> DataFrame
    var idx = find_col(df, col)
    if idx < 0:
        return df
    var scores = List[Float64]()
    var i = 0
    while i < df.nrows():
        scores.append(df.cols[idx].get_f64(i))
        i += 1
    var bot_idx = nsmallest_f64(scores, n)
    var col_names = df.col_names
    var cols = List[List[String]]()
    var c = 0
    while c < df.ncols():
        var vals = List[String]()
        var j = 0
        while j < len(bot_idx):
            vals.append(df.cols[c][bot_idx[j]])
            j += 1
        cols.append(Column.from_str(SeriesStr(col_names[c], vals)))
        c += 1
    return DataFrame(col_names, cols)

fn df_clip(df: DataFrame, col: String, lo: Float64, hi: Float64) -> DataFrame
    var idx = find_col(df, col)
    if idx < 0: return df
    var vals = List[Float64]()
    var r = 0
    while r < df.nrows():
        vals.append(df.cols[idx].get_f64(r))
        r += 1
    var clipped = clip_f64(vals, lo, hi)
    var col_names = df.col_names
    var cols = List[List[String]]()
    var c = 0
    while c < df.ncols():
        if c == idx:
            cols.append(Column.from_f64(SeriesF64(col_names[c], clipped)))
        else:
            cols.append(df.get_column_at(c))
        c += 1
    return DataFrame(col_names, cols)
fn df_fillna_i64(df: DataFrame, col: String, fill: Int64) -> DataFrame
    var idx = find_col(df, col)
    if idx < 0: return df
    var col_names = df.col_names
    var cols = List[List[String]]()
    var c = 0
    while c < df.ncols():
        if c == idx: cols.append(fillna_col_i64(df.get_column_at(c), fill)) else: cols.append(df.get_column_at(c))
        c += 1
    return DataFrame(col_names, cols)
fn df_fillna_f64(df: DataFrame, col: String, fill: Float64) -> DataFrame
    var idx = find_col(df, col)
    if idx < 0: return df
    var col_names = df.col_names
    var cols = List[List[String]]()
    var c = 0
    while c < df.ncols():
        if c == idx: cols.append(fillna_col_f64(df.get_column_at(c), fill)) else: cols.append(df.get_column_at(c))
        c += 1
    return DataFrame(col_names, cols)
fn df_fillna_bool(df: DataFrame, col: String, fill: Bool) -> DataFrame
    var idx = find_col(df, col)
    if idx < 0: return df
    var col_names = df.col_names
    var cols = List[List[String]]()
    var c = 0
    while c < df.ncols():
        if c == idx: cols.append(fillna_col_bool(df.get_column_at(c), fill)) else: cols.append(df.get_column_at(c))
        c += 1
    return DataFrame(col_names, cols)

# ---- Moved from __init__.mojo (facade helpers) ----

# [moved] series_from_list
fn series_from_list(values: List[Int], name: String, index: List[String]) -> Series:
    return _series_from_list(index, values, name)

# Print-friendly accessors (String outputs)

# [moved] series_values
fn series_values(s: Series) -> String:
    var xs = s.values_
    var out = String("[")
    var i = 0
    var n = len(xs)
    while i < n:
        out += xs[i]
        if i + 1 < n: out += ", "
        i += 1
    out += "]"
    return out

fn series_index(s: Series) -> String:
    var xs = s.index_vals
    var out = String("[")
    var i = 0
    var n = len(xs)
    while i < n:
        out += xs[i]
        if i + 1 < n: out += ", "
        i += 1
    out += "]"
    return out

# ---- DataFrame constructors & meta ----

 
fn pairs_append(pairs: List[ColPair], name: String, values: List[String]) -> List[ColPair]:
    var out = pairs.copy()
    out.append(ColPair.of_str(name, values))
    return out.copy()

fn pairs_append(pairs: List[ColPair], name: String, values: List[Int]) -> List[ColPair]:
    var out = pairs.copy()
    out.append(ColPair.of_i64(name, values))
    return out.copy()

fn pairs_append(pairs: List[ColPair], name: String, values: List[Float64]) -> List[ColPair]:
    var out = pairs.copy()
    out.append(ColPair.of_f64(name, values))
    return out.copy()
 


# ---- Drop NA (any) ----


# # ---- Categorical helper ----
# # Converts a string column to a categorical-like representation.
# # - If `categories` is provided, values outside it are mapped to "-1".
# # - If `ordered` is true, the provided order defines code order.
# # - Adds a new code column when `new_name` != "" (default: "category_code").
# fn to_category(frame: DataFrame, col: String, categories: List[String] = List[String](), ordered: Bool = False, new_name: String = String("category_code")) -> DataFrame:
#     var out = frame.copy()
#     # Find target column index
#     var idx = -1
#     var c = 0
#     while c < out.ncols():
#         if out.col_names[c] == col:
#             idx = c
#             break
#         c += 1
#     if idx < 0:
#         return out   # column not found -> no-op

#     # Build category index
#     var cats = categories
#     if len(cats) == 0:
#         # infer unique values in appearance order
#         var seen = Dict[String, Int]()
#         var i = 0
#         while i < out.cols[idx].len():
#             var s = out.cols[idx].get_string(i)
#             if seen.get(s) is None:
#                 seen[s] = len(cats)
#                 cats.append(s)
#             i += 1

#     # Map to codes
#     var code_series = List[String]()
#     var r = 0
#     while r < out.cols[idx].len():
#         var s = out.cols[idx].get_string(r)
#         var code = -1
#         var pos_opt = None
#         var j = 0
#         while j < len(cats):
#             if cats[j] == s:
#                 code = j
#                 break
#             j += 1
#         code_series.append(String(code))
#         r += 1

#     # Optionally append code column
#     if len(new_name) > 0:
#         var col_code = Column()
#         var s_code = SeriesStr(code_series, new_name)
#         col_code.from_str(s_code)
#         out.cols.append(col_code)
#         out.col_names.append(new_name)

#     return out
# ---- Index ops ----

# [moved] rename
# [removed invalid method-style rename — replaced by free-function below]
 # Build a string Column from name + data
fn col_str(name: String, data: List[String]) -> Column:
    var s = SeriesStr()
    s.set_name(name)
    s.data = data.copy()

    var n = s.len()
    s.valid.resize(n, False)
    var j = 0
    while j < n:
        var miss = (len(s.data[j]) == 0) or (s.data[j] == "NaN") or (s.data[j] == "nan")
        s.valid.set(j, not miss)
        j += 1

    var c = Column()
    c.from_str(s)    
    return c.copy()        
        
# Build a String column from name + data + validity (deep copies, no aliasing)
fn col_string_with_valid(name: String, data: List[String], valid: Bitmap) -> Column:
    # --- build SeriesStr
    var s = SeriesStr()
    s.set_name(name)

    # copy data buffer
    s.data.clear()
    s.data.reserve(len(data))
    var i = 0
    while i < len(data):
        s.data.append(data[i])
        i += 1

    # rebuild validity bitmap (deep copy)
    s.valid.resize(len(data), False)
    i = 0
    while i < len(data):
        _ = s.valid.set(i, valid.is_set(i))
        i += 1

    # wrap into Column via setter (avoids implicit copy of whole struct)
    var c = Column()
    c.set_string_series(s)
    return c.copy()

# Build an Int64 column from name + data + validity (deep copies, no aliasing)
fn col_i64_with_valid(name: String, data: List[Int], valid: Bitmap) -> Column:
    # --- build SeriesI64
    var s = SeriesI64()
    s.set_name(name)

    # copy data buffer
    s.data.clear()
    s.data.reserve(len(data))
    var i = 0
    while i < len(data):
        s.data.append(data[i])
        i += 1

    # rebuild validity bitmap (deep copy)
    s.valid.resize(len(data), False)
    i = 0
    while i < len(data):
        _ = s.valid.set(i, valid.is_set(i))
        i += 1

    # wrap into Column
    var c = Column()
    c.set_i64_series(s)        # NOTE: ensure this setter exists per your Column API
    return c.copy()


# Build a Bool column from name + data + validity (deep copies, no aliasing)
fn col_bool_with_valid(name: String, data: List[Bool], valid: Bitmap) -> Column:
    # --- build SeriesBool
    var s = SeriesBool()
    s.set_name(name)

    # copy data buffer
    s.data.clear()
    s.data.reserve(len(data))
    var i = 0
    while i < len(data):
        s.data.append(data[i])
        i += 1

    # rebuild validity bitmap (deep copy)
    s.valid.resize(len(data), False)
    i = 0
    while i < len(data):
        _ = s.valid.set(i, valid.is_set(i))
        i += 1

    # wrap into Column
    var c = Column()
    c.set_bool_series(s)       # NOTE: ensure this setter exists per your Column API
    return c.copy()


# Build a Float64 column from name + data + validity (deep copies, no aliasing)
fn col_f64_with_valid(name: String, data: List[Float64], valid: Bitmap) -> Column:
    var c = Column()
    c.tag = ColumnTag.F64()
    c.f64 = SeriesF64()
    c.f64.set_name(name)
    c.f64.data = data.copy()
    c.f64.valid = valid.copy()

    # Reset other variants to empty
    c.s   = SeriesStr()
    c.i64 = SeriesI64()
    c.b   = SeriesBool()

    # Length sync
    var n = len(c.f64.data)
    if c.f64.valid.len() != n:
        c.f64.valid.resize(n, True)
    return c.copy()

 
fn set_value(frame: DataFrame, row: Int, col: String, value: String) -> DataFrame:
    var out = frame.copy()
    var idx = out.col_index(col)
    
    if idx < 0 or row < 0 or row >= out.nrows(): 
        return out

    var c = out.get_column(col)
    if c.dtype() != ColumnTag.STR(): 
        return out 

    var n = c.s.len() 

    var data = List[String]()
    data.reserve(n)
    var valid = Bitmap()
    valid.resize(n, False)

    var i = 0
    while i < n:
        var v  = c.s.data[i]
        var ok = c.s.valid.is_set(i)
        if i == row:
            v  = value
            ok = True
        data.append(v)
        _ = valid.set(i, ok)
        i += 1
 

    var cnew = col_string_with_valid(out.col_names[idx], data, valid)
    out.set_column(cnew) 
    var c2 = out.get_column(col)  
    return out




fn set_value(frame: DataFrame, row: Int, col: String, value: Float64) -> DataFrame:
    var out = frame.copy()
    var idx = out.col_index(col)
    if idx < 0 or row < 0 or row >= out.nrows():
        return out

    var c = out.get_column(col)
    if c.dtype() != ColumnTag.F64():
        return out

    var n = c.f64.len()
    var data = List[Float64]()
    data.reserve(n)
    var valid = Bitmap()
    valid.resize(n, False)

    var i = 0
    while i < n:
        var v = c.f64.data[i]
        var ok = c.f64.valid.is_set(i)
        if i == row:
            v = value
            ok = not (value != value)
        data.append(v)
        _ = valid.set(i, ok)
        i += 1

    var cnew = col_f64(out.col_names[idx], data, valid)
    out.set_column(cnew)
    return out


fn set_value(frame: DataFrame, row: Int, col: String, value: Int) -> DataFrame:
    var out = frame.copy()
    var idx = out.col_index(col)
    if idx < 0 or row < 0 or row >= out.nrows():
        return out

    var c = out.get_column(col)
    if c.dtype() != ColumnTag.I64():
        return out

    var n = c.i64.len()
    var data = List[Int]()
    data.reserve(n)
    var valid = Bitmap()
    valid.resize(n, False)

    var i = 0
    while i < n:
        var v = c.i64.data[i]
        var ok = c.i64.valid.is_set(i)
        if i == row:
            v = value
            ok = True
        data.append(v)
        _ = valid.set(i, ok)
        i += 1

    var cnew = col_i64(out.col_names[idx], data, valid)
    out.set_column(cnew)
    return out


fn set_value(frame: DataFrame, row: Int, col: String, value: Bool) -> DataFrame:
    var out = frame.copy()
    var idx = out.col_index(col)
    if idx < 0 or row < 0 or row >= out.nrows():
        return out

    var c = out.get_column(col)
    if c.dtype() != ColumnTag.BOOL():
        return out

    var n = c.b.len()
    var data = List[Bool]()
    data.reserve(n)
    var valid = Bitmap()
    valid.resize(n, False)

    var i = 0
    while i < n:
        var v = c.b.data[i]
        var ok = c.b.valid.is_set(i)
        if i == row:
            v = value
            ok = True
        data.append(v)
        _ = valid.set(i, ok)
        i += 1

    var cnew = col_bool(out.col_names[idx], data, valid)
    out.set_column(cnew)
    return out
 
# --------------------- set_null (dtype-preserving) ---------------------

# Set a single cell to null without changing the column dtype.
# F64: write NaN and validity=false
# I64/BOOL/STR: validity=false (preserve data)
fn set_null(frame: DataFrame, row: Int, col: String) -> DataFrame:
    var out = frame.copy()
    var idx = out.col_index(col)
    if idx < 0 or row < 0 or row >= out.nrows(): 
        return out

    var c = out.get_column(col)
    var tag = c.dtype()

    # Float64
    if tag == ColumnTag.F64(): 

        var n = len(c.f64.data)
        var data = List[Float64]()
        data.reserve(n)
        var valid = Bitmap()
        valid.resize(n, False)

        var i = 0
        while i < n:
            var v = c.f64.data[i]
            var ok = c.f64.valid.is_set(i)
            if i == row:
                v = f64_nan()
                ok = False
            data.append(v)
            _ = valid.set(i, ok)
            i += 1

        var cnew = col_f64_with_valid(out.col_names[idx], data, valid)
        out.set_column(idx, cnew)   # or out.set_column(cnew)
        return out

    # Int64
    if tag == ColumnTag.I64(): 

        var n = len(c.i64.data)
        var data = List[Int]()
        data.reserve(n)
        var valid = Bitmap()
        valid.resize(n, False)

        var i = 0
        while i < n:
            data.append(c.i64.data[i])
            var ok = c.i64.valid.is_set(i)
            if i == row:
                ok = False
            _ = valid.set(i, ok)
            i += 1

        var cnew = col_i64_with_valid(out.col_names[idx], data, valid)
        out.set_column(idx, cnew) 
        return out

    # Bool
    if tag == ColumnTag.BOOL(): 

        var n = len(c.b.data)
        var data = List[Bool]()
        data.reserve(n)
        var valid = Bitmap()
        valid.resize(n, False)

        var i = 0
        while i < n:
            data.append(c.b.data[i])
            var ok = c.b.valid.is_set(i)
            if i == row:
                ok = False
            _ = valid.set(i, ok)
            i += 1

        var cnew = col_bool_with_valid(out.col_names[idx], data, valid)
        out.set_column(idx, cnew) 
        return out

    # String
    if tag == ColumnTag.STR(): 

        var n = len(c.s.data)
        var data = List[String]()
        data.reserve(n)
        var valid = Bitmap()
        valid.resize(n, False)

        var i = 0
        while i < n:
            data.append(c.s.data[i])
            var ok = c.s.valid.is_set(i)
            if i == row:
                ok = False
            _ = valid.set(i, ok)
            i += 1

        var cnew = col_string_with_valid(out.col_names[idx], data, valid)
        out.set_column(idx, cnew) 
        return out
 
    return out



# ---------- replace_values (String column) ----------
fn replace_values(frame: DataFrame, col: String, from_value: String, to_value: String) -> DataFrame:
    var out = frame.copy()
    var idx = out.col_index(col)
    if idx < 0:
        return out

    var c = out.get_column(col)
    if c.dtype() != ColumnTag.STR():
        return out

    var n = out.nrows()
    var data = List[String]()
    data.reserve(n)
    var valid = Bitmap()
    valid.resize(n, False)

    var r = 0
    while r < n:
        var s = c.s.data[r]
        var ok = c.s.valid.is_set(r)

        var out_s = s
        var out_ok = ok

        # Treat literal "<NULL>" as match for invalid cells
        if (ok and s == from_value) or ((not ok) and from_value == String("<NULL>")):
            out_s = to_value
            out_ok = True

        data.append(out_s)
        _ = valid.set(r, out_ok)
        r += 1

    var cnew = col_string_with_valid(out.col_names[idx], data, valid)
    out.set_column(idx, cnew) 
    return out

# -------------- coerce_str_to_f64 (robust replace by index) --------------

# ---------- coerce_str_to_f64 ----------
fn coerce_str_to_f64(frame: DataFrame, col: String) -> DataFrame:
    var out = frame.copy()
    var idx = out.col_index(col)
    if idx < 0:
        return out

    var c = out.get_column(col)
    if c.dtype() == ColumnTag.F64():
        return out
    if c.dtype() != ColumnTag.STR():
        return out
 

    # Build F64 buffers from string payload
    var n = len(c.s.data)
    var data = List[Float64]()
    data.reserve(n)
    var valid = Bitmap()
    valid.resize(n, False)

    var i = 0
    while i < n:
        var s = c.s.data[i]
        var ok = c.s.valid.is_set(i)
        if ok:
            if s == "" or s == "nan" or s == "NaN" or s == "None":
                ok = False
                data.append(f64_nan())
            else:
                var v = parse_f64_or_nan(s)
                if is_nan(v):
                    ok = False
                data.append(v)
        else:
            data.append(f64_nan())
        _ = valid.set(i, ok)
        i += 1

    var cnew = col_f64_with_valid(out.col_names[idx], data, valid)
    out.set_column(idx, cnew) 
    return out


 



# ---- Column statistics ----

# [moved] _find_col_idx
fn _find_col_idx(df0: DataFrame, name: String) -> Int:
    var i = 0
    while i < df0.ncols():
        if df0.col_names[i] == name:
            return i
        i += 1
    return -1



# ---- groupby (DataFrame result) ----

# ---- take_rows ----

# [moved] take_rows
fn take_rows(df0: DataFrame, idxs: List[Int]) -> DataFrame:
    var col_names = df0.col_names.copy()
    var cols = List[List[String]]()
    var c = 0
    while c < df0.ncols():
        var vals = List[String]()
        var i = 0
        while i < len(idxs):
            var r = idxs[i]
            if r >= 0 and r < df0.nrows():
                vals.append(df0.cols[c][r])
            else:
                vals.append(String(""))
            i += 1
        cols.append(vals.copy())
        c += 1
    return _df_make(col_names, cols)


# ---- concat_rows ----

# [moved] concat_rows
fn concat_rows(dfs: List[DataFrame], ignore_index: Bool) -> DataFrame:
# empty input -> empty DataFrame
    if len(dfs) == 0:
        return DataFrame()

# copy column names from the first frame
    var col_names = List[String]()
    var c = 0
    while c < dfs[0].ncols():
        col_names.append(String(dfs[0].col_names[c]))
        c += 1

# allocate output columns (one list per column)
    var cols = List[List[String]]()
    c = 0
    while c < len(col_names):
        cols.append(List[String]())
        c += 1

# append all rows from all input frames column-wise
    var total_rows = 0
    var di = 0
    while di < len(dfs):
        var df = dfs[di]
        var c2 = 0
        while c2 < df.ncols():
            var r = 0
            while r < df.nrows():
                cols[c2].append(df.cols[c2][r])
                r += 1
            c2 += 1
        total_rows += df.nrows()
        di += 1

# build output index
    var out_index = List[String]()
    var out_index_name = String("")
    if ignore_index:
# create a fresh 0..N-1 index as strings
        var r2 = 0
        while r2 < total_rows:
            out_index.append(String(r2))
            r2 += 1
        out_index_name = String("")
    else:
# preserve and concatenate existing indices in order
        out_index_name = String(dfs[0].index_name)
        di = 0
        while di < len(dfs):
            var idx = dfs[di].index_vals.copy()
            var k = 0
            while k < len(idx):
                out_index.append(String(idx[k]))
                k += 1
            di += 1

# construct the final DataFrame
    return DataFrame(col_names, cols, out_index, out_index_name)



# ---- concat_cols ----

# [moved] concat_cols
fn concat_cols(dfs: List[DataFrame]) -> DataFrame:
    if len(dfs) == 0:
        return DataFrame()
    var total_cols = 0
    var i = 0
    var height = dfs[0].nrows()
    while i < len(dfs):
        total_cols += dfs[i].ncols()
        if dfs[i].nrows() != height:
            print("[WARN] concat_cols: inconsistent heights")
        i += 1
    var col_names = List[String]()
    var out_cols = List[List[String]]()
    i = 0
    while i < len(dfs):
        var df = dfs[i]
        var c = 0
        while c < df.ncols():
            col_names.append(df.col_names[c])
            var vals = List[String]()
            var r = 0
            while r < df.nrows():
                vals.append(df.cols[c][r])
                r += 1
            out_cols.append(vals.copy())
            c += 1
        i += 1
    return _df_make(col_names, out_cols)


# ---- pivot_table (mean + margins) ----

# ---- Canonicalized (deduplicated) ----
fn copy(df0: DataFrame) -> DataFrame:
    return df0.copy()



# ---- Value setter (stub) ----

# [moved] set_value


# Facade pivot_table with pandas-like signature (limited: 'mean' only; margins unsupported yet)
fn pivot_table(
    frame: DataFrame,
    index: String,
    columns: String,
    values: String,
    agg: String = "mean",
    margins: Bool = False,
    margins_name: String = "Total"
) -> DataFrame:
# Guard unsupported aggregations
    if agg != String("mean"):
        print("[WARN] pivot_table: only 'mean' is supported currently; got agg=", agg)
    if margins:
        print("[WARN] pivot_table: 'margins' not implemented yet; ignoring.")
# Naive group-by implementation via compat/groupby if available
# Here we expect a function groupby_mean(frame, by1, by2, value) -> DataFrame
# We'll try to import a local helper; if missing, return frame unchanged as a fallback.
    from momijo.dataframe.selection import RowRange  # dummy import to ensure module is reachable
# Fallback: return input frame (no-op) to keep compilation
    return frame

# Map a single-character string "0".."9" to its int value without any raising.
@always_inline 
fn _digit(c: String, mut v: Int) -> Bool:
    if c == "0": v = 0;  return True
    if c == "1": v = 1;  return True
    if c == "2": v = 2;  return True
    if c == "3": v = 3;  return True
    if c == "4": v = 4;  return True
    if c == "5": v = 5;  return True
    if c == "6": v = 6;  return True
    if c == "7": v = 7;  return True
    if c == "8": v = 8;  return True
    if c == "9": v = 9;  return True
    return False

# Parse integer without any raising conversions.
fn _to_int(s: String, mut v: Int) -> Bool:
    if len(s) == 0: return False

    var neg = False
    var i = 0
    if s[0] == "-":
        neg = True
        i = 1
        if len(s) == 1: return False

    var acc = 0
    while i < len(s):
        var c = s[i:i+1]       
        var d = 0
        if not _digit(c, d):
            return False
        acc = acc * 10 + d
        i += 1

    v = -acc if neg else acc
    return True


# DataFrame-level rename
# cols_map: List[(old_name, new_name)]
fn rename(
    frame: DataFrame,
    cols_map: List[(String, String)],
    index_name: String = String("")
) -> DataFrame:
    var out = frame.copy()

# Rename columns according to cols_map
    var i = 0
    while i < out.ncols():
        var old_name = out.col_names[i]

        var j = 0
        while j < len(cols_map):
            var pair = cols_map[j]    # (old, new)
            if pair[0] == old_name:
                out.col_names[i] = pair[1]
                break
            j += 1

        i += 1

# Set index name only if provided
    if len(index_name) > 0:
        out.index_name = index_name

    return out.copy()

 
# Convert a string column to normalized ISO date strings (YYYY-MM-DD).
# If you have a real datetime dtype, replace the SeriesStr part with your datetime series.
fn to_datetime(frame: DataFrame, col: String, fmt: String = String("%Y-%m-%d")) -> DataFrame:
    var out = frame.copy()

    # Locate column index
    var j = -1
    var k = 0
    while k < out.ncols():
        if out.col_names[k] == col:
            j = k
            break
        k += 1
    if j < 0:
        # column not found: return a copy unchanged
        return out

    # Read source as strings
    var n = out.cols[j].len()
    var vals = List[String]()
    vals.reserve(n)

    var r = 0
    while r < n:
        var raw = out.cols[j].get_string(r)

        # Normalize into YYYY-MM-DD (supports "YYYY-MM-DD", "YYYY/M/D", "YYYYMMDD")
        var y = 0
        var m = 0
        var d = 0
        var ok = _parse_date_ymd(raw, y, m, d)

        if ok:
            vals.append(_iso_date(y, m, d))  # always YYYY-MM-DD with zero padding
        else:
            # Fallback: keep original (or choose to mark invalid if you track validity)
            vals.append(raw)
        r += 1

    # Rebuild a fresh SeriesStr and set it into the column (deep copy, no aliasing)
    var s = SeriesStr()
    s.set_name(col)
    s.data = vals.copy()
    # Optional: if you maintain validity bitmap, mark all True here
    # s.valid = Bitmap.full(n, True)

    # Requires your Column to have this setter (you mentioned adding it earlier)
    out.cols[j].set_string_series(s)

    # Optional: if you keep a dtype vector alongside columns, update it to "datetime"
    # out.dtypes[j] = String("datetime")

    return out


# --- Helpers ---

# Parse several simple date formats into y,m,d (returns True on success).
fn _parse_date_ymd(s: String, mut y: Int, mut m: Int, mut d: Int) -> Bool:
    var len_s = len(s)

    # Case 1: YYYY-MM-DD (exact)
    if len_s == 10 and s[4] == "-" and s[7] == "-":
        var ys = s[0:4]
        var ms = s[5:7]
        var ds = s[8:10]
        return _to_ymd_checked(ys, ms, ds, y, m, d)

    # Case 2: YYYY/M/D (single-digit month/day allowed, with '/')
    var slash1 = -1
    var slash2 = -1
    var i = 0
    while i < len_s:
        if s[i] == "/":
            if slash1 < 0:
                slash1 = i
            else:
                slash2 = i
                break
        i += 1
    if slash1 > 0 and slash2 > slash1 + 1 and slash2 < len_s - 1:
        var ys2 = s[0:slash1]
        var ms2 = s[slash1+1:slash2]
        var ds2 = s[slash2+1:len_s]
        return _to_ymd_checked(ys2, ms2, ds2, y, m, d)

    # Case 3: YYYYMMDD (8 digits)
    if len_s == 8:
        var ys3 = s[0:4]
        var ms3 = s[4:6]
        var ds3 = s[6:8]
        return _to_ymd_checked(ys3, ms3, ds3, y, m, d)

    return False


# Convert string year/month/day to ints and range-check; returns True if valid.
fn _to_ymd_checked(ys: String, ms: String, ds: String, mut y: Int, mut m: Int, mut d: Int) -> Bool:
    var yi = 0
    var mi = 0
    var di = 0

    if not _to_int(ys, yi): return False
    if not _to_int(ms, mi): return False
    if not _to_int(ds, di): return False

    if yi < 1: return False
    if mi < 1 or mi > 12: return False
    var maxd = _days_in_month(yi, mi)
    if di < 1 or di > maxd: return False

    y = yi
    m = mi
    d = di
    return True


 


# Days in month with leap-year check.
fn _days_in_month(y: Int, m: Int) -> Int:
    if m == 1 or m == 3 or m == 5 or m == 7 or m == 8 or m == 10 or m == 12:
        return 31
    if m == 4 or m == 6 or m == 9 or m == 11:
        return 30
    # February
    return 29 if _is_leap(y) else 28


fn _is_leap(y: Int) -> Bool:
    if (y % 400) == 0: return True
    if (y % 100) == 0: return False
    return (y % 4) == 0


# Format YYYY-MM-DD (zero-padded month/day).
fn _iso_date(y: Int, m: Int, d: Int) -> String:
    var ys = String(y)
    var ms = String(m)
    var ds = String(d)
    if len(ms) == 1: ms = String("0") + ms
    if len(ds) == 1: ds = String("0") + ds
    return ys + "-" + ms + "-" + ds

 



fn astype(frame: DataFrame, col: String, dtype: String) -> DataFrame:
    var out = frame.copy()
    var series = out.get_column(col)

    if dtype == "int":
        if series.is_i64():
            pass
        elif series.is_f64():
            var n = Int(series.len())
            var i = 0
            while i < n:
                if series.is_valid(i):
                    var v = Int64(series.f64.get(i))
                    series.i64.data[i] = Int(v[0])   # scalar + cast
                    _ = series.i64.valid.set(i, True)
                i += 1
            series.tag = ColumnTag.I64()
        elif series.is_str():
            var n = Int(series.len())
            var i = 0
            while i < n:
                if series.is_valid(i):
                    var str_val = series.s.get(i)
                    var ok = True
                    var v: Int = 0
                    try:
                        v = Int(Int64(str_val)[0])
                    except Error:
                        ok = False
                    if ok:
                        series.i64.data[i] = v
                        _ = series.i64.valid.set(i, True)
                    else:
                        _ = series.i64.valid.set(i, False)
                i += 1
            series.tag = ColumnTag.I64()

    elif dtype == "float":
        if series.is_f64():
            pass
        elif series.is_i64():
            var n = Int(series.len())
            var i = 0
            while i < n:
                if series.is_valid(i):
                    series.f64.data[i] = Float64(series.i64.get(i))
                    _ = series.f64.valid.set(i, True)
                i += 1
            series.tag = ColumnTag.F64()
        elif series.is_str():
            var n = Int(series.len())
            var i = 0
            while i < n:
                if series.is_valid(i):
                    var str_val = series.s.get(i)
                    var ok = True
                    var f: Float64 = 0.0
                    try:
                        f = Float64(str_val)  # raises if invalid
                    except Error:
                        ok = False
                    if ok:
                        series.f64.data[i] = f
                        _ = series.f64.valid.set(i, True)
                    else:
                        _ = series.f64.valid.set(i, False)
                i += 1
            series.tag = ColumnTag.F64()

    elif dtype == "str":
        var n = Int(series.len())
        var i = 0
        while i < n:
            if series.is_valid(i):
                series.s.data[i] = series.get_string(i)
            i += 1
        series.tag = ColumnTag.STR()

    out.set_column(series)
    return out.copy()








# Melt a DataFrame from wide to long format
fn melt(frame: DataFrame, id_vars: List[String], var_name: String, value_name: String) -> DataFrame:
    var melted = frame.melt(id_vars, var_name, value_name)  # Assume built-in melt exists
    return melted.copy()



fn cut_numeric(frame: DataFrame, col: String, bins: List[Int], labels: List[String]) -> List[String]:
    var out = List[String](); var i = 0; 
    while i < frame.nrows(): 
        out.append(labels[0]); 
        i += 1; 
    return out.copy()

# Overload: rename with Dict[String,String]
# Overload: rename with Dict[String,String]
fn rename(frame: DataFrame, cols_map: Dict[String, String], index_name: String = String("")) -> DataFrame:
    var pairs = List[(String, String)]()
    var keys = List[String]()
    for k in cols_map.keys():
        keys.append(String(k))
    # simple insertion sort for stable order
    var i = 0
    while i + 1 < len(keys):
        var j = i + 1
        while j < len(keys):
            if keys[j] < keys[i]:
                var tmp = keys[i]
                keys[i] = keys[j]
                keys[j] = tmp
            j += 1
        i += 1
    # safely get values
    i = 0
    while i < len(keys):
        var k = keys[i]
        var v = String("")
        var opt = cols_map.get(k)
        if opt is not None:
            v = opt.value()
        pairs.append((k, v))
        i += 1
    # Call the existing rename that accepts List[(String,String)]
    return rename(frame, pairs, index_name)


# -----------------------------------------------------------------------------
# Assign/replace columns
# -----------------------------------------------------------------------------

# Helper: deep-copy a List[T] when T is not implicitly copyable.
@always_inline
fn _copy_list_str(src: List[String]) -> List[String]:
    return src.copy()

@always_inline
fn _copy_list_f64(src: List[Float64]) -> List[Float64]:
    return src.copy()

@always_inline
fn _copy_list_i64(src: List[Int64]) -> List[Int64]:
    return src.copy()

@always_inline
fn _copy_list_bool(src: List[Bool]) -> List[Bool]:
    return src.copy()


# Flexible assign: accept Dict[String, Any] — not supported in Mojo yet.
# Keep a stub to avoid accidental usage.
fn assign_flex(df: DataFrame, newcols_any: Dict[String, Any]) -> DataFrame:
    # Not implemented: Mojo lacks a stable 'Any'-based reflection for lists here.
    # Prefer using typed overloads: assign(... List[String]/Float64/Int64/Bool ...)
    return df.copy()


# Typed helpers that convert to string columns and delegate to the main assign()

fn assign_numeric(df: DataFrame, newcols: Dict[String, List[Float64]]) -> DataFrame:
    var strmap = Dict[String, List[String]]()
    for k in newcols.keys():
        var ls = List[String]()
        var opt_src = newcols.get(k)
        if opt_src is not None:
            var src = _copy_list_f64(opt_src.value())
            var i = 0
            while i < len(src):
                ls.append(String(src[i]))
                i += 1
        strmap[k] = ls.copy()
    return assign(df, strmap)

fn assign_int(df: DataFrame, newcols: Dict[String, List[Int64]]) -> DataFrame:
    var strmap = Dict[String, List[String]]()
    for k in newcols.keys():
        var ls = List[String]()
        var opt_src = newcols.get(k)
        if opt_src is not None:
            var src = _copy_list_i64(opt_src.value())
            var i = 0
            while i < len(src):
                ls.append(String(src[i]))
                i += 1
        strmap[k] = ls.copy()
    return assign(df, strmap)

fn assign_bool(df: DataFrame, newcols: Dict[String, List[Bool]]) -> DataFrame:
    var strmap = Dict[String, List[String]]()
    for k in newcols.keys():
        var ls = List[String]()
        var opt_src = newcols.get(k)
        if opt_src is not None:
            var src = _copy_list_bool(opt_src.value())
            var i = 0
            while i < len(src):
                ls.append(String(src[i]))
                i += 1
        strmap[k] = ls.copy()
    return assign(df, strmap)




# Add/replace columns from a mapping: name -> values (as List[String])
fn assign(df: DataFrame, newcols: Dict[String, List[String]]) -> DataFrame:
    var out = df.copy()

    # name -> column index map
    var name_to_idx = Dict[String, Int]()
    var i = 0
    while i < len(out.col_names):
        name_to_idx[out.col_names[i]] = i
        i += 1

    # collect keys (we keep deterministic order by sorting lexicographically)
    var keys = List[String]()
    for k in newcols.keys():
        keys.append(String(k))

    # simple in-place sort (selection sort)
    i = 0
    while i + 1 < len(keys):
        var j = i + 1
        while j < len(keys):
            if keys[j] < keys[i]:
                var tmp = keys[i]
                keys[i] = keys[j]
                keys[j] = tmp
            j += 1
        i += 1

    # apply: create/replace string columns
    i = 0
    while i < len(keys):
        var name = keys[i]

        # fetch values safely
        var vals = List[String]()
        var opt_vals = newcols.get(name)
        if opt_vals is not None:
            vals = _copy_list_str(opt_vals.value())

        # build SeriesStr and Column
        var s = SeriesStr()
        s.set_name(name)
        s.data = vals.copy()  # List[String] deep-copied above

        var col = Column()
        col.from_str(s)

        # upsert into DataFrame
        var idx_opt = name_to_idx.get(name)
        if idx_opt is not None:
            var idx = idx_opt.value()
            out.cols[idx] = col.copy()
            out.col_names[idx] = name
        else:
            out.cols.append(col.copy())
            out.col_names.append(name)
        i += 1

    return out.copy()


# Overloads: allow direct typed dicts; route to the string-assign via conversion

fn assign(frame: DataFrame, newcols: Dict[String, List[Float64]]) -> DataFrame:
    return assign_numeric(frame, newcols)

fn assign(frame: DataFrame, newcols: Dict[String, List[Int64]]) -> DataFrame:
    return assign_int(frame, newcols)

fn assign(frame: DataFrame, newcols: Dict[String, List[Bool]]) -> DataFrame:
    return assign_bool(frame, newcols)


 

fn resolve_label_slice(index_labels: List[String], sel: LabelSlice) -> List[Int]:
    var out = List[Int]()

    # find start/end positions
    var n = len(index_labels)
    var i = 0
    var start_pos = -1
    var end_pos = -1

    # first occurrence scanning (left to right)
    while i < n:
        if start_pos < 0 and index_labels[i] == sel.start:
            start_pos = i
        if index_labels[i] == sel.end:
            end_pos = i
        i += 1

    if start_pos < 0 or end_pos < 0:
        # no match → empty selection
        return out

    # normalize order (support start after end)
    var lo = start_pos if start_pos <= end_pos else end_pos
    var hi = end_pos   if start_pos <= end_pos else start_pos

    if sel.inclusive:
        hi = hi
    else:
        hi = hi - 1

    if hi < lo:
        return out

    var r = lo
    while r <= hi:
        out.append(r)
        r += 1

    return out


# Thin wrapper so users can call: df.col_index(frame, "score")
fn col_index(df: DataFrame, col: String) -> Int:
    return df.col_index(col)

# Optional strict wrapper
fn require_col_index(df: DataFrame, col: String) raises -> Int:
    return df.require_col_index(col)


fn col_bool(name: String, data: List[Bool]) -> Column:
    var s = SeriesBoolT()
    s.set_name(name)
    s.data = data.copy()
    s.valid = Bitmap(len(data), True)
    var c = Column()
    c.from_bool(s)
    return c.copy()

fn col_bool(name: String, data: List[Bool], valid: Bitmap) -> Column:
    var s = SeriesBoolT()
    s.set_name(name)
    s.data = data.copy()
    s.valid = valid.copy()
    var n = len(s.data)
    if s.valid.len() != n:
        s.valid.resize(n, True)
    var c = Column()
    c.from_bool(s)
    return c.copy()

fn col_string(name: String, data: List[String]) -> Column:
    var s = SeriesStrT()
    s.set_name(name)
    s.data = data.copy()
    var n = s.len()
    s.valid.resize(n, False)
    var i = 0
    while i < n:
        var miss = (len(s.data[i]) == 0) or (s.data[i] == "NaN") or (s.data[i] == "nan")
        s.valid.set(i, not miss)
        i += 1
    var c = Column()
    c.from_str(s)
    return c.copy()

fn col_string(name: String, data: List[String], valid: Bitmap) -> Column:
    var s = SeriesStrT()
    s.set_name(name)
    s.data = data.copy()
    s.valid = valid.copy()
    var n = len(s.data)
    if s.valid.len() != n:
        s.valid.resize(n, True)
    var c = Column()
    c.from_str(s)
    return c.copy()

fn col_f64(name: String, data: List[Float64]) -> Column:
    var s = SeriesF64T()
    s.set_name(name)
    s.data = data.copy()
    s.valid = Bitmap(len(data), True)
    var c = Column()
    c.from_f64(s)
    return c.copy()

fn col_f64(name: String, data: List[Float64], valid: Bitmap) -> Column:
    var s = SeriesF64T()
    s.set_name(name)
    s.data = data.copy()
    s.valid = valid.copy()
    var n = len(s.data)
    if s.valid.len() != n:
        s.valid.resize(n, True)
    var c = Column()
    c.from_f64(s)
    return c.copy()

fn col_i64(name: String, data: List[Int]) -> Column:
    var s = SeriesI64()
    s.set_name(name)
    s.data = data.copy()
    s.valid = Bitmap(len(data), True)
    var c = Column()
    c.from_i64(s)
    return c.copy()

fn col_i64(name: String, data: List[Int], valid: Bitmap) -> Column:
    var s = SeriesI64()
    s.set_name(name)
    s.data = data.copy()
    s.valid = valid.copy()
    var n = len(s.data)
    if s.valid.len() != n:
        s.valid.resize(n, True)
    var c = Column()
    c.from_i64(s)
    return c.copy()


    # ---------------------------------------------------------------
# Construct a numeric range as a string-backed Series.
# stop is exclusive; step may be negative. No assert used.
# ---------------------------------------------------------------
fn range(start: Int, stop: Int, step: Int = 1, dtype: DType = DType.INT32()) -> List[Int]:
    var eff_step = step
    if eff_step == 0:
        if start <= stop:
            eff_step = 1
        else:
            eff_step = -1

    var out = List[Int]()
    if eff_step > 0:
        var v = start
        while v < stop:
            out.append(v)
            v += eff_step
    else:
        var v = start
        while v > stop:
            out.append(v)
            v += eff_step

    return out

# -----------------------------------------------------------------------------
# If you also need float or string ranges, add these helpers:
# -----------------------------------------------------------------------------
fn range_f64(start: Int, stop: Int, step: Int = 1) -> List[Float64]:
    var eff_step = step
    if eff_step == 0:
        if start <= stop:
            eff_step = 1
        else:
            eff_step = -1

    var out = List[Float64]()
    if eff_step > 0:
        var v = start
        while v < stop:
            out.append(Float64(v))
            v += eff_step
    else:
        var v = start
        while v > stop:
            out.append(Float64(v))
            v += eff_step
    return out

fn range_str(start: Int, stop: Int, step: Int = 1) -> List[String]:
    var eff_step = step
    if eff_step == 0:
        if start <= stop:
            eff_step = 1
        else:
            eff_step = -1

    var out = List[String]()
    if eff_step > 0:
        var v = start
        while v < stop:
            out.append(String(v))
            v += eff_step
    else:
        var v = start
        while v > stop:
            out.append(String(v))
            v += eff_step
    return out