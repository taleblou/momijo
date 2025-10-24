# Project:      Momijo 
# Module:       dataframe.missing
# File:         missing.mojo
# Path:         dataframe/missing.mojo
#
# Description:  dataframe.missing — NA/Null utilities for Momijo DataFrame.
#               fillna for common dtypes, dropna (any/all), isin helpers, and de-duplication.
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
#   - Structs: (none) — DataFrame/Column defined elsewhere.
#   - Key functions: fillna_str_col, fillna_col_i64/f64/bool, dropna_rows(_any),
#                    dropna_any, drop_duplicates, isin_i64/f64.
#   - Static methods present: N/A.

from momijo.arrow_core.poly_column import get_string 
from momijo.dataframe.column import Column
from momijo.dataframe.frame import DataFrame, width
from momijo.dataframe.helpers import isna_str
from momijo.dataframe.series_bool import append
from momijo.dataframe.api import *
from momijo.dataframe.column import BOOL, Column, ColumnTag, F64, I64, get_bool, get_f64, get_i64, name
from momijo.dataframe.bitmap import *

fn fillna_str_col(c: Column, fill: String) -> Column:
    var out = List[String]()
    var r = 0
    while r < c.len():
        var v = c[r]
        if isna_str(v): out.append(fill)
        else: out.append(v)
        r += 1
    return col_str(String("filled"), out)

# Drop any row that has an NA-like string cell in any column.
fn dropna_rows(df: DataFrame) -> DataFrame:
    var keep = List[Int]()
    var r = 0
    while r < df.nrows():
        var ok = True
        var c = 0
        while c < df.ncols():
            if isna_str(df.cols[c][r]):
                ok = False
            c += 1
        if ok:
            keep.append(r)
        r += 1
# build
    var col_names = df.col_names
    var cols = List[Column]()
    var cc = 0
    while cc < df.ncols():
        var vals = List[String]()
        var i = 0
        while i < len(keep):
            vals.append(df.cols[cc][keep[i]])
            i += 1
        cols.append(col_str(col_names[cc], vals))
        cc += 1
    return df_make(col_names, cols)


    fn fillna_col_i64(c: Column, fill: Int64) -> Column:
    var out = List[Int64]()
    var i = 0
    while i < c.len():
        if c.tag() == ColumnTag.I64():
            out.append(c.get_i64(i))
        else:
            out.append(Int64(0))
        i += 1
    return col_i64(c.name(), out)
fn fillna_col_f64(c: Column, fill: Float64) -> Column:
    var out = List[Float64]()
    var i = 0
    while i < c.len():
        if c.tag() == ColumnTag.F64():
            out.append(c.get_f64(i))
        else:
            out.append(0.0)
        i += 1
    return col_f64(c.name(), out)
fn fillna_col_bool(c: Column, fill: Bool) -> Column:
    var out = List[Bool]()
    var i = 0
    while i < c.len():
        if c.tag() == ColumnTag.BOOL():
            out.append(c.get_bool(i))
        else:
            out.append(fill)
        i += 1
    return col_bool(c.name(), out)

# Drop any row that has String-NA in any column (demo-level NA semantics)
fn dropna_rows_any(df: DataFrame) -> DataFrame:
    var keep = List[Int]()
    var r = 0
    while r < df.nrows():
        var ok = True
        var c = 0
        while c < df.ncols():
            var v = df.cols[c][r]
            if isna_str(v):
                ok = False
            c += 1
        if ok:
            keep.append(r)
        r += 1
    var col_names = df.col_names
    var cols = List[Column]()
    var cc = 0
    while cc < df.ncols():
        var vals = List[String]()
        var i = 0
        while i < len(keep):
            vals.append(df.cols[cc][keep[i]])
            i += 1
        cols.append(col_str(col_names[cc], vals))
        cc += 1
    return df_make(col_names, cols)
fn isin_i64(xs: List[Int64], universe: List[Int64]) -> List[Bool]:
    var out = List[Bool]()
    var i = 0
    while i < len(xs):
        var found = False
        var j = 0
        while j < len(universe):
            if xs[i] == universe[j]:
                found = True
                break
            j += 1
        out.append(found)
        i += 1
    return out
fn isin_f64(xs: List[Float64], universe: List[Float64]) -> List[Bool]:
    var out = List[Bool]()
    var i = 0
    while i < len(xs):
        var found = False
        var j = 0
        while j < len(universe):
            if xs[i] == universe[j]:
                found = True
                break
            j += 1
        out.append(found)
        i += 1
    return out

fn dropna_any(df0: DataFrame) -> DataFrame:
    var nrows = df0.nrows()
    var ncols = df0.ncols()

    var keep = List[Bool]()
    keep.reserve(nrows)

    # ---------- 1) Build keep mask (type-aware) ----------
    var r = 0
    while r < nrows:
        var any_missing = False

        var c = 0
        while c < ncols:
            var name = df0.col_names[c]
            var col  = df0.get_column(name)

            var miss = False

            if col.dtype() == ColumnTag.STR():
                # missing if invalid OR user-style "NaN" string
                var is_valid = col.s.valid.is_set(r)
                miss = (not is_valid) or (col.s.data[r] == String("NaN"))

            elif col.dtype() == ColumnTag.F64():
                var is_valid = col.f64.valid.is_set(r)
                var v = col.f64.data[r]
                # missing if invalid OR IEEE NaN
                miss = (not is_valid) or (v != v)

            elif col.dtype() == ColumnTag.I64():
                miss = not col.i64.valid.is_set(r)

            elif col.dtype() == ColumnTag.BOOL():
                miss = not col.b.valid.is_set(r)

            # else: treat unknown types as non-missing
            if miss:
                any_missing = True
                break
            c += 1

        keep.append(not any_missing)
        r += 1

    # ---------- 2) Rebuild filtered frame ----------
    var out = df0.copy()

    var c2 = 0
    while c2 < ncols:
        var name = out.col_names[c2]
        var col  = out.get_column(name)

        if col.dtype() == ColumnTag.STR():
            # preserve both data and validity for kept rows
            var data  = List[String]()
            var valid = Bitmap()
            data.reserve(nrows)
            valid.resize(0, False)        # we'll resize to exact length later

            var i = 0
            while i < nrows:
                if keep[i]:
                    data.append(col.s.data[i])
                i += 1

            # now build validity for the kept rows
            valid.resize(len(data), False)
            var j = 0
            var i2 = 0
            while i2 < nrows:
                if keep[i2]:
                    _ = valid.set(j, col.s.valid.is_set(i2))
                    j += 1
                i2 += 1
 
            var cnew = col_string_with_valid(name, data, valid)
            out.set_column(cnew)

        elif col.dtype() == ColumnTag.F64():
            var data = List[Float64]()
            data.reserve(nrows)

            var i = 0
            while i < nrows:
                if keep[i]:
                    data.append(col.f64.data[i])
                i += 1
 
            var valid = Bitmap()
            valid.resize(len(data), True)
            var cnew = col_f64(name, data, valid)
            out.set_column(cnew)

        elif col.dtype() == ColumnTag.I64():
            var data = List[Int]()
            data.reserve(nrows)

            var i = 0
            while i < nrows:
                if keep[i]:
                    data.append(col.i64.data[i])
                i += 1

            var valid = Bitmap()
            valid.resize(len(data), True)
            var cnew = col_i64(name, data, valid)
            out.set_column(cnew)

        elif col.dtype() == ColumnTag.BOOL():
            var data = List[Bool]()
            data.reserve(nrows)

            var i = 0
            while i < nrows:
                if keep[i]:
                    data.append(col.b.data[i])
                i += 1

            var valid = Bitmap()
            valid.resize(len(data), True)
            var cnew = col_bool(name, data, valid)
            out.set_column(cnew)

        # else: ignore unknown dtypes for now

        c2 += 1

    return out


fn drop_duplicates(df: DataFrame, subset: List[String], keep: String = "first") -> DataFrame:
    # ---------- normalize 'keep' ----------
    var keep_norm = keep
    if keep_norm == String("FIRST"): keep_norm = String("first")
    if keep_norm == String("Last"):  keep_norm = String("last")
    if keep_norm == String("NONE") or keep_norm == String("all"):
        keep_norm = String("none")

    # ---------- choose subset columns ----------
    var subset_idx = List[Int]()
    if len(subset) == 0:
        var c = 0
        while c < df.ncols():
            subset_idx.append(c)
            c += 1
    else:
        var j = 0
        while j < len(subset):
            var idx = df.find_col(subset[j])
            if idx >= 0:
                subset_idx.append(idx)
            j += 1
        if len(subset_idx) == 0:
            var c2 = 0
            while c2 < df.ncols():
                subset_idx.append(c2)
                c2 += 1

    # ---------- scan rows and apply keep policy ----------
    var seen_keys = List[String]()
    var last_index_for = List[Int]()
    var keep_rows = List[Bool]()

    var r = 0
    while r < df.nrows():
        var key = String("")
        var t = 0
        while t < len(subset_idx):
            key = key + String("|") + df.cols[subset_idx[t]][r]
            t += 1

        var pos = -1
        var s = 0
        while s < len(seen_keys):
            if seen_keys[s] == key:
                pos = s
                break
            s += 1

        if pos < 0:
            seen_keys.append(key)
            last_index_for.append(r)
            keep_rows.append(True)
        else:
            if keep_norm == String("first"):
                keep_rows.append(False)
            elif keep_norm == String("last"):
                var prev = last_index_for[pos]
                keep_rows[prev] = False
                keep_rows.append(True)
                last_index_for[pos] = r
            else:
                var prev2 = last_index_for[pos]
                keep_rows[prev2] = False
                keep_rows.append(False)
                last_index_for[pos] = r
        r += 1

    # ---------- collect kept row indices ----------
    var kept = List[Int]()
    var i = 0
    while i < len(keep_rows):
        if keep_rows[i]:
            kept.append(i)
        i += 1

    # ---------- build output frame (DataFrame() + set fields + set_column) ----------
    var out = DataFrame()
    out.index_name = String(df.index_name)
 
    var new_index = List[String]()
    var ii = 0
    while ii < len(kept) and ii < len(df.index_vals):
        new_index.append(df.index_vals[kept[ii]])
        ii += 1 
    if len(new_index) > 0:
        out.index_vals = new_index.copy()
 
    var c3 = 0
    while c3 < df.ncols():
        var vals = List[String]()
        var k = 0
        while k < len(kept):
            vals.append(df.cols[c3][kept[k]])
            k += 1
        var col = col_str(df.col_names[c3], vals)
        out.set_column(col)   # overload: set_column(mut self, col: Column)
        c3 += 1

    return out



# Drop rows with any NA
fn _dropna_rows(df0: DataFrame) -> DataFrame:
    var new_cols = List[List[String]]()
    var ncols = df0.ncols()

    var c = 0
    while c < ncols:
        new_cols.append(List[String]())
        c += 1

    var new_index = List[String]()

    var r = 0
    while r < df0.nrows():
        var has_na = False
        var c2 = 0
        while c2 < ncols:
            if _isna(df0.cols[c2][r]):
                has_na = True
                break
            c2 += 1

        if not has_na:
            c2 = 0
            while c2 < ncols:
                new_cols[c2].append(df0.cols[c2][r])
                c2 += 1
            if len(df0.index_vals) > r:
                new_index.append(df0.index_vals[r])
        r += 1

    return DataFrame(df0.col_names, new_cols, new_index, df0.index_name)


# ---------- fillna_value(String) ----------
fn fillna_value(frame: DataFrame, col: String, value: String) -> DataFrame:
    var out = frame.copy()
    var idx = out.col_index(col)
    if idx < 0:
        return out

    var c = out.get_column(col)
    if c.dtype() != ColumnTag.STR(): 
        return out
 

    var s = c.s.copy()
    var n = s.len()
    var filled = 0

    var i = 0
    while i < n:
        var is_null = not s.valid.is_set(i)
        var is_nan_text = (s.data[i] == String("NaN"))
        if is_null or is_nan_text:
            s.data[i] = value
            _ = s.valid.set(i, True)
            filled += 1
        i += 1 

    # Build a new Column explicitly and replace by index
    var cnew = col_string_with_valid(out.col_names[idx], s.data, s.valid)
    out.set_column(idx, cnew) 
    return out


# ---------- fillna_value(Float64) ----------
fn fillna_value(frame: DataFrame, col: String, value: Float64) -> DataFrame:
    # Ensure F64 type if it's currently string
    var out = coerce_str_to_f64(frame, col)
    var idx = out.col_index(col)
    if idx < 0:
        return out

    var c = out.get_column(col)
    if c.dtype() != ColumnTag.F64():
        return out

    var n = c.f64.len()
    var data = List[Float64]()
    data.reserve(n)
    var valid = Bitmap()
    valid.resize(n, False)

    var filled = 0
    var i = 0
    while i < n:
        var ok = c.f64.valid.is_set(i)
        var v  = c.f64.data[i]
        # Treat NaN as null regardless of 'ok'
        if (not ok) or is_nan(v):
            v = value
            ok = True
            filled += 1
        data.append(v)
        _ = valid.set(i, ok)
        i += 1
 

    var cnew = col_f64_with_valid(out.col_names[idx], data, valid)
    out.set_column(idx, cnew) 
    return out


# ---------- fillna_value(Int) ----------
fn fillna_value(frame: DataFrame, col: String, value: Int) -> DataFrame:
    var out = frame.copy()
    var idx = out.col_index(col)
    if idx < 0:
        return out

    var c = out.get_column(col)
    if c.dtype() != ColumnTag.I64(): 
        return out
 

    var s = c.i64.copy()
    var n = s.len()
    var filled = 0

    var i = 0
    while i < n:
        if not s.valid.is_set(i):
            s.data[i] = value
            _ = s.valid.set(i, True)
            filled += 1
        i += 1
 

    # Reuse the same Column shell and then replace by index
    c.from_i64(s)
    out.set_column(idx, c) 
    return out


# ---------- fillna_value(Bool) ----------
fn fillna_value(frame: DataFrame, col: String, value: Bool) -> DataFrame:
    var out = frame.copy()
    var idx = out.col_index(col)
    if idx < 0:
        return out

    var c = out.get_column(col)
    if c.dtype() != ColumnTag.BOOL(): 
        return out
 

    var s = c.b.copy()
    var n = s.len()
    var filled = 0

    var i = 0
    while i < n:
        if not s.valid.is_set(i):
            s.data[i] = value
            _ = s.valid.set(i, True)
            filled += 1
        i += 1 
    c.from_bool(s)
    out.set_column(idx, c) 
    return out






# ---------- ffill (string cells, using textual storage) ----------
fn ffill(frame: DataFrame, col: String, limit: Int = 1) -> DataFrame:
    var out = frame.copy()
    var idx = out.col_index(col)
    if idx < 0:
        return out

    var last = String("")
    var have_last = False
    var fill_count = 0

    var r = 0
    var n = out.nrows()
    while r < n:
        var s = out.cols[idx][r]
        if not _is_missing_cell(s):
            last = s
            have_last = True
            fill_count = 0
        elif have_last and (limit <= 0 or fill_count < limit):
            out = set_value(out, row=r, col=col, value=last)
            fill_count += 1
        r += 1
    return out






# ---------- interpolate_numeric (works over textual storage) ----------
fn interpolate_numeric(frame: DataFrame, col: String) -> DataFrame:
    var out = frame.copy()
    var idx = out.col_index(col)
    if idx < 0:
        return out

    var xs = List[Int]()
    var ys = List[Float64]()
    var i = 0
    while i < out.nrows():
        var s = out.cols[idx][i]
        if not _is_missing_cell(s):
            var ok = True
            var v = 0.0
            try:
                v = Float64(s)
            except _:
                ok = False
            if ok:
                xs.append(i)
                ys.append(v)
        i += 1

    if len(xs) < 2:
        return out

    var seg = 0
    while seg + 1 < len(xs):
        var x0 = xs[seg]
        var y0 = ys[seg]
        var x1 = xs[seg + 1]
        var y1 = ys[seg + 1]
        var dx = Float64(x1 - x0)
        var t = 1
        while x0 + t < x1:
            var xi = x0 + t
            if _is_missing_cell(out.cols[idx][xi]):
                var alpha = Float64(t) / dx
                var yi = y0 + (y1 - y0) * alpha
                out = set_value(out, row=xi, col=col, value=yi)
            t += 1
        seg += 1
    return out



fn _is_missing_col(col: Column, i: Int) -> Bool:
    if col.dtype() == ColumnTag.STR():
        var n = col.s.len()
        if i < 0 or i >= n:
            return True
        if not col.s.valid.is_set(i):
            return True
        var v = col.s.data[i]
        if len(v) == 0:
            return True
        if v == "NaN" or v == "nan":
            return True
        return False

    elif col.dtype() == ColumnTag.F64():
        var n = col.f64.len()
        if i < 0 or i >= n:
            return True
        if not col.f64.valid.is_set(i):
            return True
        var v = col.f64.data[i]
        return v != v

    elif col.dtype() == ColumnTag.I64():
        var n = col.i64.len()
        if i < 0 or i >= n:
            return True
        return not col.i64.valid.is_set(i)

    elif col.dtype() == ColumnTag.BOOL():
        var n = col.b.len()
        if i < 0 or i >= n:
            return True
        return not col.b.valid.is_set(i)

    return True




fn _is_missing_cell(s: String) -> Bool:
    if len(s) == 0:
        return True 
    if s == "NaN" or s == "nan" or s == "NaN ":
        return True
    return False
  

# NA detection and summary

fn _is_missing_str(s: SeriesStr, i: Int) -> Bool:
    if not s.valid.is_set(i):
        return True
    var v = s.data[i]
    if len(v) == 0:
        return True
    if v == "NaN" or v == "nan":
        return True
    return False

fn _is_missing_f64(s: SeriesF64, i: Int) -> Bool:
    if not s.valid.is_set(i):
        return True
    var v = s.data[i]
    return v != v

fn _is_missing_i64(s: SeriesI64, i: Int) -> Bool:
    return not s.valid.is_set(i)

fn _is_missing_bool(s: SeriesBool, i: Int) -> Bool:
    return not s.valid.is_set(i)

fn isna_count_by_col(df0: DataFrame) -> String:
    var out = String("")
    var c = 0
    while c < df0.ncols():
        var name = df0.col_names[c]
        var col = df0.get_column(name)
        var cnt = 0
        var r = 0
        var n = df0.nrows()
        while r < n:
            if _is_missing_col(col, r):
                cnt += 1
            r += 1
        out += name + ": " + String(cnt)
        if c + 1 < df0.ncols():
            out += "\n"
        c += 1
    return out

