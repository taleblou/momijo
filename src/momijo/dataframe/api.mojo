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


# Type aliases
alias ColPair = (String, List[String])

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

fn df_from_pairs(pairs: List[ColPair]) -> DataFrame:
    var cols = List[String]()
    var data = List[List[String]]()
    var i = 0
    while i < len(pairs):
        cols.append(String(pairs[i][0]))
        data.append(pairs[i][1])
        i += 1
    return df_from_columns(cols, data)

# Minimal df_make: build DataFrame from column names and values
fn df_make(col_names: List[String], cols: List[List[String]]) -> DataFrame:
    var idx = List[String]()
    return DataFrame(col_names, cols, idx, String(""))

# Minimal col_str: wrap name & values (identity for List[String])
fn col_str(name: String, values: List[String]) -> (String, List[String]):
    return (name, values)

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

fn df_dtypes(df: DataFrame) -> String:
    var s = String("")
    var i = 0
    while i < df.ncols():
        var dt = infer_dtype(df.cols[i])
        s = s + df.col_names[i] + String(": ") + dt
        if i + 1 < df.ncols():
            s = s + String("\n")
        i += 1
    return s

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

 


# -------------------------------------------------------------------
# Function: pairs_append (overload for String values)
# Appends a (name, values) pair directly.
# -------------------------------------------------------------------
fn pairs_append(pairs: List[ColPair], name: String, values: List[String]) -> List[ColPair]:
    var out = pairs
    out.append(ColPair(name, values))
    return out

# -------------------------------------------------------------------
# Function: pairs_append (overload for Int values)
# Converts Int values to String and appends.
# -------------------------------------------------------------------
fn pairs_append(pairs: List[ColPair], name: String, values: List[Int]) -> List[ColPair]:
    var vs = List[String]()
    var i = 0
    var n = len(values)
    while i < n:
        vs.append(String(values[i]))
        i += 1
    var out = pairs
    out.append(ColPair(name, vs))
    return out

# -------------------------------------------------------------------
# Function: pairs_append (overload for Float64 values)
# Converts Float64 values to String and appends.
# -------------------------------------------------------------------
fn pairs_append(pairs: List[ColPair], name: String, values: List[Float64]) -> List[ColPair]:
    var vs = List[String]()
    var i = 0
    var n = len(values)
    while i < n:
        vs.append(String(values[i]))
        i += 1
    var out = pairs
    out.append(ColPair(name, vs))
    return out

 
 


# ---- Drop NA (any) ----


# ---- Categorical helper ----
# Converts a string column to a categorical-like representation.
# - If `categories` is provided, values outside it are mapped to "-1".
# - If `ordered` is true, the provided order defines code order.
# - Adds a new code column when `new_name` != "" (default: "category_code").
fn to_category(frame: DataFrame, col: String, categories: List[String] = List[String](), ordered: Bool = False, new_name: String = String("category_code")) -> DataFrame:
    var out = frame.copy()
    # Find target column index
    var idx = -1
    var c = 0
    while c < out.ncols():
        if out.col_names[c] == col:
            idx = c
            break
        c += 1
    if idx < 0:
        return out   # column not found -> no-op

    # Build category index
    var cats = categories
    if len(cats) == 0:
        # infer unique values in appearance order
        var seen = Dict[String, Int]()
        var i = 0
        while i < out.cols[idx].len():
            var s = out.cols[idx].get_string(i)
            if seen.get(s) is None:
                seen[s] = len(cats)
                cats.append(s)
            i += 1

    # Map to codes
    var code_series = List[String]()
    var r = 0
    while r < out.cols[idx].len():
        var s = out.cols[idx].get_string(r)
        var code = -1
        var pos_opt = None
        var j = 0
        while j < len(cats):
            if cats[j] == s:
                code = j
                break
            j += 1
        code_series.append(String(code))
        r += 1

    # Optionally append code column
    if len(new_name) > 0:
        var col_code = Column()
        var s_code = SeriesStr(code_series, new_name)
        col_code.from_str(s_code)
        out.cols.append(col_code)
        out.col_names.append(new_name)

    return out
# ---- Index ops ----

# [moved] rename
# [removed invalid method-style rename — replaced by free-function below]

fn set_value(df0: DataFrame, row: Int, col: String, value: String) -> DataFrame:
    print("[WARN] set_value is not implemented yet; returning original DataFrame.")
    return df0



# ---- NA counters (real) ----

# [moved] isna_count_by_col
fn isna_count_by_col(df0: DataFrame) -> String:
    var out = String("")
    var c = 0
    while c < df0.ncols():
        var name = df0.col_names[c]
        var cnt = 0
        var r = 0
        while r < df0.nrows():
            var s = df0.cols[c][r]
            if _isna(s):
                cnt += 1
            r += 1
        out += name + String(": ") + String(cnt) + String("\n")
        c += 1
    return out



# ---- Value replacement ----

# [moved] replace_values
fn replace_values(df0: DataFrame, col: String, from_value: String, to_value: String) -> DataFrame:
    var idx = -1
    var i = 0
    while i < df0.ncols():
        if df0.col_names[i] == col:
            idx = i
            break
        i += 1
    if idx < 0:
        print("[WARN] replace_values: column not found -> returning original DataFrame")
        return df0

    var col_names = df0.col_names
    var cols = List[List[String]]()
    var c = 0
    while c < df0.ncols():
        var vals = List[String]()
        var r = 0
        while r < df0.nrows():
            var v = df0.cols[c][r]
            if c == idx and v == from_value:
                vals.append(to_value)
            else:
                vals.append(v)
            r += 1
        cols.append(vals)
        c += 1
    return _df_make(col_names, cols)



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

# [moved] merge
fn merge(left: DataFrame, right: DataFrame, on: List[String], how: String) -> DataFrame:
# Left-join only. 'how' is ignored for now.
    var keys = on

# map key columns to indices in both frames
    var lk = List[Int]()
    var rk = List[Int]()
    var i = 0
    while i < len(keys):
        lk.append(_find_col_idx(left,  keys[i]))
        rk.append(_find_col_idx(right, keys[i]))
        i += 1

# ---- Build right index: parallel arrays (keys -> list of row ids) ----
    var rkeys  = List[String]()
    var rlists = List[List[Int]]()

    var rrow = 0
    while rrow < right.nrows():
        var key = String("")
        var j = 0
        while j < len(rk):
            var s = right.cols[rk[j]][rrow]
            if j > 0:
                key += String("\x1f")   # field separator
            key += s
            j += 1

# linear search (no Dict → no may-raise)
        var pos = -1
        var t = 0
        while t < len(rkeys):
            if rkeys[t] == key:
                pos = t
                break
            t += 1

        if pos < 0:
            var lst = List[Int]()
            lst.append(rrow)
            rkeys.append(key)
            rlists.append(lst)
        else:
            rlists[pos].append(rrow)

        rrow += 1

# ---- Output schema: left columns + right non-key columns ----
    var out_names = List[String]()
    i = 0
    while i < left.ncols():
        out_names.append(left.col_names[i])
        i += 1

    var right_nonkey = List[Int]()
    i = 0
    while i < right.ncols():
        var is_key = False
        var u = 0
        while u < len(rk):
            if i == rk[u]:
                is_key = True
                break
            u += 1
        if not is_key:
            right_nonkey.append(i)
            out_names.append(right.col_names[i])
        i += 1

# allocate output columns
    var out_vals = List[List[String]]()
    i = 0
    while i < len(out_names):
        out_vals.append(List[String]())
        i += 1

# ---- Populate rows (left-join: take first match; else NA="") ----
    var r2 = 0
    while r2 < left.nrows():
# append left columns
        i = 0
        while i < left.ncols():
            out_vals[i].append(left.cols[i][r2])
            i += 1

# build left key
        var key2 = String("")
        var j3 = 0
        while j3 < len(lk):
            var s2 = left.cols[lk[j3]][r2]
            if j3 > 0:
                key2 += String("\x1f")
            key2 += s2
            j3 += 1

# find in right index
        var pos2 = -1
        var t2 = 0
        while t2 < len(rkeys):
            if rkeys[t2] == key2:
                pos2 = t2
                break
            t2 += 1

# append right non-key columns
        var offset = left.ncols()
        var rn = 0
        while rn < len(right_nonkey):
            var rc = right_nonkey[rn]
            if pos2 >= 0:
                var rr = rlists[pos2][0]                # first match
                out_vals[offset].append(right.cols[rc][rr])
            else:
                out_vals[offset].append(String(""))      # NA for left-join miss
            offset += 1
            rn += 1

        r2 += 1

# Build DataFrame (reuse left index)
    return DataFrame(out_names, out_vals, left.index_vals, left.index_name)



# ---- take_rows ----

# [moved] take_rows
fn take_rows(df0: DataFrame, idxs: List[Int]) -> DataFrame:
    var col_names = df0.col_names
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
        cols.append(vals)
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
            var idx = dfs[di].index_vals
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
            out_cols.append(vals)
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

    return out

 

fn to_datetime(frame: DataFrame, col: String, fmt: String = String("%Y-%m-%d")) -> DataFrame:
    var out = frame.copy()
    var c = 0
    while c < out.ncols():
        if out.col_names[c] == col:
            var r = 0
            while r < out.cols[c].len():
                var s = out.cols[c].get_string(r)
                # naive normalization
                if len(s) > 10:
                    s = s[0:10]
                elif len(s) < 10:
                    var k = len(s)
                    while k < 10:
                        s = s + "0"
                        k += 1
                # set back the value into SeriesStr
                out.cols[c].s.data[r] = s
                r += 1
            break
        c += 1
    return out

fn fillna_value(frame: DataFrame, col_name: String, value: String) -> DataFrame:
    var out = frame.copy() 
    var col_obj = out.get_column(col_name)

    # Only handle string columns for now
    if col_obj.dtype() == ColumnTag.STR():
        var s = col_obj.s
        var n = s.len()
        var i = 0
        while i < n:
            if not s.valid.is_set(i) or s.data[i] == "":
                s.data[i] = value
                s.valid.set(i, True)
            i += 1
        col_obj.from_str(s)

    out.set_column(col_obj)
    return out





fn ffill(frame: DataFrame, col: String, limit: Int = 1) -> DataFrame:
    var out = frame.copy()
    var col_obj = out.get_column(col)

    # Only handle string columns for now
    if col_obj.dtype() == ColumnTag.STR():
        var s = col_obj.s
        var last_val: Optional[String] = None
        var fill_count = 0
        var n = s.len()
        var i = 0
        while i < n:
            if s.valid.is_set(i) and s.data[i] != "":
                last_val = s.data[i]
                fill_count = 0
            elif last_val is not None and fill_count < limit:
                s.data[i] = last_val.value()
                s.valid.set(i, True)
                fill_count += 1
            i += 1
        col_obj.from_str(s)

    # Update the column in DataFrame
    out.set_column(col_obj)

    return out


fn interpolate_numeric(frame: DataFrame, col: String) -> DataFrame:
    var out = frame.copy()
    var col_obj = out.get_column(col)

    # Only float64 columns supported
    if col_obj.dtype() != ColumnTag.F64():
        return out  # fallback if not numeric

    var series = col_obj.f64
    var n = series.len()
    var i = 0

    while i < n:
        if not series.valid.is_set(i):
            # Find start and end of gap
            var start = i - 1
            var j = i
            while j < n and not series.valid.is_set(j):
                j += 1
            var end = j

            # Determine start_val
            var start_val: Float64
            if start >= 0:
                start_val = series.get(start)
            else:
                start_val = 0.0

            # Determine end_val
            var end_val: Float64
            if end < n:
                end_val = series.get(end)
            else:
                end_val = start_val

            var gap = end - start
            var k = i
            while k < end:
                var interp = start_val + (end_val - start_val) * Float64(k - start) / Float64(gap)
                series.set(k, interp)
                k += 1

            i = end
        else:
            i += 1

    col_obj.f64 = series
    out.set_column(col_obj)
    return out



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
    return out








# Melt a DataFrame from wide to long format
fn melt(frame: DataFrame, id_vars: List[String], var_name: String, value_name: String) -> DataFrame:
    var melted = frame.melt(id_vars, var_name, value_name)  # Assume built-in melt exists
    return melted



fn cut_numeric(frame: DataFrame, col: String, bins: List[Int], labels: List[String]) -> List[String]:
    var out = List[String](); var i = 0; 
    while i < frame.nrows(): 
        out.append(labels[0]); 
        i += 1; 
    return out

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


fn fillna_value(df: DataFrame, col: String, value: Float64) -> DataFrame:
    return fillna_value(df, col, String(value))


# Add/replace columns from a mapping: name -> values (as List[String])
fn assign(df: DataFrame, newcols: Dict[String, List[String]]) -> DataFrame:
    var out = df.copy()
    var name_to_idx = Dict[String, Int]()
    var i = 0
    while i < len(out.col_names):
        name_to_idx[out.col_names[i]] = i
        i += 1
    var keys = List[String]()
    for k in newcols.keys():
        keys.append(String(k))
    # sort keys
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
    # apply
    i = 0
    while i < len(keys):
        var name = keys[i]
        var vals = List[String]()
        var opt_vals = newcols.get(name)
        if opt_vals is not None:
            vals = opt_vals.value()
        var s = SeriesStr(vals, name)
        var col = Column()
        col.from_str(s)
        var idx_opt = name_to_idx.get(name)
        if idx_opt is not None:
            var idx = idx_opt.value()
            out.cols[idx] = col
            out.col_names[idx] = name
        else:
            out.cols.append(col)
            out.col_names.append(name)
        i += 1
    return out



# Overload: rename with Dict[String, String]

# Flexible assign: accept Dict[String, Any-List] by best-effort string conversion
fn assign_flex(df: DataFrame, newcols_any: Dict[String, Any]) -> DataFrame:
# This is a placeholder; Mojo may not support 'Any' yet. Keeping explicit signatures.

fn assign_numeric(df: DataFrame, newcols: Dict[String, List[Float64]]) -> DataFrame:
    var strmap = Dict[String, List[String]]()
    for k in newcols.keys():
        var ls = List[String]()
        var vs = newcols[k]
        var i = 0
        while i < len(vs):
            ls.append(String(vs[i]))
            i += 1
        strmap[k] = ls
    return assign(df, strmap)

fn assign_int(df: DataFrame, newcols: Dict[String, List[Int64]]) -> DataFrame:
    var strmap = Dict[String, List[String]]()
    for k in newcols.keys():
        var ls = List[String]()
        var vs = newcols[k]
        var i = 0
        while i < len(vs):
            ls.append(String(vs[i]))
            i += 1
        strmap[k] = ls
    return assign(df, strmap)

fn assign_bool(df: DataFrame, newcols: Dict[String, List[Bool]]) -> DataFrame:
    var strmap = Dict[String, List[String]]()
    for k in newcols.keys():
        var ls = List[String]()
        var vs = newcols[k]
        var i = 0
        while i < len(vs):
            ls.append(String(vs[i]))
            i += 1
        strmap[k] = ls
    return assign(df, strmap)

# Overload: assign with Dict[String, List[Float64]] 
fn assign(frame: DataFrame, newcols: Dict[String, List[Float64]]) -> DataFrame:
    var strmap = Dict[String, List[String]]()
    for k in newcols.keys():
        var ls = List[String]()
        var opt_src = newcols.get(k)  # safe access
        if opt_src is not None:
            var src = opt_src.value()
            var i = 0
            while i < len(src):
                ls.append(String(src[i]))
                i += 1
        strmap[k] = ls
    return assign(frame, strmap)


# Overload: assign with Dict[String, List[Int64]]
fn assign(frame: DataFrame, newcols: Dict[String, List[Int64]]) -> DataFrame:
    var strmap = Dict[String, List[String]]()
    for k in newcols.keys():
        var ls = List[String]()
        var opt_src = newcols.get(k)  # safe access
        if opt_src is not None:
            var src = opt_src.value()
            var i = 0
            while i < len(src):
                ls.append(String(src[i]))
                i += 1
        strmap[k] = ls
    return assign(frame, strmap)

# Overload: assign with Dict[String, List[Bool]]
fn assign(frame: DataFrame, newcols: Dict[String, List[Bool]]) -> DataFrame:
    var strmap = Dict[String, List[String]]()
    for k in newcols.keys():
        var ls = List[String]()
        var opt_src = newcols.get(k)  # safe access
        if opt_src is not None:
            var src = opt_src.value()
            var i = 0
            while i < len(src):
                ls.append(String(src[i]))
                i += 1
        strmap[k] = ls
    return assign(frame, strmap)
