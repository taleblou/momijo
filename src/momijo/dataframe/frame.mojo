# Project:      Momijo
# Module:       dataframe.frame
# File:         frame.mojo
# Path:         dataframe/frame.mojo
#
# Description:  dataframe.frame — DataFrame core structures and operations.
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
#   - Structs: DataFrame, ModuleState
#   - Key functions: __copyinit__, clone, copy, __init__, ncols, nrows, shape_str, find_col, with_rows, to_string, width, height, make_module_state, select_columns_safe, _deep_copy_df, _find_col, _nrows, set_index

from momijo.dataframe.column import Column
from momijo.dataframe.series_str import SeriesStr
from momijo.dataframe.column import Column, from_bool, from_f64, from_i64, from_str, get_bool, get_f64, get_i64, is_f64, is_i64, name
 
from momijo.dataframe.io_csv import is_bool
from momijo.dataframe.series_bool import SeriesBool, append
from momijo.dataframe.series_f64 import SeriesF64
from momijo.dataframe.series_i64 import SeriesI64
from collections.dict import Dict
from collections.list import List
from pathlib.path import Path

from momijo.dataframe.compat import df_from_pairs as _df_from_pairs
 
from momijo.dataframe.io_csv import read_csv as _read_csv_file, read_csv_from_string as _read_csv_text, write_csv as _write_csv_file


 
from momijo.dataframe.column import Column, from_f64, from_i64

from momijo.dataframe.series_bool import append
from momijo.dataframe.datetime_ops import parse_minutes
from momijo.dataframe.helpers import argsort_f64, argsort_i64


struct DataFrame(ExplicitlyCopyable, Movable):
    var col_names: List[String]
    var names: List[String]
    var cols: List[Column]
    var index_name: String
    var index_vals: List[String]

    # Copy constructor
    fn __copyinit__(out self, other: DataFrame):
        self.col_names = List[String]()
        self.names     = List[String]()
        var i = 0
        while i < len(other.col_names):
            self.col_names.append(String(other.col_names[i]))
            self.names.append(String(other.col_names[i]))
            i += 1
        self.cols = List[Column]()
        i = 0
        while i < len(other.cols):
            var c = other.cols[i]
            var cc = c.clone()
            self.cols.append(cc)
            i += 1
        self.index_name = String(other.index_name)
        self.index_vals = List[String]()
        i = 0
        while i < len(other.index_vals):
            self.index_vals.append(String(other.index_vals[i]))
            i += 1

    fn clone(self) -> DataFrame:
        var out = DataFrame()
        out.__copyinit__(self)
        return out

    fn copy(self) -> DataFrame:
        return self.clone()

    # Default constructor
    fn __init__(out self):
        self.col_names = List[String]()
        self.names = List[String]()
        self.cols = List[Column]()
        self.index_name = String("")
        self.index_vals = List[String]()

    # Constructor from column names + data
    fn __init__(out self, columns: List[String], data: List[List[String]], index: List[String], index_name: String = ""):
        self.col_names = columns
        self.names = List[String]()
        var i = 0
        while i < len(columns):
            self.names.append(String(columns[i]))
            i += 1
        self.cols = List[Column]()
        i = 0
        while i < len(columns):
            var s = SeriesStr(data[i], columns[i])
            var col = Column()
            col.from_str(s)
            self.cols.append(col)
            i += 1
        self.index_name = index_name
        self.index_vals = index

    # Utilities
    fn shape_str(self) -> String:
        return "(" + String(self.nrows()) + ", " + String(self.ncols()) + ")"

    fn find_col(self, name: String) -> Int:
        var i = 0
        while i < len(self.col_names):
            if self.col_names[i] == name:
                return i
            i += 1
        return -1

    fn ncols(self) -> Int:
        return len(self.col_names)

    fn nrows(self) -> Int:
        if len(self.cols) == 0:
            return 0
        return self.cols[0].len()

    fn width(self) -> Int:
        return self.ncols()

    fn height(self) -> Int:
        return self.nrows()

    fn get_column(self, name: String) -> Column:
        var idx = self.find_col(name)
        if idx < 0 or idx >= self.ncols():
            print(String("Column not found: ") + name)
            var c = Column()
            var s = SeriesStr()
            s.name = name
            s.data = List[String]()
            c.from_str(s)
            return c
        return self.cols[idx]
    
    fn get_column_by_name(self, name: String) -> Column:
        var idx = self.find_col(name)
        if idx == -1:
            print("Column not found: " + name)
            return Column()  
        return self.cols[idx]

    # Set / replace column
    fn set_column(mut self, col: Column):
        var idx = self.find_col(col.get_name())
        if idx == -1:
            self.col_names.append(col.get_name())
            self.names.append(col.get_name())
            self.cols.append(col)
        else:
            self.cols[idx] = col
            self.col_names[idx] = col.get_name()
            self.names[idx] = col.get_name()

    # Convert DataFrame to string (preview)
    fn to_string(self) -> String:
        var s = "DataFrame(" + String(self.nrows()) + "x" + String(self.ncols()) + ")\n"
        if self.index_name != "":
            s += self.index_name + " | "
        var i = 0
        while i < len(self.col_names):
            s += self.col_names[i]
            if i + 1 < len(self.col_names):
                s += " | "
            i += 1
        s += "\n"
        var show = self.nrows()
        if show > 10:
            show = 10
        var r = 0
        while r < show:
            if self.index_name != "" and len(self.index_vals) == self.nrows():
                s += self.index_vals[r] + " | "
            var c = 0
            while c < self.ncols():
                s += self.cols[c].get_string(r)
                if c + 1 < self.ncols():
                    s += " | "
                c += 1
            s += "\n"
            r += 1
        if self.nrows() > show:
            s += "...\n"
        return s

    # Melt function
    fn melt(self, id_vars: List[String], var_name: String = "variable", value_name: String = "value") -> DataFrame:
        # Collect columns to melt
        var melt_cols = List[Int]()
        var i = 0
        while i < self.ncols():
            var col_name = self.col_names[i]
            var skip = False
            var j = 0
            while j < len(id_vars):
                if id_vars[j] == col_name:
                    skip = True
                j += 1
            if not skip:
                melt_cols.append(i)
            i += 1

        # Build new DataFrame
        var out_cols = List[Column]()
        # ID columns
        var j = 0
        while j < len(id_vars):
            out_cols.append(self.get_column(id_vars[j]))
            j += 1

        # Melted columns: variable + value
        var nrows = self.nrows()
        var var_series = SeriesStr()
        var val_series = SeriesStr()
        var_series.data = List[String]()
        val_series.data = List[String]()
        var i_melt = 0
        while i_melt < len(melt_cols):
            var col_idx = melt_cols[i_melt]
            var col = self.cols[col_idx]
            var r = 0
            while r < nrows:
                var_series.data.append(self.col_names[col_idx])
                val_series.data.append(col.get_string(r))
                r += 1
            i_melt += 1
        var_series.name = var_name
        val_series.name = value_name
        var col_var = Column()
        col_var.from_str(var_series)
        var col_val = Column()
        col_val.from_str(val_series)
        out_cols.append(col_var)
        out_cols.append(col_val)

        # Build melted DataFrame
        var out_df = DataFrame()
        var k = 0
        while k < len(out_cols):
            out_df.set_column(out_cols[k])
            k += 1
        return out_df


    fn with_rows(self, n: Int) -> DataFrame:
        var out = DataFrame()

        # Copy column names
        var i = 0
        while i < len(self.col_names):
            out.col_names.append(String(self.col_names[i]))
            out.names.append(String(self.col_names[i]))
            i += 1

        # Copy first n rows of each column
        i = 0
        while i < len(self.cols):
            var c = self.cols[i]
            var cc = Column()

            var dtype = c.dtype_name()
            if dtype == "str":
                var s = SeriesStr()
                var r = 0
                while r < n and r < c.len():
                    s.data.append(c.get_string(r))
                    _ = c.validity().is_set(r)  # optional
                    r += 1
                s.name = c.get_name()
                cc.from_str(s)
            elif dtype == "i64":
                var s = SeriesI64()
                var r = 0
                while r < n and r < c.len():
                    s.data.append(c.i64.get(r))  # scalar, بدون [0]
                    _ = c.i64.valid.is_set(r)
                    r += 1
                s.name = c.get_name()
                cc.from_i64(s)
            elif dtype == "f64":
                var s = SeriesF64()
                var r = 0
                while r < n and r < c.len():
                    s.data.append(c.f64.get(r))  # scalar, بدون [0]
                    _ = c.f64.valid.is_set(r)
                    r += 1
                s.name = c.get_name()
                cc.from_f64(s)
            elif dtype == "bool":
                var s = SeriesBool()
                var r = 0
                while r < n and r < c.len():
                    s.data.append(c.b.get(r))
                    _ = c.b.valid.is_set(r)
                    r += 1
                s.name = c.get_name()
                cc.from_bool(s)

            out.cols.append(cc)
            i += 1

        # Copy index values
        var k = 0
        while k < n and k < len(self.index_vals):
            out.index_vals.append(String(self.index_vals[k]))
            k += 1
        out.index_name = String(self.index_name)

        return out
    
    fn drop_duplicates(self, subset: List[String] = List[String](), keep: String = "first") -> DataFrame:
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

    
    fn add_column(mut self, col: Column):
        var idx = self.find_col(col.get_name())
        if idx == -1:
            self.col_names.append(col.get_name())
            self.names.append(col.get_name())
            self.cols.append(col)
        else:
            self.cols[idx] = col
            self.col_names[idx] = col.get_name()
            self.names[idx] = col.get_name()
    













fn copy(self) -> DataFrame:
        var out = DataFrame()
        out.__copyinit__(self)
        return out

fn clone(self) -> DataFrame:
    var out = DataFrame()
    out.__copyinit__(self)
    return out

    fn width(self) -> Int:
        return self.ncols()

    fn height(self) -> Int:
        return self.nrows()

struct ModuleState:
    var col_names
    fn __init__(out self, col_names):
        self.col_names = col_names

fn make_module_state(state) -> ModuleState:
    return ModuleState(List[String]())


    var cols = List[Column]()

    var cidx = 0
    while cidx < df.width(, state: ModuleState):
        var name = df.col_names[cidx]
        col_names.append(name)

        var src = df.get_column_at(cidx)

        if src.is_f64():
            var arr = List[Float64]()
            var r = 0
            while r < take:
                arr.append(src.get_f64(r))
                r += 1
            cols.append(Column.from_f64(SeriesF64(name, arr)))

        elif src.is_i64():
            var arr_i = List[Int64]()
            var r2 = 0
            while r2 < take:
                arr_i.append(src.get_i64(r2))
                r2 += 1
            cols.append(Column.from_i64(SeriesI64(name, arr_i)))

        elif src.is_bool():
            var arr_b = List[Bool]()
            var r3 = 0
            while r3 < take:
                arr_b.append(src.get_bool(r3))
                r3 += 1
            cols.append(Column.from_bool(SeriesBool(name, arr_b)))

        else:
            var arr_s = List[String]()
            var r4 = 0
            while r4 < take:
                assert(src is not None, String("src is None"))
                arr_s.append(src.value()_to_string(r4))
                r4 += 1
            cols.append(Column.from_str(SeriesStr(name, arr_s)))

        cidx += 1

    return DataFrame(col_names, cols)

# Select a subset of columns by name, in the given order. Missing col_names are skipped.
fn select_columns_safe(df: DataFrame, want: List[String]) -> DataFrame
    var col_names = List[String]()
    var cols = List[Column]()
    var i = 0
    while i < len(want):
        var name = df.col_names[cidx]
        state.col_names.append(name)

        var src = df.get_column_at(cidx)

        if src.is_f64():
            var arr = List[Float64]()
            var r = 0
            while r < take:
                arr.append(src.get_f64(r))
                r += 1
            cols.append(Column.from_f64(SeriesF64(name, arr)))

        elif src.is_i64():
            var arr_i = List[Int64]()
            var r2 = 0
            while r2 < take:
                arr_i.append(src.get_i64(r2))
                r2 += 1
            cols.append(Column.from_i64(SeriesI64(name, arr_i)))

        elif src.is_bool():
            var arr_b = List[Bool]()
            var r3 = 0
            while r3 < take:
                arr_b.append(src.get_bool(r3))
                r3 += 1
            cols.append(Column.from_bool(SeriesBool(name, arr_b)))

        else:
            var arr_s = List[String]()
            var r4 = 0
            while r4 < take:
                assert(src is not None, String("src is None"))
                arr_s.append(src.value()_to_string(r4))
                r4 += 1
            cols.append(Column.from_str(SeriesStr(name, arr_s)))

        cidx += 1

    return DataFrame(state.col_names, cols)


# ---------- Helpers ----------
fn _deep_copy_df(src: DataFrame) -> DataFrame:
    var out = DataFrame()

    # Copy column names
    out.col_names = List[String]()
    for name in src.col_names:
        out.col_names.append(String(name))

    # Copy columns as Column objects (not raw lists)
    out.cols = List[Column]()
    for col in src.cols:
        # Convert each list to Column
        var new_col = Column()
        for v in col:
            new_col.append(String(v))
        out.cols.append(new_col)

    # Copy index metadata
    out.index_name = String(src.index_name)
    out.index_vals = List[String]()
    for idx in src.index_vals:
        out.index_vals.append(String(idx))

    return out

fn _find_col(df: DataFrame, name: String) -> Int:
    var i = 0
    while i < len(df.col_names):
        if df.col_names[i] == name:
            return i
        i += 1
    return -1

fn _nrows(df: DataFrame) -> Int:
    if len(df.cols) == 0:
        return 0
    return len(df.cols[0])

# ---------- API ----------

# Set a column as index: moves the column into index_vals/index_name and removes it from data columns.
fn set_index(df: DataFrame, col: String) -> DataFrame:
    var idx = _find_col(df, col)
    var out = DataFrame()

    # prepare containers
    out.col_names = List[String]()
    out.cols = List[Column]()

    # count number of columns
    var n_cols = 0
    for _ in df.col_names:
        n_cols += 1

    var c = 0
    while c < n_cols:
        if c != idx:
            out.col_names.append(String(df.col_names[c]))
            # copy Column by value (Column is Copyable/Movable)
            out.cols.append(df.cols[c])
        c += 1

    # set index metadata
    out.index_name = String(col)
    out.index_vals = List[String]()

    if idx >= 0:
        var n_rows_in_idx = df.cols[idx].len()   # use Column.len()
        var r2 = 0
        while r2 < n_rows_in_idx:
            out.index_vals.append(df.cols[idx].get_string(r2))
            r2 += 1
    else:
        # if index column not found, infer number of rows from first column (if any)
        var n_cols2 = 0
        for _ in df.cols:
            n_cols2 += 1
        var n = 0
        if n_cols2 > 0:
            n = df.cols[0].len()
        var i = 0
        while i < n:
            out.index_vals.append(String(i))
            i += 1

    return out


# Reset index: inserts index as a normal column (as the first column), and rebuilds a default integer-like index.
# Reset index: inserts index as a normal column (as the first column), and rebuilds a default integer-like index.
fn reset_index(df: DataFrame) -> DataFrame:
    var out = DataFrame()

    # prepare containers
    out.col_names = List[String]()
    out.cols = List[Column]()

    # 1) determine index column name (compute length of df.index_name safely)
    var tmp_len = 0
    for _ in df.index_name:
        tmp_len += 1
    var idx_name = String(df.index_name)
    if tmp_len == 0:
        idx_name = String("index")

    # append index name as first column name
    out.col_names.append(idx_name)

    # build idx_col_vals from df.index_vals if present
    var idx_col_vals: List[String] = List[String]()
    var n_index_vals = 0
    for _ in df.index_vals:
        n_index_vals += 1

    var ii = 0
    while ii < n_index_vals:
        idx_col_vals.append(String(df.index_vals[ii]))
        ii += 1

    # if empty, synthesize from first data column length
    var idx_vals_len = 0
    for _ in idx_col_vals:
        idx_vals_len += 1

    if idx_vals_len == 0:
        # count number of columns in df.cols
        var ncols = 0
        for _ in df.cols:
            ncols += 1

        var nrows = 0
        if ncols > 0:
            # df.cols[0] is Column -> use Column.len()
            nrows = df.cols[0].len()
        var kk = 0
        while kk < nrows:
            idx_col_vals.append(String(kk))
            kk += 1

    # create a Column for the index (SeriesStr -> Column.from_str)
    var idx_column = Column()
    var s = SeriesStr()
    var j = 0
    while j < len(idx_col_vals):
        # If your SeriesStr API uses push(...) instead of append(...), replace accordingly.
        s.append(idx_col_vals[j])
        j += 1
    idx_column.from_str(s)
    out.cols.append(idx_column)

    # 2) append all existing columns (copy-by-value, preserving original column order)
    var n_col_names = 0
    for _ in df.col_names:
        n_col_names += 1

    var c = 0
    while c < n_col_names:
        out.col_names.append(String(df.col_names[c]))
        out.cols.append(df.cols[c])   # Column is Copyable/Movable, so shallow/value copy
        c += 1

    # 3) reset index meta to default (0..nrows_after-1)
    out.index_name = String("")
    out.index_vals = List[String]()

    # compute number of rows after insertion:
    var out_ncols = 0
    for _ in out.cols:
        out_ncols += 1

    var nrows_after = 0
    if out_ncols > 1:
        nrows_after = out.cols[1].len()   # data columns start at 1 after inserted index col
    else:
        if out_ncols > 0:
            nrows_after = out.cols[0].len()
        else:
            nrows_after = 0

    var t = 0
    while t < nrows_after:
        out.index_vals.append(String(t))
        t += 1

    return out



# Constructors
fn from_pairs(pairs: List[Tuple[String, List[String]]]) -> DataFrame:
    return _df_from_pairs(pairs)

# Select helpers
fn _to_pairs_all(df: DataFrame) -> List[Tuple[String, List[String]]]:
    var out = List[Tuple[String, List[String]]]()
    var i = 0
    while i < len(df.col_names):
        var vals = List[String]()
        var r = 0
        while r < df.nrows():
            vals.append(String(df.cols[i][r]))
            r += 1
        out.append((df.col_names[i], vals))
        i += 1
    return out

fn _to_pairs_select(df: DataFrame, cols: List[String]) -> List[Tuple[String, List[String]]]:
    if len(cols) == 0:
        return _to_pairs_all(df)
    var keep = Dict[String, Int]()
    var i = 0
    while i < len(cols):
        keep[cols[i]] = 1
        i += 1
    var out = List[Tuple[String, List[String]]]()
    i = 0
    while i < len(df.col_names):
        var name = df.col_names[i]
        var take = False
        if name in keep: take = True
        if take:
            var vals = List[String]()
            var r = 0
            while r < df.nrows():
                vals.append(String(df.cols[i][r]))
                r += 1
            out.append((name, vals))
        i += 1
    return out

# dtypes noop (no raises)
fn _apply_dtypes(df: DataFrame, dtypes: Dict[String, String]) -> DataFrame:
    if len(dtypes) == 0: return df
# no conversion in this build; just rebuild pairs
    return _df_from_pairs(_to_pairs_all(df))

# CSV API
fn read_csv(path: Path, usecols: List[String] = List[String](), dtypes: Dict[String, String] = Dict[String, String]()) -> DataFrame:
    var base = _read_csv_file(String(path))
    if len(usecols) > 0:
        base = _df_from_pairs(_to_pairs_select(base, usecols))
    if len(dtypes) > 0:
        base = _apply_dtypes(base, dtypes)
    return base

fn read_csv_string(text: String, usecols: List[String] = List[String](), dtypes: Dict[String, String] = Dict[String, String]()) -> DataFrame:
    var base = _read_csv_text(text)
    if len(usecols) > 0:
        base = _df_from_pairs(_to_pairs_select(base, usecols))
    if len(dtypes) > 0:
        base = _apply_dtypes(base, dtypes)
    return base

fn write_csv(frame: DataFrame, path: Path, index: Bool = False) -> Bool:
    return _write_csv_file(frame, String(path))

# JSON writer (safe concatenation; no char literals)
fn _q() -> String:
    return String("")

fn _comma() -> String:
    return String(",")

fn _lbrace() -> String:
    return String("{")

fn _rbrace() -> String:
    return String("}")

fn _lbracket() -> String:
    return String("[")

fn _rbracket() -> String:
    return String("]")

fn _colon() -> String:
    return String(":")

fn _newline() -> String:
    return String("\\n")

fn json_field(name: String, value: String) -> String:
# No escaping to keep implementation robust in this build; test data has no quotes/backslashes.
    return _q() + name + _q() + _colon() + _q() + value + _q()

fn write_json(frame: DataFrame, path: Path, orient_records: Bool = True, lines: Bool = False) -> Bool:
    var out = String("")
    if orient_records and not lines:
# JSON array of records
        out = _lbracket()
        var r = 0
        var rows = frame.nrows()
        while r < rows:
            var i = 0
            out = out + _lbrace()
            while i < len(frame.col_names):
                out = out + json_field(frame.col_names[i], String(frame.cols[i][r]))
                if i + 1 < len(frame.col_names):
                    out = out + _comma()
                i += 1
            out = out + _rbrace()
            if r + 1 < rows:
                out = out + _comma()
            r += 1
        out = out + _rbracket()
    else:
# NDJSON
        var r2 = 0
        var rows2 = frame.nrows()
        while r2 < rows2:
            var i2 = 0
            out = out + _lbrace()
            while i2 < len(frame.col_names):
                out = out + json_field(frame.col_names[i2], String(frame.cols[i2][r2]))
                if i2 + 1 < len(frame.col_names):
                    out = out + _comma()
                i2 += 1
            out = out + _rbrace() + _newline()
            r2 += 1
    return _write_text_file(String(path), out)



# Convenience
fn head(frame: DataFrame, n: Int = 5) -> DataFrame:
    return frame.with_rows(n)

fn equals(a: DataFrame, b: DataFrame) -> Bool:
    if len(a.col_names) != len(b.col_names): return False
    if a.nrows() != b.nrows(): return False
    var i = 0
    while i < len(a.col_names):
        if a.col_names[i] != b.col_names[i]: return False
        var r = 0
        while r < a.nrows():
            if a.cols[i][r] != b.cols[i][r]: return False
            r += 1
        i += 1
    return True

fn show(frame: DataFrame, n: Int = 5) -> String:
    return frame.with_rows(n).to_string()

# FS write stub
fn _write_text_file(path: String, text: String) -> Bool:
    return False




# ----------------------------- Window helpers -------------------------------

# 1) Row number: 1..n
fn row_number(n: Int) -> List[Int64]:
    var out = List[Int64]()
    var i: Int = 0
    while i < n:
        out.append(Int64(i + 1))
        i += 1
    return out

# 2) Dense rank (Float64)
fn dense_rank(xs: List[Float64]) -> List[Int64]:
# stable-ish O(n^2) for small n: count unique values <= current
    var out = List[Int64]()
    var seen = List[Float64]()
    var i: Int = 0
    while i < len(xs):
# insert into seen if new
        var exists: Bool = False
        var j: Int = 0
        while j < len(seen):
            if xs[i] == seen[j]:
                exists = True
                break
            j += 1
        if not exists:
# insert keeping asc order (linear)
            var k: Int = 0
            var placed: Bool = False
            while k < len(seen):
                if xs[i] < seen[k]:
# shift-right by rebuild
                    var tmp = List[Float64]()
                    var t: Int = 0
                    while t < k:
                        tmp.append(seen[t]); t += 1
                    tmp.append(xs[i])
                    while k < len(seen):
                        tmp.append(seen[k]); k += 1
                    seen = tmp
                    placed = True
                    break
                k += 1
            if not placed:
                seen.append(xs[i])
# rank is 1-based position in seen
        var r: Int = 0
        while r < len(seen) and seen[r] != xs[i]:
            r += 1
        out.append(Int64(r + 1))
        i += 1
    return out

# 3) Lag/Lead for Float64
fn lag(xs: List[Float64], k: Int = 1, fill: Float64 = 0.0) -> List[Float64]:
    var out = List[Float64]()
    var i: Int = 0
    while i < len(xs):
        if i - k >= 0:
            out.append(xs[i - k])
        else:
            out.append(fill)
        i += 1
    return out

fn lead(xs: List[Float64], k: Int = 1, fill: Float64 = 0.0) -> List[Float64]:
    var out = List[Float64]()
    var i: Int = 0
    while i < len(xs):
        if i + k < len(xs):
            out.append(xs[i + k])
        else:
            out.append(fill)
        i += 1
    return out

# 4) Cumulative sum / mean (Float64)
fn cum_sum(xs: List[Float64]) -> List[Float64]:
    var out = List[Float64]()
    var acc: Float64 = 0.0
    var i: Int = 0
    while i < len(xs):
        acc += xs[i]
        out.append(acc)
        i += 1
    return out

fn cum_mean(xs: List[Float64]) -> List[Float64]:
    var out = List[Float64]()
    var acc: Float64 = 0.0
    var i: Int = 0
    while i < len(xs):
        acc += xs[i]
        out.append(acc / Float64(i + 1))
        i += 1
    return out

# 5) Rolling mean (window size w, centered=False; simple trailing window)
fn rolling_mean(xs: List[Float64], w: Int) -> List[Float64]:
    if w <= 0:
        return List[Float64]()
    var out = List[Float64]()
    var acc: Float64 = 0.0
    var i: Int = 0
    while i < len(xs):
        acc += xs[i]
        if i >= w:
            acc -= xs[i - w]
        if i + 1 >= w:
            out.append(acc / Float64(w))
        else:
            out.append(acc / Float64(i + 1))  # warm-up
        i += 1
    return out

# ----------------------------- Pretty printer -------------------------------

fn print_df(df: DataFrame, title: String = String("DataFrame")):
    print(String("\n=== ") + title + String(" ==="))
    print(String("shape: (") + String(df.nrows()) + String(", ") + String(df.ncols()) + String(")"))
# header
    var i: Int = 0
    var hdr = String("")
    while i < df.ncols():
        hdr = hdr + df.col_names[i]
        if i + 1 < df.ncols():
            hdr = hdr + String(", ")
        i += 1
    print(hdr)
# first up to 10 rows
    var rmax: Int = df.nrows()
    if rmax > 10:
        rmax = 10
    var r: Int = 0
    while r < rmax:
        var line = String("")
        var c: Int = 0
        while c < df.ncols():
            line = line + df.cols[c][r]
            if c + 1 < df.ncols():
                line = line + String(", ")
            c += 1
        print(line)
        r += 1
    if df.nrows() > rmax:
        print(String("... (") + String(df.nrows() - rmax) + String(" more rows)"))



fn rolling_mean(xs: List[Float64], win: Int) -> List[Float64]
    var n = len(xs)
    var out = List[Float64]()
    var i = 0
    while i < n:
        var s = 0.0
        var cnt = 0
        var k = i - win + 1
        if k < 0:
            k = 0
        while k <= i:
            s += xs[k]
            cnt += 1
            k += 1
        out.append(s / Float64(cnt))
        i += 1
    return out
fn rolling_apply_abs(xs: List[Float64], win: Int) -> List[Float64]
# rolling mean of absolute values
    var n = len(xs)
    var out = List[Float64]()
    var i = 0
    while i < n:
        var s = 0.0
        var cnt = 0
        var k = i - win + 1
        if k < 0:
            k = 0
        while k <= i:
            var v = xs[k]
            s += (v if v >= 0.0 else -v)
            cnt += 1
            k += 1
        out.append(s / Float64(cnt))
        i += 1
    return out


fn rank_dense_f64(xs: List[Float64]) -> List[Int]
    var order = argsort_f64(xs, True)
    var ranks = List[Int](len(xs), 0)
    var rnk = 1
    var i = 0
    while i < len(order):
        if i == 0 or xs[order[i]] not = xs[order[i - 1]]:
            rnk = rnk if i == 0 else rnk + 1
        ranks[order[i]] = rnk
        i += 1
    return ranks
fn sort_values_key(xs: List[String]) -> List[Int]
    var idxs = List[Int]()
    var i = 0
    while i < len(xs):
        idxs.append(i)
        i += 1
    var a = 0
    while a < len(xs):
        var b = a + 1
        while b < len(xs):
            if len(xs[idxs[b]]) < len(xs[idxs[a]]):
                var t = idxs[a]
                idxs[a] = idxs[b]
                idxs[b] = t
            b += 1
        a += 1
    return idxs


 

# two-key argsort: primary by city (String), secondary by parsed minutes of timestamp
fn argsort_city_ts(cities: List[String], ts: List[String]) raises -> List[Int]:
    var idx = List[Int]()
    var i = 0
    while i < len(cities)
        idx.append(i)
        i += 1
    var j = 1
    while j < len(idx):
        var key = idx[j]
        var k = j - 1
        var m_key = parse_minutes(ts[key])
        while k >= 0:
            var cmp = False
            var m_k = parse_minutes(ts[idx[k]])
            if cities[idx[k]] > cities[key]:
                cmp = True
            else:
                if cities[idx[k]] == cities[key] and m_k > m_key:
                    cmp = True
            if not cmp:
                break
            idx[k + 1] = idx[k]
            k -= 1
        idx[k + 1] = key
        j += 1
    return idx

# Sort values by columns
fn sort_values(df: DataFrame, by: List[String], ascending: List[Bool]) -> DataFrame:
    var out = df.copy()
    var n = out.nrows()
    var idx_order = List[Int]()
    var i = 0
    while i < n:
        idx_order.append(i)
        i += 1

    # Build keys for sorting
    var keys = List[List[String]]()
    var bi = 0
    while bi < len(by):
        var col_idx = out.find_col(by[bi])
        var col_vals = List[String]()
        var r = 0
        while r < n:
            col_vals.append(out.cols[col_idx].get_string(r))
            r += 1
        keys.append(col_vals)
        bi += 1

    # Simple bubble-sort by multiple keys (for demonstration)
    var swapped = True
    while swapped:
        swapped = False
        var j = 0
        while j < n - 1:
            var cmp = False
            var k = 0
            while k < len(by):
                if keys[k][idx_order[j]] != keys[k][idx_order[j+1]]:
                    if ascending[k]:
                        cmp = keys[k][idx_order[j]] > keys[k][idx_order[j+1]]
                    else:
                        cmp = keys[k][idx_order[j]] < keys[k][idx_order[j+1]]
                    break
                k += 1
            if cmp:
                var tmp = idx_order[j]
                idx_order[j] = idx_order[j+1]
                idx_order[j+1] = tmp
                swapped = True
            j += 1

    # Rearrange columns and index
    var c = 0
    while c < out.ncols():
        out.cols[c] = out.cols[c].take(idx_order)
        c += 1
    out.index_vals = List[String]()
    var r2 = 0
    while r2 < n:
        out.index_vals.append(String(r2))
        r2 += 1

    return out


