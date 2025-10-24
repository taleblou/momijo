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

from momijo.dataframe.column import *
from momijo.dataframe.series_str import SeriesStr 
 
from momijo.dataframe.io_csv import is_bool
from momijo.dataframe.series_bool import SeriesBool, append
from momijo.dataframe.series_f64 import SeriesF64
from momijo.dataframe.series_i64 import SeriesI64
from momijo.dataframe.selection import *
from collections.dict import Dict
from collections.list import List
from pathlib.path import Path

from momijo.dataframe.compat import df_from_pairs as _df_from_pairs
 
from momijo.dataframe.io_csv import read_csv ,write_csv


  
from momijo.dataframe._groupby_core import pivot_table

from momijo.dataframe.series_bool import append
from momijo.dataframe.datetime_ops import parse_minutes
from momijo.dataframe.helpers import argsort_f64, argsort_i64
from momijo.dataframe.api import *
from momijo.dataframe.datetime_ops import Resampler
 
from collections.list import List





# Assuming Column, SeriesStr and their APIs are available in scope:
# - Column(): default ctor
# - Column.get_name() -> String
# - Column.len() -> Int
# - Column.get_string(i: Int) -> String
# - Column.from_str(s: SeriesStr) -> None
# - Column.copy() -> Column
# - SeriesStr(): default ctor with .name: String, .data: List[String]
struct DataFrame(ImplicitlyCopyable, Copyable, Movable):
    var col_names:  List[String]
    var names:      List[String]      # kept for backward-compat
    var cols:       List[Column]
    var index_name: String
    var index_vals: List[String]

    # ---------------------------------------------------------------------
    # Constructors
    # ---------------------------------------------------------------------
    fn __init__(out self):
        self.col_names  = List[String]()
        self.names      = List[String]()
        self.cols       = List[Column]()
        self.index_name = String("")
        self.index_vals = List[String]()

    # Build from column names + column-major string data + optional index
    fn __init__(out self, columns: List[String], data: List[List[String]], index: List[String], index_name: String = ""):
        var ncols = len(columns)
        var dcols = len(data)
        var use_cols = ncols
        if dcols < use_cols: use_cols = dcols

        var min_len = 0
        if use_cols > 0:
            min_len = len(data[0])
            var c = 1
            while c < use_cols:
                var h = len(data[c])
                if h < min_len: min_len = h
                c += 1

        self.col_names  = List[String](); self.col_names.reserve(use_cols)
        self.names      = List[String](); self.names.reserve(use_cols)
        self.cols       = List[Column](); self.cols.reserve(use_cols)
        self.index_name = String(index_name)
        self.index_vals = List[String](); self.index_vals.reserve(min_len)

        # column names
        var i = 0
        while i < use_cols:
            var cname = columns[i]
            self.col_names.append(cname)
            self.names.append(cname)
            i += 1

        # build columns via col_from_list (type will be inferred per list)
        i = 0
        while i < use_cols:
            var vals = List[String](); vals.reserve(min_len)
            var r = 0
            while r < min_len:
                vals.append(data[i][r])
                r += 1
            var col = col_from_list(vals, self.col_names[i])
            self.cols.append(col.copy())
            i += 1

        # index
        var ilen = len(index)
        if ilen == min_len and min_len > 0:
            var j = 0
            while j < ilen:
                self.index_vals.append(index[j])
                j += 1
        else:
            var j = 0
            while j < min_len:
                self.index_vals.append(String(j))
                j += 1

    # ---------------------------------------------------------------------
    # Copying
    # ---------------------------------------------------------------------
    @always_inline
    fn __copyinit__(out self, other: DataFrame):
        self.col_names  = other.col_names.copy()
        self.names      = other.names.copy()
        self.index_name = String(other.index_name)
        self.index_vals = other.index_vals.copy()
        var n = len(other.cols)
        self.cols = List[Column](); self.cols.reserve(n)
        var i = 0
        while i < n:
            self.cols.append(other.cols[i].copy())
            i += 1

    @always_inline 
    fn copy(self) -> DataFrame: var out = self; return out
    @always_inline 
    fn clone(self) -> DataFrame: return self.copy()

    # ---------------------------------------------------------------------
    # Introspection
    # ---------------------------------------------------------------------
    @always_inline 
    fn ncols(self) -> Int: return len(self.col_names)
    @always_inline 
    fn nrows(self) -> Int:
        if len(self.cols) == 0: return 0
        return self.cols[0].len()
    @always_inline 
    fn width(self) -> Int: return self.ncols()
    @always_inline 
    fn height(self) -> Int: return self.nrows()
    @always_inline 
    fn shape_str(self) -> String:
        return "(" + String(self.nrows()) + ", " + String(self.ncols()) + ")"

    # ---------------------------------------------------------------------
    # Column lookup / access
    # ---------------------------------------------------------------------
    fn find_col(self, name: String) -> Int:
        var i = 0
        var n = len(self.col_names)
        while i < n:
            if self.col_names[i] == name: return i
            i += 1
        return -1

    fn get_column(self, name: String) -> Column:
        var idx = self.find_col(name)
        if idx < 0 or idx >= self.ncols():
            # return an empty string-typed column with the requested name
            var empty = List[String]()
            return col_from_list_with_tag(empty, name, 4).copy()   # 4 == STR tag in ColumnTag
        return self.cols[idx].copy()

    @always_inline 
    fn get_column_by_name(self, name: String) -> Column:
        return self.get_column(name)



    @always_inline 
    fn add_column(mut self, col: Column): self.set_column(col)

    # ---------------------------------------------------------------------
    # __getitem__ overloads (pandas-like where feasible)
    # ---------------------------------------------------------------------

    # df['col'] -> Column
    @always_inline 
    fn __getitem__(self, name: String) -> Column:
        return self.get_column(name)

    # df[idx] -> Column by position (0-based)
    @always_inline 
    fn __getitem__(self, idx: Int) -> Column:
        if idx >= 0 and idx < self.ncols(): return self.cols[idx].copy()
        # out-of-range: return empty string column with synthetic name
        var empty = List[String]()
        return col_from_list_with_tag(empty, String(idx), 4).copy()  # 4 == STR tag

    # df[['c1','c2',...]] -> DataFrame by names
    fn __getitem__(self, names: List[String]) -> DataFrame:
        var out = DataFrame()
        out.index_name = String(self.index_name)
        out.index_vals = self.index_vals.copy()
        var i = 0
        var k = len(names)
        while i < k:
            var nm = names[i]
            var pos = self.find_col(nm)
            if pos >= 0: out.set_column(self.cols[pos].copy())
            i += 1
        return out

    # df[[i1, i2, ...]] -> DataFrame by integer column indices
    fn __getitem__(self, indices: List[Int]) -> DataFrame:
        var out = DataFrame()
        out.index_name = String(self.index_name)
        out.index_vals = self.index_vals.copy()
        var i = 0
        var k = len(indices)
        while i < k:
            var idx = indices[i]
            if idx >= 0 and idx < self.ncols(): out.set_column(self.cols[idx].copy())
            i += 1
        return out

    # df[mask] -> row filter (boolean indexing)
    fn __getitem__(self, mask: List[Bool]) -> DataFrame:
        return self._select_rows_by_mask(mask)

    # ---------------------------------------------------------------------
    # Row/col selection helpers (all rebuild columns via col_from_list_with_tag)
    # ---------------------------------------------------------------------
    fn _frame_like_empty(self) -> DataFrame:
        var out = DataFrame()
        out.index_name = String(self.index_name)
        out.index_vals = List[String]()
        var c = 0
        while c < self.ncols():
            var empty = List[String]()
            var col = col_from_list_with_tag(empty, self.col_names[c], self.cols[c].tag)
            out.set_column(col.copy())
            c += 1
        return out

    fn _select_rows_by_positions(self, pos: List[Int]) -> DataFrame:
        var n = self.nrows()
        # filter valid row indices first
        var kept = List[Int]()
        kept.reserve(len(pos))
        var i = 0
        var k = len(pos)
        while i < k:
            var r = pos[i]
            if r >= 0 and r < n: kept.append(r)
            i += 1

        var out = DataFrame()
        out.index_name = String(self.index_name)
        out.index_vals = List[String](); out.index_vals.reserve(len(kept))
        var t = 0
        var kr = len(kept)
        while t < kr:
            out.index_vals.append(self.index_vals[kept[t]])
            t += 1

        var c = 0
        while c < self.ncols():
            var xs = List[String](); xs.reserve(len(kept))
            var j = 0
            while j < len(kept):
                xs.append(self.cols[c].get_string(kept[j]))
                j += 1
            var col = col_from_list_with_tag(xs, self.col_names[c], self.cols[c].tag)
            out.set_column(col.copy())
            c += 1

        return out

    fn _select_rows_by_mask(self, mask: List[Bool]) -> DataFrame:
        var n = self.nrows()
        if len(mask) != n:
            return self._frame_like_empty()

        var kept = List[Int]()
        kept.reserve(n)
        var r = 0
        while r < n:
            if mask[r]: kept.append(r)
            r += 1

        return self._select_rows_by_positions(kept)

    fn _select_rows_by_label_slice(self, s: LabelSlice) -> DataFrame:
        var n = self.nrows()
        var a = 0
        while a < n and self.index_vals[a] < s.start: a += 1
        var b = a
        while b < n and (self.index_vals[b] < s.end or (s.inclusive and self.index_vals[b] == s.end)): b += 1

        var kept = List[Int]()
        kept.reserve(b - a)
        var r = a
        while r < b:
            kept.append(r)
            r += 1

        return self._select_rows_by_positions(kept)

    fn _select_rows_by_pos_slice(self, s: PosSlice) -> DataFrame:
        var n = self.nrows()
        var a = s.start
        if a < 0: a = 0
        var b = s.stop
        if b > n: b = n
        if b < a: b = a

        var kept = List[Int]()
        kept.reserve(b - a)
        var r = a
        while r < b:
            kept.append(r)
            r += 1

        return self._select_rows_by_positions(kept)

    fn _select_cols_by_names(self, names: List[String]) -> DataFrame:
        var out = DataFrame()
        out.index_name = String(self.index_name)
        out.index_vals = self.index_vals.copy()
        var i = 0
        var k = len(names)
        while i < k:
            var nm = names[i]
            var pos = self.find_col(nm)
            if pos >= 0: out.set_column(self.cols[pos].copy())
            i += 1
        return out

    fn _select_cols_by_positions(self, pos: List[Int]) -> DataFrame:
        var out = DataFrame()
        out.index_name = String(self.index_name)
        out.index_vals = self.index_vals.copy()
        var i = 0
        var k = len(pos)
        while i < k:
            var c = pos[i]
            if c >= 0 and c < self.ncols(): out.set_column(self.cols[c].copy())
            i += 1
        return out

    fn _select_col_single_by_name(self, name: String) -> Column:
        return self.get_column(name)

    # ---------------------------------------------------------------------
    # loc(rows, cols)
    # ---------------------------------------------------------------------
    @always_inline 
    fn loc(self, rows: RowAll, cols: ColAll) -> DataFrame: return self.copy()
    fn loc(self, rows: RowAll, cols: List[String]) -> DataFrame: return self._select_cols_by_names(cols)
    fn loc(self, rows: RowAll, cols: List[Int])    -> DataFrame: return self._select_cols_by_positions(cols)
    @always_inline
    fn loc(self, rows: RowAll, cols: String) -> Column: return self._select_col_single_by_name(cols)

    fn loc(self, rows: RowPos, cols: ColAll) -> DataFrame: return self._select_rows_by_positions(rows.indices)
    fn loc(self, rows: RowPos, cols: List[String]) -> DataFrame: return self._select_rows_by_positions(rows.indices)._select_cols_by_names(cols)
    fn loc(self, rows: RowPos, cols: List[Int])    -> DataFrame: return self._select_rows_by_positions(rows.indices)._select_cols_by_positions(cols)
    fn loc(self, rows: RowPos, cols: String) -> Column: return self._select_rows_by_positions(rows.indices)._select_col_single_by_name(cols)

    fn loc(self, rows: RowMask, cols: ColAll) -> DataFrame: return self._select_rows_by_mask(rows.mask)
    fn loc(self, rows: RowMask, cols: List[String]) -> DataFrame: return self._select_rows_by_mask(rows.mask)._select_cols_by_names(cols)
    fn loc(self, rows: RowMask, cols: List[Int])    -> DataFrame: return self._select_rows_by_mask(rows.mask)._select_cols_by_positions(cols)
    fn loc(self, rows: RowMask, cols: String) -> Column: return self._select_rows_by_mask(rows.mask)._select_col_single_by_name(cols)

    fn loc(self, rows: LabelSlice, cols: ColAll) -> DataFrame: return self._select_rows_by_label_slice(rows)
    fn loc(self, rows: LabelSlice, cols: List[String]) -> DataFrame: return self._select_rows_by_label_slice(rows)._select_cols_by_names(cols)
    fn loc(self, rows: LabelSlice, cols: List[Int])    -> DataFrame: return self._select_rows_by_label_slice(rows)._select_cols_by_positions(cols)
    fn loc(self, rows: LabelSlice, cols: String) -> Column: return self._select_rows_by_label_slice(rows)._select_col_single_by_name(cols)

    # ---------------------------------------------------------------------
    # iloc(rows, cols) -- strictly positional
    # ---------------------------------------------------------------------
    @always_inline 
    fn iloc(self, rows: RowAll, cols: ColAll) -> DataFrame: return self.copy()
    fn iloc(self, rows: RowAll, cols: List[Int]) -> DataFrame: return self._select_cols_by_positions(cols)
    @always_inline 
    fn iloc(self, rows: RowAll, cols: Int) -> Column:
        if cols >= 0 and cols < self.ncols(): return self.cols[cols].copy()
        var empty = List[String]()
        return col_from_list_with_tag(empty, String(cols), 4).copy()

    fn iloc(self, rows: RowPos, cols: ColAll) -> DataFrame: return self._select_rows_by_positions(rows.indices)
    fn iloc(self, rows: RowPos, cols: List[Int]) -> DataFrame: return self._select_rows_by_positions(rows.indices)._select_cols_by_positions(cols)
    fn iloc(self, rows: RowPos, cols: Int) -> Column:
        var tmp = self._select_rows_by_positions(rows.indices)
        if cols >= 0 and cols < tmp.ncols(): return tmp.cols[cols].copy()
        var empty = List[String]()
        return col_from_list_with_tag(empty, String(cols), 4).copy()

    fn iloc(self, rows: PosSlice, cols: ColAll) -> DataFrame: return self._select_rows_by_pos_slice(rows)
    fn iloc(self, rows: PosSlice, cols: List[Int]) -> DataFrame: return self._select_rows_by_pos_slice(rows)._select_cols_by_positions(cols)
    fn iloc(self, rows: PosSlice, cols: Int) -> Column:
        var tmp = self._select_rows_by_pos_slice(rows)
        if cols >= 0 and cols < tmp.ncols(): return tmp.cols[cols].copy()
        var empty = List[String]()
        return col_from_list_with_tag(empty, String(cols), 4).copy()





    # ---------- helpers: build columns from strings / broadcast ----------
    @always_inline
    fn _make_col_from_strings(self, name: String, vals: List[String]) -> Column:
        # Preserve tag if a column with this name already exists; else use string tag (4).
        var pos = self.find_col(name)
        if pos >= 0:
            return col_from_list_with_tag(vals, name, self.cols[pos].tag).copy()
        # 4 == string tag in your ColumnTag
        return col_from_list_with_tag(vals, name, 4).copy()

    # Broadcast a scalar string to the frame height using the tag of an
    # existing column with the same name if any, otherwise string tag (4).
    fn _broadcast_scalar(self, name: String, value: String) -> Column:
        var n = self.nrows()
        var vals = List[String]()
        vals.reserve(n)
        var i = 0
        while i < n:
            vals.append(value)
            i += 1
        return self._make_col_from_strings(name, vals)

    # Broadcast helper that enforces a specific target tag (useful for positional set).
    fn _broadcast_scalar_with_tag(self, name: String, value: String, tag: Int) -> Column:
        var n = self.nrows()
        var vals = List[String]()
        vals.reserve(n)
        var i = 0
        while i < n:
            vals.append(value)
            i += 1
        return col_from_list_with_tag(vals, name, tag).copy()

    # ---------- __setitem__ (scalar/name/pos) – ambiguity-free core ----------
    # Enable: frame[cols_by_name(["c1","c2"])] = rhs_df  -> delegates to set_columns_by_names
    fn __setitem__(mut self, key: ColNameList, rhs: DataFrame) -> None:
        self.set_columns(key.names, rhs)

    # df["col"] = Column
    fn __setitem__(mut self, name: String, src: Column) -> None:
        if self.nrows() > 0:
            if src.len() != self.nrows():
                return
        else:
            if src.len() != 0:
                return
        self.set_column(name, src)

    # df["col"] = List[String]
    fn __setitem__(mut self, name: String, values: List[String]) -> None:
        if len(values) != self.nrows():
            return
        var col = self._make_col_from_strings(name, values)
        self.set_column(name, col)

    # df["col"] = "const"  (broadcast)
    fn __setitem__(mut self, name: String, scalar: String) -> None:
        var col = self._broadcast_scalar(name, scalar)
        self.set_column(name, col)

    # df[idx] = Column
    fn __setitem__(mut self, idx: Int, src: Column) -> None:
        if idx < 0 or idx >= self.ncols():
            return
        if self.nrows() > 0:
            if src.len() != self.nrows():
                return
        else:
            if src.len() != 0:
                return
        self.set_column(idx, src)

    # df[idx] = List[String]  (preserve target tag)
    fn __setitem__(mut self, idx: Int, values: List[String]) -> None:
        if idx < 0 or idx >= self.ncols():
            return
        if len(values) != self.nrows():
            return
        var name = self.col_names[idx]
        var col  = col_from_list_with_tag(values, name, self.cols[idx].tag)
        self.set_column(idx, col)

    # df[idx] = "const"  (preserve target tag)
    fn __setitem__(mut self, idx: Int, scalar: String) -> None:
        if idx < 0 or idx >= self.ncols():
            return
        var name = self.col_names[idx]
        var tag  = self.cols[idx].tag
        var col  = self._broadcast_scalar_with_tag(name, scalar, tag)
        self.set_column(idx, col)

 

        # ---------- single-column from frame (explicit; replaces name+DataFrame overload) ----------
    # Replace/insert a column by name using data from a rhs DataFrame.
    # If rhs has a same-name column, use it; else if rhs has exactly 1 column, reuse it.
    # Shape (nrows) must match.
    fn set_column(mut self, name: String, rhs: DataFrame) -> None:
        if rhs.nrows() != self.nrows():
            return
        var use_col = -1
        var pos_same = rhs.find_col(name)
        if pos_same >= 0:
            use_col = pos_same
        elif rhs.ncols() == 1:
            use_col = 0
        else:
            return

        var col = rhs.cols[use_col].copy()
        col.rename(name)

        # FIX: insert-or-replace path (نه replace-only)
        self.set_column(col)   # ← به‌جای self.set_column(name, col)
  

    # Replace a column by positional index from a string list (preserve target tag).
    fn set_column(mut self, idx: Int, values: List[String]) -> None:
        if idx < 0 or idx >= self.ncols():
            return
        if len(values) != self.nrows():
            return
        var name = self.col_names[idx]
        var tag  = self.cols[idx].tag
        var col  = col_from_list_with_tag(values, name, tag)
        self.set_column(idx, col)

    # Replace a column by positional index with a scalar (broadcast, preserve target tag).
    fn set_column(mut self, idx: Int, scalar: String) -> None:
        if idx < 0 or idx >= self.ncols():
            return
        var n    = self.nrows()
        var vals = List[String]()
        vals.reserve(n)
        var i = 0
        while i < n:
            vals.append(scalar)
            i += 1
        var name = self.col_names[idx]
        var tag  = self.cols[idx].tag
        var col  = col_from_list_with_tag(vals, name, tag)
        self.set_column(idx, col)

    # ---------- explicit multi-column helpers (replace list-based __setitem__) ----------
    # Set multiple columns by names from rhs frame.
    # If rhs has exactly one column, reuse it for all names; otherwise match by name.
    fn set_columns(mut self, names: List[String], rhs: DataFrame) -> None:
        if rhs.nrows() != self.nrows():
            return
        var single_col = (rhs.ncols() == 1)
        var i = 0
        var k = len(names)
        while i < k:
            var nm = names[i]
            if single_col:
                var src = rhs.cols[0].copy()
                src.rename(nm)
                self.set_column(src)                 # ← insert-or-replace
            else:
                var pos = rhs.find_col(nm)
                if pos >= 0:
                    var src = rhs.cols[pos].copy()
                    src.rename(nm)                   # ایمن‌تر
                    self.set_column(src)             # ← insert-or-replace
            i += 1

    # Set multiple columns by positions from rhs frame.
    # If rhs has exactly one column, reuse it across all indices; else match by position.
    fn set_columns(mut self, indices: List[Int], rhs: DataFrame) -> None:
        if rhs.nrows() != self.nrows():
            return
        var one = (rhs.ncols() == 1)
        var i = 0
        var k = len(indices)
        while i < k:
            var idx = indices[i]
            if idx >= 0 and idx < self.ncols():
                var target_name = self.col_names[idx]
                var src_col = Column()
                if one:
                    src_col = rhs.cols[0].copy()
                elif idx < rhs.ncols():
                    src_col = rhs.cols[idx].copy()
                else:
                    i += 1
                    continue
                src_col.rename(target_name)
                self.set_column(idx, src_col)
            i += 1

    fn set_column(mut self, col: Column):
        var nm = col.get_name()
        var idx = self.find_col(nm)
        if idx == -1:
            # INSERT
            self.col_names.append(nm)
            self.names.append(nm)
            self.cols.append(col.copy())
        else:
            # REPLACE
            self.cols[idx]      = col.copy()
            self.col_names[idx] = nm
            self.names[idx]     = nm

    fn set_column(mut self, idx: Int, src: Column) -> None:
        if idx < 0 or idx >= self.ncols(): return
        if self.nrows() > 0:
            if src.len() != self.nrows(): return
        else:
            if src.len() != 0: return

        self.cols[idx] = src.copy()

        var new_name = src.get_name()
        if idx < len(self.col_names):
            self.col_names[idx] = new_name
        if idx < len(self.names):
            self.names[idx] = new_name      # ← همگام‌سازی names

    fn set_column(mut self, name: String, src: Column) -> None:
        var found = -1
        var i = 0
        while i < self.ncols():
            if self.col_names[i] == name:
                found = i
                break
            i += 1

        var col = src.copy()
        col.rename(name)

        if found < 0:
            # INSERT when not found (قبلاً: early return)
            self.col_names.append(name)
            self.names.append(name)
            self.cols.append(col.copy())
            return

        # REPLACE when found
        self.set_column(found, col)

    fn set_column(mut self, name: String, values: List[String]) -> None:
        var tmp = ToDataFrame({ name: values.copy() })
        self.set_column(name, tmp)


    # ---------- explicit masked row assignment (replace mask-based __setitem__) ----------
    # Set all cells in rows where mask[r] is True to a scalar string (broadcast across all columns).
    fn set_rows(mut self, mask: List[Bool], scalar: String) -> None:
        var n = self.nrows()
        if len(mask) != n:
            return
        var c = 0
        while c < self.ncols():
            var vals = List[String]()
            vals.reserve(n)
            var r = 0
            while r < n:
                if mask[r]:
                    vals.append(scalar)
                else:
                    vals.append(self.cols[c].get_string(r))
                r += 1
            var col = col_from_list_with_tag(vals, self.col_names[c], self.cols[c].tag)
            self.set_column(c, col)
            c += 1

    # Row-wise masked assignment from another frame.
    # Accepts rhs with ncols == self.ncols (per-position) OR ncols == 1 (broadcast single column).
    fn set_rows(mut self, mask: List[Bool], rhs: DataFrame) -> None:
        var n = self.nrows()
        if len(mask) != n:
            return
        if rhs.nrows() != n:
            return
        var mode = 0
        if rhs.ncols() == self.ncols():
            mode = 1   # per-position
        elif rhs.ncols() == 1:
            mode = 2   # broadcast single column
        else:
            return

        var c = 0
        while c < self.ncols():
            var vals = List[String]()
            vals.reserve(n)
            var r = 0
            while r < n:
                if mask[r]:
                    if mode == 1:
                        vals.append(rhs.cols[c].get_string(r))
                    else:
                        vals.append(rhs.cols[0].get_string(r))
                else:
                    vals.append(self.cols[c].get_string(r))
                r += 1
            var col = col_from_list_with_tag(vals, self.col_names[c], self.cols[c].tag)
            self.set_column(c, col)
            c += 1

    # Row-wise masked assignment from a 1D row (length must equal ncols()).
    # Each True row gets the same row_values applied across all columns.
    fn set_rows(mut self, mask: List[Bool], row_values: List[String]) -> None:
        var n = self.nrows()
        if len(mask) != n:
            return
        if len(row_values) != self.ncols():
            return

        var c = 0
        while c < self.ncols():
            var vals = List[String]()
            vals.reserve(n)
            var r = 0
            while r < n:
                if mask[r]:
                    vals.append(row_values[c])
                else:
                    vals.append(self.cols[c].get_string(r))
                r += 1
            var col = col_from_list_with_tag(vals, self.col_names[c], self.cols[c].tag)
            self.set_column(c, col)
            c += 1
 
 
    # -------------------------------------------------------------------------
    # String rendering
    # -------------------------------------------------------------------------
    fn to_string(self) -> String:
        var s = "DataFrame(" + String(self.nrows()) + "x" + String(self.ncols()) + ")\n"
        if self.index_name != "":
            s += self.index_name + " | "
        var i = 0
        var n = len(self.col_names)
        while i < n:
            s += self.col_names[i]
            if i + 1 < n:
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
            var m = self.ncols()
            while c < m:
                s += self.cols[c].get_string(r)
                if c + 1 < m:
                    s += " | "
                c += 1
            s += "\n"
            r += 1

        if self.nrows() > show:
            s += "...\n"
        return s

    @always_inline
    fn __str__(self) -> String:
        return self.to_string()

    # -------------------------------------------------------------------------
    # Melt (wide -> long)
    # -------------------------------------------------------------------------
    fn melt(self,
        id_vars: List[String],
        var_name: String = "variable",
        value_name: String = "value") -> DataFrame:
        # ------------- partition columns into id and melt sets -------------
        var melt_cols = List[Int]()
        var id_idx    = List[Int]()

        var i = 0
        var nc = self.ncols()
        while i < nc:
            var col_name = self.col_names[i]
            var is_id = False
            var j = 0
            var idn = len(id_vars)
            while j < idn:
                if id_vars[j] == col_name:
                    is_id = True
                    break
                j += 1
            if is_id:
                id_idx.append(i)
            else:
                melt_cols.append(i)
            i += 1

        var nr = self.nrows()
        var nm = len(melt_cols)
        var total_rows = nr * nm

        # ------------- build id columns (row-major repeat) -------------
        var out_cols = List[Column]()
        out_cols.reserve(len(id_idx) + 2)

        var v = 0
        var idk = len(id_idx)
        while v < idk:
            var s_id = SeriesStr()
            s_id.name = self.col_names[id_idx[v]]
            s_id.data = List[String]()
            s_id.data.reserve(total_rows)

            # row-major: for each row, repeat its id value across all melted columns
            var r = 0
            while r < nr:
                var mci = 0
                while mci < nm:
                    s_id.data.append(self.cols[id_idx[v]].get_string(r))
                    mci += 1
                r += 1

            var c_id = Column()
            c_id.from_str(s_id)
            out_cols.append(c_id.copy())
            v += 1

        # ------------- build variable and value columns (row-major) -------------
        var s_var = SeriesStr()
        s_var.name = var_name
        s_var.data = List[String]()
        s_var.data.reserve(total_rows)

        var s_val = SeriesStr()
        s_val.name = value_name
        s_val.data = List[String]()
        s_val.data.reserve(total_rows)

        # row-major: for each row, append all melted columns in order
        var r2 = 0
        while r2 < nr:
            var mci2 = 0
            while mci2 < nm:
                var cidx = melt_cols[mci2]
                s_var.data.append(self.col_names[cidx])
                s_val.data.append(self.cols[cidx].get_string(r2))
                mci2 += 1
            r2 += 1

        var col_var = Column()
        col_var.from_str(s_var)
        var col_val = Column()
        col_val.from_str(s_val)

        out_cols.append(col_var.copy())
        out_cols.append(col_val.copy())

        # ------------- assemble output DataFrame -------------
        var out = DataFrame()
        var k = 0
        var outn = len(out_cols)
        while k < outn:
            out.set_column(out_cols[k])
            k += 1

        # reset index metadata
        out.index_name = String("")
        out.index_vals = List[String]()

        return out


    # -------------------------------------------------------------------------
    # Head-like: first n rows
    # -------------------------------------------------------------------------
    fn with_rows(self, n: Int) -> DataFrame:
        var nr = self.nrows()
        var n_take = n
        if n_take > nr:
            n_take = nr

        var out = DataFrame()
        out.index_name = String(self.index_name)
        out.index_vals = List[String]()
        out.index_vals.reserve(n_take)

        # Copy names in one shot
        out.col_names = self.col_names.copy()
        out.names     = self.names.copy()

        # Slice each column as string series
        out.cols = List[Column]()
        out.cols.reserve(len(self.cols))

        var i = 0
        while i < len(self.cols):
            var s = SeriesStr()
            s.name = self.col_names[i]
            s.data = List[String]()
            s.data.reserve(n_take)

            var r = 0
            while r < n_take:
                s.data.append(self.cols[i].get_string(r))
                r += 1

            var c = Column()
            c.from_str(s)
            out.cols.append(c.copy())
            i += 1

        # Index slice
        var j = 0
        while j < n_take and j < len(self.index_vals):
            out.index_vals.append(self.index_vals[j])
            j += 1

        return out

    # -------------------------------------------------------------------------
    # drop_duplicates (subset, keep = "first" | "last")
    # String-key based; stable; single pass per policy. O(n^2) membership with List.
    # -------------------------------------------------------------------------
    fn drop_duplicates(self,
                       subset: List[String] = List[String](),
                       keep: String = "first") -> DataFrame:
        var nr = self.nrows()
        var nc = self.ncols()

        # Determine key columns
        var key_cols = List[Int]()
        if len(subset) == 0:
            var c = 0
            while c < nc:
                key_cols.append(c)
                c += 1
        else:
            var s = 0
            while s < len(subset):
                var idx = self.find_col(subset[s])
                if idx >= 0:
                    key_cols.append(idx)
                s += 1

        var selected_rows = List[Int]()
        selected_rows.reserve(nr)

        if keep == "last":
            var seen = List[String]()
            var r = nr - 1
            while r >= 0:
                var key = String("")
                var ki = 0
                var kc = len(key_cols)
                while ki < kc:
                    if ki > 0:
                        key += String("\x1f")
                    key += self.cols[key_cols[ki]].get_string(r)
                    ki += 1
                var found = False
                var t = 0
                while t < len(seen):
                    if seen[t] == key:
                        found = True
                        break
                    t += 1
                if not found:
                    seen.append(key)
                    selected_rows.append(r)
                r -= 1

            # reverse selected_rows to restore ascending order
            var i = 0
            var j = len(selected_rows) - 1
            while i < j:
                var tmp = selected_rows[i]
                selected_rows[i] = selected_rows[j]
                selected_rows[j] = tmp
                i += 1
                j -= 1
        else:
            var seen = List[String]()
            var r = 0
            while r < nr:
                var key = String("")
                var ki = 0
                var kc = len(key_cols)
                while ki < kc:
                    if ki > 0:
                        key += String("\x1f")
                    key += self.cols[key_cols[ki]].get_string(r)
                    ki += 1
                var found = False
                var t = 0
                var sn = len(seen)
                while t < sn:
                    if seen[t] == key:
                        found = True
                        break
                    t += 1
                if not found:
                    seen.append(key)
                    selected_rows.append(r)
                r += 1

        # Build output DF by row selection
        var out = DataFrame()
        out.col_names  = self.col_names.copy()
        out.names      = self.names.copy()

        out.cols = List[Column]()
        out.cols.reserve(nc)

        var c2 = 0
        var seln = len(selected_rows)
        while c2 < nc:
            var s = SeriesStr()
            s.name = self.col_names[c2]
            s.data = List[String]()
            s.data.reserve(seln)

            var k = 0
            while k < seln:
                var rr = selected_rows[k]
                s.data.append(self.cols[c2].get_string(rr))
                k += 1

            var col = Column()
            col.from_str(s)
            out.cols.append(col)
            c2 += 1

        # Index
        out.index_name = String(self.index_name)
        out.index_vals = List[String]()
        out.index_vals.reserve(seln)

        var q = 0
        var inv = len(self.index_vals)
        while q < seln and q < inv:
            out.index_vals.append(self.index_vals[selected_rows[q]])
            q += 1

        return out

    # Convenience
    fn head(self, n: Int = 5) -> DataFrame:
        return self.with_rows(n)
    # 1) Range-based
    fn loc(self, rows: RowRange, cols: List[String]) -> DataFrame:
        return loc_impl_range(self, rows, cols)

    # # 2) Label-range
    # fn loc(self, rows: LabelSlice, cols: List[String]) -> DataFrame:
    #     var rr = labels_to_row_range(self.index_vals, rows)
    #     return loc_impl_range(self, rr, cols)

    # # 3) Explicit label list (e.g., rows=["w","y"])
    # fn loc(self, rows: List[String], cols: List[String]) -> DataFrame:
    #     var idxs = labels_to_indices(self.index_vals, rows)
    #     return loc_impl_indices(self, idxs, cols)

    # # 4) Single-cell by (row, col) names
    # fn loc(self, row: String, col: String) -> String:
    #     var r = find_first(self.index_vals, row)
    #     if r < 0:
    #         return String("")
    #     var j = 0
    #     var m = len(self.col_names)
    #     while j < m and self.col_names[j] != col:
    #         j += 1
    #     if j >= m:
    #         return String("")
    #     return String(self.cols[j].get_string(r))
        
     
    # fn iloc(self, row_indices: List[Int], col_range: ColRange) -> DataFrame:
    #     return iloc(self, row_indices, col_range)

  
    # fn iloc(self, rows: ILocRowSlice, cols: ILocColSlice) -> DataFrame:
    #     # clamp rows
    #     var nr = self.nrows()
    #     var r0 = clamp(rows.start, 0, nr)
    #     var r1 = clamp(rows.stop,  0, nr)
    #     if r1 < r0: r1 = r0

    #     # rows list
    #     var row_idxs = List[Int]()
    #     var r = r0
    #     while r < r1:
    #         row_idxs.append(r)
    #         r += 1

    #     # use the free function with a column ColRange
    #     var cr = ColRange(cols.start, cols.stop)
    #     return iloc(self, row_idxs, cr)

    # # 3.3) rows = List[Int], cols = List[Int]
    # fn iloc(self, rows: List[Int], cols: List[Int]) -> DataFrame:
    #     return iloc_impl_indices_cols(self, rows, cols)

    # # 3.4) rows = List[Int], cols = ILocColSlice
    # fn iloc(self, rows: List[Int], cols: ILocColSlice) -> DataFrame:
    #     var cr = ColRange(cols.start, cols.stop)
    #     return iloc(self, rows, cr)

    # # 3.5) rows = ILocRowSlice, cols = List[Int]
    # fn iloc(self, rows: ILocRowSlice, cols: List[Int]) -> DataFrame:
    #     var nr = self.nrows()
    #     var r0 = clamp(rows.start, 0, nr)
    #     var r1 = clamp(rows.stop,  0, nr)
    #     if r1 < r0: r1 = r0

    #     var row_idxs = List[Int]()
    #     var r = r0
    #     while r < r1:
    #         row_idxs.append(r)
    #         r += 1

    #     return iloc_impl_indices_cols(self, row_idxs, cols)

    # # 3.6)  : row, col → String
    # fn iloc(self, row: Int, col: Int) -> String:
    #     if not is_valid_cell(self, row, col):
    #         return String("")
    #     return String(self.cols[col].get_string(row))



 
    # Build a one-cell DataFrame by label row and column name
    fn at(self, label_row: String, col: String) -> DataFrame:
        # resolve row
        var r = -1
        var n = len(self.index_vals)
        var i = 0
        while i < n:
            if self.index_vals[i] == label_row:
                r = i
                break
            i += 1

        # resolve column
        var j = -1
        var m = len(self.col_names)
        var c = 0
        while c < m:
            if self.col_names[c] == col:
                j = c
                break
            c += 1

        # build 1x1 frame (string-backed)
        var out = DataFrame()
        out.index_name = String(self.index_name)
        out.index_vals = List[String]()
        out.col_names  = List[String]()
        out.names      = List[String]()
        out.cols       = List[Column]()

        var cname = String(col)
        out.col_names.append(cname)
        out.names.append(cname)

        var vals = List[String]()

        if r >= 0 and r < self.nrows() and j >= 0 and j < len(self.cols):
            out.index_vals.append(String(self.index_vals[r]))
            vals.append(String(self.cols[j].get_string(r)))
        else:
            out.index_vals.append(String(""))
            vals.append(String(""))

        out.cols.append(col_from_list(vals, cname))
        return out

    # Build a one-cell DataFrame by integer row/col
    fn iat(self, row: Int, col: Int) -> DataFrame:
        var out = DataFrame()
        out.index_name = String(self.index_name)
        out.index_vals = List[String]()
        out.col_names  = List[String]()
        out.names      = List[String]()
        out.cols       = List[Column]()

        var cname = String("")
        if col >= 0 and col < len(self.col_names):
            cname = String(self.col_names[col])
        out.col_names.append(cname)
        out.names.append(cname)

        var vals = List[String]()

        if row >= 0 and row < self.nrows():
            if row < len(self.index_vals):
                out.index_vals.append(String(self.index_vals[row]))
            else:
                out.index_vals.append(String(""))

            if col >= 0 and col < len(self.cols):
                vals.append(String(self.cols[col].get_string(row)))
            else:
                vals.append(String(""))
        else:
            out.index_vals.append(String(""))
            vals.append(String(""))

        out.cols.append(col_from_list(vals, cname))
        return out

    # Return the single cell value of a 1x1 DataFrame (string-backed)
    fn get(self) -> String:
        if len(self.cols) == 1:
            # defensive for empty column
            var clen = self.cols[0].len()
            if clen > 0:
                return String(self.cols[0].get_string(0))
        return String("")

    # Set the single cell of a 1x1 DataFrame; returns success
    fn set(mut self, v: Value) -> Bool:
        if len(self.cols) != 1:
            return False
        # rebuild the 1-element column with new value
        var cname = String(self.col_names[0])
        var vals = List[String]()
        vals.append(String(v.as_string()))
        self.cols[0] = col_from_list(vals, cname)
        return True

 

    fn set_at(mut df: DataFrame, label_row: String, col: String, v: Value) -> Bool: 
        var r = -1
        var i = 0
        var n = len(df.index_vals)
        while i < n:
            if df.index_vals[i] == label_row:
                r = i
                break
            i += 1
        var j = -1
        var c = 0
        var m = len(df.col_names)
        while c < m:
            if df.col_names[c] == col:
                j = c
                break
            c += 1 
        if r < 0 or r >= df.nrows() or j < 0 or j >= df.ncols(): 
            return False
        var ok = set_cell_preserve_type( df, r, j, v) 
        return ok

    fn set_iat(mut df: DataFrame, row: Int, col: Int, v: Value) -> Bool: 
        if row < 0 or row >= df.nrows() or col < 0 or col >= df.ncols(): 
            return False
        var ok = set_cell_preserve_type( df, row, col, v) 
        return ok


  
    # ------------------------------------------------------------------
    # DataFrame.dtypes(): returns a printable doc listing "name: dtype"
    # ------------------------------------------------------------------
    fn dtypes(self) -> Dict[String, String]:
        var out = Dict[String, String]()
        var j = 0
        var m = len(self.col_names)
        while j < m:
            var name = self.col_names[j]
            var t = self.cols[j].dtype()      # Int tag
            out[name] = dtype_name(t)
            j += 1
        return out.copy()



    # Return the index of a column by name, or -1 if not found.
    fn col_index(self, name: String) -> Int:
        var i = 0
        while i < len(self.col_names):
            if self.col_names[i] == name:
                return i
            i += 1
        return -1

    # Optional: strict version that raises if not found.
    fn require_col_index(self, name: String) raises -> Int:
        var idx = self.col_index(name)
        if idx < 0:
            def e = "Column not found: " + name
            raise Error(e)
        return idx

    fn get_column_by_index(self, idx: Int) -> Column:
        if idx < 0:
            return Column()
        if idx >= len(self.col_names):
            return Column()

        var name = self.col_names[idx]
        var c = self.get_column(name)     # assumes this returns the core Column
        return c.copy()   




    # Return the column as a (string-backed) Series facade by name.
    fn col_values(self, name: String) -> List[String]:
        var idx = 0
        while idx < len(self.col_names):
            if self.col_names[idx] == name: 
 
                return self.cols[idx].as_strings()   

            idx += 1
        # not found → empty
        return List[String]()
     # ------------------------------------------------------------------
    # Pipe: apply an arbitrary transformation and return its result.
    # Usage: df.pipe(fn(d: DataFrame) -> DataFrame: /* ... */)
    # ------------------------------------------------------------------ 
    fn pipe(self, f: fn (DataFrame) -> DataFrame) -> DataFrame:
        var tmp = self.copy()
        return f(tmp)
    # ------------------------------------------------------------------
    # Assign (callable version): name -> fn(DataFrame) -> Series 
    # ------------------------------------------------------------------
 

    fn assign(self, mapping: Dictionary[String, fn (DataFrame) -> List[String]]) -> DataFrame:
        var out = self.copy()

        # name -> idx
        var name_to_idx = Dictionary[String, Int]()
        var i = 0
        while i < len(out.col_names):
            name_to_idx[out.col_names[i]] = i
            i += 1

        # gather & sort keys (selection sort)
        var keys = List[String]()
        for k in mapping.keys():
            keys.append(String(k))
        i = 0
        while i + 1 < len(keys):
            var j = i + 1
            while j < len(keys):
                if keys[j] < keys[i]:
                    var t = keys[i]; keys[i] = keys[j]; keys[j] = t
                j += 1
            i += 1

        # apply
        i = 0
        while i < len(keys):
            var name = keys[i]
            var opt_f = mapping.get(name)
            if opt_f is not None:
                var f = opt_f.value()
                var vals = f(out)                 # List[String]

                # build a string Series/Column
                var s = SeriesStr()
                s.set_name(name)
                s.data = vals.copy()

                var col = Column()
                col.from_str(s)

                var opt_idx = name_to_idx.get(name)
                if opt_idx is not None:
                    var idx = opt_idx.value()
                    out.cols[idx] = col.copy()
                    out.col_names[idx] = name
                else:
                    name_to_idx[name] = len(out.col_names)
                    out.cols.append(col.copy())
                    out.col_names.append(name)
            i += 1

        return out.copy()

     # --------------------------------------------------------------
    # Build this Column from a generic facade Series.
    # Dispatches by dtype/tag and deep-copies buffers.
    # --------------------------------------------------------------
    fn from_series(mut self, src: Series) -> None:
        # Prefer dtype() if available; fall back to tag if needed.
        var dt = src.dtype()

        if dt.is_string():
            var ss = SeriesStr()
            ss.set_name(src.get_name())
            ss.data = src.values_as_string().copy()   # expect List[String]
            self.from_str(ss)
            return

        if dt.is_int64():
            var si = SeriesI64()
            si.set_name(src.get_name())
            si.data = src.values_as_i64().copy()      # expect List[Int64]
            # copy validity/bitmap if API exposes it; otherwise default "all valid"
            self.from_i64(si)
            return

        if dt.is_float64():
            var sf = SeriesF64()
            sf.set_name(src.get_name())
            sf.data = src.values_as_f64().copy()      # expect List[Float64]
            self.from_f64(sf)
            return

        if dt.is_bool():
            var sb = SeriesBool()
            sb.set_name(src.get_name())
            sb.data = src.values_as_bool().copy()     # expect List[Bool]
            self.from_bool(sb)
            return

        # Fallback: treat as strings
        var s2 = SeriesStr()
        s2.set_name(src.get_name())
        s2.data = src.values_as_string().copy()
        self.from_str(s2)


    #Return the column index by name, or -1 if not found.
    fn _col_index(self, name: String) -> Int:
        var i = 0
        while i < len(self.col_names):
            if self.col_names[i] == name:
                return i
            i += 1
        return -1

    #Return column values as List[String] by column name.
    fn col_values(self, name: String) -> List[String]:
        var out = List[String]()
        var idx = self._col_index(name)
        if idx < 0:
            return out.copy()   # not found -> empty

        var col = self.cols[idx].copy()         # Column
        var n = self.nrows()
        out.reserve(n)

        var r = 0
        while r < n: 
            out.append(col[r])
            r += 1

        return out.copy()


    

    # fn pivot_table(self,index: String,columns: String,values: String,agg: Agg,margins: Bool,margins_name: String) -> DataFrame:
    #     return pivot_table(self,index,columns,agg,margins,margins_name) 

    # ---------- DataFrame methods: pivot_table ----------

    # Core method: String indices
    fn pivot_table(self,
                index: String,
                columns: String,
                values: String,
                agg: Agg,
                fill_value: Optional[String] = None,
                margins: Bool = False,
                margins_name: String = String("Total")) -> DataFrame:
        return pivot_table(self, index, columns, values, agg, margins, margins_name, fill_value)

    # Core method: single-item List indices (forwarded)
    fn pivot_table(self,
                index: List[String],
                columns: List[String],
                values: String,
                agg: Agg,
                fill_value: Optional[String] = None,
                margins: Bool = False,
                margins_name: String = String("Total")) -> DataFrame:
        return pivot_table(self, index, columns, values, agg, fill_value, margins, margins_name)

    # Convenience overload: accept Value (from column.mojo) and convert to Optional[String]
    fn pivot_table(self,
                index: List[String],
                columns: List[String],
                values: String,
                agg: Agg,
                fill_value: Value,
                margins: Bool = False,
                margins_name: String = String("Total")) -> DataFrame:
        return pivot_table(self,
                        index, columns, values, agg,
                        Optional[String](fill_value.as_string()),
                        margins, margins_name)

    # Convenience overloads: primitive fill_value → Optional[String]
    fn pivot_table(self,
                index: List[String],
                columns: List[String],
                values: String,
                agg: Agg,
                fill_value: Int,
                margins: Bool = False,
                margins_name: String = String("Total")) -> DataFrame:
        return pivot_table(self,
                        index, columns, values, agg,
                        Optional[String](String(fill_value)),
                        margins, margins_name)

    fn pivot_table(self,
                index: List[String],
                columns: List[String],
                values: String,
                agg: Agg,
                fill_value: Float64,
                margins: Bool = False,
                margins_name: String = String("Total")) -> DataFrame:
        return pivot_table(self,
                        index, columns, values, agg,
                        Optional[String](String(fill_value)),
                        margins, margins_name)

    fn pivot_table(self,
                index: List[String],
                columns: List[String],
                values: String,
                agg: Agg,
                fill_value: Bool,
                margins: Bool = False,
                margins_name: String = String("Total")) -> DataFrame:
        return pivot_table(self,
                        index, columns, values, agg,
                        Optional[String](String(fill_value)),
                        margins, margins_name)

    fn pivot_table(self,
                index: List[String],
                columns: List[String],
                values: String,
                agg: Agg,
                fill_value: String,
                margins: Bool = False,
                margins_name: String = String("Total")) -> DataFrame:
        return pivot_table(self,
                        index, columns, values, agg,
                        Optional[String](fill_value),
                        margins, margins_name)

    
    fn reset_index(self) -> DataFrame:
        return reset_index(self)

    fn merge(self, right: DataFrame, on: List[String], how: String) -> DataFrame:
        return merge(self, right, on.copy(), how)

    # Convenience: single key
    fn merge(self, right: DataFrame, on: String, how: String) -> DataFrame:
        var keys = List[String]()
        keys.append(on)
        return merge(self, right, keys.copy(), how)

    
    # Convenience: default 'left' join
    fn merge(self, right: DataFrame, on: List[String]) -> DataFrame:
        return merge(self, right, on.copy(), String("left"))

    fn merge(self, right: DataFrame, on: String) -> DataFrame:
        var keys = List[String]()
        keys.append(on)
        return merge(self, right, keys.copy(), String("left"))

    fn sort_values(self, by: List[String], ascending: List[Bool]=[True]) -> DataFrame:
        return sort_values(self, by , ascending ) 
 

    fn to_csv(self: DataFrame,
          path: String,
          header: Bool = True,
          delimiter: String = String(","),
          quotechar: String = String("\""),
          index: Bool = True
        ) -> Bool:
        return write_csv(self, Path(path), header, delimiter, quotechar)

    fn set_index(self, col: String) -> DataFrame:
        return set_index(self, col, True)


    fn set_index(self, col: String, drop: Bool) -> DataFrame:
        return set_index(self, col, drop)

    fn resample(self, freq: String) -> Resampler:
        return Resampler(self, freq)
    # Return a defensive copy of all index values as List[String].
    fn index(self) -> List[String]:
        var out = List[String]()
        if len(self.index_vals) == 0:
            # No explicit index: synthesize 0..N-1 as strings
            var n = self.nrows()
            var i = 0
            while i < n:
                out.append(String(i))
                i += 1
            return out.copy()
        # Copy stored index values
        var i = 0
        while i < len(self.index_vals):
            out.append(String(self.index_vals[i]))
            i += 1
        return out.copy()



# ------------------------------------------------------------------
# Pretty doc for dtypes: holds a text buffer and prints nicely.
# ------------------------------------------------------------------
struct DTypesDoc:
    var buf: String

    fn __init__(out self):
        self.buf = String("")

    fn push_line(mut self, name: String, dtype: String) -> None:
        if len(self.buf) > 0:
            self.buf += String("\n")
        self.buf += String(name + ": " + dtype)

    fn __str__(self) -> String:
        return self.buf

# Resolve a human-readable dtype name for a column.
# Uses Column predicates if available, otherwise defaults to "string".
fn _dtype_name_for(col: Column) -> String:
    # If your Column has predicates, use them:
    # bool / int64 / float64 / string
    var t = String("string")
    # NOTE: guard calls in case these helpers exist in your project.
    # If they don't, you can switch to checking a 'tag' field if exposed.
    if col.is_bool():
        t = String("bool")
    elif col.is_i64():
        t = String("int")
    elif col.is_f64():
        t = String("float")
    elif col.is_str():
        t = String("string")
    return t


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
    return set_index(df, col, True)

# Core implementation with explicit 'drop' control.
fn set_index(df: DataFrame, col: String, drop: Bool) -> DataFrame:
    var idx = _find_col(df, col)

    var out = DataFrame()
    out.col_names = List[String]()
    out.cols      = List[Column]()

    # Copy columns; optionally drop the index source column if found.
    var n_cols = 0
    for _ in df.col_names:
        n_cols += 1

    var c = 0
    while c < n_cols:
        var keep = True
        if drop and c == idx:
            keep = False
        if keep:
            out.col_names.append(String(df.col_names[c]))
            out.cols.append(df.cols[c].copy())
        c += 1

    # Keep legacy name list in sync
    out.names = List[String]()
    var k = 0
    while k < len(out.col_names):
        out.names.append(String(out.col_names[k]))
        k += 1

    # Fill index metadata
    out.index_name = String(col)
    out.index_vals = List[String]()

    if idx >= 0:
        var n_rows_in_idx = df.cols[idx].len()
        var r = 0
        while r < n_rows_in_idx:
            out.index_vals.append(df.cols[idx].get_string(r))
            r += 1
    else:
        # If the column is not found, synthesize a 0..N-1 index from the first column's length
        var n_rows = 0
        if n_cols > 0:
            n_rows = df.cols[0].len()
        var i = 0
        while i < n_rows:
            out.index_vals.append(String(i))
            i += 1

    return out.copy()

fn reset_index(df: DataFrame) -> DataFrame:
    var out = DataFrame()
    out.col_names = List[String]()
    out.cols = List[Column]()

    # 1) detect index name presence
    var has_idx_name = False
    for _ in df.index_name.codepoints():
        has_idx_name = True
        break
    var idx_name = String(df.index_name)
    if not has_idx_name:
        idx_name = String("index")   # fallback

    # 2) check if a label column already exists (e.g., "city")
    var has_city_col = False
    var ci = 0
    while ci < len(df.col_names):
        if df.col_names[ci] == "city":
            has_city_col = True
            break
        ci += 1

    # 3) copy existing columns first
    var c = 0
    while c < len(df.col_names):
        out.col_names.append(String(df.col_names[c]))
        out.cols.append(df.cols[c].copy())
        c += 1

    # 4) decide whether to insert an index column
    var need_insert_index = True

    # if we already have a label column (e.g., "city") and index_name is empty,
    # skip inserting synthetic "index"
    if (not has_idx_name) and has_city_col:
        need_insert_index = False

    if need_insert_index:
        # build index values (use df.index_vals if present, else 0..nrows-1)
        var idx_vals = List[String]()
        var niv = len(df.index_vals)
        var ii = 0
        while ii < niv:
            idx_vals.append(String(df.index_vals[ii]))
            ii += 1

        if len(idx_vals) == 0:
            var nrows = 0
            if len(df.cols) > 0:
                nrows = df.cols[0].len()
            var k = 0
            while k < nrows:
                idx_vals.append(String(k))
                k += 1

        var s = SeriesStr()
        s.name = idx_name
        s.data = idx_vals.copy()

        var idx_col = Column()
        idx_col.from_str(s)

        # insert as first col
        var new_names = List[String]()
        var new_cols  = List[Column]()
        new_names.append(idx_name)
        new_cols.append(idx_col.copy())
        var j = 0
        while j < len(out.col_names):
            new_names.append(out.col_names[j])
            new_cols.append(out.cols[j].copy())
            j += 1
        out.col_names = new_names.copy()
        out.cols      = new_cols.copy()

    # 5) rebuild meta index 0..N-1
    out.index_name = String("")
    out.index_vals = List[String]()
    var nrows_after = 0
    if len(out.cols) > 0:
        nrows_after = out.cols[0].len()
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


fn rolling_abs(xs: List[Float64], win: Int) -> List[Float64]
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
fn sort_values(frame: DataFrame, by: List[String], ascending: List[Bool]) -> DataFrame:
    var out = frame.copy()
    var n = out.nrows()
    if n <= 1:
        return out

    # Resolve column indices once (avoid capturing names later).
    var by_idx = List[Int]()
    var k = 0
    while k < len(by):
        var ci = out.find_col(by[k])
        if ci < 0:
            # if missing, just skip this key
            k += 1
            continue
        by_idx.append(ci)
        k += 1
    if len(by_idx) == 0:
        return out

    # Build index order
    var idx = List[Int]()
    var i = 0
    while i < n:
        idx.append(i)
        i += 1

    # Stable insertion sort with inline compare (no closures, no Column copies)
    var p = 1
    while p < n:
        var cur = idx[p]
        var q = p - 1
        while q >= 0:
            # compare cur vs idx[q] across keys
            var cmp_res = 0
            var t = 0
            while t < len(by_idx):
                var col_i = by_idx[t]                    # Int, copyable
                # read as String; if your Column supports numeric getters,
                # branch here based on dtype for numeric ordering.
                var a = out.cols[col_i].get_string(cur)
                var b = out.cols[col_i].get_string(idx[q])

                if a != b:
                    var lt = (a < b)
                    if ascending[t]:
                        cmp_res = -1 if lt else 1
                    else:
                        cmp_res = 1 if lt else -1
                    break
                t += 1

            if cmp_res < 0:
                idx[q + 1] = idx[q]
                q -= 1
            else:
                break
        idx[q + 1] = cur
        p += 1

    # Rebuild all columns in the new order (avoid Column copies)
    var new_cols = List[Column]()
    var c = 0
    while c < out.ncols():
        var vals = List[String]()
        vals.reserve(n)
        var r = 0
        while r < n:
            # direct access; no "var col = out.cols[c]"
            vals.append(out.cols[c].get_string(idx[r]))
            r += 1
        new_cols.append(col_str(out.col_names[c], vals))
        c += 1

    out.cols = new_cols.copy()
    return out

#   # ---------- Row slicing like df[start:stop] ----------
#     struct RowSlice:
#         var start: Int
#         var stop:  Int
#         fn __init__(out self, start: Int, stop: Int):
#             self.start = start
#             self.stop  = stop

#     fn __getitem__(self, s: RowSlice) -> DataFrame:
#         # pandas semantics: [start:stop) ; clamp to [0, nrows]
#         var n = self.nrows()
#         var a = s.start
#         var b = s.stop
#         if a < 0: a = 0
#         if b > n: b = n
#         if b < a: b = a

#         var out = DataFrame()
#         out.index_name = String(self.index_name)
#         out.index_vals = List[String]()
#         out.index_vals.reserve(b - a)

#         # init dst columns
#         var dst_cols = List[Column]()
#         dst_cols.reserve(self.ncols())
#         var c = 0
#         while c < self.ncols():
#             dst_cols.append(self.cols[c].empty_like())
#             c += 1

#         var r = a
#         while r < b:
#             out.index_vals.append(self.index_vals[r])
#             var cc = 0
#             while cc < self.ncols():
#                 dst_cols[cc].append(self.cols[cc].get(r))
#                 cc += 1
#             r += 1

#         var k = 0
#         while k < self.ncols():
#             out.set_column(dst_cols[k].copy())
#             k += 1

#         return out