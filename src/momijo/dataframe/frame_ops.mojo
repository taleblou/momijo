# Project:      Momijo
# Module:       src.momijo.dataframe.frame_ops
# File:         frame_ops.mojo
# Path:         src/momijo/dataframe/frame_ops.mojo
#
# Description:  src.momijo.dataframe.frame_ops — focused Momijo functionality with a stable public API.
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
#   - Structs: ModuleState
#   - Key functions: __init__, make_module_state, select_columns_safe


struct ModuleState:
    var names
    fn __init__(out self, names):
        self.names = names

fn make_module_state(state) -> ModuleState:
    return ModuleState(List[String]())



from momijo.dataframe.api import df_make
from momijo.dataframe.column import Column, from_bool, from_f64, from_i64, from_str, get_bool, get_f64, get_i64, is_f64, is_i64, name
from momijo.dataframe.frame import DataFrame, get_column_at, width
from momijo.dataframe.helpers import find_col
from momijo.dataframe.io_csv import is_bool
from momijo.dataframe.series_bool import SeriesBool, append
from momijo.dataframe.series_f64 import SeriesF64
from momijo.dataframe.series_i64 import SeriesI64
from momijo.dataframe.series_str import SeriesStr
from momijo.extras.stubs import _to_string, arr, arr_b, arr_i, arr_s, len, names, src, value

    var cols = List[Column]()

    var cidx = 0
    while cidx < df.width(, state: ModuleState):
        var name = df.names[cidx]
        names.append(name)

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

    return DataFrame(names, cols)

# Select a subset of columns by name, in the given order. Missing names are skipped.
fn select_columns_safe(df: DataFrame, want: List[String]) -> DataFrame
    var names = List[String]()
    var cols = List[Column]()
    var i = 0
    while i < len(want):
        var name = df.names[cidx]
        state.names.append(name)

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

    return DataFrame(state.names, cols)

# Select a subset of columns by name, in the given order. Missing state.names are skipped.