# Project:      Momijo
# Module:       src.momijo.dataframe.encoding
# File:         encoding.mojo
# Path:         src/momijo/dataframe/encoding.mojo
#
# Description:  src.momijo.dataframe.encoding — focused Momijo functionality with a stable public API.
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
#   - Key functions: get_dummies_str, value_counts_bins


from momijo.arrow_core.poly_column import get_string
from momijo.dataframe.api import col_i64, col_str, df_make
from momijo.dataframe.column import Column
from momijo.dataframe.frame import DataFrame
from momijo.dataframe.helpers import contains_string, find_col, max_f64, min_f64
from momijo.dataframe.series_bool import append
from momijo.extras.stubs import cats, else, height, if, len, return, vals

fn get_dummies_str(df: DataFrame, col: String, drop_first: Bool = False, prefix: String = String("")) -> DataFrame
    var idx = find_col(df, col)
    if idx < 0:
        return df_make(List[String](), List[Column]())
    var cats = List[String]()
    var r = 0
    while r < df.height():
        var s = df.cols[idx].get_string(r)
        if not contains_string(cats, s):
            cats.append(s)
        r += 1
    var start = 1 if (drop_first and len(cats) > 0) else 0
    var names = df.names
    var cols = df.cols
    var j = start
    while j < len(cats):
        var nm = (prefix + String("_") + cats[j]) if len(prefix) > 0 else cats[j]
        var vals = List[Int64]()
        var rr = 0
        while rr < df.height():
            vals.append(Int64(1) if df.cols[idx].get_string(rr) == cats[j] else Int64(0))
            rr += 1
        names.append(nm)
        cols.append(col_i64(nm, vals))
        j += 1
    return df_make(names, cols)
fn value_counts_bins(xs: List[Float64], bins: Int) -> DataFrame
    if bins <= 0:
        return df_make(List[String]([String("bin"), String("count")]),
                       List[Column]([col_str(String("bin"), List[String]()), col_i64(String("count"), List[Int64]())]))
    var mn = min_f64(xs)
    var mx = max_f64(xs)
    var w = (mx - mn) / Float64(bins)
    var counts = List[Int64](bins, 0)
    var i = 0
    while i < len(xs):
        var b = Int((xs[i] - mn) / w) if w not = 0.0 else 0
        if b < 0: b = 0
        if b >= bins: b = bins - 1
        counts[b] = counts[b] + 1
        i += 1
    var labels = List[String]()
    var j = 0
    while j < bins:
        var l = mn + Float64(j) * w
        var r = l + w
        labels.append(String("[") + String(l) + String(",") + String(r) + String(")"))
        j += 1
    return df_make(List[String]([String("bin"), String("count")]),
                   List[Column]([col_str(String("bin"), labels), col_i64(String("count"), counts)]))