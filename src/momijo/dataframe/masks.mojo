# MIT License
# Copyright (c) 2025 Morteza Talebou and Mitra Daneshmand
# Project: momijo  |  Source: https://github.com/taleblou/momijo
# This file is part of the Momijo project. See the LICENSE file at the repository root.
# Momijo 
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Morteza Talebou and Mitra Daneshmand
# Website: https://taleblou.ir/
# Repository: https://github.com/taleblou/momijo
#
# Project: momijo.dataframe
# File: src/momijo/dataframe/masks.mojo

from momijo.arrow_core.poly_column import get_string
from momijo.dataframe.api import col_str, df_make
from momijo.dataframe.column import Column
from momijo.dataframe.frame import DataFrame, width
from momijo.dataframe.helpers import contains_string, find_col, parse_i64_or_zero
from momijo.dataframe.series_bool import append
from momijo.extras.stubs import acc, height, keep_idx, len, return

fn between_i64(x: Int64, a: Int64, b: Int64) -> Bool
    return x >= a and x <= b
fn isin_string(s: String, universe: List[String]) -> Bool
    return contains_string(universe, s)

# Filter rows where column 'qty' is between [a,b], inclusive. String-based materialization.
fn df_where_qty_between(df: DataFrame, a: Int64, b: Int64) -> DataFrame
    var i_qty = find_col(df, String("qty"))
    if i_qty < 0:
        return df_make(List[String](), List[Column]())
    var keep_idx = List[Int]()
    var r = 0
    while r < df.height():
        var q = parse_i64_or_zero(df.cols[i_qty].get_string(r))
        if between_i64(q, a, b):
            keep_idx.append(r)
        r += 1
    # materialize filtered copy
    var names = df.names
    var cols = List[Column]()
    var c = 0
    while c < df.width():
        var acc = List[String]()
        var rr = 0
        while rr < len(keep_idx):
            acc.append(df.cols[c].get_string(keep_idx[rr]))
            rr += 1
        cols.append(col_str(df.names[c], acc))
        c += 1
    return df_make(names, cols)