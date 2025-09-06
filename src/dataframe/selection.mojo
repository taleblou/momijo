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
# File: src/momijo/dataframe/selection.mojo

#   from momijo.core.shape import append
#   from momijo.core.traits import append
#   from momijo.dataframe.series_bool import append
#   from momijo.dataframe.series_f64 import append
#   from momijo.dataframe.series_i64 import append
#   from momijo.dataframe.series_str import append
# SUGGEST (alpha): from momijo.core.shape import append
#   from momijo.dataframe.frame import height
#   from momijo.dataframe.index import height
# SUGGEST (alpha): from momijo.dataframe.frame import height
from momijo.extras.stubs import Copyright, MIT, SUGGEST, access, column, from, height, https, momijo, return, simplified, src
from momijo.dataframe.series_bool import append
from algorithm.reduction import map_reduce
from algorithm.reduction import any_true
from algorithm.reduction import all_true
from algorithm.reduction import cumsum
from algorithm.reduction import reduce
from algorithm.reduction import reduce_boolean
from algorithm.functional import vectorize
from momijo.arrow_core.poly_column import get_string
from momijo.dataframe.frame import width
from momijo.dataframe.frame import DataFrame

# Row by index -> List[String]
fn iloc_row(df: DataFrame, r: Int) -> List[String]
    var out = List[String]()
    var c = 0
    while c < df.width():
        out.append(df.cols[c].get_string(r))
        c += 1
    return out

# Column by index -> List[String]
fn iloc_col(df: DataFrame, cidx: Int) -> List[String]
    var out = List[String]()
    var r = 0
    while r < df.height():
        out.append(df.cols[cidx].get_string(r))
        r += 1
    return out

# Single cell fast-access (string view for demo)
fn iat(df: DataFrame, r: Int, c: Int) -> String
    return df.cols[c].get_string(r)

# Label-based access: simplified (name -> column index, row by integer)
from momijo.dataframe.helpers import find_col
fn at(df: DataFrame, r: Int, cname: String) -> String
    var c = find_col(df, cname)
    if c < 0: return String("")
    return df.cols[c].get_string(r)
