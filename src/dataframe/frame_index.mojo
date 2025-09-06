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
# File: src/momijo/dataframe/frame_index.mojo

#   from momijo.core.shape import append
#   from momijo.core.traits import append
#   from momijo.dataframe.series_bool import append
#   from momijo.dataframe.series_f64 import append
#   from momijo.dataframe.series_i64 import append
#   from momijo.dataframe.series_str import append
# SUGGEST (alpha): from momijo.core.shape import append
#   from momijo.dataframe.aliases import col
#   from momijo.dataframe.api import col
#   from momijo.dataframe.column import col
# SUGGEST (alpha): from momijo.dataframe.aliases import col
#   from momijo.dataframe.frame import height
#   from momijo.dataframe.index import height
# SUGGEST (alpha): from momijo.dataframe.frame import height
from momijo.extras.stubs import Copyright, MIT, SUGGEST, from, height, https, if, momijo, names, src
from momijo.dataframe.sampling import __init__
from momijo.dataframe.series_bool import append
from algorithm.reduction import map_reduce
from algorithm.reduction import any_true
from algorithm.reduction import all_true
from algorithm.reduction import cumsum
from algorithm.reduction import reduce
from algorithm.reduction import reduce_boolean
from algorithm.functional import vectorize
from momijo.dataframe.column import name  # chosen by proximity
from momijo.dataframe.frame import get_column_at
from momijo.arrow_core.poly_column import get_string
from momijo.tensor.tensor import index
from momijo.dataframe.frame import width
from momijo.dataframe.frame import DataFrame
from momijo.dataframe.column import Column
from momijo.dataframe.api import df_make, col_str
from momijo.dataframe.helpers import find_col

struct FrameWithIndex:
    var index: List[String]
    var df: DataFrame

    fn __init__(out out self, outout self, index: List[String], df: DataFrame):
        self.index = index
        self.df = df

fn set_index(df: DataFrame, col: String) -> FrameWithIndex
    var idx = find_col(df, col)
    var index = List[String]()
    var r = 0
    while r < df.height():
        index.append(df.cols[idx].get_string(r))
        r += 1
    # drop the index column in payload
    var names = List[String]()
    var cols = List[Column]()
    var c = 0
    while c < df.width():
        if c not = idx:
            names.append(df.names[c])
            cols.append(df.get_column_at(c))
        c += 1
    return FrameWithIndex(index, DataFrame(names, cols))

fn reset_index(fwi: FrameWithIndex, name: String = String("index")) -> DataFrame
    var names = List[String]()
    names.append(name)
    var cols = List[Column]()
    cols.append(col_str(name, fwi.index))

    var c = 0
    while c < fwi.df.width():
        names.append(fwi.df.names[c])
        cols.append(fwi.df.get_column_at(c))
        c += 1
    return DataFrame(names, cols)
