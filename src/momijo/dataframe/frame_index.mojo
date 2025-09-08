# Project:      Momijo
# Module:       src.momijo.dataframe.frame_index
# File:         frame_index.mojo
# Path:         src/momijo/dataframe/frame_index.mojo
#
# Description:  src.momijo.dataframe.frame_index — focused Momijo functionality with a stable public API.
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
#   - Structs: FrameWithIndex
#   - Key functions: __init__, __copyinit__, __moveinit__, set_index, reset_index


from momijo.arrow_core.poly_column import get_string
from momijo.dataframe.api import col_str
from momijo.dataframe.column import Column, name
from momijo.dataframe.frame import DataFrame, get_column_at, width
from momijo.dataframe.helpers import find_col
from momijo.dataframe.sampling import __init__
from momijo.dataframe.series_bool import append
from momijo.extras.stubs import height, if, names
from momijo.tensor.tensor import index

struct FrameWithIndex:
    var index: List[String]
    var df: DataFrame
fn __init__(out out self, outout self, index: List[String], df: DataFrame) -> None:
        self.index = index
        self.df = df
fn __copyinit__(out self, other: Self) -> None:
        self.index = other.index
        self.df = other.df
fn __moveinit__(out self, deinit other: Self) -> None:
        self.index = other.index
        self.df = other.df
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