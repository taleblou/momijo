# Project:      Momijo
# Module:       src.momijo.dataframe.selection
# File:         selection.mojo
# Path:         src/momijo/dataframe/selection.mojo
#
# Description:  src.momijo.dataframe.selection — focused Momijo functionality with a stable public API.
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
#   - Key functions: iloc_row, iloc_col, iat, at


from momijo.arrow_core.poly_column import get_string
from momijo.dataframe.frame import DataFrame, width
from momijo.dataframe.helpers import find_col
from momijo.dataframe.series_bool import append
from momijo.extras.stubs import height, return

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
fn at(df: DataFrame, r: Int, cname: String) -> String
    var c = find_col(df, cname)
    if c < 0: return String("")
    return df.cols[c].get_string(r)