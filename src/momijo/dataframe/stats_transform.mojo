# Project:      Momijo
# Module:       src.momijo.dataframe.stats_transform
# File:         stats_transform.mojo
# Path:         src/momijo/dataframe/stats_transform.mojo
#
# Description:  src.momijo.dataframe.stats_transform — focused Momijo functionality with a stable public API.
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
#   - Key functions: value_counts_str, str_replace_all


from momijo.dataframe.api import col_i64, col_str, df_make
from momijo.dataframe.column import Column
from momijo.dataframe.frame import DataFrame
from momijo.dataframe.helpers import unique_strings_list
from momijo.dataframe.series_bool import append
from momijo.extras.stubs import and, cnt, if, len, match, not

fn value_counts_str(xs: List[String]) -> DataFrame
    var keys = unique_strings_list(xs)
    var counts = List[Int64]()
    var i = 0
    while i < len(keys):
        var k = keys[i]
        var cnt: Int64 = 0
        var j = 0
        while j < len(xs):
            if xs[j] == k:
                cnt += 1
            j += 1
        counts.append(cnt)
        i += 1
    return df_make(List[String](["value","count"]),
                   List[Column]([col_str(String("value"), keys),
                                 col_i64(String("count"), counts)]))

# naive non-overlapping replacement
fn str_replace_all(s: String, old: String, newv: String) -> String
    var out = String("")
    var i = 0
    while i < len(s):
        var match = True
        var j = 0
        while j < len(old) and (i + j) < len(s):
            if s[i + j] not = old[j]:
                match = False
            j += 1
        if match and len(old) > 0 and i + len(old) <= len(s):
            out = out + newv
            i = i + len(old)
        else:
            out = out + String(s[i])
            i += 1
    return out