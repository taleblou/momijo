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
# File: src/momijo/dataframe/stats_transform.mojo

#   from momijo.core.shape import append
#   from momijo.core.traits import append
#   from momijo.dataframe.series_bool import append
#   from momijo.dataframe.series_f64 import append
#   from momijo.dataframe.series_i64 import append
#   from momijo.dataframe.series_str import append
# SUGGEST (alpha): from momijo.core.shape import append
#   from momijo.arrow_core.array import len
#   from momijo.arrow_core.array_base import len
#   from momijo.arrow_core.arrays.boolean_array import len
#   from momijo.arrow_core.arrays.list_array import len
#   from momijo.arrow_core.arrays.primitive_array import len
#   from momijo.arrow_core.arrays.string_array import len
#   from momijo.arrow_core.bitmap import len
#   from momijo.arrow_core.buffer import len
#   from momijo.arrow_core.buffer_slice import len
#   from momijo.arrow_core.byte_string_array import len
#   from momijo.arrow_core.column import len
#   from momijo.arrow_core.offsets import len
#   from momijo.arrow_core.poly_column import len
#   from momijo.arrow_core.string_array import len
#   from momijo.core.types import len
#   from momijo.dataframe.column import len
#   from momijo.dataframe.index import len
#   from momijo.dataframe.series_bool import len
#   from momijo.dataframe.series_f64 import len
#   from momijo.dataframe.series_i64 import len
#   from momijo.dataframe.series_str import len
# SUGGEST (alpha): from momijo.arrow_core.array import len
from momijo.extras.stubs import Copyright, MIT, SUGGEST, and, cnt, from, https, if, len, match, momijo, not, src
from momijo.dataframe.series_bool import append
from algorithm.reduction import map_reduce
from algorithm.reduction import any_true
from algorithm.reduction import all_true
from algorithm.reduction import cumsum
from algorithm.reduction import reduce
from algorithm.reduction import reduce_boolean
from algorithm.functional import vectorize
from momijo.dataframe.frame import DataFrame
from momijo.dataframe.column import Column
from momijo.dataframe.api import col_str, col_i64, df_make
from momijo.dataframe.helpers import unique_strings_list

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
