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
# File: src/momijo/dataframe/series.mojo

#   from momijo.dataframe.helpers import RNG
#   from momijo.dataframe.sampling import RNG
#   from momijo.tensor.random import RNG
# SUGGEST (alpha): from momijo.dataframe.helpers import RNG
#   from momijo.core.shape import append
#   from momijo.core.traits import append
#   from momijo.dataframe.series_bool import append
#   from momijo.dataframe.series_f64 import append
#   from momijo.dataframe.series_i64 import append
#   from momijo.dataframe.series_str import append
# SUGGEST (alpha): from momijo.core.shape import append
#   from momijo.dataframe.api import df_make
#   from momijo.dataframe.frame import df_make
# SUGGEST (alpha): from momijo.dataframe.api import df_make
#   from momijo.dataframe.frame import height
#   from momijo.dataframe.index import height
# SUGGEST (alpha): from momijo.dataframe.frame import height
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
#   from momijo.dataframe.helpers import rand01
#   from momijo.dataframe.sampling import rand01
# SUGGEST (alpha): from momijo.dataframe.helpers import rand01
from momijo.extras.stubs import Copyright, MIT, SUGGEST, acc, asof, convert, from, height, https, if, keys, len, momijo, pos, replace, src, tz_localize
from momijo.dataframe.api import df_make
from momijo.dataframe.sampling import shuffle_idxs
from algorithm.reduction import map_reduce
from algorithm.reduction import any_true
from algorithm.reduction import all_true
from algorithm.reduction import cumsum
from algorithm.reduction import reduce
from algorithm.reduction import reduce_boolean
from algorithm.functional import vectorize
from momijo.dataframe.column import col_f64
from momijo.dataframe.column import col_str
from momijo.dataframe.helpers import find_col
from momijo.arrow_core.poly_column import get_string
from momijo.dataframe.frame import DataFrame
from momijo.dataframe.column import Column
from momijo.dataframe.series_bool import SeriesBool, append
from momijo.dataframe.series_i64 import SeriesI64
from momijo.dataframe.series_f64 import SeriesF64
from momijo.dataframe.series_str import SeriesStr

# ---- Auto-merged functions ----

fn groupby_sum(df: DataFrame, key: String, val: String) -> DataFrame
    var ik = find_col(df, key); var iv = find_col(df, val)
    var keys = List[String](); var sums = List[Float64]()
    var i = 0
    while i < df.height():
        var k = df.cols[ik].get_string(i)
        var x = df.cols[iv].get_string(i)
        var v = 0.0
        # naive parse
        var j = 0; var dot = False; var acc = String("")
        while j < len(x):
            acc = acc + String(x[j]); j += 1
        # here we pretend it's clean float string: replace with real parser if needed
        # (or use SeriesF64 directly and avoid strings in a real pipeline)
        # fallback:
        if len(acc) > 0: v = 0.0
        var pos = -1
        var kk = 0; var found = False
        while kk < len(keys):
            if keys[kk] == k: pos = kk; found = True
            kk += 1
        if not found:
            keys.append(k); sums.append(v)
            pos = len(keys) - 1
        sums[pos] = sums[pos] + v
        i += 1
    return df_make([key, String("sum_"+val)], [col_str(key, keys), col_f64(String("sum_"+val), sums)])
# ----------------------------------------------------------------------------
# 19) Time zones: tz_localize/convert (demo)
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# 27) Reindexing tricks + Series.asof (demo)
# ----------------------------------------------------------------------------
