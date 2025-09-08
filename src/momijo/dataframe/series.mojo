# Project:      Momijo
# Module:       src.momijo.dataframe.series
# File:         series.mojo
# Path:         src/momijo/dataframe/series.mojo
#
# Description:  src.momijo.dataframe.series — focused Momijo functionality with a stable public API.
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
#   - Key functions: groupby_sum


from momijo.arrow_core.poly_column import get_string
from momijo.dataframe.api import df_make
from momijo.dataframe.column import col_f64, col_str
from momijo.dataframe.frame import DataFrame
from momijo.dataframe.helpers import find_col
from momijo.dataframe.series_bool import append
from momijo.extras.stubs import acc, height, if, keys, len, pos

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