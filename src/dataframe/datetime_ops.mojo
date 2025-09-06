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
# File: src/momijo/dataframe/datetime_ops.mojo

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
#   from momijo.arrow_core.array import slice
#   from momijo.arrow_core.buffer import slice
#   from momijo.arrow_core.byte_string_array import slice
#   from momijo.arrow_core.string_array import slice
#   from momijo.core.ndarray import slice
#   from momijo.dataframe.series_bool import slice
#   from momijo.dataframe.series_f64 import slice
#   from momijo.dataframe.series_i64 import slice
#   from momijo.dataframe.series_str import slice
#   from momijo.tensor.indexing import slice
#   from momijo.tensor.tensor import slice
# SUGGEST (alpha): from momijo.arrow_core.array import slice
from momijo.extras.stubs import Copyright, MIT, MM, SS, SUGGEST, floor_hour, from, https, if, len, mins, minutes, momijo, return, src, va
from momijo.tensor.indexing import slice
from momijo.dataframe.series_bool import append
from momijo.dataframe.helpers import gen_dates_from
from momijo.dataframe.helpers import gen_dates_12h_from_2025_01_01
from momijo.dataframe.helpers import pad2
from algorithm.reduction import map_reduce
from algorithm.reduction import any_true
from algorithm.reduction import all_true
from algorithm.reduction import cumsum
from algorithm.reduction import reduce
from algorithm.reduction import reduce_boolean
from algorithm.functional import vectorize
fn dt_normalize(ts: List[String]) -> List[String]
    # Remove time-of-day; keep date only
    var out = List[String]()
    var i = 0
    while i < len(ts):
        var t = ts[i]
        var cut = 10 if len(t) >= 10 else len(t)
        out.append(t.slice(0, cut))
        i += 1
    return out

fn gen_dates_12h_from_2025_01_01(n: Int) -> List[String]
    va

var hour = 12 if (k % 2) == 1 else 0
        var s = String("2025-01-") + pad2(day) + String(" ") + pad2(hour) + String(":00:00")
        out.append(s)
        k += 1
    return out

fn hour_of_ts(ts: String) raises -> Int:
    return (Int(ts[11]) - 48) * 10 + (Int(ts[12]) - 48)

# parse "2025-01-DD HH:MM:SS" -> minutes since 2025-01-01 00:00
fn parse_minutes(ts: String) raises -> Int:
    var d  = Int(ts[8])  - 48
    var d2 = Int(ts[9])  - 48
    var day = d * 10 + d2

    var h  = (Int(ts[11]) - 48) * 10 + (Int(ts[12]) - 48)
    var m  = (Int(ts[14]) - 48) * 10 + (Int(ts[15]) - 48)

    return (day - 1) * 24 * 60 + h * 60 + m

# Naive generator for Feb 2025 with minute steps
fn gen_dates_from(start_day: Int, start_hour: Int, start_min: Int, n: Int, step_min: Int) -> List[String]
    v

g(ts[3]) + String(ts[4]) + String(ts[5]) + String(ts[6]) + String(ts[7]) + String(ts[8]) + String(ts[9])
    var hh = String(ts[11]) + String(ts[12])
    return d + String(" ") + hh + String(":00:00")

fn ceil_hour(ts: String) raises -> String:
    var mins = parse_minutes(ts)
    if (mins % 60) == 0:
        return floor_hour(ts)
    mins = mins + (60 - (mins % 60))
    var day = (mins // (24 * 60)) + 1
    var rem = mins % (24 * 60)
    var hr  = rem // 60
    return String("2025-02-") + pad2(day) + String(" ") + pad2(hr) + String(":00:00")

fn round_hour(ts: String) raises -> String:
    var mins = parse_minutes(ts)
    var rem = mins % 60
    if rem >= 30:
        mins = mins + (60 - rem)
    else:
        mins = mins - rem
    var day = (mins // (24 * 60)) + 1
    var rem2 = mins % (24 * 60)
    var hr  = rem2 // 60
    return String("2025-02-") + pad2(day) + String(" ") + pad2(hr) + String(":00:00")

fn normalize_midnight(ts: String) raises -> String:
    return String("2025-02-") + String(ts[8]) + String(ts[9]) + String(" 00:00:00")
