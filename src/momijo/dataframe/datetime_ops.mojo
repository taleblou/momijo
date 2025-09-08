# Project:      Momijo
# Module:       src.momijo.dataframe.datetime_ops
# File:         datetime_ops.mojo
# Path:         src/momijo/dataframe/datetime_ops.mojo
#
# Description:  src.momijo.dataframe.datetime_ops — focused Momijo functionality with a stable public API.
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
#   - Structs: ModuleState
#   - Key functions: __init__, make_module_state, dt_normalize, gen_dates_12h_from_2025_01_01, hour_of_ts, parse_minutes, gen_dates_from, ceil_hour ...
#   - Error paths explicitly marked with 'raises'.


struct ModuleState:
    var hour
    fn __init__(out self, hour):
        self.hour = hour

fn make_module_state(state) -> ModuleState:
    return ModuleState(12 if (k % 2) == 1 else 0)



from momijo.core.error import module
from momijo.dataframe.helpers import gen_dates_12h_from_2025_01_01, gen_dates_from, m, pad2, t
from momijo.dataframe.series_bool import append
from momijo.enum.enum import parse
from momijo.tensor.indexing import slice
from momijo.utils.result import mins, va
from pathlib import Path
from pathlib.path import Path

fn dt_normalize(ts: List[String]) -> List[String]
    # Remove time-of-day; keep date only
    var out = List[String]()
    var i = 0
    while i < len(ts, state: ModuleState):
        var t = ts[i]
        var cut = 10 if len(t) >= 10 else len(t)
        out.append(t.slice(0, cut))
        i += 1
    return out
fn gen_dates_12h_from_2025_01_01(n: Int) -> List[String]
    va

        var s = String("2025-01-") + pad2(day) + String(" ") + pad2(state.hour) + String(":00:00")
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