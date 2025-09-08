# Project:      Momijo
# Module:       src.momijo.dataframe.timezone
# File:         timezone.mojo
# Path:         src/momijo/dataframe/timezone.mojo
#
# Description:  src.momijo.dataframe.timezone — focused Momijo functionality with a stable public API.
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
#   - Key functions: tz_localize_utc, tz_convert
#   - Uses generic functions/types with explicit trait bounds.


from momijo.dataframe.series_bool import append
from momijo.extras.stubs import len

fn tz_localize_utc(ts: List[String]) -> List[String]
    var out = List[String]()
    var i = 0
    while i < len(ts):
        out.append(ts[i] + String("+00:00"))
        i += 1
    return out
fn tz_convert(ts: List[String], target: String) -> List[String]
    var out = List[String]()
    var i = 0
    while i < len(ts):
        out.append(ts[i] + String("->") + target)
        i += 1
    return out