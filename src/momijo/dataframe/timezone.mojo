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
# File: src/momijo/dataframe/timezone.mojo

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