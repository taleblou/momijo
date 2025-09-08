# Project:      Momijo
# Module:       src.momijo.enum.diagnostics
# File:         diagnostics.mojo
# Path:         src/momijo/enum/diagnostics.mojo
#
# Description:  src.momijo.enum.diagnostics — focused Momijo functionality with a stable public API.
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
#   - Key functions: _check_exhaustive, assert_exhaustive_or_warn, join_with, enum_expected_one_of, enum_debug_dump, enum_check_index, enum_parse_error, enum_mismatch
#   - Uses generic functions/types with explicit trait bounds.


from momijo.core.error import message, module
from momijo.core.traits import one
from momijo.dataframe.column import validity
from momijo.dataframe.helpers import t
from momijo.enum.`match` import Case, RangeCase
from momijo.enum.enum import parse
from momijo.enum.match import Case, RangeCase
from momijo.io.checkpoints.weights import items
from momijo.ir.dialects.annotations import tags
from momijo.tensor.broadcast import valid
from momijo.tensor.tensor import index
from momijo.utils.result import g
from momijo.utils.timer import start
from momijo.vision.schedule.schedule import end
from pathlib import Path
from pathlib.path import Path

fn _check_exhaustive(min_tag: Int, max_tag: Int, tags: List[Int], rstarts: List[Int], rends: List[Int]) -> Bool:
    if max_tag < min_tag:
        print("warning: empty domain")
        return True

    var size = (max_tag - min_tag) + 1
    var covered = List[Bool](length=size, fill=False)

    # precise tags
    for i in range(len(tags)):
        var t = tags[i]
        if t >= min_tag and t <= max_tag:
            covered[t - min_tag] = True

    # ranges
    var n = min(len(rstarts), len(rends))
    var idx = 0
    while idx < n:
        var s = rstarts[idx]
        var e = rends[idx]
        if e < s:
            var tmp = s
            s = e
            e = tmp
        if e >= min_tag and s <= max_tag:
            if s < min_tag: s = min_tag
            if e > max_tag: e = max_tag
            var k = s
            while k <= e:
                covered[k - min_tag] = True
                k += 1
        idx += 1

    var all_ok = True
    var j = 0
    while j < size:
        if not covered[j]:
            var miss = min_tag + j
            print("warning: uncovered tag ", miss)
            all_ok = False
        j += 1
    return all_ok

# Legacy API wrapper using Case/RangeCase
fn assert_exhaustive_or_warn(cases: List[Case], ranges: List[RangeCase], min_tag: Int, max_tag: Int) -> Bool:
    var tags = List[Int](capacity=len(cases))
    for i in range(len(cases)):
        tags.append(cases[i].tag)
    var rstarts = List[Int](capacity=len(ranges))
    var rends = List[Int](capacity=len(ranges))
    for j in range(len(ranges)):
        rstarts.append(ranges[j].start)
        rends.append(ranges[j].end)
    return _check_exhaustive(min_tag, max_tag, tags, rstarts, rends)
fn join_with(items: List[String], delim: String) -> String:
    var n = len(items)
    if n == 0:
        return ""
    if n == 1:
        return items[0]
    var out = String("")
    var i = 0
    while i < n:
        out = out + items[i]
        if i + 1 < n:
            out = out + delim
        i += 1
    return out

# "Expected one of: A, B, C" style message.
fn enum_expected_one_of(enum_name: String, names: List[String]) -> String:
    return enum_name + ": expected one of {" + join_with(names, ", ") + "}"

# Pretty dump of name->index mapping.
fn enum_debug_dump(enum_name: String, names: List[String]) -> String:
    var lines = List[String]()
    var i = 0
    while i < len(names):
        lines.append(String(i) + " => " + names[i])
        i += 1
    return enum_name + "[" + String(len(names)) + "]\n" + join_with(lines, "\n")

# Guard for index validity; returns false and a message when invalid.
fn enum_check_index(enum_name: String, i: Int, count: Int) -> (Bool, String):
    if i < 0 or i >= count:
        return (False, enum_name + ": invalid index " + String(i) + ", valid range is [0, " + String(count - 1) + "]")
    return (True, "")

# Parse error message builder.
fn enum_parse_error(enum_name: String, token: String) -> String:
    return enum_name + ": could not parse token '" + token + "'"

# Mismatch message (e.g., when wiring schemas).
fn enum_mismatch(enum_name: String, got: String, expected: String) -> String:
    return enum_name + ": got '" + got + "', expected '" + expected + "'"