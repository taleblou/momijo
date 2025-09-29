# Project:      Momijo
# Module:       dataframe.query
# File:         query.mojo
# Path:         dataframe/query.mojo
#
# Description:  dataframe.query — Query module for Momijo DataFrame.
#               Implements core data structures, algorithms, and convenience APIs for production use.
#               Designed as a stable, composable building block within the Momijo public API.
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
#   - Structs: —
#   - Key functions: _is_digit, _parse_f64, query, eval

from momijo.dataframe.selection import DataFrame, col, filter
from momijo.dataframe.utis import ge, isin, logical_and

@always_inline
fn _is_digit(ss) -> Bool:
    return (ss == "0") or (ss == "1") or (ss == "2") or (ss == "3") or (ss == "4") or            (ss == "5") or (ss == "6") or (ss == "7") or (ss == "8") or (ss == "9")

fn _parse_f64(s: StringSlice[expr]) -> Float64:
    var t = String(s)
    if len(t) == 0:
        return 0.0
    var neg = False
    var i = 0
    if t[0] == "-":
        neg = True
        i = 1
    var int_part: Int = 0
    var frac_part: Int = 0
    var frac_len: Int = 0
    var seen_dot = False
    while i < len(t):
        var ch = t[i]
        if ch == ".":
            if seen_dot: break
            seen_dot = True
        elif _is_digit(ch):
            var d: Int = 0
            if ch == "1": d = 1
            elif ch == "2": d = 2
            elif ch == "3": d = 3
            elif ch == "4": d = 4
            elif ch == "5": d = 5
            elif ch == "6": d = 6
            elif ch == "7": d = 7
            elif ch == "8": d = 8
            elif ch == "9": d = 9
            else: d = 0
            if not seen_dot:
                int_part = int_part * 10 + d
            else:
                frac_part = frac_part * 10 + d
                frac_len += 1
        else:
            break
        i += 1
    var p10 = 1.0
    var k = 0
    while k < frac_len:
        p10 = p10 * 10.0
        k += 1
    var val = Float64(int_part) + Float64(frac_part) / p10
    if neg: val = -val
    return val

fn query(df: DataFrame, expr: String) -> DataFrame:
# Supports: "col1 > num and col2 == 'X'"
    var parts = expr.split(" and ")
    var masks = List[List[Bool]]()
    for p in parts:
        var p2 = p.strip()
        if "==" in p2:
            var kv = p2.split("==")
            var name = String(kv[0].strip())
            var val = String(kv[1].strip().strip("'\""))
            masks.append(isin(col(df, name), [val]))
        elif ">" in p2:
            var kv2 = p2.split(">")
            var name2 = String(kv2[0].strip())
            var thr = _parse_f64(kv2[1].strip())
            masks.append(ge(col(df, name2), thr))
    if len(masks) == 0:
        return df
    var m = masks[0]
    var i = 1
    while i < len(masks):
        m = logical_and(m, masks[i])
        i += 1
    return filter(df, m)

fn eval(df: DataFrame, expr: String) -> DataFrame:
# "new_col = old_col * number"
    from momijo.dataframe.selection import DataFrame as DF, col as get_col
    var parts = expr.split("=")
    var new_name = String(parts[0].strip())
    var rhs = parts[1].strip()
    var mul_parts = rhs.split("*")
    var colname = String(mul_parts[0].strip())
    var factor = _parse_f64(mul_parts[1].strip())

# deep copy
    var headers = List[String]()
    for h in df.headers: headers.append(h)
    var columns = List[List[String]]()
    for c in df.columns:
        var cc = List[String]()
        for v in c: cc.append(v)
        columns.append(cc)

# compute
    var out_col = List[String]()
    for v in get_col(df, colname):
# parse v
        var t = String(v)
        var neg = False
        var i = 0
        if len(t) > 0 and t[0] == "-":
            neg = True
            i = 1
        var int_part: Int = 0
        var frac_part: Int = 0
        var frac_len: Int = 0
        var seen_dot = False
        while i < len(t):
            var ch = t[i]
            if ch == ".":
                if seen_dot: break
                seen_dot = True
            elif _is_digit(ch):
                var d: Int = 0
                if ch == "1": d = 1
                elif ch == "2": d = 2
                elif ch == "3": d = 3
                elif ch == "4": d = 4
                elif ch == "5": d = 5
                elif ch == "6": d = 6
                elif ch == "7": d = 7
                elif ch == "8": d = 8
                elif ch == "9": d = 9
                else: d = 0
                if not seen_dot:
                    int_part = int_part * 10 + d
                else:
                    frac_part = frac_part * 10 + d
                    frac_len += 1
            else:
                break
            i += 1
        var p10 = 1.0
        var k = 0
        while k < frac_len:
            p10 = p10 * 10.0
            k += 1
        var val = Float64(int_part) + Float64(frac_part) / p10
        if neg: val = -val
        out_col.append(String(val * factor))

    headers.append(new_name)
    columns.append(out_col)
    return DF(headers, columns)