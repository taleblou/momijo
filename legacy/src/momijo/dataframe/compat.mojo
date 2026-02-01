# Project:      Momijo
# Module:       dataframe.compat
# File:         compat.mojo
# Path:         dataframe/compat.mojo
#
# Description:  dataframe.compat — Compat module for Momijo DataFrame.
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
#   - Key functions: _strip, _lower, _normalize_decimal, as_f64_or_nan, as_i64_or_zero, as_bool_or_false, as_str_or_empty, map_f64_or_nan, map_i64_or_zero, map_bool_or_false, map_str_or_empty

fn _strip(s: String) -> String:
    var n = len(s)
    var i = 0
    var start = 0
    var end = n
    while i < n and (s[i] == " " or s[i] == "\t" or s[i] == "\n" or s[i] == "\r"):
        i += 1
    start = i
    var j = n - 1
    while j >= start and (s[j] == " " or s[j] == "\t" or s[j] == "\n" or s[j] == "\r"):
        j -= 1
    end = j + 1
    var out = String("")
    var k = start
    while k < end:
        out += String(s[k])
        k += 1
    return out

fn _lower(s: String) -> String:
# Minimal lowercase; relies on String.lower() if available.
    try:
        return s.lower()
    except:
        return s

fn _normalize_decimal(s: String) -> String:
# Replace comma decimal separator with dot (e.g., "12,34" -> "12.34")
    var out = String("")
    var i = 0
    while i < len(s):
        var ch = s[i]
        if ch == ",":
            out += String(".")
        else:
            out += String(ch)
        i += 1
    return out

# Float64 safe parse
fn as_f64_or_nan(s: String) -> Float64:
    var t = _normalize_decimal(_strip(s))
    if t == String("") or t == String("nan") or t == String("NaN") or t == String("NAN"):
        return Float64.nan()
    try:
        return Float64(t)
    except:
        return Float64.nan()

# Int64 safe parse (tries int, then float->int truncation), returns 0 on failure
fn as_i64_or_zero(s: String) -> Int64:
    var t = _strip(s)
    if t == String(""):
        return Int64(0)
# Try integer parse
    try:
        return Int64(t)
    except:
# Try float then cast
        var f = as_f64_or_nan(t)
        if f == f:  # not NaN
            try:
                return Int64(f)
            except:
                return Int64(0)
        return Int64(0)

# Bool safe parse
fn as_bool_or_false(s: String) -> Bool:
    var t = _lower(_strip(s))
    if t == String("true") or t == String("1") or t == String("yes") or t == String("y") or t == String("on"):
        return True
    if t == String("false") or t == String("0") or t == String("no") or t == String("n") or t == String("off"):
        return False
    return False

# String safe (identity, non-null)
fn as_str_or_empty(s: String) -> String:
# Mojo String is non-null, but we keep a guard for consistency.
    return s

# Vectorized helpers
fn map_f64_or_nan(values: List[String]) -> List[Float64]:
    var out = List[Float64]()
    var i = 0
    while i < len(values):
        out.append(as_f64_or_nan(values[i]))
        i += 1
    return out

fn map_i64_or_zero(values: List[String]) -> List[Int64]:
    var out = List[Int64]()
    var i = 0
    while i < len(values):
        out.append(as_i64_or_zero(values[i]))
        i += 1
    return out

fn map_bool_or_false(values: List[String]) -> List[Bool]:
    var out = List[Bool]()
    var i = 0
    while i < len(values):
        out.append(as_bool_or_false(values[i]))
        i += 1
    return out

fn map_str_or_empty(values: List[String]) -> List[String]:
    var out = List[String]()
    var i = 0
    while i < len(values):
        out.append(as_str_or_empty(values[i]))
        i += 1
    return out