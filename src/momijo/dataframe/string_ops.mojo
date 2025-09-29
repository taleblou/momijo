# Project:      Momijo
# Module:       dataframe.string_ops
# File:         string_ops.mojo
# Path:         dataframe/string_ops.mojo
#
# Description:  dataframe.string_ops — String Ops module for Momijo DataFrame.
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
#   - Key functions: is_alpha_code, is_digit_code, str_contains, str_split_once, str_strip, compare_str_eq, contains_digit, extract_first_alpha, extract_all_alpha_joined, split_on_delims, rpad, str_len, str_upper, str_title, str_slice, str_extract, str_replace_regex, _to_upper_one

from collections.list import List
from momijo.dataframe.selection import Mask as Mask
from momijo.dataframe._groupby_core import groupby as groupby
from momijo.dataframe.categorical import astype as astype
from momijo.dataframe.categorical import to_category as to_category
from momijo.dataframe.compat import assign as assign
from momijo.dataframe.compat import concat_cols2 as concat_cols2
from momijo.dataframe.compat import concat_rows2 as concat_rows2
from momijo.dataframe.compat import drop_duplicates as drop_duplicates
from momijo.dataframe.compat import ffill as ffill
from momijo.dataframe.compat import melt as melt
from momijo.dataframe.compat import pivot_table as pivot_table
from momijo.dataframe.compat import sort_values as sort_values
from momijo.dataframe.compat import to_datetime as to_datetime
from momijo.dataframe.df_api import concat_cols as concat_cols
from momijo.dataframe.df_api import concat_rows as concat_rows
from momijo.dataframe.df_api import copy as copy
from momijo.dataframe.df_api import head_str as head_str
from momijo.dataframe.df_api import isna_count_by_col as isna_count_by_col
from momijo.dataframe.df_api import make_pairs as make_pairs
from momijo.dataframe.df_api import merge as merge
from momijo.dataframe.df_api import rename as rename
from momijo.dataframe.df_api import replace_values as replace_values
from momijo.dataframe.df_api import series_from_list as series_from_list
from momijo.dataframe.df_api import series_values as series_values
from momijo.dataframe.df_api import set_value as set_value
from momijo.dataframe.df_api import take_rows as take_rows
from momijo.dataframe.series_str import where as where
from momijo.dataframe.stats_transform import col_mean as col_mean
from momijo.dataframe.stats_transform import cut_numeric as cut_numeric
from momijo.dataframe.stats_transform import fillna_value as fillna_value
from momijo.dataframe.stats_transform import interpolate_numeric as interpolate_numeric

from momijo.dataframe.missing import _dropna_rows as _dropna_rows

# ----------------------------- Tiny predicates ------------------------------

fn is_alpha_code(c: UInt8) -> Bool:
# A-Z or a-z
    return (c >= 65 and c <= 90) or (c >= 97 and c <= 122)

fn is_digit_code(c: UInt8) -> Bool:
# '0'..'9'
    return c >= 48 and c <= 57

# ----------------------------- Basic ops ------------------------------------



fn str_split_once(s: String, sep: String) -> (String, String):
    if len(sep) == 0:
        return (s, String(""))
    var i: Int = 0
    while i + len(sep) <= len(s):
        var ok: Bool = True
        var j: Int = 0
        while j < len(sep):
            if s[i + j] != sep[j]:
                ok = False
                break
            j += 1
        if ok:
            return (s.slice(0, i), s.slice(i + len(sep), len(s)))
        i += 1
    return (s, String(""))

fn str_strip(s: String) -> String:
    var l: Int = 0
    var r: Int = len(s)
# ASCII space or tab
    while l < r and (s.bytes()[l] == UInt8(32) or s.bytes()[l] == UInt8(9)):
        l += 1
    while r > l and (s.bytes()[r - 1] == UInt8(32) or s.bytes()[r - 1] == UInt8(9)):
        r -= 1
    return s.slice(l, r)

# Compare two equal-length lists of strings for equality; return 1/0 as Int64
fn compare_str_eq(a: List[String], b: List[String]) -> List[Int64]:
    var n = len(a)
    if len(b) != n:
        return List[Int64]()  # size mismatch => empty
    var out = List[Int64]()
    var i: Int = 0
    while i < n:
        out.append(Int64(1) if a[i] == b[i] else Int64(0))
        i += 1
    return out

# ----------------------------- Digit / alpha --------------------------------

fn contains_digit(s: String) -> Bool:
    var i: Int = 0
    while i < len(s):
        if is_digit_code(UInt8(s[i])):
            return True
        i += 1
    return False

fn extract_first_alpha(s: String) -> String:
# first contiguous [A-Za-z]+ token
    var i: Int = 0
    while i < len(s) and not is_alpha_code(UInt8(s[i])):
        i += 1
    var out = String("")
    while i < len(s) and is_alpha_code(UInt8(s[i])):
        out = out + String(s[i])
        i += 1
    return out

fn extract_all_alpha_joined(s: String) -> String:
# join all alpha tokens with '|'
    var i: Int = 0
    var out = String("")
    var first: Bool = True
    while i < len(s):
# skip non-alpha
        while i < len(s) and not is_alpha_code(UInt8(s[i])):
            i += 1
        if i >= len(s):
            break
# read token
        var tok = String("")
        while i < len(s) and is_alpha_code(UInt8(s[i])):
            tok = tok + String(s[i])
            i += 1
        if len(tok) > 0:
            if not first:
                out = out + String("|")
            out = out + tok
            first = False
    return out

# Split on lightweight delimiters: space, '-', '(', ')', '_'
fn split_on_delims(s: String) -> List[String]:
    var toks = List[String]()
    var cur = String("")
    var i: Int = 0
    while i < len(s):
       varch = UInt8(s[i])
       varis_delim = (ch == 32) or (ch == 45) or (ch == 40) or (ch == 41) or (ch == 95)
        if is_delim:
            if len(cur) > 0:
                toks.append(cur)
                cur = String("")
        else:
            cur = cur + String(s[i])
        i += 1
    if len(cur) > 0:
        toks.append(cur)
    return toks

# Right-pad to width using a (possibly multi-char) fill pattern
fn rpad(s: String, width: Int, fill: String) -> String:
    if len(s) >= width or len(fill) == 0:
        return s
    var out = s
    var i: Int = len(out)
    var fi: Int = 0
    while i < width:
        out = out + String(fill[fi])
        fi += 1
        if fi >= len(fill):
            fi = 0
        i += 1
    return out


# [moved] str_len
fn str_len(df0: DataFrame, col: String) -> List[Float64]:
        var idx = _find_col_idx(df0, col)
        var out = List[Float64]()
        var r = 0
        while r < df0.nrows():
            var s = df0.cols[idx][r]
            out.append(Float64(len(s)))
            r += 1
        return out
    
fn str_upper(df0: DataFrame, col: String) -> List[String]:
    var idx = _find_col_idx(df0, col)
    var out = List[String]()
    var r = 0
    while r < df0.nrows():
        var s = df0.cols[idx][r]
        var t = String("")
        var i = 0
        while i < len(s):
            var ch = s[i]
            var cval: UInt8
            try:
                cval = UInt8(ch[0])  # safe access with try
            except:
                i += 1
                continue
            if cval >= UInt8(ord("a")) and cval <= UInt8(ord("z")):
                t += String(chr(Int(cval) - 32))
            else:
                t += String(ch)
            i += 1
        out.append(t)
        r += 1
    return out




fn str_slice(df0: DataFrame, col: String, start: Int, length: Int) -> List[String]:
    var idx = _find_col_idx(df0, col)
    var out = List[String]()

    var st = start
    var ln = length
    if st < 0: st = 0
    if ln < 0: ln = 0

    var r = 0
    while r < df0.nrows():
        var s = df0.cols[idx][r]
        var n = len(s)
        var end = st + ln
        if st >= n:
            out.append(String(""))
        else:
            if end > n: end = n
            var sub = String("")
            var i = st
            while i < end:
                sub += String(s[i])
                i += 1
            out.append(sub)
        r += 1
    return out

# ---- concat_rows2: two-DataFrame variant (avoids List[DataFrame] trait bound) ----

# [moved] _to_upper_one
fn _to_upper_one(ch: String) -> String:
    var lower = String("abcdefghijklmnopqrstuvwxyz")
    var upper = String("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    var i = 0
    while i < len(lower):
        if ch == String(lower[i]):
            return String(upper[i])
        i += 1
    return ch

# ---- String ops ----

 

fn str_title(df0: DataFrame, col: String) -> List[String]:
    var idx = _find_col_idx(df0, col)
    var out = List[String]()
    var r = 0
    while r < df0.nrows():
        var s = df0.cols[idx][r]
        var t = String("")
        var new_word = True
        var i = 0
        while i < len(s):
            var chs = String(s[i])
            if chs == String(" ") or chs == String("\t"):
                new_word = True
                t += chs
            else:
                if new_word:
                    t += _to_upper_one(chs)
                    new_word = False
                else:
                    t += chs
            i += 1
        out.append(t)
        r += 1
    return out

fn _find_col_idx(df0: DataFrame, name: String) -> Int:
    var i = 0
    while i < df0.ncols():
        if df0.col_names[i] == name:
            return i
        i += 1
    return -1

# ---- String helpers ----

# [moved] _to_lower_one
fn _to_lower_one(ch: String) -> String:
    var upper = String("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    var lower = String("abcdefghijklmnopqrstuvwxyz")
    var i = 0
    while i < len(upper):
        if ch == String(upper[i]):
            return String(lower[i])
        i += 1
    return ch

fn _to_lower_str(s: String) -> String:
    var t = String("")
    var i = 0
    while i < len(s):
        t += _to_lower_one(String(s[i]))
        i += 1
    return t


fn str_extract(df0: DataFrame, col: String, pattern: String) -> List[String]:
    var idx = _find_col_idx(df0, col)
    var out = List[String]()

    var accept = False
    if pattern == String("^([A-Z])") or pattern == String(r"^([A-Z])"):
        accept = True

    var r = 0
    while r < df0.nrows():
        var s = df0.cols[idx][r]
        if accept and len(s) >= 1:
            var first_char: UInt8 = UInt8(0)
            try:
                first_char = UInt8(s[0])
            except:
                out.append(String(""))
                r += 1
                continue

            if first_char >= UInt8(ord("A")) and first_char <= UInt8(ord("Z")):
                out.append(String(s[0]))
            else:
                out.append(String(""))
        else:
            out.append(String(""))
        r += 1
    return out





# ---- str_replace_regex ----
fn str_replace_regex(df0: DataFrame, col: String, pattern: String, repl: String) -> List[String]:
    var idx = _find_col_idx(df0, col)
    var out = List[String]()

    var vowel_pattern = String("[aeiou]")
    var accept = (pattern == vowel_pattern) or (pattern == String(r"[aeiou]"))

    var r = 0
    while r < df0.nrows():
        var s = df0.cols[idx][r]
        if not accept:
            pass  # do nothing
        else:
            var t = String("")
            var i = 0
            while i < len(s):
                var ch = s[i]
                if ch == "a" or ch == "e" or ch == "i" or ch == "o" or ch == "u":
                    t += repl
                else:
                    t += String(ch)
                i += 1
            out.append(t)
        r += 1
    return out


# ---- Moved from __init__.mojo ----
fn col_str_concat(df0: DataFrame, col: String, prefix: String = String(""), suffix: String = String("")) -> List[String]:
    var idx = _find_col_idx(df0, col)
    if idx < 0:
        print("[WARN] col_str_concat: column not found -> returning empty list")
        return List[String]()
    var out = List[String]()
    var r = 0
    while r < df0.nrows():
        var s = df0.cols[idx][r]
        out.append(prefix + s + suffix)
        r += 1
    return out
# ---- Small sentinels ----
# ---- Small sentinels ----
fn NaN() -> String:     return String("NaN")
fn NoneStr() -> String: return String("")

 

# Helper: check if string is NA/empty
fn _isna(s: String) -> Bool:
    return s == String("")

# Core: check if 'sub' exists in 's'
fn str_contains_s(s: String, sub: String) -> Bool:
    if len(sub) == 0:
        return True
    var i = 0
    while i + len(sub) <= len(s):
        var is_match: Bool = True
        var j = 0
        while j < len(sub):
            if s[i+j] != sub[j]:
                is_match = False
                break
            j += 1
        if is_match:
            return True
        i += 1
    return False

# Main: str_contains for DataFrame column
fn str_contains(df0: DataFrame, col: String, sub: String, case_insensitive: Bool, na_false: Bool) -> List[String]:
    var idx = _find_col_idx(df0, col)
    var out = List[String]()

    var needle = sub
    if case_insensitive:
        needle = _to_lower_str(sub)

    var r = 0
    while r < df0.nrows():
        var s = df0.cols[idx][r]
        if _isna(s):
            out.append(String("False") if na_false else String(""))
            r += 1
            continue

        var hay = s
        if case_insensitive:
            hay = _to_lower_str(s)

        var found = str_contains_s(hay, needle)
        out.append(String("True") if found else String("False"))
        r += 1

    return out
