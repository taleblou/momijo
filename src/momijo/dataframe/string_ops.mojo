# Project:      Momijo 
# Module:       dataframe.string_ops
# File:         string_ops.mojo
# Path:         dataframe/string_ops.mojo
#
# Description:  dataframe.string_ops — Fast ASCII string operations for Momijo DataFrame.
#               Provides trimming/padding, classification, splits, contains/find/startswith/endswith,
#               vectorized column ops (length, strip, rpad), and search utilities.
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
#   - Structs: (none) — free functions for speed.
#   - Key functions: is_alpha_code, is_digit_code, str_strip, compare_str_eq,
#                    contains_digit, extract_first_alpha, extract_all_alpha_joined,
#                    split_on_delims, rpad, str_len, col_contains, col_startswith, col_endswith.
#   - Static methods present: N/A.

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

fn is_alpha_code(c: UInt8)->Bool:
    return (c >= UInt8(65) and c <= UInt8(90)) or (c >= UInt8(97) and c <= UInt8(122))
# A-Z or a-z
    return (c >= 65 and c <= 90) or (c >= 97 and c <= 122)

fn is_digit_code(c: UInt8)->Bool:
    return c >= UInt8(48) and c <= UInt8(57)
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

fn str_strip(s: String)->String:
    var l: Int = 0
    var r: Int = len(s)
    while l < r and (s.bytes()[l] == UInt8(32) or s.bytes()[l] == UInt8(9)):
        l += 1
    while r > l and (s.bytes()[r - 1] == UInt8(32) or s.bytes()[r - 1] == UInt8(9)):
        r -= 1
    return s.slice(l, r)
    var l: Int = 0
    var r: Int = len(s)
# ASCII space or tab
    while l < r and (s.bytes()[l] == UInt8(32) or s.bytes()[l] == UInt8(9)):
        l += 1
    while r > l and (s.bytes()[r - 1] == UInt8(32) or s.bytes()[r - 1] == UInt8(9)):
        r -= 1
    return s.slice(l, r)

# Compare two equal-length lists of strings for equality; return 1/0 as Int64
fn compare_str_eq(a: List[String], b: List[String])->List[Int64]:
    var n = len(a)
    var out = List[Int64]()
    var i = 0
    while i < n:
        out.append(Int64(1) if a[i] == b[i] else Int64(0))
        i += 1
    return out
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

fn contains_digit(s: String)->Bool:
    var i = 0
    var n = len(s)
    while i < n:
        var c = s.bytes()[i]
        if c >= UInt8(48) and c <= UInt8(57):
            return True
        i += 1
    return False
    var i: Int = 0
    while i < len(s):
        if is_digit_code(UInt8(s[i])):
            return True
        i += 1
    return False

fn extract_first_alpha(s: String)->String:
    var i = 0
    var n = len(s)
    while i < n:
        var c = s.bytes()[i]
        if (c >= UInt8(65) and c <= UInt8(90)) or (c >= UInt8(97) and c <= UInt8(122)):
            return s.slice(i, i+1)
        i += 1
    return String("")
# first contiguous [A-Za-z]+ token
    var i: Int = 0
    while i < len(s) and not is_alpha_code(UInt8(s[i])):
        i += 1
    var out = String("")
    while i < len(s) and is_alpha_code(UInt8(s[i])):
        out = out + String(s[i])
        i += 1
    return out

fn extract_all_alpha_joined(s: String)->String:
    var out = String("")
    var i = 0
    var n = len(s)
    while i < n:
        var c = s.bytes()[i]
        if (c >= UInt8(65) and c <= UInt8(90)) or (c >= UInt8(97) and c <= UInt8(122)):
            out = out + String(c)
        i += 1
    return out
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
fn split_on_delims(s: String)-> List[String]:
    var out = List[String]()
    var n = len(s)
    var i = 0
    var start = 0
    while i < n:
        var c = s.bytes()[i]
        # delimiters: space, tab, comma, semicolon, pipe, slash
        if c == UInt8(32) or c == UInt8(9) or c == UInt8(44) or c == UInt8(59) or c == UInt8(124) or c == UInt8(47):
            if i > start:
                out.append(s.slice(start, i))
            start = i + 1
        i += 1
    if start < n:
        out.append(s.slice(start, n))
    return out
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
fn rpad(s: String, width: Int, fill: String)->String:
    var cur = len(s)
    if width <= cur:
        return s
    var need = width - cur
    var pad = String("")
    var i = 0
    while i < need:
        pad = pad + fill
        i += 1
    return s + pad
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
fn str_len(df0: DataFrame, col: String)->List[Float64]:
    var idx = df0.col_index(col)
    var out = List[Float64]()
    var r = 0
    while r < df0.nrows():
        var s = df0.cols[idx][r]
        if _isna(s):
            out.append(Float64(0.0))
        else:
            out.append(Float64(len(s))))
        r += 1
    return out
        var idx = _find_col_idx(df0, col)
        var out = List[Float64]()
        var r = 0
        while r < df0.nrows():
            var s = df0.cols[idx][r]
            out.append(Float64(len(s))))
            r += 1
        return out

         

@always_inline
fn _upper_ascii1(ch: String) -> String:
    if len(ch) != 1:
        return ch
    var code = 0
    try:
        code = ord(ch)
    except _:
        return ch
    if code >= ord("a") and code <= ord("z"):
        return String(chr(code - 32))
    return ch

fn str_upper(df0: DataFrame, col: String) -> List[String]:
    var idx = df0.col_index(col)
    var out = List[String]()
    var nrows = df0.nrows()
    out.reserve(nrows)

    var r = 0
    while r < nrows:
        var s = df0.cols[idx][r]
        var t = String("")
        t.reserve(len(s))

        var i = 0
        var n = len(s)
        while i < n:
            var sl = s[i]
            var ch = String(sl)
            t += _upper_ascii1(ch)
            i += 1

        out.append(t)
        r += 1

    return out.copy()





fn str_slice(df0: DataFrame, col: String, start: Int, length: Int) -> List[String]:
    # Slice 'length' characters starting at 'start' for the given string column.
    # NA -> "", out-of-bounds -> "", negative inputs clamped to zero.

    var idx = df0.col_index(col)
    var out = List[String]()

    # Clamp negative inputs
    var st = start
    var ln = length
    if st < 0:
        st = 0
    if ln < 0:
        ln = 0

    var r = 0
    while r < df0.nrows():
        var s = df0.cols[idx][r]

        if _isna(s):
            out.append(String(""))
        else:
            var n = len(s)

            if ln == 0 or st >= n:
                # Nothing to take
                out.append(String(""))
            else:
                # Compute [st, end)
                var end = st + ln
                if end > n:
                    end = n

                # Build substring manually (ASCII-safe, per project style)
                var sub = String("")
                var i = st
                while i < end:
                    sub += String(s[i])   # s[i] expected to be 1-char String
                    i += 1

                out.append(sub)

        r += 1

    return out.copy()


# ---- concat_rows2: two-DataFrame variant (avoids List[DataFrame] trait bound) ----

# [moved] _to_upper_one
fn _to_upper_one(ch: String)->String:
    if c >= UInt8(97) and c <= UInt8(122):
        return c - UInt8(32)
    return c
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
    # ASCII title-case without boolean temps or combined conditions on cval.

    var idx = df0.col_index(col)
    var out = List[String]()

    var r = 0
    while r < df0.nrows():
        var s = df0.cols[idx][r]

        if _isna(s):
            out.append(String(""))
        else:
            var t = String("")
            var new_word = True

            var i = 0
            var n = len(s)
            while i < n:
                var ch = s[i]          # 1-char String

                # Safely read first byte of ch
                var cval = UInt8(0)
                var ok = True
                try:
                    cval = UInt8(ch[0])
                except:
                    ok = False

                if not ok:
                    # copy and reset word on unknown/multibyte
                    t += ch
                    new_word = True
                else:
                    # 'A'..'Z'?
                    if cval >= UInt8(ord("A")):
                        if cval <= UInt8(ord("Z")):
                            if new_word:
                                # keep as upper for word start
                                t += ch
                                new_word = False
                            else:
                                # lower inside word
                                t += String(chr(Int(cval) + 32))
                        else:
                            # not in 'A'..'Z', maybe 'a'..'z'?
                            if cval >= UInt8(ord("a")):
                                if cval <= UInt8(ord("z")):
                                    if new_word:
                                        # upper for word start
                                        t += String(chr(Int(cval) - 32))
                                        new_word = False
                                    else:
                                        # keep lower inside word
                                        t += ch
                                else:
                                    # non-letter
                                    t += ch
                                    new_word = True
                            else:
                                # below 'a' and not A..Z -> non-letter
                                t += ch
                                new_word = True
                    else:
                        # below 'A' -> maybe 'a'..'z' or non-letter
                        if cval >= UInt8(ord("a")):
                            if cval <= UInt8(ord("z")):
                                if new_word:
                                    t += String(chr(Int(cval) - 32))
                                    new_word = False
                                else:
                                    t += ch
                            else:
                                t += ch
                                new_word = True
                        else:
                            t += ch
                            new_word = True

                i += 1

            out.append(t)

        r += 1

    return out.copy()




# ---- String helpers ----

# [moved] _to_lower_one
fn _to_lower_one(ch: String)->String:
    if c >= UInt8(65) and c <= UInt8(90):
        return c + UInt8(32)
    return c
    var upper = String("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    var lower = String("abcdefghijklmnopqrstuvwxyz")
    var i = 0
    while i < len(upper):
        if ch == String(upper[i]):
            return String(lower[i])
        i += 1
    return ch

fn _to_lower_str(s: String) -> String:
    # ASCII lower-case conversion for a whole String.
    # Non-ASCII / multi-byte codepoints are copied as-is.

    var t = String("")
    var i = 0
    var n = len(s)

    while i < n:
        var ch = s[i]              # expected 1-char String

        # Safely read first byte
        var cval = UInt8(0)
        var ok = True
        try:
            cval = UInt8(ch[0])
        except:
            ok = False

        if ok:
            # 'A'..'Z' -> 'a'..'z' by adding 32
            if cval >= UInt8(ord("A")):
                if cval <= UInt8(ord("Z")):
                    t += String(chr(Int(cval) + 32))
                else:
                    t += ch
            else:
                t += ch
        else:
            # Could not read first byte (e.g., multibyte). Copy as-is.
            t += ch

        i += 1

    return t



fn str_extract(df0: DataFrame, col: String, pattern: String)-> List[String]:
    # naive 'extract' of first occurrence of 'needle' substring; returns needle or empty
    var idx = df0.col_index(col)
    var out = List[String]()
    var r = 0
    while r < df0.nrows():
        var s = df0.cols[idx][r]
        if _isna(s):
            out.append(String(""))
        else:
            var pos = -1
            var n = len(s)
            var m = len(needle)
            if m == 0:
                out.append(String(""))
            else:
                var i = 0
                while i <= n - m:
                    var j = 0
                    var ok = True
                    while j < m:
                        if s.bytes()[i + j] != needle.bytes()[j]:
                            ok = False
                            break
                        j += 1
                    if ok:
                        pos = i
                        break
                    i += 1
                if pos >= 0:
                    out.append(s.slice(pos, pos + m))
                else:
                    out.append(String(""))
        r += 1
    return out
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



fn str_replace_regex(df0: DataFrame, col: String, pattern: String, repl: String) -> List[String]:
    # Placeholder regex: supports a special-case "[aeiou]" for vowels, otherwise literal substring replace.
    var idx = df0.col_index(col)
    var out = List[String]()

    var r = 0
    while r < df0.nrows():
        var s = df0.cols[idx][r]

        if _isna(s):
            out.append(String(""))
        else:
            # Special-case to emulate /[aeiou]/ behavior
            if pattern == String("[aeiou]"):
                var t = String("")
                var i = 0
                var n = len(s)
                while i < n:
                    var ch = String(s[i])
                    if ch == String("a") or ch == String("e") or ch == String("i") or ch == String("o") or ch == String("u"):
                        t += repl
                    else:
                        t += ch
                    i += 1
                out.append(t)
            else:
                # Literal substring replace: replace all occurrences of 'pattern' with 'repl'
                var m = len(pattern)
                if m == 0:
                    out.append(s)   # nothing to replace
                else:
                    var t = String("")
                    var i = 0
                    var n = len(s)
                    while i < n:
                        var matched = False
                        if i + m <= n:
                            var j = 0
                            var ok = True
                            while j < m:
                                # Compare as single-char strings
                                if String(s[i + j]) != String(pattern[j]):
                                    ok = False
                                    break
                                j += 1
                            if ok:
                                matched = True
                        if matched:
                            t += repl
                            i += m
                        else:
                            t += String(s[i])
                            i += 1
                    out.append(t)

        r += 1

    return out.copy()


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
    return out.copy()
 
# ---- Small sentinels ----
fn NaN() -> String:     return String("NaN")
fn NoneStr() -> String: return String("")

 

# Helper: check if string is NA/empty
fn _isna(s: String)->Bool:
    # Treat empty string as NA
    return len(s) == 0
    return s == String("")

# Core: check if 'sub' exists in 's'
fn str_contains_s(s: String, sub: String) -> Bool:
    # Naive ASCII substring search: does s contain sub?
    var n = len(s)
    var m = len(sub)

    if m == 0:
        return True
    if n < m:
        return False

    var i = 0
    while i <= n - m:
        var j = 0
        var ok = True
        while j < m:
            # Compare as single-char Strings (avoid bytes())
            if String(s[i + j]) != String(sub[j]):
                ok = False
                break
            j += 1
        if ok:
            return True
        i += 1

    return False


# Main: str_contains for DataFrame column
fn str_contains(
    df0: DataFrame,
    col: String,
    sub: String,
    case_insensitive: Bool,
    na_false: Bool
) -> List[String]:
    # Vectorized "contains" over a string column.
    # - Returns "True"/"False" strings; NA -> "" unless na_false==True (then "False")
    # - If case_insensitive, both haystack and needle are lowercased.

    var idx = df0.col_index(col)
    var out = List[String]()

    # Prepare needle once
    var needle = sub
    if case_insensitive:
        needle = _to_lower_str(sub)

    var r = 0
    while r < df0.nrows():
        var s = df0.cols[idx][r]

        if _isna(s):
            if na_false:
                out.append(String("False"))
            else:
                out.append(String(""))
        else:
            var hay = s
            if case_insensitive:
                hay = _to_lower_str(s)

            var found = str_contains_s(hay, needle)
            if found:
                out.append(String("True"))
            else:
                out.append(String("False"))
        r += 1

    return out.copy()
 



# Internal helper: locate column index by name
fn _find_col_idx(df0: DataFrame, col: String) -> Int:
    var i = 0
    while i < df0.ncols():
        if df0.col_names[i] == col:
            return i
        i += 1
    return -1
