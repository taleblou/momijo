# MIT License
# Copyright (c) 2025 Morteza Talebou and Mitra Daneshmand
# Project: momijo  |  Source: https://github.com/taleblou/momijo
# SPDX-License-Identifier: MIT
# Website: https://taleblou.ir/
# Repository: https://github.com/taleblou/momijo
#
# Project: momijo.utils
# File: src/momijo/utils/string_utils.mojo
#
# Checklist compliance:
# - Only 'var' (no 'let'); no 'export'; no global mutable state
# - Explicit imports; English comments
# - Prefer mut/out over inout
# - Deterministic behavior; avoid exceptions in library code
# - Use byte-level safe access (s.bytes()[i]) for indexing
#
# Notes:
# - These utilities assume ASCII-safe operations for case conversion and classification.
# - For UTF-8 strings, byte-level ops may split codepoints; functions document ASCII-only semantics.

from stdlib.list import List
from stdlib.string import String

# ---------------------------------------------
# ASCII classification helpers
# ---------------------------------------------

fn is_ascii_alpha(b: UInt8) -> Bool:
    return (b >= UInt8(65) and b <= UInt8(90)) or (b >= UInt8(97) and b <= UInt8(122))

fn is_ascii_digit(b: UInt8) -> Bool:
    return b >= UInt8(48) and b <= UInt8(57)

fn is_ascii_alnum(b: UInt8) -> Bool:
    return is_ascii_alpha(b) or is_ascii_digit(b)

fn is_ascii_space(b: UInt8) -> Bool:
    # space, tab, CR, LF
    return b == UInt8(32) or b == UInt8(9) or b == UInt8(10) or b == UInt8(13)

# ---------------------------------------------
# Case conversion (ASCII only)
# ---------------------------------------------

fn to_lower_ascii(s: String) -> String:
    var out = String("")
    var n = len(s)
    var i = 0
    while i < n:
        var b = s.bytes()[i]
        if b >= UInt8(65) and b <= UInt8(90):
            b = UInt8(b + UInt8(32))
        out = out + String.from_utf8([b])
        i += 1
    return out

fn to_upper_ascii(s: String) -> String:
    var out = String("")
    var n = len(s)
    var i = 0
    while i < n:
        var b = s.bytes()[i]
        if b >= UInt8(97) and b <= UInt8(122):
            b = UInt8(b - UInt8(32))
        out = out + String.from_utf8([b])
        i += 1
    return out

# ---------------------------------------------
# Trim / strip
# ---------------------------------------------

fn lstrip_ws(s: String) -> String:
    var n = len(s)
    var i = 0
    while i < n and is_ascii_space(s.bytes()[i]):
        i += 1
    var out = String("")
    while i < n:
        out = out + String.from_utf8([s.bytes()[i]])
        i += 1
    return out

fn rstrip_ws(s: String) -> String:
    var n = len(s)
    if n == 0: return s
    var i = n - 1
    while Int64(i) >= Int64(0) and is_ascii_space(s.bytes()[i]):
        if Int64(i) == Int64(0):
            # all whitespace
            return String("")
        i = i - 1
    var out = String("")
    var j = 0
    while j <= i:
        out = out + String.from_utf8([s.bytes()[j]])
        j += 1
    return out

fn strip_ws(s: String) -> String:
    return rstrip_ws(lstrip_ws(s))

# ---------------------------------------------
# Starts/Ends/Contains (ASCII-safe byte scan)
# ---------------------------------------------

fn starts_with(s: String, prefix: String) -> Bool:
    var ns = len(s)
    var np = len(prefix)
    if np > ns: return False
    var i = 0
    while i < np:
        if s.bytes()[i] != prefix.bytes()[i]:
            return False
        i += 1
    return True

fn ends_with(s: String, suffix: String) -> Bool:
    var ns = len(s)
    var nf = len(suffix)
    if nf > ns: return False
    var i = 0
    while i < nf:
        if s.bytes()[ns - nf + i] != suffix.bytes()[i]:
            return False
        i += 1
    return True

fn contains_substr(s: String, sub: String) -> Bool:
    return find_substr(s, sub) >= Int64(0)

# ---------------------------------------------
# Find / rfind (naive scan)
# ---------------------------------------------

fn find_substr(s: String, sub: String) -> Int64:
    var ns = len(s)
    var np = len(sub)
    if np == 0: return Int64(0)
    if np > ns: return Int64(-1)
    var i = 0
    while i + np <= ns:
        var j = 0
        var ok = True
        while j < np:
            if s.bytes()[i + j] != sub.bytes()[j]:
                ok = False
                break
            j += 1
        if ok: return Int64(i)
        i += 1
    return Int64(-1)

fn rfind_substr(s: String, sub: String) -> Int64:
    var ns = len(s)
    var np = len(sub)
    if np == 0: return Int64(ns)
    if np > ns: return Int64(-1)
    var i = ns - np
    while True:
        var j = 0
        var ok = True
        while j < np:
            if s.bytes()[i + j] != sub.bytes()[j]:
                ok = False
                break
            j += 1
        if ok: return Int64(i)
        if Int64(i) == Int64(0): break
        i = i - 1
    return Int64(-1)

# ---------------------------------------------
# Replace (all occurrences, non-overlapping)
# ---------------------------------------------

fn replace_all(s: String, old: String, newv: String) -> String:
    var ns = len(s)
    var no = len(old)
    if no == 0:
        # insert newv between characters and at ends
        var out = String("")
        var i = 0
        out = out + newv
        while i < ns:
            out = out + String.from_utf8([s.bytes()[i]]) + newv
            i += 1
        return out
    var out = String("")
    var i = 0
    while i < ns:
        var k = find_substr(slice_bytes(s, i, ns), old)
        if k < Int64(0):
            # append rest
            while i < ns:
                out = out + String.from_utf8([s.bytes()[i]])
                i += 1
            break
        var kk = Int64(i) + k
        # append bytes s[i:kk]
        var j = i
        while Int64(j) < kk:
            out = out + String.from_utf8([s.bytes()[j]])
            j += 1
        # append new
        out = out + newv
        i = Int64(kk) + Int64(no)
    return out

# ---------------------------------------------
# Split / join
# ---------------------------------------------

fn split_once(s: String, sep: String) -> (String, String, Bool):
    var k = find_substr(s, sep)
    if k < Int64(0):
        return (s, String(""), False)
    var a = slice_str(s, Int64(0), k)
    var b = slice_str(s, k + Int64(len(sep)), Int64(len(s)))
    return (a, b, True)

fn rsplit_once(s: String, sep: String) -> (String, String, Bool):
    var k = rfind_substr(s, sep)
    if k < Int64(0):
        return (s, String(""), False)
    var a = slice_str(s, Int64(0), k)
    var b = slice_str(s, k + Int64(len(sep)), Int64(len(s)))
    return (a, b, True)

fn split_all(s: String, sep: String) -> List[String]:
    var out = List[String]()
    if len(sep) == 0:
        # split into single bytes
        var i = 0
        var n = len(s)
        while i < n:
            out.append(String.from_utf8([s.bytes()[i]]))
            i += 1
        return out
    var start = Int64(0)
    var n = Int64(len(s))
    while start <= n:
        var k = find_substr(slice_str(s, start, n), sep)
        if k < Int64(0):
            out.append(slice_str(s, start, n))
            break
        var mid = start + k
        out.append(slice_str(s, start, mid))
        start = mid + Int64(len(sep))
    return out

fn join_with(sep: String, parts: List[String]) -> String:
    var out = String("")
    var n = len(parts)
    var i = 0
    while i < n:
        if i > 0: out = out + sep
        out = out + parts[i]
        i += 1
    return out

# ---------------------------------------------
# Line utilities
# ---------------------------------------------

fn split_lines(s: String, keepends: Bool = False) -> List[String]:
    var out = List[String]()
    var n = len(s)
    var i = 0
    var start = 0
    while i < n:
        var b = s.bytes()[i]
        if b == UInt8(10): # LF
            if keepends:
                out.append(slice_str(s, start, i + 1))
            else:
                out.append(slice_str(s, start, i))
            start = i + 1
        elif b == UInt8(13): # CR
            # treat CRLF as single line break
            if i + 1 < n and s.bytes()[i + 1] == UInt8(10):
                if keepends:
                    out.append(slice_str(s, start, i + 2))
                else:
                    out.append(slice_str(s, start, i))
                start = i + 2
                i += 1
            else:
                if keepends:
                    out.append(slice_str(s, start, i + 1))
                else:
                    out.append(slice_str(s, start, i))
                start = i + 1
        i += 1
    if start <= n:
        out.append(slice_str(s, start, n))
    return out

# ---------------------------------------------
# Whitespace split (ASCII)
# ---------------------------------------------

fn split_whitespace(s: String) -> List[String]:
    var out = List[String]()
    var n = len(s)
    var i = 0
    while i < n:
        # skip spaces
        while i < n and is_ascii_space(s.bytes()[i]):
            i += 1
        if i >= n: break
        # collect token
        var start = i
        while i < n and not is_ascii_space(s.bytes()[i]):
            i += 1
        out.append(slice_str(s, start, i))
    return out

# ---------------------------------------------
# Quote helpers
# ---------------------------------------------

fn strip_quotes(s: String) -> String:
    var n = len(s)
    if n >= 2:
        var a = s.bytes()[0]
        var b = s.bytes()[n - 1]
        if (a == UInt8(34) and b == UInt8(34)) or (a == UInt8(39) and b == UInt8(39)):
            return slice_str(s, Int64(1), Int64(n - 1))
    return s

fn ensure_quoted(s: String, quote: UInt8 = UInt8(34)) -> String:
    var q = quote
    var n = len(s)
    if n >= 2 and s.bytes()[0] == q and s.bytes()[n - 1] == q:
        return s
    return String.from_utf8([q]) + s + String.from_utf8([q])

# ---------------------------------------------
# Padding / repeat
# ---------------------------------------------

fn repeat_char(ch: UInt8, times: Int64) -> String:
    var out = String("")
    var t = times
    if t <= Int64(0): return out
    var i = Int64(0)
    while i < t:
        out = out + String.from_utf8([ch])
        i += Int64(1)
    return out

fn pad_left(s: String, width: Int64, ch: UInt8 = UInt8(32)) -> String:
    var n = Int64(len(s))
    if n >= width: return s
    return repeat_char(ch, width - n) + s

fn pad_right(s: String, width: Int64, ch: UInt8 = UInt8(32)) -> String:
    var n = Int64(len(s))
    if n >= width: return s
    return s + repeat_char(ch, width - n)

# ---------------------------------------------
# Safe slicing helpers (byte-indexed)
# ---------------------------------------------

fn slice_bytes(s: String, start: Int64, end: Int64) -> List[UInt8]:
    var ns = Int64(len(s))
    var a = start
    var b = end
    if a < Int64(0): a = Int64(0)
    if b > ns: b = ns
    if b < a: b = a
    var out = List[UInt8]()
    var i = a
    while i < b:
        out.append(s.bytes()[i])
        i += Int64(1)
    return out

fn slice_str(s: String, start: Int64, end: Int64) -> String:
    var bs = slice_bytes(s, start, end)
    return String.from_utf8(bs)

# ---------------------------------------------
# CSV-safe join (simple): quote + escape inner quotes
# ---------------------------------------------

fn csv_escape_field(s: String) -> String:
    var needs = False
    var n = len(s)
    var i = 0
    while i < n:
        var b = s.bytes()[i]
        if b == UInt8(44) or b == UInt8(34) or b == UInt8(10) or b == UInt8(13):
            needs = True
            break
        i += 1
    if not needs:
        return s
    # escape inner quotes by doubling
    var out = String("")
    i = 0
    while i < n:
        var b2 = s.bytes()[i]
        if b2 == UInt8(34):
            out = out + String("""")
        else:
            out = out + String.from_utf8([b2])
        i += 1
    return String(""") + out + String(""")

fn csv_join(fields: List[String]) -> String:
    var out = String("")
    var n = len(fields)
    var i = 0
    while i < n:
        if i > 0: out = out + String(",")
        out = out + csv_escape_field(fields[i])
        i += 1
    return out

# ---------------------------------------------
# Self-test
# ---------------------------------------------

fn _self_test() -> Bool:
    var ok = True

    ok = ok and (to_lower_ascii(String("AbC")) == String("abc"))
    ok = ok and (to_upper_ascii(String("aBc")) == String("ABC"))

    ok = ok and (strip_ws(String("  x\n")).bytes()[0] == UInt8(120))  # 'x'

    ok = ok and starts_with(String("hello"), String("he"))
    ok = ok and ends_with(String("hello"), String("lo"))
    ok = ok and contains_substr(String("hello"), String("ell"))
    ok = ok and (find_substr(String("hello"), String("z")) == Int64(-1))

    var parts = split_all(String("a,b,,c"), String(","))
    ok = ok and (len(parts) == 4)

    var joined = join_with(String("|"), parts)
    ok = ok and (contains_substr(joined, String("|")))

    var lines = split_lines(String("a\r\nb\nc"))
    ok = ok and (len(lines) == 3)

    var ws = split_whitespace(String(" a  b\t c"))
    ok = ok and (len(ws) == 3)

    var rep = repeat_char(UInt8(120), Int64(3))  # 'x'
    ok = ok and (rep == String("xxx"))

    var padded = pad_left(String("x"), Int64(3), UInt8(46))  # '.'
    ok = ok and (padded == String("..x"))

    var csv = csv_join(parts)
    ok = ok and contains_substr(csv, String(","))

    var (a, b, oksp) = split_once(String("a=b=c"), String("="))
    ok = ok and oksp and (a == String("a")) and (b == String("b=c"))

    var (ra, rb, okrp) = rsplit_once(String("a=b=c"), String("="))
    ok = ok and okrp and (ra == String("a=b")) and (rb == String("c"))

    return ok
