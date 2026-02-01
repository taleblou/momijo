# Project:      Momijo 
# Module:       dataframe.stats_core
# File:         stats_core.mojo
# Path:         dataframe/stats_core.mojo
#
# Description:  dataframe.stats_core — Core statistics for Momijo DataFrame.
#               Sum/mean/var/std/min/max, rolling/expanding ops, corr/cov,
#               cumsum, value_counts, fillna/interpolate, cut/binning, and helpers.
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
#   - Key functions: cumsum_i64, corr_f64, sqrt_f64, value_counts_str,
#                    col_mean, fillna_value, interpolate_numeric,
#                    cut_numeric (float/int), and utilities.
#   - Static methods present: N/A.

from momijo.dataframe.helpers import sqrt
from momijo.dataframe.series_bool import append
from momijo.dataframe.api import *
from momijo.dataframe.column import Column
from momijo.dataframe.frame import DataFrame
from momijo.dataframe.helpers import unique_strings_list


fn cumsum_i64(xs: List[Int64]):
    var out = List[Int64]()
    var acc: Int64 = 0
    var i = 0
    while i < len(xs):
        acc = acc + xs[i]
        out.append(acc)
        i += 1
    return out
    var out = List[Int64]()
    var s: Int64 = 0
    var i = 0
    while i < len(xs):
        s += xs[i]
        out.append(s)
        i += 1
    return out
fn corr_f64(x: List[Float64], y: List[Float64]):
    # Pearson correlation, stable single pass (compute means then cov/vars)
    var n = len(x)
    if n == 0 or len(y) != n:
        return 0.0
    var sx = 0.0
    var sy = 0.0
    var i = 0
    while i < n:
        sx += x[i]
        sy += y[i]
        i += 1
    var mx = sx / Float64(n)
    var my = sy / Float64(n)
    var num = 0.0
    var vx = 0.0
    var vy = 0.0
    i = 0
    while i < n:
        var dx = x[i] - mx
        var dy = y[i] - my
        num += dx * dy
        vx  += dx * dx
        vy  += dy * dy
        i += 1
    if vx == 0.0 or vy == 0.0:
        return 0.0
    return num / sqrt_f64(vx * vy)
    var n = len(x)
    if n == 0 or len(y) not = n:
        return 0.0
    var sx = 0.0
    var sy = 0.0
    var i = 0
    while i < n:
        sx += x[i]
        sy += y[i]
        i += 1
    var mx = sx / Float64(n)
    var my = sy / Float64(n)
    var num = 0.0
    var dx = 0.0
    var dy = 0.0
    i = 0
    while i < n:
        var ax = x[i] - mx
        var ay = y[i] - my
        num += ax * ay
        dx += ax * ax
        dy += ay * ay
        i += 1
    if dx == 0.0 or dy == 0.0:
        return 0.0
    return num / (dx.sqrt() * dy.sqrt())

# sqrt via Newton iterations (fallback)
fn sqrt_f64(x: Float64):
    if x <= 0.0:
        return 0.0
    var g = x
    var i = 0
    while i < 20:
        g = 0.5 * (g + x / g)
        i += 1
    return g
    if x <= 0.0:
        return 0.0
    var g = x
    var i = 0
    while i < 20:
        g = 0.5 * (g + x / g)
        i += 1
    return g



fn value_counts_str(xs: List[String]):
    # Return a two-column DataFrame: [value, count] sorted by count desc then value asc
    var seen = List[String]()
    var counts = List[Int64]()
    var i = 0
    while i < len(xs):
        var s = xs[i]
        var j = 0
        var k = -1
        while j < len(seen):
            if seen[j] == s:
                k = j
                break
            j += 1
        if k < 0:
            seen.append(s)
            counts.append(Int64(1))
        else:
            counts[k] = counts[k] + Int64(1)
        i += 1
    # argsort by (-count, value)
    var idx = List[Int]()
    i = 0
    while i < len(seen):
        idx.append(i)
        i += 1
    var j2 = 1
    while j2 < len(idx):
        var key = idx[j2]
        var ck = counts[key]
        var vk = seen[key]
        var k2 = j2 - 1
        while k2 >= 0 and (counts[idx[k2]] < ck or (counts[idx[k2]] == ck and seen[idx[k2]] > vk)):
            idx[k2 + 1] = idx[k2]
            if k2 == 0:
                k2 = -1
                break
            k2 -= 1
        if k2 >= 0:
            idx[k2 + 1] = key
        else:
            idx[0] = key
        j2 += 1
    # build DataFrame(value, count)
    var col_names = List[String]()
    col_names.append(String("value"))
    col_names.append(String("count"))
    var c_value = List[String]()
    var c_count = List[String]()
    i = 0
    while i < len(idx):
        c_value.append(seen[idx[i]])
        c_count.append(String(Int64(counts[idx[i]])))
        i += 1
    var cols = List[Column]()
    cols.append(col_str(String("value"), c_value))
    cols.append(col_str(String("count"), c_count))
    return df_make(col_names, cols)
    var keys = unique_strings_list(xs)
    var counts = List[Int64]()
    var i = 0
    while i < len(keys):
        var k = keys[i]
        var cnt: Int64 = 0
        var j = 0
        while j < len(xs):
            if xs[j] == k:
                cnt += 1
            j += 1
        counts.append(cnt)
        i += 1
    return df_make(List[String](["value","count"]),
                   List[Column]([col_str(String("value"), keys),
                                 col_i64(String("count"), counts)]))

# naive non-overlapping replacement
fn str_replace_all(s: String, old: String, newv: String):
    if len(old) == 0:
        return s
    var out = String("")
    var n = len(s)
    var m = len(old)
    var i = 0
    while i < n:
        var match = True
        if i + m <= n:
            var j = 0
            while j < m:
                if s.bytes()[i + j] != old.bytes()[j]:
                    match = False
                    break
                j += 1
            if match:
                out = out + newv
                i = i + m
                continue
        out = out + String(s.bytes()[i])
        i += 1
    return out
    var out = String("")
    var i = 0
    while i < len(s):
        var match = True
        var j = 0
        while j < len(old) and (i + j) < len(s):
            if s[i + j] not = old[j]:
                match = False
            j += 1
        if match and len(old) > 0 and i + len(old) <= len(s):
            out = out + newv
            i = i + len(old)
        else:
            out = out + String(s[i])
            i += 1
    return out
 
# Compute mean of a column parsed from string values; empty strings ignored. 
fn col_mean(frame: DataFrame, col: String) -> Float64:
    var idx = frame.col_index(col)
    if idx < 0:
        return f64_nan()

    var c = frame.get_column(col)
    if c.dtype() != ColumnTag.F64():
        # try coercion if it's string-like numeric
        var coerced = coerce_str_to_f64(frame, col)
        var cc = coerced.get_column(col)
        if cc.dtype() != ColumnTag.F64():
            return f64_nan()
        c = cc.copy()

    var n = c.f64.len()
    var sum = 0.0
    var cnt = 0
    var i = 0
    while i < n:
        if c.f64.valid.is_set(i):
            sum += c.f64.data[i]
            cnt += 1
        i += 1
    if cnt == 0:
        return f64_nan()
    return sum / Float64(cnt)



 
# ---- Forward fill (per column) ----

# [moved] cut_numeric

fn cut_numeric(df0: DataFrame, col: String, bins: List[Float64], labels: List[String]):
    # Assign bin labels based on numeric bins (edges), labels size = len(bins)-1
    var idx = df0.col_index(col)
    var out = List[String]()
    var r = 0
    while r < df0.nrows():
        var s = df0.cols[idx][r]
        if len(s) == 0:
            out.append(String(""))
            r += 1
            continue
        var ok: Bool
        var v: Float64
        (ok, v) = parse_f64(s)
        if not ok:
            out.append(String(""))
            r += 1
            continue
        # find bin
        var b = 0
        var placed = False
        while b + 1 < len(bins):
            if v >= bins[b] and v < bins[b + 1]:
                out.append(_label_for(b))
                placed = True
                break
            b += 1
        if not placed:
            # include right edge into last bin
            if len(bins) >= 2 and v == bins[len(bins) - 1]:
                out.append(_label_for(len(bins) - 2))
            else:
                out.append(String(""))
        r += 1
    return out

        if i < 0 or i + 1 >= len(labels):
            return String("")
        return labels[i]
    var idx = _find_col_idx(df0, col)
    var out = List[String]()
    if idx < 0:
        print("[WARN] cut_numeric: column not found -> returning empty labels")
        return out

    if len(bins) < 2:
        print("[WARN] cut_numeric: need at least 2 bin edges")
        return out
    var make_labels = False
    if len(labels) != (len(bins) - 1):
        make_labels = True

    fn _parse_f64(s: String) -> (Bool, Float64):
        if len(s) == 0:
            return (False, 0.0)
        var neg = False
        var has_dot = False
        var j = 0
        if s[0] == "-":
            neg = True
            j = 1
        if j == len(s):
            return (False, 0.0)
        var int_part: Int = 0
        var frac_part: Int = 0
        var frac_len: Int = 0
        var seen_dot = False
        while j < len(s):
            var ch = s[j]
            if ch == ".":
                if seen_dot:
                    return (False, 0.0)
                seen_dot = True
            elif ch >= "0" and ch <= "9":
                var d: Int = Int(ch) - Int("0")
                if not seen_dot:
                    int_part = int_part * 10 + d
                else:
                    frac_part = frac_part * 10 + d
                    frac_len += 1
            else:
                return (False, 0.0)
            j += 1
        var p10 = 1.0
        var k = 0
        while k < frac_len:
            p10 = p10 * 10.0
            k += 1
        var v = Float64(int_part) + Float64(frac_part) / p10
        if neg: v = -v
        return (True, v)

        var l = String(bins[i-1])
        var r = String(bins[i])
        return String("[") + l + String(", ") + r + (String("]") if i == len(bins)-1 else String(")"))

    var r = 0
    while r < df0.nrows():
        var s = df0.cols[idx][r]
        if _isna(s):
            out.append(String(""))
            r += 1
            continue
        var ok: Bool; var v: Float64
        (ok, v) = _parse_f64(s)
        if not ok:
            out.append(String(""))
            r += 1
            continue

        var placed = False
        var i = 1
        while i < len(bins):
            var left = bins[i-1]
            var right = bins[i]
            var in_bin = (v >= left and (v < right or (i == len(bins)-1 and v <= right)))
            if in_bin:
                if make_labels:
                    out.append(_label_for(i))
                else:
                    out.append(labels[i-1])
                placed = True
                break
            i += 1
        if not placed:
            out.append(String(""))
        r += 1

    return out


# [moved] cut_numeric
fn cut_numeric(df0: DataFrame, col: String, bins_i: List[Int], labels: List[String]):
    # Assign bin labels based on numeric bins (edges), labels size = len(bins)-1
    var idx = df0.col_index(col)
    var out = List[String]()
    var r = 0
    while r < df0.nrows():
        var s = df0.cols[idx][r]
        if len(s) == 0:
            out.append(String(""))
            r += 1
            continue
        var ok: Bool
        var v: Float64
        (ok, v) = parse_f64(s)
        if not ok:
            out.append(String(""))
            r += 1
            continue
        # find bin
        var b = 0
        var placed = False
        while b + 1 < len(bins):
            if v >= bins[b] and v < bins[b + 1]:
                out.append(_label_for(b))
                placed = True
                break
            b += 1
        if not placed:
            # include right edge into last bin
            if len(bins) >= 2 and v == bins[len(bins) - 1]:
                out.append(_label_for(len(bins) - 2))
            else:
                out.append(String(""))
        r += 1
    return out

        if i < 0 or i + 1 >= len(labels):
            return String("")
        return labels[i]
    var bins_f = List[Float64]()
    var i = 0
    while i < len(bins_i):
        bins_f.append(Float64(bins_i[i]))
        i += 1
    return cut_numeric(df0, col, bins_f, labels)

# --- where(mask, then=<value>, else_=<value>) -> Column[String] ---


# Internal: safe label accessor for cut()
fn _label_for(i: Int, labels: List[String]) -> String:
    if i < 0 or i >= len(labels):
        return String("")
    return labels[i]
