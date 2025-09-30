# Project:      Momijo
# Module:       dataframe.stats_core
# File:         stats_core.mojo
# Path:         dataframe/stats_core.mojo
#
# Description:  dataframe.stats_core — Stats Core module for Momijo DataFrame.
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
#   - Key functions: cumsum_i64, corr_f64, sqrt_f64, value_counts_str, str_replace_all, col_mean, fillna_value, interpolate_numeric, cut_numeric, _parse_f64, _label_for



from momijo.dataframe.helpers import sqrt
from momijo.dataframe.series_bool import append
from momijo.dataframe.api import col_i64, col_str, df_make
from momijo.dataframe.column import Column
from momijo.dataframe.frame import DataFrame
from momijo.dataframe.helpers import unique_strings_list


fn cumsum_i64(xs: List[Int64]) -> List[Int64]
    var out = List[Int64]()
    var s: Int64 = 0
    var i = 0
    while i < len(xs):
        s += xs[i]
        out.append(s)
        i += 1
    return out
fn corr_f64(x: List[Float64], y: List[Float64]) -> Float64
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
fn sqrt_f64(x: Float64) -> Float64
    if x <= 0.0:
        return 0.0
    var g = x
    var i = 0
    while i < 20:
        g = 0.5 * (g + x / g)
        i += 1
    return g



fn value_counts_str(xs: List[String]) -> DataFrame
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
fn str_replace_all(s: String, old: String, newv: String) -> String
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

# ---- Moved from __init__.mojo (facade helpers) ----

# [moved] col_mean
fn col_mean(df0: DataFrame, col: String) -> Float64:
    var idx = -1
    var i = 0
    while i < df0.ncols():
        if df0.col_names[i] == col:
            idx = i
            break
        i += 1
    if idx < 0:
        print("[WARN] col_mean: column not found; returning 0.0")
        return 0.0
    var vals = List[String]()
    var r = 0
    while r < df0.nrows():
        vals.append(df0.cols[idx][r])
        r += 1
    var st = _compute_stats(vals)
    return st.mean


# ---- NA fill with a specific value (numeric-friendly) ----

# [moved] fillna_value
fn fillna_value(df0: DataFrame, col: String, value: Float64) -> DataFrame:
    var idx = -1
    var i = 0
    while i < df0.ncols():
        if df0.col_names[i] == col:
            idx = i
            break
        i += 1
    if idx < 0:
        print("[WARN] fillna_value: column not found -> returning original DataFrame")
        return df0

    var col_names = df0.col_names
    var cols = List[Column]()
    var c = 0
    while c < df0.ncols():
        var vals = List[String]()
        var r = 0
        while r < df0.nrows():
            var s = df0.cols[c][r]
            if c == idx and _isna(s):
                vals.append(String(value))
            else:
                vals.append(s)
            r += 1
        cols.append(_col_str(col_names[c], vals))
        c += 1
    return _df_make(col_names, cols)


# ---- Linear interpolation for numeric columns ----

# [moved] interpolate_numeric
fn interpolate_numeric(df0: DataFrame, col: String) -> DataFrame:
    var idx = -1
    var i = 0
    while i < df0.ncols():
        if df0.col_names[i] == col:
            idx = i
            break
        i += 1
    if idx < 0:
        print("[WARN] interpolate_numeric: column not found -> returning original DataFrame")
        return df0

# Collect numeric values and mask
    var n = df0.nrows()
    var vals = List[Float64]()
    var known = List[Bool]()
    var r = 0
    while r < n:
        var s = df0.cols[idx][r]
        if _isna(s):
            vals.append(0.0)
            known.append(False)
        else:
# simple parse: allow leading '-', digits and one '.'
            var ok = True
            var has_dot = False
            var j = 0
            if (len(s) > 0) and (s[0] == "-"):
                j = 1
            if j == len(s):
                ok = False
            while j < len(s):
                var ch = s[j]
                if ch == ".":
                    if has_dot:
                        ok = False
                        break
                    has_dot = True
                elif (ch == "0") or (ch == "1") or (ch == "2") or (ch == "3") or (ch == "4") or (ch == "5") or (ch == "6") or (ch == "7") or (ch == "8") or (ch == "9"):
                    # ok
                    pass
                else:
                    ok = False
                    break
                j += 1
            if ok:
                # convert
                # minimal parse without pow10 accumulation: split by '.'
                var neg = False
                var k = 0
                if (len(s) > 0) and (s[0] == "-"):
                    neg = True
                    k = 1
                var int_part: Int = 0
                var frac_part: Int = 0
                var frac_len: Int = 0
                var seen_dot = False
                while k < len(s):
                    var ch2 = s[k]
                    if ch2 == ".":
                        seen_dot = True
                    else:
                        var d: Int = 0
                        if ch2 == "1": d = 1
                        elif ch2 == "2": d = 2
                        elif ch2 == "3": d = 3
                        elif ch2 == "4": d = 4
                        elif ch2 == "5": d = 5
                        elif ch2 == "6": d = 6
                        elif ch2 == "7": d = 7
                        elif ch2 == "8": d = 8
                        elif ch2 == "9": d = 9
                        else: d = 0
                        if not seen_dot:
                            int_part = int_part * 10 + d
                        else:
                            frac_part = frac_part * 10 + d
                            frac_len += 1
                    k += 1
                var p10 = 1.0
                var t = 0
                while t < frac_len:
                    p10 = p10 * 10.0
                    t += 1
                var v = Float64(int_part) + Float64(frac_part) / p10
                if neg: v = -v
                vals.append(v)
                known.append(True)
            else:
                vals.append(0.0)
                known.append(False)
        r += 1

# If none known, return original
    var any_known = False
    var idx_first = -1
    var idx_last_known = -1
    r = 0
    while r < n:
        if known[r]:
            any_known = True
            idx_last_known = r
            if idx_first < 0:
                idx_first = r
        r += 1
    if not any_known:
        print("[WARN] interpolate_numeric: no numeric values -> returning original DataFrame")
        return df0

# Fill leading
    var lead = 0
    while lead < idx_first:
        vals[lead] = vals[idx_first]
        lead += 1

# Interpolate gaps
    var last_i = idx_first
    var last_v = vals[idx_first]
    var pos = idx_first + 1
    while pos < n:
        if known[pos]:
            var gap = pos - last_i
            if gap > 1:
                var step = (vals[pos] - last_v) / Float64(gap)
                var k2 = 1
                while k2 < gap:
                    vals[last_i + k2] = last_v + step * Float64(k2)
                    k2 += 1
            last_i = pos
            last_v = vals[pos]
        pos += 1

# Fill trailing
    var tail = last_i + 1
    while tail < n:
        vals[tail] = last_v
        tail += 1

# Build new DataFrame
    var col_names = df0.col_names
    var cols = List[Column]()
    var c = 0
    while c < df0.ncols():
        var out_vals = List[String]()
        var r2 = 0
        while r2 < df0.nrows():
            if c == idx:
                out_vals.append(String(vals[r2]))
            else:
                out_vals.append(df0.cols[c][r2])
            r2 += 1
        cols.append(_col_str(col_names[c], out_vals))
        c += 1
    return _df_make(col_names, cols)


# ---- Forward fill (per column) ----

# [moved] cut_numeric

fn cut_numeric(df0: DataFrame, col: String, bins: List[Float64], labels: List[String]) -> List[String]:
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

    fn _label_for(i: Int) -> String:
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
fn cut_numeric(df0: DataFrame, col: String, bins_i: List[Int], labels: List[String]) -> List[String]:
    var bins_f = List[Float64]()
    var i = 0
    while i < len(bins_i):
        bins_f.append(Float64(bins_i[i]))
        i += 1
    return cut_numeric(df0, col, bins_f, labels)

# --- where(mask, then=..., else_=...) -> Column[String] ---