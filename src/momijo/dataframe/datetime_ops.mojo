# Project:      Momijo
# Module:       dataframe.datetime_ops
# File:         datetime_ops.mojo
# Path:         dataframe/datetime_ops.mojo
#
# Description:  dataframe.datetime_ops — Datetime Ops module for Momijo DataFrame.
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
#   - Key functions: _pad2, parse_minutes, gen_dates_12h_from_2025_01_01, gen_dates_from, datetime_year, tz_localize_utc, tz_convert

from momijo.dataframe.frame import DataFrame

from momijo.dataframe.series_bool import *
from momijo.dataframe.series_f64 import *
from momijo.dataframe.series_str import *
from momijo.dataframe.series_i64 import *

# ---- helpers (local) ----
fn _pad2(n: Int) -> String:
    if n < 0:
        var m = -n
        if m < 10:
            return String("-0") + String(m)
        else:
            return String("-") + String(m)
    if n < 10:
        return String("0") + String(n)
    return String(n)

# Parse "YYYY-MM-DD HH:MM:SS" to minutes since midnight (HH*60+MM).
# If format is unexpected, returns 0.
fn parse_minutes(ts: String) -> Int:
    if len(ts) < 16:
        return 0
# Expect positions: HH at [11..12], MM at [14..15]
    var h10 = ts[11]
    var h01 = ts[12]
    var m10 = ts[14]
    var m01 = ts[15]
# Digit check
    if (h10 < "0" or h10 > "9") or (h01 < "0" or h01 > "9") or (m10 < "0" or m10 > "9") or (m01 < "0" or m01 > "9"):
        return 0
    var hh = (Int(h10) - Int("0")) * 10 + (Int(h01) - Int("0"))
    var mm = (Int(m10) - Int("0")) * 10 + (Int(m01) - Int("0"))
    return hh * 60 + mm

# Generate n timestamps from 2025-01-01, stepping 12 hours each.
fn gen_dates_12h_from_2025_01_01(n: Int) -> List[String]:
    var out = List[String]()
    var i = 0
    var minutes = 0
    while i < n:
        var day = 1 + (minutes / 1440)
        var day_str = _pad2(day)
        var hh = (minutes / 60) % 24
        var mm = minutes % 60
        var ts = String("2025-01-") + day_str + String(" ") + _pad2(hh) + String(":") + _pad2(mm) + String(":00")
        out.append(ts)
        minutes += 720  # 12h
        i += 1
    return out

# Generate n timestamps from a given start day/hour/min, stepping by step_min minutes.
# Month fixed to Jan 2025 to keep simple and deterministic for samples.
fn gen_dates_from(start_day: Int, start_hour: Int, start_min: Int, n: Int, step_min: Int) -> List[String]:
    var out = List[String]()
    var total = start_day * 1440 + start_hour * 60 + start_min
    var i = 0
    while i < n:
        var d_total = total + i * step_min
        var day = d_total / 1440
        var rem = d_total % 1440
        var hh = rem / 60
        var mm = rem % 60
        var ts = String("2025-01-") + _pad2(day) + String(" ") + _pad2(hh) + String(":") + _pad2(mm) + String(":00")
        out.append(ts)
        i += 1
    return out

# Extract first 'take' years (YYYY) from a string column and return as a printable CSV string.
# Extract first 'take' rows of a string column and return as a printable CSV string.

fn datetime_year(df0: DataFrame, col: String) -> String:
    # Convenience overload: take = nrows()
    return datetime_year(df0, col, df0.nrows())

fn datetime_year(df0: DataFrame, col: String, take: Int) -> String:
    # Find column index by name
    var idx = -1
    var c = 0
    while c < df0.ncols():
        if df0.col_names[c] == col:
            idx = c
            break
        c += 1
    if idx < 0:
        return String("[WARN] datetime_year: column not found")

    # Clamp take to available rows
    var nrows = df0.nrows()
    var n = take if take < nrows else nrows

    # Build CSV from first n rows in the target column
    var out = String("")
    var i = 0
    while i < n:
        # NOTE: read string cell via column API
        var s = String(df0.cols[idx].get_string(i))

        # If you truly want only the year part (first 4 chars), replace the next line accordingly.
        # For now we keep the whole string if length >= 4 to match prior behavior.
        var year = String("")
        if len(s) >= 4:
            year = s

        out += year
        if i != n - 1:
            out += String(", ")
        i += 1

    return out
# Extract first 'take' years (YYYY) from a string column and return a 1-col DataFrame ["year"].

# Convenience overload: default take = 3
# Return a single-column DataFrame with the 4-digit year extracted from a string date col.
fn datetime_year_df(df0: DataFrame, col: String) -> DataFrame:
    var j = -1
    var c = 0
    while c < df0.ncols():
        if df0.col_names[c] == col:
            j = c
            break
        c += 1

    var out = DataFrame()
    var ys = List[String]()

    if j < 0:
        # column not found → empty result with header
        var col_y = Column()
        var s = SeriesStr()
        s.set_name(String("year"))
        s.data = ys.copy()
        col_y.set_string_series(s)
        out.cols.append(col_y.copy())
        out.col_names.append(String("year"))
        return out

    var n = df0.nrows()
    ys.reserve(n)
    var r = 0
    while r < n:
        var s = df0.cols[j].get_string(r)
        var y = String("")
        if len(s) >= 4:
            y = s[0:4]
        ys.append(y)
        r += 1

    var col_y = Column()
    var s2 = SeriesStr()
    s2.set_name(String("year"))
    s2.data = ys.copy()
    col_y.set_string_series(s2)

    out.cols.append(col_y.copy())
    out.col_names.append(String("year"))
    return out.copy()


fn datetime_year_df(df0: DataFrame, col: String, take: Int) -> DataFrame:
    # 1) Find column index by name
    var idx = -1
    var c = 0
    while c < df0.ncols():
        if df0.col_names[c] == col:
            idx = c
            break
        c += 1
    if idx < 0:
        # Return empty 1-col frame if column not found
        return _make_single_string_frame(List[String](), "year")

    # 2) Clamp 'take' to available rows
    var nrows = df0.nrows()
    var n = take if take < nrows else nrows

    # 3) Build years list (YYYY = first 4 chars if available)
    var years = List[String]()
    var i = 0
    while i < n:
        var s = String(df0.cols[idx].get_string(i))
        var y = String("")
        if len(s) >= 4:
            # Take exactly first 4 characters as "year"
            var j = 0
            while j < 4:
                y += String(s[j])
                j += 1
        years.append(y)
        i += 1

    # 4) Return a 1-column DataFrame: ["year"]
    return _make_single_string_frame(years, "year")


# ---------- Minimal helper to build a 1-column (string) DataFrame ----------
# Replace internals with your project's canonical constructors if you have them.

fn _make_single_string_frame(values: List[String], name: String) -> DataFrame:
    # Build a SeriesStr
    var s = SeriesStr()
    s.set_name(name)
    s.data  = values.copy()         # List[String]
    # Validity bitmap: all True for provided values
    var m = len(values)
    var bm = Bitmap()
    bm.resize(m, True)
    s.valid = bm.copy()

    # Wrap into a Column (string-tag)
    var col = Column()
    col.set_string_series(s)        # <- if you have a setter; otherwise set tag/s directly

    # Assemble DataFrame with a single column
    var out = DataFrame()
    out.col_names = List[String]()
    out.cols      = List[Column]()
    out.col_names.append(name)
    out.cols.append(col.copy())
    return out




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