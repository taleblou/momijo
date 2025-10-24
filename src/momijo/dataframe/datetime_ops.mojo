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
@always_inline
fn atoi(s: String) -> Int:
    var n = len(s)
    if n == 0:
        return 0
    var sign = 1
    var i = 0
    var c0 = s[0]
    if c0 == '-':
        sign = -1
        i = 1
    elif c0 == '+':
        i = 1

    var val = 0
    while i < n:
        var ch = s[i]
        if ch < '0' or ch > '9':
            break
        val = val * 10 + (ord(ch) - ord('0'))
        i += 1
    return sign * val

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

  
 

# -------------------- DataFrame API --------------------
fn resample(self, freq: String) -> Resampler:
    return Resampler(self, freq)
 

  
# --- Add to: dataframe/frame.mojo -------------------------------------------------
# English-only comments. Conforms to Momijo style (var-only, no raises, Copyable.)
# Minimal time-based resampling over a datetime-like String index.
# Index values must be ISO-like: "YYYY-MM-DD" or "YYYY-MM-DD HH:MM:SS".
# Supported frequencies: "<k>D" (days), e.g., "1D", "2D", "7D".

# -------------------- tiny date helpers (Gregorian) --------------------
@always_inline
fn _clip_date_part(s: String, start: Int, n: Int) -> String:
    var out = String("")
    var i = 0
    # while i < n and (start + i) < len(s):
    #     out = out + String(s[start + i])
    #     i += 1
    return out

@always_inline
fn _parse_date_ymd(date_str: String) -> (Bool, Int, Int, Int):
    # Accepts "YYYY-MM-DD" or longer; only first 10 chars are used.
    if len(date_str) < 10:
        return (False, 0, 0, 0)
    var y_str = _clip_date_part(date_str, 0, 4)
    var m_str = _clip_date_part(date_str, 5, 2)
    var d_str = _clip_date_part(date_str, 8, 2)
    var ok = True
    var y = 0
    var m = 0
    var d = 0
    var i = 0
    while i < len(y_str):
        var c = y_str[i]
        if c < '0' or c > '9':
            ok = False
        i += 1
    i = 0
    while i < len(m_str):
        var c2 = m_str[i]
        if c2 < '0' or c2 > '9':
            ok = False
        i += 1
    i = 0
    while i < len(d_str):
        var c3 = d_str[i]
        if c3 < '0' or c3 > '9':
            ok = False
        i += 1
    if not ok:
        return (False, 0, 0, 0)
    y = atoi(y_str)
    m = atoi(m_str)
    d = atoi(d_str)
    return (True, y, m, d)
@always_inline
fn _days_from_civil(y0: Int, m0: Int, d0: Int) -> Int:
    # Howard Hinnant's integer-only algorithm, shifted so 1970-01-01 maps to 0.
    var y: Int = y0
    var m: Int = m0
    var d: Int = d0

    if m <= 2:
        y -= 1

    var era: Int
    if y >= 0:
        era = Int(y / 400)
    else:
        era = Int((y - 399) / 400)

    var yoe: Int = y - era * 400

    var mp: Int
    if m > 2:
        mp = m - 3
    else:
        mp = m + 9

    var doy: Int = Int((153 * mp + 2) / 5 + d - 1)
    var doe: Int =  Int(yoe * 365 + yoe / 4 - yoe / 100 + doy)

    return era * 146097 + doe - 719468
@always_inline
fn _civil_from_days(z: Int) -> (Int, Int, Int):
    # Inverse of _days_from_civil with the same 719468 shift.
    var z2: Int = z + 719468

    var era: Int
    if z2 >= 0:
        era = Int(z2 / 146097)
    else:
        era = Int((z2 - 146096) / 146097)

    var doe: Int = z2 - era * 146097
    var yoe: Int = Int((doe - doe / 1460 + doe / 36524 - doe / 146096) / 365)  # spaces intentional to avoid accidental token merge; keep as single identifiers in your code
    var y: Int = yoe + era * 400
    var doy: Int = doe - Int(365 * yoe + yoe / 4 - yoe / 100)
    var mp: Int = Int((5 * doy + 2) / 153)
    var d: Int = Int(doy - (153 * mp + 2) / 5 + 1)

    var m: Int
    if mp < 10:
        m = mp + 3
    else:
        m = mp - 9

    if m <= 2:
        y += 1

    return (y, m, d)
@always_inline
fn _ymd_to_string(y: Int, m: Int, d: Int) -> String:
    # Format YYYY-MM-DD with zero-padded month/day.
    var yy: String = String(y)

    var mm: String = String(m)
    if m < 10:
        mm = String("0") + mm

    var dd: String = String(d)
    if d < 10:
        dd = String("0") + dd

    return yy + String("-") + mm + String("-") + dd
@always_inline
fn _pow10_abs(e: Int) -> Float64:
    var out = 1.0
    var i = 0
    while i < e:
        out = out * 10.0
        i += 1
    return out

@always_inline
fn _char_at(s: String, i: Int) -> (Bool, String):
    if i < 0: return (False, String(""))
    if i >= len(s): return (False, String(""))
    return (True, _clip_date_part(s, i, 1))   # safe 1-char slice

@always_inline
fn _is_digit1(ch: String) -> Bool:
    if len(ch) != 1: return False
    var c = ch[0]
    return (c >= '0' and c <= '9')

@always_inline
fn _digit1_to_int(ch: String) -> Int:
    # call only if _is_digit1(ch) is True
    var c = ch[0]
    return ord(c) - ord('0')


@always_inline
fn _atof_noraise(s: String) -> (Bool, Float64):
    var n = len(s)
    if n == 0: return (False, 0.0)

    var i = 0
    var sign = 1.0

    # sign
    if i < n:
        var ok0: Bool; var ch0: String
        (ok0, ch0) = _char_at(s, i)
        if ok0 and ch0 == String("-"):
            sign = -1.0
            i = i + 1
        elif ok0 and ch0 == String("+"):
            i = i + 1

    var seen_digit = False
    var val = 0.0

    # integer part
    while i < n:
        var ok: Bool; var ch: String
        (ok, ch) = _char_at(s, i)
        if not ok: break
        if _is_digit1(ch):
            seen_digit = True
            val = val * 10.0 + Float64(_digit1_to_int(ch))
            i = i + 1
        else:
            break

    # fractional part
    if i < n:
        var okd: Bool; var dot: String
        (okd, dot) = _char_at(s, i)
        if okd and dot == String("."):
            i = i + 1
            var scale = 1.0
            while i < n:
                var okf: Bool; var ch2: String
                (okf, ch2) = _char_at(s, i)
                if not okf: break
                if _is_digit1(ch2):
                    seen_digit = True
                    scale = scale * 10.0
                    val = val + Float64(_digit1_to_int(ch2)) / scale
                    i = i + 1
                else:
                    break

    # exponent
    var exp_val = 0
    var exp_sign = 1
    if i < n:
        var oke: Bool; var ce: String
        (oke, ce) = _char_at(s, i)
        if oke and (ce == String("e") or ce == String("E")):
            i = i + 1
            if i < n:
                var oks: Bool; var cs: String
                (oks, cs) = _char_at(s, i)
                if oks and (cs == String("+") or cs == String("-")):
                    if cs == String("-"):
                        exp_sign = -1
                    i = i + 1
            var exp_seen = False
            while i < n:
                var okx: Bool; var ch3: String
                (okx, ch3) = _char_at(s, i)
                if not okx: break
                if _is_digit1(ch3):
                    exp_seen = True
                    exp_val = exp_val * 10 + _digit1_to_int(ch3)
                    i = i + 1
                else:
                    break
            if not exp_seen:
                exp_val = 0
                exp_sign = 1

    if not seen_digit:
        return (False, 0.0)

    var f = val * sign
    if exp_val != 0:
        var p = _pow10_abs(exp_val)
        if exp_sign < 0:
            f = f / p
        else:
            f = f * p

    return (True, f)



# -------------------- freq parser --------------------
@always_inline
fn _parse_day_freq(spec: String) -> Int:
    # supports "D" and "kD" (k>=1)
    var n = len(spec)
    if n == 0:
        return 1
    # trailing must be 'D'
    var last = _clip_date_part(spec, n - 1, 1)
    if last != String("D"):
        return 1
    # read leading integer (if any)
    var num_str = String("")
    var i = 0
    while i < n - 1:
        var c = spec[i]
        if c >= '0' and c <= '9':
            num_str = num_str + String(c)
        i += 1
    if len(num_str) == 0:
        return 1
    var k = atoi(num_str)
    if k <= 0:
        k = 1
    return k

@always_inline
fn _try_parse_float(s: String) -> (Bool, Float64):
    if len(s) == 0:
        return (False, 0.0)
    return _atof_noraise(s)


# -------------------- Resampler --------------------
struct Resampler(Copyable, Movable):
    var base: DataFrame
    var step_days: Int

    fn __init__(out self, base: DataFrame, freq: String):
        self.base = base.copy()
        self.step_days = _parse_day_freq(freq)

    # Sum reducer; if numeric_only is True, only numeric-looking columns are included.
    fn sum(self, numeric_only: Bool = True) -> DataFrame:
        var out_names = List[String]()
        var out_data  = List[List[String]]()
        var out_index = List[String]()

        # Build list of candidate value columns (exclude the index)
        var val_cols = List[Int]()
        var c = 0
        while c < len(self.base.col_names):
            # exclude original index column if it exists among columns
            var is_idx_col = (self.base.index_name == self.base.col_names[c])
            if not is_idx_col:
                val_cols.append(c)
            c += 1

        # Decide which columns to keep based on numeric_only
        var keep_flags = List[Bool]()
        var j = 0
        while j < len(val_cols):
            var cj = val_cols[j]
            var probe = String("")
            if self.base.cols[cj].len() > 0:
                probe = self.base.cols[cj].get_string(0)
            var is_num = False
            var tmp_ok: Bool
            var tmp_val: Float64
            (tmp_ok, tmp_val) = _try_parse_float(probe)
            is_num = tmp_ok
            if numeric_only:
                keep_flags.append(is_num)
            else:
                keep_flags.append(True)
            j += 1

        # Map bin_label -> position in accum arrays
        var bin_labels = List[String]()
        var bin_pos = Dict[String, Int]()

        # Prepare accumulators per kept column (created lazily once we see first bin)
        var have_inited = False
        var acc = List[List[Float64]]()

        # Anchor: first valid date in index
        var nrows = len(self.base.index_vals)
        var anchor_days = 0
        var found_anchor = False
        var r0 = 0
        while r0 < nrows and not found_anchor:
            var ok: Bool
            var y: Int
            var m: Int
            var d: Int
            (ok, y, m, d) = _parse_date_ymd(self.base.index_vals[r0])
            if ok:
                anchor_days = _days_from_civil(y, m, d)
                found_anchor = True
            r0 += 1
        if not found_anchor:
            # No valid dates; return empty copy with same columns
            var k = 0
            while k < len(val_cols):
                if keep_flags[k]:
                    out_names.append(self.base.col_names[val_cols[k]])
                    out_data.append(List[String]())
                k += 1
            var out_df = DataFrame(out_names, out_data, out_index, self.base.index_name)
            return out_df.copy()

        # Iterate rows and accumulate
        var r = 0
        while r < nrows:
            var ok2: Bool
            var yy: Int
            var mm: Int
            var dd: Int
            (ok2, yy, mm, dd) = _parse_date_ymd(self.base.index_vals[r])
            if ok2:
                var ord = _days_from_civil(yy, mm, dd)
                var bucket = (ord - anchor_days) // self.step_days
                var start_ord = anchor_days + bucket * self.step_days
                var sy: Int
                var sm: Int
                var sd: Int
                (sy, sm, sd) = _civil_from_days(Int(start_ord))
                var label = _ymd_to_string(sy, sm, sd)

                # lookup/create bin position
                var pos_opt = bin_pos.get(label)
                var pos = -1
                if pos_opt is None:
                    pos = len(bin_labels)
                    bin_pos[label] = pos
                    bin_labels.append(label)
                    if not have_inited:
                        # init accum arrays on first bin
                        var k2 = 0
                        while k2 < len(val_cols):
                            if keep_flags[k2]:
                                var arr = List[Float64]()
                                acc.append(arr.copy())
                            k2 += 1
                        have_inited = True
                    # extend acc arrays with a zero slot for new bin
                    var a = 0
                    while a < len(acc):
                        acc[a].append(0.0)
                        a += 1
                else:
                    pos = pos_opt.value()

                # add row values
                var kk = 0
                var kept_idx = 0
                while kk < len(val_cols):
                    if keep_flags[kk]:
                        var col_idx = val_cols[kk]
                        var s = self.base.cols[col_idx].get_string(r)
                        var okv: Bool
                        var fv: Float64
                        (okv, fv) = _try_parse_float(s)
                        if okv:
                            acc[kept_idx][pos] = acc[kept_idx][pos] + fv
                        kept_idx += 1
                    kk += 1
            r += 1

        # Build output columns (as strings) in kept order
        var out_cols = List[List[String]]()
        var ci = 0
        while ci < len(val_cols):
            if keep_flags[ci]:
                var arr = List[String]()
                var p = 0
                while p < len(bin_labels):
                    arr.append(String(acc[ci][p]))
                    p += 1
                out_cols.append(arr.copy())
                out_names.append(self.base.col_names[val_cols[ci]])
            ci += 1

        # Assemble DataFrame with resampled index
        var out_df2 = DataFrame(out_names, out_cols, bin_labels, self.base.index_name)
        return out_df2.copy()

    fn mean(self, numeric_only: Bool = True) -> DataFrame:
        var out_names = List[String]()
        var out_data  = List[List[String]]()
        var out_index = List[String]()

        # candidate value columns (exclude original index column name)
        var val_cols = List[Int]()
        var c = 0
        while c < len(self.base.col_names):
            var is_idx_col = (self.base.index_name == self.base.col_names[c])
            if not is_idx_col:
                val_cols.append(c)
            c += 1

        # decide which columns to keep
        var keep_flags = List[Bool]()
        var j = 0
        while j < len(val_cols):
            var cj = val_cols[j]
            var probe = String("")
            if self.base.cols[cj].len() > 0:
                probe = self.base.cols[cj].get_string(0)
            var okp: Bool
            var tmp: Float64
            (okp, tmp) = _try_parse_float(probe)
            if numeric_only:
                keep_flags.append(okp)
            else:
                keep_flags.append(True)
            j += 1

        # bins
        var bin_labels = List[String]()
        var bin_pos = Dict[String, Int]()

        # accumulators: sum and count for each kept column
        var have_inited = False
        var acc_sum = List[List[Float64]]()
        var acc_cnt = List[List[Int]]()

        # anchor from first valid date
        var nrows = len(self.base.index_vals)
        var anchor_days = 0
        var found_anchor = False
        var r0 = 0
        while r0 < nrows and not found_anchor:
            var ok: Bool; var y: Int; var m: Int; var d: Int
            (ok, y, m, d) = _parse_date_ymd(self.base.index_vals[r0])
            if ok:
                anchor_days = _days_from_civil(y, m, d)
                found_anchor = True
            r0 += 1
        if not found_anchor:
            var k = 0
            while k < len(val_cols):
                if keep_flags[k]:
                    out_names.append(self.base.col_names[val_cols[k]])
                    out_data.append(List[String]())
                k += 1
            var out_df_empty = DataFrame(out_names, out_data, out_index, self.base.index_name)
            return out_df_empty.copy()

        # iterate rows
        var r = 0
        while r < nrows:
            var ok2: Bool; var yy: Int; var mm: Int; var dd: Int
            (ok2, yy, mm, dd) = _parse_date_ymd(self.base.index_vals[r])
            if ok2:
                var ord = _days_from_civil(yy, mm, dd)
                var bucket = (ord - anchor_days) // self.step_days
                var start_ord = anchor_days + bucket * self.step_days
                var sy: Int; var sm: Int; var sd: Int
                (sy, sm, sd) = _civil_from_days(start_ord)
                var label = _ymd_to_string(sy, sm, sd)

                # bin position
                var pos_opt = bin_pos.get(label)
                var pos = -1
                if pos_opt is None:
                    pos = len(bin_labels)
                    bin_pos[label] = pos
                    bin_labels.append(label)
                    if not have_inited:
                        var k2 = 0
                        while k2 < len(val_cols):
                            if keep_flags[k2]:
                                var arr_s = List[Float64]()
                                var arr_c = List[Int]()
                                acc_sum.append(arr_s.copy())
                                acc_cnt.append(arr_c.copy())
                            k2 += 1
                        have_inited = True
                    var a = 0
                    while a < len(acc_sum):
                        acc_sum[a].append(0.0)
                        acc_cnt[a].append(0)
                        a += 1
                else:
                    pos = pos_opt.value()

                # accumulate
                var kk = 0
                var kept_idx = 0
                while kk < len(val_cols):
                    if keep_flags[kk]:
                        var col_idx = val_cols[kk]
                        var s = self.base.cols[col_idx].get_string(r)
                        var okv: Bool; var fv: Float64
                        (okv, fv) = _try_parse_float(s)
                        if okv:
                            acc_sum[kept_idx][pos] = acc_sum[kept_idx][pos] + fv
                            acc_cnt[kept_idx][pos] = acc_cnt[kept_idx][pos] + 1
                        kept_idx += 1
                    kk += 1
            r += 1

        # build output columns (sum / count)
        var out_cols = List[List[String]]()
        var ci = 0
        while ci < len(val_cols):
            if keep_flags[ci]:
                var arr = List[String]()
                var p = 0
                while p < len(bin_labels):
                    var cnt = acc_cnt[ci][p]
                    if cnt > 0:
                        arr.append(String(acc_sum[ci][p] / Float64(cnt)))
                    else:
                        arr.append(String(""))   # empty if no numeric
                    p += 1
                out_cols.append(arr.copy())
                out_names.append(self.base.col_names[val_cols[ci]])
            ci += 1

        var out_df = DataFrame(out_names, out_cols, bin_labels, self.base.index_name)
        return out_df.copy()


        # ---- add INSIDE struct Resampler --------------------------------

    fn min(self, numeric_only: Bool = True) -> DataFrame:
        var out_names = List[String]()
        var out_data  = List[List[String]]()
        var out_index = List[String]()

        var val_cols = List[Int]()
        var c = 0
        while c < len(self.base.col_names):
            var is_idx_col = (self.base.index_name == self.base.col_names[c])
            if not is_idx_col:
                val_cols.append(c)
            c += 1

        var keep_flags = List[Bool]()
        var j = 0
        while j < len(val_cols):
            var cj = val_cols[j]
            var probe = String("")
            if self.base.cols[cj].len() > 0:
                probe = self.base.cols[cj].get_string(0)
            var okp: Bool; var tmp: Float64
            (okp, tmp) = _try_parse_float(probe)
            if numeric_only:
                keep_flags.append(okp)
            else:
                keep_flags.append(True)
            j += 1

        var bin_labels = List[String]()
        var bin_pos = Dict[String, Int]()
        var have_inited = False
        var acc_min = List[List[Float64]]()
        var acc_cnt = List[List[Int]]()

        var nrows = len(self.base.index_vals)
        var anchor_days = 0
        var found_anchor = False
        var r0 = 0
        while r0 < nrows and not found_anchor:
            var ok: Bool; var y: Int; var m: Int; var d: Int
            (ok, y, m, d) = _parse_date_ymd(self.base.index_vals[r0])
            if ok:
                anchor_days = _days_from_civil(y, m, d)
                found_anchor = True
            r0 += 1
        if not found_anchor:
            var k = 0
            while k < len(val_cols):
                if keep_flags[k]:
                    out_names.append(self.base.col_names[val_cols[k]])
                    out_data.append(List[String]())
                k += 1
            var out_df_empty = DataFrame(out_names, out_data, out_index, self.base.index_name)
            return out_df_empty.copy()

        var r = 0
        while r < nrows:
            var ok2: Bool; var yy: Int; var mm: Int; var dd: Int
            (ok2, yy, mm, dd) = _parse_date_ymd(self.base.index_vals[r])
            if ok2:
                var ord = _days_from_civil(yy, mm, dd)
                var bucket = (ord - anchor_days) // self.step_days
                var start_ord = anchor_days + bucket * self.step_days
                var sy: Int; var sm: Int; var sd: Int
                (sy, sm, sd) = _civil_from_days(start_ord)
                var label = _ymd_to_string(sy, sm, sd)

                var pos_opt = bin_pos.get(label)
                var pos = -1
                if pos_opt is None:
                    pos = len(bin_labels)
                    bin_pos[label] = pos
                    bin_labels.append(label)
                    if not have_inited:
                        var k2 = 0
                        while k2 < len(val_cols):
                            if keep_flags[k2]:
                                acc_min.append(List[Float64]())
                                acc_cnt.append(List[Int]())
                            k2 += 1
                        have_inited = True
                    var a = 0
                    while a < len(acc_min):
                        acc_min[a].append(0.0)
                        acc_cnt[a].append(0)
                        a += 1
                else:
                    pos = pos_opt.value()

                var kk = 0
                var kept_idx = 0
                while kk < len(val_cols):
                    if keep_flags[kk]:
                        var col_idx = val_cols[kk]
                        var s = self.base.cols[col_idx].get_string(r)
                        var okv: Bool; var fv: Float64
                        (okv, fv) = _try_parse_float(s)
                        if okv:
                            var cnt = acc_cnt[kept_idx][pos]
                            if cnt == 0:
                                acc_min[kept_idx][pos] = fv
                            else:
                                if fv < acc_min[kept_idx][pos]:
                                    acc_min[kept_idx][pos] = fv
                            acc_cnt[kept_idx][pos] = cnt + 1
                        kept_idx += 1
                    kk += 1
            r += 1

        var out_cols = List[List[String]]()
        var ci = 0
        while ci < len(val_cols):
            if keep_flags[ci]:
                var arr = List[String]()
                var p = 0
                while p < len(bin_labels):
                    var cnt = acc_cnt[ci][p]
                    if cnt > 0:
                        arr.append(String(acc_min[ci][p]))
                    else:
                        arr.append(String(""))
                    p += 1
                out_cols.append(arr.copy())
                out_names.append(self.base.col_names[val_cols[ci]])
            ci += 1

        var out_df = DataFrame(out_names, out_cols, bin_labels, self.base.index_name)
        return out_df.copy()


    fn max(self, numeric_only: Bool = True) -> DataFrame:
        var out_names = List[String]()
        var out_data  = List[List[String]]()
        var out_index = List[String]()

        var val_cols = List[Int]()
        var c = 0
        while c < len(self.base.col_names):
            var is_idx_col = (self.base.index_name == self.base.col_names[c])
            if not is_idx_col:
                val_cols.append(c)
            c += 1

        var keep_flags = List[Bool]()
        var j = 0
        while j < len(val_cols):
            var cj = val_cols[j]
            var probe = String("")
            if self.base.cols[cj].len() > 0:
                probe = self.base.cols[cj].get_string(0)
            var okp: Bool; var tmp: Float64
            (okp, tmp) = _try_parse_float(probe)
            if numeric_only:
                keep_flags.append(okp)
            else:
                keep_flags.append(True)
            j += 1

        var bin_labels = List[String]()
        var bin_pos = Dict[String, Int]()
        var have_inited = False
        var acc_max = List[List[Float64]]()
        var acc_cnt = List[List[Int]]()

        var nrows = len(self.base.index_vals)
        var anchor_days = 0
        var found_anchor = False
        var r0 = 0
        while r0 < nrows and not found_anchor:
            var ok: Bool; var y: Int; var m: Int; var d: Int
            (ok, y, m, d) = _parse_date_ymd(self.base.index_vals[r0])
            if ok:
                anchor_days = _days_from_civil(y, m, d)
                found_anchor = True
            r0 += 1
        if not found_anchor:
            var k = 0
            while k < len(val_cols):
                if keep_flags[k]:
                    out_names.append(self.base.col_names[val_cols[k]])
                    out_data.append(List[String]())
                k += 1
            var out_df_empty = DataFrame(out_names, out_data, out_index, self.base.index_name)
            return out_df_empty.copy()

        var r = 0
        while r < nrows:
            var ok2: Bool; var yy: Int; var mm: Int; var dd: Int
            (ok2, yy, mm, dd) = _parse_date_ymd(self.base.index_vals[r])
            if ok2:
                var ord = _days_from_civil(yy, mm, dd)
                var bucket = (ord - anchor_days) // self.step_days
                var start_ord = anchor_days + bucket * self.step_days
                var sy: Int; var sm: Int; var sd: Int
                (sy, sm, sd) = _civil_from_days(start_ord)
                var label = _ymd_to_string(sy, sm, sd)

                var pos_opt = bin_pos.get(label)
                var pos = -1
                if pos_opt is None:
                    pos = len(bin_labels)
                    bin_pos[label] = pos
                    bin_labels.append(label)
                    if not have_inited:
                        var k2 = 0
                        while k2 < len(val_cols):
                            if keep_flags[k2]:
                                acc_max.append(List[Float64]())
                                acc_cnt.append(List[Int]())
                            k2 += 1
                        have_inited = True
                    var a = 0
                    while a < len(acc_max):
                        acc_max[a].append(0.0)
                        acc_cnt[a].append(0)
                        a += 1
                else:
                    pos = pos_opt.value()

                var kk = 0
                var kept_idx = 0
                while kk < len(val_cols):
                    if keep_flags[kk]:
                        var col_idx = val_cols[kk]
                        var s = self.base.cols[col_idx].get_string(r)
                        var okv: Bool; var fv: Float64
                        (okv, fv) = _try_parse_float(s)
                        if okv:
                            var cnt = acc_cnt[kept_idx][pos]
                            if cnt == 0:
                                acc_max[kept_idx][pos] = fv
                            else:
                                if fv > acc_max[kept_idx][pos]:
                                    acc_max[kept_idx][pos] = fv
                            acc_cnt[kept_idx][pos] = cnt + 1
                        kept_idx += 1
                    kk += 1
            r += 1

        var out_cols = List[List[String]]()
        var ci = 0
        while ci < len(val_cols):
            if keep_flags[ci]:
                var arr = List[String]()
                var p = 0
                while p < len(bin_labels):
                    var cnt = acc_cnt[ci][p]
                    if cnt > 0:
                        arr.append(String(acc_max[ci][p]))
                    else:
                        arr.append(String(""))
                    p += 1
                out_cols.append(arr.copy())
                out_names.append(self.base.col_names[val_cols[ci]])
            ci += 1

        var out_df = DataFrame(out_names, out_cols, bin_labels, self.base.index_name)
        return out_df.copy()


    fn var(self, numeric_only: Bool = True, ddof: Int = 0) -> DataFrame:
        var out_names = List[String]()
        var out_data  = List[List[String]]()
        var out_index = List[String]()

        var val_cols = List[Int]()
        var c = 0
        while c < len(self.base.col_names):
            var is_idx_col = (self.base.index_name == self.base.col_names[c])
            if not is_idx_col:
                val_cols.append(c)
            c += 1

        var keep_flags = List[Bool]()
        var j = 0
        while j < len(val_cols):
            var cj = val_cols[j]
            var probe = String("")
            if self.base.cols[cj].len() > 0:
                probe = self.base.cols[cj].get_string(0)
            var okp: Bool; var tmp: Float64
            (okp, tmp) = _try_parse_float(probe)
            if numeric_only:
                keep_flags.append(okp)
            else:
                keep_flags.append(True)
            j += 1

        var bin_labels = List[String]()
        var bin_pos = Dict[String, Int]()
        var have_inited = False
        var acc_sum = List[List[Float64]]()
        var acc_sumsq = List[List[Float64]]()
        var acc_cnt = List[List[Int]]()

        var nrows = len(self.base.index_vals)
        var anchor_days = 0
        var found_anchor = False
        var r0 = 0
        while r0 < nrows and not found_anchor:
            var ok: Bool; var y: Int; var m: Int; var d: Int
            (ok, y, m, d) = _parse_date_ymd(self.base.index_vals[r0])
            if ok:
                anchor_days = _days_from_civil(y, m, d)
                found_anchor = True
            r0 += 1
        if not found_anchor:
            var k = 0
            while k < len(val_cols):
                if keep_flags[k]:
                    out_names.append(self.base.col_names[val_cols[k]])
                    out_data.append(List[String]())
                k += 1
            var out_df_empty = DataFrame(out_names, out_data, out_index, self.base.index_name)
            return out_df_empty.copy()

        var r = 0
        while r < nrows:
            var ok2: Bool; var yy: Int; var mm: Int; var dd: Int
            (ok2, yy, mm, dd) = _parse_date_ymd(self.base.index_vals[r])
            if ok2:
                var ord = _days_from_civil(yy, mm, dd)
                var bucket = (ord - anchor_days) // self.step_days
                var start_ord = anchor_days + bucket * self.step_days
                var sy: Int; var sm: Int; var sd: Int
                (sy, sm, sd) = _civil_from_days(start_ord)
                var label = _ymd_to_string(sy, sm, sd)

                var pos_opt = bin_pos.get(label)
                var pos = -1
                if pos_opt is None:
                    pos = len(bin_labels)
                    bin_pos[label] = pos
                    bin_labels.append(label)
                    if not have_inited:
                        var k2 = 0
                        while k2 < len(val_cols):
                            if keep_flags[k2]:
                                acc_sum.append(List[Float64]())
                                acc_sumsq.append(List[Float64]())
                                acc_cnt.append(List[Int]())
                            k2 += 1
                        have_inited = True
                    var a = 0
                    while a < len(acc_sum):
                        acc_sum[a].append(0.0)
                        acc_sumsq[a].append(0.0)
                        acc_cnt[a].append(0)
                        a += 1
                else:
                    pos = pos_opt.value()

                var kk = 0
                var kept_idx = 0
                while kk < len(val_cols):
                    if keep_flags[kk]:
                        var col_idx = val_cols[kk]
                        var s = self.base.cols[col_idx].get_string(r)
                        var okv: Bool; var fv: Float64
                        (okv, fv) = _try_parse_float(s)
                        if okv:
                            acc_sum[kept_idx][pos] = acc_sum[kept_idx][pos] + fv
                            acc_sumsq[kept_idx][pos] = acc_sumsq[kept_idx][pos] + fv * fv
                            acc_cnt[kept_idx][pos] = acc_cnt[kept_idx][pos] + 1
                        kept_idx += 1
                    kk += 1
            r += 1

        var out_cols = List[List[String]]()
        var ci = 0
        while ci < len(val_cols):
            if keep_flags[ci]:
                var arr = List[String]()
                var p = 0
                while p < len(bin_labels):
                    var n = acc_cnt[ci][p]
                    if n > ddof and n > 0:
                        var sumv = acc_sum[ci][p]
                        var sumsq = acc_sumsq[ci][p]
                        var nf = Float64(n)
                        var num = sumsq - (sumv * sumv / nf)
                        var denom = Float64(n - ddof)
                        arr.append(String(num / denom))
                    else:
                        arr.append(String(""))
                    p += 1
                out_cols.append(arr.copy())
                out_names.append(self.base.col_names[val_cols[ci]])
            ci += 1

        var out_df = DataFrame(out_names, out_cols, bin_labels, self.base.index_name)
        return out_df.copy()


    fn std(self, numeric_only: Bool = True, ddof: Int = 0) -> DataFrame:
        var v = self.var(numeric_only, ddof)
        # take sqrt of each numeric cell (string-backed)
        var out = v.copy()
        var c = 0
        while c < len(out.cols):
            var r = 0
            while r < out.cols[c].len():
                var s = out.cols[c].get_string(r)
                if len(s) > 0:
                    var val = atof(s)
                    out.cols[c].set_string(r, String(_sqrt64(val)))
                r += 1
            c += 1
        return out.copy()

    