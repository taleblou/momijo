# Project:      Momijo 
# Module:       dataframe.selection
# File:         selection.mojo
# Path:         dataframe/selection.mojo
#
# Description:  dataframe.selection — Row/column selection (loc/iloc/select), masks & filters.
#               Helpers for building Column/Mask, and vectorized predicates.
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
#   - Structs: Mask, RowRange, ColRange (assumed pre-defined in this module or imports).
#   - Key functions: select, loc, iloc, filter, where, mask_and, col_ge, col_isin.
#   - Static methods present: N/A.

from momijo.dataframe.frame import DataFrame as DataFrame
from momijo.dataframe.column import *
from momijo.dataframe.series_str import SeriesStr
from momijo.dataframe.io_bytes import str_to_bytes


struct RowRange(Copyable, Movable):
    var start: Int
    var stop: Int
    fn __init__(out self, start: Int, stop: Int):
        self.start = start
        self.stop = stop
    fn __init__(out self):
        self.start = 0
        self.stop = -1

struct LabelSlice(Copyable, Movable):
    var start: String
    var end: String
    var inclusive: Bool

    fn __init__(out self):
        self.start = String("")
        self.end = String("")
        self.inclusive = True

struct ColRange:
    var start: Int
    var stop: Int
    fn __init__(out self, start: Int, stop: Int):
        self.start = start
        self.stop = stop

# Copy basic index metadata from src into out_df (out parameter)
fn _copy_meta(src: DataFrame, out out_df: DataFrame) -> None:
    # copy index name
    out_df.index_name = String(src.index_name)

    # deep-copy index values to avoid aliasing
    out_df.index_vals = List[String]()
    var i = 0
    var n = len(src.index_vals)
    while i < n:
        out_df.index_vals.append(String(src.index_vals[i]))
        i += 1
    return

struct ILocRowSlice:
    var start: Int
    var stop: Int
    fn __init__(out self, start: Int, stop: Int):
        self.start = start
        self.stop  = stop

struct ILocColSlice:
    var start: Int
    var stop: Int
    fn __init__(out self, start: Int, stop: Int):
        self.start = start
        self.stop  = stop

# Public facades
fn slice_rows(start: Int, stop: Int) -> ILocRowSlice:
    return ILocRowSlice(start, stop)

fn slice_cols(start: Int, stop: Int) -> ILocColSlice:
    return ILocColSlice(start, stop)

# Build a Column from a list of strings with a name
# Uses SeriesStr + Column.from_str(...) path that exists in your codebase.
# Select columns by names (keeps the given order; skips missing names)
# Select columns by names (keeps the given order; skips missing names)
fn select(df: DataFrame, cols: List[String]) -> DataFrame:
    var out = DataFrame()

    # ---- Copy index metadata (deep copies) ----
    out.index_name = String(df.index_name)
    out.index_vals = List[String]()
    var ii = 0
    var nidx = len(df.index_vals)
    while ii < nidx:
        out.index_vals.append(String(df.index_vals[ii]))
        ii += 1

    # ---- Prepare output column containers ----
    out.col_names = List[String]()
    out.names     = List[String]()
    out.cols      = List[Column]()

    # Nothing requested → return a shallow-structure copy carrying only the index
    var nreq = len(cols)
    if nreq == 0:
        return out.copy()

    # ---- Select requested columns by name (preserve request order; skip misses) ----
    var nsrc = len(df.col_names)

    # Small fast-path: if requested list exactly matches the source order,
    # we can avoid the inner name scan (still deep-copy columns).
    var same_order = (nreq == nsrc)
    if same_order:
        var k = 0
        while k < nsrc:
            if df.col_names[k] != cols[k]:
                same_order = False
                break
            k += 1
    if same_order:
        var j = 0
        while j < nsrc:
            out.col_names.append(String(df.col_names[j]))
            out.names.append(String(df.col_names[j]))
            out.cols.append(df.cols[j].copy())   # Column expected Copyable; replace with a real clone if needed
            j += 1
        return out.copy()

    # General path: for each requested name, find first match in source and append
    var i = 0
    while i < nreq:
        var name = cols[i]

        # Linear search over source column names
        var j = 0
        while j < nsrc and df.col_names[j] != name:
            j += 1

        # If found, append to output (names arrays kept in sync)
        if j < nsrc:
            var src_name = String(df.col_names[j])
            out.col_names.append(src_name)
            out.names.append(src_name)
            out.cols.append(df.cols[j].copy())   # ensure deep copy of column payload
        # If not found → silently skip this requested name
        i += 1

    return out.copy()



# loc: slice rows by inclusive RowRange, and columns by names
# - rows is inclusive: [start, stop], both ends included when valid
# - columns selected by exact names, order preserved, missing names are skipped
fn loc(df: DataFrame, rows: RowRange, cols: List[String]) -> DataFrame:
    # 1) First pick the requested columns (keeps order, skips missing)
    var sub = select(df, cols)

    # 2) Determine nrows from selected df (prefer column length; fallback to index size)
    var nrows = 0
    if len(sub.cols) > 0:
        nrows = sub.cols[0].len()
    else:
        nrows = len(sub.index_vals)

    # 3) Clamp inclusive [start, stop] to valid row range
    var start = rows.start
    var stop  = rows.stop
    if start < 0:
        start = 0
    if stop < 0:
        stop = -1               # forces empty if negative
    if stop >= nrows:
        stop = nrows - 1
    if start > stop or nrows == 0:
        # return empty slice with same columns metadata, zero rows
        var empty = DataFrame()
        empty.index_name = String(sub.index_name)
        empty.index_vals = List[String]()      # no rows
        empty.col_names  = List[String]()
        empty.names      = List[String]()
        empty.cols       = List[Column]()

        var c = 0
        var cn = len(sub.col_names)
        while c < cn:
            empty.col_names.append(String(sub.col_names[c]))
            empty.names.append(String(sub.col_names[c]))
            # build empty column with same name
            var vals = List[String]()
            empty.cols.append(col_from_list(vals, sub.col_names[c]))
            c += 1
        return empty

    # 4) Build output DataFrame with sliced rows
    var out = DataFrame()

    # index meta (name + sliced values)
    out.index_name = String(sub.index_name)
    out.index_vals = List[String]()
    var r = start
    while r <= stop:
        if r < len(sub.index_vals):
            out.index_vals.append(String(sub.index_vals[r]))
        else:
            # If index_vals is shorter/missing, synthesize as empty string
            out.index_vals.append(String(""))
        r += 1

    # columns meta
    out.col_names = List[String]()
    out.names     = List[String]()
    out.cols      = List[Column]()

    var c = 0
    var cn = len(sub.col_names)
    while c < cn:
        out.col_names.append(String(sub.col_names[c]))
        out.names.append(String(sub.col_names[c]))

        # gather string cell values for rows [start..stop]
        var vals = List[String]()
        var rr = start
        while rr <= stop:
            vals.append(String(sub.cols[c].get_string(rr)))
            rr += 1

        out.cols.append(col_from_list(vals, sub.col_names[c]))
        c += 1

    return out


# range rows + named columns → DataFrame
fn loc_impl_range(df: DataFrame, rows: RowRange, cols: List[String]) -> DataFrame:
    var sub = select(df, cols)

    var nrows = 0
    if len(sub.cols) > 0:
        nrows = sub.cols[0].len()
    else:
        nrows = len(sub.index_vals)

    var start = rows.start
    var stop  = rows.stop
    if start < 0:
        start = 0
    if stop < 0:
        stop = -1
    if stop >= nrows:
        stop = nrows - 1

    if start > stop or nrows == 0:
        var empty = DataFrame()
        empty.index_name = String(sub.index_name)
        empty.index_vals = List[String]()
        empty.col_names  = List[String]()
        empty.names      = List[String]()
        empty.cols       = List[Column]()
        var c = 0
        var cn = len(sub.col_names)
        while c < cn:
            empty.col_names.append(String(sub.col_names[c]))
            empty.names.append(String(sub.col_names[c]))
            var vals = List[String]()
            empty.cols.append(col_from_list(vals, sub.col_names[c]))
            c += 1
        return empty

    var out = DataFrame()
    out.index_name = String(sub.index_name)
    out.index_vals = List[String]()
    var r = start
    while r <= stop:
        if r < len(sub.index_vals):
            out.index_vals.append(String(sub.index_vals[r]))
        else:
            out.index_vals.append(String(""))
        r += 1

    out.col_names = List[String]()
    out.names     = List[String]()
    out.cols      = List[Column]()

    var c = 0
    var cn = len(sub.col_names)
    while c < cn:
        out.col_names.append(String(sub.col_names[c]))
        out.names.append(String(sub.col_names[c]))
        var vals = List[String]()
        var rr = start
        while rr <= stop:
            vals.append(String(sub.cols[c].get_string(rr)))
            rr += 1
        out.cols.append(col_from_list(vals, sub.col_names[c]))
        c += 1

    return out

# arbitrary row indices + named columns → DataFrame
fn loc_impl_indices(df: DataFrame, row_idxs: List[Int], cols: List[String]) -> DataFrame:
    var sub = select(df, cols)

    var out = DataFrame()
    out.index_name = String(sub.index_name)
    out.index_vals = List[String]()
    var i = 0
    var k = len(row_idxs)
    while i < k:
        var r = row_idxs[i]
        if r >= 0 and r < len(sub.index_vals):
            out.index_vals.append(String(sub.index_vals[r]))
        else:
            out.index_vals.append(String(""))
        i += 1

    out.col_names = List[String]()
    out.names     = List[String]()
    out.cols      = List[Column]()

    var c = 0
    var cn = len(sub.col_names)
    while c < cn:
        out.col_names.append(String(sub.col_names[c]))
        out.names.append(String(sub.col_names[c]))
        var vals = List[String]()
        i = 0
        while i < k:
            var r = row_idxs[i]
            vals.append(String(sub.cols[c].get_string(r)))
            i += 1
        out.cols.append(col_from_list(vals, sub.col_names[c]))
        c += 1

    return out
 

# boolean mask to filter rows
struct Mask:
    var vals: List[Bool]
    fn __init__(out self):
        self.vals = List[Bool]()
    fn __copyinit__(out self, other: Mask):
        self.vals = List[Bool]()
        var i = 0
        while i < len(other.vals):
            self.vals.append(other.vals[i])
            i += 1
# Logical AND of two masks (length = min(len(a), len(b)))
fn mask_and(a: Mask, b: Mask) -> Mask:
    var out = Mask()
    out.vals = List[Bool]()
    var n = len(a.vals)
    var m = len(b.vals)
    var L = n
    if m < L:
        L = m
    var i = 0
    while i < L:
        out.vals.append(a.vals[i] and b.vals[i])
        i += 1
    return out


# Filter rows of a DataFrame using a boolean mask.
# Keeps rows where mask[i] == true, in order; extra mask entries are ignored.
fn filter_rows(df: DataFrame, mask: List[Bool]) -> DataFrame:
    # Determine number of rows from first column (fallback to index length)
    var nrows = 0
    if len(df.cols) > 0:
        nrows = df.cols[0].len()
    else:
        nrows = len(df.index_vals)

    # Build row indices to keep (guard against mask shorter than nrows)
    var keep = List[Int]()
    var r = 0
    var mlen = len(mask)
    while r < nrows and r < mlen:
        if mask[r]:
            keep.append(r)
        r += 1

    # Prepare output frame
    var out = DataFrame()
    out.col_names = List[String]()
    out.names     = List[String]()
    out.cols      = List[Column]()

    # Copy index name and slice index values with the same mask
    out.index_name = String(df.index_name)
    out.index_vals = List[String]()
    var ki = 0
    var kn = len(keep)
    while ki < kn:
        var ridx = keep[ki]
        if ridx < len(df.index_vals):
            out.index_vals.append(String(df.index_vals[ridx]))
        else:
            # If index_vals is missing/shorter, synthesize empty string
            out.index_vals.append(String(""))
        ki += 1

    # Copy/construct selected columns with filtered rows
    var c = 0
    var cn = len(df.col_names)
    while c < cn:
        var cname = String(df.col_names[c])
        out.col_names.append(cname)
        out.names.append(cname)

        # Gather string cells for kept rows
        var vals = List[String]()
        var j = 0
        while j < kn:
            var rr = keep[j]
            # Assume Column.get_string(rr) exists
            vals.append(String(df.cols[c].get_string(rr)))
            j += 1

        out.cols.append(col_from_list(vals, cname))
        c += 1

    return out

# Build mask: column >= value (numeric compare; non-parsable → False)
fn col_ge(df: DataFrame, name: String, value: Int) -> Mask:
    var out = Mask()
    out.vals = List[Bool]()

    # Resolve column index by name
    var j = 0
    var ncols = len(df.col_names)
    while j < ncols and df.col_names[j] != name:
        j += 1

    var nrows = df.nrows()

    # If column not found → all False (proper length)
    if j >= ncols:
        var r0 = 0
        while r0 < nrows:
            out.vals.append(False)
            r0 += 1
        return out

    # ASCII constants
    var ZERO:  UInt8 = UInt8(48)  # '0'
    var NINE:  UInt8 = UInt8(57)  # '9'
    var MINUS: UInt8 = UInt8(45)  # '-'
    var PLUS:  UInt8 = UInt8(43)  # '+'

    # Row-wise parse and compare
    var r = 0
    while r < nrows:
        # Read cell as string
        var s = String(df.cols[j].get_string(r))
        var bs = s.bytes()
        var n = len(bs)

        # Default = invalid
        var bad = (n == 0)
        var sign = 1
        var i = 0

        # Optional leading sign
        if (not bad) and (bs[0] == MINUS or bs[0] == PLUS):
            if n == 1:
                bad = True
            else:
                if bs[0] == MINUS:
                    sign = -1
                else:
                    sign = 1
                i = 1

        # Parse decimal integer
        var acc = 0
        while (not bad) and (i < n):
            var ch = bs[i]
            if (ch < ZERO) or (ch > NINE):
                bad = True
                break
            # acc = acc * 10 + (ch - '0')
            acc = acc * 10 + (Int(ch) - Int(ZERO))
            i += 1

        if not bad:
            acc = acc * sign
            out.vals.append(acc >= value)
        else:
            out.vals.append(False)

        r += 1

    return out

# Build mask: column string value is in a given set of strings
fn col_isin(df: DataFrame, name: String, values: List[String]) -> Mask:
    var out = Mask()
    out.vals = List[Bool]()

    # resolve column index
    var j = 0
    var ncols = len(df.col_names)
    while j < ncols and df.col_names[j] != name:
        j += 1

    var nrows = df.nrows()

    # column not found => all False with proper length
    if j >= ncols:
        var r0 = 0
        while r0 < nrows:
            out.vals.append(False)
            r0 += 1
        return out

    # membership check
    var r = 0
    while r < nrows:
        var s = String(df.cols[j].get_string(r))
        var found = False

        var k = 0
        var m = len(values)
        while k < m:
            if s == values[k]:
                found = True
                break
            k += 1

        out.vals.append(found)
        r += 1

    return out


# where: return only rows where mask is True
fn where(df: DataFrame, mask: Mask) -> DataFrame:
    # filter_rows already guards for mask/data length mismatches
    return filter_rows(df, mask.vals)
 

# iloc: select by explicit row indices and a half-open column range [start, stop)
fn clamp(v: Int, lo: Int, hi: Int) -> Int:
    var x = v
    if x < lo: x = lo
    if x > hi: x = hi
    return x

# Build a frame from arbitrary row and column index lists (no range)
fn iloc_impl_indices_cols(df: DataFrame, row_idxs: List[Int], col_idxs: List[Int]) -> DataFrame:
    var out = DataFrame()

    # index
    out.index_name = String(df.index_name)
    out.index_vals = List[String]()
    var i = 0
    var rn = len(row_idxs)
    var nrows = df.nrows()
    while i < rn:
        var r = row_idxs[i]
        if r >= 0 and r < nrows and r < len(df.index_vals):
            out.index_vals.append(String(df.index_vals[r]))
        elif r >= 0 and r < nrows:
            out.index_vals.append(String(""))
        i += 1

    # columns
    out.col_names = List[String]()
    out.names     = List[String]()
    out.cols      = List[Column]()

    var ci = 0
    var cn = len(col_idxs)
    while ci < cn:
        var c = col_idxs[ci]
        # skip invalid columns silently
        if c >= 0 and c < len(df.col_names):
            var cname = String(df.col_names[c])
            out.col_names.append(cname)
            out.names.append(cname)

            var vals = List[String]()
            i = 0
            while i < rn:
                var r = row_idxs[i]
                if r >= 0 and r < nrows:
                    vals.append(String(df.cols[c].get_string(r)))
                i += 1

            out.cols.append(col_from_list(vals, cname))
        ci += 1

    return out

# iloc: select by explicit row indices and a half-open column range [start, stop)
fn iloc(df: DataFrame, row_indices: List[Int], col_range: ColRange) -> DataFrame:
    var cstart = col_range.start
    var cstop  = col_range.stop
    if cstart < 0:
        cstart = 0
    var ncols = df.ncols()
    if cstop < 0 or cstop > ncols:
        cstop = ncols
    if cstart > cstop:
        cstart = cstop

    # build col index list [cstart..cstop)
    var col_idxs = List[Int]()
    var c = cstart
    while c < cstop:
        col_idxs.append(c)
        c += 1

    return iloc_impl_indices_cols(df, row_indices, col_idxs)



# Public alias expected by All.mojo
fn filter(df: DataFrame, mask: List[Bool]) -> DataFrame:
    return filter_rows(df, mask)


# Overload: accept Mask directly (for All.mojo compatibility)
fn filter(df: DataFrame, mask: Mask) -> DataFrame:
    return filter_rows(df, mask.vals)


struct ColumnBoolOps:
    name: String
    fn __init__(out self, name: String):
        self.name = name
    fn ge(self, value: Int) -> Mask: return Mask()
    fn isin(self, values: List[String]) -> Mask: return Mask()
    fn between(self, lo: Int, hi: Int) -> Mask: return Mask()

 
 

# -----------------------------------------------------------------------------
# Label-based row slicing token + helpers
# -----------------------------------------------------------------------------
# If you already defined this elsewhere, keep your original.

fn slice_labels(start: String, end: String, inclusive: Bool = True) -> LabelSlice:
    var s = LabelSlice()
    s.start = start
    s.end = end
    s.inclusive = inclusive
    return s.copy()

fn find_first(labels: List[String], target: String) -> Int:
    var i = 0
    var n = len(labels)
    while i < n:
        if labels[i] == target:
            return i
        i += 1
    return -1

fn labels_to_row_range(index_vals: List[String], sel: LabelSlice) -> RowRange:
    # FIX: RowRange needs (start, stop) because of your __init__ signature
    var rr = RowRange(0, -1)

    var a = find_first(index_vals, sel.start)
    var b = find_first(index_vals, sel.end)

    # Missing labels -> empty
    if a < 0 or b < 0:
        return rr.copy()

    # Normalize order
    var lo = a if a <= b else b
    var hi = b if a <= b else a

    if sel.inclusive:
        rr.start = lo
        rr.stop  = hi
    else:
        rr.start = lo
        rr.stop  = hi - 1

    # If exclusive made it invalid, encode empty
    if rr.stop < rr.start:
        rr.start = 0
        rr.stop = -1

    return rr.copy()