from collections.list import List
from collections.dict import Dict
from pathlib.path import Path
from momijo.dataframe.io_bytes import string_to_bytes, u8_to_char
from momijo.dataframe.frame import DataFrame
# از df_from_pairs استفاده نمی‌کنیم تا به ColPair وابسته نشویم

# --- Internal helpers ---
@always_inline
fn _first_byte_or(s: String, fallback: UInt8) -> UInt8:
    var bs = string_to_bytes(s)
    if len(bs) == 0:
        return fallback
    return bs[0]

fn _split_lines(s: String) -> List[String]:
    var out = List[String]()
    var cur = String("")
    var bs = string_to_bytes(s)
    var i = 0
    var n = len(bs)
    while i < n:
        var ch = bs[i]
        if ch == UInt8(13):
            i += 1
            continue
        if ch == UInt8(10):
            out.append(cur.copy())
            cur = String("")
            i += 1
            continue
        cur = cur + u8_to_char(ch)
        i += 1
    out.append(cur.copy())
    return out.copy()

fn _parse_csv_line(line: String, delim: UInt8, quote: UInt8) -> List[String]:
    var out = List[String]()
    var buf = String("")
    var inq = False

    var bs = string_to_bytes(line)
    var i = 0
    var n = len(bs)

    while i < n:
        var ch = bs[i]
        if inq:
            if ch == quote:
                if i + 1 < n and bs[i + 1] == quote:
                    buf = buf + u8_to_char(quote)
                    i += 2
                    continue
                inq = False
                i += 1
                continue
            buf = buf + u8_to_char(ch)
            i += 1
            continue
        else:
            if ch == quote:
                inq = True
                i += 1
                continue
            if ch == delim:
                out.append(buf.copy())
                buf = String("")
                i += 1
                continue
            if ch == UInt8(13) or ch == UInt8(10):
                i += 1
                continue
            buf = buf + u8_to_char(ch)
            i += 1
            continue

    out.append(buf.copy())
    return out.copy()

fn _string(v: Int) -> String:
    return String(v)

# ---------------- CSV: parse from in-memory string into DataFrame ----------------

fn read_csv_from_string(
    text: String,
    header: Bool = True,
    delimiter: String = String(","),
    quotechar: String = String("\"")
) -> DataFrame:
    var lines = _split_lines(text)
    var d = _first_byte_or(delimiter, UInt8(44))
    var q = _first_byte_or(quotechar, UInt8(34))

    var rows = List[List[String]]()
    var i = 0
    while i < len(lines):
        var parts = _parse_csv_line(lines[i], d, q)

        var is_empty = True
        var k = 0
        while k < len(parts):
            if len(parts[k]) > 0:
                is_empty = False
                break
            k += 1

        if not (is_empty and i == len(lines) - 1):
            rows.append(parts.copy())
        i += 1

    if len(rows) == 0:
        return DataFrame(List[String](), List[List[String]](), List[String](), String(""))

    var headers = List[String]()
    var start_row = 0
    var width = 0

    if header:
        var first = rows[0].copy()
        width = len(first)
        var j = 0
        while j < width:
            var name = first[j]
            if len(name) == 0:
                name = String("col") + String(String(j))
            headers.append(name.copy())
            j += 1
        start_row = 1
    else:
        width = len(rows[0])
        var j2 = 0
        while j2 < width:
            var gen = String("col_") + _string(j2)
            headers.append(gen.copy())
            j2 += 1

    var cols = List[List[String]]()
    var c = 0
    while c < width:
        cols.append(List[String]())
        c += 1

    var r = start_row
    while r < len(rows):
        var row = rows[r].copy()
        var c2 = 0
        while c2 < width:
            var cell = String("")
            if c2 < len(row):
                cell = row[c2]
            cols[c2].append(cell.copy())
            c2 += 1
        r += 1

    var index = List[String]()
    var nrows = 0
    if width > 0:
        nrows = len(cols[0])
    var rr = 0
    while rr < nrows:
        var idxs = _string(rr)
        index.append(idxs.copy())
        rr += 1

    return DataFrame(headers, cols, index, String(""))

# ---------------- CSV: to-string writer ----------------

fn to_csv_string(
    df: DataFrame,
    header: Bool = True,
    delimiter: String = String(","),
    quotechar: String = String("\"")
) -> String:
    var d = _first_byte_or(delimiter, UInt8(44))
    var q = _first_byte_or(quotechar, UInt8(34))

    fn needs_quote(cell: String, d: UInt8, q: UInt8) -> Bool:
        var bs = string_to_bytes(cell)
        var i = 0
        while i < len(bs):
            var ch = bs[i]
            if ch == d or ch == q or ch == UInt8(10) or ch == UInt8(13):
                return True
            i += 1
        if len(bs) > 0 and (bs[0] == UInt8(32) or bs[len(bs) - 1] == UInt8(32)):
            return True
        return False

    fn write_cell(cell: String, d: UInt8, q: UInt8) -> String:
        if needs_quote(cell, d, q):
            var out = String("")
            out = out + u8_to_char(q)
            var bs = string_to_bytes(cell)
            var i = 0
            while i < len(bs):
                var ch = bs[i]
                if ch == q:
                    out = out + u8_to_char(q) + u8_to_char(q)
                else:
                    out = out + u8_to_char(ch)
                i += 1
            out = out + u8_to_char(q)
            return out
        return cell

    var out = String("")
    var ncols = df.ncols()

    if header:
        var c = 0
        while c < ncols:
            out = out + write_cell(df.col_names[c], d, q)
            if c + 1 < ncols:
                out = out + String(delimiter)
            c += 1
        out = out + String("\n")

    var nrows = df.nrows()
    var r = 0
    while r < nrows:
        var c2 = 0
        while c2 < ncols:
            out = out + write_cell(df.cols[c2][r], d, q)
            if c2 + 1 < ncols:
                out = out + String(delimiter)
            c2 += 1
        out = out + String("\n")
        r += 1

    return out

# ---------------- column selection (build List[List[String]]) & dtypes (noop) ----------------

fn _select_columns(df: DataFrame, cols: List[String]) -> DataFrame:
    if len(cols) == 0:
        return df

    var name_to_idx = Dict[String, Int]()
    var i = 0
    while i < len(df.col_names):
        name_to_idx[df.col_names[i]] = i
        i += 1

    var new_names = List[String]()
    var new_cols = List[List[String]]()

    var k = 0
    while k < len(cols):
        var opt_idx = name_to_idx.get(cols[k])
        if opt_idx is not None:
            var idx = opt_idx.value()

            var vals = List[String]()
            var r = 0
            var nrows = df.nrows()
            while r < nrows:
                var v = String(df.cols[idx][r])
                vals.append(v.copy())
                r += 1

            new_names.append(df.col_names[idx].copy())
            new_cols.append(vals.copy())
        k += 1

    var nrows2 = 0
    if len(new_cols) > 0:
        nrows2 = len(new_cols[0])

    var new_index = List[String]()
    var r2 = 0
    while r2 < nrows2:
        var s = _string(r2)
        new_index.append(s.copy())
        r2 += 1

    return DataFrame(new_names, new_cols, new_index, String(""))

fn _apply_dtypes(df: DataFrame, dtypes: Dict[String, String]) -> DataFrame:
    if len(dtypes) == 0:
        return df
    return df  # placeholder

# ---------------- public API ----------------

fn read_csv_string(
    text: String,
    header: Bool = True,
    delimiter: String = String(","),
    quotechar: String = String("\""),
    usecols: List[String] = List[String](),
    dtypes: Dict[String, String] = Dict[String, String]()
) -> DataFrame:
    var base = read_csv_from_string(text, header, delimiter, quotechar)
    if len(usecols) > 0:
        base = _select_columns(base, usecols)
    if len(dtypes) > 0:
        base = _apply_dtypes(base, dtypes)
    return base

fn read_csv(
    path: Path,
    header: Bool = True,
    delimiter: String = String(","),
    quotechar: String = String("\"")
) raises -> DataFrame:
    var f = open(String(path), "r")
    var s = f.read()
    f.close()
    return read_csv_string(s, header, delimiter, quotechar)

fn write_csv(
    df: DataFrame,
    path: Path,
    header: Bool = True,
    delimiter: String = String(","),
    quotechar: String = String("\"")
) -> Bool:
    var s = to_csv_string(df, header, delimiter, quotechar)
    try:
        var f = open(String(path), "w")
        _ = f.write(s)
        f.close()
        return True
    except _:
        return False
