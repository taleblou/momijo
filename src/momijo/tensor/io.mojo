# MIT License
# SPDX-License-Identifier: MIT
# Project: momijo.tensor
# File: src/momijo/tensor/io.mojo
# Description: Text/structured I/O helpers for Tensor. Pure Mojo parsers for
#              CSV and minimal JSON/XML, plus string-based codecs. Path-based
#              wrappers use stdlib file I/O (io.file.open) with safe try/except.

from momijo.tensor.tensor import Tensor
from io.file import open

# =============================================================================
# Notes & Scope
# =============================================================================
# - All comments are in English (project rule).
# - No wildcards in imports; var-only; 4-space indent; no globals.
# - CSV: simple RFC4180-like subset, quotes + doubled quotes supported.
# - JSON: minimal numeric arrays (1D / 2D).
# - XML: minimal numeric rows/values tags.
# - We avoid raising in public APIs; file access is wrapped in try/except.

# =============================================================================
# Tiny utilities
# =============================================================================
@always_inline 
fn _is_space_ch(ch: String) -> Bool:
    # Single-char String whitespace check
    return ch == " " or ch == "\t" or ch == "\n" or ch == "\r"

@always_inline
fn _trim_ascii(s: String) -> String:
    # Use StringSlice and compare single-char slices as Strings
    var sl = s.as_string_slice()
    var n = len(sl)

    var i = 0
    while i < n and _is_space_ch(String(sl[i:i+1])):
        i += 1

    var j = n
    while j > i and _is_space_ch(String(sl[j-1:j])):
        j -= 1

    return String(sl[i:j])


# ---- helpers for char access as String ---- 

@always_inline
fn _ch(sl: StringSlice, i: Int) -> String:
    return String(sl[i:i+1])

@always_inline
fn _is_plus(ch: String) -> Bool:  return ch == "+"
@always_inline
fn _is_minus(ch: String) -> Bool: return ch == "-"
@always_inline
fn _is_dot(ch: String) -> Bool:   return ch == "."
@always_inline
fn _is_e(ch: String) -> Bool:     return ch == "e" or ch == "E"

@always_inline
fn _is_digit(ch: String) -> Bool:
    # lexicographic compare works for ASCII digits on 1-char strings
    return ch >= "0" and ch <= "9"

@always_inline
fn _digit_of(ch: String) -> (Bool, Int):
    if ch == "0": return (True, 0)
    if ch == "1": return (True, 1)
    if ch == "2": return (True, 2)
    if ch == "3": return (True, 3)
    if ch == "4": return (True, 4)
    if ch == "5": return (True, 5)
    if ch == "6": return (True, 6)
    if ch == "7": return (True, 7)
    if ch == "8": return (True, 8)
    if ch == "9": return (True, 9)
    return (False, 0)


# ---- REPLACE _parse_f64 with this version ----
@always_inline
fn _parse_f64(s: String) -> (Bool, Float64):
    # Very small numeric parser supporting: sign, integer, fraction, exponent.
    var t = _trim_ascii(s)
    var tl = t.as_string_slice()
    var m = len(tl)
    if m == 0:
        return (False, 0.0)

    var k = 0
    var sign = 1.0
    if _is_plus(_ch(tl, k)):
        k += 1
    elif _is_minus(_ch(tl, k)):
        sign = -1.0
        k += 1

    var int_part = 0.0
    var has_int = False
    while k < m and _is_digit(_ch(tl, k)):
        var (okd, dv) = _digit_of(_ch(tl, k))
        if not okd:
            break
        int_part = int_part * 10.0 + Float64(dv)
        has_int = True
        k += 1

    var frac_part = 0.0
    var frac_scale = 1.0
    if k < m and _is_dot(_ch(tl, k)):
        k += 1
        var any_frac = False
        while k < m and _is_digit(_ch(tl, k)):
            var (okd2, dv2) = _digit_of(_ch(tl, k))
            if not okd2:
                break
            frac_part = frac_part * 10.0 + Float64(dv2)
            frac_scale = frac_scale * 10.0
            any_frac = True
            k += 1
        if not any_frac and not has_int:
            return (False, 0.0)

    var e = 0
    var has_exp = False
    if k < m and _is_e(_ch(tl, k)):
        k += 1
        has_exp = True
        var esign = 1
        if k < m and (_is_plus(_ch(tl, k)) or _is_minus(_ch(tl, k))):
            if _is_minus(_ch(tl, k)):
                esign = -1
            k += 1
        var any_exp = False
        while k < m and _is_digit(_ch(tl, k)):
            var (okd3, dv3) = _digit_of(_ch(tl, k))
            if not okd3:
                break
            e = e * 10 + dv3
            any_exp = True
            k += 1
        if not any_exp:
            return (False, 0.0)
        e = e * esign

    var val = int_part + frac_part / frac_scale
    var pow10 = 1.0
    if has_exp:
        if e < 0:
            var i2 = 0
            while i2 < -e:
                pow10 = pow10 / 10.0
                i2 += 1
        else:
            var i3 = 0
            while i3 < e:
                pow10 = pow10 * 10.0
                i3 += 1

    return (True, sign * val * pow10)



@always_inline
fn _to_string_f64(x: Float64) -> String:
    # Basic dtoa; integer fast-path then 6 decimals.
    var s = String("")
    var xi = Int(x)
    if x == Float64(xi):
        s = s + xi.__str__()
        return s
    var frac = x
    var sign = ""
    if frac < 0.0:
        sign = "-"
        frac = -frac
    var whole = Int(frac)
    var rest = frac - Float64(whole)
    s = s + sign + whole.__str__() + "."
    var i = 0
    while i < 6:
        rest = rest * 10.0
        var d = Int(rest)
        s = s + d.__str__()
        rest = rest - Float64(d)
        i += 1
    return s

# =============================================================================
# Core CSV codec (string-based)
# =============================================================================
struct CsvOptions(Copyable, Movable):
    var delimiter: String
    var has_header: Bool

    fn __init__(out self, delimiter: String = ",", has_header: Bool = False):
        self.delimiter = delimiter
        self.has_header = has_header

@always_inline
fn _csv_split_line(line: String, delim: String) -> List[String]:
    # Splits a CSV line handling quotes and doubled quotes.
    var out = List[String]()
    var ls = line.as_string_slice()
    var ds = delim.as_string_slice()
    var n = len(ls)
    var i = 0
    var cur = String("")
    var in_q = False
    # compare single-char Strings (not UInt8)
    var q = String("\"")           # quote as 1-char String
    var d = String(ds[0:1])        # delimiter as 1-char String

    while i < n:
        var c = String(ls[i:i+1])  # current char as 1-char String
        if in_q:
            if c == q:
                var next_i = i + 1
                if next_i < n and String(ls[next_i:next_i+1]) == q:
                    # doubled quote -> append one quote
                    cur = cur + "\""
                    i = i + 2
                    continue
                in_q = False
                i += 1
                continue
            # append current char
            cur = cur + c
            i += 1
        else:
            if c == q:
                in_q = True
                i += 1
            elif c == d:
                out.append(cur.copy())
                cur = String("")
                i += 1
            else:
                cur = cur + c
                i += 1
    out.append(cur.copy())
    return out.copy()


fn csv_to_tensor_with[T: ImplicitlyCopyable & Copyable & Movable](
    text: String,
    from_f64: fn (Float64) -> T,
    opts: CsvOptions = CsvOptions()
) -> Tensor[T]:
    # Parse CSV text (rows × cols) into a dense row-major Tensor[T].
    var lines = text.split("\n")
    var rows = List[List[String]]()

    var skip_header = opts.has_header
    var i = 0
    var nlines = len(lines)
    while i < nlines:
        var raw = _trim_ascii(String(lines[i]))
        i += 1
        if len(raw) == 0:
            continue
        if skip_header:
            skip_header = False
            continue
        var fields = _csv_split_line(raw, opts.delimiter)
        rows.append(fields.copy())

    var r = len(rows)
    if r == 0:
        var empty = List[T]()
        var shape = List[Int]()
        shape.append(0)
        return Tensor[T](empty, shape)

    var c = len(rows[0])
    var shape2 = List[Int]()
    shape2.append(r)
    shape2.append(c)

    var data = List[T]()
    data.reserve(r * c)

    var rr = 0
    while rr < r:
        var cc = 0
        var row = rows[rr].copy()
        while cc < c:
            var f = 0.0
            var ok = False
            if cc < len(row):
                 (ok, f) = _parse_f64(row[cc])
            var v = from_f64(f)
            data.append(v)
            cc += 1
        rr += 1

    return Tensor[T](data, shape2)

@always_inline
fn tensor_to_csv[T: ImplicitlyCopyable & Copyable & Movable](
    x: Tensor[T],
    to_f64: fn (T) -> Float64,
    delim: String = ",",
    header: Optional[List[String]] = None
) -> String:
    var s = String("")
    if not (header is None):
        var h = header.value().copy()
        var j0 = 0
        var m0 = len(h)
        while j0 < m0:
            if j0 > 0:
                s = s + delim
            s = s + h[j0]
            j0 += 1
        s = s + "\n"

    var ndim = len(x._shape)
    if ndim == 1:
        var n = x._shape[0]
        var i = 0
        while i < n:
            if i > 0:
                s = s + delim
            s = s + _to_string_f64(to_f64(x._data[i]))
            i += 1
        s = s + "\n"
        return s

    if ndim >= 2:
        var r = x._shape[0]
        var c = x._shape[1]
        var rs = x._strides[0]
        var cs = x._strides[1]
        var i2 = 0
        while i2 < r:
            var j2 = 0
            while j2 < c:
                if j2 > 0:
                    s = s + delim
                var idx = i2 * rs + j2 * cs + x._offset
                s = s + _to_string_f64(to_f64(x._data[idx]))
                j2 += 1
            s = s + "\n"
            i2 += 1
        return s

    # ndim == 0
    if ndim == 0:
        s = s + _to_string_f64(to_f64(x._data[0])) + "\n"
    return s

# =============================================================================
# JSON (string-based, minimal)
# =============================================================================
fn json_numbers_to_tensor_with[T: ImplicitlyCopyable & Copyable & Movable](
    text: String,
    from_f64: fn (Float64) -> T
) -> Tensor[T]:
    # Accepts either: [n0, n1, ...] or [[...], [...], ...]
    var s = text
    var on = s.find("[")
    var cn = s.rfind("]")
    if on < 0 or cn < 0 or cn <= on:
        var empty = List[T]()
        var shape = List[Int]()
        shape.append(0)
        return Tensor[T](empty, shape)

    # Body as String
    var body = String(s.as_string_slice()[on + 1:cn])

    # Detect nested arrays by looking for '['
    var nested = body.find("[") >= 0

    if not nested:
        # 1D: split by commas
        var toks = body.split(",")
        var data = List[T]()
        var n = len(toks)
        data.reserve(n)
        var i = 0
        while i < n:
            var (_ok, f) = _parse_f64(String(toks[i]))
            data.append(from_f64(f))
            i += 1
        var shape = List[Int]()
        shape.append(n)
        return Tensor[T](data, shape)

    # 2D: split top-level items by bracket depth
    var rows = List[List[Float64]]()
    var depth = 0
    var cur = String("")
    var bl = body.as_string_slice()
    var i2 = 0
    var n2 = len(bl)
    while i2 < n2:
        var chs = _ch(bl, i2)   # یک کاراکتر به‌صورت String
        if chs == "[":
            depth += 1
            if depth == 1:
                cur = String("")
            else:
                cur = cur + "["
        elif chs == "]":
            depth -= 1
            if depth == 0:
                var cells_s = cur.split(",")
                var row = List[Float64]()
                var k = 0
                var m = len(cells_s)
                while k < m:
                    var (_ok2, f2) = _parse_f64(String(cells_s[k]))
                    row.append(f2)
                    k += 1
                rows.append(row.copy())
            else:
                cur = cur + "]"
        else:
            if depth >= 1:
                # افزودن کاراکتر جاری به cur
                cur = cur + String(bl[i2:i2+1])
        i2 += 1

    var r = len(rows)
    if r == 0:
        var empty2 = List[T]()
        var shape0 = List[Int]()
        shape0.append(0)
        return Tensor[T](empty2, shape0)

    var c = len(rows[0])
    var data2 = List[T]()
    data2.reserve(r * c)
    var i3 = 0
    while i3 < r:
        var j3 = 0
        while j3 < c:
            var v = from_f64(rows[i3][j3])
            data2.append(v)
            j3 += 1
        i3 += 1
    var shape2 = List[Int]()
    shape2.append(r)
    shape2.append(c)
    return Tensor[T](data2, shape2)


@always_inline
fn tensor_to_json_array[T: ImplicitlyCopyable & Copyable & Movable](
    x: Tensor[T], to_f64: fn (T) -> Float64, pretty: Bool = True
) -> String:
    var s = String("")
    var pad = String("")
    var nl = String("")
    if pretty:
        nl = "\n"
        pad = "  "

    var ndim = len(x._shape)
    if ndim == 0:
        s = s + _to_string_f64(to_f64(x._data[0]))
        return s

    if ndim == 1:
        var n = x._shape[0]
        s = s + "[" + nl
        var i = 0
        while i < n:
            if i > 0:
                s = s + "," + nl
            s = s + pad + _to_string_f64(to_f64(x._data[i]))
            i += 1
        s = s + nl + "]"
        return s

    # 2D or higher: serialize first two dims in nested arrays (others flattened)
    var r = x._shape[0]
    var c = x._shape[1]
    var rs = x._strides[0]
    var cs = x._strides[1]
    s = s + "[" + nl
    var i2 = 0
    while i2 < r:
        if i2 > 0:
            s = s + "," + nl
        s = s + pad + "["
        var j2 = 0
        while j2 < c:
            if j2 > 0:
                s = s + ","
            var idx = i2 * rs + j2 * cs + x._offset
            s = s + _to_string_f64(to_f64(x._data[idx]))
            j2 += 1
        s = s + "]"
        i2 += 1
    s = s + nl + "]"
    return s

# =============================================================================
# XML (string-based, minimal)
# =============================================================================
fn xml_numbers_to_tensor_with[T: ImplicitlyCopyable & Copyable & Movable](
    text: String,
    from_f64: fn (Float64) -> T,
    row_tag: String = "row",
    item_tag: String = "v"
) -> Tensor[T]:
    # Expected shape:
    # <tensor>
    #   <row> <v>1</v> <v>2</v> ... </row>
    #   <row> ... </row>
    # </tensor>
    var rows = List[List[Float64]]()
    var pos = 0
    var n = len(text)
    while True:
        var start_row = text.find("<" + row_tag + ">", pos)
        if start_row < 0:
            break
        var end_row = text.find("</" + row_tag + ">", start_row)
        if end_row < 0:
            break
        var row_body = String(text.as_string_slice()[start_row + len(row_tag) + 2 : end_row])

        var vals = List[Float64]()
        var p2 = 0
        var m = len(row_body)
        while True:
            var st = row_body.find("<" + item_tag + ">", p2)
            if st < 0:
                break
            var en = row_body.find("</" + item_tag + ">", st)
            if en < 0:
                break
            var num_s = String(row_body.as_string_slice()[st + len(item_tag) + 3 : en])
             
            var (_ok, f)= _parse_f64(num_s)
            vals.append(f)
            p2 = en + len(item_tag) + 3
        rows.append(vals.copy())
        pos = end_row + len(row_tag) + 3

    var r = len(rows)
    if r == 0:
        var empty = List[T]()
        var shp = List[Int]()
        shp.append(0)
        return Tensor[T](empty, shp)

    var c = len(rows[0])
    var data = List[T]()
    data.reserve(r * c)
    var i = 0
    while i < r:
        var j = 0
        while j < c:
            data.append(from_f64(rows[i][j]))
            j += 1
        i += 1
    var shape2 = List[Int]()
    shape2.append(r)
    shape2.append(c)
    return Tensor[T](data, shape2)

@always_inline
fn tensor_to_xml_numbers[T: ImplicitlyCopyable & Copyable & Movable](
    x: Tensor[T], to_f64: fn (T) -> Float64,
    root_tag: String = "tensor", row_tag: String = "row", item_tag: String = "v"
) -> String:
    var s = String("")
    s = s + "<" + root_tag + ">\n"
    var ndim = len(x._shape)
    if ndim == 1:
        var n = x._shape[0]
        s = s + "  <" + row_tag + ">"
        var i = 0
        while i < n:
            if i > 0:
                s = s + " "
            s = s + "<" + item_tag + ">" + _to_string_f64(to_f64(x._data[i])) + "</" + item_tag + ">"
            i += 1
        s = s + "</" + row_tag + ">\n"
    else:
        var r = x._shape[0]
        var c = 1
        if len(x._shape) > 1:
            c = x._shape[1]
        var rs = 0
        var cs = 0
        if len(x._strides) > 0:
            rs = x._strides[0]
        if len(x._strides) > 1:
            cs = x._strides[1]
        var i2 = 0
        while i2 < r:
            s = s + "  <" + row_tag + ">"
            var j2 = 0
            while j2 < c:
                if j2 > 0:
                    s = s + " "
                var idx = i2 * rs + j2 * cs + x._offset
                s = s + "<" + item_tag + ">" + _to_string_f64(to_f64(x._data[idx])) + "</" + item_tag + ">"
                j2 += 1
            s = s + "</" + row_tag + ">\n"
            i2 += 1
    s = s + "</" + root_tag + ">\n"
    return s

# =============================================================================
# Path-based wrappers (use stdlib file I/O). Wrapped with try/except (no raises).
# =============================================================================
@always_inline
fn _read_text_file(path: String) -> String:
    try:
        var f = open(path, "r")
        return f.read()
    except e:
        return String("")

@always_inline
fn _write_text_file(path: String, content: String) -> Bool:
    try:
        var f = open(path, "w")
        f.write(content)
        f.close()
        return True
    except e:
        return False

@always_inline
fn read_csv_with[T: ImplicitlyCopyable & Copyable & Movable](
    path: String, from_f64: fn (Float64) -> T, opts: CsvOptions = CsvOptions()
) -> Tensor[T]:
    var txt = _read_text_file(path)
    return csv_to_tensor_with[T](txt, from_f64, opts)

@always_inline
fn write_csv[T: ImplicitlyCopyable & Copyable & Movable](
    x: Tensor[T], to_f64: fn (T) -> Float64, path: String,
    delim: String = ",", header: Optional[List[String]] = None
) -> Bool:
    var txt = tensor_to_csv[T](x, to_f64, delim, header)
    return _write_text_file(path, txt)

@always_inline
fn read_json_numbers_with[T: ImplicitlyCopyable & Copyable & Movable](
    path: String, from_f64: fn (Float64) -> T
) -> Tensor[T]:
    var txt = _read_text_file(path)
    return json_numbers_to_tensor_with[T](txt, from_f64)

@always_inline
fn write_json_numbers[T: ImplicitlyCopyable & Copyable & Movable](
    x: Tensor[T], to_f64: fn (T) -> Float64, path: String, pretty: Bool = True
) -> Bool:
    var txt = tensor_to_json_array[T](x, to_f64, pretty)
    return _write_text_file(path, txt)

@always_inline
fn read_xml_numbers_with[T: ImplicitlyCopyable & Copyable & Movable](
    path: String, from_f64: fn (Float64) -> T, row_tag: String = "row", item_tag: String = "v"
) -> Tensor[T]:
    var txt = _read_text_file(path)
    return xml_numbers_to_tensor_with[T](txt, from_f64, row_tag, item_tag)

@always_inline
fn write_xml_numbers[T: ImplicitlyCopyable & Copyable & Movable](
    x: Tensor[T], to_f64: fn (T) -> Float64, path: String,
    root_tag: String = "tensor", row_tag: String = "row", item_tag: String = "v"
) -> Bool:
    var txt = tensor_to_xml_numbers[T](x, to_f64, root_tag, row_tag, item_tag)
    return _write_text_file(path, txt)

# =============================================================================
# Small converter helpers (avoid inline fn literals in call sites)
# =============================================================================
@always_inline
fn _id_f64(x: Float64) -> Float64:
    return x

@always_inline
fn _f64_to_f32(x: Float64) -> Float32:
    return Float32(x)

@always_inline
fn _f32_to_f64(x: Float32) -> Float64:
    return Float64(x)

@always_inline
fn _f64_to_int(x: Float64) -> Int:
    return Int(x)

@always_inline
fn _int_to_f64(x: Int) -> Float64:
    return Float64(x)

# =============================================================================
# Convenience overloads for Float64 / Float32 / Int (common cases)
# =============================================================================
# --- Float64 ---
@always_inline
fn read_csv_f64(path: String, opts: CsvOptions = CsvOptions()) -> Tensor[Float64]:
    return read_csv_with[Float64](path, _id_f64, opts)

@always_inline
fn write_csv_f64(
    x: Tensor[Float64], path: String, delim: String = ",",
    header: Optional[List[String]] = None
) -> Bool:
    return write_csv[Float64](x, _id_f64, path, delim, header)

@always_inline
fn read_json_f64(path: String) -> Tensor[Float64]:
    return read_json_numbers_with[Float64](path, _id_f64)

@always_inline
fn write_json_f64(x: Tensor[Float64], path: String, pretty: Bool = True) -> Bool:
    return write_json_numbers[Float64](x, _id_f64, path, pretty)

@always_inline
fn read_xml_f64(path: String, row_tag: String = "row", item_tag: String = "v") -> Tensor[Float64]:
    return read_xml_numbers_with[Float64](path, _id_f64, row_tag, item_tag)

@always_inline
fn write_xml_f64(
    x: Tensor[Float64], path: String,
    root_tag: String = "tensor", row_tag: String = "row", item_tag: String = "v"
) -> Bool:
    return write_xml_numbers[Float64](x, _id_f64, path, root_tag, row_tag, item_tag)

# --- Float32 ---
@always_inline
fn read_csv_f32(path: String, opts: CsvOptions = CsvOptions()) -> Tensor[Float32]:
    return read_csv_with[Float32](path, _f64_to_f32, opts)

@always_inline
fn write_csv_f32(
    x: Tensor[Float32], path: String, delim: String = ",",
    header: Optional[List[String]] = None
) -> Bool:
    return write_csv[Float32](x, _f32_to_f64, path, delim, header)

@always_inline
fn read_json_f32(path: String) -> Tensor[Float32]:
    return read_json_numbers_with[Float32](path, _f64_to_f32)

@always_inline
fn write_json_f32(x: Tensor[Float32], path: String, pretty: Bool = True) -> Bool:
    return write_json_numbers[Float32](x, _f32_to_f64, path, pretty)

@always_inline
fn read_xml_f32(path: String, row_tag: String = "row", item_tag: String = "v") -> Tensor[Float32]:
    return read_xml_numbers_with[Float32](path, _f64_to_f32, row_tag, item_tag)

@always_inline
fn write_xml_f32(
    x: Tensor[Float32], path: String,
    root_tag: String = "tensor", row_tag: String = "row", item_tag: String = "v"
) -> Bool:
    return write_xml_numbers[Float32](x, _f32_to_f64, path, root_tag, row_tag, item_tag)

# --- Int ---
@always_inline
fn read_csv_int(path: String, opts: CsvOptions = CsvOptions()) -> Tensor[Int]:
    return read_csv_with[Int](path, _f64_to_int, opts)

@always_inline
fn write_csv_int(
    x: Tensor[Int], path: String, delim: String = ",",
    header: Optional[List[String]] = None
) -> Bool:
    return write_csv[Int](x, _int_to_f64, path, delim, header)

@always_inline
fn read_json_int(path: String) -> Tensor[Int]:
    return read_json_numbers_with[Int](path, _f64_to_int)

@always_inline
fn write_json_int(x: Tensor[Int], path: String, pretty: Bool = True) -> Bool:
    return write_json_numbers[Int](x, _int_to_f64, path, pretty)

@always_inline
fn read_xml_int(path: String, row_tag: String = "row", item_tag: String = "v") -> Tensor[Int]:
    return read_xml_numbers_with[Int](path, _f64_to_int, row_tag, item_tag)

@always_inline
fn write_xml_int(
    x: Tensor[Int], path: String,
    root_tag: String = "tensor", row_tag: String = "row", item_tag: String = "v"
) -> Bool:
    return write_xml_numbers[Int](x, _int_to_f64, path, root_tag, row_tag, item_tag)
