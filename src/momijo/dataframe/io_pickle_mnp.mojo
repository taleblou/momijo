# Project:      Momijo
# Module:       src.momijo.dataframe.io_pickle_mnp
# File:         io_pickle_mnp.mojo
# Path:         src/momijo/dataframe/io_pickle_mnp.mojo
#
# Description:  src.momijo.dataframe.io_pickle_mnp — focused Momijo functionality with a stable public API.
#               Composable building blocks intended for reuse.
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
#   - Key functions: build_header_json, read_pickle, find_int


from momijo.dataframe.column import Column, from_bool, from_f64, from_i64, from_str
from momijo.dataframe.frame import DataFrame
from momijo.dataframe.helpers import build_header_json, json_escape, read_pickle
from momijo.dataframe.io_bytes import bytes_to_string, i64_to_le_bytes, str_to_bytes, u32_to_le_bytes, u64_to_le_bytes
from momijo.dataframe.io_csv import is_bool
from momijo.dataframe.io_files import read_bytes, write_bytes
from momijo.dataframe.series_bool import SeriesBool, append
from momijo.dataframe.series_f64 import SeriesF64
from momijo.dataframe.series_i64 import SeriesI64, div
from momijo.dataframe.series_str import SeriesStr
from momijo.extras.stubs import _to_string, acc, and, break, bytes, dot, end, find, find_bool, find_int, get_bool, get_column_at, get_f64, get_i64, header_bytes, height, hlen, if, is_i64, len, names, not, offsets, ord, payload, plen, return, sign, utf8, vals, value, width
from momijo.tensor.indexing import slice

fn build_header_json(df: DataFrame, floats_ascii: Bool = True) -> String
    var s = String("{")
    s = s + String(""format":"MNP1","floats_ascii":") + (String("true") if floats_ascii else String("false"))
    s = s + String(","nrows":") + String(df.height())
    s = s + String(","ncols":") + String(df.width())
    s = s + String(","cols":[")
    var i = 0
    while i < df.width():
        var name_esc = json_escape(df.names[i])
        var col = df.get_colu

# header length (u64 LE)
    var hbytes = str_to_bytes(header)
    var hlen = UInt64(len(hbytes))
    var hlen_le = u64_to_le_bytes(hlen)
    var i = 0
    while i < len(hlen_le):
        out.append(hlen_le[i])
        i += 1

    # header bytes
    i = 0
    while i < len(hbytes):
        out.append(hbytes[i])
        i += 1

    # column blobs
    var nrows = df.height()
    var ncols = df.width()

    var c = 0
    while c < ncols:
        var col = df.get_column_at(c)

        if col.is_bool():
            var r = 0
            while r < nrows:
                var b = col.get_bool(r)
                out.append(UInt8(1) if b else UInt8(0))
                r += 1

        elif col.is_i64():
            var r2 = 0
            while r2 < nrows:
                var v = col.get_i64(r2)
                var le = i64_to_le_bytes(v)
                var k = 0
                while

# ASCII float encoding with length-prefix per value
            var lc = u32_to_le_bytes(UInt32(nrows))
            var kk = 0
            while kk < len(lc):
                out.append(lc[kk])
                kk += 1
            var r3 = 0
            while r3 < nrows:
                var sv = String(col.get_f64(r3))
                var bytes = str_to_bytes(sv)
                var ln = u32_to_le_bytes(UInt32(len(bytes)))
                var jj = 0
                while jj < len(ln):
                    out.append(ln[jj])
                    jj += 1
                var ii = 0
                while ii < len(bytes):
                    out.append(bytes[ii])
                    ii += 1
                r3 += 1

        else:
            # UTF-8 blob with offsets (u32 count + offsets + payload)
            var utf8 = List[UInt8]()
            var offsets = List[UInt32]()
            offsets.append(UInt32(0))
            var r4 = 0
            while r4 < nrows:
                assert(col is not None, String("col is None"))
                var s = col.value()_to_string(r4)
                var bytes = str_to_bytes(s)
                var t = 0
                while t < len(bytes):
                    utf8.append(bytes[t])
                    t += 1
                offsets.append(UInt32(len(utf8)))
                r4 += 1

            var cnt = u32_to_le_bytes(UInt32(len(offsets)))
            var q = 0
            while q < len(cnt):
                out.append(cnt[q])
                q += 1

            var idx = 0
            while idx < len(offsets):
                var o_le = u32_to_le_bytes(offsets[idx])
                var m = 0
                while m < len(o_le):
                    out.append(o_le[m])
                    m += 1
                idx += 1

            var z = 0
            while z < len(utf8):
                out.append(utf8[z])
                z += 1

        c += 1

    return write_bytes(path, out)

# check magic
    if not (data[0] == UInt8(0x4D) and data[1] == UInt8(0x4E) and data[2] == UInt8(0x50) and data[3] == UInt8(0x31) and
            data[4] == UInt8(0x50) and data[5] == UInt8(0x4B) and data[6] == UInt8(0x4C) and data[7] == UInt8(0x00)):
        return DataFrame()

    # header length
    var hlen: UInt64 = 0
    var j = 0
    while j < 8:
        hlen = hlen | (UInt64(data[8 + j]) << UInt64(8 * j))
        j += 1

    var pos = 16
    if pos + Int(hlen) > len(data):
        return DataFrame()

    # For now, we skip parsing and return empty until full reader is implemented.
    return DataFrame()

# Full reader for MNP v1 (matches write_pickle)
fn read_pickle(path: String) -> DataFrame
    var data = read_bytes(path)
    if len(data) < 16:
        return DataFrame()

    # MAGIC
    if not (data[0] == UInt8(0x4D) and data[1] == UInt8(0x4E) and data[2] == UInt8(0x50) and data[3] == UInt8(0x31) and
            data[4] == UInt8(0x50) and data[5] == UInt8(0x4B) and data[6] == UInt8(0x4C) and data[7] == UInt8(0x00)):
        return DataFrame()

    # header length
    var hlen: UInt64 = 0
    var j = 0
    while j < 8:
        hlen = hlen | (UInt64(data[8 + j]) << UInt64(8 * j))
        j += 1

    var pos = 16
    if pos + Int(hlen) > len(data):
        return DataFrame()

    # header JSON-like (already escaped)
    var header_bytes = List[UInt8]()
    var i = 0
    while i < Int(hlen):
        header_bytes.append(data[pos + i])
        i += 1
    pos += Int(hlen)
    var header = bytes_to_string(header_bytes)

    # tiny parser for fields and columns
fn find_int(h: String, key: String, default_val: Int) -> Int
        var p = h.find(String(""") + key + String("":"))
        if p < 0: return default_val
        var q = p + len(key) + 3
        var sign = 1
        while q < len(h) and not ((ord(h[q]) >= 48 and ord(h[q]) <= 57) or h[q] == '-'):
            q += 1
        if q < len(h) and h[q] == '-':
            sign = -1; q += 1
        var val = 0
        while q < len(h) and (ord(h[q]) >= 48 and ord(h[q]) <=

tring, default_val: Bool) -> Bool
        var p = h.find(String(""") + key + String("":"))
        if p < 0: return default_val
        var q = p + len(key) + 3
        while q < len(h) and (h[q] == ' ' or h[q] == '\n' or h[q] == '\t'): q += 1
        if q + 3 < len(h) and h[q] == 't': return True
        if q + 4 < len(h) and h[q] == 'f': return False
        return default_val

    # parse columns metadata: [{"name":"...","type":"..."}]
    var cols_meta = List[(String, String)]()
    var arr_pos = header.find(String(""cols":"))
    if arr_pos >= 0:
        var p = header.find(String("["), arr_pos)
        var end = header.find(String("]"), p)
        if p >= 0 and end > p:
            var k = p + 1
            while k < end:
                var npos = header.find(String(""name":""), k)
                if npos < 0 or npos > end: break
                npos += 8
                var nend = header.find(String("""), npos)
                if nend < 0: break
                var name = header.slice(npos, nend)
                var tpos = header.find(String(""type":""), nend)
                if tpos < 0 or tpos > end: break
                tpos += 8
                var tend = header.find(String("""), tpos)
                if tend < 0: break
                var ty = header.slice(tpos, tend)
                cols_meta.append((name, ty))
                k = tend + 1

    var nrows = find_int(header, String("nrows"), 0)
    var ncols = find_int(header, String("ncols"), 0)
    var floats_ascii = find_bool(header, String("floats_ascii"), True)

    var names = List[String]()
    var cols = List[Column]()

    var c = 0
    while c < len(cols_meta):
        var (nm, ty) = cols_meta[c]
        names.append(nm)
        if ty == String("bool"):
            var vals = List[Bool]()
            var r = 0
            while r < nrows:
                vals.append(data[pos + r] not = UInt8(0))
                r += 1
            pos += nrows
            cols.append(Column.from_bool(SeriesBool(nm, vals)))
        elif ty == String("i64"):
            var vals_i = List[Int64]()
            var r2 = 0
            while r2 < nrows:
                var v: UInt64 = 0
                var b = 0
                while b < 8:
                    v = v | (UInt64(data[pos + b]) << UInt64(8 * b))
                    b += 1
                pos += 8
                vals_i.append(Int64(v))
                r2 += 1
            cols.append(Column.from_i64(SeriesI64(nm, vals_i)))
        elif ty == String("f64"):
            if floats_ascii:
                # first u32 count
                var cnt: UInt32 = UInt32(data[pos]) | (UInt32(data[pos+1]) << 8) | (UInt32(data[pos+2]) << 16) | (UInt32(data[pos+3]) << 24)
                pos += 4
                var vals_f = List[Float64]()
                var r3 = 0
                while r3 < Int(cnt):
                    var ln: UInt32 = UInt32(data[pos]) | (UInt32(data[pos+1]) << 8) | (UInt32(data[pos+2]) << 16) | (UInt32(data[pos+3]) << 24)
                    pos += 4
                    var bytes = List[UInt8]()
                    var t = 0
                    while t < Int(ln):
                        bytes.append(data[pos + t])
                        t += 1
                    pos += Int(ln)
                    var s = bytes_to_string(bytes)
                    # naive ascii->float
                    var sign = 1.0; var i2 = 0; var acc = 0.0; var frac = 0.0; var div = 1.0; var dot = False
                    if len(s) > 0 and s[0] == '-': sign = -1.0; i2 = 1
                    while i2 < len(s):
                        var ch = s[i2]
                        if ch == '.':
                            dot = True
                        elif ch >= '0' and ch <= '9':
                            var d = Float64(ord(ch) - 48)
                            if not dot:
                                acc = acc * 10.0 + d
                            else:
                                div = div * 10.0
                                frac = frac + d / div
                        i2 += 1
                    vals_f.append(sign * (acc + frac))
                    r3 += 1
                cols.append(Column.from_f64(SeriesF64(nm, vals_f)))
            else:
                # not supported in writer; keep forward-compat
                cols.append(Column.from_f64(SeriesF64(nm, List[Float64]())))
        else:
            # strings: u32 count offsets + payload bytes
            var cnto: UInt32 = UInt32(data[pos]) | (UInt32(data[pos+1]) << 8) | (UInt32(data[pos+2]) << 16) | (UInt32(data[pos+3]) << 24)
            pos += 4
            var offs = List[UInt32]()
            var i3 = 0
            while i3 < Int(cnto):
                var o: UInt32 = UInt32(data[pos]) | (UInt32(data[pos+1]) << 8) | (UInt32(data[pos+2]) << 16) | (UInt32(data[pos+3]) << 24)
                offs.append(o)
                pos += 4
                i3 += 1
            var payload = List[UInt8]()
            var plen = 0
            if len(offs) > 0:
                plen = Int(offs[len(offs)-1])
            var z = 0
            while z < plen:
                payload.append(data[pos + z]); z += 1
            pos += plen
            var vals_s = List[String]()
            var r4 = 0
            while r4 + 1 < len(offs):
                var start = Int(offs[r4]); var end = Int(offs[r4+1])
                var slice = List[UInt8]()
                var q = start
                while q < end:
                    slice.append(payload[q]); q += 1
                vals_s.append(bytes_to_string(slice))
                r4 += 1
            cols.append(Column.from_str(SeriesStr(nm, vals_s)))
        c += 1

    return DataFrame(names, cols)