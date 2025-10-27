# MIT License
# SPDX-License-Identifier: MIT
# Project: momijo.vision
# File: src/momijo/vision/io/decode_png.mojo
# Description: PNG reader: Gray/GA/RGB/RGBA + Indexed; bit depths 1/2/4/8/16; filters 0..4; interlace none/Adam7;
#              inflate stored/fixed/dynamic.
#   - Public APIs:
#       decode_png_bytes(...) -> u8
#       decode_png_bytes_u16(...) -> u16 (only when IHDR bit_depth==16 & non-indexed)
#   - read_png(...) elsewhere wraps u8 API to Image.

from collections.list import List
from momijo.vision.io.inflate import zlib_inflate
from momijo.vision.io.unfilter import png_unfilter

@always_inline
fn _be32(b0: UInt8, b1: UInt8, b2: UInt8, b3: UInt8) -> Int:
    return (Int(b0) << 24) | (Int(b1) << 16) | (Int(b2) << 8) | Int(b3)

@always_inline
fn _ceil_div(a: Int, b: Int) -> Int:
    return (a + b - 1) // b

fn _expand_bits_to_bytes(bits: Int, packed: List[UInt8], pixels: Int) -> List[UInt8]:
    var out = List[UInt8]()
    var i = 0; var idx = 0
    while i < len(packed) and idx < pixels:
        var byte = Int(packed[i])
        if bits == 1:
            var k = 7
            while k >= 0 and idx < pixels:
                var v = ((byte >> k) & 1) * 255
                out.append(UInt8(v)); k -= 1; idx += 1
        elif bits == 2:
            var k2 = 6
            while k2 >= 0 and idx < pixels:
                var n = (byte >> k2) & 3
                var v = (n * 255) // 3
                out.append(UInt8(v)); k2 -= 2; idx += 1
        else:
            var hi = (byte >> 4) & 15; var lo = byte & 15
            var vhi = (hi * 255) // 15; var vlo = (lo * 255) // 15
            out.append(UInt8(vhi)); idx += 1
            if idx < pixels: out.append(UInt8(vlo)); idx += 1
        i += 1
    return out.copy()

fn _expand_indexed_bits_to_indices(bits: Int, packed: List[UInt8], pixels: Int) -> List[UInt8]:
    var out = List[UInt8]()
    var i = 0; var idx = 0
    while i < len(packed) and idx < pixels:
        var byte = Int(packed[i])
        if bits == 1:
            var k = 7
            while k >= 0 and idx < pixels:
                var v = (byte >> k) & 1
                out.append(UInt8(v)); k -= 1; idx += 1
        elif bits == 2:
            var k2 = 6
            while k2 >= 0 and idx < pixels:
                var n = (byte >> k2) & 3
                out.append(UInt8(n)); k2 -= 2; idx += 1
        else:
            var hi = (byte >> 4) & 15; var lo = byte & 15
            out.append(UInt8(hi)); idx += 1
            if idx < pixels: out.append(UInt8(lo)); idx += 1
        i += 1
    return out.copy()

fn _expand_indexed_to_rgb(plte: List[UInt8], trns: List[UInt8], indices: List[UInt8]) -> (Int, List[UInt8]):
    var has_alpha = len(trns) > 0
    var out = List[UInt8]()
    var i = 0
    while i < len(indices):
        var idx = Int(indices[i])
        var p = idx * 3
        if p + 2 >= len(plte): p = 0
        out.append(plte[p+0]); out.append(plte[p+1]); out.append(plte[p+2])
        if has_alpha:
            var a: UInt8 = UInt8(255)
            if idx < len(trns): a = trns[idx]
            out.append(a)
        i += 1
    var n_channels: Int = 3
    if has_alpha:
        n_channels = 4
    return (n_channels, out.copy())


fn _downscale16_to_8(buf16: List[UInt8]) -> List[UInt8]:
    var out = List[UInt8]()
    var i = 0
    while i + 1 < len(buf16):
        out.append(buf16[i])   # MSB
        i += 2
    return out.copy()

# Common parse that returns (ok,w,h,bit_depth,color_type,interlace,plte,trns,idat)
fn _parse_chunks(bytes: List[UInt8]) -> (Bool, Int, Int, Int, Int, Int, List[UInt8], List[UInt8], List[UInt8]):
    if len(bytes) < 8: return (False,0,0,0,0,0,List[UInt8](),List[UInt8](),List[UInt8]())
    if not (bytes[0]==UInt8(0x89) and bytes[1]==UInt8(0x50) and bytes[2]==UInt8(0x4E) and bytes[3]==UInt8(0x47) and bytes[4]==UInt8(0x0D) and bytes[5]==UInt8(0x0A) and bytes[6]==UInt8(0x1A) and bytes[7]==UInt8(0x0A)):
        return (False,0,0,0,0,0,List[UInt8](),List[UInt8](),List[UInt8]())
    var pos = 8
    var w = 0; var h = 0; var bit_depth = 0; var color_type = 0; var interlace = 0
    var plte = List[UInt8](); var trns = List[UInt8](); var idat = List[UInt8]()
    while pos + 8 <= len(bytes):
        var L = _be32(bytes[pos], bytes[pos+1], bytes[pos+2], bytes[pos+3]); pos += 4
        var t0 = bytes[pos]; var t1 = bytes[pos+1]; var t2 = bytes[pos+2]; var t3 = bytes[pos+3]; pos += 4
        if pos + L + 4 > len(bytes): return (False,0,0,0,0,0,List[UInt8](),List[UInt8](),List[UInt8]())
        var start = pos; var end = pos + L; pos = end; pos += 4
        if t0==UInt8(0x49) and t1==UInt8(0x48) and t2==UInt8(0x44) and t3==UInt8(0x52):
            if L < 13: return (False,0,0,0,0,0,List[UInt8](),List[UInt8](),List[UInt8]())
            w = _be32(bytes[start+0], bytes[start+1], bytes[start+2], bytes[start+3])
            h = _be32(bytes[start+4], bytes[start+5], bytes[start+6], bytes[start+7])
            bit_depth  = Int(bytes[start+8]); color_type = Int(bytes[start+9]); interlace  = Int(bytes[start+12])
        elif t0==UInt8(0x50) and t1==UInt8(0x4C) and t2==UInt8(0x54) and t3==UInt8(0x45):
            var i = start; 
            while i < end: plte.append(bytes[i]); i += 1
        elif t0==UInt8(0x74) and t1==UInt8(0x52) and t2==UInt8(0x4E) and t3==UInt8(0x53):
            var j = start; 
            while j < end: trns.append(bytes[j]); j += 1
        elif t0==UInt8(0x49) and t1==UInt8(0x44) and t2==UInt8(0x41) and t3==UInt8(0x54):
            var k = start; 
            while k < end: idat.append(bytes[k]); k += 1
        elif t0==UInt8(0x49) and t1==UInt8(0x45) and t2==UInt8(0x4E) and t3==UInt8(0x44):
            break
    return (True,w,h,bit_depth,color_type,interlace,plte.copy(),trns.copy(),idat.copy())

fn _collect_scan(w: Int, h: Int, scan_channels: Int, bit_depth: Int, interlace: Int, infl: List[UInt8]) -> (Bool, List[UInt8]):
    if interlace == 0:
        var ok_u = png_unfilter(w, h, scan_channels, infl.copy(), bit_depth)
        if not ok_u[0]: return (False, List[UInt8]())
        return (True, ok_u[1].copy())
    # Adam7
    @always_inline
    fn _ceil_div(a: Int, b: Int) -> Int: return (a + b - 1) // b
    var XOFF = [0,4,0,2,0,1,0]; var YOFF = [0,0,4,0,2,0,1]
    var XSP  = [8,8,4,4,2,2,1]; var YSP  = [8,8,8,4,4,2,2]
    var out = List[UInt8](); var cur = 0; var pass_idx = 0
    while pass_idx < 7:
        var x0 = XOFF[pass_idx]; var y0 = YOFF[pass_idx]
        var xs = XSP[pass_idx]; var ys = YSP[pass_idx]
        var pw = 0; var ph = 0
        if w > x0: pw = _ceil_div(w - x0, xs)
        if h > y0: ph = _ceil_div(h - y0, ys)
        if pw <= 0 or ph <= 0: pass_idx += 1; continue
        var row_bytes = 0
        if bit_depth == 8: row_bytes = pw * scan_channels
        elif bit_depth == 16: row_bytes = pw * scan_channels * 2
        else: row_bytes = ((pw * bit_depth + 7) // 8) * scan_channels
        var need = ph * (1 + row_bytes)
        if cur + need > len(infl): return (False, List[UInt8]())
        var sub = List[UInt8](); var t = 0; 
        while t < need: sub.append(infl[cur+t]); t += 1
        cur += need
        var okp = png_unfilter(pw, ph, scan_channels, sub.copy(), bit_depth)
        if not okp[0]: return (False, List[UInt8]())
        out.extend(okp[1].copy()); pass_idx += 1
    return (True, out.copy())

# u8 path (downscale 16->8 MSB; expand indexed/gray low-bit)
fn decode_png_bytes(bytes: List[UInt8]) -> (Bool, Int, Int, Int, List[UInt8]):
    var P = _parse_chunks(bytes.copy())
    if not P[0]: return (False,0,0,0,List[UInt8]())
    var w = P[1]; var h = P[2]; var bit_depth = P[3]; var color_type = P[4]; var interlace = P[5]
    var plte = P[6].copy(); var trns = P[7].copy(); var idat = P[8].copy()

    var ok_inf = zlib_inflate(idat.copy())
    if not ok_inf[0]: return (False,0,0,0,List[UInt8]())
    var infl = ok_inf[1].copy()

    var scan_channels = 0
    if color_type == 0: scan_channels = 1
    elif color_type == 2: scan_channels = 3
    elif color_type == 3: scan_channels = 1
    elif color_type == 4: scan_channels = 2
    elif color_type == 6: scan_channels = 4
    else: return (False,0,0,0,List[UInt8]())

    var pk = _collect_scan(w,h,scan_channels,bit_depth,interlace,infl.copy())
    if not pk[0]: return (False,0,0,0,List[UInt8]())
    var packed = pk[1].copy()

    if bit_depth == 16:
        return (True, w, h, scan_channels, _downscale16_to_8(packed.copy()))
    if bit_depth == 8:
        if color_type == 3:
            var c = _expand_indexed_to_rgb(plte.copy(), trns.copy(), packed.copy())
            return (True, w, h, c[0], c[1].copy())
        return (True, w, h, scan_channels, packed.copy())
    if bit_depth == 1 or bit_depth == 2 or bit_depth == 4:
        if color_type == 0:
            var count = w*h; var gray = _expand_bits_to_bytes(bit_depth, packed.copy(), count)
            return (True, w, h, 1, gray.copy())
        elif color_type == 3:
            var count2 = w*h; var idxs = _expand_indexed_bits_to_indices(bit_depth, packed.copy(), count2)
            var c2 = _expand_indexed_to_rgb(plte.copy(), trns.copy(), idxs.copy())
            return (True, w, h, c2[0], c2[1].copy())
    return (False,0,0,0,List[UInt8]())

# u16 path: only non-indexed with bit_depth==16, returns packed big-endian samples as UInt16 list
# Caller can interpret as native-endian values (we store as UInt16 with host endianness)
fn decode_png_bytes_u16(bytes: List[UInt8]) -> (Bool, Int, Int, Int, List[UInt16]):
    var P = _parse_chunks(bytes.copy())
    if not P[0]: return (False,0,0,0,List[UInt16]())
    var w = P[1]; var h = P[2]; var bit_depth = P[3]; var color_type = P[4]; var interlace = P[5]
    if bit_depth != 16 or color_type == 3: return (False,0,0,0,List[UInt16]())

    var idat = P[8].copy()
    var ok_inf = zlib_inflate(idat.copy())
    if not ok_inf[0]: return (False,0,0,0,List[UInt16]())
    var infl = ok_inf[1].copy()

    var scan_channels = 0
    if color_type == 0: scan_channels = 1
    elif color_type == 2: scan_channels = 3
    elif color_type == 4: scan_channels = 2
    elif color_type == 6: scan_channels = 4
    else: return (False,0,0,0,List[UInt16]())

    var pk = _collect_scan(w,h,scan_channels,bit_depth,interlace,infl.copy())
    if not pk[0]: return (False,0,0,0,List[UInt16]())
    var bytes16 = pk[1].copy()

    var out = List[UInt16]()
    var i = 0
    while i + 1 < len(bytes16):
        var hi = Int(bytes16[i]); var lo = Int(bytes16[i+1])
        var v = (hi << 8) | lo
        out.append(UInt16(v))
        i += 2
    return (True, w, h, scan_channels, out.copy())