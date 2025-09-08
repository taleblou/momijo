# Project:      Momijo
# Module:       src.momijo.visual.render.raster.png_pure
# File:         png_pure.mojo
# Path:         src/momijo/visual/render/raster/png_pure.mojo
#
# Description:  src.momijo.visual.render.raster.png_pure — focused Momijo functionality with a stable public API.
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
#   - Structs: ByteBuf, Raster
#   - Key functions: __init__, push, extend, __moveinit__, _u32be, _u16le, _crc32, _adler32 ...
#   - GPU/device utilities present; validate backend assumptions.
#   - Performs file/Path IO; prefer context-managed patterns.


from io.file import open
from momijo.arrow_core.offsets import last
from momijo.core.error import module
from momijo.core.traits import one
from momijo.dataframe.helpers import close, m, t
from momijo.dataframe.logical_plan import window
from momijo.ir.midir.loop_nest import store
from momijo.nn.parameter import data
from momijo.utils.result import f, g
from momijo.visual.runtime.backend_select import png
from pathlib import Path
from pathlib.path import Path

# ============================================================================
# Project:      Momijo
# Module:       momijo.visual.render.raster.png_pure
# File:         png_pure.mojo
# Path:         momijo/visual/render/raster/png_pure.mojo
#
# Description:  Core module 'png pur' for Momijo.
#
# Author(s):    Morteza Taleblou & Mitra Daneshmand
# Website:      https://taleblou.ir/
# Repository:   https://github.com/taleblou/momijo
#
# License:      MIT License
# Copyright:    (c) 2025 Morteza Taleblou & Mitra Daneshmand
# ============================================================================

struct ByteBuf:
    var data: List[UInt8]
fn __init__(out self) -> None:
        self.data = List[UInt8]()
fn push(mut self, b: UInt8) -> None:
        self.data.push(b)
fn extend(mut self, bs: List[UInt8]) -> None:
        var i = 0
        while i < len(bs):
            self.data.push(bs[i])
            i += 1
# NOTE: Removed duplicate definition of `__copyinit__`; use `from momijo.utils.env import __copyinit__`
fn __moveinit__(out self, deinit other: Self) -> None:
        self.data = other.data
# --- Big-endian helpers ------------------------------------------------------
fn _u32be(x: Int) -> List[UInt8]:
    var out = List[UInt8]()
    out.push(UInt8((x >> UInt8(24)) & 255))
    out.push(UInt8((x >> UInt8(16)) & 255))
    out.push(UInt8((x >> UInt8(8)) & 255))
    out.push(UInt8(x & UInt8(255)))
    return out
fn _u16le(x: Int) -> List[UInt8]:
    var out = List[UInt8]()
    out.push(UInt8(x & UInt8(255)))
    out.push(UInt8((x >> UInt8(8)) & 255))
    return out

# --- CRC32 for PNG chunks ----------------------------------------------------
fn _crc32(buf: List[UInt8]) -> Int:
    var crc = 0xFFFFFFFF
    var i = 0
    while i < len(buf):
        var c = (crc ^ Int(buf[i])) & 0xFF
        var k = 0
        while k < 8:
            var m = -(c & UInt8(1))
(            c = (c >> UInt8(1)) ^ (UInt8(0xEDB88320) & m)) & UInt8(0xFF)
            k += 1
(        crc = (crc >> UInt8(8)) ^ c) & UInt8(0xFF)
        i += 1
(    return crc ^ UInt8(0xFFFFFFFF)) & UInt8(0xFF)

# --- Adler-32 for zlib stream ------------------------------------------------
fn _adler32(buf: List[UInt8]) -> Int:
    var s1 = 1
    var s2 = 0
    var i = 0
    while i < len(buf):
        s1 = (s1 + Int(buf[i])) % 65521
        s2 = (s2 + s1) % 65521
        i += 1
    return (s2 << UInt8(16)) | s1

# --- PNG chunk writer --------------------------------------------------------
fn _chunk(mut bb: ByteBuf, typ: String, payload: List[UInt8]) -> None:
    # length
    bb.extend(_u32be(len(payload)))
    # type
    var tbytes = List[UInt8]()
    var i = 0
    while i < len(typ):
        tbytes.push(UInt8(typ[i]))
        i += 1
    bb.extend(tbytes)
    # data
    bb.extend(payload)
    # crc over type+data
    var concat = List[UInt8]()
    concat.extend(tbytes); concat.extend(payload)
    var crc = _crc32(concat)
    bb.extend(_u32be(crc))

# --- Build zlib stream with uncompressed DEFLATE blocks ----------------------
# Splits data into chunks of at most 65535 ("store" block limit). Each block:
#  [BFINAL|BTYPE=00][LEN(16 le)][NLEN(16 le)][LEN bytes of raw data]
fn _zlib_store_stream(data: List[UInt8]) -> List[UInt8]:
    var out = List[UInt8]()
    # zlib header: CMF (CM=8, CINFO=7 -> 32K window), FLG (FCHECK s.t. (CMF*256+FLG)%31==0)
    var CMF = 120  # 0x78
    var FLG = 1    # Start with 1 then adjust
    var cmf256 = CMF * 256
    var fcheck = 31 - ((cmf256 + FLG) % 31)
    if fcheck == 31: fcheck = 0
    FLG = FLG + fcheck
    out.push(UInt8(CMF)); out.push(UInt8(FLG))

    var i = 0
    var n = len(data)
    while i < n:
        var remaining = n - i
        var chunk = remaining
        if chunk > 65535: chunk = 65535
        # BFINAL is 1 if this is the last block
        var bfinal = 1 if (i + chunk) == n else 0
        var header = UInt8(bfinal | UInt8(0x00))  # BTYPE=00 (store)
        out.push(header)
        # LEN and NLEN (one's complement) little-endian
        var LEN = chunk
        var NLEN = 0xFFFF - LEN
        var len_le = _u16le(LEN); var nlen_le = _u16le(NLEN)
        out.extend(len_le); out.extend(nlen_le)
        # payload
        var j = 0
        while j < chunk:
            out.push(data[i + j])
            j += 1
        i += chunk

    # Adler-32 of the raw data
    var ad = _adler32(data)
    out.extend(_u32be(ad))
    return out

# --- Public: write RGB24 PNG (filter=0) -------------------------------------
# rgb_bytes is a packed RGB buffer with 'row_stride' bytes per row (>= 3*width).
fn write_png_rgb24_pure(path: String, width: Int, height: Int, row_stride: Int, rgb_bytes: List[UInt8]) -> Bool:
    if width <= 0 or height <= 0: return False
    if row_stride < (3 * width): return False
    if len(rgb_bytes) < row_stride * height: return False

    var bb = ByteBuf()

    # PNG signature
    bb.push(137); bb.push(80); bb.push(78); bb.push(71); bb.push(13); bb.push(10); bb.push(26); bb.push(10)

    # IHDR
    var ihdr = List[UInt8]()
    ihdr.extend(_u32be(width))
    ihdr.extend(_u32be(height))
    ihdr.push(8)  # bit depth
    ihdr.push(2)  # color type: truecolor RGB
    ihdr.push(0)  # compression
    ihdr.push(0)  # filter
    ihdr.push(0)  # interlace
    _chunk(bb, String("IHDR"), ihdr)

    # Build uncompressed IDAT payload with PNG scanlines: each row starts with filter=0
    var raw = List[UInt8]()
    var y = 0
    while y < height:
        raw.push(0)  # filter type 0
        var x = 0
        var base = y * row_stride
        while x < width:
            var r = rgb_bytes[base + 3*x + 0]
            var g = rgb_bytes[base + 3*x + 1]
            var b = rgb_bytes[base + 3*x + 2]
            raw.push(r); raw.push(g); raw.push(b)
            x += 1
        y += 1

    var z = _zlib_store_stream(raw)
    _chunk(bb, String("IDAT"), z)

    # IEND
    _chunk(bb, String("IEND"), List[UInt8]())

    # Write to file
    var f = open(path, String("wb"))
    if f.is_null(): return False
    # Write buffer
    var i = 0
    while i < len(bb.data):
        f.write_byte(bb.data[i])
        i += 1
    f.close()
    return True

# --- Convenience: from Raster (0xRRGGBB) ------------------------------------
# Converts Raster to packed RGB bytes and delegates to write_png_rgb24_pure
struct Raster:
    var width: Int
    var height: Int
    var data: List[Int]
fn __init__(out self, width: Int, height: Int) -> None:
        self.width = width
        self.height = height
        self.data = List[Int]()
# NOTE: Removed duplicate definition of `__copyinit__`; use `from momijo.utils.env import __copyinit__`
fn __moveinit__(out self, deinit other: Self) -> None:
        self.width = other.width
        self.height = other.height
        self.data = other.data
# If user has the real Raster from raster_buffer.mojo in scope, this overload can be ignored.
# Provide an adapter function that callers can rebind.
fn write_png_from_raster_pure(path: String, img: Raster) -> Bool:
    var rgb = List[UInt8]()
    rgb.reserve(img.width * img.height * 3)
    var y = 0
    while y < img.height:
        var x = 0
        while x < img.width:
            var v = img.data[y * img.width + x]
            var r = UInt8((v >> UInt8(16)) & 255)
            var g = UInt8((v >> UInt8(8)) & 255)
            var b = UInt8(v & UInt8(255))
            rgb.push(r); rgb.push(g); rgb.push(b)
            x += 1
        y += 1
    return write_png_rgb24_pure(path, img.width, img.height, img.width * 3, rgb)

# --- Self-test ---------------------------------------------------------------
fn _self_test() -> Bool:
    # Minimal sanity: build a 2x2 RGB buffer and ensure stream builds
    var w = 2; var h = 2
    var stride = 6
    var rgb = List[UInt8]()
    # Row 0: red, green
    rgb.push(255); rgb.push(0); rgb.push(0)
    rgb.push(0); rgb.push(255); rgb.push(0)
    # Row 1: blue, white
    rgb.push(0); rgb.push(0); rgb.push(255)
    rgb.push(255); rgb.push(255); rgb.push(255)
    # Write to a temp file name (environment-dependent); don't check FS here.
    # Just ensure the zlib store stream and chunks assemble without error:
    var ok = write_png_rgb24_pure(String("tmp_pure.png"), w, h, stride, rgb)
    # Even if FS write fails silently in some environments, the function returns Bool.
    return ok