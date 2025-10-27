# MIT License
# SPDX-License-Identifier: MIT
# Project: momijo.vision.io
# File: src/momijo/vision/io/jpeg_decoder.mojo
# Description: Minimal JPEG structure inspection and stub decoder that parses SOF0 to get width/height.
# Notes:
# - No globals (var-only), var-only style, Copyable & Movable where needed.
# - This module exposes: is_jpeg, parse_segment, inspect_jpeg_structure, decode_jpeg.

from momijo.vision.image import Image, ImageMeta, ColorSpace
from momijo.vision.tensor import Tensor
from momijo.vision.transforms.array import full

# -------------------------------
# Helpers
# -------------------------------

@always_inline
fn read_be_u16(p: UnsafePointer[UInt8]) -> Int:
    return (Int(p[0]) << 8) | Int(p[1])

@always_inline
fn is_jpeg(p: UnsafePointer[UInt8]) -> Bool:
    return p[0] == UInt8(0xFF) and p[1] == UInt8(0xD8)  # SOI marker

# Marker getters (no globals)
@always_inline fn MARK_SOI() -> Int:  return 0xFFD8
@always_inline fn MARK_EOI() -> Int:  return 0xFFD9
@always_inline fn MARK_APP0() -> Int: return 0xFFE0
@always_inline fn MARK_APP1() -> Int: return 0xFFE1
@always_inline fn MARK_DQT()  -> Int: return 0xFFDB
@always_inline fn MARK_SOF0() -> Int: return 0xFFC0
@always_inline fn MARK_DHT()  -> Int: return 0xFFC4
@always_inline fn MARK_SOS()  -> Int: return 0xFFDA
@always_inline fn MARK_TEM()  -> Int: return 0xFF01
@always_inline fn MARK_RST0() -> Int: return 0xFFD0  # ... up to RST7 (RST0 + 7)

@always_inline
fn is_rst(marker: Int) -> Bool:
    return marker >= MARK_RST0() and marker <= (MARK_RST0() + 7)

@always_inline
fn has_length_field(marker: Int) -> Bool:
    # Markers with NO 2-byte length field: SOI, EOI, TEM, RST0..RST7
    if marker == MARK_SOI(): return False
    if marker == MARK_EOI(): return False
    if marker == MARK_TEM(): return False
    if is_rst(marker): return False
    return True

@always_inline
fn _fallback_image() -> Image:
    # 1x1x3 black, packed HWC u8
    var arr = full((1, 1, 3), UInt8(0))
    var t = arr.tensor()
    var meta = ImageMeta().with_colorspace(ColorSpace.SRGB())
    return Image(meta.copy(), t.copy())

# -------------------------------
# Segment type
# -------------------------------

struct JpegSegment(Copyable, Movable):
    var marker: Int
    var length: Int     # length field value (includes 2 bytes of length)
    var data_ptr: UnsafePointer[UInt8]  # payload pointer (length-2 bytes)

    fn __init__(out self, marker: Int, length: Int, data_ptr: UnsafePointer[UInt8]):
        self.marker = marker
        self.length = length
        self.data_ptr = data_ptr

    fn __copyinit__(out self, other: Self):
        self.marker = other.marker
        self.length = other.length
        self.data_ptr = other.data_ptr

# Small result wrapper for parsing (avoids tuple unpack issues)
struct ParseSegResult(Copyable, Movable):
    var ok: Bool
    var seg: JpegSegment
    fn __init__(out self, ok: Bool, seg: JpegSegment):
        self.ok = ok
        self.seg = seg.copy()
    fn __copyinit__(out self, other: Self):
        self.ok = other.ok
        self.seg = other.seg.copy()

# -------------------------------
# Segment parser
# -------------------------------
# Parse when marker HAS a 2-byte length field.
# 'ptr' must point to 0xFF (then marker byte).
# Returns ok=False if truncated/invalid.
fn parse_segment(ptr: UnsafePointer[UInt8], end: UnsafePointer[UInt8]) -> ParseSegResult:
    if (ptr + 4) > end:
        return ParseSegResult(False, JpegSegment(0, 0, ptr))
    var marker = read_be_u16(ptr)
    var length = read_be_u16(ptr + 2)  # includes the 2 bytes of length itself
    if length < 2:
        return ParseSegResult(False, JpegSegment(marker, length, ptr + 4))
    var seg_total = 2 + length  # 2(marker) + length (2 + payload)
    if (ptr + seg_total) > end:
        return ParseSegResult(False, JpegSegment(marker, length, ptr + 4))
    return ParseSegResult(True, JpegSegment(marker, length, ptr + 4))

# -------------------------------
# Pretty helpers
# -------------------------------

fn _print_marker_name(marker: Int):
    if marker == MARK_APP0():
        print("[JPEG] APP0 (JFIF / JFXX)")
    else:
        if marker == MARK_APP1():
            print("[JPEG] APP1 (EXIF / XMP)")
        else:
            if marker == MARK_DQT():
                print("[JPEG] DQT (Quantization Table)")
            else:
                if marker == MARK_SOF0():
                    print("[JPEG] SOF0 (Baseline DCT)")
                else:
                    if marker == MARK_DHT():
                        print("[JPEG] DHT (Huffman Table)")
                    else:
                        if marker == MARK_SOS():
                            print("[JPEG] SOS (Start of Scan)")
                        else:
                            print("[JPEG] Marker 0x" + String(marker))

# -------------------------------
# SOF0 parsing (extract width/height/components)
# -------------------------------
# SOF0 payload layout:
#   P (1) | Y (2, height) | X (2, width) | Nf (1) | Nf * (Ci (1), HiVi (1), Tqi (1))
fn parse_sof0_dims(sof0_payload: UnsafePointer[UInt8], payload_len: Int) -> (Bool, Int, Int, Int):
    if payload_len < 6:
        return (False, 0, 0, 0)
    var precision = Int(sof0_payload[0])  # usually 8
    var height = read_be_u16(sof0_payload + 1)
    var width  = read_be_u16(sof0_payload + 3)
    var nf     = Int(sof0_payload[5])

    var need = 6 + 3 * nf
    if payload_len < need:
        return (False, 0, 0, 0)

    if precision != 8:
        return (False, 0, 0, 0)
    if width <= 0 or height <= 0:
        return (False, 0, 0, 0)

    return (True, width, height, nf)

# -------------------------------
# Public: structure inspection
# -------------------------------

fn inspect_jpeg_structure(ptr: UnsafePointer[UInt8], length: Int):
    if length < 4 or not is_jpeg(ptr):
        print("[JPEG] Not a JPEG file")
        return

    var end = ptr + length
    var cursor = ptr + 2  # after SOI
    print("[JPEG] SOI detected")

    while (cursor + 2) <= end:
        if cursor[0] != UInt8(0xFF):
            print("[JPEG] Non-marker byte encountered; stopping scan")
            break

        var marker = read_be_u16(cursor)

        if marker == MARK_EOI():
            print("[JPEG] EOI reached")
            cursor = cursor + 2
            break

        if not has_length_field(marker):
            if marker == MARK_TEM():
                print("[JPEG] TEM")
            else:
                if is_rst(marker):
                    print("[JPEG] RST" + String(marker - MARK_RST0()))
                else:
                    print("[JPEG] Standalone marker 0x" + String(marker))
            cursor = cursor + 2
            continue

        var res = parse_segment(cursor, end)
        if not res.ok:
            print("[JPEG] Truncated/invalid segment; aborting")
            break

        _print_marker_name(marker)

        # Advance: 2(marker) + length bytes
        cursor = cursor + (2 + res.seg.length)

    print("[JPEG] Structure parsing complete")

# -------------------------------
# Public: stub decoder that returns an all-zero image with parsed SOF0 dims
# -------------------------------

fn decode_jpeg(ptr: UnsafePointer[UInt8], length: Int) -> (Bool, Image):
    if length < 4 or not is_jpeg(ptr):
        return (False, _fallback_image())

    var end = ptr + length
    var cursor = ptr + 2

    var found_sof0 = False
    var width = 0
    var height = 0
    var ncomp = 0

    # Walk segments until SOS/EOI; capture SOF0 dims.
    while (cursor + 2) <= end:
        if cursor[0] != UInt8(0xFF):
            # Likely in entropy-coded data; stop structural pass
            break

        var marker = read_be_u16(cursor)

        if marker == MARK_EOI():
            cursor = cursor + 2
            break

        if not has_length_field(marker):
            # SOI/TEM/RSTn
            cursor = cursor + 2
            continue

        var res = parse_segment(cursor, end)
        if not res.ok:
            break

        if marker == MARK_SOF0():
            # res.seg.length includes its own 2 bytes; payload length = length - 2
            var payload_len = res.seg.length - 2
            var dims = parse_sof0_dims(res.seg.data_ptr, payload_len)
            if dims[0]:
                found_sof0 = True
                width  = dims[1]
                height = dims[2]
                ncomp  = dims[3]

        if marker == MARK_SOS():
            # Start of Scan — entropy-coded data follows; stop scanning structure
            break

        cursor = cursor + (2 + res.seg.length)

    if not found_sof0 or width <= 0 or height <= 0:
        return (False, _fallback_image())

    # Create a zero image with the parsed dimensions (1 or 3 channels).
    var c = 1
    if ncomp >= 3:
        c = 3

    # Allocate zero buffer and pack into a HWC tensor.
    var arr = full((height, width, c), UInt8(0))
    var t = arr.tensor()
    var meta = ImageMeta().with_colorspace(ColorSpace.SRGB())
    var img = Image(meta.copy(), t.copy())

    # Placeholder image (not real decode), but dimensions/channels are correct.
    return (True, img)
