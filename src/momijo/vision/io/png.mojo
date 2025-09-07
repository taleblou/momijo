# MIT License
# Copyright (c) 2025 Morteza Talebou and Mitra Daneshmand
# Project: momijo  |  Source: https://github.com/taleblou/momijo
# This file is part of the Momijo project. See the LICENSE file at the repository root.
# Momijo 
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Morteza Talebou and Mitra Daneshmand
# Website: https://taleblou.ir/
# Repository: https://github.com/taleblou/momijo
#
# Project: momijo.vision.io
# File: src/momijo/vision/io/png.mojo

from momijo.dataframe.helpers import t
from momijo.nn.parameter import data
from momijo.utils.timer import start, stop
from momijo.visual.runtime.backend_select import png
from pathlib import Path
from pathlib.path import Path

struct ImageRGBAU8(Copyable, Movable):
    var h: Int
    var w: Int
    var data: List[UInt8]   # packed RGBA, HWC
fn __init__(out self, h: Int, w: Int, data: List[UInt8]) -> None:
        self.h = h; self.w = w
        var expected = h * w * 4
        if len(data) != expected:
            var buf: List[UInt8] = List[UInt8]()
            var i = 0
            while i < expected:
                buf.append(0)
                i += 1
            self.data = buf
        else:
            self.data = data

# -------------------------
# PNG Info
# -------------------------
struct PngInfo(Copyable, Movable):
    var ok: Bool
    var width: Int
    var height: Int
    var bit_depth: Int
    var color_type: Int
    var interlace: Int
    var has_srgb: Bool
fn __init__(out self) -> None:
        self.ok = False
        self.width = 0
        self.height = 0
        self.bit_depth = 0
        self.color_type = 0
        self.interlace = 0
        self.has_srgb = False

# -------------------------
# Helpers
# -------------------------
@staticmethod
fn _read_be_u32(buf: List[UInt8], i: Int) -> (Bool, Int, Int):
    # returns (ok, value, next_index)
    if i + 3 >= len(buf):
        return (False, 0, i)
    var v = (Int(buf[i]) << 24) | (Int(buf[i+1]) << 16) | (Int(buf[i+2]) << 8) | Int(buf[i+3])
    return (True, v, i + 4)

@staticmethod
fn _eq4(buf: List[UInt8], i: Int, a: UInt8, b: UInt8, c: UInt8, d: UInt8) -> Bool:
    if i + 3 >= len(buf):
        return False
    return buf[i] == a and buf[i+1] == b and buf[i+2] == c and buf[i+3] == d

# -------------------------
# Signature
# -------------------------
@staticmethod
fn is_png(buf: List[UInt8]) -> Bool:
    # 8-byte signature: 89 50 4E 47 0D 0A 1A 0A
    if len(buf) < 8:
        return False
    return (buf[0] == UInt8(0x89) and buf[1] == UInt8(0x50) and buf[2] == UInt8(0x4E) and buf[3] == UInt8(0x47) and
            buf[4] == UInt8(0x0D) and buf[5] == UInt8(0x0A) and buf[6] == UInt8(0x1A) and buf[7] == UInt8(0x0A))

# -------------------------
# Parse IHDR (+ sRGB detection)
# -------------------------
@staticmethod
fn parse_info(buf: List[UInt8]) -> PngInfo:
    var info = PngInfo()
    var n = len(buf)
    if n < 33:  # signature (8) + minimal IHDR chunk (~25)
        return info
    if not is_png(buf):
        return info
    var i = 8  # after signature

    while i + 12 <= n:
        # length (4), type (4), data (len), CRC (4)
        var ok_len = False; var length = 0; var j = i
        (ok_len, length, j) = _read_be_u32(buf, j)
        if not ok_len:
            return info
        var type_i = j
        j += 4
        var data_i = j
        var data_end = data_i + length
        var crc_i = data_end
        var next_chunk = crc_i + 4
        if next_chunk > n:
            return info

        # IHDR
        if _eq4(buf, type_i, UInt8(0x49), UInt8(0x48), UInt8(0x44), UInt8(0x52)):  # 'IHDR'
            if length == 13 and (data_i + 13) <= n:
                var ok_w = False; var w = 0; var p = data_i
                (ok_w, w, p) = _read_be_u32(buf, p)
                if not ok_w: return info
                var ok_h = False; var h = 0
                (ok_h, h, p) = _read_be_u32(buf, p)
                if not ok_h: return info
                var bit_depth = Int(buf[p]); p += 1
                var color_type = Int(buf[p]); p += 1
                var compression = Int(buf[p]); p += 1
                var filter_method = Int(buf[p]); p += 1
                var interlace = Int(buf[p]); p += 1
                # basic validation of fields
                if compression != 0 or filter_method != 0:
                    return info
                info.width = w
                info.height = h
                info.bit_depth = bit_depth
                info.color_type = color_type
                info.interlace = interlace
                # don't return yet; continue to see if sRGB exists before IDAT
                info.ok = True

        # sRGB chunk detection (no payload use here)
        if _eq4(buf, type_i, UInt8(0x73), UInt8(0x52), UInt8(0x47), UInt8(0x42)):  # 'sRGB'
            info.has_srgb = True

        # IDAT indicates start of image data; we can stop if IHDR was found
        if _eq4(buf, type_i, UInt8(0x49), UInt8(0x44), UInt8(0x41), UInt8(0x54)):  # 'IDAT'
            return info

        # move to next chunk
        i = next_chunk

    return info

# -------------------------
# Placeholders for future codec wiring
# -------------------------
@staticmethod
fn decode_to_rgba_u8(buf: List[UInt8]) -> (Bool, ImageRGBAU8):
    # Placeholder: real decoder not implemented.
    var img = ImageRGBAU8(0, 0, List[UInt8]())
    return (False, img)

@staticmethod
fn encode_from_rgba_u8(img: ImageRGBAU8) -> (Bool, List[UInt8]):
    # Placeholder: real encoder not implemented.
    return (False, List[UInt8]())

# -------------------------
# Minimal smoke test
# -------------------------
@staticmethod
fn __self_test__() -> Bool:
    # Build a tiny synthetic PNG-like buffer:
    # signature + IHDR(len=13) + dummy CRC + IDAT(len=0) + dummy CRC + IEND
    var b: List[UInt8] = List[UInt8]()

    # signature
    b.append(0x89); b.append(0x50); b.append(0x4E); b.append(0x47); b.append(0x0D); b.append(0x0A); b.append(0x1A); b.append(0x0A)

    # IHDR
    # len=13
    b.append(0x00); b.append(0x00); b.append(0x00); b.append(0x0D)
    # type 'IHDR'
    b.append(0x49); b.append(0x48); b.append(0x44); b.append(0x52)
    # data: w=3,h=2,bit=8,color=6(RGBA),comp=0,filter=0,interlace=0
    b.append(0x00); b.append(0x00); b.append(0x00); b.append(0x03)   # width 3
    b.append(0x00); b.append(0x00); b.append(0x00); b.append(0x02)   # height 2
    b.append(0x08)  # bit depth
    b.append(0x06)  # color type (RGBA)
    b.append(0x00)  # compression
    b.append(0x00)  # filter
    b.append(0x00)  # interlace
    # dummy CRC
    b.append(0x00); b.append(0x00); b.append(0x00); b.append(0x00)

    # IDAT len=0
    b.append(0x00); b.append(0x00); b.append(0x00); b.append(0x00)
    b.append(0x49); b.append(0x44); b.append(0x41); b.append(0x54)   # 'IDAT'
    # no data
    b.append(0x00); b.append(0x00); b.append(0x00); b.append(0x00)   # CRC

    # IEND len=0
    b.append(0x00); b.append(0x00); b.append(0x00); b.append(0x00)
    b.append(0x49); b.append(0x45); b.append(0x4E); b.append(0x44)   # 'IEND'
    b.append(0x00); b.append(0x00); b.append(0x00); b.append(0x00)   # CRC

    if not is_png(b): return False
    var info = parse_info(b)
    if not info.ok: return False
    if not (info.width == 3 and info.height == 2 and info.bit_depth == 8 and info.color_type == 6 and info.interlace == 0):
        return False

    var (ok_dec, img) = decode_to_rgba_u8(b)
    if ok_dec: return False  # placeholder should be False

    var (ok_enc, buf2) = encode_from_rgba_u8(ImageRGBAU8(2, 2, List[UInt8]()))
    if ok_enc: return False

    return True