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
# File: src/momijo/vision/io/jpeg.mojo

from momijo.core.device import id
from momijo.ir.passes.cse import find
from momijo.nn.parameter import data
from momijo.utils.result import g
from momijo.utils.timer import start
from pathlib import Path
from pathlib.path import Path

# NOTE: Removed duplicate definition of `ImageRGBU8`; use `from momijo.vision.io.ppm import ImageRGBU8`

fn __init__(out self, h: Int, w: Int, data: List[UInt8]) -> None:
        self.h = h; self.w = w
        var expected = h * w * 3
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
# JPEG Info
# -------------------------
struct JpegInfo(Copyable, Movable):
    var ok: Bool
    var jfif: Bool
    var precision: Int
    var height: Int
    var width: Int
    var num_components: Int
fn __init__(out self) -> None:
        self.ok = False
        self.jfif = False
        self.precision = 0
        self.height = 0
        self.width = 0
        self.num_components = 0

# -------------------------
# Helpers
# -------------------------
@staticmethod
fn _read_be_u16(buf: List[UInt8], i: Int) -> (Bool, Int, Int):
    # returns (ok, value, next_index)
    if i + 1 >= len(buf):
        return (False, 0, i)
    var v = (Int(buf[i]) << 8) | Int(buf[i+1])
    return (True, v, i + 2)

@staticmethod
fn _is_ff(byte: UInt8) -> Bool:
    return byte == UInt8(0xFF)

@staticmethod
fn _is_sof(marker: Int) -> Bool:
    # Baseline DCT (C0), Progressive (C2) etc. We accept common SOF0 and SOF2.
    return marker == 0xC0 or marker == 0xC1 or marker == 0xC2 or marker == 0xC3

# -------------------------
# Signature
# -------------------------
@staticmethod
fn is_jpeg(buf: List[UInt8]) -> Bool:
    if len(buf) < 2:
        return False
    return buf[0] == UInt8(0xFF) and buf[1] == UInt8(0xD8)

# -------------------------
# Parse metadata (walk markers until SOF)
# -------------------------
@staticmethod
fn parse_info(buf: List[UInt8]) -> JpegInfo:
    var info = JpegInfo()
    var n = len(buf)
    if n < 4:
        return info
    # SOI
    if not (buf[0] == UInt8(0xFF) and buf[1] == UInt8(0xD8)):
        return info
    var i = 2

    # iterate over segments
    while i + 3 < n:
        # find 0xFF
        while i < n and not _is_ff(buf[i]):
            i += 1
        if i + 1 >= n:
            return info
        # skip fill FFs
        while i < n and _is_ff(buf[i]):
            i += 1
        if i >= n:
            return info
        var marker = Int(buf[i])   # e.g., E0, DB, C0, DA, D9, ...
        i += 1

        # Standalone markers without length: SOI (D8), EOI (D9), RSTn (D0..D7)
        if marker == 0xD8 or marker == 0xD9 or (marker >= 0xD0 and marker <= 0xD7):
            if marker == 0xD9:
                # EOI
                return info
            continue

        # Read segment length (includes the 2 bytes of length)
        var ok_len = False
        var seg_len = 0
        var next_i = i
        (ok_len, seg_len, next_i) = _read_be_u16(buf, i)
        if not ok_len or seg_len < 2:
            return info
        i = next_i
        var seg_end = i + (seg_len - 2)
        if seg_end > n:
            return info

        # APP0 (JFIF)
        if marker == 0xE0:
            # 'JFIF\0' at start?
            if (i + 5) <= seg_end:
                if buf[i] == UInt8(0x4A) and buf[i+1] == UInt8(0x46) and buf[i+2] == UInt8(0x49) and buf[i+3] == UInt8(0x46) and buf[i+4] == UInt8(0x00):
                    info.jfif = True

        # SOF*
        if _is_sof(marker):
            # SOF layout:
            # precision (1), height (2), width (2), num_components (1), then comps
            if i + 6 <= seg_end:
                info.precision = Int(buf[i+0])
                var ok1 = False; var h = 0; var j = i + 1
                (ok1, h, j) = _read_be_u16(buf, j)
                if not ok1:
                    return info
                var ok2 = False; var w = 0
                (ok2, w, j) = _read_be_u16(buf, j)
                if not ok2:
                    return info
                info.height = h
                info.width = w
                info.num_components = Int(buf[j])
                info.ok = True
                return info

        # advance to next marker
        i = seg_end

    return info

# -------------------------
# Placeholders for future codec wiring
# -------------------------
@staticmethod
fn decode_to_rgb_u8(buf: List[UInt8]) -> (Bool, ImageRGBU8):
    # Placeholder: real decoder not implemented.
    var img = ImageRGBU8(0, 0, List[UInt8]())
    return (False, img)

@staticmethod
fn encode_from_rgb_u8(img: ImageRGBU8, quality: Int) -> (Bool, List[UInt8]):
    # Placeholder: real encoder not implemented.
    return (False, List[UInt8]())

# -------------------------
# Minimal smoke test
# -------------------------
@staticmethod
fn __self_test__() -> Bool:
    # Build a tiny synthetic JPEG-like buffer:
    # SOI FF D8
    # APP0 FFE0 length=16 (0x0010) + 'JFIF\0' + 11 padding bytes
    # SOF0 FFC0 length=17 (0x0011): [precision=8][h=0x00 0x02][w=0x00 0x03][ncomp=3][3*3 bytes comps]
    # EOI FF D9
    var b: List[UInt8] = List[UInt8]()

    # SOI
    b.append(0xFF); b.append(0xD8)

    # APP0
    b.append(0xFF); b.append(0xE0)
    b.append(0x00); b.append(0x10)       # length 16
    b.append(0x4A); b.append(0x46); b.append(0x49); b.append(0x46); b.append(0x00) # 'JFIF\0'
    # padding (11 bytes total payload for len=16 includes the 2 length bytes; we've used 5, need 9 more)
    b.append(0x01); b.append(0x01); b.append(0x00); b.append(0x00); b.append(0x01); b.append(0x00); b.append(0x01); b.append(0x00); b.append(0x00)

    # SOF0
    b.append(0xFF); b.append(0xC0)
    b.append(0x00); b.append(0x11)       # length 17
    b.append(0x08)                       # precision 8
    b.append(0x00); b.append(0x02)       # height = 2
    b.append(0x00); b.append(0x03)       # width = 3
    b.append(0x03)                       # num components = 3
    # components (id, sampling, qtbl) * 3
    b.append(0x01); b.append(0x11); b.append(0x00)
    b.append(0x02); b.append(0x11); b.append(0x01)
    b.append(0x03); b.append(0x11); b.append(0x01)

    # EOI
    b.append(0xFF); b.append(0xD9)

    if not is_jpeg(b): return False
    var info = parse_info(b)
    if not info.ok: return False
    if not (info.jfif and info.precision == 8 and info.height == 2 and info.width == 3 and info.num_components == 3):
        return False

    # decode/encode placeholders
    var (ok_dec, img) = decode_to_rgb_u8(b)
    if ok_dec: return False  # should be False for placeholder

    var (ok_enc, buf2) = encode_from_rgb_u8(ImageRGBU8(2, 2, List[UInt8]()), 90)
    if ok_enc: return False  # placeholder should be False

    return True