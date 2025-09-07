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
# File: src/momijo/vision/io/ppm.mojo

from momijo.core.error import unsupported
from momijo.dataframe.helpers import read
from momijo.nn.parameter import data
from momijo.tensor.tensor import index
from momijo.utils.timer import start
from momijo.vision.schedule.schedule import end
from pathlib import Path
from pathlib.path import Path

struct ImageRGBU8(Copyable, Movable):
    var h: Int
    var w: Int
    var data: List[UInt8]   # packed RGB, HWC
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
# PPM Info
# -------------------------
struct PpmInfo(Copyable, Movable):
    var ok: Bool
    var magic: Int          # 6 for P6, 3 for P3, 0 otherwise
    var width: Int
    var height: Int
    var maxval: Int
    var data_offset: Int    # index in the buffer where pixel data starts (P6) or first number (P3)
fn __init__(out self) -> None:
        self.ok = False
        self.magic = 0
        self.width = 0
        self.height = 0
        self.maxval = 0
        self.data_offset = 0

# -------------------------
# Helpers
# -------------------------
@staticmethod
fn _is_ws(b: UInt8) -> Bool:
    # space, tab, CR, LF
    return b == UInt8(32) or b == UInt8(9) or b == UInt8(13) or b == UInt8(10)

@staticmethod
fn _skip_ws_and_comments(buf: List[UInt8], i: Int) -> Int:
    var n = len(buf)
    var p = i
    while p < n:
        var b = buf[p]
        if b == UInt8(35):  # '#'
            # skip to end of line
            p += 1
            while p < n and not (buf[p] == UInt8(10)):  # newline
                p += 1
            # consume newline if present
            if p < n: p += 1
            continue
        if _is_ws(b):
            p += 1
            continue
        break
    return p

@staticmethod
fn _read_uint(buf: List[UInt8], i: Int) -> (Bool, Int, Int):
    var n = len(buf)
    var p = _skip_ws_and_comments(buf, i)
    if p >= n:
        return (False, 0, p)
    var v = 0
    var saw = False
    while p < n:
        var b = buf[p]
        if b >= UInt8(48) and b <= UInt8(57):  # '0'..'9'
            v = v * 10 + (Int(b) - 48)
            p += 1
            saw = True
        else:
            break
    if not saw:
        return (False, 0, p)
    return (True, v, p)

@staticmethod
fn _scale_sample(v: Int, maxval: Int) -> UInt8:
    if maxval <= 0: return UInt8(0)
    if maxval == 255: return UInt8(v & UInt8(0xFF))
    # scale to 0..255 with rounding
    var num = v * 255 + (maxval // 2)
    var val = num // maxval
    if val < 0: return UInt8(0)
    if val > 255: return UInt8(255)
    return UInt8(val)

@staticmethod
fn _append_byte(mut out: List[UInt8], b: UInt8) -> List[UInt8]:
    out.append(b)
    return out

@staticmethod
fn _append_uint_ascii(mut out: List[UInt8], v: Int) -> List[UInt8]:
    if v == 0:
        out.append(UInt8(48))
        return out
    var tmp: List[UInt8] = List[UInt8]()
    var x = v
    while x > 0:
        var d = x % 10
        tmp.append(UInt8(48 + d))
        x = x // 10
    # reverse
    var i = len(tmp) - 1
    while i >= 0:
        out.append(tmp[i])
        i -= 1
    return out

# -------------------------
# Signature
# -------------------------
@staticmethod
fn is_ppm(buf: List[UInt8]) -> Bool:
    if len(buf) < 2: return False
    if not (buf[0] == UInt8(80)):  # 'P'
        return False
    return buf[1] == UInt8(54) or buf[1] == UInt8(51)  # '6' or '3'

# -------------------------
# Parse metadata
# -------------------------
@staticmethod
fn parse_info(buf: List[UInt8]) -> PpmInfo:
    var info = PpmInfo()
    var n = len(buf)
    if n < 2: return info
    if buf[0] != UInt8(80): return info  # 'P'
    var magic = 0
    if buf[1] == UInt8(54): magic = 6
    elif buf[1] == UInt8(51): magic = 3
    else: return info
    var p = 2
    # read width, height, maxval
    var ok = False; var w = 0; var h = 0; var mv = 0
    (ok, w, p) = _read_uint(buf, p); if not ok: return info
    (ok, h, p) = _read_uint(buf, p); if not ok: return info
    (ok, mv, p) = _read_uint(buf, p); if not ok: return info
    # For P6, next non-ws/comment should be pixel data start
    p = _skip_ws_and_comments(buf, p)
    info.ok = True
    info.magic = magic
    info.width = w
    info.height = h
    info.maxval = mv
    info.data_offset = p
    return info

# -------------------------
# Decode
# -------------------------
@staticmethod
fn decode_to_rgb_u8(buf: List[UInt8]) -> (Bool, ImageRGBU8):
    var info = parse_info(buf)
    if not info.ok:
        return (False, ImageRGBU8(0, 0, List[UInt8]()))
    if info.width <= 0 or info.height <= 0:
        return (False, ImageRGBU8(0, 0, List[UInt8]()))
    if info.maxval <= 0:
        return (False, ImageRGBU8(0, 0, List[UInt8]()))

    if info.magic == 6:
        # Binary
        if info.maxval > 255:
            # 2-byte samples unsupported in this minimal impl
            return (False, ImageRGBU8(0, 0, List[UInt8]()))
        var expected = info.width * info.height * 3
        var start = info.data_offset
        if start + expected > len(buf):
            return (False, ImageRGBU8(0, 0, List[UInt8]()))
        var out: List[UInt8] = List[UInt8]()
        var i = 0
        while i < expected:
            var v = Int(buf[start + i])
            out.append(_scale_sample(v, info.maxval))
            i += 1
        return (True, ImageRGBU8(info.height, info.width, out))

    # ASCII P3
    var out3: List[UInt8] = List[UInt8]()
    var total = info.width * info.height * 3
    var p = info.data_offset
    var count = 0
    while count < total:
        var okn = False; var val = 0
        (okn, val, p) = _read_uint(buf, p)
        if not okn:
            return (False, ImageRGBU8(0, 0, List[UInt8]()))
        out3.append(_scale_sample(val, info.maxval))
        count += 1
    return (True, ImageRGBU8(info.height, info.width, out3))

# -------------------------
# Encode (P6)
# -------------------------
@staticmethod
fn encode_p6_from_rgb_u8(img: ImageRGBU8, maxval: Int) -> (Bool, List[UInt8]):
    if img.h <= 0 or img.w <= 0:
        return (False, List[UInt8]())
    if maxval != 255:
        # For simplicity, only 255 supported
        return (False, List[UInt8]())
    var out: List[UInt8] = List[UInt8]()
    # Header: P6\n<width> <height>\n<maxval>\n
    out = _append_byte(out, UInt8(80))  # 'P'
    out = _append_byte(out, UInt8(54))  # '6'
    out = _append_byte(out, UInt8(10))  # \n
    out = _append_uint_ascii(out, img.w)
    out = _append_byte(out, UInt8(32))  # space
    out = _append_uint_ascii(out, img.h)
    out = _append_byte(out, UInt8(10))  # \n
    out = _append_uint_ascii(out, 255)
    out = _append_byte(out, UInt8(10))  # \n

    # Data
    var expected = img.h * img.w * 3
    var i = 0
    while i < expected:
        out.append(img.data[i])
        i += 1
    return (True, out)

# -------------------------
# Minimal smoke test
# -------------------------
@staticmethod
fn __self_test__() -> Bool:
    # Build a tiny P6: 2x1, max=255, pixels: (255,0,0) (0,255,0)
    var p6: List[UInt8] = List[UInt8]()
    # Header: P6\n2 1\n255\n
    p6.append(80); p6.append(54); p6.append(10)   # P6\n
    p6.append(50); p6.append(32); p6.append(49); p6.append(10)  # "2 1\n"
    p6.append(50); p6.append(53); p6.append(53); p6.append(10)  # "255\n"
    # Data: R,G,B | R,G,B
    p6.append(255); p6.append(0); p6.append(0)
    p6.append(0);   p6.append(255); p6.append(0)

    if not is_ppm(p6): return False
    var info6 = parse_info(p6)
    if not (info6.ok and info6.magic == 6 and info6.width == 2 and info6.height == 1 and info6.maxval == 255): return False
    var (ok6, img6) = decode_to_rgb_u8(p6)
    if not ok6: return False
    if not (img6.h == 1 and img6.w == 2 and len(img6.data) == 6): return False
    if not (img6.data[0] == 255 and img6.data[1] == 0 and img6.data[2] == 0): return False

    # Build a tiny P3: 2x1, max=255, pixels: (10,20,30) (40,50,60) with comments
    var p3: List[UInt8] = List[UInt8]()
    # "P3\n# comment\n2 1\n255\n10 20 30  # px0\n40 50 60\n"
    p3.append(80); p3.append(51); p3.append(10)  # P3\n
    p3.append(35); p3.append(32); p3.append(99); p3.append(111); p3.append(109); p3.append(109); p3.append(101); p3.append(110); p3.append(116); p3.append(10)  # "# comment\n"
    p3.append(50); p3.append(32); p3.append(49); p3.append(10)  # "2 1\n"
    p3.append(50); p3.append(53); p3.append(53); p3.append(10)  # "255\n"
    # numbers with a trailing comment
    # "10 20 30  # px0\n40 50 60\n"
    p3.append(49); p3.append(48); p3.append(32); p3.append(50); p3.append(48); p3.append(32); p3.append(51); p3.append(48); p3.append(32)
    p3.append(35); p3.append(32); p3.append(112); p3.append(120); p3.append(48); p3.append(10)
    p3.append(52); p3.append(48); p3.append(32); p3.append(53); p3.append(48); p3.append(32); p3.append(54); p3.append(48); p3.append(10)

    var info3 = parse_info(p3)
    if not (info3.ok and info3.magic == 3 and info3.width == 2 and info3.height == 1 and info3.maxval == 255): return False
    var (ok3, img3) = decode_to_rgb_u8(p3)
    if not ok3: return False
    if not (img3.data[0] == 10 and img3.data[1] == 20 and img3.data[2] == 30 and img3.data[3] == 40 and img3.data[4] == 50 and img3.data[5] == 60): return False

    # Encode test (P6)
    var img = ImageRGBU8(1, 2, img6.data)
    var (ok_enc, enc) = encode_p6_from_rgb_u8(img, 255)
    if not ok_enc: return False
    var info_enc = parse_info(enc)
    if not (info_enc.ok and info_enc.magic == 6 and info_enc.width == 2 and info_enc.height == 1 and info_enc.maxval == 255): return False

    return True