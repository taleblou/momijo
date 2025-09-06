# MIT License
# Copyright (c) 2025 Morteza Talebou and Mitra Daneshmand
# Project: momijo  |  Source: https://github.com/taleblou/momijo
# This file is part of the Momijo project. See the LICENSE file at the repository root.
# Momijo
# SPDX-License-Identifier: MIT
# Website: https://taleblou.ir/
# Repository: https://github.com/taleblou/momijo
#
# Project: momijo.vision.io
# File: src/momijo/vision/io/registry.mojo
#
# Minimal, dependency-light IO registry for Momijo Vision.
# Goals:
#   - Detect basic image formats from bytes (PPM P6/P3, PNG, JPEG)
#   - Provide unified decoding entry: read_image_from_bytes -> TensorU8HWC
#   - Provide extension-based dispatcher (path-only placeholder; no FS access here)
# Style:
#   - No 'export', no 'let', no 'inout'
#   - Constructors use `fn __init__(out self, ...)`
#
# NOTE: This file is standalone and does not import other modules. It includes
#       tiny decoders for PPM (P6/P3). PNG/JPEG decoding are left as placeholders
#       (metadata detection only), matching the minimal scope of this phase.

# -------------------------
# Minimal Tensor (U8/HWC)
# -------------------------
struct TensorU8HWC(Copyable, Movable):
    var h: Int
    var w: Int
    var c: Int
    var data: List[UInt8]

    fn __init__(out self, h: Int, w: Int, c: Int, data: List[UInt8]):
        self.h = h; self.w = w; self.c = c
        var expected = h * w * c
        if len(data) != expected:
            var buf: List[UInt8] = List[UInt8]()
            var i = 0
            while i < expected:
                buf.append(0)
                i += 1
            self.data = buf
        else:
            self.data = data

    @staticmethod
    fn zeros(h: Int, w: Int, c: Int) -> TensorU8HWC:
        var total = h * w * c
        var buf: List[UInt8] = List[UInt8]()
        var i = 0
        while i < total:
            buf.append(0)
            i += 1
        return TensorU8HWC(h, w, c, buf)

# -------------------------
# Common helpers
# -------------------------
@staticmethod
fn _is_ws(b: UInt8) -> Bool:
    return b == UInt8(32) or b == UInt8(9) or b == UInt8(13) or b == UInt8(10)

@staticmethod
fn _skip_ws_and_comments(buf: List[UInt8], i: Int) -> Int:
    var n = len(buf)
    var p = i
    while p < n:
        var b = buf[p]
        if b == UInt8(35):  # '#'
            p += 1
            while p < n and not (buf[p] == UInt8(10)):
                p += 1
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
        if b >= UInt8(48) and b <= UInt8(57):
            v = v * 10 + (Int(b) - 48)
            p += 1
            saw = True
        else:
            break
    if not saw:
        return (False, 0, p)
    return (True, v, p)

# -------------------------
# Signatures
# -------------------------
@staticmethod
fn _is_ppm(buf: List[UInt8]) -> Bool:
    if len(buf) < 2: return False
    if buf[0] != UInt8(80): return False  # 'P'
    return buf[1] == UInt8(54) or buf[1] == UInt8(51)  # '6' or '3'

@staticmethod
fn _is_png(buf: List[UInt8]) -> Bool:
    if len(buf) < 8: return False
    return (buf[0] == UInt8(0x89) and buf[1] == UInt8(0x50) and buf[2] == UInt8(0x4E) and buf[3] == UInt8(0x47) and
            buf[4] == UInt8(0x0D) and buf[5] == UInt8(0x0A) and buf[6] == UInt8(0x1A) and buf[7] == UInt8(0x0A))

@staticmethod
fn _is_jpeg(buf: List[UInt8]) -> Bool:
    if len(buf) < 2: return False
    return buf[0] == UInt8(0xFF) and buf[1] == UInt8(0xD8)

# -------------------------
# PPM decoding (P6/P3)
# -------------------------
struct _PpmInfo(Copyable, Movable):
    var ok: Bool
    var magic: Int   # 6 or 3
    var w: Int
    var h: Int
    var maxval: Int
    var data_offset: Int
    fn __init__(out self):
        self.ok = False; self.magic = 0; self.w = 0; self.h = 0; self.maxval = 0; self.data_offset = 0

@staticmethod
fn _ppm_parse(buf: List[UInt8]) -> _PpmInfo:
    var info = _PpmInfo()
    var n = len(buf)
    if n < 2 or buf[0] != UInt8(80): return info
    var magic = 0
    if buf[1] == UInt8(54): magic = 6
    elif buf[1] == UInt8(51): magic = 3
    else: return info
    var p = 2
    var ok = False; var w = 0; var h = 0; var mv = 0
    (ok, w, p) = _read_uint(buf, p); if not ok: return info
    (ok, h, p) = _read_uint(buf, p); if not ok: return info
    (ok, mv, p) = _read_uint(buf, p); if not ok: return info
    p = _skip_ws_and_comments(buf, p)
    info.ok = True; info.magic = magic; info.w = w; info.h = h; info.maxval = mv; info.data_offset = p
    return info

@staticmethod
fn _scale_sample(v: Int, maxval: Int) -> UInt8:
    if maxval <= 0: return UInt8(0)
    if maxval == 255: return UInt8(v & 0xFF)
    var num = v * 255 + (maxval // 2)
    var s = num // maxval
    if s < 0: return UInt8(0)
    if s > 255: return UInt8(255)
    return UInt8(s)

@staticmethod
fn _decode_ppm(buf: List[UInt8]) -> (Bool, TensorU8HWC):
    var info = _ppm_parse(buf)
    if not info.ok or info.w <= 0 or info.h <= 0 or info.maxval <= 0:
        return (False, TensorU8HWC.zeros(0, 0, 3))
    if info.magic == 6:
        if info.maxval > 255: return (False, TensorU8HWC.zeros(0,0,3))
        var expected = info.w * info.h * 3
        if info.data_offset + expected > len(buf): return (False, TensorU8HWC.zeros(0,0,3))
        var out: List[UInt8] = List[UInt8]()
        var i = 0
        while i < expected:
            out.append(_scale_sample(Int(buf[info.data_offset + i]), info.maxval))
            i += 1
        return (True, TensorU8HWC(info.h, info.w, 3, out))
    # P3
    var total = info.w * info.h * 3
    var out3: List[UInt8] = List[UInt8]()
    var p = info.data_offset
    var cnt = 0
    while cnt < total:
        var okn = False; var val = 0
        (okn, val, p) = _read_uint(buf, p)
        if not okn: return (False, TensorU8HWC.zeros(0,0,3))
        out3.append(_scale_sample(val, info.maxval))
        cnt += 1
    return (True, TensorU8HWC(info.h, info.w, 3, out3))

# -------------------------
# Unified detect & decode from bytes
# -------------------------
@staticmethod
fn detect_format(buf: List[UInt8]) -> String:
    if _is_ppm(buf): return String("ppm")
    if _is_png(buf): return String("png")
    if _is_jpeg(buf): return String("jpeg")
    return String("unknown")

@staticmethod
fn read_image_from_bytes(buf: List[UInt8]) -> (Bool, TensorU8HWC):
    if _is_ppm(buf):
        return _decode_ppm(buf)
    if _is_png(buf):
        # Placeholder: PNG decode not yet implemented in minimal registry
        return (False, TensorU8HWC.zeros(0,0,4))
    if _is_jpeg(buf):
        # Placeholder: JPEG decode not yet implemented
        return (False, TensorU8HWC.zeros(0,0,3))
    return (False, TensorU8HWC.zeros(0,0,3))

# -------------------------
# Extension-based dispatcher (no filesystem access here)
# -------------------------
@staticmethod
fn read_image(path: String) -> (Bool, TensorU8HWC):
    # Placeholder: no disk IO in this minimal registry. Returns False.
    return (False, TensorU8HWC.zeros(0,0,3))

# -------------------------
# Minimal smoke test
# -------------------------
@staticmethod
fn __self_test__() -> Bool:
    # Tiny P6 buffer: 2x1 pixels (255,0,0) (0,255,0)
    var p6: List[UInt8] = List[UInt8]()
    p6.append(80); p6.append(54); p6.append(10)   # P6\n
    p6.append(50); p6.append(32); p6.append(49); p6.append(10)  # "2 1\n"
    p6.append(50); p6.append(53); p6.append(53); p6.append(10)  # "255\n"
    p6.append(255); p6.append(0); p6.append(0)    # R,G,B
    p6.append(0);   p6.append(255); p6.append(0)

    if detect_format(p6) != String("ppm"): return False
    var (ok, t) = read_image_from_bytes(p6)
    if not ok: return False
    if not (t.h == 1 and t.w == 2 and t.c == 3): return False
    if not (t.data[0] == 255 and t.data[1] == 0 and t.data[2] == 0): return False

    # PNG/JPEG signatures should be detected but decode return False
    var png_sig: List[UInt8] = List[UInt8]()
    png_sig.append(0x89); png_sig.append(0x50); png_sig.append(0x4E); png_sig.append(0x47)
    png_sig.append(0x0D); png_sig.append(0x0A); png_sig.append(0x1A); png_sig.append(0x0A)
    if detect_format(png_sig) != String("png"): return False
    var (okp, _) = read_image_from_bytes(png_sig)
    if okp: return False

    var jpg_sig: List[UInt8] = List[UInt8]()
    jpg_sig.append(0xFF); jpg_sig.append(0xD8)
    if detect_format(jpg_sig) != String("jpeg"): return False
    var (okj, _) = read_image_from_bytes(jpg_sig)
    if okj: return False

    return True
