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
# Project: momijo.vision.transforms
# File: src/momijo/vision/transforms/random_rotation.mojo

struct ChannelOrder(Copyable, Movable):
    var id: Int
fn __init__(out self, id: Int) -> None: self.id = id
    @staticmethod fn GRAY() -> ChannelOrder: return ChannelOrder(1)
    @staticmethod fn RGB()  -> ChannelOrder: return ChannelOrder(2)
    @staticmethod fn BGR()  -> ChannelOrder: return ChannelOrder(3)
    @staticmethod fn RGBA() -> ChannelOrder: return ChannelOrder(4)
    @staticmethod fn BGRA() -> ChannelOrder: return ChannelOrder(5)
fn __eq__(self, other: ChannelOrder) -> Bool: return self.id == other.id

@staticmethod
fn _num_ch(order: ChannelOrder) -> Int:
    if order == ChannelOrder.GRAY(): return 1
    if order == ChannelOrder.RGB() or order == ChannelOrder.BGR(): return 3
    return 4

struct Image(Copyable, Movable):
    var h: Int
    var w: Int
    var order: ChannelOrder
    var data: List[UInt8]
fn __init__(out self, h: Int, w: Int, order: ChannelOrder, data: List[UInt8]) -> None:
        self.h = h; self.w = w; self.order = order
        var expected = h*w*_num_ch(order)
        if len(data) != expected:
            var buf: List[UInt8] = List[UInt8]()
            var i = 0
            while i < expected: buf.append(0); i += 1
            self.data = buf
        else:
            self.data = data

# -------------------------
# Enums / Pad mode
# -------------------------
struct PadMode(Copyable, Movable):
    var id: Int
fn __init__(out self, id: Int) -> None: self.id = id
    @staticmethod fn ZERO() -> PadMode: return PadMode(0)
    @staticmethod fn EDGE() -> PadMode: return PadMode(1)
fn __eq__(self, other: PadMode) -> Bool: return self.id == other.id
fn to_string(self) -> String:
        if self.id == 0: return String("ZERO")
        if self.id == 1: return String("EDGE")
        return String("UNKNOWN")

# -------------------------
# Helpers
# -------------------------
@staticmethod
fn _offset(w: Int, c: Int, x: Int, y: Int, ch: Int) -> Int:
    return ((y * w) + x) * c + ch

@staticmethod
fn _alloc_u8(n: Int) -> List[UInt8]:
    var out: List[UInt8] = List[UInt8]()
    var i = 0
    while i < n: out.append(0); i += 1
    return out

@staticmethod
fn _clamp_i(v: Int, lo: Int, hi: Int) -> Int:
    if v < lo: return lo
    if v > hi: return hi
    return v

@staticmethod
fn _abs_i(v: Int) -> Int:
    if v < 0: return -v
    return v

# -------------------------
# Fixed-point trig (Q16)
# -------------------------
@staticmethod
fn _deg_to_rad_q16(deg: Int) -> Int:
    var pi_q16 = 205887  # round(pi * 65536)
    return (deg * pi_q16) // 180

@staticmethod
fn _wrap_pi_q16(x: Int) -> Int:
    var pi_q16 = 205887
    var tp = pi_q16 * 2
    var r = x % tp
    if r > pi_q16: r = r - tp
    if r < -pi_q16: r = r + tp
    return r

@staticmethod
fn _sin_q16(x_q16: Int) -> Int:

    var A = 83443    # 1.27323954 * 65536
    var B = 26519    # 0.405284735 * 65536
    var x = _wrap_pi_q16(x_q16)
    var term1 = (A * x) >> 16
    var term2 = (B * ((x * _abs_i(x)) >> 16)) >> 16
    return term1 - term2

@staticmethod
fn _cos_q16(x_q16: Int) -> Int:
    var p2 = 102943  # (pi/2)*65536
    return _sin_q16(x_q16 + p2)

# -------------------------
# Bilinear sampling helper
# -------------------------
@staticmethod
fn _sample_bilinear_u8_hwc(h: Int, w: Int, c: Int, src: List[UInt8], fx_q16: Int, fy_q16: Int, pad: PadMode) -> List[UInt8]:
    # returns c-length vector for one sample
    var out: List[UInt8] = List[UInt8]()
    var x = fx_q16 >> UInt8(16)
    var y = fy_q16 >> UInt8(16)

    if x < 0 or y < 0 or x >= w or y >= h:
        if pad == PadMode.ZERO():
            var i = 0
            while i < c: out.append(0); i += 1
            return out
        # EDGE pad: clamp coordinates and return nearest
        x = _clamp_i(x, 0, w-1)
        y = _clamp_i(y, 0, h-1)
        var ch = 0
        while ch < c:
            out.append(src[_offset(w, c, x, y, ch)])
            ch += 1
        return out

    var x1 = x + 1; if x1 >= w: x1 = w - 1
    var y1 = y + 1; if y1 >= h: y1 = h - 1

    var fx = fx_q16 & UInt8(0xFFFF)
    var fy = fy_q16 & UInt8(0xFFFF)
    var wx0 = 65536 - fx
    var wy0 = 65536 - fy

    var w00 = (wx0 * wy0) >> 16
    var w10 = (fx  * wy0) >> 16
    var w01 = (wx0 * fy)  >> 16
    var w11 = (fx  * fy)  >> 16

    var ch = 0
    while ch < c:
        var p00 = Int(src[_offset(w, c, x,  y,  ch)])
        var p10 = Int(src[_offset(w, c, x1, y,  ch)])
        var p01 = Int(src[_offset(w, c, x,  y1, ch)])
        var p11 = Int(src[_offset(w, c, x1, y1, ch)])
        var acc = p00*w00 + p10*w10 + p01*w01 + p11*w11
        var v = (acc + 32768) >> 16
        if v < 0: v = 0
        if v > 255: v = 255
        out.append(UInt8(v))
        ch += 1
    return out

# -------------------------
# Core rotate kernels
# -------------------------
@staticmethod
fn rotate_bilinear_u8_hwc_keep(h: Int, w: Int, c: Int, src: List[UInt8], deg: Int, pad: PadMode) -> List[UInt8]:
    # same size output, rotate around center
    if deg % 360 == 0:
        # copy
        var out0: List[UInt8] = List[UInt8](); var n = h*w*c; var i0 = 0
        while i0 < n: out0.append(src[i0]); i0 += 1
        return out0

    var ang = _deg_to_rad_q16(deg)
    var s = _sin_q16(ang)   # Q16
    var co = _cos_q16(ang)  # Q16

    var cx_q16 = ((w << UInt8(16)) - 65536) >> 1  # (w-1)/2 in Q16
    var cy_q16 = ((h << UInt8(16)) - 65536) >> 1

    var out = _alloc_u8(h*w*c)
    var y = 0
    while y < h:
        var dy_q16 = (y << UInt8(16)) - cy_q16
        var x = 0
        while x < w:
            var dx_q16 = (x << UInt8(16)) - cx_q16
            # inverse map
            var sx_q16 = ((co * dx_q16) >> 16) + ((s * dy_q16) >> 16) + cx_q16
            var sy_q16 = ((-s * dx_q16) >> 16) + ((co * dy_q16) >> 16) + cy_q16
            var sample = _sample_bilinear_u8_hwc(h, w, c, src, sx_q16, sy_q16, pad)
            var ch = 0
            while ch < c:
                out[_offset(w, c, x, y, ch)] = sample[ch]
                ch += 1
            x += 1
        y += 1
    return out

@staticmethod
fn _rotated_size(h: Int, w: Int, ang_q16: Int) -> (Int, Int):
    var s = _sin_q16(ang_q16)   # Q16
    var c = _cos_q16(ang_q16)   # Q16
    var aw = _abs_i((w * c) >> 16) + _abs_i((h * s) >> 16)
    var ah = _abs_i((h * c) >> 16) + _abs_i((w * s) >> 16)
    if aw < 1: aw = 1
    if ah < 1: ah = 1
    return (ah, aw)

@staticmethod
fn rotate_bilinear_u8_hwc_auto(h: Int, w: Int, c: Int, src: List[UInt8], deg: Int, pad: PadMode) -> (Int, Int, List[UInt8]):
    if deg % 360 == 0:
        var copy: List[UInt8] = List[UInt8](); var n = h*w*c; var i0 = 0
        while i0 < n: copy.append(src[i0]); i0 += 1
        return (h, w, copy)

    var ang = _deg_to_rad_q16(deg)
    var s = _sin_q16(ang)   # Q16
    var co = _cos_q16(ang)  # Q16

    var oh = 0; var ow = 0
    (oh, ow) = _rotated_size(h, w, ang)

    var cx_src_q16 = ((w << UInt8(16)) - 65536) >> 1
    var cy_src_q16 = ((h << UInt8(16)) - 65536) >> 1
    var cx_dst_q16 = ((ow << UInt8(16)) - 65536) >> 1
    var cy_dst_q16 = ((oh << UInt8(16)) - 65536) >> 1

    var out = _alloc_u8(oh*ow*c)
    var y = 0
    while y < oh:
        var dy_q16 = (y << UInt8(16)) - cy_dst_q16
        var x = 0
        while x < ow:
            var dx_q16 = (x << UInt8(16)) - cx_dst_q16
            # map dest->src
            var sx_q16 = ((co * dx_q16) >> 16) + ((s * dy_q16) >> 16) + cx_src_q16
            var sy_q16 = ((-s * dx_q16) >> 16) + ((co * dy_q16) >> 16) + cy_src_q16
            var sample = _sample_bilinear_u8_hwc(h, w, c, src, sx_q16, sy_q16, pad)
            var ch = 0
            while ch < c:
                out[_offset(ow, c, x, y, ch)] = sample[ch]
                ch += 1
            x += 1
        y += 1
    return (oh, ow, out)

# -------------------------
# RNG + random API
# -------------------------
struct LcgRng(Copyable, Movable):
    var state: Int
fn __init__(out self, seed: Int) -> None:
        if seed == 0: seed = 1
        self.state = seed & UInt8(0x7FFFFFFF)
fn next_u32(mut self) -> (LcgRng, Int):
        var a = 1664525
        var c = 1013904223
        self.state = (a * self.state + c) & 0x7FFFFFFF
        return (self, self.state)
fn uniform_int(mut self, lo: Int, hi: Int) -> (LcgRng, Int):
        var r = 0; (self, r) = self.next_u32()
        if hi <= lo: return (self, lo)
        var span = hi - lo + 1
        var v = lo + (r % span)
        return (self, v)

@staticmethod
fn apply_random_rotation(img: Image, seed: Int, min_deg: Int, max_deg: Int, pad: PadMode, keep_size: Int) -> (Image, Int):
    var rng = LcgRng(seed)
    var deg = 0; (rng, deg) = rng.uniform_int(min_deg, max_deg)
    var c = _num_ch(img.order)
    if keep_size != 0:
        var buf = rotate_bilinear_u8_hwc_keep(img.h, img.w, c, img.data, deg, pad)
        return (Image(img.h, img.w, img.order, buf), deg)
    var oh = 0; var ow = 0; var out_buf: List[UInt8] = List[UInt8]()
    (oh, ow, out_buf) = rotate_bilinear_u8_hwc_auto(img.h, img.w, c, img.data, deg, pad)
    return (Image(oh, ow, img.order, out_buf), deg)

# -------------------------
# Minimal smoke test
# -------------------------
@staticmethod
fn __self_test__() -> Bool:
    # Build 3x3 RGB with distinct pattern
    var h = 3; var w = 3
    var data: List[UInt8] = List[UInt8]()
    var y = 0
    while y < h:
        var x = 0
        while x < w:
            data.append(UInt8(x*20))
            data.append(UInt8(y*30))
            data.append(UInt8((x+y)*10))
            x += 1
        y += 1
    var img = Image(h, w, ChannelOrder.RGB(), data)

    # 1) deg=0 keep-size -> must equal input
    var buf0 = rotate_bilinear_u8_hwc_keep(h,w,3, data, 0, PadMode.ZERO())
    if len(buf0) != len(data): return False
    var i = 0
    while i < len(data):
        if buf0[i] != data[i]: return False
        i += 1

    # 2) small angle auto-size -> output dims >= input dims occasionally; just check buffer size matches
    var oh = 0; var ow = 0; var b: List[UInt8] = List[UInt8]()
    (oh, ow, b) = rotate_bilinear_u8_hwc_auto(h,w,3, data, 15, PadMode.EDGE())
    if not (len(b) == oh*ow*3 and oh >= 3 and ow >= 3): return False

    # 3) random API reproducibility for fixed seed ranges
    var out1: Image; var d1 = 0
    (out1, d1) = apply_random_rotation(img, 1234, -10, 10, PadMode.EDGE(), 1)
    var out2: Image; var d2 = 0
    (out2, d2) = apply_random_rotation(img, 1234, -10, 10, PadMode.EDGE(), 1)
    if d1 != d2: return False
    if not (len(out1.data) == len(out2.data) and out1.h == out2.h and out1.w == out2.w): return False
    i = 0
    while i < len(out1.data):
        if out1.data[i] != out2.data[i]: return False
        i += 1

    return True