# Project:      Momijo
# Module:       src.momijo.vision.transforms.color_jitter
# File:         color_jitter.mojo
# Path:         src/momijo/vision/transforms/color_jitter.mojo
#
# Description:  src.momijo.vision.transforms.color_jitter — focused Momijo functionality with a stable public API.
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
#   - Structs: ChannelOrder, Image, LcgRng, JitterRanges, JitterParams
#   - Key functions: __init__, GRAY, RGB, BGR, RGBA, BGRA, __eq__, _num_ch ...
#   - Static methods present.
#   - Uses generic functions/types with explicit trait bounds.


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
        var expected = h * w * _num_ch(order)
        if len(data) != expected:
            var buf: List[UInt8] = List[UInt8]()
            var i = 0
            while i < expected:
                buf.append(0); i += 1
            self.data = buf
        else:
            self.data = data

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
    while i < n:
        out.append(0)
        i += 1
    return out

@staticmethod
fn _clamp_u8(v: Int) -> UInt8:
    if v < 0: return UInt8(0)
    if v > 255: return UInt8(255)
    return UInt8(v)

# -------------------------
# Brightness / Contrast / Saturation
# -------------------------
@staticmethod
fn adjust_brightness_u8_hwc(h: Int, w: Int, c: Int, src: List[UInt8], delta: Int) -> List[UInt8]:
    if h <= 0 or w <= 0 or c <= 0:
        return List[UInt8]()
    var out = _alloc_u8(h * w * c)
    var i = 0
    while i < h * w:
        var chn = 0
        while chn < c:
            # keep alpha (if last channel and c==4)
            if c == 4 and chn == 3:
                out[i*c + chn] = src[i*c + chn]
            else:
                var v = Int(src[i*c + chn]) + delta
                out[i*c + chn] = _clamp_u8(v)
            chn += 1
        i += 1
    return out

@staticmethod
fn adjust_contrast_u8_hwc(h: Int, w: Int, c: Int, src: List[UInt8], factor_q8: Int) -> List[UInt8]:
    # factor_q8: 256 == 1.0; >256 increases contrast around 128.
    if h <= 0 or w <= 0 or c <= 0:
        return List[UInt8]()
    var out = _alloc_u8(h * w * c)
    var i = 0
    while i < h * w:
        var chn = 0
        while chn < c:
            if c == 4 and chn == 3:
                out[i*c + chn] = src[i*c + chn]
            else:
                var x = Int(src[i*c + chn])
                var y = ((x - 128) * factor_q8) >> 8
                var v = y + 128
                out[i*c + chn] = _clamp_u8(v)
            chn += 1
        i += 1
    return out

@staticmethod
fn adjust_saturation_u8_hwc(h: Int, w: Int, c: Int, src: List[UInt8], factor_q8: Int) -> List[UInt8]:
    # factor_q8: 0 -> grayscale, 256 -> no change, >256 -> boost
    if h <= 0 or w <= 0 or c <= 0:
        return List[UInt8]()
    var out = _alloc_u8(h * w * c)
    var i = 0
    while i < h * w:
        # fetch RGB (or single-channel replicate)
        var r = 0; var g = 0; var b = 0; var a = 255
        if c == 1:
            var v = Int(src[i])
            r = v; g = v; b = v
        else:
            r = Int(src[i*c + 0])
            g = Int(src[i*c + 1])
            b = Int(src[i*c + 2])
            if c == 4: a = Int(src[i*c + 3])
        # gray = 0.299 0.587 0.114 with integer weights (77,150,29)/256
        var gray = (77*r + 150*g + 29*b) >> 8
        var rr = gray + (((r - gray) * factor_q8) >> 8)
        var gg = gray + (((g - gray) * factor_q8) >> 8)
        var bb = gray + (((b - gray) * factor_q8) >> 8)
        if c == 1:
            out[i] = _clamp_u8(rr)
        elif c == 3:
            out[i*3 + 0] = _clamp_u8(rr)
            out[i*3 + 1] = _clamp_u8(gg)
            out[i*3 + 2] = _clamp_u8(bb)
        else:
            out[i*4 + 0] = _clamp_u8(rr)
            out[i*4 + 1] = _clamp_u8(gg)
            out[i*4 + 2] = _clamp_u8(bb)
            out[i*4 + 3] = _clamp_u8(a)
        i += 1
    return out

# -------------------------
# Hue via YIQ rotation (fixed-point)
# -------------------------
# Fixed-point scales:
#   Q14 for YIQ coefficients (const * 16384)
#   Q16.16 for sin/cos
#
@staticmethod
fn _deg_to_rad_q16(deg: Int) -> Int:
    # rad = deg * pi / 180; pi_q16 = round(pi * 65536)
    var pi_q16 = 205887  # 3.14159265 * 65536
    return (deg * pi_q16) // 180

@staticmethod
fn _wrap_pi(x_q16: Int) -> Int:
    # wrap to [-pi, pi] in Q16
    var pi_q16 = 205887
    var two_pi = pi_q16 * 2
    var r = x_q16 % two_pi
    if r > pi_q16:
        r = r - two_pi
    if r < -pi_q16:
        r = r + two_pi
    return r

@staticmethod
fn _abs_i(v: Int) -> Int:
    if v < 0: return -v
    return v

@staticmethod
fn _sin_q16(x_q16: Int) -> Int:


    var A = 83443    # round(1.27323954 * 65536)
    var B = 26519    # round(0.405284735 * 65536)
    var x = _wrap_pi(x_q16)
    var term1 = (A * x) >> 16
    var term2 = (B * ((x * _abs_i(x)) >> 16)) >> 16
    return term1 - term2

@staticmethod
fn _cos_q16(x_q16: Int) -> Int:
    # cos(x) = sin(x + pi/2)
    var pi_over_2 = 102943  # (pi/2)*65536
    return _sin_q16(x_q16 + pi_over_2)

@staticmethod
fn adjust_hue_u8_hwc(h: Int, w: Int, c: Int, src: List[UInt8], deg: Int) -> List[UInt8]:
    if h <= 0 or w <= 0 or c <= 0:
        return List[UInt8]()
    # Early out if deg==0
    if deg == 0:
        var copy: List[UInt8] = List[UInt8]()
        var n = h*w*c; var i0 = 0
        while i0 < n:
            copy.append(src[i0]); i0 += 1
        return copy

    var ang = _deg_to_rad_q16(deg)
    var s = _sin_q16(ang)  # Q16
    var co = _cos_q16(ang) # Q16

    # Q14 constants
    var YR = 4905   # 0.299 * 16384
    var YG = 9629   # 0.587 * 16384
    var YB = 1860   # 0.114 * 16384
    var IR = 9761   # 0.596 * 16384
    var IG = -4483  # -0.274 * 16384
    var IB = -5272  # -0.322 * 16384
    var QR = 3452   # 0.211 * 16384
    var QG = -8567  # -0.523 * 16384
    var QB = 5118   # 0.312 * 16384

    var RY = 15678  # 0.956 * 16384
    var RQ = 10180  # 0.621 * 16384
    var GY = 16384  # 1.000 * 16384 (approx for Y term)
    var GI = -4461  # -0.272 * 16384
    var GQ = -10603 # -0.647 * 16384
    var BY = 16384  # 1.000 * 16384
    var BI = -18125 # -1.106 * 16384
    var BQ = 27904  # 1.703 * 16384

    var out = _alloc_u8(h * w * c)
    var i = 0
    while i < h * w:
        var r = 0; var g = 0; var b = 0; var a = 255
        if c == 1:
            var v = Int(src[i]); r = v; g = v; b = v
        else:
            r = Int(src[i*c + 0])
            g = Int(src[i*c + 1])
            b = Int(src[i*c + 2])
            if c == 4: a = Int(src[i*c + 3])

        # YIQ in Q14
        var Y = YR*r + YG*g + YB*b
        var I = IR*r + IG*g + IB*b
        var Q = QR*r + QG*g + QB*b

        # rotate (I,Q): I' = I*co - Q*s ; Q' = I*s + Q*co    (co,s in Q16, I,Q in Q14)
        var I2 = ((I * co) >> 16) - ((Q * s) >> 16)
        var Q2 = ((I * s) >> 16) + ((Q * co) >> 16)

        # back to RGB: r = Y + 0.956*I2 + 0.621*Q2  (all in Q14)
        var r14 = Y + (RY * I2 >> UInt8(14)) + (RQ * Q2 >> UInt8(14))
        var g14 = (GY * (Y >> UInt8(14)) << 14) + (GI * I2 >> UInt8(14)) + (GQ * Q2 >> UInt8(14))  # keep Y scale
        var b14 = (BY * (Y >> UInt8(14)) << 14) + (BI * I2 >> UInt8(14)) + (BQ * Q2 >> UInt8(14))

        # Convert Q14 to Int
        var rr = r14 >> UInt8(14)
        var gg = g14 >> UInt8(14)
        var bb = b14 >> UInt8(14)

        if c == 1:
            var gy = (77*rr + 150*gg + 29*bb) >> 8
            out[i] = _clamp_u8(gy)
        elif c == 3:
            out[i*3 + 0] = _clamp_u8(rr)
            out[i*3 + 1] = _clamp_u8(gg)
            out[i*3 + 2] = _clamp_u8(bb)
        else:
            out[i*4 + 0] = _clamp_u8(rr)
            out[i*4 + 1] = _clamp_u8(gg)
            out[i*4 + 2] = _clamp_u8(bb)
            out[i*4 + 3] = _clamp_u8(a)
        i += 1
    return out

# -------------------------
# Combined pipeline
# -------------------------
@staticmethod
fn apply_color_jitter_u8_hwc(h: Int, w: Int, c: Int, src: List[UInt8],
                             brightness_delta: Int, contrast_q8: Int, saturation_q8: Int, hue_deg: Int) -> List[UInt8]:
    var buf = src
    if brightness_delta != 0:
        buf = adjust_brightness_u8_hwc(h, w, c, buf, brightness_delta)
    if contrast_q8 != 256:
        buf = adjust_contrast_u8_hwc(h, w, c, buf, contrast_q8)
    if saturation_q8 != 256:
        buf = adjust_saturation_u8_hwc(h, w, c, buf, saturation_q8)
    if hue_deg != 0:
        buf = adjust_hue_u8_hwc(h, w, c, buf, hue_deg)
    return buf

# -------------------------
# Simple RNG (LCG) + param sampling
# -------------------------
struct LcgRng(Copyable, Movable):
    var state: Int
fn __init__(out self, seed: Int) -> None:
        if seed == 0: seed = 1
        self.state = seed
fn next_u32(mut self) -> (LcgRng, Int):
        # LCG parameters (Numerical Recipes)
        var a = 1664525
        var c = 1013904223
        self.state = (a * self.state + c) & 0x7FFFFFFF
        return (self, self.state)
fn uniform_int(mut self, lo: Int, hi: Int) -> (LcgRng, Int):
        var s = 0; (self, s) = self.next_u32()
        var span = hi - lo + 1
        var v = lo + (s % span)
        return (self, v)
fn uniform_q8(mut self, lo_q8: Int, hi_q8: Int) -> (LcgRng, Int):
        var i = 0; (self, i) = self.uniform_int(lo_q8, hi_q8)
        return (self, i)

struct JitterRanges(Copyable, Movable):
    var bright_min: Int
    var bright_max: Int
    var contrast_min_q8: Int
    var contrast_max_q8: Int
    var sat_min_q8: Int
    var sat_max_q8: Int
    var hue_min_deg: Int
    var hue_max_deg: Int
fn __init__(out self) -> None:
        self.bright_min = -32
        self.bright_max = 32
        self.contrast_min_q8 = 192   # 0.75x
        self.contrast_max_q8 = 320   # 1.25x
        self.sat_min_q8 = 192
        self.sat_max_q8 = 320
        self.hue_min_deg = -10
        self.hue_max_deg = 10

struct JitterParams(Copyable, Movable):
    var brightness_delta: Int
    var contrast_q8: Int
    var saturation_q8: Int
    var hue_deg: Int
fn __init__(out self, bd: Int, cq8: Int, sq8: Int, hd: Int) -> None:
        self.brightness_delta = bd
        self.contrast_q8 = cq8
        self.saturation_q8 = sq8
        self.hue_deg = hd

@staticmethod
fn sample_jitter(mut rng: LcgRng, ranges: JitterRanges) -> (LcgRng, JitterParams):
    var bd = 0; (rng, bd) = rng.uniform_int(ranges.bright_min, ranges.bright_max)
    var cq8 = 256; (rng, cq8) = rng.uniform_q8(ranges.contrast_min_q8, ranges.contrast_max_q8)
    var sq8 = 256; (rng, sq8) = rng.uniform_q8(ranges.sat_min_q8, ranges.sat_max_q8)
    var hd = 0; (rng, hd) = rng.uniform_int(ranges.hue_min_deg, ranges.hue_max_deg)
    return (rng, JitterParams(bd, cq8, sq8, hd))

# -------------------------
# Image wrappers
# -------------------------
@staticmethod
fn apply_color_jitter(img: Image, brightness_delta: Int, contrast_q8: Int, saturation_q8: Int, hue_deg: Int) -> Image:
    var c = _num_ch(img.order)
    var buf = apply_color_jitter_u8_hwc(img.h, img.w, c, img.data, brightness_delta, contrast_q8, saturation_q8, hue_deg)
    return Image(img.h, img.w, img.order, buf)

@staticmethod
fn apply_color_jitter_random(img: Image, seed: Int, ranges: JitterRanges) -> (Image, JitterParams):
    var rng = LcgRng(seed)
    var jp: JitterParams = JitterParams(0,256,256,0)
    (rng, jp) = sample_jitter(rng, ranges)
    var out = apply_color_jitter(img, jp.brightness_delta, jp.contrast_q8, jp.saturation_q8, jp.hue_deg)
    return (out, jp)

# -------------------------
# Minimal smoke test
# -------------------------
@staticmethod
fn __self_test__() -> Bool:
    # Build 2x2 RGB
    var data: List[UInt8] = List[UInt8]()
    # (255,0,0) (0,255,0)
    data.append(255); data.append(0);   data.append(0)
    data.append(0);   data.append(255); data.append(0)
    # (0,0,255) (255,255,255)
    data.append(0);   data.append(0);   data.append(255)
    data.append(255); data.append(255); data.append(255)
    var img = Image(2, 2, ChannelOrder.RGB(), data)

    # No-op jitter should equal input
    var buf0 = apply_color_jitter_u8_hwc(2,2,3, data, 0, 256, 256, 0)
    if len(buf0) != 12: return False
    var i = 0
    while i < 12:
        if buf0[i] != data[i]: return False
        i += 1

    # Brightness +10
    var b1 = apply_color_jitter_u8_hwc(2,2,3, data, 10, 256, 256, 0)
    if not (len(b1) == 12 and b1[0] == UInt8(255)): return False

    # Contrast 0.5 (128 pivot)
    var c1 = apply_color_jitter_u8_hwc(2,2,3, data, 0, 128, 256, 0)
    if len(c1) != 12: return False

    # Saturation 0 (grayscale)
    var s0 = apply_color_jitter_u8_hwc(2,2,3, data, 0, 256, 0, 0)
    if len(s0) != 12: return False

    # Hue shift small
    var h1 = apply_color_jitter_u8_hwc(2,2,3, data, 0, 256, 256, 15)
    if len(h1) != 12: return False

    # Random sample
    var ranges = JitterRanges()
    var out_img: Image; var jp: JitterParams = JitterParams(0,256,256,0)
    (out_img, jp) = apply_color_jitter_random(img, 12345, ranges)
    if not (len(out_img.data) == 12): return False

    return True