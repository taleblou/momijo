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
# Project: momijo.vision.ops
# File: src/momijo/vision/ops/resize.mojo

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
                buf.append(0)
                i += 1
            self.data = buf
        else:
            self.data = data

# -------------------------
# Interpolation mode
# -------------------------
struct Interp(Copyable, Movable):
    var id: Int
fn __init__(out self, id: Int) -> None: self.id = id
    @staticmethod fn NEAREST() -> Interp:  return Interp(0)
    @staticmethod fn BILINEAR() -> Interp: return Interp(1)
fn __eq__(self, other: Interp) -> Bool: return self.id == other.id

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
# Core resize kernels (UInt8/HWC, generic channels)
# -------------------------
@staticmethod
fn resize_nearest_u8_hwc(h: Int, w: Int, c: Int, src: List[UInt8], oh: Int, ow: Int) -> List[UInt8]:
    if h <= 0 or w <= 0 or c <= 0 or oh <= 0 or ow <= 0:
        return List[UInt8]()
    var out = _alloc_u8(oh * ow * c)
    var y = 0
    while y < oh:
        var sy = (y * h) // oh
        var x = 0
        while x < ow:
            var sx = (x * w) // ow
            var chn = 0
            while chn < c:
                out[_offset(ow, c, x, y, chn)] = src[_offset(w, c, sx, sy, chn)]
                chn += 1
            x += 1
        y += 1
    return out

@staticmethod
fn resize_bilinear_u8_hwc(h: Int, w: Int, c: Int, src: List[UInt8], oh: Int, ow: Int) -> List[UInt8]:
    if h <= 0 or w <= 0 or c <= 0 or oh <= 0 or ow <= 0:
        return List[UInt8]()
    var out = _alloc_u8(oh * ow * c)
    var y = 0
    while y < oh:
        var sy_fp = 0
        if oh > 1:
            sy_fp = (y * (h - 1) * 256) // (oh - 1)
        var y0 = sy_fp >> UInt8(8)
        var fy = sy_fp & UInt8(255)
        var y1 = y0 + 1
        if y1 >= h: y1 = h - 1
        var wy0 = 256 - fy

        var x = 0
        while x < ow:
            var sx_fp = 0
            if ow > 1:
                sx_fp = (x * (w - 1) * 256) // (ow - 1)
            var x0 = sx_fp >> UInt8(8)
            var fx = sx_fp & UInt8(255)
            var x1 = x0 + 1
            if x1 >= w: x1 = w - 1

            var wx0 = 256 - fx
            var w00 = wy0 * wx0
            var w10 = wy0 * fx
            var w01 = fy * wx0
            var w11 = fy * fx

            var chn = 0
            while chn < c:
                var p00 = Int(src[_offset(w, c, x0, y0, chn)])
                var p10 = Int(src[_offset(w, c, x1, y0, chn)])
                var p01 = Int(src[_offset(w, c, x0, y1, chn)])
                var p11 = Int(src[_offset(w, c, x1, y1, chn)])
                var accum = p00 * w00 + p10 * w10 + p01 * w01 + p11 * w11
                var val = (accum + 32768) >> 16
                out[_offset(ow, c, x, y, chn)] = _clamp_u8(val)
                chn += 1
            x += 1
        y += 1
    return out

# -------------------------
# Image wrappers (preserve order)
# -------------------------
@staticmethod
fn apply_resize_nearest(img: Image, out_h: Int, out_w: Int) -> Image:
    var c = _num_ch(img.order)
    var out_buf = resize_nearest_u8_hwc(img.h, img.w, c, img.data, out_h, out_w)
    return Image(out_h, out_w, img.order, out_buf)

@staticmethod
fn apply_resize_bilinear(img: Image, out_h: Int, out_w: Int) -> Image:
    var c = _num_ch(img.order)
    var out_buf = resize_bilinear_u8_hwc(img.h, img.w, c, img.data, out_h, out_w)
    return Image(out_h, out_w, img.order, out_buf)

@staticmethod
fn apply_resize(img: Image, out_h: Int, out_w: Int, mode: Interp) -> Image:
    if mode == Interp.NEAREST():
        return apply_resize_nearest(img, out_h, out_w)
    return apply_resize_bilinear(img, out_h, out_w)

# -------------------------
# Aspect-ratio helper (optional)
# -------------------------
@staticmethod
fn keep_aspect_fit(h: Int, w: Int, max_h: Int, max_w: Int) -> (Int, Int):
    # returns (out_h, out_w) that fits inside max_h x max_w while keeping aspect
    if h <= 0 or w <= 0 or max_h <= 0 or max_w <= 0:
        return (0, 0)
    # compare ratios without float
    # scale_h = max_h / h, scale_w = max_w / w -> choose min
    var num_h = max_h * w
    var num_w = max_w * h
    if num_h < num_w:
        # limited by height
        var out_h = max_h
        var out_w = (w * max_h) // h
        if out_w < 1: out_w = 1
        return (out_h, out_w)
    else:
        var out_w = max_w
        var out_h = (h * max_w) // w
        if out_h < 1: out_h = 1
        return (out_h, out_w)

# -------------------------
# Minimal smoke test
# -------------------------
@staticmethod
fn __self_test__() -> Bool:
    # RGB 2x2
    var data: List[UInt8] = List[UInt8]()
    # (255,0,0) (0,255,0)
    data.append(255); data.append(0);   data.append(0)
    data.append(0);   data.append(255); data.append(0)
    # (0,0,255) (255,255,255)
    data.append(0);   data.append(0);   data.append(255)
    data.append(255); data.append(255); data.append(255)
    var img = Image(2, 2, ChannelOrder.RGB(), data)

    var n1 = resize_nearest_u8_hwc(2, 2, 3, data, 4, 4)
    if len(n1) != 4*4*3: return False
    var b1 = resize_bilinear_u8_hwc(2, 2, 3, data, 5, 7)
    if len(b1) != 5*7*3: return False

    var outN = apply_resize_nearest(img, 4, 4)
    if not (outN.h == 4 and outN.w == 4 and len(outN.data) == 4*4*3): return False
    var outB = apply_resize_bilinear(img, 3, 5)
    if not (outB.h == 3 and outB.w == 5 and len(outB.data) == 3*5*3): return False

    # keep-aspect helper
    var (oh, ow) = keep_aspect_fit(1080, 1920, 256, 256)
    if not (oh == 144 and ow == 256): return False

    # generic channels (c=1)
    var g: List[UInt8] = List[UInt8]()
    g.append(10); g.append(20); g.append(30); g.append(40)
    var g2 = resize_nearest_u8_hwc(2, 2, 1, g, 1, 4)
    if len(g2) != 4: return False

    return True