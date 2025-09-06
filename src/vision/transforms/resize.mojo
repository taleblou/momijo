# MIT License
# Copyright (c) 2025 
# Project: momijo  |  Source: https://github.com/taleblou/momijo
# SPDX-License-Identifier: MIT
#
# Project: momijo.vision
# File: src/momijo/vision/resize.mojo
#
# High-level resize helpers for Momijo Vision (standalone).
# Style: no 'export', no 'let', no 'inout'. Constructors use `fn __init__(out self, ...)`.
#
# Features:
#   - UInt8/HWC core kernels:
#       * Nearest + Bilinear with optional align_corners
#       * Optional anti-alias (box prefilter) for downscaling
#   - Image wrapper with ChannelOrder
#   - Interp enum + helpers keep_aspect_fit/fill and letterbox size compute
#   - __self_test__()
#
# Notes:
# - Anti-alias uses a simple 1D separable box filter when scaling < 1.0 (downsampling).
# - align_corners semantics:
#     False (default): standard half-pixel centers (PyTorch default for interpolate(mode='bilinear', align_corners=False)).
#     True: maps corner pixels exactly (OpenCV-like alignCorners).
# - All functions allocate fresh outputs to avoid aliasing.

# -------------------------
# Lightweight Image (U8/HWC) + ChannelOrder
# -------------------------
struct ChannelOrder(Copyable, Movable):
    var id: Int
    fn __init__(out self, id: Int): self.id = id
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

    fn __init__(out self, h: Int, w: Int, order: ChannelOrder, data: List[UInt8]):
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
# Interp enum
# -------------------------
struct Interp(Copyable, Movable):
    var id: Int
    fn __init__(out self, id: Int): self.id = id
    @staticmethod fn NEAREST() -> Interp:  return Interp(0)
    @staticmethod fn BILINEAR() -> Interp: return Interp(1)
    fn __eq__(self, other: Interp) -> Bool: return self.id == other.id
    fn to_string(self) -> String:
        if self.id == 0: return String("NEAREST")
        if self.id == 1: return String("BILINEAR")
        return String("Interp(") + String(self.id) + String(")")

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

@staticmethod
fn _min(a: Int, b: Int) -> Int:
    if a < b: return a
    return b

@staticmethod
fn _max(a: Int, b: Int) -> Int:
    if a > b: return a
    return b

# -------------------------
# 1D Box filter (for anti-alias downscale)
# -------------------------
@staticmethod
fn _box_filter_horizontal_u8(h: Int, w: Int, c: Int, src: List[UInt8], radius: Int) -> List[UInt8]:
    var out = _alloc_u8(h*w*c)
    if radius <= 0: 
        var i = 0; var n = h*w*c
        while i < n: out[i] = src[i]; i += 1
        return out
    var win = radius * 2 + 1
    var y = 0
    while y < h:
        var x = 0
        while x < w:
            var x0 = x - radius; if x0 < 0: x0 = 0
            var x1 = x + radius; if x1 >= w: x1 = w - 1
            var chn = 0
            while chn < c:
                var sum = 0; var xi = x0
                while xi <= x1:
                    sum = sum + Int(src[_offset(w, c, xi, y, chn)])
                    xi += 1
                var denom = (x1 - x0 + 1)
                out[_offset(w, c, x, y, chn)] = _clamp_u8( (sum + (denom//2)) // denom )
                chn += 1
            x += 1
        y += 1
    return out

@staticmethod
fn _box_filter_vertical_u8(h: Int, w: Int, c: Int, src: List[UInt8], radius: Int) -> List[UInt8]:
    var out = _alloc_u8(h*w*c)
    if radius <= 0: 
        var i = 0; var n = h*w*c
        while i < n: out[i] = src[i]; i += 1
        return out
    var win = radius * 2 + 1
    var y = 0
    while y < h:
        var x = 0
        while x < w:
            var chn = 0
            while chn < c:
                var y0 = y - radius; if y0 < 0: y0 = 0
                var y1 = y + radius; if y1 >= h: y1 = h - 1
                var sum = 0; var yi = y0
                while yi <= y1:
                    sum = sum + Int(src[_offset(w, c, x, yi, chn)])
                    yi += 1
                var denom = (y1 - y0 + 1)
                out[_offset(w, c, x, y, chn)] = _clamp_u8( (sum + (denom//2)) // denom )
                chn += 1
            x += 1
        y += 1
    return out

@staticmethod
fn _antialias_prefilter_if_needed(h: Int, w: Int, c: Int, src: List[UInt8], oh: Int, ow: Int, antialias: Bool) -> List[UInt8]:
    if not antialias: 
        var copy: List[UInt8] = List[UInt8](); var n = h*w*c; var i0 = 0
        while i0 < n: copy.append(src[i0]); i0 += 1
        return copy
    # radius ~ ceil(0.5 * downscale_factor) per axis
    var rh = 0; var rw = 0
    if oh < h: rh = (h // _max(1, oh)) // 2
    if ow < w: rw = (w // _max(1, ow)) // 2
    if rh < 0: rh = 0
    if rw < 0: rw = 0
    if rh == 0 and rw == 0:
        var copy2: List[UInt8] = List[UInt8](); var n2 = h*w*c; var j0 = 0
        while j0 < n2: copy2.append(src[j0]); j0 += 1
        return copy2
    var tmp = _box_filter_horizontal_u8(h, w, c, src, rw)
    var out = _box_filter_vertical_u8(h, w, c, tmp, rh)
    return out

# -------------------------
# Core resize kernels (UInt8/HWC, generic channels)
# -------------------------
@staticmethod
fn resize_nearest_u8_hwc(h: Int, w: Int, c: Int, src: List[UInt8], oh: Int, ow: Int, align_corners: Bool) -> List[UInt8]:
    if h <= 0 or w <= 0 or c <= 0 or oh <= 0 or ow <= 0:
        return List[UInt8]()
    var out = _alloc_u8(oh * ow * c)

    var y = 0
    while y < oh:
        var sy = 0
        if align_corners and oh > 1:
            sy = (y * (h - 1) + (oh - 1) // 2) // (oh - 1)
        else:
            sy = (( (2*y + 1) * h ) // (2*oh))
        var x = 0
        while x < ow:
            var sx = 0
            if align_corners and ow > 1:
                sx = (x * (w - 1) + (ow - 1) // 2) // (ow - 1)
            else:
                sx = (( (2*x + 1) * w ) // (2*ow))
            var chn = 0
            while chn < c:
                out[_offset(ow, c, x, y, chn)] = src[_offset(w, c, sx, sy, chn)]
                chn += 1
            x += 1
        y += 1
    return out

@staticmethod
fn resize_bilinear_u8_hwc(h: Int, w: Int, c: Int, src: List[UInt8], oh: Int, ow: Int, align_corners: Bool, antialias: Bool) -> List[UInt8]:
    if h <= 0 or w <= 0 or c <= 0 or oh <= 0 or ow <= 0:
        return List[UInt8]()

    var pre = _antialias_prefilter_if_needed(h, w, c, src, oh, ow, antialias)

    var out = _alloc_u8(oh * ow * c)
    var y = 0
    while y < oh:
        # map y in [0..oh-1] to src space
        var sy_fp = 0
        if align_corners:
            if oh > 1:
                sy_fp = (y * (h - 1) * 256) // (oh - 1)
            else:
                sy_fp = 0
        else:
            # half-pixel centers
            sy_fp = (((2*y + 1) * h - oh) * 256) // (2 * oh)
        var y0 = sy_fp >> 8
        var fy = sy_fp & 255
        var y1 = y0 + 1
        if y1 >= h: y1 = h - 1
        var wy0 = 256 - fy

        var x = 0
        while x < ow:
            var sx_fp = 0
            if align_corners:
                if ow > 1:
                    sx_fp = (x * (w - 1) * 256) // (ow - 1)
                else:
                    sx_fp = 0
            else:
                sx_fp = (((2*x + 1) * w - ow) * 256) // (2 * ow)

            var x0 = sx_fp >> 8
            var fx = sx_fp & 255
            var x1 = x0 + 1
            if x1 >= w: x1 = w - 1

            var wx0 = 256 - fx
            var w00 = wy0 * wx0
            var w10 = wy0 * fx
            var w01 = fy * wx0
            var w11 = fy * fx

            var chn = 0
            while chn < c:
                var p00 = Int(pre[_offset(w, c, x0, y0, chn)])
                var p10 = Int(pre[_offset(w, c, x1, y0, chn)])
                var p01 = Int(pre[_offset(w, c, x0, y1, chn)])
                var p11 = Int(pre[_offset(w, c, x1, y1, chn)])
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
fn apply_resize_nearest(img: Image, out_h: Int, out_w: Int, align_corners: Bool) -> Image:
    var c = _num_ch(img.order)
    var out_buf = resize_nearest_u8_hwc(img.h, img.w, c, img.data, out_h, out_w, align_corners)
    return Image(out_h, out_w, img.order, out_buf)

@staticmethod
fn apply_resize_bilinear(img: Image, out_h: Int, out_w: Int, align_corners: Bool, antialias: Bool) -> Image:
    var c = _num_ch(img.order)
    var out_buf = resize_bilinear_u8_hwc(img.h, img.w, c, img.data, out_h, out_w, align_corners, antialias)
    return Image(out_h, out_w, img.order, out_buf)

@staticmethod
fn apply_resize(img: Image, out_h: Int, out_w: Int, mode: Interp, align_corners: Bool, antialias: Bool) -> Image:
    if mode == Interp.NEAREST():
        return apply_resize_nearest(img, out_h, out_w, align_corners)
    return apply_resize_bilinear(img, out_h, out_w, align_corners, antialias)

# -------------------------
# Aspect helpers
# -------------------------
@staticmethod
fn keep_aspect_fit(h: Int, w: Int, max_h: Int, max_w: Int) -> (Int, Int):
    if h <= 0 or w <= 0 or max_h <= 0 or max_w <= 0: return (0, 0)
    var num_h = max_h * w
    var num_w = max_w * h
    if num_h < num_w:
        var out_h = max_h
        var out_w = (w * max_h) // h
        if out_w < 1: out_w = 1
        return (out_h, out_w)
    else:
        var out_w = max_w
        var out_h = (h * max_w) // w
        if out_h < 1: out_h = 1
        return (out_h, out_w)

@staticmethod
fn keep_aspect_fill(h: Int, w: Int, min_h: Int, min_w: Int) -> (Int, Int):
    # smallest size that fully covers min_h x min_w
    if h <= 0 or w <= 0 or min_h <= 0 or min_w <= 0: return (0, 0)
    var num_h = min_h * w
    var num_w = min_w * h
    if num_h > num_w:
        var out_h = min_h
        var out_w = (w * min_h + h - 1) // h
        return (out_h, out_w)
    else:
        var out_w = min_w
        var out_h = (h * min_w + w - 1) // w
        return (out_h, out_w)

@staticmethod
fn letterbox_size(h: Int, w: Int, box_h: Int, box_w: Int) -> (Int, Int, Int, Int):
    # returns (oh, ow, pad_top, pad_left) to place resized image centered in a box
    var oh = 0; var ow = 0
    (oh, ow) = keep_aspect_fit(h, w, box_h, box_w)
    var pad_t = (box_h - oh) // 2
    var pad_l = (box_w - ow) // 2
    return (oh, ow, pad_t, pad_l)

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

    # Nearest (align_corners=False)
    var n1 = resize_nearest_u8_hwc(2, 2, 3, data, 4, 4, False)
    if len(n1) != 4*4*3: return False

    # Bilinear with align_corners=True and antialias=True downscale
    var b1 = resize_bilinear_u8_hwc(2, 2, 3, data, 5, 7, True, False)
    if len(b1) != 5*7*3: return False
    var b2 = resize_bilinear_u8_hwc(8, 6, 3, _alloc_u8(8*6*3), 2, 2, False, True)
    if len(b2) != 2*2*3: return False

    # Wrappers
    var outN = apply_resize(img, 4, 4, Interp.NEAREST(), False, False)
    if not (outN.h == 4 and outN.w == 4 and len(outN.data) == 4*4*3): return False
    var outB = apply_resize(img, 3, 5, Interp.BILINEAR(), True, False)
    if not (outB.h == 3 and outB.w == 5 and len(outB.data) == 3*5*3): return False

    # Aspect helpers
    var oh = 0; var ow = 0
    (oh, ow) = keep_aspect_fit(1080, 1920, 256, 256)
    if not (oh == 144 and ow == 256): return False
    (oh, ow) = keep_aspect_fill(1080, 1920, 256, 256)
    if not (oh == 256 and ow >= 256): return False

    var oh2 = 0; var ow2 = 0; var pt = 0; var pl = 0
    (oh2, ow2, pt, pl) = letterbox_size(1080, 1920, 256, 256)
    if not (oh2 == 144 and ow2 == 256 and pt == 56 and pl == 0): return False

    return True
