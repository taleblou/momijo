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
# File: src/momijo/vision/ops/convert_color.mojo

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
            while i < expected: buf.append(0); i += 1
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

# -------------------------
# Raw buffer kernels (UInt8/HWC)
# -------------------------
@staticmethod
fn bgr_to_rgb_u8_hwc(h: Int, w: Int, src: List[UInt8]) -> List[UInt8]:
    var total = h * w * 3
    var out = _alloc_u8(total)
    var y = 0
    while y < h:
        var x = 0
        while x < w:
            var base = _offset(w, 3, x, y, 0)
            out[base + 0] = src[base + 2]
            out[base + 1] = src[base + 1]
            out[base + 2] = src[base + 0]
            x += 1
        y += 1
    return out

@staticmethod
fn rgb_to_gray_u8_hwc(h: Int, w: Int, src: List[UInt8]) -> List[UInt8]:
    var out = _alloc_u8(h * w)
    var y = 0
    while y < h:
        var x = 0
        while x < w:
            var base = _offset(w, 3, x, y, 0)
            var r = src[base + 0]
            var g = src[base + 1]
            var b = src[base + 2]
            var gy = UInt16(77) * UInt16(r) + UInt16(150) * UInt16(g) + UInt16(29) * UInt16(b)
            out[y * w + x] = UInt8((gy >> UInt8(8)) & UInt16(0xFF))
            x += 1
        y += 1
    return out

@staticmethod
fn rgba_to_rgb_u8_hwc(h: Int, w: Int, src: List[UInt8]) -> List[UInt8]:
    var out = _alloc_u8(h * w * 3)
    var y = 0
    while y < h:
        var x = 0
        while x < w:
            var s = _offset(w, 4, x, y, 0)
            var d = _offset(w, 3, x, y, 0)
            out[d + 0] = src[s + 0]
            out[d + 1] = src[s + 1]
            out[d + 2] = src[s + 2]
            x += 1
        y += 1
    return out

@staticmethod
fn bgra_to_bgr_u8_hwc(h: Int, w: Int, src: List[UInt8]) -> List[UInt8]:
    var out = _alloc_u8(h * w * 3)
    var y = 0
    while y < h:
        var x = 0
        while x < w:
            var s = _offset(w, 4, x, y, 0)
            var d = _offset(w, 3, x, y, 0)
            out[d + 0] = src[s + 0]
            out[d + 1] = src[s + 1]
            out[d + 2] = src[s + 2]
            x += 1
        y += 1
    return out

@staticmethod
fn argmin_channel_u8_hwc(h: Int, w: Int, src: List[UInt8]) -> List[UInt8]:
    var out = _alloc_u8(h * w)
    var y = 0
    while y < h:
        var x = 0
        while x < w:
            var base = _offset(w, 3, x, y, 0)
            var v0 = src[base + 0]; var v1 = src[base + 1]; var v2 = src[base + 2]
            var arg = UInt8(0); var minv = v0
            if v1 < minv: minv = v1; arg = UInt8(1)
            if v2 < minv: arg = UInt8(2)
            out[y * w + x] = arg
            x += 1
        y += 1
    return out

# -------------------------
# Image wrappers
# -------------------------
@staticmethod
fn apply_bgr_to_rgb(img: Image) -> Image:
    if not (img.order == ChannelOrder.BGR() and _num_ch(img.order) == 3):
        # pass-through copy as RGB if already RGB; else best effort swap for 3ch
        if img.order == ChannelOrder.RGB():
            var copy: List[UInt8] = List[UInt8]()
            var i = 0; var n = img.h * img.w * 3
            while i < n: copy.append(img.data[i]); i += 1
            return Image(img.h, img.w, ChannelOrder.RGB(), copy)
        # unknown: try swap indices 0 and 2
    var out = bgr_to_rgb_u8_hwc(img.h, img.w, img.data)
    return Image(img.h, img.w, ChannelOrder.RGB(), out)

@staticmethod
fn apply_rgb_to_gray(img: Image) -> Image:
    # If already gray, just return copy
    if img.order == ChannelOrder.GRAY():
        var copy: List[UInt8] = List[UInt8]()
        var i = 0; var n = img.h * img.w
        while i < n: copy.append(img.data[i]); i += 1
        return Image(img.h, img.w, ChannelOrder.GRAY(), copy)
    # Expect RGB/BGR; if BGR convert first
    var rgb_buf: List[UInt8] = List[UInt8]()
    if img.order == ChannelOrder.BGR():
        rgb_buf = bgr_to_rgb_u8_hwc(img.h, img.w, img.data)
    elif img.order == ChannelOrder.RGB():
        # copy to avoid aliasing
        var n3 = img.h * img.w * 3
        var i2 = 0
        while i2 < n3: rgb_buf.append(img.data[i2]); i2 += 1
    else:
        # For RGBA/BGRA take first three channels as RGB
        var rgb: List[UInt8] = List[UInt8]()
        var y = 0
        while y < img.h:
            var x = 0
            while x < img.w:
                var base = _offset(img.w, _num_ch(img.order), x, y, 0)
                rgb.append(img.data[base + 0])
                rgb.append(img.data[base + 1])
                rgb.append(img.data[base + 2])
                x += 1
            y += 1
        rgb_buf = rgb
    var gray = rgb_to_gray_u8_hwc(img.h, img.w, rgb_buf)
    return Image(img.h, img.w, ChannelOrder.GRAY(), gray)

@staticmethod
fn apply_drop_alpha(img: Image) -> Image:
    if _num_ch(img.order) == 3:
        # copy
        var copy: List[UInt8] = List[UInt8]()
        var i = 0; var n = img.h * img.w * 3
        while i < n: copy.append(img.data[i]); i += 1
        return Image(img.h, img.w, img.order, copy)
    if img.order == ChannelOrder.RGBA():
        var rgb = rgba_to_rgb_u8_hwc(img.h, img.w, img.data)
        return Image(img.h, img.w, ChannelOrder.RGB(), rgb)
    if img.order == ChannelOrder.BGRA():
        var bgr = bgra_to_bgr_u8_hwc(img.h, img.w, img.data)
        return Image(img.h, img.w, ChannelOrder.BGR(), bgr)
    # default: keep first 3 channels
    var out: List[UInt8] = List[UInt8]()
    var y = 0
    while y < img.h:
        var x = 0
        while x < img.w:
            var base = _offset(img.w, _num_ch(img.order), x, y, 0)
            out.append(img.data[base + 0])
            out.append(img.data[base + 1])
            out.append(img.data[base + 2])
            x += 1
        y += 1
    # assume RGB
    return Image(img.h, img.w, ChannelOrder.RGB(), out)

# -------------------------
# Minimal smoke test
# -------------------------
@staticmethod
fn __self_test__() -> Bool:
    # 2x2 RGB pattern
    var rgb: List[UInt8] = List[UInt8]()
    rgb.append(255); rgb.append(0);   rgb.append(0)
    rgb.append(0);   rgb.append(255); rgb.append(0)
    rgb.append(0);   rgb.append(0);   rgb.append(255)
    rgb.append(255); rgb.append(255); rgb.append(255)

    var img_rgb = Image(2, 2, ChannelOrder.RGB(), rgb)
    var g = apply_rgb_to_gray(img_rgb)
    if not (_num_ch(g.order) == 1 and len(g.data) == 4): return False

    # BGR -> RGB
    var bgr = bgr_to_rgb_u8_hwc(2, 2, rgb)  # treat as BGR swap
    var img_bgr = Image(2, 2, ChannelOrder.BGR(), bgr)
    var rgb2 = apply_bgr_to_rgb(img_bgr)
    if not (_num_ch(rgb2.order) == 3 and len(rgb2.data) == 12): return False

    # RGBA drop alpha
    var rgba: List[UInt8] = List[UInt8]()
    # px0 (1,2,3,4) px1 (5,6,7,8) px2 (9,10,11,12) px3 (13,14,15,16)
    rgba.append(1); rgba.append(2); rgba.append(3); rgba.append(4)
    rgba.append(5); rgba.append(6); rgba.append(7); rgba.append(8)
    rgba.append(9); rgba.append(10); rgba.append(11); rgba.append(12)
    rgba.append(13); rgba.append(14); rgba.append(15); rgba.append(16)
    var img_rgba = Image(2, 2, ChannelOrder.RGBA(), rgba)
    var rgb3 = apply_drop_alpha(img_rgba)
    if not (_num_ch(rgb3.order) == 3 and len(rgb3.data) == 12): return False

    # argmin_channel sanity
    var am = argmin_channel_u8_hwc(2, 2, rgb)
    if len(am) != 4: return False

    return True