# MIT License
# Copyright (c) 2025 Morteza Talebou and Mitra Daneshmand
# Project: momijo  |  Source: https://github.com/taleblou/momijo
# This file is part of the Momijo project. See the LICENSE file at the repository root.
# Momijo 
# SPDX-License-Identifier: MIT
# Website: https://taleblou.ir/
# Repository: https://github.com/taleblou/momijo
#
# Project: momijo.vision
# File: src/momijo/vision/image.mojo

from momijo.vision.dtypes import DType, Layout, ColorSpace
from momijo.vision.tensor import Tensor, packed_hwc_strides
from momijo.vision.memory import alloc_u8

# -----------------------------------------------------------------------------
# Image metadata
# -----------------------------------------------------------------------------

 
struct ImageMeta(ExplicitlyCopyable, Movable):
    var _layout: Layout
    var _cs: ColorSpace
    var _alpha_premultiplied: Bool

    # Provide a default constructor so ImageMeta() works.
    fn __init__(out self,
                layout: Layout = Layout.HWC(),
                cs: ColorSpace = ColorSpace.SRGB(),
                alpha_premultiplied: Bool = False):
        self._layout = layout
        self._cs = cs
        self._alpha_premultiplied = alpha_premultiplied

    fn __copyinit__(out self, other: Self):
        self._layout = other._layout
        self._cs = other._cs
        self._alpha_premultiplied = other._alpha_premultiplied

    fn layout(self) -> Layout:
        return self._layout

    fn colorspace(self) -> ColorSpace:
        return self._cs

    fn is_premultiplied(self) -> Bool:
        return self._alpha_premultiplied

    fn with_colorspace(self, cs: ColorSpace) -> ImageMeta:
        return ImageMeta(self._layout, cs, self._alpha_premultiplied)

    fn with_layout(self, layout: Layout) -> ImageMeta:
        return ImageMeta(layout, self._cs, self._alpha_premultiplied)

    fn set_premultiplied(mut self, v: Bool):
        self._alpha_premultiplied = v

# -----------------------------------------------------------------------------
# Image wrapper around a Tensor
# -----------------------------------------------------------------------------
@fieldwise_init
struct Image(ExplicitlyCopyable, Movable):
    var _tensor: Tensor
    var _meta: ImageMeta

    fn __init__(out self, meta: ImageMeta, tensor: Tensor):
        self._meta = meta
        self._tensor = tensor

    fn __copyinit__(out self, other: Self):
        self._tensor = other._tensor
        self._meta = other._meta

    # --- basic getters ---
    fn tensor(self) -> Tensor:
        return self._tensor

    fn meta(self) -> ImageMeta:
        return self._meta

    fn height(self) -> Int:
        return self._tensor.height()

    fn width(self) -> Int:
        return self._tensor.width()

    fn channels(self) -> Int:
        return self._tensor.channels()

    fn dtype(self) -> DType:
        return self._tensor.dtype()

    fn colorspace(self) -> ColorSpace:
        return self._meta.colorspace()

    fn layout(self) -> Layout:
        return self._meta.layout()

    # --- layout / contiguity helpers ---
    fn is_empty(self) -> Bool:
        return self.height() <= 0 or self.width() <= 0 or self.channels() <= 0

    fn is_hwc(self) -> Bool:
        return self._meta.layout() == Layout.HWC()

    fn is_u8(self) -> Bool:
        return self._tensor.dtype() == DType.UInt8()

    fn is_contiguous_hwc_u8(self) -> Bool:
        return self.is_hwc() and self.is_u8() and self._tensor.is_contiguous_hwc_u8()

    # Returns a copy that is packed HWC/u8 if copy_if_needed is True; otherwise returns self unchanged.
    fn ensure_packed_hwc_u8(self, copy_if_needed: Bool = True) -> Image:
        if self.is_contiguous_hwc_u8():
            return self
        if copy_if_needed:
            var t = self._tensor.copy_to_packed_hwc()
            return Image(t, self._meta)
        return self

    # --- cloning / views ---
    fn clone(self) -> Image:
        var copied = self._tensor.clone()      # deep copy
        return Image(self._meta, copied)       # دقت: (meta, tensor)


    fn roi(self, y: Int, x: Int, h: Int, w: Int) -> Image:
        # Safe ROI for HWC layout; clamps to image bounds.
        if not self.is_hwc():
            return self

        var H = self.height()
        var W = self.width()
        if H <= 0 or W <= 0:
            return self

        var y0 = y
        if y0 < 0: y0 = 0
        if y0 >= H: y0 = H - 1

        var x0 = x
        if x0 < 0: x0 = 0
        if x0 >= W: x0 = W - 1

        var y1 = y + h
        if y1 <= y0 + 1: y1 = y0 + 1
        if y1 > H: y1 = H

        var x1 = x + w
        if x1 <= x0 + 1: x1 = x0 + 1
        if x1 > W: x1 = W

        var hh = y1 - y0
        var ww = x1 - x0
        # Use copy-based ROI since Tensor has copy_roi.
        var v = self._tensor.copy_roi(y0, x0, hh, ww)
        return Image(v, self._meta)

    # Keep API parity; currently returns self since zero-copy view not supported.
    fn as_hwc_view(self, h: Int, w: Int, c: Int, s0: Int, s1: Int, s2: Int) -> Image:
        return self

    # --- meta transforms ---
    fn with_meta(self, meta: ImageMeta) -> Image:
        return Image(self._tensor, meta)

    fn with_colorspace(self, cs: ColorSpace) -> Image:
        var m = self._meta.with_colorspace(cs)
        return Image(self._tensor, m)

    # --- pixel access (requires packed HWC/UInt8). No asserts; early returns. ---
    fn set_u8(self, y: Int, x: Int, ch: Int, v: UInt8):
        # Guards
        if not self.is_contiguous_hwc_u8():
            return
        var H = self.height()
        var W = self.width()
        var C = self.channels()
        if y < 0 or y >= H:
            return
        if x < 0 or x >= W:
            return
        if ch < 0 or ch >= C:
            return

        # Flat HWC index
        var idx = (y * W + x) * C + ch

        # Write via raw pointer (no field reassign → no mut self needed)
        var p = self._tensor.data()   # UnsafePointer[UInt8]
        p[idx] = v


    # Read one u8 channel at (y, x, ch) from a contiguous HWC/u8 image.
    fn at_u8(self, y: Int, x: Int, ch: Int) -> UInt8:
        # Guards
        if not self.is_contiguous_hwc_u8():
            return UInt8(0)

        var H = self.height()
        var W = self.width()
        var C = self.channels()

        if y < 0 or y >= H:
            return UInt8(0)
        if x < 0 or x >= W:
            return UInt8(0)
        if ch < 0 or ch >= C:
            return UInt8(0)

        # Flat HWC index
        var idx = (y * W + x) * C + ch

        # Read via underlying tensor pointer
        var t = self._tensor
        return t.load_u8_at(idx)


    fn get_u8(self, y: Int, x: Int, ch: Int) -> UInt8:
        if not self.is_contiguous_hwc_u8(): return UInt8(0)
        if y < 0 or y >= self.height(): return UInt8(0)
        if x < 0 or x >= self.width(): return UInt8(0)
        if ch < 0 or ch >= self.channels(): return UInt8(0)
        var idx = (y * self.width() + x) * self.channels() + ch
        return self._tensor.load_u8_at(idx)

    fn set_rgb_u8(mut self, y: Int, x: Int, r: UInt8, g: UInt8, b: UInt8):
        self.set_u8(y, x, 0, r)
        self.set_u8(y, x, 1, g)
        self.set_u8(y, x, 2, b)

    # --- alpha utilities (only when channels == 4, packed HWC/UInt8) ---
    fn premultiply_alpha(self) -> Image:
        if not self.is_contiguous_hwc_u8(): return self
        if self.channels() != 4: return self

        var out = self.clone()
        var h = out.height()
        var w = out.width()
        var y = 0
        while y < h:
            var x = 0
            while x < w:
                var a = out.get_u8(y, x, 3)
                var r = out.get_u8(y, x, 0)
                var g = out.get_u8(y, x, 1)
                var b = out.get_u8(y, x, 2)

                var rr = (Int(r) * Int(a) + 127) // 255
                var gg = (Int(g) * Int(a) + 127) // 255
                var bb = (Int(b) * Int(a) + 127) // 255

                if rr > 255: rr = 255
                if gg > 255: gg = 255
                if bb > 255: bb = 255

                out.set_u8(y, x, 0, UInt8(rr))
                out.set_u8(y, x, 1, UInt8(gg))
                out.set_u8(y, x, 2, UInt8(bb))
                x += 1
            y += 1
        return out

    fn unpremultiply_alpha(self) -> Image:
        if not self.is_contiguous_hwc_u8(): return self
        if self.channels() != 4: return self

        var out = self.clone()
        var h = out.height()
        var w = out.width()
        var y = 0
        while y < h:
            var x = 0
            while x < w:
                var a = out.get_u8(y, x, 3)
                if a != UInt8(0):
                    var r = out.get_u8(y, x, 0)
                    var g = out.get_u8(y, x, 1)
                    var b = out.get_u8(y, x, 2)

                    var rr = (Int(r) * 255 + Int(a) // 2) // Int(a)
                    var gg = (Int(g) * 255 + Int(a) // 2) // Int(a)
                    var bb = (Int(b) * 255 + Int(a) // 2) // Int(a)

                    if rr > 255: rr = 255
                    if gg > 255: gg = 255
                    if bb > 255: bb = 255

                    out.set_u8(y, x, 0, UInt8(rr))
                    out.set_u8(y, x, 1, UInt8(gg))
                    out.set_u8(y, x, 2, UInt8(bb))
                x += 1
            y += 1
        return out

    # --- factory: create packed HWC/UInt8 image filled with a constant ---
    @staticmethod
    fn new_hwc_u8(h: Int, w: Int, c: Int,
                  value: UInt8 = UInt8(0),
                  cs: ColorSpace = ColorSpace.SRGB(),
                  layout: Layout = Layout.HWC()) -> Image:
        var t = _alloc_tensor_u8(h, w, c, value)
        var m = ImageMeta(layout, cs, False)
        return Image(t, m)

    @staticmethod
    fn full_hwc_u8(h: Int, w: Int, c: Int, value: UInt8) -> Image:
        return Image.new_hwc_u8(h, w, c, value)

    @staticmethod
    fn zeros_hwc_u8(h: Int, w: Int, c: Int) -> Image:
        return Image.new_hwc_u8(h, w, c, UInt8(0))

# -----------------------------------------------------------------------------
# Helpers / factory functions
# -----------------------------------------------------------------------------

# Allocate a Tensor that is packed HWC/UInt8 and optionally filled with 'value'
fn _alloc_tensor_u8(h: Int, w: Int, c: Int, value: UInt8) -> Tensor:
    var hh = h
    if hh <= 0:
        hh = 1

    var ww = w
    if ww <= 0:
        ww = 1

    var cc = c
    if cc <= 0:
        cc = 1

    var (s0, s1, s2) = packed_hwc_strides(hh, ww, cc)
    var n = hh * ww * cc
    var buf = alloc_u8(n)

    var i = 0
    while i < n:
        buf[i] = value
        i += 1

    return Tensor(buf, n, hh, ww, cc, s0, s1, s2, DType.UInt8())


# Wrap an existing UInt8 buffer in packed HWC
fn make_u8_hwc(out img: Image, data: UnsafePointer[UInt8], h: Int, w: Int, c: Int, cs: ColorSpace):
    if h <= 0 or w <= 0 or c <= 0:
        # Produce an empty image if shape is invalid
        var empty_t = Tensor(UnsafePointer[UInt8](), 0, 0, 0, 0, 0, 0, 0, DType.UInt8(), False)
        var empty_m = ImageMeta(Layout.HWC(), cs, False)
        img = Image(empty_t, empty_m)
        return

    var (s0, s1, s2) = packed_hwc_strides(h, w, c)
    var byte_len = h * w * c
    # owns = False because 'data' is external
    var t = Tensor(data, byte_len, h, w, c, s0, s1, s2, DType.UInt8(), False)
    var m = ImageMeta(Layout.HWC(), cs, False)
    img = Image(t, m)

# Allocate a fresh zero-initialized u8 HWC buffer and wrap as Image
fn make_zero_u8_hwc(h: Int, w: Int, c: Int, cs: ColorSpace) -> Image:
    if h <= 0 or w <= 0 or c <= 0:
        var empty_t = Tensor(UnsafePointer[UInt8](), 0, 0, 0, 0, 0, 0, 0, DType.UInt8(), False)
        var empty_m = ImageMeta(Layout.HWC(), cs, False)
        return Image(empty_t, empty_m)

    var (s0, s1, s2) = packed_hwc_strides(h, w, c)
    var n = h * w * c
    var buf = alloc_u8(n)
    var i = 0
    while i < n:
        buf[i] = 0
        i += 1
    var t = Tensor(buf, n, h, w, c, s0, s1, s2, DType.UInt8(), True)
    var m = ImageMeta(Layout.HWC(), cs, False)
    return Image(t, m)

# Validate basic invariants; returns True if OK (no assertions)
fn validate_image(self: Image) -> Bool:
    if self.height() <= 0: return False
    if self.width()  <= 0: return False
    if self.channels() <= 0: return False
    if not (self.layout() == Layout.HWC()): return False
    if not (self.dtype() == DType.UInt8()): return False
    return True
