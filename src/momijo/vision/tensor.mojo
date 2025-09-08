# Project:      Momijo
# Module:       src.momijo.vision.tensor
# File:         tensor.mojo
# Path:         src/momijo/vision/tensor.mojo
#
# Description:  src.momijo.vision.tensor — focused Momijo functionality with a stable public API.
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
#   - Structs: DType, Layout, Tensor
#   - Key functions: __init__, UInt8, Float32, __eq__, to_string, dtype_bytes, __init__, HWC ...
#   - Static methods present.


struct DType(Copyable, Movable):
    var id: Int
fn __init__(out self, id: Int) -> None:
        self.id = id
    @staticmethod
fn UInt8() -> DType: return DType(1)
    @staticmethod
fn Float32() -> DType: return DType(2)   # reserved
fn __eq__(self, other: DType) -> Bool: return self.id == other.id
fn to_string(self) -> String:
        if self.id == 1: return String("UInt8")
        if self.id == 2: return String("Float32")
        return String("UnknownDType")

@staticmethod
fn dtype_bytes(dt: DType) -> Int:
    if dt == DType.UInt8(): return 1
    if dt == DType.Float32(): return 4
    return 1

struct Layout(Copyable, Movable):
    var id: Int
fn __init__(out self, id: Int) -> None:
        self.id = id
    @staticmethod
fn HWC() -> Layout: return Layout(1)
    @staticmethod
fn CHW() -> Layout: return Layout(2)
fn __eq__(self, other: Layout) -> Bool: return self.id == other.id
fn to_string(self) -> String:
        if self.id == 1: return String("HWC")
        if self.id == 2: return String("CHW")
        return String("UnknownLayout")

# -------------------------
# Stride helpers
# -------------------------
@staticmethod
fn packed_hwc_strides(h: Int, w: Int, c: Int) -> (Int, Int, Int):
    # (s0, s1, s2) for indices (y, x, ch)
    var s2 = 1
    var s1 = c
    var s0 = w * c
    return (s0, s1, s2)

@staticmethod
fn packed_chw_strides(c: Int, h: Int, w: Int) -> (Int, Int, Int):
    # (s0, s1, s2) for indices (ch, y, x)
    var s2 = 1
    var s1 = w
    var s0 = h * w
    return (s0, s1, s2)

# -------------------------
# Tensor (uint8 only storage for now)
# -------------------------
struct Tensor(Copyable, Movable):
    # Logical dims always stored as image dims (h, w, c) regardless of layout.
    var h: Int
    var w: Int
    var c: Int
    var layout: Layout
    var dtype: DType
    # Strides correspond to the layout's indexing order:
    # - HWC: s0,s1,s2 for (y,x,ch)
    # - CHW: s0,s1,s2 for (ch,y,x)
    var s0: Int
    var s1: Int
    var s2: Int
    var data: List[UInt8]
fn __init__(out self, h: Int, w: Int, c: Int, layout: Layout, dtype: DType, s0: Int, s1: Int, s2: Int, data: List[UInt8]) -> None:
        self.h = h
        self.w = w
        self.c = c
        self.layout = layout
        self.dtype = dtype
        self.s0 = s0
        self.s1 = s1
        self.s2 = s2
        # validate size (U8 only)
        var elem = dtype_bytes(dtype)
        var expected = h * w * c * elem
        if len(data) != expected:
            var fixed: List[UInt8] = List[UInt8]()
            var i = 0
            while i < expected:
                fixed.append(0)
                i += 1
            self.data = fixed
        else:
            self.data = data

    # Constructors
    @staticmethod
fn zeros_hwc(h: Int, w: Int, c: Int) -> Tensor:
        var total = h * w * c
        var buf: List[UInt8] = List[UInt8]()
        var i = 0
        while i < total:
            buf.append(0)
            i += 1
        var (s0, s1, s2) = packed_hwc_strides(h, w, c)
        return Tensor(h, w, c, Layout.HWC(), DType.UInt8(), s0, s1, s2, buf)

    @staticmethod
fn zeros_chw(c: Int, h: Int, w: Int) -> Tensor:
        var total = h * w * c
        var buf: List[UInt8] = List[UInt8]()
        var i = 0
        while i < total:
            buf.append(0)
            i += 1
        var (s0, s1, s2) = packed_chw_strides(c, h, w)
        # logical dims still (h,w,c)
        return Tensor(h, w, c, Layout.CHW(), DType.UInt8(), s0, s1, s2, buf)

    # Metadata
fn height(self) -> Int: return self.h
fn width(self) -> Int: return self.w
fn channels(self) -> Int: return self.c
fn is_hwc(self) -> Bool: return self.layout == Layout.HWC()
fn is_chw(self) -> Bool: return self.layout == Layout.CHW()
fn size(self) -> Int: return self.h * self.w * self.c

# -------------------------
# Indexing
# -------------------------
@staticmethod
fn _offset(t: Tensor, x: Int, y: Int, ch: Int) -> Int:
    # Returns flat byte index for element at (x,y,ch) in logical image coords.
    if t.layout == Layout.HWC():
        # order: (y, x, ch)
        return (y * t.s0) + (x * t.s1) + (ch * t.s2)
    # CHW layout; order: (ch, y, x)
    return (ch * t.s0) + (y * t.s1) + (x * t.s2)

@staticmethod
fn get_u8(t: Tensor, x: Int, y: Int, ch: Int) -> UInt8:
    var idx = _offset(t, x, y, ch)
    return t.data[idx]

@staticmethod
fn set_u8(mut t: Tensor, x: Int, y: Int, ch: Int, v: UInt8) -> Tensor:
    var idx = _offset(t, x, y, ch)
    t.data[idx] = v
    return t

# -------------------------
# Layout conversions (copying)
# -------------------------
@staticmethod
fn to_packed_hwc(t: Tensor) -> Tensor:
    if t.is_hwc():
        # ensure contiguous; if already packed return copy with packed strides
        var (ps0, ps1, ps2) = packed_hwc_strides(t.h, t.w, t.c)
        var out = Tensor.zeros_hwc(t.h, t.w, t.c)
        var y = 0
        while y < t.h:
            var x = 0
            while x < t.w:
                var ch = 0
                while ch < t.c:
                    out = set_u8(out, x, y, ch, get_u8(t, x, y, ch))
                    ch += 1
                x += 1
            y += 1
        # set packed strides explicitly
        out.s0 = ps0; out.s1 = ps1; out.s2 = ps2
        return out
    # CHW -> HWC
    var out2 = Tensor.zeros_hwc(t.h, t.w, t.c)
    var y2 = 0
    while y2 < t.h:
        var x2 = 0
        while x2 < t.w:
            var ch2 = 0
            while ch2 < t.c:
                out2 = set_u8(out2, x2, y2, ch2, get_u8(t, x2, y2, ch2))
                ch2 += 1
            x2 += 1
        y2 += 1
    return out2

@staticmethod
fn to_packed_chw(t: Tensor) -> Tensor:
    if t.is_chw():
        var (ps0, ps1, ps2) = packed_chw_strides(t.c, t.h, t.w)
        var out = Tensor.zeros_chw(t.c, t.h, t.w)
        var ch = 0
        while ch < t.c:
            var y = 0
            while y < t.h:
                var x = 0
                while x < t.w:
                    out = set_u8(out, x, y, ch, get_u8(t, x, y, ch))
                    x += 1
                y += 1
            ch += 1
        out.s0 = ps0; out.s1 = ps1; out.s2 = ps2
        return out
    # HWC -> CHW
    var out2 = Tensor.zeros_chw(t.c, t.h, t.w)
    var ch2 = 0
    while ch2 < t.c:
        var y2 = 0
        while y2 < t.h:
            var x2 = 0
            while x2 < t.w:
                out2 = set_u8(out2, x2, y2, ch2, get_u8(t, x2, y2, ch2))
                x2 += 1
            y2 += 1
        ch2 += 1
    return out2

# -------------------------
# Basic transforms
# -------------------------
@staticmethod
fn crop_hwc(t: Tensor, x0: Int, y0: Int, cw: Int, ch: Int) -> Tensor:
    # Crop on logical image coords; result is HWC packed
    var x1 = x0 + cw
    var y1 = y0 + ch
    if x0 < 0: x0 = 0
    if y0 < 0: y0 = 0
    if x1 > t.w: x1 = t.w
    if y1 > t.h: y1 = t.h
    var ow = x1 - x0
    var oh = y1 - y0
    if ow <= 0 or oh <= 0:
        return Tensor.zeros_hwc(0, 0, t.c)
    var src = to_packed_hwc(t)
    var out = Tensor.zeros_hwc(oh, ow, t.c)
    var y = 0
    while y < oh:
        var x = 0
        while x < ow:
            var chn = 0
            while chn < t.c:
                out = set_u8(out, x, y, chn, get_u8(src, x0 + x, y0 + y, chn))
                chn += 1
            x += 1
        y += 1
    return out

@staticmethod
fn flip_horizontal(t: Tensor) -> Tensor:
    var src = to_packed_hwc(t)
    var out = Tensor.zeros_hwc(src.h, src.w, src.c)
    var y = 0
    while y < src.h:
        var x = 0
        while x < src.w:
            var chn = 0
            while chn < src.c:
                var v = get_u8(src, src.w - 1 - x, y, chn)
                out = set_u8(out, x, y, chn, v)
                chn += 1
            x += 1
        y += 1
    return out

@staticmethod
fn flip_vertical(t: Tensor) -> Tensor:
    var src = to_packed_hwc(t)
    var out = Tensor.zeros_hwc(src.h, src.w, src.c)
    var y = 0
    while y < src.h:
        var x = 0
        while x < src.w:
            var chn = 0
            while chn < src.c:
                var v = get_u8(src, x, src.h - 1 - y, chn)
                out = set_u8(out, x, y, chn, v)
                chn += 1
            x += 1
        y += 1
    return out

# -------------------------
# Summary
# -------------------------
@staticmethod
fn summary(t: Tensor) -> String:
    var s = String("Tensor(") + String(t.h) + String("x") + String(t.w) + String("x") + String(t.c) + String(", ")
    s = s + t.layout.to_string() + String(", ") + t.dtype.to_string() + String(", strides=(")
    s = s + String(t.s0) + String(",") + String(t.s1) + String(",") + String(t.s2) + String("))")
    return s

# -------------------------
# Minimal smoke test
# -------------------------
@staticmethod
fn __self_test__() -> Bool:
    # Build 2x2x3 RGB-like tensor in HWC
    var t = Tensor.zeros_hwc(2, 2, 3)
    # Fill a simple pattern on channel 0
    t = set_u8(t, 0, 0, 0, 10)
    t = set_u8(t, 1, 0, 0, 20)
    t = set_u8(t, 0, 1, 0, 30)
    t = set_u8(t, 1, 1, 0, 40)
    # Check roundtrip HWC -> CHW -> HWC
    var chw = to_packed_chw(t)
    if not chw.is_chw(): return False
    var hwc_rt = to_packed_hwc(chw)
    if not hwc_rt.is_hwc(): return False
    # Crop center 1x1 at (1,1)
    var cr = crop_hwc(hwc_rt, 1, 1, 1, 1)
    if cr.h != 1 or cr.w != 1 or cr.c != 3: return False
    # Flips keep shape
    var fh = flip_horizontal(hwc_rt)
    var fv = flip_vertical(hwc_rt)
    if fh.h != 2 or fh.w != 2: return False
    if fv.h != 2 or fv.w != 2: return False
    return True