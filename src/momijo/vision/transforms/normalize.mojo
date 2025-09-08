# Project:      Momijo
# Module:       src.momijo.vision.transforms.normalize
# File:         normalize.mojo
# Path:         src/momijo/vision/transforms/normalize.mojo
#
# Description:  src.momijo.vision.transforms.normalize — focused Momijo functionality with a stable public API.
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
#   - Structs: ChannelOrder, ImageU8, ImageF32
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

struct ImageU8(Copyable, Movable):
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

struct ImageF32(Copyable, Movable):
    var h: Int
    var w: Int
    var order: ChannelOrder
    var data: List[Float32]
fn __init__(out self, h: Int, w: Int, order: ChannelOrder, data: List[Float32]) -> None:
        self.h = h; self.w = w; self.order = order
        var expected = h*w*_num_ch(order)
        if len(data) != expected:
            var buf: List[Float32] = List[Float32]()
            var i = 0
            while i < expected: buf.append(Float32(0.0)); i += 1
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
    while i < n: out.append(0); i += 1
    return out

@staticmethod
fn _alloc_f32(n: Int) -> List[Float32]:
    var out: List[Float32] = List[Float32]()
    var i = 0
    while i < n: out.append(Float32(0.0)); i += 1
    return out

@staticmethod
fn _clamp_u8(v: Int) -> UInt8:
    if v < 0: return UInt8(0)
    if v > 255: return UInt8(255)
    return UInt8(v)

@staticmethod
fn _abs_f(x: Float32) -> Float32:
    if x < Float32(0.0): return -x
    return x

# -------------------------
# Min/Max stats (UInt8, per-channel)
# -------------------------
@staticmethod
fn compute_minmax_u8_per_channel(h: Int, w: Int, c: Int, src: List[UInt8]) -> (List[Int], List[Int]):
    var mins: List[Int] = List[Int]()
    var maxs: List[Int] = List[Int]()
    var k = 0
    while k < c:
        mins.append(255)
        maxs.append(0)
        k += 1
    var i = 0
    while i < h*w:
        var ch = 0
        while ch < c:
            var v = Int(src[i*c + ch])
            if v < mins[ch]: mins[ch] = v
            if v > maxs[ch]: maxs[ch] = v
            ch += 1
        i += 1
    return (mins, maxs)

# -------------------------
# Min/Max normalization to UInt8
# -------------------------
@staticmethod
fn normalize_minmax_u8_global(h: Int, w: Int, c: Int, src: List[UInt8], out_min: Int, out_max: Int) -> List[UInt8]:
    # Uses global min/max over all channels
    var mn = 255; var mx = 0
    var i = 0
    while i < h*w*c:
        var v = Int(src[i])
        if v < mn: mn = v
        if v > mx: mx = v
        i += 1
    if mx == mn:
        # constant image -> map to mid
        var out = _alloc_u8(h*w*c)
        var mid = (out_min + out_max) // 2
        var j = 0
        while j < h*w*c: out[j] = _clamp_u8(mid); j += 1
        return out
    var out = _alloc_u8(h*w*c)
    var scale_num = out_max - out_min
    i = 0
    while i < h*w*c:
        var v = Int(src[i])
        var t = ( (v - mn) * scale_num ) // (mx - mn) + out_min
        out[i] = _clamp_u8(t)
        i += 1
    return out

@staticmethod
fn normalize_minmax_u8_per_channel(h: Int, w: Int, c: Int, src: List[UInt8], out_min: Int, out_max: Int) -> List[UInt8]:
    var mins: List[Int] = List[Int](); var maxs: List[Int] = List[Int]()
    (mins, maxs) = compute_minmax_u8_per_channel(h, w, c, src)
    var out = _alloc_u8(h*w*c)
    var i = 0
    while i < h*w:
        var ch = 0
        while ch < c:
            var mn = mins[ch]; var mx = maxs[ch]
            var v = Int(src[i*c + ch])
            var t = 0
            if mx == mn:
                t = (out_min + out_max) // 2
            else:
                t = ( (v - mn) * (out_max - out_min) ) // (mx - mn) + out_min
            out[i*c + ch] = _clamp_u8(t)
            ch += 1
        i += 1
    return out

# -------------------------
# UInt8 <-> Float32 (unit scaling)
# -------------------------
@staticmethod
fn u8_to_f32_unit(h: Int, w: Int, c: Int, src: List[UInt8]) -> List[Float32]:
    var out = _alloc_f32(h*w*c)
    var i = 0
    while i < h*w*c:
        out[i] = Float32(Int(src[i])) / Float32(255.0)
        i += 1
    return out

@staticmethod
fn f32_unit_to_u8(h: Int, w: Int, c: Int, src: List[Float32]) -> List[UInt8]:
    var out = _alloc_u8(h*w*c)
    var i = 0
    while i < h*w*c:
        var v = src[i]
        if v < Float32(0.0): v = Float32(0.0)
        if v > Float32(1.0): v = Float32(1.0)
        var iv = Int(v * Float32(255.0) + Float32(0.5))
        out[i] = _clamp_u8(iv)
        i += 1
    return out

# -------------------------
# Mean/Std normalization (Float32, per-channel)
# out = (x - mean[ch]) / max(std[ch], eps)
# -------------------------
@staticmethod
fn standardize_per_channel_f32(h: Int, w: Int, c: Int, src: List[Float32], means: List[Float32], stds: List[Float32], eps: Float32) -> List[Float32]:
    var out = _alloc_f32(h*w*c)
    var i = 0
    while i < h*w:
        var ch = 0
        while ch < c:
            var m = means[ch]
            var s = stds[ch]
            if s < eps: s = eps
            var v = src[i*c + ch]
            out[i*c + ch] = (v - m) / s
            ch += 1
        i += 1
    return out

# -------------------------
# Image wrappers
# -------------------------
@staticmethod
fn apply_u8_to_unit_f32(img: ImageU8) -> ImageF32:
    var c = _num_ch(img.order)
    var buf = u8_to_f32_unit(img.h, img.w, c, img.data)
    return ImageF32(img.h, img.w, img.order, buf)

@staticmethod
fn apply_unit_f32_to_u8(img: ImageF32) -> ImageU8:
    var c = _num_ch(img.order)
    var buf = f32_unit_to_u8(img.h, img.w, c, img.data)
    return ImageU8(img.h, img.w, img.order, buf)

@staticmethod
fn apply_standardize_u8_to_f32(img: ImageU8, means: List[Float32], stds: List[Float32]) -> ImageF32:
    var c = _num_ch(img.order)
    var f = u8_to_f32_unit(img.h, img.w, c, img.data)
    var eps = Float32(1e-6)
    var z = standardize_per_channel_f32(img.h, img.w, c, f, means, stds, eps)
    return ImageF32(img.h, img.w, img.order, z)

@staticmethod
fn apply_minmax_u8_global(img: ImageU8, out_min: Int, out_max: Int) -> ImageU8:
    var c = _num_ch(img.order)
    var buf = normalize_minmax_u8_global(img.h, img.w, c, img.data, out_min, out_max)
    return ImageU8(img.h, img.w, img.order, buf)

@staticmethod
fn apply_minmax_u8_per_channel(img: ImageU8, out_min: Int, out_max: Int) -> ImageU8:
    var c = _num_ch(img.order)
    var buf = normalize_minmax_u8_per_channel(img.h, img.w, c, img.data, out_min, out_max)
    return ImageU8(img.h, img.w, img.order, buf)

# -------------------------
# Minimal smoke test
# -------------------------
@staticmethod
fn __self_test__() -> Bool:
    # 2x2 RGB with strong values
    var data: List[UInt8] = List[UInt8]()
    # row0: (255,0,0) (0,255,0)
    data.append(255); data.append(0);   data.append(0)
    data.append(0);   data.append(255); data.append(0)
    # row1: (0,0,255) (128,128,128)
    data.append(0);   data.append(0);   data.append(255)
    data.append(128); data.append(128); data.append(128)

    var img = ImageU8(2, 2, ChannelOrder.RGB(), data)

    # 1) UInt8 -> Float32 [0,1]
    var f = apply_u8_to_unit_f32(img)
    if not (len(f.data) == 12): return False
    # Check first component ~ 1.0 and second ~ 0.0
    var a0 = f.data[0]; var a1 = f.data[1]
    if a0 < Float32(0.99) or a0 > Float32(1.01): return False
    if a1 < Float32(-0.01) or a1 > Float32(0.01): return False

    # 2) Per-channel minmax to [0..255]
    var mm = apply_minmax_u8_per_channel(img, 0, 255)
    if not (len(mm.data) == 12): return False
    # R channel had max 255, min 0 -> first value should remain 255
    if mm.data[0] != UInt8(255): return False

    # 3) Mean/Std standardization (ImageNet means/stds style)
    var means: List[Float32] = List[Float32](); means.append(Float32(0.485)); means.append(Float32(0.456)); means.append(Float32(0.406))
    var stds:  List[Float32] = List[Float32](); stds.append(Float32(0.229));  stds.append(Float32(0.224));  stds.append(Float32(0.225))
    var z = apply_standardize_u8_to_f32(img, means, stds)
    if not (len(z.data) == 12): return False

    # 4) Roundtrip f32 [0,1] -> u8
    var u = apply_unit_f32_to_u8(f)
    if not (len(u.data) == 12 and u.data[0] == UInt8(255) and u.data[1] == UInt8(0)): return False

    return True