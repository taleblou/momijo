# MIT License
# Copyright (c) 2025 
# Project: momijo  |  Source: https://github.com/taleblou/momijo
# SPDX-License-Identifier: MIT
#
# Project: momijo.vision
# File: src/momijo/vision/to_tensor.mojo
#
# Standalone HWC(UInt8) → Tensor(Float32) converters for Momijo Vision.
# Style: no 'export', no 'let', no 'inout'. Constructors use `fn __init__(out self, ...)`.
#
# Implemented:
#   - ChannelOrder + ImageU8
#   - TensorF32 (shape: List[Int], data: List[Float32])
#   - HWC(UInt8) → CHW(Float32) unit-scaled [0,1]
#   - HWC(UInt8) → CHW(Float32) normalized by mean/std (per-channel)
#   - Batch packers: List[ImageU8] → NCHW(Float32) (unit or mean/std)
#   - Options handled:
#       * RGB/BGR/GRAY/RGBA/BGRA inputs
#       * Drop alpha if present
#       * Expand GRAY → 3-channel by replication (optional)
#       * BGR → RGB channel swap for model-friendly layout
#   - __self_test__()
#
# Notes:
# - This module is dependency-light and self-contained for early integration.
# - Mean/Std arrays must match output channel count (1 or 3).

# -------------------------
# ChannelOrder + ImageU8
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

struct ImageU8(Copyable, Movable):
    var h: Int
    var w: Int
    var order: ChannelOrder
    var data: List[UInt8]
    fn __init__(out self, h: Int, w: Int, order: ChannelOrder, data: List[UInt8]):
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
# TensorF32
# -------------------------
struct TensorF32(Copyable, Movable):
    var shape: List[Int]      # e.g., [C,H,W] or [N,C,H,W]
    var data: List[Float32]   # row-major for the given shape
    fn __init__(out self, shape: List[Int], data: List[Float32]):
        self.shape = shape
        self.data = data

# -------------------------
# Helpers
# -------------------------
@staticmethod
fn _offset_hwc(w: Int, c: Int, x: Int, y: Int, ch: Int) -> Int:
    return ((y * w) + x) * c + ch

@staticmethod
fn _alloc_f32(n: Int) -> List[Float32]:
    var out: List[Float32] = List[Float32]()
    var i = 0
    while i < n: out.append(Float32(0.0)); i += 1
    return out

@staticmethod
fn _append_f32(mut v: List[Float32], x: Float32) -> List[Float32]:
    v.append(x); return v

@staticmethod
fn _min(a: Int, b: Int) -> Int:
    if a < b: return a
    return b

# Map various input orders to RGB triple (ints)
@staticmethod
fn _read_rgb_triplet(order: ChannelOrder, base: Int, c: Int, buf: List[UInt8]) -> (Int, Int, Int):
    if order == ChannelOrder.RGB():
        return (Int(buf[base+0]), Int(buf[base+1]), Int(buf[base+2]))
    if order == ChannelOrder.BGR():
        return (Int(buf[base+2]), Int(buf[base+1]), Int(buf[base+0]))
    if order == ChannelOrder.RGBA():
        return (Int(buf[base+0]), Int(buf[base+1]), Int(buf[base+2]))
    if order == ChannelOrder.BGRA():
        return (Int(buf[base+2]), Int(buf[base+1]), Int(buf[base+0]))
    # GRAY -> replicate
    var g = Int(buf[base+0])
    return (g, g, g)

# -------------------------
# Core converters (single image)
# -------------------------
@staticmethod
fn to_tensor_chw_unit(img: ImageU8, expand_gray_to3: Bool) -> TensorF32:
    # Output: CHW Float32 in [0,1], RGB channel order (C ∈ {1,3})
    var h = img.h; var w = img.w; var c_in = _num_ch(img.order)
    var c_out = 3
    if img.order == ChannelOrder.GRAY() and not expand_gray_to3:
        c_out = 1
    var out = _alloc_f32(c_out * h * w)

    if c_out == 1:
        var i = 0
        while i < h*w:
            out[i] = Float32(Int(img.data[i])) / Float32(255.0)
            i += 1
        return TensorF32([1, h, w], out)

    # c_out == 3
    var y = 0
    while y < h:
        var x = 0
        while x < w:
            var base = _offset_hwc(w, c_in, x, y, 0)
            var (r, g, b) = _read_rgb_triplet(img.order, base, c_in, img.data)
            var idx = y*w + x
            out[0*h*w + idx] = Float32(r) / Float32(255.0)
            out[1*h*w + idx] = Float32(g) / Float32(255.0)
            out[2*h*w + idx] = Float32(b) / Float32(255.0)
            x += 1
        y += 1
    return TensorF32([3, h, w], out)

@staticmethod
fn to_tensor_chw_norm(img: ImageU8, means: List[Float32], stds: List[Float32], expand_gray_to3: Bool) -> TensorF32:
    # Output: CHW Float32, normalized as (x/255 - mean)/std per channel
    var h = img.h; var w = img.w; var c_in = _num_ch(img.order)
    var c_out = 3
    if img.order == ChannelOrder.GRAY() and not expand_gray_to3:
        c_out = 1
    # expect means/stds length == c_out; if not, clamp
    var m0 = Float32(0.0); var m1 = Float32(0.0); var m2 = Float32(0.0)
    var s0 = Float32(1.0); var s1 = Float32(1.0); var s2 = Float32(1.0)
    if c_out == 1:
        if len(means) > 0: m0 = means[0]
        if len(stds)  > 0: s0 = stds[0]
    else:
        if len(means) > 0: m0 = means[0]
        if len(means) > 1: m1 = means[1]
        if len(means) > 2: m2 = means[2]
        if len(stds)  > 0: s0 = stds[0]
        if len(stds)  > 1: s1 = stds[1]
        if len(stds)  > 2: s2 = stds[2]
        if s0 == Float32(0.0): s0 = Float32(1.0)
        if s1 == Float32(0.0): s1 = Float32(1.0)
        if s2 == Float32(0.0): s2 = Float32(1.0)

    var out = _alloc_f32(c_out * h * w)

    if c_out == 1:
        var i = 0
        while i < h*w:
            var v = Float32(Int(img.data[i])) / Float32(255.0)
            out[i] = (v - m0) / s0
            i += 1
        return TensorF32([1, h, w], out)

    var y = 0
    while y < h:
        var x = 0
        while x < w:
            var base = _offset_hwc(w, c_in, x, y, 0)
            var (r, g, b) = _read_rgb_triplet(img.order, base, c_in, img.data)
            var idx = y*w + x
            var rf = Float32(r) / Float32(255.0)
            var gf = Float32(g) / Float32(255.0)
            var bf = Float32(b) / Float32(255.0)
            out[0*h*w + idx] = (rf - m0) / s0
            out[1*h*w + idx] = (gf - m1) / s1
            out[2*h*w + idx] = (bf - m2) / s2
            x += 1
        y += 1
    return TensorF32([3, h, w], out)

# -------------------------
# Batch packers: List[ImageU8] → NCHW
# -------------------------
@staticmethod
fn batch_to_nchw_unit(imgs: List[ImageU8], expand_gray_to3: Bool) -> TensorF32:
    var n = len(imgs)
    if n == 0:
        return TensorF32([0,0,0,0], List[Float32]())

    var h0 = imgs[0].h; var w0 = imgs[0].w
    var c0 = 3
    if imgs[0].order == ChannelOrder.GRAY() and not expand_gray_to3: c0 = 1
    var out: List[Float32] = List[Float32]()
    var i = 0
    while i < n:
        var t = to_tensor_chw_unit(imgs[i], expand_gray_to3)
        # enforce same shape (H,W,C)
        if t.shape[1] != h0 or t.shape[2] != w0 or t.shape[0] != c0:
            # simple fallback: skip mismatched sizes
            i += 1
            continue
        # append data
        var k = 0
        while k < len(t.data):
            out.append(t.data[k]); k += 1
        i += 1
    return TensorF32([n, c0, h0, w0], out)

@staticmethod
fn batch_to_nchw_norm(imgs: List[ImageU8], means: List[Float32], stds: List[Float32], expand_gray_to3: Bool) -> TensorF32:
    var n = len(imgs)
    if n == 0:
        return TensorF32([0,0,0,0], List[Float32]())

    var h0 = imgs[0].h; var w0 = imgs[0].w
    var c0 = 3
    if imgs[0].order == ChannelOrder.GRAY() and not expand_gray_to3: c0 = 1
    var out: List[Float32] = List[Float32]()
    var i = 0
    while i < n:
        var t = to_tensor_chw_norm(imgs[i], means, stds, expand_gray_to3)
        if t.shape[1] != h0 or t.shape[2] != w0 or t.shape[0] != c0:
            i += 1
            continue
        var k = 0
        while k < len(t.data):
            out.append(t.data[k]); k += 1
        i += 1
    return TensorF32([n, c0, h0, w0], out)

# -------------------------
# Minimal smoke test
# -------------------------
@staticmethod
fn __self_test__() -> Bool:
    # 2x2 BGR → CHW RGB
    var bgr: List[UInt8] = List[UInt8]()
    # px0 (B,G,R)=(0,0,255) ; px1=(0,255,0) ; px2=(255,0,0) ; px3=(255,255,255)
    bgr.append(0); bgr.append(0);   bgr.append(255)
    bgr.append(0); bgr.append(255); bgr.append(0)
    bgr.append(255); bgr.append(0); bgr.append(0)
    bgr.append(255); bgr.append(255); bgr.append(255)
    var img_bgr = ImageU8(2, 2, ChannelOrder.BGR(), bgr)

    var t = to_tensor_chw_unit(img_bgr, True)
    if not (len(t.data) == 3*2*2 and t.shape[0] == 3 and t.shape[1] == 2 and t.shape[2] == 2): return False
    # Check first channel (R) first value ~ 1.0
    var r0 = t.data[0*4 + 0]
    if r0 < Float32(0.99) or r0 > Float32(1.01): return False

    # GRAY 2x2 expand→3
    var gy: List[UInt8] = List[UInt8]()
    gy.append(0); gy.append(128); gy.append(200); gy.append(255)
    var img_g = ImageU8(2,2, ChannelOrder.GRAY(), gy)
    var tg = to_tensor_chw_unit(img_g, True)
    if not (tg.shape[0] == 3 and len(tg.data) == 12): return False

    # Normalized
    var means: List[Float32] = List[Float32](); means.append(Float32(0.5)); means.append(Float32(0.5)); means.append(Float32(0.5))
    var stds:  List[Float32] = List[Float32](); stds.append(Float32(0.5));  stds.append(Float32(0.5));  stds.append(Float32(0.5))
    var tn = to_tensor_chw_norm(img_bgr, means, stds, True)
    if not (len(tn.data) == 12): return False

    # Batch
    var batch: List[ImageU8] = List[ImageU8](); batch.append(img_bgr); batch.append(img_bgr)
    var nb = batch_to_nchw_unit(batch, True)
    if not (nb.shape[0] == 2 and nb.shape[1] == 3 and nb.shape[2] == 2 and nb.shape[3] == 2): return False

    return True
