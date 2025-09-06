# MIT License
# Copyright (c) 2025 Morteza Talebou and Mitra Daneshmand
# Project: momijo  |  Source: https://github.com/taleblou/momijo
# This file is part of the Momijo project. See the LICENSE file at the repository root.
# Momijo
# SPDX-License-Identifier: MIT
# Website: https://taleblou.ir/
# Repository: https://github.com/taleblou/momijo
#
# Project: momijo.vision.adapters
# File: src/momijo/vision/adapters/arrow.mojo
#
# Minimal Arrow-style adapter for Vision tensors (uint8, HWC) using raw pointers.
# Goal: Provide zero-copy views over external buffers (e.g., Arrow) without
#       importing other Momijo modules. No allocation/free here.
# Style: no 'export', no 'let', no 'inout'; constructors as `fn __init__(out self, ...)`.
#
# NOTE: This is not Apache Arrow proper; it's an adapter that assumes you already
#       hold a contiguous uint8 buffer laid out as HWC. When Momijo Arrow becomes
#       stable, this adapter can switch to real Arrow arrays.

# -------------------------
# Tags & helpers
# -------------------------
struct DType(Copyable, Movable):
    var id: Int
    fn __init__(out self, id: Int):
        self.id = id
    @staticmethod
    fn UInt8() -> DType: return DType(1)
    fn __eq__(self, other: DType) -> Bool: return self.id == other.id
    fn to_string(self) -> String:
        if self.id == 1: return String("UInt8")
        return String("UnknownDType")

@staticmethod
fn dtype_bytes(dt: DType) -> Int:
    if dt == DType.UInt8(): return 1
    return 1

struct Layout(Copyable, Movable):
    var id: Int
    fn __init__(out self, id: Int):
        self.id = id
    @staticmethod
    fn HWC() -> Layout: return Layout(1)
    fn __eq__(self, other: Layout) -> Bool: return self.id == other.id
    fn to_string(self) -> String:
        if self.id == 1: return String("HWC")
        return String("UnknownLayout")

@staticmethod
fn packed_hwc_strides(h: Int, w: Int, c: Int) -> (Int, Int, Int):
    # (s0, s1, s2) for (y, x, ch)
    var s2 = 1
    var s1 = c
    var s0 = w * c
    return (s0, s1, s2)

# -------------------------
# Pointer-based tensor view (uint8, HWC)
# -------------------------
struct TensorU8HWCView(Copyable, Movable):
    var data: Pointer[UInt8]
    var nbytes: Int
    var h: Int
    var w: Int
    var c: Int
    var s0: Int
    var s1: Int
    var s2: Int
    var dtype: DType
    var layout: Layout

    fn __init__(out self,
                data: Pointer[UInt8],
                nbytes: Int,
                h: Int,
                w: Int,
                c: Int,
                s0: Int,
                s1: Int,
                s2: Int):
        self.data = data
        self.nbytes = nbytes
        self.h = h
        self.w = w
        self.c = c
        self.s0 = s0
        self.s1 = s1
        self.s2 = s2
        self.dtype = DType.UInt8()
        self.layout = Layout.HWC()

    # Meta
    fn height(self) -> Int: return self.h
    fn width(self) -> Int: return self.w
    fn channels(self) -> Int: return self.c
    fn byte_len(self) -> Int: return self.nbytes

# Create a view from a contiguous HWC array (no copy)
@staticmethod
fn tensor_from_array_u8_hwc(data: Pointer[UInt8], nbytes: Int, h: Int, w: Int, c: Int) -> TensorU8HWCView:
    var expected = h * w * c * dtype_bytes(DType.UInt8())
    # We don't change dims if nbytes < expected; caller is responsible.
    var (s0, s1, s2) = packed_hwc_strides(h, w, c)
    return TensorU8HWCView(data, nbytes, h, w, c, s0, s1, s2)

# Return the underlying pointer (no copy)
@staticmethod
fn array_from_tensor_u8_hwc(t: TensorU8HWCView) -> Pointer[UInt8]:
    return t.data

# ROI view (no copy): x:[x0,x0+cw), y:[y0,y0+ch)
@staticmethod
fn view_roi_hwc(t: TensorU8HWCView, x0: Int, y0: Int, cw: Int, ch: Int) -> TensorU8HWCView:
    # Clamp to bounds
    var x1 = x0 + cw
    var y1 = y0 + ch
    if x0 < 0: x0 = 0
    if y0 < 0: y0 = 0
    if x1 > t.w: x1 = t.w
    if y1 > t.h: y1 = t.h
    var ow = x1 - x0
    var oh = y1 - y0
    if ow <= 0 or oh <= 0:
        return TensorU8HWCView(Pointer[UInt8](), 0, 0, 0, t.c, t.s0, t.s1, t.s2)
    # Compute byte offset of (x0, y0, ch=0)
    var elem = dtype_bytes(t.dtype)
    var offset_elems = (y0 * t.s0) + (x0 * t.s1)  # + ch*s2 (ch=0)
    var offset_bytes = offset_elems * elem
    var new_ptr = t.data + offset_bytes
    # Strides unchanged for the cropped view; byte_len is conservative
    var new_bytes = ow * oh * t.c * elem
    var (s0, s1, s2) = (t.s0, t.s1, t.s2)
    return TensorU8HWCView(new_ptr, new_bytes, oh, ow, t.c, s0, s1, s2)

# -------------------------
# Summary helper
# -------------------------
@staticmethod
fn summary(t: TensorU8HWCView) -> String:
    var s = String("TensorU8HWCView(") + String(t.h) + String("x") + String(t.w) + String("x") + String(t.c) + String(", ")
    s = s + t.layout.to_string() + String(", ") + t.dtype.to_string() + String(", strides=(")
    s = s + String(t.s0) + String(",") + String(t.s1) + String(",") + String(t.s2) + String("), nbytes=") + String(t.nbytes) + String(")")
    return s

# -------------------------
# Minimal smoke test (stride math only; does not dereference the pointer)
# -------------------------
@staticmethod
fn __self_test__() -> Bool:
    var dummy = Pointer[UInt8]()   # null pointer; we don't read from it
    var h = 3; var w = 5; var c = 3
    var (s0, s1, s2) = packed_hwc_strides(h, w, c) # (15, 3, 1)
    var nbytes = h * w * c * dtype_bytes(DType.UInt8())
    var view = TensorU8HWCView(dummy, nbytes, h, w, c, s0, s1, s2)
    if view.byte_len() != 45: return False
    # ROI math check: center 1x1
    var roi = view_roi_hwc(view, 2, 1, 1, 1)
    if not (roi.h == 1 and roi.w == 1 and roi.c == 3): return False
    # roundtrip pointer
    var p = array_from_tensor_u8_hwc(view)
    # We only check that returning didn't crash and type matches
    return True
