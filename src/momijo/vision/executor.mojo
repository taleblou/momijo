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
# Project: momijo.vision
# File: src/momijo/vision/executor.mojo

from momijo.arrow_core.dtype_arrow import uint8
from momijo.core.device import id, kind
from momijo.core.version import major
from momijo.dataframe.diagnostics import safe
from momijo.dataframe.expr import Ops
from momijo.ir.dialects.annotations import integer
from momijo.nn.parameter import data
from momijo.tensor.tensor import index
from momijo.utils.result import g
from pathlib import Path
from pathlib.path import Path
from sys import version

# NOTE: Removed duplicate definition of `Layout`; use `from momijo.vision.image import Layout`
# NOTE: Removed duplicate definition of `__init__`; use `from momijo.core.version import __init__`

fn HWC() -> Layout:
        return Layout(1)
    @staticmethod
# NOTE: Removed duplicate definition of `CHW`; use `from momijo.vision.dtypes import CHW`
fn __eq__(self, other: Layout) -> Bool:
        return self.id == other.id

# -------------------------
# Image container: uint8 HWC only (by design, to keep it simple and fast)
# -------------------------
struct ImageU8HWC(Copyable, Movable):
    var h: Int
    var w: Int
    var c: Int
    var data: List[UInt8]
fn __init__(out self, h: Int, w: Int, c: Int, data: List[UInt8]) -> None:
        self.h = h
        self.w = w
        self.c = c
        self.data = data
        # basic shape validation
        var expected = h * w * c
        if len(self.data) != expected:
            # If sizes mismatch, initialize with zeros to be safe.
            var fixed: List[UInt8] = List[UInt8]()
            var i = 0
            while i < expected:
                fixed.append(0)
                i += 1
            self.data = fixed
fn layout(self) -> Layout:
        return Layout.HWC()
fn channels(self) -> Int:
        return self.c

# pixel access helpers
@staticmethod
fn _offset(h: Int, w: Int, c: Int, x: Int, y: Int, ch: Int) -> Int:
    # Row-major HWC: ((y * w) + x) * c + ch
    return ((y * w) + x) * c + ch

@staticmethod
fn get_px(img: ImageU8HWC, x: Int, y: Int, ch: Int) -> UInt8:
    var idx = _offset(img.h, img.w, img.c, x, y, ch)
    return img.data[idx]

@staticmethod
fn set_px(mut img: ImageU8HWC, x: Int, y: Int, ch: Int, v: UInt8) -> ImageU8HWC:
    var idx = _offset(img.h, img.w, img.c, x, y, ch)
    img.data[idx] = v
    return img

# -------------------------
# Ops definition via a small tagged-union style
# -------------------------
# NOTE: Removed duplicate definition of `OpKind`; use `from momijo.vision.ir.ir import OpKind`
# NOTE: Removed duplicate definition of `__init__`; use `from momijo.core.version import __init__`
fn ResizeNearest() -> OpKind:
        return OpKind(1)
    @staticmethod
fn RGBToGray() -> OpKind:
        return OpKind(2)
# NOTE: Removed duplicate definition of `__eq__`; use `from momijo.vision.ir.ir import __eq__`

struct Node(Copyable, Movable):
    var kind: OpKind
    var p1: Int
    var p2: Int
fn __init__(out self, kind: OpKind, p1: Int, p2: Int) -> None:
        self.kind = kind
        self.p1 = p1
        self.p2 = p2

# -------------------------
# Implementations
# -------------------------
@staticmethod
fn resize_nearest_u8_hwc(src: ImageU8HWC, oh: Int, ow: Int) -> ImageU8HWC:
    var out_data: List[UInt8] = List[UInt8]()
    var total = oh * ow * src.c
    var i = 0
    while i < total:
        out_data.append(0)
        i += 1

    var dst = ImageU8HWC(oh, ow, src.c, out_data)

    # precompute scale as float via integer ratio (avoid float types for simplicity)
    # nearest index calculation: sx = (x * src.w + ow//2) // ow (simple rounding)
    var y = 0
    while y < oh:
        var sx_num_y = y * src.h
        var sy = sx_num_y // oh
        var x = 0
        while x < ow:
            var sx = (x * src.w) // ow
            var ch = 0
            while ch < src.c:
                var v = get_px(src, sx, sy, ch)
                dst = set_px(dst, x, y, ch, v)
                ch += 1
            x += 1
        y += 1
    return dst

@staticmethod
fn rgb_to_gray(src: ImageU8HWC) -> ImageU8HWC:
    # Expect c == 3; if not, pass-through
    if src.c != 3:
        return src
    var out_data: List[UInt8] = List[UInt8]()
    var total = src.h * src.w
    var i = 0
    while i < total:
        out_data.append(0)
        i += 1
    var dst = ImageU8HWC(src.h, src.w, 1, out_data)

    var y = 0
    while y < src.h:
        var x = 0
        while x < src.w:
            # integer-approx luma: (0.299, 0.587, 0.114) -> (77, 150, 29) / 256
            var r = get_px(src, x, y, 0)
            var g = get_px(src, x, y, 1)
            var b = get_px(src, x, y, 2)
            var gray_u16 = UInt16(77) * UInt16(r) + UInt16(150) * UInt16(g) + UInt16(29) * UInt16(b)
            var gray = UInt8((gray_u16 >> UInt8(8)) & UInt16(0xFF))
            dst = set_px(dst, x, y, 0, gray)
            x += 1
        y += 1
    return dst

# -------------------------
# Executor
# -------------------------
struct Executor(Copyable, Movable):
    var nodes: List[Node]
fn __init__(out self) -> None:
        self.nodes = List[Node]()
fn add_resize(mut self, oh: Int, ow: Int) -> Executor:
        self.nodes.append(Node(OpKind.ResizeNearest(), oh, ow))
        return self
fn add_rgb_to_gray(mut self) -> Executor:
        self.nodes.append(Node(OpKind.RGBToGray(), 0, 0))
        return self

# Run the pipeline
@staticmethod
fn run_pipeline(exec: Executor, src: ImageU8HWC) -> ImageU8HWC:
    var cur = src
    var i = 0
    while i < len(exec.nodes):
        var n = exec.nodes[i]
        if n.kind == OpKind.ResizeNearest():
            cur = resize_nearest_u8_hwc(cur, n.p1, n.p2)
        elif n.kind == OpKind.RGBToGray():
            cur = rgb_to_gray(cur)
        i += 1
    return cur

# Single-op helper
@staticmethod
fn run_single_op(kind: OpKind, src: ImageU8HWC, oh: Int, ow: Int) -> ImageU8HWC:
    if kind == OpKind.ResizeNearest():
        return resize_nearest_u8_hwc(src, oh, ow)
    if kind == OpKind.RGBToGray():
        return rgb_to_gray(src)
    return src

# -------------------------
# Minimal smoke test
# -------------------------
@staticmethod
fn __self_test__() -> Bool:
    var h = 2
    var w = 2
    var c = 3
    # 2x2 RGB test pattern:
    # (255,0,0) (0,255,0)
    # (0,0,255) (255,255,255)
    var data: List[UInt8] = List[UInt8]()
    data.append(255); data.append(0);   data.append(0)
    data.append(0);   data.append(255); data.append(0)
    data.append(0);   data.append(0);   data.append(255)
    data.append(255); data.append(255); data.append(255)

    var img = ImageU8HWC(h, w, c, data)
    var ex = Executor()
    ex = ex.add_rgb_to_gray()
    ex = ex.add_resize(4, 4)
    var out = run_pipeline(ex, img)
    # Check shape
    if out.h != 4 or out.w != 4 or out.c != 1:
        return False
    return True