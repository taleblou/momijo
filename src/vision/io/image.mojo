# MIT License
# Copyright (c) 2025 Morteza Talebou and Mitra Daneshmand
# Project: momijo  |  Source: https://github.com/taleblou/momijo
# This file is part of the Momijo project. See the LICENSE file at the repository root.
# Momijo
# SPDX-License-Identifier: MIT
# Website: https://taleblou.ir/
# Repository: https://github.com/taleblou/momijo
#
# Project: momijo.vision.io
# File: src/momijo/vision/io/image.mojo
#
# Minimal, dependency-light Image IO shim for Momijo Vision.
# Goals:
#  - Provide a stable Image container and construction helpers usable before full IO stack lands.
#  - Avoid imports from WIP modules; keep this file self-contained and compilable.
#  - Define function names that can be easily wired to real decoders later.
# Style:
#  - No 'export', no 'let', no 'inout'.
#  - Constructors use `fn __init__(out self, ...)`.
#
# NOTE: This is not a real file decoder. The `read_image_*` funcs construct images
#       from already-available buffers. When momijo.io decoders are ready, they can
#       replace or call into these helpers.

# -------------------------
# Local tag types (kept in-sync with momijo.vision.dtypes later)
# -------------------------
struct ChannelOrder(Copyable, Movable):
    var id: Int
    fn __init__(out self, id: Int): self.id = id
    @staticmethod fn GRAY() -> ChannelOrder: return ChannelOrder(1)
    @staticmethod fn RGB()  -> ChannelOrder: return ChannelOrder(2)
    @staticmethod fn BGR()  -> ChannelOrder: return ChannelOrder(3)
    @staticmethod fn RGBA() -> ChannelOrder: return ChannelOrder(4)
    @staticmethod fn BGRA() -> ChannelOrder: return ChannelOrder(5)
    @staticmethod fn ARGB() -> ChannelOrder: return ChannelOrder(6)
    fn __eq__(self, other: ChannelOrder) -> Bool: return self.id == other.id
    fn to_string(self) -> String:
        if self.id == 1: return String("GRAY")
        if self.id == 2: return String("RGB")
        if self.id == 3: return String("BGR")
        if self.id == 4: return String("RGBA")
        if self.id == 5: return String("BGRA")
        if self.id == 6: return String("ARGB")
        return String("UNKNOWN")

@staticmethod
fn num_channels(order: ChannelOrder) -> Int:
    if order == ChannelOrder.GRAY(): return 1
    if order == ChannelOrder.RGB() or order == ChannelOrder.BGR(): return 3
    return 4

struct AlphaMode(Copyable, Movable):
    var id: Int
    fn __init__(out self, id: Int): self.id = id
    @staticmethod fn NONE() -> AlphaMode:          return AlphaMode(0)
    @staticmethod fn STRAIGHT() -> AlphaMode:      return AlphaMode(1)
    @staticmethod fn PREMULTIPLIED() -> AlphaMode: return AlphaMode(2)
    fn __eq__(self, other: AlphaMode) -> Bool: return self.id == other.id
    fn to_string(self) -> String:
        if self.id == 0: return String("NONE")
        if self.id == 1: return String("STRAIGHT")
        if self.id == 2: return String("PREMULTIPLIED")
        return String("UNKNOWN")

struct Layout(Copyable, Movable):
    var id: Int
    fn __init__(out self, id: Int): self.id = id
    @staticmethod fn HWC() -> Layout: return Layout(1)
    @staticmethod fn CHW() -> Layout: return Layout(2)
    fn __eq__(self, other: Layout) -> Bool: return self.id == other.id
    fn to_string(self) -> String:
        if self.id == 1: return String("HWC")
        if self.id == 2: return String("CHW")
        return String("UNKNOWN")

# -------------------------
# Image container (U8/HWC storage)
# -------------------------
struct Image(Copyable, Movable):
    var h: Int
    var w: Int
    var order: ChannelOrder
    var alpha: AlphaMode
    var layout: Layout      # always HWC for storage
    var data: List[UInt8]

    fn __init__(out self, h: Int, w: Int, order: ChannelOrder, alpha: AlphaMode, data: List[UInt8]):
        self.h = h; self.w = w; self.order = order; self.alpha = alpha
        self.layout = Layout.HWC()
        var c = num_channels(order)
        var expected = h * w * c
        if len(data) != expected:
            var fixed: List[UInt8] = List[UInt8]()
            var i = 0
            while i < expected:
                fixed.append(0)
                i += 1
            self.data = fixed
        else:
            self.data = data

    @staticmethod
    fn zeros(h: Int, w: Int, order: ChannelOrder, alpha: AlphaMode) -> Image:
        var c = num_channels(order)
        var total = h * w * c
        var buf: List[UInt8] = List[UInt8]()
        var i = 0
        while i < total:
            buf.append(0)
            i += 1
        return Image(h, w, order, alpha, buf)

    # metadata
    fn height(self) -> Int: return self.h
    fn width(self)  -> Int: return self.w
    fn channels(self) -> Int: return num_channels(self.order)
    fn size(self) -> Int: return self.h * self.w * self.channels()
    fn is_hwc(self) -> Bool: return self.layout == Layout.HWC()

# -------------------------
# Basic pixel helpers
# -------------------------
@staticmethod
fn _offset(w: Int, c: Int, x: Int, y: Int, ch: Int) -> Int:
    return ((y * w) + x) * c + ch

@staticmethod
fn get_px(img: Image, x: Int, y: Int, ch: Int) -> UInt8:
    return img.data[_offset(img.w, img.channels(), x, y, ch)]

@staticmethod
fn set_px(mut img: Image, x: Int, y: Int, ch: Int, v: UInt8) -> Image:
    img.data[_offset(img.w, img.channels(), x, y, ch)] = v
    return img

# -------------------------
# Construction helpers (simulating decoders)
# -------------------------
@staticmethod
fn read_image_from_rgb_u8(h: Int, w: Int, buf: List[UInt8]) -> Image:
    # Treat buf as packed RGB (HWC) data
    return Image(h, w, ChannelOrder.RGB(), AlphaMode.NONE(), buf)

@staticmethod
fn read_image_from_bgr_u8(h: Int, w: Int, buf: List[UInt8]) -> Image:
    return Image(h, w, ChannelOrder.BGR(), AlphaMode.NONE(), buf)

@staticmethod
fn read_image_from_rgba_u8(h: Int, w: Int, buf: List[UInt8]) -> Image:
    return Image(h, w, ChannelOrder.RGBA(), AlphaMode.STRAIGHT(), buf)

@staticmethod
fn read_image_from_bgra_u8(h: Int, w: Int, buf: List[UInt8]) -> Image:
    return Image(h, w, ChannelOrder.BGRA(), AlphaMode.STRAIGHT(), buf)

@staticmethod
fn read_image_from_gray_u8(h: Int, w: Int, buf: List[UInt8]) -> Image:
    return Image(h, w, ChannelOrder.GRAY(), AlphaMode.NONE(), buf)

# Placeholder for path-based read. Returns a zero image with failure flag.
@staticmethod
fn read_image(path: String) -> (Bool, Image):
    var img = Image.zeros(0, 0, ChannelOrder.RGB(), AlphaMode.NONE())
    return (False, img)

# -------------------------
# Export helpers (encode to raw channel-major buffers)
# -------------------------
@staticmethod
fn encode_to_rgb_u8(img: Image) -> List[UInt8]:
    # If already RGB with 3 channels, return a copy; if BGR swap; if GRAY repeat channel; if RGBA drop alpha.
    var c = img.channels()
    if img.order == ChannelOrder.RGB() and c == 3:
        var out: List[UInt8] = List[UInt8]()
        var i = 0; var n = img.h * img.w * 3
        while i < n:
            out.append(img.data[i]); i += 1
        return out

    # allocate
    var out2: List[UInt8] = List[UInt8]()
    var total = img.h * img.w * 3
    var j = 0
    while j < total: out2.append(0); j += 1

    var y = 0
    while y < img.h:
        var x = 0
        while x < img.w:
            if img.order == ChannelOrder.BGR() and c == 3:
                var b = get_px(img, x, y, 0)
                var g = get_px(img, x, y, 1)
                var r = get_px(img, x, y, 2)
                var base = _offset(img.w, 3, x, y, 0)
                out2[base+0] = r; out2[base+1] = g; out2[base+2] = b
            elif img.order == ChannelOrder.RGBA() and c == 4:
                var base3 = _offset(img.w, 4, x, y, 0)
                var base = _offset(img.w, 3, x, y, 0)
                out2[base+0] = img.data[base3+0]
                out2[base+1] = img.data[base3+1]
                out2[base+2] = img.data[base3+2]
            elif img.order == ChannelOrder.BGRA() and c == 4:
                var base4 = _offset(img.w, 4, x, y, 0)
                var baseB = _offset(img.w, 3, x, y, 0)
                out2[baseB+0] = img.data[base4+2]
                out2[baseB+1] = img.data[base4+1]
                out2[baseB+2] = img.data[base4+0]
            elif img.order == ChannelOrder.GRAY() and c == 1:
                var g1 = get_px(img, x, y, 0)
                var baseG = _offset(img.w, 3, x, y, 0)
                out2[baseG+0] = g1; out2[baseG+1] = g1; out2[baseG+2] = g1
            else:
                # Fallback: just copy first 3 channels if present, else zeros
                var k = 0
                while k < 3:
                    var v: UInt8 = 0
                    if k < c: v = get_px(img, x, y, k)
                    out2[_offset(img.w, 3, x, y, k)] = v
                    k += 1
            x += 1
        y += 1
    return out2

# -------------------------
# String summary
# -------------------------
@staticmethod
fn summary(img: Image) -> String:
    var s = String("Image(") + String(img.h) + String("x") + String(img.w) + String("x") + String(img.channels()) + String(", ")
    s = s + img.layout.to_string() + String(", ") + img.order.to_string() + String(", alpha=") + img.alpha.to_string() + String(")")
    return s

# -------------------------
# Minimal smoke test
# -------------------------
@staticmethod
fn __self_test__() -> Bool:
    # Build 2x2 RGB buffer
    var data: List[UInt8] = List[UInt8]()
    data.append(255); data.append(0);   data.append(0)
    data.append(0);   data.append(255); data.append(0)
    data.append(0);   data.append(0);   data.append(255)
    data.append(255); data.append(255); data.append(255)

    var img = read_image_from_rgb_u8(2, 2, data)
    if img.size() != 12: return False
    var rgb = encode_to_rgb_u8(img)
    if len(rgb) != 12: return False

    var bgr = read_image_from_bgr_u8(2, 2, data)
    var rgb2 = encode_to_rgb_u8(bgr)
    if len(rgb2) != 12: return False

    var g = read_image_from_gray_u8(2, 2, List[UInt8]())
    if g.channels() != 1: return False

    return True
