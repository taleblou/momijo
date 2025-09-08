# Project:      Momijo
# Module:       src.momijo.vision.image
# File:         image.mojo
# Path:         src/momijo/vision/image.mojo
#
# Description:  src.momijo.vision.image — focused Momijo functionality with a stable public API.
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
#   - Structs: ChannelOrder, Layout, AlphaMode, Image
#   - Key functions: __init__, GRAY, RGB, BGR, RGBA, BGRA, ARGB, __eq__ ...
#   - Static methods present.


struct ChannelOrder(Copyable, Movable):
    var id: Int
fn __init__(out self, id: Int) -> None:
        self.id = id
    @staticmethod
fn GRAY() -> ChannelOrder: return ChannelOrder(1)
    @staticmethod
fn RGB() -> ChannelOrder:  return ChannelOrder(2)
    @staticmethod
fn BGR() -> ChannelOrder:  return ChannelOrder(3)
    @staticmethod
fn RGBA() -> ChannelOrder: return ChannelOrder(4)
    @staticmethod
fn BGRA() -> ChannelOrder: return ChannelOrder(5)
    @staticmethod
fn ARGB() -> ChannelOrder: return ChannelOrder(6)
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
fn has_alpha(order: ChannelOrder) -> Bool:
    return order == ChannelOrder.RGBA() or order == ChannelOrder.BGRA() or order == ChannelOrder.ARGB()

@staticmethod
fn num_channels(order: ChannelOrder) -> Int:
    if order == ChannelOrder.GRAY(): return 1
    if order == ChannelOrder.RGB() or order == ChannelOrder.BGR(): return 3
    return 4

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
        return String("UNKNOWN")

struct AlphaMode(Copyable, Movable):
    var id: Int
fn __init__(out self, id: Int) -> None:
        self.id = id
    @staticmethod
fn NONE() -> AlphaMode:          return AlphaMode(0)
    @staticmethod
fn STRAIGHT() -> AlphaMode:      return AlphaMode(1)
    @staticmethod
fn PREMULTIPLIED() -> AlphaMode: return AlphaMode(2)
fn __eq__(self, other: AlphaMode) -> Bool: return self.id == other.id
fn to_string(self) -> String:
        if self.id == 0: return String("NONE")
        if self.id == 1: return String("STRAIGHT")
        if self.id == 2: return String("PREMULTIPLIED")
        return String("UNKNOWN")

# -------------------------
# Image container: UInt8 HWC with metadata
# -------------------------
struct Image(Copyable, Movable):
    var h: Int
    var w: Int
    var order: ChannelOrder
    var alpha: AlphaMode
    var layout: Layout   # always HWC for storage
    var data: List[UInt8]
fn __init__(out self, h: Int, w: Int, order: ChannelOrder, alpha: AlphaMode, data: List[UInt8]) -> None:
        self.h = h
        self.w = w
        self.order = order
        self.alpha = alpha
        self.layout = Layout.HWC()
        var c = num_channels(order)
        # Validate data length; if mismatch, zero-fill conservatively
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

    # Convenience constructor: zeros
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

    # read-only metadata
fn height(self) -> Int: return self.h
fn width(self) -> Int:  return self.w
fn channels(self) -> Int: return num_channels(self.order)
fn size(self) -> Int: return self.h * self.w * self.channels()
fn is_hwc(self) -> Bool: return self.layout == Layout.HWC()

# -------------------------
# Pixel addressing and get/set
# -------------------------
@staticmethod
fn _offset(h: Int, w: Int, c: Int, x: Int, y: Int, ch: Int) -> Int:
    # Row-major HWC
    return ((y * w) + x) * c + ch

@staticmethod
fn get_px(img: Image, x: Int, y: Int, ch: Int) -> UInt8:
    var idx = _offset(img.h, img.w, img.channels(), x, y, ch)
    return img.data[idx]

@staticmethod
fn set_px(mut img: Image, x: Int, y: Int, ch: Int, v: UInt8) -> Image:
    var idx = _offset(img.h, img.w, img.channels(), x, y, ch)
    img.data[idx] = v
    return img

# -------------------------
# Common transforms
# -------------------------
@staticmethod
fn rgb_to_gray(img: Image) -> Image:
    # For RGB/BGR 3-channel; pass-through otherwise
    var c = img.channels()
    if c != 3:
        return img
    var out = Image.zeros(img.h, img.w, ChannelOrder.GRAY(), AlphaMode.NONE())
    var y = 0
    while y < img.h:
        var x = 0
        while x < img.w:
            var r: UInt8 = 0
            var g: UInt8 = 0
            var b: UInt8 = 0
            if img.order == ChannelOrder.RGB():
                r = get_px(img, x, y, 0)
                g = get_px(img, x, y, 1)
                b = get_px(img, x, y, 2)
            else:
                # BGR
                b = get_px(img, x, y, 0)
                g = get_px(img, x, y, 1)
                r = get_px(img, x, y, 2)
            # integer-approx luma: (77, 150, 29)/256
            var gray_u16 = UInt16(77) * UInt16(r) + UInt16(150) * UInt16(g) + UInt16(29) * UInt16(b)
            var gray = UInt8((gray_u16 >> UInt8(8)) & UInt16(0xFF))
            out = set_px(out, x, y, 0, gray)
            x += 1
        y += 1
    return out

@staticmethod
fn drop_alpha(img: Image) -> Image:
    # If alpha present, return image without alpha channel (RGB/BGR); else pass-through.
    if not has_alpha(img.order):
        return img
    var base_order = ChannelOrder.RGB()
    if img.order == ChannelOrder.BGRA() or img.order == ChannelOrder.ARGB():
        base_order = ChannelOrder.BGR()
    # Map channel indices based on order:
    # RGBA -> RGB (0,1,2)
    # BGRA -> BGR (0,1,2)
    # ARGB -> RGB (1,2,3)
    var out = Image.zeros(img.h, img.w, base_order, AlphaMode.NONE())
    var y = 0
    while y < img.h:
        var x = 0
        while x < img.w:
            if img.order == ChannelOrder.RGBA():
                out = set_px(out, x, y, 0, get_px(img, x, y, 0))
                out = set_px(out, x, y, 1, get_px(img, x, y, 1))
                out = set_px(out, x, y, 2, get_px(img, x, y, 2))
            elif img.order == ChannelOrder.BGRA():
                out = set_px(out, x, y, 0, get_px(img, x, y, 0))
                out = set_px(out, x, y, 1, get_px(img, x, y, 1))
                out = set_px(out, x, y, 2, get_px(img, x, y, 2))
            else:
                # ARGB
                out = set_px(out, x, y, 0, get_px(img, x, y, 1))
                out = set_px(out, x, y, 1, get_px(img, x, y, 2))
                out = set_px(out, x, y, 2, get_px(img, x, y, 3))
            x += 1
        y += 1
    return out

@staticmethod
fn bgr_to_rgb(img: Image) -> Image:
    if img.channels() != 3 or not (img.order == ChannelOrder.BGR() or img.order == ChannelOrder.RGB()):
        return img
    if img.order == ChannelOrder.RGB():
        return img
    var out = Image.zeros(img.h, img.w, ChannelOrder.RGB(), img.alpha)
    var y = 0
    while y < img.h:
        var x = 0
        while x < img.w:
            out = set_px(out, x, y, 0, get_px(img, x, y, 2)) # R <- B
            out = set_px(out, x, y, 1, get_px(img, x, y, 1)) # G <- G
            out = set_px(out, x, y, 2, get_px(img, x, y, 0)) # B <- R
            x += 1
        y += 1
    return out

@staticmethod
fn resize_nearest(img: Image, oh: Int, ow: Int) -> Image:
    var out = Image.zeros(oh, ow, img.order, img.alpha)
    var c = img.channels()
    var y = 0
    while y < oh:
        var sy = (y * img.h) // oh
        var x = 0
        while x < ow:
            var sx = (x * img.w) // ow
            var ch = 0
            while ch < c:
                out = set_px(out, x, y, ch, get_px(img, sx, sy, ch))
                ch += 1
            x += 1
        y += 1
    return out

# -------------------------
# String summary (for logging)
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
    # Build 2x2 RGB pattern
    var data: List[UInt8] = List[UInt8]()
    # (255,0,0) (0,255,0)
    data.append(255); data.append(0);   data.append(0)
    data.append(0);   data.append(255); data.append(0)
    # (0,0,255) (255,255,255)
    data.append(0);   data.append(0);   data.append(255)
    data.append(255); data.append(255); data.append(255)

    var img = Image(2, 2, ChannelOrder.RGB(), AlphaMode.NONE(), data)
    if img.size() != 12: return False

    var gray = rgb_to_gray(img)
    if gray.channels() != 1: return False
    var rsz = resize_nearest(gray, 4, 4)
    if rsz.h != 4 or rsz.w != 4: return False

    var bgr = bgr_to_rgb(Image(2, 2, ChannelOrder.BGR(), AlphaMode.NONE(), data))
    if bgr.order != ChannelOrder.RGB(): return False

    var noalpha = drop_alpha(Image.zeros(2, 2, ChannelOrder.RGBA(), AlphaMode.STRAIGHT()))
    if noalpha.channels() != 3: return False

    return True