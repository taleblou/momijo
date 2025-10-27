# MIT License
# Project: momijo.vision
# File: momijo/vision/bitwise.mojo
# SPDX-License-Identifier: MIT

from momijo.vision.image import Image

fn _min(a: Int, b: Int) -> Int:
    if a < b:
        return a
    return b

fn _and_u8(a: UInt8, b: UInt8) -> UInt8:
    var r = (Int(a) & Int(b)) & 255
    return UInt8(r)

fn _or_u8(a: UInt8, b: UInt8) -> UInt8:
    var r = (Int(a) | Int(b)) & 255
    return UInt8(r)

fn _not_u8(a: UInt8) -> UInt8:
    # bitwise NOT in 8-bit domain
    var r = 255 - Int(a)
    return UInt8(r)

# dst = a & b  (UInt8 per-channel). Sizes/channels = overlap of inputs.
fn bitwise_and(a: Image, b: Image) -> Image:
    var A = a.ensure_packed_hwc_u8(True)
    var B = b.ensure_packed_hwc_u8(True)

    var h = _min(A.height(), B.height())
    var w = _min(A.width(),  B.width())
    var c = _min(A.channels(), B.channels())

    var out = Image.new_hwc_u8(h, w, c, UInt8(0))

    var y = 0
    while y < h:
        var x = 0
        while x < w:
            var ch = 0
            while ch < c:
                var va = A.get_u8(y, x, ch)
                var vb = B.get_u8(y, x, ch)
                out.set_u8(y, x, ch, _and_u8(va, vb))
                ch += 1
            x += 1
        y += 1
    return out.copy()

# dst = a & b, with mask (apply where mask>0, else 0)
fn bitwise_and(a: Image, b: Image, mask: Image) -> Image:
    var A = a.ensure_packed_hwc_u8(True)
    var B = b.ensure_packed_hwc_u8(True)
    var M = mask.ensure_packed_hwc_u8(True)

    var h = _min(_min(A.height(), B.height()), M.height())
    var w = _min(_min(A.width(),  B.width()),  M.width())
    var c = _min(A.channels(), B.channels())

    var out = Image.new_hwc_u8(h, w, c, UInt8(0))

    var y = 0
    while y < h:
        var x = 0
        while x < w:
            var m = M.get_u8(y, x, 0)
            var ch = 0
            if m > UInt8(0):
                while ch < c:
                    var va = A.get_u8(y, x, ch)
                    var vb = B.get_u8(y, x, ch)
                    out.set_u8(y, x, ch, _and_u8(va, vb))
                    ch += 1
            else:
                while ch < c:
                    out.set_u8(y, x, ch, UInt8(0))
                    ch += 1
            x += 1
        y += 1
    return out.copy()

# dst = a | b
fn bitwise_or(a: Image, b: Image) -> Image:
    var A = a.ensure_packed_hwc_u8(True)
    var B = b.ensure_packed_hwc_u8(True)

    var h = _min(A.height(), B.height())
    var w = _min(A.width(),  B.width())
    var c = _min(A.channels(), B.channels())

    var out = Image.new_hwc_u8(h, w, c, UInt8(0))

    var y = 0
    while y < h:
        var x = 0
        while x < w:
            var ch = 0
            while ch < c:
                var va = A.get_u8(y, x, ch)
                var vb = B.get_u8(y, x, ch)
                out.set_u8(y, x, ch, _or_u8(va, vb))
                ch += 1
            x += 1
        y += 1
    return out.copy()

# dst = a | b, with mask (apply where mask>0, else 0)
fn bitwise_or(a: Image, b: Image, mask: Image) -> Image:
    var A = a.ensure_packed_hwc_u8(True)
    var B = b.ensure_packed_hwc_u8(True)
    var M = mask.ensure_packed_hwc_u8(True)

    var h = _min(_min(A.height(), B.height()), M.height())
    var w = _min(_min(A.width(),  B.width()),  M.width())
    var c = _min(A.channels(), B.channels())

    var out = Image.new_hwc_u8(h, w, c, UInt8(0))

    var y = 0
    while y < h:
        var x = 0
        while x < w:
            var m = M.get_u8(y, x, 0)
            var ch = 0
            if m > UInt8(0):
                while ch < c:
                    var va = A.get_u8(y, x, ch)
                    var vb = B.get_u8(y, x, ch)
                    out.set_u8(y, x, ch, _or_u8(va, vb))
                    ch += 1
            else:
                while ch < c:
                    out.set_u8(y, x, ch, UInt8(0))
                    ch += 1
            x += 1
        y += 1
    return out.copy()

# dst = ~a
fn bitwise_not(a: Image) -> Image:
    var A = a.ensure_packed_hwc_u8(True)

    var h = A.height()
    var w = A.width()
    var c = A.channels()

    var out = Image.new_hwc_u8(h, w, c, UInt8(0))

    var y = 0
    while y < h:
        var x = 0
        while x < w:
            var ch = 0
            while ch < c:
                var va = A.get_u8(y, x, ch)
                out.set_u8(y, x, ch, _not_u8(va))
                ch += 1
            x += 1
        y += 1
    return out.copy()

# dst = ~a where mask>0, else 0
fn bitwise_not(a: Image, mask: Image) -> Image:
    var A = a.ensure_packed_hwc_u8(True)
    var M = mask.ensure_packed_hwc_u8(True)

    var h = _min(A.height(), M.height())
    var w = _min(A.width(),  M.width())
    var c = A.channels()

    var out = Image.new_hwc_u8(h, w, c, UInt8(0))

    var y = 0
    while y < h:
        var x = 0
        while x < w:
            var m = M.get_u8(y, x, 0)
            var ch = 0
            if m > UInt8(0):
                while ch < c:
                    var va = A.get_u8(y, x, ch)
                    out.set_u8(y, x, ch, _not_u8(va))
                    ch += 1
            else:
                while ch < c:
                    out.set_u8(y, x, ch, UInt8(0))
                    ch += 1
            x += 1
        y += 1
    return out.copy()
