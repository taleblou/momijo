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
# Project: momijo.vision.transforms
# File: src/momijo/vision/transforms/random_resized_crop.mojo

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
        var expected = h*w*_num_ch(order)
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
    while i < n: out.append(0); i += 1
    return out

# -------------------------
# Deterministic flip
# -------------------------
@staticmethod
fn flip_h_u8_hwc(h: Int, w: Int, c: Int, src: List[UInt8]) -> List[UInt8]:
    if h <= 0 or w <= 0 or c <= 0:
        return List[UInt8]()
    var out = _alloc_u8(h*w*c)
    var y = 0
    while y < h:
        var x = 0
        while x < w:
            var xr = (w - 1) - x
            var ch = 0
            while ch < c:
                out[_offset(w, c, xr, y, ch)] = src[_offset(w, c, x, y, ch)]
                ch += 1
            x += 1
        y += 1
    return out

@staticmethod
fn maybe_flip_h_u8_hwc(h: Int, w: Int, c: Int, src: List[UInt8], do_flip: Bool) -> List[UInt8]:
    if do_flip:
        return flip_h_u8_hwc(h, w, c, src)
    # copy
    var out = _alloc_u8(h*w*c)
    var i = 0
    while i < h*w*c:
        out[i] = src[i]
        i += 1
    return out

# -------------------------
# RNG (LCG) + Bernoulli
# -------------------------
struct LcgRng(Copyable, Movable):
    var state: Int
fn __init__(out self, seed: Int) -> None:
        if seed == 0: seed = 1
        self.state = seed & UInt8(0x7FFFFFFF)
fn next_u32(mut self) -> (LcgRng, Int):
        var a = 1664525
        var c = 1013904223
        self.state = (a * self.state + c) & 0x7FFFFFFF
        return (self, self.state)
fn bernoulli_q8(mut self, p_q8: Int) -> (LcgRng, Bool):
        var s = 0; (self, s) = self.next_u32()
        var r = s & UInt8(255)               # 0..255
        # Accept if r < p_q8 (p_q8==255 -> ~100%)
        var ok = False
        if r < p_q8: ok = True
        return (self, ok)

# -------------------------
# Random flip (single image)
# -------------------------
@staticmethod
fn random_flip_h_u8_hwc(h: Int, w: Int, c: Int, src: List[UInt8], seed: Int, p_q8: Int) -> (List[UInt8], Bool):
    var rng = LcgRng(seed)
    var dof = False
    (rng, dof) = rng.bernoulli_q8(p_q8)
    var out = maybe_flip_h_u8_hwc(h, w, c, src, dof)
    return (out, dof)

# -------------------------
# Image wrapper
# -------------------------
@staticmethod
fn apply_random_hflip(img: Image, seed: Int, p_q8: Int) -> (Image, Bool):
    var c = _num_ch(img.order)
    var buf: List[UInt8] = List[UInt8](); var flipped = False
    (buf, flipped) = random_flip_h_u8_hwc(img.h, img.w, c, img.data, seed, p_q8)
    return (Image(img.h, img.w, img.order, buf), flipped)

# -------------------------
# Batch utilities
# -------------------------
@staticmethod
fn sample_masks(n: Int, seed: Int, p_q8: Int) -> List[Bool]:
    var rng = LcgRng(seed)
    var out: List[Bool] = List[Bool]()
    var i = 0
    while i < n:
        var b = False
        (rng, b) = rng.bernoulli_q8(p_q8)
        out.append(b)
        i += 1
    return out

@staticmethod
fn apply_random_hflip_batch(imgs: List[Image], seed: Int, p_q8: Int) -> (List[Image], List[Bool]):
    var masks = sample_masks(len(imgs), seed, p_q8)
    var out_imgs: List[Image] = List[Image]()
    var i = 0
    while i < len(imgs):
        var img = imgs[i]
        var c = _num_ch(img.order)
        var buf = maybe_flip_h_u8_hwc(img.h, img.w, c, img.data, masks[i])
        out_imgs.append(Image(img.h, img.w, img.order, buf))
        i += 1
    return (out_imgs, masks)

# -------------------------
# Minimal smoke test
# -------------------------
@staticmethod
fn __self_test__() -> Bool:
    # Build a 2x3 RGB image with distinct pattern:
    # row0: (1,2,3) (4,5,6) (7,8,9)
    # row1: (10,11,12) (13,14,15) (16,17,18)
    var h = 2; var w = 3; var c = 3
    var src: List[UInt8] = List[UInt8]()
    # row0
    src.append(1); src.append(2); src.append(3)
    src.append(4); src.append(5); src.append(6)
    src.append(7); src.append(8); src.append(9)
    # row1
    src.append(10); src.append(11); src.append(12)
    src.append(13); src.append(14); src.append(15)
    src.append(16); src.append(17); src.append(18)

    # 1) Deterministic flip
    var f = flip_h_u8_hwc(h, w, c, src)
    # After flip, row0 pixels order should reverse by 3-tuples
    # row0 becomes: (7,8,9) (4,5,6) (1,2,3)
    if not (f[0] == UInt8(7) and f[1] == UInt8(8) and f[2] == UInt8(9)): return False
    if not (f[3] == UInt8(4) and f[4] == UInt8(5) and f[5] == UInt8(6)): return False
    if not (f[6] == UInt8(1) and f[7] == UInt8(2) and f[8] == UInt8(3)): return False

    # 2) Random flip with p=255 -> always flip
    var buf: List[UInt8] = List[UInt8](); var flipped = False
    (buf, flipped) = random_flip_h_u8_hwc(h, w, c, src, 42, 255)
    if not flipped: return False
    if not (buf[0] == UInt8(7) and buf[1] == UInt8(8) and buf[2] == UInt8(9)): return False

    # 3) Random flip with p=0 -> never flip (copy)
    var buf2: List[UInt8] = List[UInt8](); var flipped2 = True
    (buf2, flipped2) = random_flip_h_u8_hwc(h, w, c, src, 42, 0)
    if flipped2: return False
    if not (buf2[0] == UInt8(1) and buf2[1] == UInt8(2) and buf2[2] == UInt8(3)): return False

    # 4) Image wrapper + reproducibility
    var img = Image(h, w, ChannelOrder.RGB(), src)
    var out_img: Image; var fl = False
    (out_img, fl) = apply_random_hflip(img, 12345, 200)
    var out_img2: Image; var fl2 = False
    (out_img2, fl2) = apply_random_hflip(img, 12345, 200)
    if not (fl == fl2): return False
    var i = 0
    while i < len(out_img.data):
        if out_img.data[i] != out_img2.data[i]: return False
        i += 1

    # 5) Batch masks length and correspondence
    var imgs: List[Image] = List[Image](); imgs.append(img); imgs.append(img)
    var outs: List[Image] = List[Image](); var masks: List[Bool] = List[Bool]()
    (outs, masks) = apply_random_hflip_batch(imgs, 7, 128)
    if not (len(outs) == 2 and len(masks) == 2): return False

    return True