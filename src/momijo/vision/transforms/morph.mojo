# MIT License
# SPDX-License-Identifier: MIT
# Project: momijo.vision | File: src/momijo/vision/transforms/morph.mojo

from momijo.vision.image import Image

# -----------------------------------------------------------------------------
# Configuration and pixel access hooks
# -----------------------------------------------------------------------------

fn _PIXEL_IO_AVAILABLE() -> Bool:
    # We have get_u8 / set_u8 on Image, so enable pixel IO path.
    return True

# Read a single channel at (x,y). Channel index in [0..c-1].
fn _peek(img: Image, x: Int, y: Int, ch: Int) -> UInt8:
    return img.get_u8(y, x, ch)

# Write B,G,R (or generally ch0,ch1,ch2) at (x,y). If image has <3 channels,
# we write into available channels only.
fn _poke(mut img: Image, x: Int, y: Int, b: UInt8, g: UInt8, r: UInt8):
    var c = img.channels()
    if c >= 1:
        img.set_u8(y, x, 0, b)
    if c >= 2:
        img.set_u8(y, x, 1, g)
    if c >= 3:
        img.set_u8(y, x, 2, r)

# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------

fn _in_bounds(img: Image, x: Int, y: Int) -> Bool:
    return (x >= 0 and x < img.width() and y >= 0 and y < img.height())

fn _clamp_int(v: Int, lo: Int, hi: Int) -> Int:
    var x = v
    if x < lo:
        x = lo
    if x > hi:
        x = hi
    return x

# Normalize ksize to a positive odd integer (1,3,5,...).
fn _normalize_ksize(ksize: Int) -> Int:
    var k = ksize
    if k < 1:
        k = 1
    if (k & 1) == 0:
        k = k + 1
    return k

# Pick safe channel indices for G and R based on image channel count.
fn _safe_ch1(c: Int) -> Int:
    if c > 1:
        return 1
    return 0

fn _safe_ch2(c: Int) -> Int:
    if c > 2:
        return 2
    if c > 1:
        return 1
    return 0

# Naive (O(k^2)) neighborhood min for each channel separately.
fn _erode_naive(mut dst: Image, src: Image, k: Int) -> Image:
    var rad = k // 2
    var h = src.height()
    var w = src.width()
    var c = src.channels()
    var ch1 = _safe_ch1(c)
    var ch2 = _safe_ch2(c)

    var y = 0
    while y < h:
        var x = 0
        while x < w:
            var mb: UInt8 = UInt8(255)
            var mg: UInt8 = UInt8(255)
            var mr: UInt8 = UInt8(255)

            var yy = y - rad
            while yy <= y + rad:
                var xx = x - rad
                while xx <= x + rad:
                    if _in_bounds(src, xx, yy):
                        var b = _peek(src, xx, yy, 0)
                        var g = _peek(src, xx, yy, ch1)
                        var r = _peek(src, xx, yy, ch2)
                        if b < mb:
                            mb = b
                        if g < mg:
                            mg = g
                        if r < mr:
                            mr = r
                    xx = xx + 1
                yy = yy + 1

            _poke(dst, x, y, mb, mg, mr)
            x = x + 1
        y = y + 1
    return dst

# Naive (O(k^2)) neighborhood max for each channel separately.
fn _dilate_naive(mut dst: Image, src: Image, k: Int) -> Image:
    var rad = k // 2
    var h = src.height()
    var w = src.width()
    var c = src.channels()
    var ch1 = _safe_ch1(c)
    var ch2 = _safe_ch2(c)

    var y = 0
    while y < h:
        var x = 0
        while x < w:
            var Mb: UInt8 = UInt8(0)
            var Mg: UInt8 = UInt8(0)
            var Mr: UInt8 = UInt8(0)

            var yy = y - rad
            while yy <= y + rad:
                var xx = x - rad
                while xx <= x + rad:
                    if _in_bounds(src, xx, yy):
                        var b = _peek(src, xx, yy, 0)
                        var g = _peek(src, xx, yy, ch1)
                        var r = _peek(src, xx, yy, ch2)
                        if b > Mb:
                            Mb = b
                        if g > Mg:
                            Mg = g
                        if r > Mr:
                            Mr = r
                    xx = xx + 1
                yy = yy + 1

            _poke(dst, x, y, Mb, Mg, Mr)
            x = x + 1
        y = y + 1
    return dst

# Allocate a new Image with the same shape as src (zero-initialized).
fn _like(src: Image) -> Image:
    var h = src.height()
    var w = src.width()
    var c = src.channels()
    return Image.new_hwc_u8(h, w, c, UInt8(0))

# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

fn erode(img: Image, ksize: Int) -> Image:
    var k = _normalize_ksize(ksize)
    if not _PIXEL_IO_AVAILABLE():
        return img
    var dst = _like(img)
    return _erode_naive(dst, img, k)

# Morphological erosion (stub/naive for kernel Image; iterations supported).
fn erode(img: Image, kernel: Image, iterations: Int = 1) -> Image:
    var out = img
    var i = 0
    while i < iterations:
        # TODO: implement real erosion using the kernel.
        out = erode(out, kernel.width())  # use kernel width as ksize fallback
        i = i + 1
    return out

fn dilate(img: Image, ksize: Int) -> Image:
    var k = _normalize_ksize(ksize)
    if not _PIXEL_IO_AVAILABLE():
        return img
    var dst = _like(img)
    return _dilate_naive(dst, img, k)

# Morphological dilation (stub/naive for kernel Image; iterations supported).
fn dilate(img: Image, kernel: Image, iterations: Int = 1) -> Image:
    var out = img
    var i = 0
    while i < iterations:
        # TODO: implement real dilation using the kernel.
        out = dilate(out, kernel.width())  # use kernel width as ksize fallback
        i = i + 1
    return out

# op codes:
#   0: OPEN     (erode then dilate)
#   1: CLOSE    (dilate then erode)
#   2: GRADIENT (approximate: return dilated image)
fn morphology(img: Image, op: Int, ksize: Int) -> Image:
    var k = _normalize_ksize(ksize)
    if not _PIXEL_IO_AVAILABLE():
        return img

    if op == 0:
        # OPEN
        var tmp = _like(img)
        tmp = _erode_naive(tmp, img, k)
        var out = _like(img)
        out = _dilate_naive(out, tmp, k)
        return out
    elif op == 1:
        # CLOSE
        var tmp2 = _like(img)
        tmp2 = _dilate_naive(tmp2, img, k)
        var out2 = _like(img)
        out2 = _erode_naive(out2, tmp2, k)
        return out2
    elif op == 2:
        # GRADIENT (approximation without subtraction)
        var out3 = _like(img)
        out3 = _dilate_naive(out3, img, k)
        return out3
    else:
        # Unknown op: return input unchanged
        return img

fn morphology(img: Image, op: Int, kernel: Image, iterations: Int = 1) -> Image:
    if op == MORPH_ERODE():
        return erode(img, kernel, iterations)
    elif op == MORPH_DILATE():
        return dilate(img, kernel, iterations)
    elif op == MORPH_OPEN():
        # opening = erode then dilate
        var tmp = erode(img, kernel, iterations)
        return dilate(tmp, kernel, iterations)
    elif op == MORPH_CLOSE():
        # closing = dilate then erode
        var tmp2 = dilate(img, kernel, iterations)
        return erode(tmp2, kernel, iterations)
    elif op == MORPH_GRADIENT():
        # gradient = dilate - erode (stub with dilate)
        return dilate(img, kernel, iterations)
    else:
        # unknown op → return input
        return img

# --- Operation codes (no enum in project standard) ---
fn MORPH_ERODE() -> Int:    return 0
fn MORPH_DILATE() -> Int:   return 1
fn MORPH_OPEN() -> Int:     return 2
fn MORPH_CLOSE() -> Int:    return 3
fn MORPH_GRADIENT() -> Int: return 4
