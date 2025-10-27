# MIT License
# SPDX-License-Identifier: MIT
# Project: momijo.vision | File: src/momijo/vision/transforms/color.mojo

from momijo.vision.image import Image
from momijo.vision.image import make_zero_u8_hwc
from momijo.vision.dtypes import ColorSpace
from momijo.vision.transforms.draw import line  # for plot_hist_u8

# ---------------------------
# BGR <-> RGB channel swaps
# ---------------------------

# Swap channels: BGR -> RGB (creates a new image, does not modify in-place)
fn bgr_to_rgb(img: Image) -> Image:
    var x = img.ensure_packed_hwc_u8(True)
    var h = x.height()
    var w = x.width()
    var c = x.channels()
    if h <= 0 or w <= 0 or c < 3:
        return x.copy()

    var out = x.clone()
    var pin  = x.tensor().data()
    var pout = out.tensor().data()
    var s0 = w * c
    var s1 = c

    var y = 0
    while y < h:
        var xj = 0
        while xj < w:
            var base = y * s0 + xj * s1
            # BGR -> RGB: swap channel 0 and 2
            pout[base + 0] = pin[base + 2]
            pout[base + 1] = pin[base + 1]
            pout[base + 2] = pin[base + 0]
            xj += 1
        y += 1
    return out.copy()

# Symmetric alias
fn rgb_to_bgr(img: Image) -> Image:
    return bgr_to_rgb(img)

# ---------------------------
# BGR -> Grayscale (luma)
# ---------------------------

# Converts BGR image to 1-channel grayscale using ITU-R BT.601 luma:
# Y = 0.114*B + 0.587*G + 0.299*R
fn bgr_to_gray(img: Image) -> Image:
    var x = img.ensure_packed_hwc_u8(True)
    var h = x.height()
    var w = x.width()
    var c = x.channels()

    if h <= 0 or w <= 0:
        return x.copy()
    if c == 1:
        # already single-channel
        return x.copy()

    if c < 3:
        # degenerate: copy first channel into gray
        var out = make_zero_u8_hwc(h, w, 1, ColorSpace.Gray())
        var pin  = x.tensor().data()
        var pout = out.tensor().data()
        var s0_in = w * c
        var s1_in = c

        var y = 0
        while y < h:
            var xj = 0
            while xj < w:
                var base_in = y * s0_in + xj * s1_in
                pout[y * w + xj] = pin[base_in + 0]
                xj += 1
            y += 1
        return out.copy()

    # Normal 3+ channel case (use B,G,R)
    var out = make_zero_u8_hwc(h, w, 1, ColorSpace.Gray())
    var pin  = x.tensor().data()
    var pout = out.tensor().data()
    var s0 = w * c
    var s1 = c

    var y = 0
    while y < h:
        var xj = 0
        while xj < w:
            var base = y * s0 + xj * s1
            var b = pin[base + 0]
            var g = pin[base + 1]
            var r = pin[base + 2]
            # compute luma (use Float64 for accuracy)
            var lum = 0.114 * Float64(b) + 0.587 * Float64(g) + 0.299 * Float64(r)
            if lum < 0.0: lum = 0.0
            if lum > 255.0: lum = 255.0
            pout[y * w + xj] = UInt8(Int(lum))
            xj += 1
        y += 1
    return out.copy()

# ---------------------------
# Split / Merge channels
# ---------------------------

# Split into three single-channel images (B, G, R). If channels < 3,
# missing outputs are zero-initialized 1-channel placeholders.
fn split3(img: Image) -> (Image, Image, Image):
    var x = img.ensure_packed_hwc_u8(True)
    var h = x.height()
    var w = x.width()
    var c = x.channels()

    var b_img = make_zero_u8_hwc(h, w, 1, ColorSpace.Gray())
    var g_img = make_zero_u8_hwc(h, w, 1, ColorSpace.Gray())
    var r_img = make_zero_u8_hwc(h, w, 1, ColorSpace.Gray())

    if h <= 0 or w <= 0 or c <= 0:
        return (b_img.copy(), g_img.copy(), r_img.copy())

    var pin = x.tensor().data()
    var pb = b_img.tensor().data()
    var pg = g_img.tensor().data()
    var pr = r_img.tensor().data()

    var s0 = w * c
    var s1 = c

    var y = 0
    while y < h:
        var xj = 0
        while xj < w:
            var base = y * s0 + xj * s1
            # channel 0 always exists
            pb[y * w + xj] = pin[base + 0]
            if c > 1:
                pg[y * w + xj] = pin[base + 1]
            if c > 2:
                pr[y * w + xj] = pin[base + 2]
            xj += 1
        y += 1

    return (b_img.copy(), g_img.copy(), r_img.copy())

# Merge three single-channel images into a 3-channel BGR image.
# Inputs are provided as (r, g, b) for convenience; output is BGR (0:B,1:G,2:R).
fn merge3(r: Image, g: Image, b: Image) -> Image:
    var rr = r.ensure_packed_hwc_u8(True)
    var gg = g.ensure_packed_hwc_u8(True)
    var bb = b.ensure_packed_hwc_u8(True)

    var h = rr.height()
    if gg.height() < h: h = gg.height()
    if bb.height() < h: h = bb.height()

    var w = rr.width()
    if gg.width() < w: w = gg.width()
    if bb.width() < w: w = bb.width()

    if h <= 0 or w <= 0:
        return make_zero_u8_hwc(0, 0, 3, ColorSpace.SRGB())

    var out = make_zero_u8_hwc(h, w, 3, ColorSpace.SRGB())
    var pr = rr.tensor().data()
    var pg = gg.tensor().data()
    var pb = bb.tensor().data()
    var pout = out.tensor().data()

    var c = 3
    var s0 = w * c
    var s1 = c

    var y = 0
    while y < h:
        var xj = 0
        while xj < w:
            var base = y * s0 + xj * s1
            # BGR order in output
            pout[base + 0] = pb[y * w + xj]  # B
            pout[base + 1] = pg[y * w + xj]  # G
            pout[base + 2] = pr[y * w + xj]  # R
            xj += 1
        y += 1
    return out.copy()

# ---------------------------
# Histogram and Equalization
# ---------------------------

# 256-bin histogram of the first channel (returns Int counts).
fn histogram(img: Image) -> List[Int]:
    var x = img.ensure_packed_hwc_u8(True)
    var h = x.height()
    var w = x.width()
    var c = x.channels()

    var hist = List[Int]()
    var i = 0
    while i < 256:
        hist.append(0)
        i += 1

    if h <= 0 or w <= 0 or c <= 0:
        return hist.copy()

    var pin = x.tensor().data()
    var s0 = w * c
    var s1 = c

    var y = 0
    while y < h:
        var xj = 0
        while xj < w:
            var base = y * s0 + xj * s1
            var v = Int(pin[base + 0])  # first channel
            hist[v] = hist[v] + 1
            xj += 1
        y += 1
    return hist.copy()

# Computes a per-pixel histogram for a grayscale image (channel 0).
# If the image has multiple channels, it uses channel 0.
# Works on packed HWC/UInt8; converts if needed.
fn histogram(img: Image, bins: Int = 256) -> List[Int]:
    var base = img.ensure_packed_hwc_u8(True)

    var h = base.height()
    var w = base.width()

    # Allocate and zero-initialize the histogram
    var hist = List[Int]()
    var i = 0
    while i < bins:
        hist.append(0)
        i += 1

    # Iterate pixels and accumulate counts from channel 0
    var y = 0
    while y < h:
        var x = 0
        while x < w:
            var v = base.get_u8(y, x, 0)
            var idx = (Int(v) * bins) // 256
            if idx < 0:
                idx = 0
            if idx >= bins:
                idx = bins - 1
            hist[idx] = hist[idx] + 1
            x += 1
        y += 1

    return hist.copy()


fn histogram(img: List[List[UInt8]], bins: Int = 256) -> List[Int]:
    if bins <= 0:
        return List[Int]()

    var hist = List[Int]()
    hist.reserve(bins)
    var i = 0
    while i < bins:
        hist.append(0)
        i += 1

    var h = len(img)
    if h == 0:
        return hist.copy()
    var w = len(img[0])

    var y = 0
    while y < h:
        var x = 0
        while x < w:
            var v = Int(img[y][x])
            if 0 <= v and v < bins:
                hist[v] = hist[v] + 1
            x += 1
        y += 1

    return hist.copy()

# Histogram equalization for 1-channel images. If the image is not single-channel,
# it is returned unchanged.
fn equalize_hist(img: Image) -> Image:
    var x = img.ensure_packed_hwc_u8(True)
    var h = x.height()
    var w = x.width()
    var c = x.channels()

    if h <= 0 or w <= 0:
        return x.copy()
    if c != 1:
        # Only grayscale equalization supported
        return x.copy()

    # Compute histogram
    var hist = List[Int]()
    var i = 0
    while i < 256:
        hist.append(0)
        i += 1

    var pin = x.tensor().data()
    var y = 0
    while y < h:
        var xj = 0
        while xj < w:
            var v = Int(pin[y * w + xj])
            hist[v] = hist[v] + 1
            xj += 1
        y += 1

    # Compute CDF
    var cdf = List[Int]()
    i = 0
    var accum = 0
    while i < 256:
        accum = accum + hist[i]
        cdf.append(accum)
        i += 1

    var total = h * w
    if total == 0:
        return x.copy()

    # Find the first non-zero cdf value
    var cdf_min = 0
    i = 0
    while i < 256:
        if cdf[i] > 0:
            cdf_min = cdf[i]
            break
        i += 1

    # Map using equalization formula:
    # v' = round( (cdf[v] - cdf_min) / (total - cdf_min) * 255 )
    var out = x.clone()
    var pout = out.tensor().data()
    y = 0
    while y < h:
        var xj = 0
        while xj < w:
            var idx = y * w + xj
            var v = Int(pin[idx])
            var denom = total - cdf_min
            var mapped = 0
            if denom > 0:
                mapped = Int(((cdf[v] - cdf_min) * 255) // denom)
            if mapped < 0: mapped = 0
            if mapped > 255: mapped = 255
            pout[idx] = UInt8(mapped)
            xj += 1
        y += 1

    return out.copy()

# ---------------------------
# Histogram plotting
# ---------------------------

# Render a simple histogram canvas of size HxW = 100 x N (3-channel, BGR).
# Draws vertical bars whose heights are proportional to the counts.
fn plot_hist_u8(hist: List[Int]) -> Image:
    var nb = len(hist)
    if nb <= 0:
        nb = 256

    var H = 100
    var W = nb
    var canvas = make_zero_u8_hwc(H, W, 3, ColorSpace.SRGB())

    # Find max value to scale heights
    var maxv = 0
    var i = 0
    while i < nb and i < len(hist):
        if hist[i] > maxv: maxv = hist[i]
        i += 1
    if maxv <= 0:
        return canvas.copy()

    # Draw bars (white) using line()
    var y_bottom = H - 1
    var x = 0
    while x < W and x < len(hist):
        var h = hist[x]
        var bar_h = Int((Float64(h) / Float64(maxv)) * Float64(H - 1))
        if bar_h < 0: bar_h = 0
        var y_top = y_bottom - bar_h
        canvas = line(canvas, x, y_bottom, x, y_top, (UInt8(255), UInt8(255), UInt8(255)), 1)
        x += 1

    return canvas.copy()


fn plot_hist_u8(hist: List[Int], width: Int = 256, height: Int = 200) -> Image:
    var bins = len(hist)
    if bins == 0:
        return Image.new_hwc_u8(1, 1, 3, UInt8(0))

    # create black image
    var img = Image.new_hwc_u8(1, 1, 3, UInt8(0))

    # find max value for scaling
    var max_val = 0
    var i = 0
    while i < bins:
        if hist[i] > max_val:
            max_val = hist[i]
        i += 1
    if max_val == 0:
        max_val = 1

    # scale histogram into [0, height]
    var scale = Float64(height - 1) / Float64(max_val)

    # draw as vertical bars (white color)
    var x = 0
    while x < width and x < bins:
        var hval = Int(Float64(hist[x]) * scale)
        var y = height - 1
        while y >= height - hval:
            img.set_u8(y, x, 0, 255)
            img.set_u8(y, x, 1, 255)
            img.set_u8(y, x, 2, 255)
            y -= 1
        x += 1

    return img.copy()
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
# File: momijo/vision/transforms/color_jitter.mojo

 
from momijo.vision.tensor import Tensor, packed_hwc_strides
from momijo.vision.dtypes import DType

# ----------------------------------------
# Utilities
# ----------------------------------------
fn _clamp_u8_f32(x: Float32) -> UInt8:
    var v = x
    if v < 0.0: v = 0.0
    if v > 255.0: v = 255.0
    return UInt8(Int(v + 0.5))  # round-to-nearest

fn _apply_brightness(r: Float32, g: Float32, b: Float32, factor: Float32) -> (Float32, Float32, Float32):
    if factor == 1.0: return (r, g, b)
    return (r * factor, g * factor, b * factor)

fn _apply_contrast(r: Float32, g: Float32, b: Float32, factor: Float32) -> (Float32, Float32, Float32):
    if factor == 1.0: return (r, g, b)
    # pivot around 128
    var p: Float32 = 128.0
    return ((r - p) * factor + p, (g - p) * factor + p, (b - p) * factor + p)

# RGB <-> HSV helpers (Float32, RGB/HSV in [0,255] / [0..360, 0..1, 0..1] ranges)
fn _rgb_to_hsv(r: Float32, g: Float32, b: Float32) -> (Float32, Float32, Float32):
    var rf = r / 255.0
    var gf = g / 255.0
    var bf = b / 255.0

    var maxv = rf
    if gf > maxv: maxv = gf
    if bf > maxv: maxv = bf
    var minv = rf
    if gf < minv: minv = gf
    if bf < minv: minv = bf

    var v = maxv
    var d = maxv - minv
    var s: Float32 = 0.0
    if maxv > 0.0:
        s = d / maxv
    var h: Float32 = 0.0
    if d != 0.0:
        if maxv == rf:
            h = 60.0 * ((gf - bf) / d % 6.0)
        elif maxv == gf:
            h = 60.0 * ((bf - rf) / d + 2.0)
        else:
            h = 60.0 * ((rf - gf) / d + 4.0)
        if h < 0.0:
            h += 360.0
    return (h, s, v)

fn _hsv_to_rgb(h: Float32, s: Float32, v: Float32) -> (Float32, Float32, Float32):
    # h in [0,360), s,v in [0,1]
    if s == 0.0:
        var x = v * 255.0
        return (x, x, x)
    var hh = (h / 60.0)
    var i = Int(hh)
    var f = hh - Float32(i)
    var p = v * (1.0 - s)
    var q = v * (1.0 - s * f)
    var t = v * (1.0 - s * (1.0 - f))
    var rf: Float32 = 0.0
    var gf: Float32 = 0.0
    var bf: Float32 = 0.0
    if i == 0:
        rf = v; gf = t; bf = p
    elif i == 1:
        rf = q; gf = v; bf = p
    elif i == 2:
        rf = p; gf = v; bf = t
    elif i == 3:
        rf = p; gf = q; bf = v
    elif i == 4:
        rf = t; gf = p; bf = v
    else:
        rf = v; gf = p; bf = q
    return (rf * 255.0, gf * 255.0, bf * 255.0)

fn _apply_saturation_hue(r: Float32, g: Float32, b: Float32, sat_factor: Float32, hue_delta_deg: Float32) -> (Float32, Float32, Float32):
    if sat_factor == 1.0 and hue_delta_deg == 0.0:
        return (r, g, b)
    var (h, s, v) = _rgb_to_hsv(r, g, b)
    # adjust
    s = s * sat_factor
    if s < 0.0: s = 0.0
    if s > 1.0: s = 1.0
    h = h + hue_delta_deg
    # wrap hue
    while h < 0.0: h += 360.0
    while h >= 360.0: h -= 360.0
    return _hsv_to_rgb(h, s, v)

# ----------------------------------------
# Transform
# ----------------------------------------
@fieldwise_init
struct ColorJitter:
    var _brightness: Float32
    var _contrast:   Float32
    var _saturation: Float32
    var _hue_deg:    Float32   # degrees

    fn __init__(out self, brightness: Float64, contrast: Float64, saturation: Float64, hue: Float64):
        self._brightness = Float32(brightness)
        self._contrast   = Float32(contrast)
        self._saturation = Float32(saturation)
        self._hue_deg    = Float32(hue)

    # Apply to an HWC/u8 tensor; returns a NEW packed HWC/u8 tensor.
    fn __call__(self, x_in: Tensor) -> Tensor:
        assert(x_in.dtype() == DType.UInt8, "ColorJitter: only UInt8 supported")
        var h = x_in.height()
        var w = x_in.width()
        var c = x_in.channels()
        assert(h > 0 and w > 0 and c > 0, "ColorJitter: bad input shape")

        # Ensure we have a packed HWC source
        var x = x_in
        if not (x.stride2() == 1 and x.stride1() == c and x.stride0() == w * c):
            x = x_in.copy_to_packed_hwc()

        # Prepare output
        var (s0_out, s1_out, s2_out) = packed_hwc_strides(h, w, c)
        var out_len = h * w * c
        var out_buf = UnsafePointer[UInt8].alloc(out_len)
        var out = Tensor(out_buf, out_len, h, w, c, s0_out, s1_out, s2_out, DType.UInt8)

        var s0 = x.stride0()
        var s1 = x.stride1()
        var s2 = x.stride2()
        var sp = x.data()

        var y:Int = 0
        if c == 1:
            # Grayscale: brightness + contrast only
            while y < h:
                var xcol:Int = 0
                while xcol < w:
                    var base = y * s0 + xcol * s1
                    var v0 = Float32(sp[base])
                    var v1 = v0 * self._brightness
                    var v2 = (v1 - 128.0) * self._contrast + 128.0
                    out_buf[y * s0_out + xcol * s1_out] = _clamp_u8_f32(v2)
                    xcol += 1
                y += 1
            return out.copy()

        # RGB (C>=3): full pipeline
        while y < h:
            var xcol:Int = 0
            while xcol < w:
                var base = y * s0 + xcol * s1
                var r = Float32(sp[base + 0 * s2])
                var g = Float32(sp[base + 1 * s2])
                var b = Float32(sp[base + 2 * s2])

                # brightness
                (r, g, b) = _apply_brightness(r, g, b, self._brightness)
                # contrast
                (r, g, b) = _apply_contrast(r, g, b, self._contrast)
                # saturation & hue
                (r, g, b) = _apply_saturation_hue(r, g, b, self._saturation, self._hue_deg)

                out_buf[y * s0_out + xcol * s1_out + 0 * s2_out] = _clamp_u8_f32(r)
                out_buf[y * s0_out + xcol * s1_out + 1 * s2_out] = _clamp_u8_f32(g)
                out_buf[y * s0_out + xcol * s1_out + 2 * s2_out] = _clamp_u8_f32(b)

                # If there are extra channels (e.g., alpha), copy-through
                var ch = 3
                while ch < c:
                    out_buf[y * s0_out + xcol * s1_out + ch * s2_out] = sp[base + ch * s2]
                    ch += 1

                xcol += 1
            y += 1

        return out.copy()

# Convenience functional wrapper
fn color_jitter(x: Tensor, brightness: Float64, contrast: Float64, saturation: Float64, hue: Float64) -> Tensor:
    var op = ColorJitter(brightness, contrast, saturation, hue)
    return op(x)



fn clahe_color_bgr(img: List[List[List[UInt8]]], clip_limit: Float64 = 2.0, tile: Int = 8) -> List[List[List[UInt8]]]:
    var h = len(img)
    if h == 0: return img.copy()
    var w = len(img[0])
    if w == 0: return img.copy()

    # Step 1: Convert BGR -> LAB (only L channel will be enhanced)
    # LAB range simplified: L in [0,255], A,B in [0,255]
    var L = List[List[UInt8]]()
    var A = List[List[UInt8]]()
    var B = List[List[UInt8]]()
    L.reserve(h); A.reserve(h); B.reserve(h)

    var y = 0
    while y < h:
        var rowL = List[UInt8]()
        var rowA = List[UInt8]()
        var rowB = List[UInt8]()
        var x = 0
        while x < w:
            var b = img[y][x][0]
            var g = img[y][x][1]
            var r = img[y][x][2]

            # naive luminance as L, just average (placeholder)
            var l_val = UInt8((Int(r) + Int(g) + Int(b)) // 3)
            rowL.append(l_val)
            rowA.append(g)  # placeholder
            rowB.append(r)  # placeholder
            x += 1
        L.append(rowL.copy())
        A.append(rowA.copy())
        B.append(rowB.copy())
        y += 1

    # Step 2: Apply CLAHE to L
    var L_eq = _clahe_gray(L, clip_limit, tile)

    # Step 3: Merge back LAB -> BGR
    var out = List[List[List[UInt8]]]()
    out.reserve(h)
    y = 0
    while y < h:
        var row = List[List[UInt8]]()
        row.reserve(w)
        var x = 0
        while x < w:
            var px = List[UInt8]()
            px.append(L_eq[y][x])     # B
            px.append(A[y][x])       # G
            px.append(B[y][x])       # R
            row.append(px.copy())
            x += 1
        out.append(row.copy())
        y += 1
    return out.copy()

# Helper: CLAHE on grayscale
fn _clahe_gray(img: List[List[UInt8]], clip_limit: Float64, tile: Int) -> List[List[UInt8]]:
    # Very simplified version: just call histogram equalization per tile
    # (Not full CLAHE, but placeholder)
    return img.copy()



# Helper: convert Image -> HxWx3 UInt8 list
fn _image_to_u8_3d(img: Image) -> List[List[List[UInt8]]]:
    var h = img.height()
    var w = img.width()
    var out = List[List[List[UInt8]]]()
    out.reserve(h)
    var y = 0
    while y < h:
        var row = List[List[UInt8]]()
        row.reserve(w)
        var x = 0
        while x < w:
            var px = List[UInt8]()
            px.append(img.at_u8(y, x, 0))
            px.append(img.at_u8(y, x, 1))
            px.append(img.at_u8(y, x, 2))
            row.append(px.copy())
            x += 1
        out.append(row.copy())
        y += 1
    return out.copy()

# Helper: convert HxWx3 UInt8 list -> Image
fn _u8_3d_to_image(rgb: List[List[List[UInt8]]]) -> Image:
    var h = len(rgb)
    if h == 0:
        return full((1, 1, 3), UInt8(0))
    var w = len(rgb[0])
    var img = full((h, w, 3), UInt8(0))
    var y = 0
    while y < h:
        var x = 0
        while x < w:
            var px = rgb[y][x].copy()  # [R,G,B] or [B,G,R] depending on your convention
            img.set_u8(y, x, 0, px[0])
            img.set_u8(y, x, 1, px[1])
            img.set_u8(y, x, 2, px[2])
            x += 1
        y += 1
    return img.copy()

# Overload: CLAHE on Image by adapting to existing list-based implementation.
# Assumes you already have: fn clahe_color_bgr(rgb: List[List[List[UInt8]]],
#                                             clip_limit: Float64 = 2.0,
#                                             tile: Int = 8) -> List[List[List[UInt8]]]
fn clahe_color_bgr(img: Image, clip_limit: Float64 = 2.0, tile: Int = 8) -> Image:
    var rgb = _image_to_u8_3d(img)
    var enhanced = clahe_color_bgr(rgb, clip_limit, tile)   # resolves to the list-based overload
    return _u8_3d_to_image(enhanced)