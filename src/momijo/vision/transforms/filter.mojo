# MIT License
# SPDX-License-Identifier: MIT
# Project: momijo.vision | File: src/momijo/vision/transforms/filter.mojo

from momijo.vision.image import Image
from math import exp

# --------------------------- helpers ---------------------------

fn _odd_at_least(v: Int, lo: Int = 1) -> Int:
    var x = v
    if x < lo:
        x = lo
    if (x & 1) == 0:
        x = x + 1
    return x

fn _clamp_i(x: Int, lo: Int, hi: Int) -> Int:
    var v = x
    if v < lo:
        v = lo
    if v > hi:
        v = hi
    return v

# --------------------------- filters ---------------------------

# Gaussian blur: validates ksize/sigma; keeps image content as-is.
# Internal helper: validates odd kernel sizes and non-negative sigma.
fn _sanitize_kernel(v: Int, minimum: Int = 1) -> Int:
    var x = v
    if x < minimum:
        x = minimum
    if (x % 2) == 0:
        x = x + 1
    return x

fn _sanitize_sigma(s: Float64) -> Float64:
    if s < 0.0:
        return 0.0
    return s

# TODO: Replace with real Gaussian implementation (separable conv).
fn _gaussian_blur_impl(img: Image, kx: Int, ky: Int, sigma: Float64) -> Image:
    var _kx = _sanitize_kernel(kx, 1)
    var _ky = _sanitize_kernel(ky, 1)
    var _s  = _sanitize_sigma(sigma)
    # Placeholder: return input image unchanged for now.
    return img

# --- Overload 1: single ksize (positional sigma) ---
fn gaussian_blur(img: Image, ksize: Int, sigma: Float64) -> Image:
    var k = _sanitize_kernel(ksize, 1)
    var s = _sanitize_sigma(sigma)
    return _gaussian_blur_impl(img, k, k, s)

# --- Overload 2: kx, ky with keyword-only sigma ---
fn gaussian_blur(img: Image, kx: Int, ky: Int, *, sigma: Float64 = 0.0) -> Image:
    var k0 = _sanitize_kernel(kx, 1)
    var k1 = _sanitize_kernel(ky, 1)
    var s  = _sanitize_sigma(sigma)
    return _gaussian_blur_impl(img, k0, k1, s)

# Median blur: validates odd ksize.
fn median_blur(img: Image, ksize: Int) -> Image:
    var k = _odd_at_least(ksize, 1)
    # Placeholder: return input image unchanged.
    return img

# Sobel: validates derivative orders and kernel size {1,3,5,7}.
fn sobel(img: Image, dx: Int, dy: Int, ksize: Int = 3) -> Image:
    var k = ksize
    if k != 1 and k != 3 and k != 5 and k != 7:
        # normalize to nearest supported odd size
        k = _odd_at_least(ksize, 1)
        if k > 7:
            k = 7
    var dxx = dx
    var dyy = dy
    if dxx < 0:
        dxx = 0
    if dyy < 0:
        dyy = 0
    # Placeholder: return input image unchanged.
    return img

# Laplacian: validates kernel size {1,3,5,7}.
fn laplacian(img: Image, ksize: Int = 3) -> Image:
    var k = ksize
    if k != 1 and k != 3 and k != 5 and k != 7:
        k = _odd_at_least(ksize, 1)
        if k > 7:
            k = 7
    # Placeholder: return input image unchanged.
    return img

# Canny: normalizes thresholds and aperture (odd in [3,7]).
fn canny(img: Image, t1: Int, t2: Int, aperture: Int = 3) -> Image:
    var low = t1
    var high = t2
    if low > high:
        var tmp = low
        low = high
        high = tmp

    var ap = aperture
    ap = _odd_at_least(ap, 3)
    ap = _clamp_i(ap, 3, 7)

    # Placeholder: return input image unchanged.
    return img

# Simple sqrt approximation (Newton iterations), good enough for gradients.
fn _sqrt64(x: Float64) -> Float64:
    if x <= 0.0:
        return 0.0
    var g = x
    var i = 0
    while i < 8:
        g = 0.5 * (g + x / g)
        i = i + 1
    return g

# Computes gradient magnitude from gx, gy (use channel 0), output UInt8 in [0..255].
# Works on packed HWC/UInt8; will convert inputs if needed.
fn magnitude_u8(gx: Image, gy: Image) -> Image:
    # Ensure packed HWC/UInt8 for safe pixel access
    var ax = gx.ensure_packed_hwc_u8(True)
    var ay = gy.ensure_packed_hwc_u8(True)

    # Use overlap size in case of slight mismatch
    var h = ax.height()
    var w = ax.width()
    var h2 = ay.height()
    var w2 = ay.width()
    if h2 < h:
        h = h2
    if w2 < w:
        w = w2
    if h <= 0 or w <= 0:
        return Image.new_hwc_u8(1, 1, 1, UInt8(0))

    var out = Image.new_hwc_u8(h, w, 1, UInt8(0))

    var y = 0
    while y < h:
        var x = 0
        while x < w:
            # Read channel 0 (assumes grayscale or first channel is intensity)
            var vx_u8 = ax.get_u8(y, x, 0)
            var vy_u8 = ay.get_u8(y, x, 0)
            var vx = Float64(vx_u8)
            var vy = Float64(vy_u8)

            var mag = _sqrt64(vx * vx + vy * vy)
            if mag > 255.0:
                mag = 255.0
            if mag < 0.0:
                mag = 0.0

            out.set_u8(y, x, 0, UInt8(mag))
            x = x + 1
        y = y + 1

    return out

# Create a single-channel UInt8 image of size (h, w) filled with ones.
fn ones_u8(h: Int, w: Int) -> Image:
    var out = Image.full_hwc_u8(h, w, 1, UInt8(1))
    return out

# ----- Bilateral Filter utils -----

fn _exp_neg_half_over_sigma2(x2: Float64, sigma: Float64) -> Float64:
    # exp( - x2 / (2 * sigma^2) )
    var s = sigma
    if s <= 0.0:
        # When sigma is zero or negative, fall back to a hard weight of 0 (except zero-distance).
        # Caller avoids using this branch for dx=dy=0.
        return 0.0
    var denom = 2.0 * s * s
    return exp(-x2 / denom)

# Precompute spatial Gaussian weights for a given radius and sigma_space.
# Returns a flattened (2r+1)*(2r+1) kernel in row-major: idx = (dy + r) * size + (dx + r)
fn _make_spatial_kernel(radius: Int, sigma_space: Float64) -> (List[Float64], Int):
    var r = radius
    var size = 2 * r + 1
    var kern = List[Float64]()
    var dy = -r
    while dy <= r:
        var dx = -r
        while dx <= r:
            var dist2 = Float64(dx * dx + dy * dy)
            var w = _exp_neg_half_over_sigma2(dist2, sigma_space)
            kern.append(w)
            dx = dx + 1
        dy = dy + 1
    return (kern, size)

# ----- Bilateral Filter (naive O(H*W*d*d)) -----

# Applies bilateral filtering on an image.
# - d: diameter of the pixel neighborhood (odd; if <= 0, it will be forced to >= 1 and odd)
# - sigma_color: range sigma (intensity domain)
# - sigma_space: spatial sigma (distance domain)
#
# Notes:
# - Handles both single-channel and multi-channel images.
# - Border handling: clamp-to-edge.
# - Output type matches input type; assumes 8-bit images for simplicity (UInt8 channels).
fn bilateral_filter(img: Image, d: Int, sigma_color: Float64, sigma_space: Float64) -> Image:
    var hh = img.height()
    var ww = img.width()
    var cc = img.channels()

    if hh <= 0 or ww <= 0 or cc <= 0:
        return Image.new_hwc_u8(1, 1, 1, UInt8(0))

    # Force odd diameter and compute radius
    var diameter = _odd_at_least(d, 1)
    var radius = diameter // 2

    # Precompute spatial kernel
    var kernel_and_size = _make_spatial_kernel(radius, sigma_space)
    var spatial = kernel_and_size[0]
    var ksize = kernel_and_size[1]

    # Prepare output image with same shape/type
    var out = Image.new_hwc_u8(hh, ww, cc, UInt8(0))

    var y = 0
    while y < hh:
        var x = 0
        while x < ww:
            var ch = 0
            while ch < cc:
                # Center value (as Float64)
                var center_v = Float64(img.get_u8(y, x, ch))

                # Accumulators
                var wsum = 0.0
                var acc = 0.0

                var dy = -radius
                while dy <= radius:
                    var ny = y + dy
                    if ny < 0:
                        ny = 0
                    if ny >= hh:
                        ny = hh - 1

                    var dx = -radius
                    while dx <= radius:
                        var nx = x + dx
                        if nx < 0:
                            nx = 0
                        if nx >= ww:
                            nx = ww - 1

                        # Spatial weight from precomputed kernel
                        var kidx = (dy + radius) * ksize + (dx + radius)
                        var w_spatial = spatial[kidx]

                        # Range (color) weight
                        var neigh_v = Float64(img.get_u8(ny, nx, ch))
                        var diff = neigh_v - center_v
                        var diff2 = diff * diff
                        var w_range = _exp_neg_half_over_sigma2(diff2, sigma_color)

                        var w = w_spatial * w_range

                        wsum = wsum + w
                        acc = acc + (w * neigh_v)

                        dx = dx + 1
                    dy = dy + 1

                # Normalize and write
                var out_v = center_v
                if wsum > 0.0:
                    out_v = acc / wsum

                # Clamp to 0..255 and cast to UInt8
                if out_v < 0.0:
                    out_v = 0.0
                if out_v > 255.0:
                    out_v = 255.0
                out.set_u8(y, x, ch, UInt8(out_v))

                ch = ch + 1
            x = x + 1
        y = y + 1

    return out

# Alias for OpenCV-style naming: bilateral_blur
fn bilateral_blur(img: Image, d: Int, sigma_color: Float64, sigma_space: Float64) -> Image:
    return bilateral_filter(img, d, sigma_color, sigma_space)
