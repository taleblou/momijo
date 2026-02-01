# MIT License
# SPDX-License-Identifier: MIT
# Project: momijo.vision | File: src/momijo/vision/transforms/draw.mojo

from momijo.vision.image import Image
from momijo.vision.dtypes import DType  # for explicit checks
from momijo.vision.transforms.array import full
from momijo.vision.transforms.features import Keypoint, keypoint_xy, len_keypoints

from momijo.vision.transforms.glyphs import _glyph5x7_rowmajor

from collections.list import List
from math import sqrt

fn as_u8(xs: List[Int]) -> List[UInt8]:
    var out = List[UInt8]()
    out.reserve(len(xs))
    var i = 0
    while i < len(xs):
        var v = xs[i]
        if v < 0: v = 0
        if v > 255: v = 255
        out.append(UInt8(v))
        i += 1
    return out.copy()
# ---------- Small integer helpers ----------
fn iabs(v: Int) -> Int:
    return -v if v < 0 else v

# ---------- Pixel write hook ----------
# Writes a BGR pixel into a packed HWC/UInt8 image at (x,y).
fn _poke(mut img: Image, x: Int, y: Int, b: UInt8, g: UInt8, r: UInt8):
    # --- Dimensions & bounds ---
    var w = img.width()
    var h = img.height()
    if x < 0 or y < 0 or x >= w or y >= h:
        return

    # --- Channels ---
    var c = img.channels()
    if c <= 0:
        return

    # --- Linear index & guard ---
    var base = (y * w + x) * c
    var total = w * h * c
    if base < 0 or base + (c - 1) >= total:
        return

    # --- Pointer and BEFORE values ---
    var p = img.tensor().data()

    var before0: UInt8 = p[base]
    var before1: UInt8 = 0
    var before2: UInt8 = 0
    if c > 1:
        before1 = p[base + 1]
    if c > 2:
        before2 = p[base + 2]



    # --- WRITE ---
    p[base] = b
    if c > 1:
        p[base + 1] = g
    if c > 2:
        p[base + 2] = r



    # --- AFTER values ---
    var after0: UInt8 = p[base]
    var after1: UInt8 = 0
    var after2: UInt8 = 0
    if c > 1:
        after1 = p[base + 1]
    if c > 2:
        after2 = p[base + 2]








# ---------- Helpers ----------

fn _in_bounds(img: Image, x: Int, y: Int) -> Bool:
    return (x >= 0 and x < img.width() and y >= 0 and y < img.height())



fn _draw_disk(mut img: Image, cx: Int, cy: Int, r: Int, color: (UInt8, UInt8, UInt8)):
    var rr = r
    if rr <= 0:
        return
    var y = -rr
    while y <= rr:
        var x = -rr
        while x <= rr:
            if x * x + y * y <= rr * rr:
                _draw_point(img, cx + x, cy + y, color)
            x += 1
        y += 1

# ---------- Public API ----------
 # then draw in-place on that buffer and return the (possibly new) Image.



# line overloads
fn line(mut img: Image, x1: Int, y1: Int, x2: Int, y2: Int, color: (Int, Int, Int), thickness: Int = 1) -> Image:
    return line(img, x1, y1, x2, y2, _to_u8_color_from_int3(color), thickness)

fn line(mut img: Image, x1: Int, y1: Int, x2: Int, y2: Int, color: List[Int], thickness: Int = 1) -> Image:
    return line(img, x1, y1, x2, y2, _to_u8_color_from_list(color), thickness)

# Accept List[UInt8] and forward to the List[Int] overload
fn line(mut img: Image,x1: Int, y1: Int, x2: Int, y2: Int,color: List[UInt8],thickness: Int = 1) -> Image:
    var tmp = List[Int]()
    var i = 0
    while i < len(color):
        tmp.append(Int(color[i]))
        i += 1
    return line(img, x1, y1, x2, y2, tmp, thickness)


fn line(mut img: Image, x1: Int, y1: Int, x2: Int, y2: Int,
        color: (UInt8, UInt8, UInt8), thickness: Int = 1) -> Image:
    img = img.ensure_packed_hwc_u8(True)

    # Bresenham with optional thickness via orthogonal disk stamping
    var dx = iabs(x2 - x1)
    var sx = 1 if x1 < x2 else -1
    var dy = -iabs(y2 - y1)
    var sy = 1 if y1 < y2 else -1
    var err = dx + dy
    var px = x1
    var py = y1
    var t = thickness
    if t < 1: t = 1

    while True:
        if t == 1:
            _draw_point(img, px, py, color)
        else:
            _draw_disk(img, px, py, t // 2, color)

        if px == x2 and py == y2:
            break
        var e2 = 2 * err
        if e2 >= dy:
            err = err + dy
            px = px + sx
        if e2 <= dx:
            err = err + dx
            py = py + sy
    return img.copy()

# -------- helpers --------
fn _min(a: Int, b: Int) -> Int:
    if a < b: return a
    return b

fn _max(a: Int, b: Int) -> Int:
    if a > b: return a
    return b

fn _clamp(v: Int, lo: Int, hi: Int) -> Int:
    if v < lo: return lo
    if v > hi: return hi
    return v

fn _put_color(dst: Image, y: Int, x: Int, color: List[UInt8]):
    # Writes color, broadcasting
    var c = dst.channels()
    if len(color) == 0:
        var ch = 0
        while ch < c:
            dst.set_u8(y, x, ch, 0)
            ch += 1
        return
    var ch = 0
    while ch < c:
        var idx = if ch < len(color) then ch else (len(color) - 1)
        dst.set_u8(y, x, ch, color[idx])
        ch += 1

fn _put_gray(dst: Image, y: Int, x: Int, v: UInt8):
    var c = dst.channels()
    var ch = 0
    while ch < c:
        dst.set_u8(y, x, ch, v)
        ch += 1

# -------- core draw --------
# thickness > 0: outline with that thickness
# thickness == 0: no-op
# thickness < 0: filled rectangle
# --- Core: draws either filled (thickness < 0) or outlined (thickness > 0) ---
# Color order is (b, g, r) to match your _poke signature.
fn _rectangle_core(src: Image,
                   x0: Int, y0: Int, x1: Int, y1: Int,
                   thickness: Int,
                   b: UInt8, g: UInt8, r: UInt8) -> Image:
    var h = src.height()
    var w = src.width()
    if h == 0 or w == 0:
        return src.copy()

    var x_min = _max(_min(x0, x1), 0)
    var x_max = _min(_max(x0, x1), w - 1)
    var y_min = _max(_min(y0, y1), 0)
    var y_max = _min(_max(y0, y1), h - 1)

    if x_min > x_max or y_min > y_max:
        return src.copy()

    # Clone first, then if packed HWC/u8
    var dst = src.ensure_packed_hwc_u8(True).clone()


    if thickness < 0:
        var y = y_min
        while y <= y_max:
            var x = x_min
            while x <= x_max:
                _poke(dst, x, y, b, g, r)
                x += 1
            y += 1
        return dst.copy()

    if thickness == 0:
        return dst.copy()

    var t = thickness

    # Top
    var y = y_min
    while y < _min(y_min + t, y_max + 1):
        var x = x_min
        while x <= x_max:
            _poke(dst, x, y, b, g, r)
            x += 1
        y += 1

    # Bottom
    y = _max(y_max - t + 1, y_min)
    while y <= y_max:
        var x2 = x_min
        while x2 <= x_max:
            _poke(dst, x2, y, b, g, r)
            x2 += 1
        y += 1

    # Left
    var xL = x_min
    while xL < _min(x_min + t, x_max + 1):
        var yy = y_min
        while yy <= y_max:
            _poke(dst, xL, yy, b, g, r)
            yy += 1
        xL += 1

    # Right
    var xR = _max(x_max - t + 1, x_min)
    while xR <= x_max:
        var yy2 = y_min
        while yy2 <= y_max:
            _poke(dst, xR, yy2, b, g, r)
            yy2 += 1
        xR += 1

    return dst.copy()


# ---------------- Public APIs ----------------

# Grayscale rectangle (single channel value expanded to BGR)
fn rectangle_gray(src: Image,
                  x0: Int, y0: Int, x1: Int, y1: Int,
                  gray: UInt8, thickness: Int = 1) -> Image:
    return _rectangle_core(src, x0, y0, x1, y1, thickness,
                           gray, gray, gray)

# Color rectangle with (UInt8, UInt8, UInt8) in BGR order
fn rectangle_color(src: Image,
                   x0: Int, y0: Int, x1: Int, y1: Int,
                   color: (UInt8, UInt8, UInt8),
                   thickness: Int = 1) -> Image:
    return _rectangle_core(src, x0, y0, x1, y1, thickness,
                           color[0], color[1], color[2])


# Overloads for convenience (same names, different 6th arg type)
fn rectangle(src: Image,
             x0: Int, y0: Int, x1: Int, y1: Int,
             gray: UInt8, thickness: Int = 1) -> Image:
    return rectangle_gray(src, x0, y0, x1, y1, gray, thickness)

fn rectangle(src: Image,
             x0: Int, y0: Int, x1: Int, y1: Int,
             color: (UInt8, UInt8, UInt8), thickness: Int = 1) -> Image:
    return rectangle_color(src, x0, y0, x1, y1, color, thickness)



# circle overloads
fn circle(mut img: Image, cx: Int, cy: Int, r: Int, color: (Int, Int, Int), thickness: Int = 1) -> Image:
    return circle(img, cx, cy, r, _to_u8_color_from_int3(color), thickness)

fn circle(mut img: Image, cx: Int, cy: Int, r: Int, color: List[Int], thickness: Int = 1) -> Image:
    return circle(img, cx, cy, r, _to_u8_color_from_list(color), thickness)

# Accept List[UInt8] and forward to the List[Int] overload.
fn circle(mut img: Image, cx: Int, cy: Int, r: Int, color: List[UInt8], thickness: Int = 1) -> Image:
    var tmp = List[Int]()
    var i = 0
    while i < len(color):
        tmp.append(Int(color[i]))
        i += 1
    return circle(img, cx, cy, r, tmp, thickness)


fn circle(mut img: Image, cx: Int, cy: Int, r: Int,
          color: (UInt8, UInt8, UInt8), thickness: Int = 1) -> Image:
    img = img.ensure_packed_hwc_u8(True)
    if r <= 0:
        return img.copy()
    if thickness < 0:
        _draw_disk(img, cx, cy, r, color)
        return img.copy()

    # Midpoint circle algorithm
    var x = r
    var y = 0
    var err = 1 - x
    var t = thickness
    if t < 1: t = 1

    fn _stamp(mut img: Image, px: Int, py: Int, t: Int, color: (UInt8, UInt8, UInt8)):
        if t == 1:
            _draw_point(img, px, py, color)
        else:
            _draw_disk(img, px, py, t // 2, color)

    while x >= y:
        _stamp(img, cx + x, cy + y, t, color)
        _stamp(img, cx + y, cy + x, t, color)
        _stamp(img, cx - y, cy + x, t, color)
        _stamp(img, cx - x, cy + y, t, color)
        _stamp(img, cx - x, cy - y, t, color)
        _stamp(img, cx - y, cy - x, t, color)
        _stamp(img, cx + y, cy - x, t, color)
        _stamp(img, cx + x, cy - y, t, color)

        y += 1
        if err < 0:
            err += 2 * y + 1
        else:
            x -= 1
            err += 2 * (y - x) + 1
    return img.copy()
# --- fill_poly main implementation -------------------------------------------
# Canonical overload: pts = List[(Int, Int)], color = (UInt8, UInt8, UInt8)
fn fill_poly(mut img: Image, pts: List[(Int, Int)], color: (UInt8, UInt8, UInt8)) -> Image:
    img = img.ensure_packed_hwc_u8(True)

    var n = len(pts)
    if n < 3:
        return img.copy()

    # Early exit if image is empty
    var W = img.width()
    var H = img.height()
    if W <= 0 or H <= 0:
        return img.copy()

    # Compute scanline range
    var min_y = H - 1
    var max_y = 0
    var i = 0
    while i < n:
        var (_, py) = pts[i]
        if py < min_y: min_y = py
        if py > max_y: max_y = py
        i += 1

    # Clamp to image bounds
    if min_y < 0: min_y = 0
    if max_y >= H: max_y = H - 1
    if min_y > max_y:
        return img.copy()

    var y = min_y
    while y <= max_y:
        # Collect intersections with scanline y
        var xs = List[Int]()
        var j = 0
        while j < n:
            var k = (j + 1) % n
            var (x0, y0) = pts[j]
            var (x1, y1) = pts[k]

            # Count edges that cross the scanline (exclude horizontal)
            if ((y0 <= y and y1 > y) or (y1 <= y and y0 > y)):
                # Safe: y1 - y0 != 0 here by the condition above
                var x = x0 + (y - y0) * (x1 - x0) // (y1 - y0)
                xs.append(x)
            j += 1

        # Insertion sort (xs is tiny per scanline)
        var a = 1
        while a < len(xs):
            var key = xs[a]
            var b = a - 1
            while b >= 0 and xs[b] > key:
                xs[b + 1] = xs[b]
                b -= 1
            xs[b + 1] = key
            a += 1

        # Fill spans in pairs
        var p = 0
        while p + 1 < len(xs):
            var x0s = xs[p]
            var x1s = xs[p + 1]
            # Order and clamp
            if x0s > x1s:
                var tmp = x0s
                x0s = x1s
                x1s = tmp
            if x0s < 0: x0s = 0
            if x1s >= W: x1s = W - 1
            if x0s <= x1s:
                img = line(img, x0s, y, x1s, y, color, 1)
            p += 2
        y += 1

    return img.copy()


# --------- Helpers to normalize colors ----------
@always_inline
fn _to_u8(x: Int) -> UInt8:
    var v = x
    if v < 0: v = 0
    if v > 255: v = 255
    return UInt8(v)

@always_inline
fn _color_from_int_tuple(color: (Int, Int, Int)) -> (UInt8, UInt8, UInt8):
    return (_to_u8(color[0]), _to_u8(color[1]), _to_u8(color[2]))

@always_inline
fn _color_from_list_int(color: List[Int]) -> (UInt8, UInt8, UInt8):
    var r = 0; var g = 0; var b = 0
    if len(color) > 0: b = color[0]
    if len(color) > 1: g = color[1]
    if len(color) > 2: r = color[2]
    return (_to_u8(b), _to_u8(g), _to_u8(r))

@always_inline
fn _color_from_list_u8(color: List[UInt8]) -> (UInt8, UInt8, UInt8):
    var b: UInt8 = UInt8(0)
    var g: UInt8 = UInt8(0)
    var r: UInt8 = UInt8(0)
    if len(color) > 0: b = color[0]
    if len(color) > 1: g = color[1]
    if len(color) > 2: r = color[2]
    return (b, g, r)


# --------- Point converters (assumed provided) ----------
# Here are minimal stand-ins to avoid ambiguity:

@always_inline
fn _pts_from_flat_int(flat: List[Int]) -> List[(Int, Int)]:
    var out = List[(Int, Int)]()
    var i = 0
    while i + 1 < len(flat):
        out.append((flat[i], flat[i + 1]))
        i += 2
    return out.copy()

@always_inline
fn _pts_from_list_list_int(nested: List[List[Int]]) -> List[(Int, Int)]:
    var out = List[(Int, Int)]()
    var i = 0
    while i < len(nested):
        var row = nested[i].copy()
        var x = 0; var y = 0
        if len(row) > 0: x = row[0]
        if len(row) > 1: y = row[1]
        out.append((x, y))
        i += 1
    return out.copy()


# --------- Overloads: pts as List[Int] (flat) ----------
fn fill_poly(mut img: Image, pts: List[Int], color: (UInt8, UInt8, UInt8)) -> Image:
    return fill_poly(img, _pts_from_flat_int(pts), color)

fn fill_poly(mut img: Image, pts: List[Int], color: (Int, Int, Int)) -> Image:
    return fill_poly(img, _pts_from_flat_int(pts), _color_from_int_tuple(color))

fn fill_poly(mut img: Image, pts: List[Int], color: List[Int]) -> Image:
    return fill_poly(img, _pts_from_flat_int(pts), _color_from_list_int(color))

fn fill_poly(mut img: Image, pts: List[Int], color: List[UInt8]) -> Image:
    return fill_poly(img, _pts_from_flat_int(pts), _color_from_list_u8(color))


# --------- Overloads: pts as List[List[Int]] ----------
fn fill_poly(mut img: Image, pts: List[List[Int]], color: (UInt8, UInt8, UInt8)) -> Image:
    return fill_poly(img, _pts_from_list_list_int(pts), color)

fn fill_poly(mut img: Image, pts: List[List[Int]], color: (Int, Int, Int)) -> Image:
    return fill_poly(img, _pts_from_list_list_int(pts), _color_from_int_tuple(color))

fn fill_poly(mut img: Image, pts: List[List[Int]], color: List[Int]) -> Image:
    return fill_poly(img, _pts_from_list_list_int(pts), _color_from_list_int(color))

fn fill_poly(mut img: Image, pts: List[List[Int]], color: List[UInt8]) -> Image:
    return fill_poly(img, _pts_from_list_list_int(pts), _color_from_list_u8(color))




# convenience overload: accepts (Int,Int,Int) and casts to UInt8
# Draw multiple circles with a given BGR color (UInt8), without nested helpers.
# thickness < 0  -> filled circle
# thickness == 0 -> no-op
# thickness > 0  -> ring with outer radius = r and inner radius = r - thickness + 1
fn _draw_circles_u8(
    mut img: Image,
    circles: List[(Int, Int, Int)],          # (cx, cy, radius)
    color: (UInt8, UInt8, UInt8),            # B, G, R
    thickness: Int,
    center_color: (UInt8, UInt8, UInt8),     # B, G, R
    center_radius: Int
) -> Image:
    var (b, g, r) = color
    var (cb, cg, cr) = center_color

    for (cx, cy, rad) in circles:
        if rad <= 0:
            continue

        # Optional filled center dot
        if center_radius > 0:
            var cr_sq = center_radius * center_radius
            var y0 = cy - center_radius
            var y1 = cy + center_radius
            var x0 = cx - center_radius
            var x1 = cx + center_radius
            var y = y0
            while y <= y1:
                var dy = y - cy
                var x = x0
                while x <= x1:
                    var dx = x - cx
                    var d2 = dx * dx + dy * dy
                    if d2 <= cr_sq:
                        _poke(img, x, y, cb, cg, cr)  # (x, y), BGR
                    x += 1
                y += 1

        # Determine fill/ring parameters
        if thickness == 0:
            # no-op
            continue

        var r_out = rad
        var r_in = 0
        if thickness >= 0:
            r_in = rad - thickness + 1
        if r_in < 0:
            r_in = 0

        if r_in < 0:
            r_in = 0
        if r_in > r_out:
            # nothing to draw
            continue

        var r_out_sq = r_out * r_out
        var r_in_sq  = r_in * r_in

        # Scan bounding box once and test d^2 against the annulus range
        var yA = cy - r_out
        var yB = cy + r_out
        var xA = cx - r_out
        var xB = cx + r_out

        var y2 = yA
        while y2 <= yB:
            var dy2 = y2 - cy
            var x2 = xA
            while x2 <= xB:
                var dx2 = x2 - cx
                var d2  = dx2 * dx2 + dy2 * dy2
                if d2 <= r_out_sq and d2 >= r_in_sq:
                    _poke(img, x2, y2, b, g, r)  # (x, y), BGR
                x2 += 1
            y2 += 1

    return img.copy()


 # Int-color wrapper that converts to UInt8 BGR and calls the u8 version.
# Input colors are assumed RGB; convert to internal BGR ordering.
# fn draw_circles(
#     mut img: Image,
#     circles: List[(Int, Int, Int)],
#     color_rgb: (Int, Int, Int),
#     thickness: Int = 1,
#     center_color_rgb: (Int, Int, Int) = (0, 0, 0),
#     center_radius: Int = 0
# ) -> Image:
#     var (r, g, b) = color_rgb
#     var (cr, cg, cb) = center_color_rgb
#     var col_bgr: (UInt8, UInt8, UInt8)  = (UInt8(b),  UInt8(g),  UInt8(r))
#     var ccol_bgr: (UInt8, UInt8, UInt8) = (UInt8(cb), UInt8(cg), UInt8(cr))
#     return _draw_circles_u8(img, circles, col_bgr, thickness, ccol_bgr, center_radius)

# Int-based signature (RGB), converts internally to BGR UInt8 and forwards.
fn draw_circles(
    mut img: Image,
    circles: List[(Int, Int, Int)],
    color_rgb: (Int, Int, Int),
    thickness: Int = 1,
    center_color_rgb: (Int, Int, Int) = (0, 0, 0),
    center_radius: Int = 0
) -> Image:
    var (r, g, b) = color_rgb
    var (cr, cg, cb) = center_color_rgb
    var col_bgr: (UInt8, UInt8, UInt8)  = (UInt8(b & 255),  UInt8(g & 255),  UInt8(r & 255))
    var ccol_bgr: (UInt8, UInt8, UInt8) = (UInt8(cb & 255), UInt8(cg & 255), UInt8(cr & 255))
    return _draw_circles_u8(img, circles, col_bgr, thickness, ccol_bgr, center_radius)

fn draw_circles(
    mut img: Image,
    circles: List[(Int, Int, Int)],
    color_rgb: (UInt8, UInt8, UInt8),
    thickness: Int = 1,
    center_color_rgb: (UInt8, UInt8, UInt8) = (UInt8(0), UInt8(0), UInt8(0)),
    center_radius: Int = 0
) -> Image:
    var (r8, g8, b8) = color_rgb
    var (cr8, cg8, cb8) = center_color_rgb
    var col_bgr: (UInt8, UInt8, UInt8)  = (b8, g8, r8)
    var ccol_bgr: (UInt8, UInt8, UInt8) = (cb8, cg8, cr8)
    return _draw_circles_u8(img, circles, col_bgr, thickness, ccol_bgr, center_radius)



# Draw contours onto an image.
# - img: base image (source; drawing happens on a copy)
# - contours: list of contours (each contour = list of (x, y) integer points)
# - color: (B, G, R) as UInt8 tuple
# - thickness:
#     > 0 : outline mode; stamp a (2*thickness+1) x (2*thickness+1) square at each contour point
#      0 : no-op
#     < 0 : fill the polygon interior using even–odd rule (scanline fill)


# simple insertion sort for List[Int]
fn _sort_int_list(mut xs: List[Int]):
    var i = 1
    while i < len(xs):
        var key = xs[i]
        var j = i - 1
        while j >= 0 and xs[j] > key:
            xs[j + 1] = xs[j]
            j -= 1
        xs[j + 1] = key
        i += 1

fn draw_contours(
    img: Image,
    contours: List[List[(Int, Int)]],
    color: (UInt8, UInt8, UInt8) = (UInt8(0), UInt8(255), UInt8(0)),
    thickness: Int = 1
) -> Image:
    var h = img.height()
    var w = img.width()
    if h == 0 or w == 0 or len(contours) == 0:
        return img.copy()

    var out = img.copy()
    var (b, g, r) = color

    if thickness == 0:
        return out.copy()

    fn _safe_poke(mut im: Image, x: Int, y: Int, bb: UInt8, gg: UInt8, rr: UInt8):
        if _in_bounds(im, x, y):
            _poke(im, x, y, bb, gg, rr)

    # Outline mode
    if thickness > 0:
        var t = thickness
        var ci = 0
        while ci < len(contours):
            var c = contours[ci].copy()
            var pi = 0
            while pi < len(c):
                var px = c[pi][0]
                var py = c[pi][1]
                var dy = -t
                while dy <= t:
                    var dx = -t
                    while dx <= t:
                        _safe_poke(out, px + dx, py + dy, b, g, r)
                        dx += 1
                    dy += 1
                pi += 1
            ci += 1
        return out.copy()

    # Fill mode (even–odd)
    var ci2 = 0
    while ci2 < len(contours):
        var poly = contours[ci2].copy()
        var n = len(poly)
        if n >= 3:
            var min_y = poly[0][1]
            var max_y = poly[0][1]
            var i = 1
            while i < n:
                var py = poly[i][1]
                if py < min_y: min_y = py
                if py > max_y: max_y = py
                i += 1
            if min_y < 0: min_y = 0
            if max_y >= h: max_y = h - 1

            var y = min_y
            while y <= max_y:
                var xs = List[Int]()
                var j = 0
                while j < n:
                    var k = (j + 1) % n
                    var x0 = poly[j][0]
                    var y0 = poly[j][1]
                    var x1 = poly[k][0]
                    var y1 = poly[k][1]
                    if ((y0 > y) != (y1 > y)):
                        var num = (y - y0) * (x1 - x0)
                        var den = (y1 - y0)
                        var x_int = x0 + (num // den)
                        xs.append(x_int)
                    j += 1
                _sort_int_list(xs)
                var xi = 0
                while xi + 1 < len(xs):
                    var x_start = xs[xi]
                    var x_end = xs[xi + 1]
                    if x_start > x_end:
                        var tmp = x_start
                        x_start = x_end
                        x_end = tmp
                    if x_start < 0: x_start = 0
                    if x_end >= w: x_end = w - 1
                    var x = x_start
                    while x <= x_end:
                        _safe_poke(out, x, y, b, g, r)
                        x += 1
                    xi += 2
                y += 1
        ci2 += 1

    return out.copy()


# --- utils ---
@always_inline
fn _imax(a: Int, b: Int) -> Int:
    return a if a > b else b

@always_inline
fn _iabs(x: Int) -> Int:
    return x if x >= 0 else -x

@always_inline
fn _clamp_i(x: Int, lo: Int, hi: Int) -> Int:
    var v = x
    if v < lo: v = lo
    if v > hi: v = hi
    return v

# --- paste src into dst at (x0,y0), HWC/UInt8 ---
fn _paste_hwc_u8(mut dst: Image, src: Image, x0: Int, y0: Int) -> Image:
    var D = dst.ensure_packed_hwc_u8(True)
    var S = src.ensure_packed_hwc_u8(True)
    var Hd = D.height(); var Wd = D.width();  var Cd = D.channels()
    var Hs = S.height(); var Ws = S.width(); var Cs = S.channels()
    if Hd <= 0 or Wd <= 0 or Hs <= 0 or Ws <= 0 or Cd != Cs:
        return D.copy()

    var y = 0
    while y < Hs:
        var yd = y0 + y
        if yd >= 0 and yd < Hd:
            var x = 0
            while x < Ws:
                var xd = x0 + x
                if xd >= 0 and xd < Wd:
                    var ch = 0
                    while ch < Cd:
                        D.set_u8(yd, xd, ch, S.get_u8(y, x, ch))
                        ch = ch + 1
                x = x + 1
        y = y + 1
    return D.copy()

# --- Bresenham line (BGR UInt8) ---
fn _draw_line_bgr_u8(mut img: Image, x0: Int, y0: Int, x1: Int, y1: Int, col: (UInt8, UInt8, UInt8)) -> Image:
    var I = img.ensure_packed_hwc_u8(True)
    var H = I.height(); var W = I.width()
    var (b, g, r) = col

    var dx = _iabs(x1 - x0)
    var sx = 1 if x0 < x1 else -1
    var dy = -_iabs(y1 - y0)
    var sy = 1 if y0 < y1 else -1
    var err = dx + dy

    var cx = x0
    var cy = y0
    while True:
        if 0 <= cx and cx < W and 0 <= cy and cy < H:
            I.set_u8(cy, cx, 0, b)
            I.set_u8(cy, cx, 1, g)
            I.set_u8(cy, cx, 2, r)
        if cx == x1 and cy == y1:
            break
        var e2 = 2 * err
        if e2 >= dy:
            err = err + dy
            cx = cx + sx
        if e2 <= dx:
            err = err + dx
            cy = cy + sy
    return I.copy()

# --- Filled disk (BGR UInt8) ---
fn _draw_disk_bgr_u8(mut img: Image, cx: Int, cy: Int, rad: Int, col: (UInt8, UInt8, UInt8)) -> Image:
    var I = img.ensure_packed_hwc_u8(True)
    var H = I.height(); var W = I.width()
    var (b, g, r) = col
    if rad <= 0: return I.copy()

    var y = cy - rad
    var r2 = rad * rad
    while y <= cy + rad:
        var dy = y - cy
        var x = cx - rad
        while x <= cx + rad:
            var dx = x - cx
            if (dx * dx + dy * dy) <= r2:
                if 0 <= x and x < W and 0 <= y and y < H:
                    I.set_u8(y, x, 0, b)
                    I.set_u8(y, x, 1, g)
                    I.set_u8(y, x, 2, r)
            x = x + 1
        y = y + 1
    return I.copy()

# --- Main: side-by-side matches visualization (kps as (x,y)) ---
fn draw_matches(
    img1: Image,
    kps1: List[(Int, Int)],
    img2: Image,
    kps2: List[(Int, Int)],
    matches: List[(Int, Int)]
) -> Image:
    var a = img1.ensure_packed_hwc_u8(True)
    var b = img2.ensure_packed_hwc_u8(True)

    var h = _imax(a.height(), b.height())
    var w = a.width() + b.width()
    var canvas = Image.new_hwc_u8(h, w, 3, UInt8(0))

    canvas = _paste_hwc_u8(canvas, a, 0, 0)
    canvas = _paste_hwc_u8(canvas, b, a.width(), 0)

    var line_color: (UInt8, UInt8, UInt8) = (UInt8(0),  UInt8(255), UInt8(0))    # green
    var end_color:  (UInt8, UInt8, UInt8) = (UInt8(0),  UInt8(0),   UInt8(255))  # red
    var end_radius = 2

    var n1 = len(kps1)
    var n2 = len(kps2)
    var m  = len(matches)

    var i = 0
    while i < m:
        var (idx1, idx2) = matches[i]
        if (0 <= idx1) and (idx1 < n1) and (0 <= idx2) and (idx2 < n2):
            var (x0, y0) = kps1[idx1]
            var (x1, y1) = kps2[idx2]

            x1 = x1 + a.width()  # shift right image

            # Optional clamps
            x0 = _clamp_i(x0, 0, w - 1); y0 = _clamp_i(y0, 0, h - 1)
            x1 = _clamp_i(x1, 0, w - 1); y1 = _clamp_i(y1, 0, h - 1)

            canvas = _draw_line_bgr_u8(canvas, x0, y0, x1, y1, line_color)
            canvas = _draw_disk_bgr_u8(canvas, x0, y0, end_radius, end_color)
            canvas = _draw_disk_bgr_u8(canvas, x1, y1, end_radius, end_color)
        i = i + 1

    return canvas.copy()
# Helper: drop the 3rd component (score) from keypoints
@always_inline
fn _drop_score(kps3: List[(Int, Int, Float32)]) -> List[(Int, Int)]:
    var out = List[(Int, Int)]()
    var i = 0
    while i < len(kps3):
        var (x, y, _) = kps3[i]
        out.append((x, y))
        i = i + 1
    return out.copy()

# Overload: accept (Int, Int, Float32) keypoints and forward
fn draw_matches(
    img1: Image,
    kps1: List[(Int, Int, Float32)],
    img2: Image,
    kps2: List[(Int, Int, Float32)],
    matches: List[(Int, Int)]
) -> Image:
    var k1 = _drop_score(kps1)
    var k2 = _drop_score(kps2)
    return draw_matches(img1, k1, img2, k2, matches)


fn label_to_color(lbl: Int) -> (UInt8, UInt8, UInt8):
    # Simple deterministic palette based on label id (BGR)
    var lut = [
        (UInt8(  0), UInt8(  0), UInt8(  0)), # 0 black
        (UInt8(255), UInt8(  0), UInt8(  0)), # 1 blue-ish in BGR
        (UInt8(  0), UInt8(255), UInt8(  0)), # 2 green
        (UInt8(  0), UInt8(  0), UInt8(255)), # 3 red
        (UInt8(255), UInt8(255), UInt8(  0)), # 4 cyan
        (UInt8(255), UInt8(  0), UInt8(255)), # 5 magenta
        (UInt8(  0), UInt8(255), UInt8(255)), # 6 yellow
        (UInt8(128), UInt8(128), UInt8(128)), # 7 gray
        (UInt8(255), UInt8(128), UInt8(  0)), # 8 orange
        (UInt8(128), UInt8(  0), UInt8(255))  # 9 purple
    ]
    var n = len(lut)
    var idx = lbl
    if idx < 0: idx = -idx
    idx = idx % n
    return lut[idx]


# اگر keypoints از نوع (Int, Int) هستند:
fn draw_matches(
    img1: Image,
    kps1: List[(Int, Int)],
    img2: Image,
    kps2: List[(Int, Int)],
    matches: List[(Int, Int, Int)]
) -> Image:
    var pairs = List[(Int, Int)]()
    var i = 0
    while i < len(matches):
        var (qi, ti, _) = matches[i]
        pairs.append((qi, ti))
        i = i + 1
    return draw_matches(img1, kps1, img2, kps2, pairs)

# اگر keypoints از نوع (Int, Int, Float32) هستند:
fn draw_matches(
    img1: Image,
    kps1: List[(Int, Int, Float32)],
    img2: Image,
    kps2: List[(Int, Int, Float32)],
    matches: List[(Int, Int, Int)]
) -> Image:
    # drop score از keypoints
    var k1 = List[(Int, Int)]()
    var i = 0
    while i < len(kps1):
        var (x, y, _) = kps1[i]
        k1.append((x, y))
        i = i + 1
    var k2 = List[(Int, Int)]()
    i = 0
    while i < len(kps2):
        var (x, y, _) = kps2[i]
        k2.append((x, y))
        i = i + 1


    var pairs = List[(Int, Int)]()
    i = 0
    while i < len(matches):
        var (qi, ti, _) = matches[i]
        pairs.append((qi, ti))
        i = i + 1

    return draw_matches(img1, k1, img2, k2, pairs)





# --- Arrowed line ------------------------------------------------------------
fn _arrow_head_points(x1: Int, y1: Int, x2: Int, y2: Int, tip_len: Float64) -> ((Int, Int), (Int, Int)):
    var dx = x2 - x1
    var dy = y2 - y1
    var md = iabs(dx)
    if iabs(dy) > md: md = iabs(dy)
    if md == 0:
        return ((x2, y2), (x2, y2))  # degenerate

    # head length w.r.t L∞ norm
    var hl = Int(Float64(md) * tip_len)
    if hl < 2: hl = 2

    # base point of arrow head (on the line, 'hl' pixels from tip)
    var bx = x2 - dx * hl // md
    var by = y2 - dy * hl // md

    # perpendicular vector scaled ~ hl/2
    var px = -dy
    var py = dx
    var mp = iabs(px)
    if iabs(py) > mp: mp = iabs(py)
    if mp == 0: mp = 1
    var side = hl // 2
    var sx = px * side // mp
    var sy = py * side // mp

    # two wing points
    var left  = (bx + sx, by + sy)
    var right = (bx - sx, by - sy)
    return (left, right)

# Core arrowed_line (UInt8 tuple color)
fn arrowed_line(
    mut img: Image,
    x1: Int, y1: Int,
    x2: Int, y2: Int,
    color: (UInt8, UInt8, UInt8),
    thickness: Int = 1,
    tip_len: Float64 = 0.1
) -> Image:
    # draw shaft
    img = line(img, x1, y1, x2, y2, color, thickness)

    # wing points
    var (pL, pR) = _arrow_head_points(x1, y1, x2, y2, tip_len)

    # option A: destructure
    var (lx, ly) = pL
    var (rx, ry) = pR

    img = line(img, x2, y2, lx, ly, color, thickness)
    img = line(img, x2, y2, rx, ry, color, thickness)
    return img.copy()

# Overloads to accept (Int,Int,Int) and List[Int]
fn arrowed_line(mut img: Image, x1: Int, y1: Int, x2: Int, y2: Int,
                color: (Int, Int, Int),
                thickness: Int = 1, tip_len: Float64 = 0.1) -> Image:
    return arrowed_line(img, x1, y1, x2, y2, _to_u8_color_from_int3(color), thickness, tip_len)

fn arrowed_line(mut img: Image, x1: Int, y1: Int, x2: Int, y2: Int,
                color: List[Int],
                thickness: Int = 1, tip_len: Float64 = 0.1) -> Image:
    return arrowed_line(img, x1, y1, x2, y2, _to_u8_color_from_list(color), thickness, tip_len)

# Accept List[UInt8] and forward to the List[Int] overload
fn arrowed_line(mut img: Image,x1: Int, y1: Int,x2: Int, y2: Int,color: List[UInt8],thickness: Int = 1,tip_len: Float64 = 0.1) -> Image:
    var tmp = List[Int]()
    var i = 0
    while i < len(color):
        tmp.append(Int(color[i]))
        i += 1
    return arrowed_line(img, x1, y1, x2, y2, tmp, thickness, tip_len)
# Base implementation: uses UInt8 tuple color and supports optional 'aa'
# --- 5x7 glyphs (ROW-major): each of 7 rows has 5 bits (bit 0 = leftmost col) ---
# 5x7 ROW-major glyphs.
# Each return is 7 rows; in each row, 5 LSBits encode pixels (bit 0 = leftmost column).



# --- Fill a solid rectangle [x0..x1]×[y0..y1] using horizontal lines (in-place) ---
@always_inline
fn _fill_rect(mut img: Image,
              x0: Int, y0: Int, x1: Int, y1: Int,
              col: (UInt8, UInt8, UInt8)) -> Image:
    var xx0 = x0; var yy0 = y0; var xx1 = x1; var yy1 = y1
    if xx0 > xx1:
        var t = xx0; xx0 = xx1; xx1 = t
    if yy0 > yy1:
        var t2 = yy0; yy0 = yy1; yy1 = t2

    # clamp to image bounds
    var W = img.width(); var H = img.height()
    if xx1 < 0 or yy1 < 0 or xx0 >= W or yy0 >= H:
        return img.copy()
    if xx0 < 0: xx0 = 0
    if yy0 < 0: yy0 = 0
    if xx1 >= W: xx1 = W - 1
    if yy1 >= H: yy1 = H - 1

    var y = yy0
    while y <= yy1:
        img = line(img, xx0, y, xx1, y, col, 1)
        y += 1
    return img.copy()


# --- One font "bit" as a solid block ---
@always_inline
fn _plot_block(mut img: Image, x: Int, y: Int, block: Int,
               col: (UInt8, UInt8, UInt8)) -> Image:
    var b = block
    if b < 2: b = 2
    return _fill_rect(img, x, y, x + b - 1, y + b - 1, col)


# --- Public API: draw text (supports T/E/S/O/N/R/I; extend as needed) ---
fn put_text(
    mut img: Image,
    text: String,
    x: Int,
    y: Int,
    font: Int = 1,                          # kept for API compatibility
    size: Float64 = 3.0,                    # strong default scale
    color: (UInt8, UInt8, UInt8) = (UInt8(255), UInt8(255), UInt8(255)),
    thickness: Int = 2,
    aa: Bool = False
) raises -> Image:
    img = img.ensure_packed_hwc_u8(True)

    # scale → block size per font pixel
    var block = Int(size * 6.0) + (thickness - 1)
    if block < 3: block = 3
    var gap = block // 2                    # letter spacing

    var cursor_x = x
    var cursor_y = y

    var i = 0
    var n = text.__len__()
    while i < n:
        # Each index returns a StringSlice (may raise) → row-major glyph
        var chs = text[i]
        var rows = _glyph5x7_rowmajor(chs)   # 7 rows

        # draw rows×cols (7×5)
        var row = 0
        while row < 7:
            var row_bits = rows[row]
            var col = 0
            while col < 5:
                if (Int(row_bits) >> col) & 1 == 1:
                    var px = cursor_x + col * block
                    var py = cursor_y + row * block
                    img = _plot_block(img, px, py, block, color)
                col += 1
            row += 1

        # advance to next character
        cursor_x += 5 * block + gap
        i += 1

    return img.copy()


# Overload: color as (Int, Int, Int) → convert to UInt8 tuple
fn put_text(
    mut img: Image,
    text: String,
    x: Int,
    y: Int,
    font: Int,
    size: Float64,
    color: (Int, Int, Int),
    thickness: Int,
    aa: Bool = False
)raises -> Image:
    return put_text(img, text, x, y, font, size, _to_u8_color_from_int3(color), thickness, aa)

# Convenience wrapper with defaults for (Int, Int, Int) color
fn put_text(
    mut img: Image,
    text: String,
    x: Int,
    y: Int,
    color: (Int, Int, Int)
)raises -> Image:
    return put_text(img, text, x, y, 1, 1.0, _to_u8_color_from_int3(color), 1, False)

# Overload: color as List[Int] → convert to UInt8 tuple
fn put_text(
    mut img: Image,
    text: String,
    x: Int,
    y: Int,
    font: Int,
    size: Float64,
    color: List[Int],
    thickness: Int,
    aa: Bool = False
)raises -> Image:
    return put_text(img, text, x, y, font, size, _to_u8_color_from_list(color), thickness, aa)

# Convenience wrapper with defaults for List[Int] color
fn put_text(
    mut img: Image,
    text: String,
    x: Int,
    y: Int,
    color: List[Int]
)raises -> Image:
    return put_text(img, text, x, y, 1, 1.0, _to_u8_color_from_list(color), 1, False)

# Overload: color as List[UInt8] → forward to List[Int]
fn put_text(
    mut img: Image,
    text: String,
    x: Int,
    y: Int,
    font: Int,
    size: Float64,
    color: List[UInt8],
    thickness: Int,
    aa: Bool = False
)raises -> Image:
    var tmp = List[Int]()
    var i = 0
    while i < len(color):
        tmp.append(Int(color[i]))
        i += 1
    return put_text(img, text, x, y, font, size, tmp, thickness, aa)

# Convenience wrapper with defaults for List[UInt8] color
fn put_text(
    mut img: Image,
    text: String,
    x: Int,
    y: Int,
    color: List[UInt8]
)raises -> Image:
    var tmp = List[Int]()
    var i = 0
    while i < len(color):
        tmp.append(Int(color[i]))
        i += 1
    return put_text(img, text, x, y, 1, 1.0, tmp, 1, False)


# --- Compat: Int32Mat --------------------------------------------------------
# Usage: var pts = Int32Mat([[600,40],[760,40],[720,140],[580,140]])
fn Int32Mat(a: List[List[Int]]) -> List[(Int, Int)]:
    var out = List[(Int, Int)]()
    var i = 0
    while i < len(a):
        var row = a[i].copy()
        var x = 0
        var y = 0
        if len(row) > 0: x = row[0]
        if len(row) > 1: y = row[1]
        out.append((x, y))
        i += 1
    return out.copy()

# --- Compat: fonts -----------------------------------------------------------
fn FONT_SIMPLEX() -> Int:
    return 1

# --- Color adapters (once per file) ------------------------------------------

fn _to_u8_color_from_int3(c: (Int, Int, Int)) -> (UInt8, UInt8, UInt8):
    var (r, g, b) = c
    return (
        UInt8(_clamp_i(r, 0, 255)),
        UInt8(_clamp_i(g, 0, 255)),
        UInt8(_clamp_i(b, 0, 255))
    )

fn _to_u8_color_from_list(c: List[Int]) -> (UInt8, UInt8, UInt8):
    var b = 0; var g = 0; var r = 0
    if len(c) > 0: b = c[0]
    if len(c) > 1: g = c[1]
    if len(c) > 2: r = c[2]
    return (UInt8(_clamp_i(b, 0, 255)), UInt8(_clamp_i(g, 0, 255)), UInt8(_clamp_i(r, 0, 255)))

fn _to_u8_color_from_list_u8(c: List[UInt8]) -> (UInt8, UInt8, UInt8):
    var b: UInt8 = 0; var g: UInt8 = 0; var r: UInt8 = 0
    if len(c) > 0: b = c[0]
    if len(c) > 1: g = c[1]
    if len(c) > 2: r = c[2]
    return (b, g, r)


# Utility: map label image to RGB for visualization
# Background (0) -> black. Other labels -> fixed pseudo-colors.
fn label_to_color(labels: List[List[Int]], num_labels: Int) -> List[List[List[UInt8]]]:
    var h = len(labels)
    if h == 0: return List[List[List[UInt8]]]()
    var w = len(labels[0])

    # simple deterministic palette (avoid 0)
    fn _color_for(lbl: Int) -> (UInt8, UInt8, UInt8):
        if lbl == 0: return (0, 0, 0)
        # hash-like mapping
        var v = UInt32(lbl) * 2654435761  # Knuth
        var r = UInt8((v >> 16) & 0xFF)
        var g = UInt8((v >> 8) & 0xFF)
        var b = UInt8(v & 0xFF)
        # avoid too dark colors
        if r < 32: r = 32
        if g < 32: g = 32
        if b < 32: b = 32
        return (r, g, b)

    var out = List[List[List[UInt8]]]()
    out.reserve(h)
    var y = 0
    while y < h:
        var row = List[List[UInt8]]()
        row.reserve(w)
        var x = 0
        while x < w:
            var (r, g, b) = _color_for(labels[y][x])
            var px = List[UInt8]()
            px.append(r); px.append(g); px.append(b)  # RGB
            row.append(px)
            x += 1
        out.append(row)
        y += 1
    return out.copy()

# Deterministic pseudo-color for label ids
fn _color_for(lbl: Int) -> (UInt8, UInt8, UInt8):
    if lbl == 0: return (UInt8(0), UInt8(0), UInt8(0))  # background: black
    var v = UInt32(lbl) * 2654435761  # Knuth multiplicative hash
    var r = UInt8((v >> 16) & 0xFF)
    var g = UInt8((v >> 8) & 0xFF)
    var b = UInt8(v & 0xFF)
    if r < 32: r = 32
    if g < 32: g = 32
    if b < 32: b = 32
    return (r, g, b)

# Labels (H x W, Int) -> RGB Image (UInt8)

fn label_to_color_image(labels: List[List[Int]], num_labels: Int) -> Image:
    var h = len(labels)
    if h == 0:
        # minimal valid image using Image factory
        return Image.new_hwc_u8(1, 1, 3, UInt8(0))

    var w = len(labels[0])

    # create an HxWx3 UInt8 image (all zeros initially)
    var img = Image.new_hwc_u8(h, w, 3, UInt8(0))

    var y = 0
    while y < h:
        var x = 0
        while x < w:
            var (r, g, b) = _color_for(labels[y][x])
            img.set_u8(y, x, 0, r)
            img.set_u8(y, x, 1, g)
            img.set_u8(y, x, 2, b)
            x += 1
        y += 1

    return img.copy()

fn _draw_point(mut img: Image, x: Int, y: Int, color: (UInt8, UInt8, UInt8)):
    # Guard: skip if out of bounds
    if not _in_bounds(img, x, y):
        return

    var (b, g, r) = color   # assuming color tuple is (B, G, R)
    _poke(img, x, y, b, g, r)


fn _draw_thick_line(
    mut img: Image,
    x1: Int,
    y1: Int,
    x2: Int,
    y2: Int,
    color: (UInt8, UInt8, UInt8),
    thickness: Int
):
    if thickness <= 1:
        _draw_line(img, x1, y1, x2, y2, color)
        return

    var dx = x2 - x1
    var dy = y2 - y1
    var len = sqrt(Float64(dx * dx + dy * dy))
    if len == 0.0:
        _draw_point(img, x1, y1, color)
        return

    # Unit normal vector
    var nx = -dy / len
    var ny = dx / len
    var r = thickness // 2
    var i = -r
    while i <= r:
        var ox = _round_to_int(Float64(i) * nx)
        var oy = _round_to_int(Float64(i) * ny)
        _draw_line(img, x1 + ox, y1 + oy, x2 + ox, y2 + oy, color)
        i += 1

fn _draw_line(mut img: Image, x1: Int, y1: Int, x2: Int, y2: Int, color: (UInt8,UInt8,UInt8)):
    var dx = abs(x2 - x1)
    var sx = 1 if x1 < x2 else -1
    var dy = -abs(y2 - y1)
    var sy = 1 if y1 < y2 else -1
    var err = dx + dy
    var x = x1
    var y = y1
    while True:
        _draw_point(img, x, y, color)
        if x == x2 and y == y2: break
        var e2 = 2 * err
        if e2 >= dy:
            err = err + dy
            x = x + sx
        if e2 <= dx:
            err = err + dx
            y = y + sy

fn draw_lines_p(mut img: Image, lines: List[(Int,Int,Int,Int)], color: (UInt8,UInt8,UInt8)) -> Image:
    var i = 0
    while i < len(lines):
        var (x1, y1, x2, y2) = lines[i]
        _draw_line(img, x1, y1, x2, y2, color)
        i += 1
    return img.copy()

fn draw_lines_p(
    mut img: Image,
    lines: List[(Int, Int, Int, Int)],
    color: (UInt8, UInt8, UInt8),
    thickness: Int = 1
) -> Image:
    var i = 0
    while i < len(lines):
        var (x1, y1, x2, y2) = lines[i]
        _draw_thick_line(img, x1, y1, x2, y2, color, thickness)
        i += 1
    return img.copy()


# --- internal blit (u8 HWC only) ---
fn _blit(mut dst: Image, dx: Int, dy: Int, src: Image):
    var h = src.height(); var w = src.width()
    var yy = 0
    while yy < h:
        var xx = 0
        while xx < w:
            # Read pixel from src
            var (r, g, b) = _peek(src, xx, yy)   # assumes _peek exists in this file
            # Write into dst at offset
            _poke(dst, dx + xx, dy + yy, r, g, b) # assumes _poke exists in this file
            xx = xx + 1
        yy = yy + 1

# --- New overload: OpenCV-style draw_matches ---
# As in: draw_matches(img1, kp1, img2, kp2, matches)


fn bgr_u8(b: Int, g: Int, r: Int) -> (UInt8, UInt8, UInt8):
    return (UInt8(b), UInt8(g), UInt8(r))


# floor for Float64 → Int (works without math)
fn _floor_to_int(x: Float64) -> Int:
    var i = Int(x)                 # truncates toward 0
    if Float64(i) > x:             # if we truncated a negative with fractional part
        return i - 1
    return i

# round half-up to Int (no math.floor)
fn _round_to_int(x: Float64) -> Int:
    if x >= 0.0:
        return _floor_to_int(x + 0.5)
    else:
        return -_floor_to_int((-x) + 0.5)
