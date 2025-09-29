# MIT License
# SPDX-License-Identifier: MIT
# Project: momijo.vision | File: src/momijo/vision/transforms/draw.mojo

from momijo.vision.image import Image
from momijo.vision.dtypes import DType  # for explicit checks if needed
from momijo.vision.transforms.array import full
from momijo.vision.transforms.features import Keypoint, keypoint_xy, len_keypoints
 
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
    return out
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
    return img

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
    # Writes color, broadcasting if needed
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
        return src

    var x_min = _max(_min(x0, x1), 0)
    var x_max = _min(_max(x0, x1), w - 1)
    var y_min = _max(_min(y0, y1), 0)
    var y_max = _min(_max(y0, y1), h - 1)

    if x_min > x_max or y_min > y_max:
        return src

    # Clone first, then ensure packed HWC/u8
    var dst = src.ensure_packed_hwc_u8(True).clone()
 

    if thickness < 0:
        var y = y_min
        while y <= y_max:
            var x = x_min
            while x <= x_max:
                _poke(dst, x, y, b, g, r)
                x += 1
            y += 1
        return dst

    if thickness == 0:
        return dst

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
 
    return dst


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




fn circle(mut img: Image, cx: Int, cy: Int, r: Int,
          color: (UInt8, UInt8, UInt8), thickness: Int = 1) -> Image:
    img = img.ensure_packed_hwc_u8(True)
    if r <= 0:
        return img
    if thickness < 0:
        _draw_disk(img, cx, cy, r, color)
        return img

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
    return img
# --- fill_poly main implementation -------------------------------------------
fn fill_poly(mut img: Image, pts: List[(Int, Int)], color: (UInt8, UInt8, UInt8)) -> Image:
    img = img.ensure_packed_hwc_u8(True)

    var n = len(pts)
    if n < 3:
        return img

    # Compute min_y, max_y
    var min_y = img.height()
    var max_y = 0
    var i = 0
    while i < n:
        var (px, py) = pts[i]
        if py < min_y: min_y = py
        if py > max_y: max_y = py
        i += 1

    if img.height() <= 0 or img.width() <= 0:
        return img

    min_y = _clamp(min_y, 0, img.height() - 1)
    max_y = _clamp(max_y, 0, img.height() - 1)

    var y = min_y
    while y <= max_y:
        # Build intersection list
        var xs = List[Int]()
        var j = 0
        while j < n:
            var k = (j + 1) % n
            var (x0, y0) = pts[j]
            var (x1, y1) = pts[k]
            if (y0 <= y and y1 > y) or (y1 <= y and y0 > y):
                var x = x0 + (y - y0) * (x1 - x0) // (y1 - y0)
                xs.append(x)
            j += 1
        # Sort xs (insertion sort for small lists)
        var a = 1
        while a < len(xs):
            var key = xs[a]
            var b = a - 1
            while b >= 0 and xs[b] > key:
                xs[b + 1] = xs[b]
                b -= 1
            xs[b + 1] = key
            a += 1
        # Fill pairs
        var p = 0
        while p + 1 < len(xs):
            var x_start = _clamp(xs[p], 0, img.width() - 1)
            var x_end   = _clamp(xs[p + 1], 0, img.width() - 1)
            img = line(img, x_start, y, x_end, y, color, 1)
            p += 2
        y += 1
    return img

# --- fill_poly overloads -----------------------------------------------------

# Overload for (Int, Int, Int) → converted to UInt8 tuple
fn fill_poly(mut img: Image, pts: List[(Int, Int)], color: (Int, Int, Int)) -> Image:
    return fill_poly(img, pts, _to_u8_color_from_int3(color))

# Overload for List[Int] → converted to UInt8 tuple
fn fill_poly(mut img: Image, pts: List[(Int, Int)], color: List[Int]) -> Image:
    return fill_poly(img, pts, _to_u8_color_from_list(color))


fn draw_circles(
    mut img: Image,
    circles: List[(Int, Int, Int)],
    color: (UInt8, UInt8, UInt8),
    thickness: Int = 1,
    center_color: (UInt8, UInt8, UInt8) = (0, 0, 0),
    center_radius: Int = 0
) -> Image:
    img = img.ensure_packed_hwc_u8(True)
    var i = 0
    while i < len(circles):
        var (cx, cy, r) = circles[i]
        # main circle
        img = circle(img, cx, cy, r, color, thickness)
        # optional center dot
        if center_radius > 0:
            img = circle(img, cx, cy, center_radius, center_color, -1)  # filled
        i += 1
    return img

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

    return img


 # Int-color wrapper that converts to UInt8 BGR and calls the u8 version.
# Input colors are assumed RGB; convert to internal BGR ordering.
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
    var col_bgr: (UInt8, UInt8, UInt8)  = (UInt8(b),  UInt8(g),  UInt8(r))
    var ccol_bgr: (UInt8, UInt8, UInt8) = (UInt8(cb), UInt8(cg), UInt8(cr))
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
        return img

    var out = img.copy()
    var (b, g, r) = color

    if thickness == 0:
        return out

    fn _safe_poke(mut im: Image, x: Int, y: Int, bb: UInt8, gg: UInt8, rr: UInt8):
        if _in_bounds(im, x, y):
            _poke(im, x, y, bb, gg, rr)

    # Outline mode
    if thickness > 0:
        var t = thickness
        var ci = 0
        while ci < len(contours):
            var c = contours[ci]
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
        return out

    # Fill mode (even–odd)
    var ci2 = 0
    while ci2 < len(contours):
        var poly = contours[ci2]
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

    return out



fn draw_matches(mut img: Image, pts1: List[(Int, Int)], pts2: List[(Int, Int)]) -> Image:
    img = img.ensure_packed_hwc_u8(True)
    # Draw lines between corresponding points and small disks at endpoints
    var n1 = len(pts1)
    var n2 = len(pts2)
    var n = n1 if n1 < n2 else n2
    var line_color: (UInt8, UInt8, UInt8) = (UInt8(0), UInt8(255), UInt8(0))
    var end_color:  (UInt8, UInt8, UInt8) = (UInt8(0), UInt8(0), UInt8(255))
    var i = 0
    while i < n:
        var (x0, y0) = pts1[i]
        var (x1, y1) = pts2[i]
        img = line(img, x0, y0, x1, y1, line_color, 1)
        _draw_disk(img, x0, y0, 2, end_color)
        _draw_disk(img, x1, y1, 2, end_color)
        i += 1
    return img

 


fn draw_matches(
    img1: Image,
    kps1: List[(Int, Int)],
    img2: Image,
    kps2: List[(Int, Int)],
    matches: List[(Int, Int)]
) -> Image:
    # Ensure format
    var a = img1.ensure_packed_hwc_u8(True)
    var b = img2.ensure_packed_hwc_u8(True)

    # Build side-by-side canvas
    var h = a.height() if a.height() > b.height() else b.height()
    var w = a.width() + b.width()
    var canvas = full(h, w, 3, UInt8(0))

    # Paste the two images (expects you have a paste utility)
    canvas = paste(canvas, a, 0, 0)
    canvas = paste(canvas, b, a.width(), 0)

    # Colors
    var line_color: (UInt8, UInt8, UInt8) = (UInt8(0), UInt8(255), UInt8(0))
    var end_color:  (UInt8, UInt8, UInt8) = (UInt8(0), UInt8(0), UInt8(255))

    # Draw each match
    var m = len(matches)
    var n1 = len(kps1)
    var n2 = len(kps2)

    var i = 0
    while i < m:
        var (idx1, idx2) = matches[i]
        if (0 <= idx1) and (idx1 < n1) and (0 <= idx2) and (idx2 < n2):
            var (x0, y0) = kps1[idx1]
            var (x1, y1) = kps2[idx2]
            # Offset x for the right image
            x1 = x1 + a.width()

            canvas = line(canvas, x0, y0, x1, y1, line_color, 1)
            _draw_disk(canvas, x0, y0, 2, end_color)
            _draw_disk(canvas, x1, y1, 2, end_color)
        i += 1

    return canvas


fn draw_matches(
    img1: Image,
    kps1: Int,
    img2: Image,
    kps2: Int,
    matches: List[(Int, Int)]
) -> Image:
    # Ensure format
    var a = img1.ensure_packed_hwc_u8(True)
    var b = img2.ensure_packed_hwc_u8(True)

    # Build side-by-side canvas
    var h = a.height() if a.height() > b.height() else b.height()
    var w = a.width() + b.width()
    var canvas = full(h, w, 3, UInt8(0))

    # Paste the two images (expects you have a paste utility)
    canvas = paste(canvas, a, 0, 0)
    canvas = paste(canvas, b, a.width(), 0)

    # Colors
    var line_color: (UInt8, UInt8, UInt8) = (UInt8(0), UInt8(255), UInt8(0))
    var end_color:  (UInt8, UInt8, UInt8) = (UInt8(0), UInt8(0), UInt8(255))

    # Draw each match
    var m = len(matches)
    var n1 = len(kps1)
    var n2 = len(kps2)

    var i = 0
    while i < m:
        var (idx1, idx2) = matches[i]
        if (0 <= idx1) and (idx1 < n1) and (0 <= idx2) and (idx2 < n2):
            var (x0, y0) = kps1[idx1]
            var (x1, y1) = kps2[idx2]
            # Offset x for the right image
            x1 = x1 + a.width()

            canvas = line(canvas, x0, y0, x1, y1, line_color, 1)
            _draw_disk(canvas, x0, y0, 2, end_color)
            _draw_disk(canvas, x1, y1, 2, end_color)
        i += 1

    return canvas

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








# line overloads
fn line(mut img: Image, x1: Int, y1: Int, x2: Int, y2: Int, color: (Int, Int, Int), thickness: Int = 1) -> Image:
    return line(img, x1, y1, x2, y2, _to_u8_color_from_int3(color), thickness)

fn line(mut img: Image, x1: Int, y1: Int, x2: Int, y2: Int, color: List[Int], thickness: Int = 1) -> Image:
    return line(img, x1, y1, x2, y2, _to_u8_color_from_list(color), thickness)

# circle overloads
fn circle(mut img: Image, cx: Int, cy: Int, r: Int, color: (Int, Int, Int), thickness: Int = 1) -> Image:
    return circle(img, cx, cy, r, _to_u8_color_from_int3(color), thickness)

fn circle(mut img: Image, cx: Int, cy: Int, r: Int, color: List[Int], thickness: Int = 1) -> Image:
    return circle(img, cx, cy, r, _to_u8_color_from_list(color), thickness)


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
    return img

# Overloads to accept (Int,Int,Int) and List[Int]
fn arrowed_line(mut img: Image, x1: Int, y1: Int, x2: Int, y2: Int,
                color: (Int, Int, Int),
                thickness: Int = 1, tip_len: Float64 = 0.1) -> Image:
    return arrowed_line(img, x1, y1, x2, y2, _to_u8_color_from_int3(color), thickness, tip_len)

fn arrowed_line(mut img: Image, x1: Int, y1: Int, x2: Int, y2: Int,
                color: List[Int],
                thickness: Int = 1, tip_len: Float64 = 0.1) -> Image:
    return arrowed_line(img, x1, y1, x2, y2, _to_u8_color_from_list(color), thickness, tip_len)


# Base implementation: uses UInt8 tuple color and supports optional 'aa'
fn put_text(
    mut img: Image,
    text: String,
    x: Int,
    y: Int,
    font: Int = 1,
    size: Float64 = 1.0,
    color: (UInt8, UInt8, UInt8) = (UInt8(255), UInt8(255), UInt8(255)),
    thickness: Int = 1,
    aa: Bool = False
) -> Image:
    img = img.ensure_packed_hwc_u8(True)
    # Minimal placeholder: underline and a small marker at start.
    var yb = y
    var xe = x + Int(Float64(text.__len__()) * 6.0 * size)
    img = line(img, x, yb, xe, yb, color, thickness)
    _draw_disk(img, x, y, 2, color)
    return img

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
) -> Image:
    return put_text(img, text, x, y, font, size, _to_u8_color_from_int3(color), thickness, aa)

# Convenience wrapper with defaults for (Int, Int, Int) color
fn put_text(
    mut img: Image,
    text: String,
    x: Int,
    y: Int,
    color: (Int, Int, Int)
) -> Image:
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
) -> Image:
    return put_text(img, text, x, y, font, size, _to_u8_color_from_list(color), thickness, aa)

# Convenience wrapper with defaults for List[Int] color
fn put_text(
    mut img: Image,
    text: String,
    x: Int,
    y: Int,
    color: List[Int]
) -> Image:
    return put_text(img, text, x, y, 1, 1.0, _to_u8_color_from_list(color), 1, False)

# --- Compat: Int32Mat --------------------------------------------------------
# Usage: var pts = Int32Mat([[600,40],[760,40],[720,140],[580,140]])
fn Int32Mat(a: List[List[Int]]) -> List[(Int, Int)]:
    var out = List[(Int, Int)]()
    var i = 0
    while i < len(a):
        var row = a[i]
        var x = 0
        var y = 0
        if len(row) > 0: x = row[0]
        if len(row) > 1: y = row[1]
        out.append((x, y))
        i += 1
    return out

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
    return out
 
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

    return img

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
    return img

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
    return img
 

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
# matches: list of (query_idx, train_idx, distance)

fn draw_matches(img1: Image,
                kp1: List[Keypoint],
                img2: Image,
                kp2: List[Keypoint],
                matches: List[Tuple[Int, Int, Int]],
                thickness: Int = 1) -> Image:

    var h1 = img1.height()
    var w1 = img1.width()
    var h2 = img2.height()
    var w2 = img2.width()
    var H = if h1 > h2 then h1 else h2
    var W = w1 + w2

    # Create a black canvas of proper size
    var canvas = full(H, W, 3, UInt8(0))
    _blit(canvas, 0, 0, img1)
    _blit(canvas, w1, 0, img2)

    var i = 0
    while i < len(matches):
        var (qi, ti, _) = matches[i]
        # Defensive index check
        if qi >= 0 and qi < len(kp1) and ti >= 0 and ti < len(kp2):
            var (x1, y1) = keypoint_xy(kp1, qi)
            var (x2, y2) = keypoint_xy(kp2, ti)
            var sx2 = x2 + w1

            # Draw keypoints
            _draw_circle(canvas, x1, y1, 3, (UInt8(0), UInt8(255), UInt8(0)))
            _draw_circle(canvas, sx2, y2, 3, (UInt8(0), UInt8(255), UInt8(0)))

            # Draw line connecting keypoints
            _draw_thick_line(canvas, x1, y1, sx2, y2, (UInt8(255), UInt8(0), UInt8(0)), thickness)
        i = i + 1

    return canvas

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