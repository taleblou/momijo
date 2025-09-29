# MIT License
# SPDX-License-Identifier: MIT
# Project: momijo.vision | File: src/momijo/vision/transforms/hough.mojo

from momijo.vision.image import Image
from collections.list import List


import math 

fn abs_f64(x: Float64) -> Float64:
    if x < 0.0:
        return -x
    else:
        return x


# -------------------------- Pixel access helpers --------------------------

fn _euclid2(ax: Int, ay: Int, bx: Int, by: Int) -> Int:
    var dx = ax - bx
    var dy = ay - by
    return dx * dx + dy * dy

fn _in_bounds(img: Image, x: Int, y: Int) -> Bool:
    return (x >= 0 and x < img.width() and y >= 0 and y < img.height())

fn _gray_at(img: Image, x: Int, y: Int) -> UInt8:
    if not _in_bounds(img, x, y):
        return UInt8(0)
    var base = img.ensure_packed_hwc_u8(True)
    var c = base.channels()
    if c >= 3:
        var b = base.get_u8(y, x, 0)
        var g = base.get_u8(y, x, 1)
        var r = base.get_u8(y, x, 2)
        var m = b
        if g > m:
            m = g
        if r > m:
            m = r
        return m
    else:
        return base.get_u8(y, x, 0)

fn _poke(mut img: Image, x: Int, y: Int, b: UInt8, g: UInt8, r: UInt8):
    if not _in_bounds(img, x, y):
        return
    var base = img.ensure_packed_hwc_u8(True)
    var c = base.channels()
    if c >= 3:
        base.set_u8(y, x, 0, b)
        base.set_u8(y, x, 1, g)
        base.set_u8(y, x, 2, r)
    else:
        base.set_u8(y, x, 0, r)

# -------------------------- Hough Lines (θ–ρ) -----------------------------

fn _clip_to_image(img: Image, rho: Float64, ct: Float64, st: Float64) -> (Bool, Int, Int, Int, Int):
    # Intersect x=0, x=w-1, y=0, y=h-1 with x*ct + y*st = rho
    var w = img.width()
    var h = img.height()
    var pts = List[(Float64, Float64)]()

    if abs_f64(st) > 1e-8:
        var y0 = rho / st
        if y0 >= 0.0 and y0 <= Float64(h - 1):
            pts.append((0.0, y0))
        var yw = (rho - Float64(w - 1) * ct) / st
        if yw >= 0.0 and yw <= Float64(h - 1):
            pts.append((Float64(w - 1), yw))
    if abs_f64(ct) > 1e-8:
        var x0 = rho / ct
        if x0 >= 0.0 and x0 <= Float64(w - 1):
            pts.append((x0, 0.0))
        var xh = (rho - Float64(h - 1) * st) / ct
        if xh >= 0.0 and xh <= Float64(w - 1):
            pts.append((xh, Float64(h - 1)))

    if len(pts) < 2:
        return (False, 0, 0, 0, 0)

    # Pick the two farthest points on the rectangle
    var best_d2 = -1.0
    var a = 0
    var b = 1
    var i = 0
    while i < len(pts):
        var j = i + 1
        while j < len(pts):
            var dx = pts[i][0] - pts[j][0]
            var dy = pts[i][1] - pts[j][1]
            var d2 = dx * dx + dy * dy
            if d2 > best_d2:
                best_d2 = d2
                a = i
                b = j
            j += 1
        i += 1

    var x1 = Int(math.floor(pts[a][0] + 0.5))
    var y1 = Int(math.floor(pts[a][1] + 0.5))
    var x2 = Int(math.floor(pts[b][0] + 0.5))
    var y2 = Int(math.floor(pts[b][1] + 0.5))
    return (True, x1, y1, x2, y2)

# Standard Hough lines: returns clipped line endpoints for each accumulator peak
fn hough_lines(img: Image, rho: Float64, theta: Float64, thresh: Int) -> List[(Int, Int, Int, Int)]:
    var h = img.height()
    var w = img.width()
    var diag = Int(math.ceil(math.sqrt(Float64(w * w + h * h))))
    var rr = rho
    var th = theta
    if rr <= 0.0:
        rr = 1.0
    if th <= 0.0:
        th = math.pi / 180.0
    var n_rho = Int(math.floor((2.0 * Float64(diag)) / rr)) + 1
    var n_theta = Int(math.floor(math.pi / th)) + 1
    if n_rho <= 0 or n_theta <= 0:
        return List[(Int, Int, Int, Int)]()

    # cos/sin LUT
    var cos_tbl = List[Float64]()
    var sin_tbl = List[Float64]()
    var ti = 0
    while ti < n_theta:
        var ang = Float64(ti) * th
        cos_tbl.append(math.cos(ang))
        sin_tbl.append(math.sin(ang))
        ti += 1

    # accumulator
    var acc = List[Int]()
    var total = n_rho * n_theta
    var k = 0
    while k < total:
        acc.append(0)
        k += 1

    # vote
    var y = 0
    while y < h:
        var x = 0
        while x < w:
            if _gray_at(img, x, y) > UInt8(0):
                var tt = 0
                while tt < n_theta:
                    var r = Float64(x) * cos_tbl[tt] + Float64(y) * sin_tbl[tt]
                    var r_idx = Int(math.floor((r + Float64(diag)) / rr + 0.5))
                    if r_idx >= 0 and r_idx < n_rho:
                        var idx = r_idx * n_theta + tt
                        acc[idx] = acc[idx] + 1
                    tt += 1
            x += 1
        y += 1

    # collect peaks with 3x3 NMS
    var lines = List[(Int, Int, Int, Int)]()
    var r_idx = 1
    while r_idx < n_rho - 1:
        var t_idx = 1
        while t_idx < n_theta - 1:
            var v = acc[r_idx * n_theta + t_idx]
            if v >= thresh:
                var is_max = True
                var dr = -1
                while dr <= 1:
                    var dt = -1
                    while dt <= 1:
                        if not (dr == 0 and dt == 0):
                            var vv = acc[(r_idx + dr) * n_theta + (t_idx + dt)]
                            if vv > v:
                                is_max = False
                        dt += 1
                    dr += 1
                if is_max:
                    var ct = cos_tbl[t_idx]
                    var st = sin_tbl[t_idx]
                    var r_val = Float64(r_idx) * rr - Float64(diag)
                    var ok = False
                    var x1 = 0
                    var y1 = 0
                    var x2 = 0
                    var y2 = 0
                    (ok, x1, y1, x2, y2) = _clip_to_image(img, r_val, ct, st)
                    if ok:
                        lines.append((x1, y1, x2, y2))
            t_idx += 1
        r_idx += 1

    return lines

# Probabilistic-like wrapper: splits standard Hough lines into segments using a gap/length policy
fn hough_lines_p(
    img: Image,
    rho: Float64,
    theta: Float64,
    threshold: Int,
    min_len: Int = 0,
    max_gap: Int = 0
) -> List[(Int, Int, Int, Int)]:
    var raw_lines = hough_lines(img, rho, theta, threshold)
    var output = List[(Int, Int, Int, Int)]()

    var i = 0
    while i < len(raw_lines):
        var x0 = raw_lines[i][0]
        var y0 = raw_lines[i][1]
        var x1 = raw_lines[i][2]
        var y1 = raw_lines[i][3]

        var dx = x1 - x0
        var dy = y1 - y0
        var length = math.sqrt(Float64(dx * dx + dy * dy))
        if length < Float64(min_len):
            i += 1
            continue

        var n_points = Int(math.ceil(length))
        if n_points < 2:
            i += 1
            continue

        var step_x = Float64(dx) / Float64(n_points)
        var step_y = Float64(dy) / Float64(n_points)

        var prev_x = Int(math.floor(Float64(x0)))
        var prev_y = Int(math.floor(Float64(y0)))
        var seg_start_x = prev_x
        var seg_start_y = prev_y
        var gap = 0

        var j = 1
        while j <= n_points:
            var fx = Float64(x0) + step_x * Float64(j)
            var fy = Float64(y0) + step_y * Float64(j)
            var cx = Int(math.floor(fx + 0.5))
            var cy = Int(math.floor(fy + 0.5))

            if cx == prev_x and cy == prev_y:
                j += 1
                continue

            if _gray_at(img, cx, cy) > UInt8(0):
                if gap > max_gap:
                    var dx2 = prev_x - seg_start_x
                    var dy2 = prev_y - seg_start_y
                    var seg_len = math.sqrt(Float64(dx2 * dx2 + dy2 * dy2))
                    if seg_len >= Float64(min_len):
                        output.append((seg_start_x, seg_start_y, prev_x, prev_y))
                    seg_start_x = cx
                    seg_start_y = cy
                gap = 0
            else:
                gap += 1

            prev_x = cx
            prev_y = cy
            j += 1

        var dx3 = prev_x - seg_start_x
        var dy3 = prev_y - seg_start_y
        var last_len = math.sqrt(Float64(dx3 * dx3 + dy3 * dy3))
        if last_len >= Float64(min_len):
            output.append((seg_start_x, seg_start_y, prev_x, prev_y))

        i += 1

    return output

# -------------------------- Hough Circles (integer) ------------------------

fn _circle_offsets(r: Int) -> List[(Int, Int)]:
    var rr = r
    var pts = List[(Int, Int)]()
    if rr <= 0:
        return pts
    var x = rr
    var y = 0
    var err = 1 - x
    while x >= y:
        pts.append(( x,  y))
        pts.append(( y,  x))
        pts.append((-y,  x))
        pts.append((-x,  y))
        pts.append((-x, -y))
        pts.append((-y, -x))
        pts.append(( y, -x))
        pts.append(( x, -y))
        y += 1
        if err < 0:
            err += 2 * y + 1
        else:
            x -= 1
            err += 2 * (y - x) + 1

    # Deduplicate
    var unique = List[(Int, Int)]()
    var i = 0
    while i < len(pts):
        var keep = True
        var j = 0
        while j < len(unique):
            if pts[i][0] == unique[j][0] and pts[i][1] == unique[j][1]:
                keep = False
                break
            j += 1
        if keep:
            unique.append(pts[i])
        i += 1
    return unique

fn _hough_circles_impl(
    img: Image,
    method: Int,
    dp: Float64,
    minDist: Float64,
    param1: Float64,
    param2: Float64,
    min_r: Int,
    max_r: Int
) -> List[(Int, Int, Int)]:
    var h = img.height()
    var w = img.width()
    var res = List[(Int, Int, Int)]()
    if h <= 2 or w <= 2:
        return res

    # radius bounds
    var r_min = min_r
    var r_max = max_r
    if r_min < 6:
        r_min = 6
    if r_max <= 0:
        if h < w:
            r_max = h // 4
        else:
            r_max = w // 4
    if r_max < r_min:
        r_max = r_min

    # radii set
    var radii = List[Int]()
    var r = r_min
    while r <= r_max:
        radii.append(r)
        r += 4

    # precompute offsets per radius
    var offs = List[List[(Int, Int)]]()
    var ri = 0
    while ri < len(radii):
        offs.append(_circle_offsets(radii[ri]))
        ri += 1

    # accumulators per radius (flattened h*w)
    var accs = List[List[Int]]()
    var rr = 0
    while rr < len(radii):
        var acc = List[Int]()
        var k = 0
        while k < h * w:
            acc.append(0)
            k += 1
        accs.append(acc)
        rr += 1

    # vote along circle perimeters for each edge pixel
    var y = 0
    while y < h:
        var x = 0
        while x < w:
            if _gray_at(img, x, y) > UInt8(0):
                var ridx = 0
                while ridx < len(radii):
                    var of = offs[ridx]
                    var oi = 0
                    while oi < len(of):
                        var cx = x - of[oi][0]
                        var cy = y - of[oi][1]
                        if cx >= 0 and cx < w and cy >= 0 and cy < h:
                            var idx = cy * w + cx
                            accs[ridx][idx] = accs[ridx][idx] + 1
                        oi += 1
                    ridx += 1
            x += 1
        y += 1

    # peak picking with minDist suppression
    var min_d2 = Int(minDist * minDist)
    if min_d2 <= 0:
        min_d2 = 64

    rr = 0
    while rr < len(radii):
        var acc = accs[rr]
        var perim = len(offs[rr])
        var base_thr = (perim * 3) // 4
        if base_thr < 4:
            base_thr = 4
        var thr = base_thr
        if param2 > 30.0:
            thr = base_thr + Int((param2 - 30.0) * 0.1)

        var cy = 0
        while cy < h:
            var cx = 0
            while cx < w:
                var v = acc[cy * w + cx]
                if v >= thr:
                    # 3x3 NMS
                    var is_max = True
                    var ddy = -1
                    while ddy <= 1:
                        var ddx = -1
                        while ddx <= 1:
                            if not (ddx == 0 and ddy == 0):
                                var xx = cx + ddx
                                var yy = cy + ddy
                                if xx >= 0 and xx < w and yy >= 0 and yy < h:
                                    if acc[yy * w + xx] > v:
                                        is_max = False
                            ddx += 1
                        ddy += 1
                    if is_max:
                        # enforce minDist from already accepted centers
                        var ok = True
                        var i = 0
                        while i < len(res):
                            var dx = cx - res[i][0]
                            var dy = cy - res[i][1]
                            if dx * dx + dy * dy < min_d2:
                                ok = False
                                break
                            i += 1
                        if ok:
                            res.append((cx, cy, radii[rr]))
                cx += 1
            cy += 1
        rr += 1

    return res

# Backward-compatible positional signature
fn hough_circles(img: Image, method: Int, dp: Float64, minDist: Float64) -> List[(Int, Int, Int)]:
    return _hough_circles_impl(img, method, dp, minDist, 100.0, 30.0, 0, 0)

# OpenCV-style signature
fn hough_circles(
    img: Image,
    dp: Float64 = 1.0,
    min_dist: Int = 50,
    param1: Float64 = 100.0,
    param2: Float64 = 30.0,
    min_r: Int = 0,
    max_r: Int = 0,
    method: Int = 0
) -> List[(Int, Int, Int)]:
    return _hough_circles_impl(
        img,
        method,
        dp,
        Float64(min_dist),
        param1,
        param2,
        min_r,
        max_r
    )


