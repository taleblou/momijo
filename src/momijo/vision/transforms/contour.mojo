# MIT License
# SPDX-License-Identifier: MIT
# Project: momijo.vision | File: src/momijo/vision/transforms/contour.mojo

from momijo.vision.image import Image


# --- Pixel read hook --------------------------------------------------------
# Replace this when Image exposes a real packed HWC UInt8 buffer.
fn _peek(_: Image, _: Int, _: Int) -> UInt8:
    # Return 0 => background. Wire to actual pixels later.
    return UInt8(0)

fn _is_on(img: Image, x: Int, y: Int, thresh: UInt8 = UInt8(1)) -> Bool:
    return _peek(img, x, y) >= thresh

# --- Small math helpers -----------------------------------------------------
fn _fabs(x: Float64) -> Float64:
    return -x if x < 0.0 else x
 

fn _order_boundary(points: List[(Int, Int)], W: Int, H: Int) -> List[(Int, Int)]:
    var contour = List[(Int, Int)]()
    var n = len(points)
    if n == 0:
        return contour.copy()

    # Presence map
    var present = List[List[Bool]]()
    var yy = 0
    while yy < H:
        var row = List[Bool]()
        var xx = 0
        while xx < W:
            row.append(False)
            xx += 1
        present.append(row.copy())
        yy += 1

    var i = 0
    while i < n:
        var px = points[i][0]
        var py = points[i][1]
        if px >= 0 and px < W and py >= 0 and py < H:
            present[py][px] = True
        i += 1

    # Used map
    var used = List[List[Bool]]()
    yy = 0
    while yy < H:
        var row2 = List[Bool]()
        var xx2 = 0
        while xx2 < W:
            row2.append(False)
            xx2 += 1
        used.append(row2.copy())
        yy += 1

    # 8-connected directions (clockwise)
    var dirs = List[(Int, Int)]()
    dirs.append((1, 0))
    dirs.append((1, 1))
    dirs.append((0, 1))
    dirs.append((-1, 1))
    dirs.append((-1, 0))
    dirs.append((-1, -1))
    dirs.append((0, -1))
    dirs.append((1, -1))

    # Start at lexicographically smallest (y, x)
    var start_idx = 0
    var best_x = points[0][0]
    var best_y = points[0][1]
    i = 1
    while i < n:
        var px2 = points[i][0]
        var py2 = points[i][1]
        var better = (py2 < best_y) or (py2 == best_y and px2 < best_x)
        if better:
            best_x = px2
            best_y = py2
            start_idx = i
        i += 1

    var sx = best_x
    var sy = best_y
    var cx = sx
    var cy = sy
    if sx < 0 or sx >= W or sy < 0 or sy >= H:
        return contour.copy()
    if not present[sy][sx]:
        return contour.copy()

    contour.append((cx, cy))
    used[cy][cx] = True

    # Greedy walk
    var steps = 0
    var max_steps = n * 4
    while steps < max_steps:
        var found = False
        var k = 0
        while k < len(dirs):
            var nx = cx + dirs[k][0]
            var ny = cy + dirs[k][1]
            if nx >= 0 and nx < W and ny >= 0 and ny < H:
                if present[ny][nx] and not used[ny][nx]:
                    cx = nx
                    cy = ny
                    contour.append((cx, cy))
                    used[cy][cx] = True
                    found = True
                    break
            k += 1
        if not found:
            break
        if cx == sx and cy == sy and len(contour) > 2:
            break
        steps += 1

    return contour.copy()

# Unique helper: checks channel 0 >= thresh on packed HWC/UInt8.
fn is_on_u8(img: Image, x: Int, y: Int, thresh: UInt8) -> Bool:
    var base = img.ensure_packed_hwc_u8(True)
    if x < 0 or x >= base.width() or y < 0 or y >= base.height():
        return False
    var v = base.get_u8(y, x, 0)
    return v >= thresh

fn find_contours(img: Image) -> List[List[(Int, Int)]]:
    var H = img.height()
    var W = img.width()
    var contours = List[List[(Int, Int)]]()
    if H <= 0 or W <= 0:
        return contours.copy()

    var seen = List[List[Bool]]()
    var y = 0
    while y < H:
        var row = List[Bool]()
        var x = 0
        while x < W:
            row.append(False)
            x += 1
        seen.append(row.copy())
        y += 1

    var qx = List[Int]()
    var qy = List[Int]()

    var nb4 = List[(Int, Int)]()
    nb4.append((1, 0))
    nb4.append((-1, 0))
    nb4.append((0, 1))
    nb4.append((0, -1))

    y = 0
    while y < H:
        var x = 0
        while x < W:
            if not seen[y][x] and is_on_u8(img, x, y, UInt8(1)):
                qx = List[Int]()
                qy = List[Int]()
                qx.append(x)
                qy.append(y)
                seen[y][x] = True

                var comp_boundary = List[(Int, Int)]()
                var qi = 0
                while qi < len(qx):
                    var cx = qx[qi]
                    var cy = qy[qi]

                    var is_boundary = False
                    var t = 0
                    while t < 4:
                        var nx = cx + nb4[t][0]
                        var ny = cy + nb4[t][1]
                        if nx < 0 or nx >= W or ny < 0 or ny >= H:
                            is_boundary = True
                        else:
                            if not is_on_u8(img, nx, ny, UInt8(1)):
                                is_boundary = True
                        t += 1
                    if is_boundary:
                        comp_boundary.append((cx, cy))

                    t = 0
                    while t < 4:
                        var nx2 = cx + nb4[t][0]
                        var ny2 = cy + nb4[t][1]
                        if nx2 >= 0 and nx2 < W and ny2 >= 0 and ny2 < H:
                            if not seen[ny2][nx2] and is_on_u8(img, nx2, ny2, UInt8(1)):
                                seen[ny2][nx2] = True
                                qx.append(nx2)
                                qy.append(ny2)
                        t += 1

                    qi += 1

                var contour = _order_boundary(comp_boundary, W, H)
                if len(contour) > 0:
                    contours.append(contour.copy())

            x += 1
        y += 1

    return contours.copy()






# Assumes _order_boundary(points, W, H) is defined elsewhere in your module set.

fn find_contours(img: Image, external_only: Bool = False) -> List[List[(Int, Int)]]:
    var H = img.height()
    var W = img.width()
    var contours = List[List[(Int, Int)]]()
    if H <= 0 or W <= 0:
        return contours.copy()

    # Seen mask
    var seen = List[List[Bool]]()
    var y = 0
    while y < H:
        var row = List[Bool]()
        var x = 0
        while x < W:
            row.append(False)
            x += 1
        seen.append(row.copy())
        y += 1

    # BFS queues
    var qx = List[Int]()
    var qy = List[Int]()

    # 4-neighborhood offsets
    var dx4 = List[Int]()
    dx4.append(1); dx4.append(-1); dx4.append(0); dx4.append(0)
    var dy4 = List[Int]()
    dy4.append(0); dy4.append(0); dy4.append(1); dy4.append(-1)

    # Scan all pixels
    y = 0
    while y < H:
        var x = 0
        while x < W:
            if (not seen[y][x]) and is_on_u8(img, x, y, UInt8(1)):
                # BFS over this component
                var comp_boundary = List[(Int, Int)]()
                qx = List[Int]()
                qy = List[Int]()
                qx.append(x)
                qy.append(y)
                seen[y][x] = True

                var qi = 0
                while qi < len(qx):
                    var cx = qx[qi]
                    var cy = qy[qi]

                    # boundary test: any 4-neighbor is off or OOB
                    var is_boundary = False
                    var t = 0
                    while t < 4:
                        var nx = cx + dx4[t]
                        var ny = cy + dy4[t]
                        if (nx < 0) or (nx >= W) or (ny < 0) or (ny >= H):
                            is_boundary = True
                        else:
                            if not is_on_u8(img, nx, ny, UInt8(1)):
                                is_boundary = True
                        t += 1
                    if is_boundary:
                        comp_boundary.append((cx, cy))

                    # expand BFS
                    t = 0
                    while t < 4:
                        var nx2 = cx + dx4[t]
                        var ny2 = cy + dy4[t]
                        if (nx2 >= 0) and (nx2 < W) and (ny2 >= 0) and (ny2 < H):
                            if (not seen[ny2][nx2]) and is_on_u8(img, nx2, ny2, UInt8(1)):
                                seen[ny2][nx2] = True
                                qx.append(nx2)
                                qy.append(ny2)
                        t += 1

                    qi += 1

                # Order boundary points into a contour polyline
                var contour = _order_boundary(comp_boundary, W, H)

                # external_only hook (placeholder): implement hierarchy filtering if needed
                if len(contour) > 0:
                    contours.append(contour.copy())
            x += 1
        y += 1

    return contours.copy()


fn get_contour(contours: List[List[(Int,Int)]], idx: Int) -> List[(Int,Int)]:
    if idx < 0 or idx >= len(contours):
        return List[(Int,Int)]()
    return contours[idx].copy()


fn len_contours(cs: List[List[(Int, Int)]]) -> Int:
    return len(cs)
 

# Returns True if pixel at (x,y) is considered "on" in a binary mask.
# If strict=True and on_value is provided, it requires equality; otherwise any nonzero is on.
fn _is_on(img: Image, x: Int, y: Int, on_value: UInt8 = UInt8(1), strict: Bool = False) -> Bool:
    var v = img.get(y, x, 0)
    if strict:
        return v == on_value
    return v != UInt8(0)

# 8-neighborhood adjacency check
fn _is_neighbor8(ax: Int, ay: Int, bx: Int, by: Int) -> Bool:
    var dx = ax - bx
    if dx < 0: dx = -dx
    var dy = ay - by
    if dy < 0: dy = -dy
    return (dx <= 1) and (dy <= 1) and not (dx == 0 and dy == 0)



# --- Contour utilities --------------------------------------------------------

# Axis-aligned bounding rectangle for a contour.
# Returns (x, y, w, h). If the contour is empty, returns (0, 0, 0, 0).
fn bounding_rect(contour: List[(Int, Int)]) -> (Int, Int, Int, Int):
    var n = len(contour)
    if n == 0:
        return (0, 0, 0, 0)

    var min_x = contour[0][0]
    var max_x = contour[0][0]
    var min_y = contour[0][1]
    var max_y = contour[0][1]

    var i = 1
    while i < n:
        var px = contour[i][0]
        var py = contour[i][1]
        if px < min_x:
            min_x = px
        if px > max_x:
            max_x = px
        if py < min_y:
            min_y = py
        if py > max_y:
            max_y = py
        i += 1

    # Inclusive bounds → +1
    var w = (max_x - min_x) + 1
    var h = (max_y - min_y) + 1
    if w < 0:
        w = 0
    if h < 0:
        h = 0
    return (min_x, min_y, w, h)

# Polygon area via the shoelace formula.
# Assumes the contour is a polyline around the object; returns absolute area.
# For degenerate/short contours, returns 0.0.
fn contour_area(contour: List[(Int, Int)]) -> Float64:
    var n = len(contour)
    if n < 3:
        return 0.0

    var acc = 0.0
    var i = 0
    while i < n:
        var j = i + 1
        if j == n:
            j = 0
        var xi = Float64(contour[i][0])
        var yi = Float64(contour[i][1])
        var xj = Float64(contour[j][0])
        var yj = Float64(contour[j][1])
        acc = acc + (xi * yj - xj * yi)
        i += 1

    if acc < 0.0:
        acc = -acc
    return 0.5 * acc


# --------- scalar helpers ----------
fn _abs(x: Float64) -> Float64:
    if x >= 0.0: return x
    return -x

# Safe square root using Newton-Raphson iteration
fn _sqrt(v: Float64) -> Float64:
    if v <= 0.0:
        return 0.0

    # Initial guess: use the input itself or 1.0 if very small
    var x = v
    if x < 1.0:
        x = 1.0


    var i = 0
    while i < 10:  # fixed number of iterations
        var next = 0.5 * (x + v / x)
        if (x - next) < 1e-12 and (next - x) < 1e-12:
            # converged
            if next < 0.0:
                return 0.0
            else:
                return next

        x = next
        i += 1

    # final guard
    if x < 0.0:
        return 0.0
    return x

fn _hypot(dx: Float64, dy: Float64) -> Float64:
    var ax = _abs(dx)
    var ay = _abs(dy)
    if ax < ay:
        if ay == 0.0: return 0.0
        var r = ax / ay
        return ay * _sqrt(1.0 + r * r)
    else:
        if ax == 0.0: return 0.0
        var r = ay / ax
        return ax * _sqrt(1.0 + r * r)

# --------- core length on flat coords [x0,y0,x1,y1,...] ----------
fn _arc_length_flat(coords: List[Float64], closed: Bool) -> Float64:
    var n = len(coords)
    if (n < 4) or ((n % 2) != 0):
        return 0.0
    var total: Float64 = 0.0
    var i = 0
    while (i + 3) < n:
        var x0 = coords[i]
        var y0 = coords[i + 1]
        var x1 = coords[i + 2]
        var y1 = coords[i + 3]
        total += _hypot(x1 - x0, y1 - y0)
        i += 2
    if closed:
        total += _hypot(coords[0] - coords[n - 2], coords[1] - coords[n - 1])
    return total

# --------- flatten helpers ----------
fn _flatten_xy(points: List[List[Float64]]) -> List[Float64]:
    var flat = List[Float64]()
    flat.reserve(len(points) * 2)
    var i = 0
    while i < len(points):
        var p = points[i].copy()
        if len(p) >= 2:
            flat.append(p[0])
            flat.append(p[1])
        i += 1
    return flat.copy()

fn _flatten_xy_f32(points: List[List[Float32]]) -> List[Float64]:
    var flat = List[Float64]()
    flat.reserve(len(points) * 2)
    var i = 0
    while i < len(points):
        var p = points[i].copy()
        if len(p) >= 2:
            flat.append(Float64(p[0]))
            flat.append(Float64(p[1]))
        i += 1
    return flat.copy()

fn _flatten_xy_i(points: List[List[Int]]) -> List[Float64]:
    var flat = List[Float64]()
    flat.reserve(len(points) * 2)
    var i = 0
    while i < len(points):
        var p = points[i].copy()
        if len(p) >= 2:
            flat.append(Float64(p[0]))
            flat.append(Float64(p[1]))
        i += 1
    return flat.copy()
# --------- public overloads (2D: [[x,y],...]) ----------
fn arc_length(points: List[List[Float64]], closed: Bool = False) -> Float64:
    return _arc_length_flat(_flatten_xy(points), closed)

fn arc_length(points: List[List[Float32]], closed: Bool = False) -> Float64:
    return _arc_length_flat(_flatten_xy_f32(points), closed)

fn arc_length(points: List[List[Int]], closed: Bool = False) -> Float64:
    return _arc_length_flat(_flatten_xy_i(points), closed)

# --------- public overloads (flat: [x0,y0,x1,y1,...]) ----------
fn arc_length(coords: List[Float64], closed: Bool = False) -> Float64:
    return _arc_length_flat(coords, closed)

fn arc_length(coords: List[Float32], closed: Bool = False) -> Float64:
    var tmp = List[Float64]()
    tmp.reserve(len(coords))
    var i = 0
    while i < len(coords):
        tmp.append(Float64(coords[i]))
        i += 1
    return _arc_length_flat(tmp, closed)

fn arc_length(coords: List[Int], closed: Bool = False) -> Float64:
    var tmp = List[Float64]()
    tmp.reserve(len(coords))
    var i = 0
    while i < len(coords):
        tmp.append(Float64(coords[i]))
        i += 1
    return _arc_length_flat(tmp, closed)

# --------- public overloads (tuple contours: [(x,y), ...]) ----------
fn arc_length(contour: List[(Int, Int)], closed: Bool = False) -> Float64:
    var n = len(contour)
    if n < 2:
        return 0.0
    var perim = 0.0
    var i = 0
    var limit = n - 1
    while i < limit:
        var dx = Float64(contour[i + 1][0] - contour[i][0])
        var dy = Float64(contour[i + 1][1] - contour[i][1])
        perim = perim + _hypot(dx, dy)
        i += 1
    if closed:
        var dxc = Float64(contour[0][0] - contour[n - 1][0])
        var dyc = Float64(contour[0][1] - contour[n - 1][1])
        perim = perim + _hypot(dxc, dyc)
    return perim

fn arc_length(contour: List[(Float32, Float32)], closed: Bool = False) -> Float64:
    var n = len(contour)
    if n < 2:
        return 0.0
    var perim = 0.0
    var i = 0
    var limit = n - 1
    while i < limit:
        var dx = Float64(contour[i + 1][0]) - Float64(contour[i][0])
        var dy = Float64(contour[i + 1][1]) - Float64(contour[i][1])
        perim = perim + _hypot(dx, dy)
        i += 1
    if closed:
        var dxc = Float64(contour[0][0]) - Float64(contour[n - 1][0])
        var dyc = Float64(contour[0][1]) - Float64(contour[n - 1][1])
        perim = perim + _hypot(dxc, dyc)
    return perim

fn arc_length(contour: List[(Float64, Float64)], closed: Bool = False) -> Float64:
    var n = len(contour)
    if n < 2:
        return 0.0
    var perim = 0.0
    var i = 0
    var limit = n - 1
    while i < limit:
        var dx = contour[i + 1][0] - contour[i][0]
        var dy = contour[i + 1][1] - contour[i][1]
        perim = perim + _hypot(dx, dy)
        i += 1
    if closed:
        var dxc = contour[0][0] - contour[n - 1][0]
        var dyc = contour[0][1] - contour[n - 1][1]
        perim = perim + _hypot(dxc, dyc)
    return perim