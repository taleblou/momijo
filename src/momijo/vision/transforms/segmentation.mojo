# MIT License
# Project: momijo.vision
# File: momijo/vision/segmentation.mojo
# SPDX-License-Identifier: MIT

# Connected Components Labeling (binary -> labels)
# - Input 'bw' is a binary image-like 2D list:
#     0 = background, non-zero = foreground
# - Returns: (num_labels, labels_2d) where labels_2d[y][x] in [0..num_labels]
#   Label 0 is background. Foreground components are 1..num_labels.
# - connectivity: 4 or 8

fn _is_fg(v: Int) -> Bool:
    return v != 0

fn _dirs4() -> List[(Int, Int)]:
    var d = List[(Int, Int)]()
    d.append(( 1, 0))
    d.append((-1, 0))
    d.append(( 0, 1))
    d.append(( 0,-1))
    return d

fn _dirs8() -> List[(Int, Int)]:
    var d = _dirs4()
    d.append(( 1, 1))
    d.append(( 1,-1))
    d.append((-1, 1))
    d.append((-1,-1))
    return d

fn _cc_label_core(bw: List[List[Int]], connectivity: Int) -> (Int, List[List[Int]]):
    var h = len(bw)
    if h == 0: 
        return (0, List[List[Int]]())
    var w = len(bw[0])

    # labels initialized to 0 (background)
    var labels = List[List[Int]]()
    labels.reserve(h)
    var y = 0
    while y < h:
        var row = List[Int]()
        row.reserve(w)
        var x = 0
        while x < w:
            row.append(0)
            x += 1
        labels.append(row)
        y += 1

    var dirs = _dirs4()
    if connectivity == 8:
        dirs = _dirs8()

    var current = 0

    # simple stack-based DFS
    var y0 = 0
    while y0 < h:
        var x0 = 0
        while x0 < w:
            if _is_fg(bw[y0][x0]) and (labels[y0][x0] == 0):
                current += 1
                # push
                var stack_y = List[Int]()
                var stack_x = List[Int]()
                stack_y.append(y0)
                stack_x.append(x0)
                labels[y0][x0] = current

                while len(stack_y) > 0:
                    var sy = stack_y.pop()
                    var sx = stack_x.pop()

                    var k = 0
                    while k < len(dirs):
                        var dy = dirs[k].1
                        var dx = dirs[k].0
                        var ny = sy + dy
                        var nx = sx + dx
                        if 0 <= ny and ny < h and 0 <= nx and nx < w:
                            if _is_fg(bw[ny][nx]) and (labels[ny][nx] == 0):
                                labels[ny][nx] = current
                                stack_y.append(ny)
                                stack_x.append(nx)
                        k += 1
                # component done
            x0 += 1
        y0 += 1

    return (current, labels)
 
 

# ----- Matrix image overloads -----
# ---------------- Core: Int mask (HxW, values 0/1) ----------------

fn connected_components(bw: List[List[Int]], connectivity: Int = 4) -> (Int, List[List[Int]]):
    var h = len(bw)
    if h == 0:
        return (0, List[List[Int]]())
    var w = len(bw[0])
    if w == 0:
        return (0, List[List[Int]]())

    # labels initialized to 0
    var labels = List[List[Int]]()
    var y = 0
    while y < h:
        var row = List[Int]()
        var x = 0
        while x < w:
            row.append(0)
            x += 1
        labels.append(row)
        y += 1

    # neighbor offsets
    var dx = List[Int]()
    var dy = List[Int]()
    if connectivity == 8:
        dx.append(1);  dy.append(0)
        dx.append(-1); dy.append(0)
        dx.append(0);  dy.append(1)
        dx.append(0);  dy.append(-1)
        dx.append(1);  dy.append(1)
        dx.append(1);  dy.append(-1)
        dx.append(-1); dy.append(1)
        dx.append(-1); dy.append(-1)
    else:
        # default to 4-connectivity
        dx.append(1);  dy.append(0)
        dx.append(-1); dy.append(0)
        dx.append(0);  dy.append(1)
        dx.append(0);  dy.append(-1)

    var num = 0
    var qx = List[Int]()
    var qy = List[Int]()

    y = 0
    while y < h:
        var x = 0
        while x < w:
            if bw[y][x] != 0 and labels[y][x] == 0:
                num = num + 1
                qx = List[Int]()
                qy = List[Int]()
                qx.append(x)
                qy.append(y)
                labels[y][x] = num

                var qi = 0
                while qi < len(qx):
                    var cx = qx[qi]
                    var cy = qy[qi]

                    var k = 0
                    while k < len(dx):
                        var nx = cx + dx[k]
                        var ny = cy + dy[k]
                        if nx >= 0 and nx < w and ny >= 0 and ny < h:
                            if bw[ny][nx] != 0 and labels[ny][nx] == 0:
                                labels[ny][nx] = num
                                qx.append(nx)
                                qy.append(ny)
                        k += 1
                    qi += 1
            x += 1
        y += 1

    return (num, labels)

# ---------------- Overloads: different input types ----------------

# HxW UInt8 grayscale (foreground if > 0)
fn connected_components(bw: List[List[UInt8]], connectivity: Int = 4) -> (Int, List[List[Int]]):
    var h = len(bw)
    if h == 0:
        return (0, List[List[Int]]())
    var w = len(bw[0])
    var mask = List[List[Int]]()
    var y = 0
    while y < h:
        var row = List[Int]()
        var x = 0
        while x < w:
            var v = bw[y][x]
            if v > UInt8(0):
                row.append(1)
            else:
                row.append(0)
            x += 1
        mask.append(row)
        y += 1
    return connected_components(mask, connectivity)

# HxWxC UInt8 color (foreground if any channel > 0)
fn connected_components(bw: List[List[List[UInt8]]], connectivity: Int = 4) -> (Int, List[List[Int]]):
    var h = len(bw)
    if h == 0:
        return (0, List[List[Int]]())
    var w = len(bw[0])
    var mask = List[List[Int]]()
    var y = 0
    while y < h:
        var row = List[Int]()
        var x = 0
        while x < w:
            var pix = bw[y][x]
            var fg = 0
            var c = 0
            while c < len(pix):
                if pix[c] > UInt8(0):
                    fg = 1
                    break
                c += 1
            row.append(fg)
            x += 1
        mask.append(row)
        y += 1
    return connected_components(mask, connectivity)

# HxW Float32 (foreground if > 0.0)
fn connected_components(bw: List[List[Float32]], connectivity: Int = 4) -> (Int, List[List[Int]]):
    var h = len(bw)
    if h == 0:
        return (0, List[List[Int]]())
    var w = len(bw[0])
    var mask = List[List[Int]]()
    var y = 0
    while y < h:
        var row = List[Int]()
        var x = 0
        while x < w:
            if bw[y][x] > 0.0:
                row.append(1)
            else:
                row.append(0)
            x += 1
        mask.append(row)
        y += 1
    return connected_components(mask, connectivity)

# HxW Float64 (foreground if > 0.0)
fn connected_components(bw: List[List[Float64]], connectivity: Int = 4) -> (Int, List[List[Int]]):
    var h = len(bw)
    if h == 0:
        return (0, List[List[Int]]())
    var w = len(bw[0])
    var mask = List[List[Int]]()
    var y = 0
    while y < h:
        var row = List[Int]()
        var x = 0
        while x < w:
            if bw[y][x] > 0.0:
                row.append(1)
            else:
                row.append(0)
            x += 1
        mask.append(row)
        y += 1
    return connected_components(mask, connectivity)

# ---------------- Overload: Image (expects packed HWC/UInt8 or will convert) ----------------

fn connected_components(img: Image, connectivity: Int = 4) -> (Int, List[List[Int]]):
    var base = img.ensure_packed_hwc_u8(True)
    var h = base.height()
    if h == 0:
        return (0, List[List[Int]]())
    var w = base.width()

    var mask = List[List[Int]]()
    var y = 0
    while y < h:
        var row = List[Int]()
        var x = 0
        while x < w:
            var v = base.get_u8(y, x, 0)  # uses channel 0
            if v > UInt8(0):
                row.append(1)
            else:
                row.append(0)
            x += 1
        mask.append(row)
        y += 1

    return connected_components(mask, connectivity)

 