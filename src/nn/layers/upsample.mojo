# MIT License
# Copyright (c) 2025
# SPDX-License-Identifier: MIT
#
# Module: momijo.nn.upsample
# Path:   src/momijo/nn/upsample.mojo
#
# Minimal 2D upsampling (nearest & bilinear) for pedagogy/smoke tests.
# List-based Float64 implementation supporting:
#   - Single image [C,H,W] and batch [N,C,H,W]
#   - Integer scale factors OR explicit output size
#   - align_corners flag for bilinear (and coordinate mapping)
#
# Momijo style:
# - No global vars, no `export`. Use `var` (not `let`).
# - Constructors: fn __init__(out self, ...)
# - Prefer `mut/out` over `inout`.

# --- Helpers ---
fn zeros2d(h: Int, w: Int) -> List[List[Float64]]:
    var y = List[List[Float64]]()
    for i in range(h):
        var row = List[Float64]()
        for j in range(w): row.push(0.0)
        y.push(row)
    return y

fn zeros3d(c: Int, h: Int, w: Int) -> List[List[List[Float64]]]:
    var y = List[List[List[Float64]]]()
    for ch in range(c):
        y.push(zeros2d(h, w))
    return y

fn zeros4d(n: Int, c: Int, h: Int, w: Int) -> List[List[List[List[Float64]]]]:
    var y = List[List[List[List[Float64]]]]()
    for i in range(n):
        y.push(zeros3d(c, h, w))
    return y

fn clamp_int(x: Int, lo: Int, hi: Int) -> Int:
    var v = x
    if v < lo: v = lo
    if v > hi: v = hi
    return v

fn coord_map(out_idx: Int, in_size: Int, out_size: Int, align_corners: Bool) -> Float64:
    if out_size <= 1: return 0.0
    if align_corners:
        if out_size == 1:
            return 0.0
        # map: i_in = i_out * (in-1)/(out-1)
        return Float64(out_idx) * (Float64(in_size - 1) / Float64(out_size - 1))
    else:
        # map: (i_out + 0.5) * in/out - 0.5
        return (Float64(out_idx) + 0.5) * (Float64(in_size) / Float64(out_size)) - 0.5

# --- Nearest (single image, CHW) ---
fn upsample2d_single_nearest(x: List[List[List[Float64]]], out_h: Int, out_w: Int, align_corners: Bool = False) -> List[List[List[Float64]]]:
    var C = len(x)
    if C == 0: return x
    var H = len(x[0])
    var W = 0
    if H > 0: W = len(x[0][0])
    var y = zeros3d(C, out_h, out_w)
    for c in range(C):
        for i in range(out_h):
            var src_i_f = coord_map(i, H, out_h, align_corners)
            var src_i = clamp_int(Int(round(src_i_f)), 0, H - 1)
            for j in range(out_w):
                var src_j_f = coord_map(j, W, out_w, align_corners)
                var src_j = clamp_int(Int(round(src_j_f)), 0, W - 1)
                y[c][i][j] = x[c][src_i][src_j]
    return y

# --- Bilinear (single image, CHW) ---
fn floori(x: Float64) -> Int:
    var xi = Int(x)
    if Float64(xi) > x: xi -= 1
    return xi

fn lerp(a: Float64, b: Float64, t: Float64) -> Float64:
    return a + (b - a) * t

fn get_safe(x: List[List[Float64]], i: Int, j: Int) -> Float64:
    var H = len(x)
    if H == 0: return 0.0
    var W = len(x[0])
    var ii = i
    var jj = j
    if ii < 0: ii = 0
    if jj < 0: jj = 0
    if ii >= H: ii = H - 1
    if jj >= W: jj = W - 1
    return x[ii][jj]

fn upsample2d_single_bilinear(x: List[List[List[Float64]]], out_h: Int, out_w: Int, align_corners: Bool = False) -> List[List[List[Float64]]]:
    var C = len(x)
    if C == 0: return x
    var H = len(x[0])
    var W = 0
    if H > 0: W = len(x[0][0])
    var y = zeros3d(C, out_h, out_w)
    for c in range(C):
        for i in range(out_h):
            var src_y = coord_map(i, H, out_h, align_corners)
            var y0 = floori(src_y)
            var y1 = y0 + 1
            var wy = src_y - Float64(y0)
            for j in range(out_w):
                var src_x = coord_map(j, W, out_w, align_corners)
                var x0 = floori(src_x)
                var x1 = x0 + 1
                var wx = src_x - Float64(x0)
                var v00 = get_safe(x[c], y0, x0)
                var v01 = get_safe(x[c], y0, x1)
                var v10 = get_safe(x[c], y1, x0)
                var v11 = get_safe(x[c], y1, x1)
                var v0 = lerp(v00, v01, wx)
                var v1 = lerp(v10, v11, wx)
                y[c][i][j] = lerp(v0, v1, wy)
    return y

# --- Batch wrappers [N,C,H,W] given output size ---
fn upsample2d_batch_nearest(x: List[List[List[List[Float64]]]], out_h: Int, out_w: Int, align_corners: Bool = False) -> List[List[List[List[Float64]]]]:
    var N = len(x)
    if N == 0: return x
    var C = len(x[0])
    var y = zeros4d(N, C, out_h, out_w)
    for n in range(N):
        y[n] = upsample2d_single_nearest(x[n], out_h, out_w, align_corners)
    return y

fn upsample2d_batch_bilinear(x: List[List[List[List[Float64]]]], out_h: Int, out_w: Int, align_corners: Bool = False) -> List[List[List[List[Float64]]]]:
    var N = len(x)
    if N == 0: return x
    var C = len(x[0])
    var y = zeros4d(N, C, out_h, out_w)
    for n in range(N):
        y[n] = upsample2d_single_bilinear(x[n], out_h, out_w, align_corners)
    return y

# --- Convenience: compute output from scales ---
fn out_from_scale(h: Int, w: Int, sh: Int, sw: Int) -> (Int, Int):
    var oh = h * (sh if sh > 0 else 1)
    var ow = w * (sw if sw > 0 else 1)
    return (oh, ow)

# --- Module wrapper ---
struct Upsample2d:
    var use_scale: Bool
    var scale_h: Int
    var scale_w: Int
    var out_h: Int
    var out_w: Int
    var bilinear: Bool
    var align_corners: Bool

    # If out_h/out_w > 0, explicit size is used; otherwise scale_h/scale_w must be >0.
    fn __init__(out self, scale_h: Int = 0, scale_w: Int = 0, out_h: Int = 0, out_w: Int = 0, bilinear: Bool = False, align_corners: Bool = False):
        self.scale_h = scale_h
        self.scale_w = scale_w
        self.out_h = out_h
        self.out_w = out_w
        self.bilinear = bilinear
        self.align_corners = align_corners
        self.use_scale = not (out_h > 0 and out_w > 0)

    fn forward_chw(self, x: List[List[List[Float64]]]) -> List[List[List[Float64]]]:
        var H = 0
        var W = 0
        if len(x) > 0:
            H = len(x[0])
            if H > 0: W = len(x[0][0])
        var oh = self.out_h
        var ow = self.out_w
        if self.use_scale:
            var sc_h = self.scale_h if self.scale_h > 0 else 1
            var sc_w = self.scale_w if self.scale_w > 0 else 1
            oh = H * sc_h
            ow = W * sc_w
        if self.bilinear:
            return upsample2d_single_bilinear(x, oh, ow, self.align_corners)
        else:
            return upsample2d_single_nearest(x, oh, ow, self.align_corners)

    fn forward_nchw(self, x: List[List[List[List[Float64]]]]) -> List[List[List[List[Float64]]]]:
        var H = 0
        var W = 0
        if len(x) > 0 and len(x[0]) > 0:
            H = len(x[0][0])
            if H > 0: W = len(x[0][0][0])
        var oh = self.out_h
        var ow = self.out_w
        if self.use_scale:
            var sc_h = self.scale_h if self.scale_h > 0 else 1
            var sc_w = self.scale_w if self.scale_w > 0 else 1
            oh = H * sc_h
            ow = W * sc_w
        if self.bilinear:
            return upsample2d_batch_bilinear(x, oh, ow, self.align_corners)
        else:
            return upsample2d_batch_nearest(x, oh, ow, self.align_corners)

# --- Smoke tests ---
fn _self_test() -> Bool:
    var ok = True

    # 1) Simple CHW 1x2x2 nearest up x2 -> 1x4x4
    var C = 1; var H = 2; var W = 2
    var x = zeros3d(C, H, W)
    x[0][0][0] = 1.0; x[0][0][1] = 2.0
    x[0][1][0] = 3.0; x[0][1][1] = 4.0
    var up_n = Upsample2d(scale_h=2, scale_w=2, bilinear=False, align_corners=False)
    var y_n = up_n.forward_chw(x)
    ok = ok and (len(y_n) == 1) and (len(y_n[0]) == 4) and (len(y_n[0][0]) == 4)

    # 2) Bilinear to explicit size 5x3 with/without align_corners
    var up_b = Upsample2d(out_h=5, out_w=3, bilinear=True, align_corners=False)
    var y_b = up_b.forward_chw(x)
    ok = ok and (len(y_b[0]) == 5) and (len(y_b[0][0]) == 3)
    var up_b2 = Upsample2d(out_h=5, out_w=3, bilinear=True, align_corners=True)
    var y_b2 = up_b2.forward_chw(x)
    ok = ok and (len(y_b2[0]) == 5) and (len(y_b2[0][0]) == 3)

    # 3) Batch path N=2
    var N = 2
    var xb = zeros4d(N, C, H, W)
    xb[0] = x; xb[1] = x
    var yb = up_n.forward_nchw(xb)
    ok = ok and (len(yb) == 2) and (len(yb[0][0]) == 4) and (len(yb[0][0][0]) == 4)

    return ok
 
