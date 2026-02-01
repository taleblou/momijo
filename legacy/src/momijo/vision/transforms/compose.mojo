# MIT License
# SPDX-License-Identifier: MIT
# Project:      Momijo
# Module:       vision.transforms.compose
# File:         compose.mojo
#
# Description:
#   - Compose[T] with image-specialized pipeline for Tensor[Float32] (CHW).
#   - No module-level globals; parameters live inside Step structs.
#   - Removed ambiguous generic __call__; only image __call__ remains. For non-image types use apply(x).
#   - Replaced mistaken "static methods" on Step with free factory functions.
#
# Assumptions:
#   - Image tensors are CHW Float32; if HW provided, promoted to CHW.
#   - Random affine uses bilinear sampling and zero padding.

from collections.list import List
from math import  cos, sin, floor
from momijo.tensor import tensor

# Kind helpers (replaces const)
@always_inline
fn _KIND_AFFINE() -> Int: return 0
@always_inline
fn _KIND_NORMALIZE() -> Int: return 1

struct StepAffine(Copyable, Movable):
    var degrees:     Float32
    var translate_x: Float32
    var translate_y: Float32
    var scale_min:   Float32
    var scale_max:   Float32

    fn __init__(out self,
                degrees: Float32,
                translate_x: Float32,
                translate_y: Float32,
                scale_min: Float32,
                scale_max: Float32):
        self.degrees     = degrees
        self.translate_x = translate_x
        self.translate_y = translate_y
        self.scale_min   = scale_min
        self.scale_max   = scale_max

struct StepNorm(Copyable, Movable):
    var mean: List[Float32]
    var std:  List[Float32]

    fn __init__(out self, mean: List[Float32], std: List[Float32]):
        self.mean = mean.copy()
        self.std  = std.copy()

struct Step(Copyable, Movable):
    var kind:   Int
    var affine: StepAffine
    var norm:   StepNorm

    fn __init__(out self, kind: Int, affine: StepAffine, norm: StepNorm):
        self.kind   = kind
        self.affine = affine .copy()
        self.norm   = norm .copy()

# ----- Free factory helpers (avoids invalid 'self' methods) -----
fn make_step_affine(a: StepAffine) -> Step:
    var s = StepAffine(a.degrees, a.translate_x, a.translate_y, a.scale_min, a.scale_max)
    return Step(_KIND_AFFINE(), s, StepNorm(List[Float32](), List[Float32]()))

fn make_step_norm(nm: StepNorm) -> Step:
    var z = StepAffine(0.0, 0.0, 0.0, 1.0, 1.0)
    return Step(_KIND_NORMALIZE(), z, StepNorm(nm.mean, nm.std))


# ------------------------------
# Generic function-chain Compose
# ------------------------------
struct Compose[T: Copyable & Movable](Copyable, Movable):
    var _fns:   List[fn(T) -> T]   # optional, for non-image pipelines via apply()
    var _steps: List[Step]         # image steps (used when T == Tensor[Float32])

    fn __init__(out self):
        self._fns   = List[fn(T) -> T]()
        self._steps = List[Step]()

    fn __init__(out self, fns: List[fn(T) -> T]):
        var tmp = List[fn(T) -> T]()
        var i = 0
        var n = len(fns)
        while i < n:
            tmp.append(fns[i])
            i = i + 1
        self._fns = tmp.copy()
        self._steps = List[Step]()

    fn __copyinit__(out self, other: Self):
        self._fns   = other._fns.copy()
        self._steps = other._steps.copy()

    # For non-image types, explicitly call apply()
    fn apply(self, x: T) -> T:
        var y = x.copy()
        var i = 0
        var n = len(self._fns)
        while i < n:
            y = self._fns[i](y)
            i = i + 1
        return y

    # Image builders (meaningful when T == Tensor[Float32])
    fn add_random_affine(mut self,
                         degrees: Float32,
                         translate_x: Float32,
                         translate_y: Float32,
                         scale_min: Float32,
                         scale_max: Float32) -> Compose[T]:
        var st = make_step_affine(StepAffine(degrees, translate_x, translate_y, scale_min, scale_max))
        self._steps.append(st .copy())
        return self.copy()

    fn add_normalize(mut self,
                     mean: List[Float32],
                     std:  List[Float32]) -> Compose[T]:
        var st = make_step_norm(StepNorm(mean, std))
        self._steps.append(st .copy())
        return self.copy()

    # ---- Specialized operator-call ONLY for images ----
    fn __call__(self, x_in: tensor.Tensor[Float32]) -> tensor.Tensor[Float32]:
        var x = x_in.copy()
        if len(x.shape()) == 2:
            x = x.unsqueeze(0)  # promote HW -> CHW
        var y = x.copy()

        # Apply recorded image steps
        var i = 0
        var m = len(self._steps)
        while i < m:
            var st = self._steps[i].copy()
            if st.kind == _KIND_AFFINE():
                y = _apply_random_affine_with_params(y, st.affine)
            elif st.kind == _KIND_NORMALIZE():
                y = _apply_normalize_with_params(y, st.norm.mean, st.norm.std)
            i = i + 1

        # NOTE: We intentionally do NOT apply generic _fns here to avoid casts/ambiguity.
        return y .copy()


# -------------- Helpers --------------

@always_inline
fn _abs_f64(v: Float32) -> Float32:
    return (v if v >= 0.0 else -v)

@always_inline
fn _broadcast_get(vals: List[Float32], idx: Int) -> Float32:
    var m = len(vals)
    if m == 0: return 0.0
    if m == 1: return vals[0]
    if idx < m: return vals[idx]
    return vals[m - 1]

# Return ~uniform-ish in (-1,1) using normal -> bounded transform
@always_inline
fn _rand_symm() -> Float32:
    var n = tensor.randn([1])._data[0]
    var a = _abs_f64(n)
    return n / (1.0 + a)  # in (-1,1)

@always_inline
fn _lerp(a: Float32, b: Float32, t: Float32) -> Float32:
    return a + (b - a) * t


# -------------- Normalize --------------
fn _apply_normalize_with_params(img_in: tensor.Tensor[Float32],
                                mean: List[Float32],
                                std:  List[Float32]) -> tensor.Tensor[Float32]:
    var x = img_in .copy()
    # Expect CHW
    var C = x.shape()[0]
    var H = x.shape()[1]
    var W = x.shape()[2]

    var y = tensor.zeros_like(x)
    var c = 0
    while c < C:
        var mean_c = _broadcast_get(mean, c)
        var std_c  = _broadcast_get(std,  c)
        if std_c == 0.0:
            std_c = 1.0
        var hw = H * W
        var base = c * hw
        var h = 0
        while h < H:
            var w = 0
            while w < W:
                var idx = base + h * W + w
                y._data[idx] = (x._data[idx] - mean_c) / std_c
                w = w + 1
            h = h + 1
        c = c + 1
    return y .copy()


# -------------- Random Affine --------------
# Apply per-image random parameters derived from StepAffine bounds.
fn _apply_random_affine_with_params(img: tensor.Tensor[Float32],
                                    p: StepAffine) -> tensor.Tensor[Float32]:
    var C = img.shape()[0]
    var H = img.shape()[1]
    var W = img.shape()[2]

    # Draw randoms
    var r_ang = _rand_symm()
    var r_tx  = _rand_symm()
    var r_ty  = _rand_symm()
    var r_sc  = _rand_symm()

    var angle_deg = p.degrees * r_ang
    var tx_pix    = p.translate_x * Float32(W) * r_tx
    var ty_pix    = p.translate_y * Float32(H) * r_ty
    var t_sc      = (r_sc + 1.0) * 0.5
    var scale     = _lerp(p.scale_min, p.scale_max, t_sc)

    var pi :Float32= 3.141592653589793
    var rad  :Float32= angle_deg * (pi / 180.0)
    var cs  :Float32= cos(rad)
    var sn  :Float32= sin(rad)

    # Center coordinates
    var cx  :Float32= (Float32(W) - 1.0) * 0.5
    var cy  :Float32= (Float32(H) - 1.0) * 0.5

    var out = tensor.zeros_like(img)

    var c = 0
    while c < C:
        var base = c * H * W
        var yy = 0
        while yy < H:
            var xx = 0
            while xx < W:
                var dx = Float32(xx) - cx
                var dy = Float32(yy) - cy

                # Inverse mapping: inv(A) = (1/scale) * R(-theta)
                var inv_scale = 1.0 / scale
                var ics = cs
                var isn = -sn

                var rx = ics * dx - isn * dy
                var ry = isn * dx + ics * dy

                var sx = rx * inv_scale - tx_pix + cx
                var sy = ry * inv_scale - ty_pix + cy

                var val = _bilinear_sample(img, c, sx, sy, W, H)
                out._data[base + yy * W + xx] = val
                xx = xx + 1
            yy = yy + 1
        c = c + 1

    return out .copy()


fn _bilinear_sample(src: tensor.Tensor[Float32],
                    c: Int,
                    fx: Float32,
                    fy: Float32,
                    W: Int,
                    H: Int) -> Float32:
    if fx < 0.0 or fy < 0.0 or fx > Float32(W - 1) or fy > Float32(H - 1):
        return 0.0

    var x0 = Int(floor(fx))
    var y0 = Int(floor(fy))
    var x1 = x0 + 1
    var y1 = y0 + 1

    var dx = fx - Float32(x0)
    var dy = fy - Float32(y0)

    if x1 >= W: x1 = W - 1
    if y1 >= H: y1 = H - 1

    var base = c * H * W

    var i00 = base + y0 * W + x0
    var i10 = base + y0 * W + x1
    var i01 = base + y1 * W + x0
    var i11 = base + y1 * W + x1

    var v00 = src._data[i00]
    var v10 = src._data[i10]
    var v01 = src._data[i01]
    var v11 = src._data[i11]

    var vx0 = v00 * (1.0 - dx) + v10 * dx
    var vx1 = v01 * (1.0 - dx) + v11 * dx
    var v   = vx0 * (1.0 - dy) + vx1 * dy
    return v
