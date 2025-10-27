# MIT License
# SPDX-License-Identifier: MIT
# Project: momijo.vision | File: src/momijo/vision/transforms/geom.mojo

from momijo.vision.image import Image
from momijo.vision.transforms.array import full, zeros

# MIT License
# Project: momijo.vision.transforms
# File: momijo/vision/transforms/geom.mojo
# Shape-only geometric transforms. No pixel resampling yet.

 

# --- helpers ---------------------------------------------------------------

fn _pos_or(default_v: Int, v: Int) -> Int:
    var x = v
    if x <= 0: x = default_v
    return x

fn _norm_rot90(k: Int) -> Int:
    var t = k % 4
    if t < 0: t += 4
    return t


 

# --- public API ------------------------------------------------------------

fn resize(img: Image, new_w: Int, new_h: Int, inter: Int = 0) -> Image:
    # Shape-only resize. 'inter' is accepted for API compatibility.
    var W = _pos_or(img.width(), new_w)
    var H = _pos_or(img.height(), new_h)
    var C = img.channels()

    # For now just allocate a new blank image of target size.
    # Replace with actual resampling later.
    return Image.new_hwc_u8(H, W, C, UInt8(0))

    
# basic rotation
fn rotate_basic(img: Image, angle: Float64) -> Image:
    if angle == 0.0:
        return img.copy()
    elif angle == 90.0 or angle == -270.0:
        return Image.new_hwc_u8(img.width(), img.height(), img.channels(), UInt8(0))
    elif angle == 180.0 or angle == -180.0:
        return Image.new_hwc_u8(img.height(), img.width(), img.channels(), UInt8(0))
    elif angle == 270.0 or angle == -90.0:
        return Image.new_hwc_u8(img.width(), img.height(), img.channels(), UInt8(0))
    else:
        return Image.new_hwc_u8(img.height(), img.width(), img.channels(), UInt8(0))

# overload with extra args
fn rotate(img: Image, angle: Float64, scale: Float64 = 1.0,
          center_x: Float64 = -1.0, center_y: Float64 = -1.0,
          border: Int = 0) -> Image:
    return rotate_basic(img, angle)


# --- rotate in 90-degree steps ---
fn rotate90(img: Image, k: Int = 1) -> Image:
    var t = _norm_rot90(k)
    if t == 0:
        return img.copy()
    elif t == 1:
        # 90°: swap H/W
        return Image.new_hwc_u8(img.width(), img.height(), img.channels(), UInt8(0))
    elif t == 2:
        # 180°: keep H/W
        return Image.new_hwc_u8(img.height(), img.width(), img.channels(), UInt8(0))
    else:
        # t == 3 → 270°: swap H/W
        return Image.new_hwc_u8(img.width(), img.height(), img.channels(), UInt8(0))

# --- translate (shape-only, no pixel shift yet) ---
fn translate(img: Image, dx: Int, dy: Int) -> Image:
    return Image.new_hwc_u8(img.height(), img.width(), img.channels(), UInt8(0))

# --- flip (shape-only) ---
# axis: 0 = vertical, 1 = horizontal, -1 = both; shape unchanged
fn flip(img: Image, axis: Int = 0) -> Image:
    if axis == 0 or axis == 1 or axis == -1:
        return Image.new_hwc_u8(img.height(), img.width(), img.channels(), UInt8(0))
    # unknown mode → no-op same shape
    return Image.new_hwc_u8(img.height(), img.width(), img.channels(), UInt8(0))

# --- optional: affine (shape-only stub) ---
# Expects 2x3 row-major matrix: [a00, a01, a02, a10, a11, a12]
# Uncomment when you wire real resampler.
# from collections.list import List
# fn affine(img: Image, matrix: List[Float64]) -> Image:
#     if len(matrix) != 6:
#         return Image.new_hwc_u8(img.height(), img.width(), img.channels(), UInt8(0))
#     return Image.new_hwc_u8(img.height(), img.width(), img.channels(), UInt8(0))

 


@fieldwise_init("implicit")
struct RandomHorizontalFlip:
    var p: Float32
    var _rng_state: UInt64

    fn __init__(out self, p: Float32 = 0.5, seed: UInt64 = 88172645463325252):
        self.p = p
        self._rng_state = seed

    # Simple LCG RNG → Float32 in [0,1)
    fn _rand01(mut self) -> Float32:
        self._rng_state = self._rng_state * 6364136223846793005 + 1
        return Float32((self._rng_state >> 33) & 0xFFFFFFFF) / 4294967296.0

    fn __call__(mut self, src: UnsafePointer[Float32], c: Int, h: Int, w: Int) -> Pointer[Float32]:
        assert(c > 0 and h > 0 and w > 0, "RandomHorizontalFlip: invalid input shape")

        var flip = self._rand01() < self.p

        var dst_elems = c * h * w
        var dst = UnsafePointer[Float32].alloc(dst_elems)

        var sC = h * w
        var sH = w
        var sW = 1

        var ch: Int = 0
        while ch < c:
            var y: Int = 0
            while y < h:
                var x: Int = 0
                while x < w:
                    var src_x = if flip: (w - 1 - x) else: x
                    var src_idx = ch * sC + y * sH + src_x * sW
                    var dst_idx = ch * sC + y * sH + x * sW
                    dst[dst_idx] = src[src_idx]
                    x += 1
                y += 1
            ch += 1

        return dst

# Convenience wrapper
fn random_horizontal_flip(ptr: UnsafePointer[Float32], c: Int, h: Int, w: Int,
                          p: Float32 = 0.5, seed: UInt64 = 88172645463325252) -> UnsafePointer[Float32]:
    var op = RandomHorizontalFlip(p, seed)
    return op(ptr, c, h, w)



# ----------------------------------------
# RNG (LCG)
# ----------------------------------------
@fieldwise_init("implicit")
struct _LCG:
    var _state: UInt64

    fn __init__(out self, seed: UInt64):
        self._state = seed

    fn rand01(mut self) -> Float32:
        self._state = self._state * 6364136223846793005 + 1
        return Float32((self._state >> 33) & 0xFFFFFFFF) / 4294967296.0

    fn uniform(mut self, a: Float32, b: Float32) -> Float32:
        return a + (b - a) * self.rand01()

# ----------------------------------------
# Math helpers
# ----------------------------------------
fn _deg2rad(x_deg: Float64) -> Float64:
    return x_deg * 3.141592653589793 / 180.0

fn _clamp_i(x: Int, lo: Int, hi: Int) -> Int:
    if x < lo: return lo
    if x > hi: return hi
    return x

# ----------------------------------------
# Core rotation (nearest), keeps same H×W
# ----------------------------------------
fn _rotate_nearest_hwc_u8(src: Tensor, angle_deg: Float64) -> Tensor:
    assert(src.dtype() == DType.UInt8, "random_rotation: only UInt8 supported")
    var h = src.height()
    var w = src.width()
    var c = src.channels()
    assert(h > 0 and w > 0 and c > 0, "random_rotation: bad input shape")

    # Ensure packed source for simpler addressing
    var x = src
    if not (x.stride2() == 1 and x.stride1() == c and x.stride0() == w * c):
        x = src.copy_to_packed_hwc()

    var (s0_out, s1_out, s2_out) = packed_hwc_strides(h, w, c)
    var out_len = h * w * c
    var out_buf = UnsafePointer[UInt8].alloc(out_len)
    var out = Tensor(out_buf, out_len, h, w, c, s0_out, s1_out, s2_out, DType.UInt8)

    var s0 = x.stride0(); var s1 = x.stride1(); var s2 = x.stride2()
    var sp = x.data()

    # Center coordinates (treat pixels as at integer coords; center between pixels for even sizes)
    var cx = (Float64(w) - 1.0) * 0.5
    var cy = (Float64(h) - 1.0) * 0.5

    var theta = _deg2rad(angle_deg)
    var ct = cos(theta)
    var st = sin(theta)

    var oy:Int = 0
    while oy < h:
        var fy = Float64(oy)
        var ox:Int = 0
        while ox < w:
            var fx = Float64(ox)

            # Inverse mapping: (ox,oy) in output → (sx,sy) in input
            var dx = fx - cx
            var dy = fy - cy
            var sx_f =  dx * ct + dy * st + cx
            var sy_f = -dx * st + dy * ct + cy

            # Nearest neighbor
            var sx_i = Int(round(sx_f))
            var sy_i = Int(round(sy_f))

            var base_out = oy * s0_out + ox * s1_out
            if (sx_i >= 0 and sx_i < w and sy_i >= 0 and sy_i < h):
                var base_in = sy_i * s0 + sx_i * s1
                var ch:Int = 0
                while ch < c:
                    out_buf[base_out + ch * s2_out] = sp[base_in + ch * s2]
                    ch += 1
            else:
                # fill out-of-bounds with 0
                var ch2:Int = 0
                while ch2 < c:
                    out_buf[base_out + ch2 * s2_out] = 0
                    ch2 += 1
            ox += 1
        oy += 1

    return out.copy()

# ----------------------------------------
# Transform
# ----------------------------------------
@fieldwise_init("implicit")
struct RandomRotation:
    var degrees: Float64
    var p: Float32
    var _rng: _LCG

    fn __init__(out self, degrees: Float64, p: Float32 = 1.0, seed: UInt64 = 0xC0FFEE1234):
        assert(degrees >= 0.0, "RandomRotation: degrees must be non-negative")
        self.degrees = degrees
        self.p = p
        self._rng = _LCG(seed)

    fn __call__(mut self, x_in: Tensor) -> Tensor:
        assert(x_in.dtype() == DType.UInt8, "RandomRotation: only UInt8 (HWC) supported")
        # Maybe skip
        if self._rng.rand01() >= self.p:
            # Return a packed copy for consistency
            return x_in.copy_to_packed_hwc()

        # Sample angle in [-degrees, +degrees]
        var a = Float32(self.degrees)
        var angle = Float64(self._rng.uniform(-a, a))
        return _rotate_nearest_hwc_u8(x_in, angle)

# Convenience functional wrapper
fn random_rotation(x: Tensor, degrees: Float64, p: Float32 = 1.0, seed: UInt64 = 0xC0FFEE1234) -> Tensor:
    var op = RandomRotation(degrees, p, seed)
    return op(x)


# --- crop ROI ---------------------------------------------------------------
# OpenCV-style crop: (y, x, h, w). Clamps to image bounds and returns a view/copy as implemented by Image.roi().
fn crop(img: Image, y: Int, x: Int, h: Int, w: Int) -> Image:
    return img.copy().roi(y, x, h, w)



# Overload: rotate90(img, clockwise=True|False)
fn rotate90(img: Image, clockwise: Bool) -> Image:
    var k = 0
    if clockwise:
        k = 1      # 90° clockwise
    else:
        k = 3      # 90° counter-clockwise
    return rotate90(img, k)



# ---------------- Border specification ----------------
# 0 = CONSTANT, 1 = REPLICATE, 2 = REFLECT
struct BorderSpec(Copyable, Movable):
    var _mode: Int
    var _cval: UInt8

    # Copy initializer required by Copyable
    fn __copyinit__(out self, other: Self):
        self._mode = other._mode
        self._cval = other._cval

    # Optional convenience init
    fn __init__(out self, mode: Int, cval: UInt8):
        self._mode = mode
        self._cval = cval

    # Getters
    fn mode(self) -> Int:
        return self._mode

    fn cval(self) -> UInt8:
        return self._cval

    # Static factories
    @staticmethod
    fn CONSTANT(val: UInt8 = UInt8(0)) -> BorderSpec:
        return BorderSpec(0, val)

    @staticmethod
    fn REPLICATE() -> BorderSpec:
        return BorderSpec(1, UInt8(0))

    @staticmethod
    fn REFLECT() -> BorderSpec:
        return BorderSpec(2, UInt8(0))


# Convenience factories returning BorderSpec (explicit names)
fn BORDER_CONSTANT_BS(val: UInt8 = 0) -> BorderSpec: return BorderSpec.CONSTANT(val)
fn BORDER_REPLICATE_BS() -> BorderSpec: return BorderSpec.REPLICATE()
fn BORDER_REFLECT_BS() -> BorderSpec: return BorderSpec.REFLECT()

# ---------------- Integer constants as getters ----------------
# These return Int codes useful for generic APIs or config.
# Keep codes aligned with BorderSpec: 0=CONSTANT, 1=REPLICATE, 2=REFLECT.

# Border codes
fn BORDER_CONSTANT() -> Int: return 0
fn BORDER_REPLICATE() -> Int: return 1
fn BORDER_REFLECT()  -> Int: return 2

# Interpolation codes
fn INTER_LINEAR() -> Int: return 0
fn INTER_AREA()   -> Int: return 1

# Morphology op codes
fn MORPH_OPEN()     -> Int: return 0
fn MORPH_CLOSE()    -> Int: return 1
fn MORPH_GRADIENT() -> Int: return 2

# Adaptive threshold method codes
fn ADAPTIVE_GAUSSIAN() -> Int: return 0

# Font codes
fn FONT_PLAIN()   -> Int: return 0
fn FONT_SIMPLEX() -> Int: return 1

# Convert Int border code to BorderSpec
fn to_border_spec(code: Int, val: UInt8 = 0) -> BorderSpec:
    if code == 0: return BorderSpec.CONSTANT(val)   # CONSTANT
    if code == 1: return BorderSpec.REPLICATE()     # REPLICATE
    # default: REFLECT
    return BorderSpec.REFLECT()


# Pixel fetch with border handling
fn _fetch_u8(img: Image, y: Int, x: Int, ch: Int, border: BorderSpec) -> UInt8:
    var h = img.height()
    var w = img.width()

    if 0 <= y and y < h and 0 <= x and x < w:
        return img.copy().at_u8(y, x, ch)

    # CONSTANT
    if border._mode == 0:
        return border._cval

    # REPLICATE
    if border._mode == 1:
        var yy = y
        var xx = x
        if yy < 0: yy = 0
        if yy >= h: yy = h - 1
        if xx < 0: xx = 0
        if xx >= w: xx = w - 1
        return img.copy().at_u8(yy, xx, ch)

    # REFLECT (OpenCV-like: reflect without repeating edge pixel)
    var yy2 = y
    var xx2 = x
    if h > 1:
        while yy2 < 0 or yy2 >= h:
            if yy2 < 0: yy2 = -yy2 - 1
            elif yy2 >= h: yy2 = 2 * h - yy2 - 1
    else:
        yy2 = 0
    if w > 1:
        while xx2 < 0 or xx2 >= w:
            if xx2 < 0: xx2 = -xx2 - 1
            elif xx2 >= w: xx2 = 2 * w - xx2 - 1
    else:
        xx2 = 0
    return img.copy().at_u8(yy2, xx2, ch)


fn translate(src: Image, dx: Int, dy: Int, border_code: Int) -> Image:
    return translate(src, dx, dy, to_border_spec(border_code, 0))
# Translate image by (dx, dy). Positive dx shifts right, positive dy shifts down.
# border: how to sample outside source bounds.
fn translate(src: Image, dx: Int, dy: Int, border: BorderSpec = BorderSpec.CONSTANT(0)) -> Image:
    var h = src.height()
    var w = src.width()
    var c = src.channels()
    var dst = vision.full(h, w, c, 0)


    var y = 0
    while y < h:
        var x = 0
        # Inverse mapping: dst(y,x) <- src(y - dy, x - dx)
        var sy = y - dy
        while x < w:
            var sx = x - dx
            var ch = 0
            while ch < c:
                var v = _fetch_u8(src, sy, sx, ch, border)
                dst.set_u8(y, x, ch, v)
                ch += 1
            x += 1
        y += 1
    return dst.copy()


 
 

# ---------------- Scalars / small helpers ----------------

fn _abs(x: Float64) -> Float64:
    if x >= 0.0: return x
    return -x
  

# ---------------- 3x3 helpers (row-major) ----------------
fn _det3(m: List[Float64]) -> Float64:
    # m length=9: [a b c; d e f; g h i]
    return (
        m[0]*(m[4]*m[8] - m[5]*m[7])
      - m[1]*(m[3]*m[8] - m[5]*m[6])
      + m[2]*(m[3]*m[7] - m[4]*m[6])
    )

fn _inv3(m: List[Float64]) -> (Bool, List[Float64]):
    var det = _det3(m)
    if det == 0.0:
        return (False, List[Float64]())
    var inv = List[Float64]()
    inv.reserve(9)
    var a = m[0]; var b = m[1]; var c = m[2]
    var d = m[3]; var e = m[4]; var f = m[5]
    var g = m[6]; var h = m[7]; var i = m[8]
    # adjugate / det
    inv.append( (e*i - f*h) / det )
    inv.append( (c*h - b*i) / det )
    inv.append( (b*f - c*e) / det )
    inv.append( (f*g - d*i) / det )
    inv.append( (a*i - c*g) / det )
    inv.append( (c*d - a*f) / det )
    inv.append( (d*h - e*g) / det )
    inv.append( (b*g - a*h) / det )
    inv.append( (a*e - b*d) / det )
    return (True, inv.copy())

fn _mul3x3_vec(m: List[Float64], x: Float64, y: Float64, w: Float64) -> (Float64, Float64, Float64):
    var u = m[0]*x + m[1]*y + m[2]*w
    var v = m[3]*x + m[4]*y + m[5]*w
    var z = m[6]*x + m[7]*y + m[8]*w
    return (u, v, z)

# ---------------- Affine from 3 point pairs ----------------
# Solve two 3x3 systems built from rows [xi yi 1]
fn _affine_from_3pts(src: List[(Float64, Float64)], dst: List[(Float64, Float64)]) -> (Bool, List[Float64]):
    var M = List[Float64]()
    M.reserve(9)
    var X = List[Float64](); X.reserve(3)
    var Y = List[Float64](); Y.reserve(3)

    var i = 0
    while i < 3:
        var (x, y) = src[i]
        var (Xv, Yv) = dst[i]
        M.append(x); M.append(y); M.append(1.0)
        X.append(Xv); Y.append(Yv)
        i += 1

    var req = _inv3(M)
    var ok=req[0]
    var Minv=req[1].copy()
    if not ok:
        return (False, List[Float64]())

    fn _mul3x3_vec3(m: List[Float64], v: List[Float64]) -> (Float64, Float64, Float64):
        var r0 = m[0]*v[0] + m[1]*v[1] + m[2]*v[2]
        var r1 = m[3]*v[0] + m[4]*v[1] + m[5]*v[2]
        var r2 = m[6]*v[0] + m[7]*v[1] + m[8]*v[2]
        return (r0, r1, r2)

    var (a, b, c) = _mul3x3_vec3(Minv, X)
    var (d, e, f) = _mul3x3_vec3(Minv, Y)

    var A = List[Float64]()
    A.append(a); A.append(b); A.append(c)
    A.append(d); A.append(e); A.append(f)
    A.append(0.0); A.append(0.0); A.append(1.0)
    return (True, A.copy())

# ---------------- Perspective (homography) from 4 point pairs ----------------
# Solve 8 unknowns (h0..h7) with Gaussian elimination; set h8 = 1
fn _solve_homography(src: List[(Float64, Float64)], dst: List[(Float64, Float64)]) -> (Bool, List[Float64]):
    var A = List[List[Float64]]()
    var b = List[Float64]()
    var i = 0
    while i < 4:
        var (x, y) = src[i]
        var (X, Y) = dst[i]
        var row1 = List[Float64](); row1.reserve(8)
        row1.append(x); row1.append(y); row1.append(1.0); row1.append(0.0); row1.append(0.0); row1.append(0.0); row1.append(-x*X); row1.append(-y*X)
        A.append(row1.copy()); b.append(X)
        var row2 = List[Float64](); row2.reserve(8)
        row2.append(0.0); row2.append(0.0); row2.append(0.0); row2.append(x); row2.append(y); row2.append(1.0); row2.append(-x*Y); row2.append(-y*Y)
        A.append(row2.copy()); b.append(Y)
        i += 1

    var n = 8
    var col = 0
    while col < n:
        # pivot
        var piv = col
        var r = col + 1
        while r < n:
            if _abs(A[r][col]) > _abs(A[piv][col]):
                piv = r
            r += 1
        if _abs(A[piv][col]) < 1e-12:
            return (False, List[Float64]())
        # swap rows
        if piv != col:
            var tmp = A[piv].copy(); A[piv] = A[col].copy(); A[col] = tmp.copy()
            var tb = b[piv]; b[piv] = b[col]; b[col] = tb
        # normalize
        var lead = A[col][col]
        var j = col
        while j < n:
            A[col][j] = A[col][j] / lead
            j += 1
        b[col] = b[col] / lead
        # eliminate
        var i2 = 0
        while i2 < n:
            if i2 != col:
                var factor = A[i2][col]
                if factor != 0.0:
                    var j2 = col
                    while j2 < n:
                        A[i2][j2] = A[i2][j2] - factor * A[col][j2]
                        j2 += 1
                    b[i2] = b[i2] - factor * b[col]
            i2 += 1
        col += 1

    var H = List[Float64]()
    H.reserve(9)
    var k = 0
    while k < n:
        H.append(b[k])
        k += 1
    H.append(1.0)
    return (True, H.copy())

# ---------------- Warp kernels (nearest) ----------------
fn _warp_affine(src_img: Image, A_inv: List[Float64], border: BorderSpec) -> Image:
    var h = src_img.height()
    var w = src_img.width()
    var c = src_img.channels()
    var out = zeros(h, w, c, UInt8(0))

    var y = 0
    while y < h:
        var x = 0
        while x < w:
            var (ux, uy, uz) = _mul3x3_vec(A_inv, Float64(x), Float64(y), 1.0)
            var sx = Int(ux + 0.5)  # uz == 1 for affine
            var sy = Int(uy + 0.5)
            var ch = 0
            while ch < c:
                var v = _fetch_u8(src_img, sy, sx, ch, border)
                out.set_u8(y, x, ch, v)
                ch += 1
            x += 1
        y += 1
    return out.copy()

fn _warp_perspective(src_img: Image, H_inv: List[Float64], border: BorderSpec) -> Image:
    var h = src_img.height()
    var w = src_img.width()
    var c = src_img.channels()
    var out = zeros(h, w, c, UInt8(0))

    var y = 0
    while y < h:
        var x = 0
        while x < w:
            var (ux, uy, uz) = _mul3x3_vec(H_inv, Float64(x), Float64(y), 1.0)
            if uz == 0.0:
                var ch0 = 0
                while ch0 < c:
                    out.set_u8(y, x, ch0, 0)
                    ch0 += 1
            else:
                var sx_f = ux / uz
                var sy_f = uy / uz
                var sx = Int(sx_f + 0.5)
                var sy = Int(sy_f + 0.5)
                var ch = 0
                while ch < c:
                    var v = _fetch_u8(src_img, sy, sx, ch, border)
                    out.set_u8(y, x, ch, v)
                    ch += 1
            x += 1
        y += 1
    return out.copy()

# ---------------- Tuple → List adapters (for named-arg API) ----------------
fn _to_list3_f64(t: ((Float64, Float64), (Float64, Float64), (Float64, Float64))) -> List[(Float64, Float64)]:
    var L = List[(Float64, Float64)]()
    L.append(t[0]); L.append(t[1]); L.append(t[2])
    return L.copy()

fn _to_list4_f64(t: ((Float64, Float64), (Float64, Float64), (Float64, Float64), (Float64, Float64))) -> List[(Float64, Float64)]:
    var L = List[(Float64, Float64)]()
    L.append(t[0]); L.append(t[1]); L.append(t[2]); L.append(t[3])
    return L.copy()

fn _to_list3_f32(t: ((Float32, Float32), (Float32, Float32), (Float32, Float32))) -> List[(Float64, Float64)]:
    var L = List[(Float64, Float64)]()
    L.append((Float64(t[0][0]), Float64(t[0][1])))
    L.append((Float64(t[1][0]), Float64(t[1][1])))
    L.append((Float64(t[2][0]), Float64(t[2][1])))
    return L.copy()

fn _to_list4_f32(t: ((Float32, Float32), (Float32, Float32), (Float32, Float32), (Float32, Float32))) -> List[(Float64, Float64)]:
    var L = List[(Float64, Float64)]()
    L.append((Float64(t[0][0]), Float64(t[0][1])))
    L.append((Float64(t[1][0]), Float64(t[1][1])))
    L.append((Float64(t[2][0]), Float64(t[2][1])))
    L.append((Float64(t[3][0]), Float64(t[3][1])))
    return L.copy()

# ---------------- Public API: affine (List + Tuple overloads) ----------------
from collections.list import List

fn affine(
    img: Image,
    src: List[(Float64, Float64)],
    dst: List[(Float64, Float64)],
    border: BorderSpec = BorderSpec.CONSTANT(0)
) -> Image:
    # Require exactly three source and three destination points
    if len(src) != 3 or len(dst) != 3:
        return img.copy()

    # Compute affine coefficients from 3 point pairs
    var tmpA = _affine_from_3pts(src, dst)  # expected (Bool, List[Float64]) with len=6
    var okA = tmpA[0]
    var A = tmpA[1].copy()
    if not okA or len(A) != 6:
        return img.copy()

    # Invert affine 2x3 matrix
    var tmpInv = _inv3(A)  # expected (Bool, List[Float64]) with len=6
    var okInv = tmpInv[0]
    var A_inv = tmpInv[1].copy()
    if not okInv or len(A_inv) != 6:
        return img.copy()

    # Apply warp with backward mapping
    return _warp_affine(img, A_inv, border)

# Int border overload (named 'border' as Int)
fn affine(
    img: Image,
    src: List[(Float64, Float64)],
    dst: List[(Float64, Float64)],
    border: Int
) -> Image:
    return affine(img, src, dst, to_border_spec(border, 0))

# Tuple-of-tuples overloads
fn affine(
    img: Image,
    src: ((Float64, Float64), (Float64, Float64), (Float64, Float64)),
    dst: ((Float64, Float64), (Float64, Float64), (Float64, Float64)),
    border: BorderSpec = BorderSpec.CONSTANT(0)
) -> Image:
    return affine(img, _to_list3_f64(src), _to_list3_f64(dst), border)

fn affine(
    img: Image,
    src: ((Float64, Float64), (Float64, Float64), (Float64, Float64)),
    dst: ((Float64, Float64), (Float64, Float64), (Float64, Float64)),
    border: Int
) -> Image:
    return affine(img, _to_list3_f64(src), _to_list3_f64(dst), to_border_spec(border, 0))

fn affine(
    img: Image,
    src: ((Float32, Float32), (Float32, Float32), (Float32, Float32)),
    dst: ((Float32, Float32), (Float32, Float32), (Float32, Float32)),
    border: BorderSpec = BorderSpec.CONSTANT(0)
) -> Image:
    return affine(img, _to_list3_f32(src), _to_list3_f32(dst), border)

fn affine(
    img: Image,
    src: ((Float32, Float32), (Float32, Float32), (Float32, Float32)),
    dst: ((Float32, Float32), (Float32, Float32), (Float32, Float32)),
    border: Int
) -> Image:
    return affine(img, _to_list3_f32(src), _to_list3_f32(dst), to_border_spec(border, 0))

# ---------------- Public API: perspective (List + Tuple overloads) ----------------

fn perspective(
    img: Image,
    src: List[(Float64, Float64)],
    dst: List[(Float64, Float64)],
    border: BorderSpec = BorderSpec.CONSTANT(0)
) -> Image:
    # Require exactly four point pairs; otherwise no-op
    if len(src) != 4 or len(dst) != 4:
        return img.copy()

    # Solve homography H (3x3 row-major flattened, len==9)
    var tmpH = _solve_homography(src, dst)   # expected (Bool, List[Float64])
    var okH = tmpH[0]
    var H = tmpH[1].copy()
    if not okH:
        return img.copy()
    if len(H) != 9:
        return img.copy()

    # Invert H for backward mapping
    var tmpInv = _inv3(H)                    # expected (Bool, List[Float64])
    var okInv = tmpInv[0]
    var H_inv = tmpInv[1].copy()
    if not okInv:
        return img.copy()
    if len(H_inv) != 9:
        return img.copy()

    # Warp using the inverse homography (backward warping)
    return _warp_perspective(img, H_inv, border)


fn perspective(
    img: Image,
    src: List[(Float64, Float64)],
    dst: List[(Float64, Float64)],
    border: Int
) -> Image:
    return perspective(img, src, dst, to_border_spec(border, 0))

fn perspective(
    img: Image,
    src: ((Float64, Float64), (Float64, Float64), (Float64, Float64), (Float64, Float64)),
    dst: ((Float64, Float64), (Float64, Float64), (Float64, Float64), (Float64, Float64)),
    border: BorderSpec = BorderSpec.CONSTANT(0)
) -> Image:
    return perspective(img, _to_list4_f64(src), _to_list4_f64(dst), border)

fn perspective(
    img: Image,
    src: ((Float64, Float64), (Float64, Float64), (Float64, Float64), (Float64, Float64)),
    dst: ((Float64, Float64), (Float64, Float64), (Float64, Float64), (Float64, Float64)),
    border: Int
) -> Image:
    return perspective(img, _to_list4_f64(src), _to_list4_f64(dst), to_border_spec(border, 0))

fn perspective(
    img: Image,
    src: ((Float32, Float32), (Float32, Float32), (Float32, Float32), (Float32, Float32)),
    dst: ((Float32, Float32), (Float32, Float32), (Float32, Float32), (Float32, Float32)),
    border: BorderSpec = BorderSpec.CONSTANT(0)
) -> Image:
    return perspective(img, _to_list4_f32(src), _to_list4_f32(dst), border)

fn perspective(
    img: Image,
    src: ((Float32, Float32), (Float32, Float32), (Float32, Float32), (Float32, Float32)),
    dst: ((Float32, Float32), (Float32, Float32), (Float32, Float32), (Float32, Float32)),
    border: Int
) -> Image:
    return perspective(img, _to_list4_f32(src), _to_list4_f32(dst), to_border_spec(border, 0))


# Correct reflect101 mapping into [0, n-1]
fn _reflect_coord(n: Int, s: Int) -> Int:
    if n <= 1:
        return 0
    var period = 2 * n - 2
    var t = s % period
    if t < 0:
        t = t + period
    if t < n:
        return t
    return period - t

# Helper: pick constant value for a channel (fallback to border._cval if list is empty).
fn _const_ch(val: List[UInt8], ch: Int, fallback: UInt8) -> UInt8:
    if len(val) == 0:
        return fallback
    var idx = ch
    if idx >= len(val):
        idx = len(val) - 1
    if idx < 0:
        idx = 0
    return val[idx]

# BorderSpec contract assumed:
#   border._mode: 0=CONSTANT, 1=REPLICATE, else=REFLECT
#   border._cval: UInt8 (used when CONSTANT)
fn copy_make_border(
    src: Image,
    top: Int, bottom: Int, left: Int, right: Int,
    border: BorderSpec = BorderSpec.CONSTANT(0),
    value: List[UInt8] = []
) -> Image:
    # Ensure readable packed HWC/UInt8
    var base = src.ensure_packed_hwc_u8(True)

    var h = base.height()
    var w = base.width()
    var c = base.channels()
    var H = h + top + bottom
    var W = w + left + right
    if H <= 0 or W <= 0 or c <= 0:
        return Image.new_hwc_u8(1, 1, 3, UInt8(0))

    # Allocate destination (zero-filled)
    var dst = Image.new_hwc_u8(H, W, c, UInt8(0))

    var y = 0
    while y < H:
        var x = 0
        while x < W:
            var sy = y - top
            var sx = x - left

            if sy >= 0 and sy < h and sx >= 0 and sx < w:
                # Copy from source
                var ch = 0
                while ch < c:
                    var pv = base.get_u8(sy, sx, ch)
                    dst.set_u8(y, x, ch, pv)
                    ch += 1
            else:
                # Outside source: handle by border mode
                if border._mode == 0:
                    # CONSTANT
                    var ch2 = 0
                    while ch2 < c:
                        var cv = _const_ch(value, ch2, border._cval)
                        dst.set_u8(y, x, ch2, cv)
                        ch2 += 1
                elif border._mode == 1:
                    # REPLICATE
                    var ry = sy
                    if ry < 0:
                        ry = 0
                    if ry >= h:
                        ry = h - 1
                    var rx = sx
                    if rx < 0:
                        rx = 0
                    if rx >= w:
                        rx = w - 1
                    var ch3 = 0
                    while ch3 < c:
                        var pv2 = base.get_u8(ry, rx, ch3)
                        dst.set_u8(y, x, ch3, pv2)
                        ch3 += 1
                else:
                    # REFLECT (no edge repeat)
                    var ry2 = _reflect_coord(h, sy)
                    var rx2 = _reflect_coord(w, sx)
                    var ch4 = 0
                    while ch4 < c:
                        var pv3 = base.get_u8(ry2, rx2, ch4)
                        dst.set_u8(y, x, ch4, pv3)
                        ch4 += 1
            x += 1
        y += 1

    return dst.copy()

# Int border overloads (so calls like BORDER_CONSTANT() work) ------------

fn copy_make_border(
    src: Image,
    top: Int, bottom: Int, left: Int, right: Int,
    border: Int
) -> Image:
    return copy_make_border(src, top, bottom, left, right, to_border_spec(border, 0))

fn copy_make_border(
    src: Image,
    top: Int, bottom: Int, left: Int, right: Int,
    border: Int,
    value: List[UInt8]
) -> Image:
    return copy_make_border(src, top, bottom, left, right, to_border_spec(border, 0), value)


fn deg2rad(deg: Float64) -> Float64:
    return deg * 3.141592653589793 / 180.0


