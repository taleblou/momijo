# MIT License
# SPDX-License-Identifier: MIT
# Project: momijo.vision
# File: src/momijo/vision/transforms/array.mojo
# Description: Array helpers for Image (Tensor + ImageMeta backed)

from momijo.vision.image import Image, ImageMeta, ColorSpace, Layout
from momijo.vision.tensor import Tensor, packed_hwc_strides
from momijo.vision.dtypes import DType

# -------------------------------------------------------------------
# Internal helpers
# -------------------------------------------------------------------

fn _default_meta_hwc_u8() -> ImageMeta:
    var m = ImageMeta()
    m = m.with_colorspace(ColorSpace.SRGB())
    return m.copy()

fn _alloc_tensor_u8(h_in: Int, w_in: Int, c_in: Int, value: UInt8) -> Tensor:
    # Replace assert with a safe guard
    var h = h_in
    var w = w_in
    var c = c_in
    if h <= 0:
        h = 1
    if w <= 0:
        w = 1
    if c <= 0:
        c = 1

    var (s0, s1, s2) = packed_hwc_strides(h, w, c)
    var n = h * w * c
    var buf = UnsafePointer[UInt8].alloc(n)
    var i = 0
    while i < n:
        buf[i] = value
        i = i + 1
    return Tensor(buf, n, h, w, c, s0, s1, s2, DType.UInt8())

fn _u8_from_f64(x: Float64) -> UInt8:
    var v = x
    if v < 0.0:
        v = 0.0
    if v > 255.0:
        v = 255.0
    return UInt8(Int(v))

# -------------------------------------------------------------------
# Core allocators
# -------------------------------------------------------------------

# OpenCV-style convenience: zeros((h,w,c), dtype) and zeros(h,w,c,dtype)
fn zeros(shape: (Int, Int, Int), dtype: UInt8) -> Image:
    return full(shape, UInt8(0))

fn zeros(h: Int, w: Int, c: Int, dtype: UInt8) -> Image:
    return full((h, w, c), UInt8(0))

# -------------------------------------------------------------------
# Fill helpers
# -------------------------------------------------------------------

# Fill with scalar (UInt8)
# fn fill(img: Image, scalar: UInt8) -> Image:
#     var v = scalar
#     return full_like(img, v)

# Adapter: accept List[UInt8] and forward to List[Int]
fn fill(img: Image, color_u8: List[UInt8]) -> Image:
    var color_i = List[Int]()
    var i = 0
    while i < len(color_u8):
        color_i.append(Int(color_u8[i]))
        i += 1
    return fill(img, color_i)


# Main: List[Int] -> write per-channel into packed HWC UInt8 buffer
fn fill(img: Image, color_i: List[Int]) -> Image:
    # Normalize to UInt8
    var col_u8 = List[UInt8]()
    var k = 0
    while k < len(color_i):
        col_u8.append(UInt8(color_i[k] & 255))
        k += 1

    # Handle empty / single-value cases
    if len(col_u8) == 0:
        return full_like(img, UInt8(0))
    if len(col_u8) == 1:
        return full_like(img, col_u8[0])

    # Ensure packed HWC UInt8 before writing raw bytes
    var out = img.copy()

    var h = out.height()
    var w = out.width()
    var c = out.channels()
    if h <= 0 or w <= 0 or c <= 0:
        return out.copy()

    var t = out.tensor()
    var s0 = t.stride0()            # bytes per row = w * c
    var s1 = t.stride1()            # bytes per pixel = c
    var s2 = t.stride2()            # must be 1 for packed HWC
    var base = t.data()             # UnsafePointer[UInt8]

    # Safety: fallback if somehow not packed (shouldn't happen with ensure_)
    if s2 != 1:
        # Slow path using set_u8 (still correct)
        var y = 0
        while y < h:
            var x = 0
            while x < w:
                var ch = 0
                while ch < c:
                    var idx = ch
                    if idx >= len(col_u8): idx = len(col_u8) - 1
                    out.set_u8(y, x, ch, col_u8[idx])
                    ch += 1
                x += 1
            y += 1
        return out.copy()

    # Fast path: raw pointer writes (packed HWC)
    var y = 0
    while y < h:
        var row_off = y * s0
        var x = 0
        while x < w:
            var px = row_off + x * s1
            var ch = 0
            while ch < c:
                var idx = ch
                if idx >= len(col_u8): idx = len(col_u8) - 1
                base[px + ch] = col_u8[idx]
                ch += 1
            x += 1
        y += 1

    return out.copy()


# -------------------------------------------------------------------
# Arithmetic / utilities
# -------------------------------------------------------------------

# Weighted sum: a*alpha + b*beta + gamma (per-pixel when contiguous HWC/u8)
fn add_weighted(a: Image, alpha: Float64, b: Image, beta: Float64, gamma: Float64) -> Image:
    var aa = a.ensure_packed_hwc_u8(True)
    var bb = b.ensure_packed_hwc_u8(True)

    # Shape guard
    if aa.height() != bb.height() or aa.width() != bb.width() or aa.channels() != bb.channels():
        return a.copy()  # fallback

    var h = aa.height(); var w = aa.width(); var c = aa.channels()
    var (s0, s1, s2) = packed_hwc_strides(h, w, c)
    var n = h * w * c

    var ap = aa.tensor().data()
    var bp = bb.tensor().data()

    var out_buf = UnsafePointer[UInt8].alloc(n)
    var i = 0
    while i < n:
        var av = Float64(Int(ap[i]))
        var bv = Float64(Int(bp[i]))
        out_buf[i] = _u8_from_f64(alpha * av + beta * bv + gamma)
        i = i + 1

    var t = Tensor(out_buf, n, h, w, c, s0, s1, s2, DType.UInt8())
    return Image(aa.meta(),t)

# Absolute value for u8 is identity; implement as a byte-for-byte copy.
fn abs_u8(img: Image) -> Image:
    var ii = img.ensure_packed_hwc_u8(True)
    var h = ii.height(); var w = ii.width(); var c = ii.channels()
    var (s0, s1, s2) = packed_hwc_strides(h, w, c)
    var n = h * w * c
    var src = ii.tensor().data()

    var dst = UnsafePointer[UInt8].alloc(n)
    var i = 0
    while i < n:
        dst[i] = src[i]
        i = i + 1

    var t = Tensor(dst, n, h, w, c, s0, s1, s2, DType.UInt8())
    return Image(ii.meta(),t)

# Copy src into dst (safe semantics): replace dst with a clone of src.
fn copy_to(src: Image, mut dst: Image):
    dst = src.clone()

# Produce indices [0, 1, ..., k-1]
fn top_k_indices(k: Int) -> List[Int]:
    var out = List[Int]()
    var i = 0
    while i < k:
        out.append(i)
        i = i + 1
    return out.copy()

# -------------------------------------------------------------------
# Fill allocators
# -------------------------------------------------------------------

fn full(shape: (Int, Int, Int), value: UInt8) -> Image:
    var h = shape[0]; var w = shape[1]; var c = shape[2]
    var t = _alloc_tensor_u8(h, w, c, value)
    var m = _default_meta_hwc_u8()
    return Image(m.copy(), t.copy())

fn full_like(img: Image, value: UInt8) -> Image:
    var h = img.height(); var w = img.width(); var c = img.channels()
    var t = _alloc_tensor_u8(h, w, c, value)
    return Image(img.meta(),t.copy())

fn full(h: Int, w: Int, c: Int, value: UInt8) -> Image:
    return full((h, w, c), value)

# -------------------------------------------------------------------
# Feature & matching stubs (no alias; use tuples and nested lists)
# -------------------------------------------------------------------

# Keypoints as tuples: (x: Int, y: Int, response: Float32)
# Descriptors as 2D list of bytes: List[List[UInt8]]
# Matches as tuples: (query_idx: Int, train_idx: Int, distance: Int)

fn _descriptor_empty(n: Int, length: Int) -> List[List[UInt8]]:
    var d = List[List[UInt8]]()
    var i = 0
    while i < n:
        var row = List[UInt8]()
        var j = 0
        while j < length:
            row.append(0)
            j = j + 1
        d.append(row.copy())
        i = i + 1
    return d.copy()

fn _descriptor_set_zero(mut d: List[List[UInt8]], idx: Int):
    if 0 <= idx and idx < len(d):
        var j = 0
        var L = len(d[idx])
        while j < L:
            d[idx][j] = 0
            j = j + 1

# Public helpers expected by user code
fn valid_descriptors(des: List[List[UInt8]]) -> Bool:
    return not _descriptor_is_null(des)

fn len_keypoints(kps: List[(Int, Int, Float32)]) -> Int:
    return len(kps)

# ORB-like placeholder returning empty results
fn orb_detect_and_compute(img: Image, n_features: Int = 500) -> (List[(Int, Int, Float32)], List[List[UInt8]]):
    var kps = List[(Int, Int, Float32)]()
    var desc = _descriptor_empty(0, 32)  # typical ORB length
    return (kps.copy(), desc.copy())

# ---------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------

fn _descriptor_is_null(d: List[List[UInt8]]) -> Bool:
    # Null if empty or any row is empty
    if len(d) == 0:
        return True
    if len(d[0]) == 0:
        return True
    return False

fn _row_len_uniform(d: List[List[UInt8]]) -> Int:
    # Returns common row length if uniform; otherwise returns -1
    var n = len(d)
    var L = len(d[0])
    var i = 1
    while i < n:
        if len(d[i]) != L:
            return -1
        i = i + 1
    return L

fn _popcount8(x: Int) -> Int:
    # Kernighan’s popcount for 0..255 stored in Int
    var v = x
    var c = 0
    while v != 0:
        v = v & (v - 1)
        c = c + 1
    return c

fn _hamming_row(x: List[UInt8], y: List[UInt8]) -> Int:
    # Hamming distance between two equal-length byte rows
    var Lx = len(x)
    var Ly = len(y)
    var L = Lx
    if Ly < L:
        L = Ly

    var dist = 0
    var k = 0
    while k < L:
        # XOR as Int to avoid UInt8 bit-op pitfalls; limit to 0..255
        var xi = Int(x[k])
        var yi = Int(y[k])
        var xr = (xi ^ yi) & 255
        dist = dist + _popcount8(xr)
        k = k + 1

    # If lengths differ, count remaining bits of the extra tail
    if Lx > L:
        var t = L
        while t < Lx:
            dist = dist + _popcount8(Int(x[t]) & 255)
            t = t + 1
    else:
        if Ly > L:
            var t2 = L
            while t2 < Ly:
                dist = dist + _popcount8(Int(y[t2]) & 255)
                t2 = t2 + 1

    return dist

# ---------------------------------------------------------------
# Public API
# ---------------------------------------------------------------

# Returns list of (idx_a, idx_b, distance), sorted by distance asc.
fn bf_match_hamming(
    a: List[List[UInt8]],
    b: List[List[UInt8]],
    cross_check: Bool = True
) -> List[(Int, Int, Int)]:
    var out = List[(Int, Int, Int)]()
    if _descriptor_is_null(a) or _descriptor_is_null(b):
        return out.copy()

    # Ensure rows are uniform length (typical ORB=32 bytes)
    var La = _row_len_uniform(a)
    var Lb = _row_len_uniform(b)
    if (La < 0) or (Lb < 0):
        # Non-uniform rows; return empty for safety
        return out.copy()

    var na = len(a)
    var nb = len(b)

    # Best matches A->B
    var best_j = List[Int]()
    var best_d = List[Int]()
    # reserve() may be unsupported; simple push will grow as needed
    # best_j.reserve(na); best_d.reserve(na)

    var i = 0
    while i < na:
        var j = 0
        var bj = -1
        var bd = 1000000000  # large sentinel
        while j < nb:
            var d = _hamming_row(a[i], b[j])
            if d < bd:
                bd = d
                bj = j
            j = j + 1
        best_j.append(bj)
        best_d.append(bd)
        i = i + 1

    if not cross_check:
        # One-way matches (A->B)
        var k = 0
        while k < na:
            if best_j[k] >= 0:
                out.append((k, best_j[k], best_d[k]))
            k = k + 1
    else:
        # Cross-check: also compute B->A best and keep only mutual bests
        var best_i = List[Int]()

        var j2 = 0
        while j2 < nb:
            var i2 = 0
            var bi = -1
            var bd2 = 1000000000
            while i2 < na:
                var d2 = _hamming_row(b[j2], a[i2])
                if d2 < bd2:
                    bd2 = d2
                    bi = i2
                i2 = i2 + 1
            best_i.append(bi)
            j2 = j2 + 1

        var p = 0
        while p < na:
            var q = best_j[p]
            if (q >= 0) and (best_i[q] == p):
                out.append((p, q, best_d[p]))
            p = p + 1

    # In-place insertion sort by distance ascending
    var nout = len(out)
    var s = 1
    while s < nout:
        var key = out[s]
        var dkey = key[2]
        var t = s - 1
        while (t >= 0) and (out[t][2] > dkey):
            var tmp0 = out[t][0]
            var tmp1 = out[t][1]
            var tmp2 = out[t][2]
            out[t + 1] = (tmp0, tmp1, tmp2)
            t = t - 1
        out[t + 1] = key
        s = s + 1

    return out.copy()

fn top_k_matches(matches: List[(Int, Int, Int)], k: Int) -> List[(Int, Int, Int)]:
    var n = len(matches)
    if k >= n:
        return matches.copy()
    var out = List[(Int, Int, Int)]()
    var i = 0
    while i < k:
        out.append(matches[i])
        i = i + 1
    return out.copy()
