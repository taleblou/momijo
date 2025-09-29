# =========================
# List helpers (shared)
# =========================

fn list_select[T: Copyable & Movable](xs: List[T], dim: Int, index: Int) -> List[T]:
    if len(xs) == 0:
        return List[T]()
    var idx = index
    if idx < 0: idx = 0
    if idx >= len(xs): idx = len(xs) - 1
    var out = List[T]()
    out.append(xs[idx])
    return out

fn list_slice[T: Copyable & Movable](xs: List[T], dim: Int, start: Int, end: Int, step: Int = 1) -> List[T]:
    var n = len(xs)
    if n == 0 or step == 0:
        return List[T]()
    var s = start
    var e = end
    if s < 0: s = 0
    if e > n: e = n
    if e < s:
        return List[T]()
    var out = List[T]()
    var i = s
    while i < e:
        out.append(xs[i])
        i += step
    return out


# =========================
# Tensor helpers (shared)
# =========================

fn compute_strides_shared(shape: List[Int]) -> List[Int]:
    var r = len(shape)
    var strides = List[Int]()
    strides.resize(r)
    var acc = 1
    var i = r - 1
    while i >= 0:
        strides[i] = acc
        acc = acc * shape[i]
        i -= 1
    return strides

fn clamp_idx_shared(idx: Int, lo: Int, hi: Int) -> Int:
    var x = idx
    if x < lo: x = lo
    if x >= hi: x = hi - 1
    return x

fn norm_range_shared(start: Int, stop: Int, step: Int, dim: Int) -> (Int, Int, Int):
    var s = step
    if s == 0: s = 1
    var a = start
    var b = stop
    if a < 0: a = dim + a
    if b <= 0: b = dim + b
    if a < 0: a = 0
    if b < 0: b = 0
    if a > dim: a = dim
    if b > dim: b = dim
    return (a, b, s)

fn count_range(a: Int, b: Int, s: Int) -> Int:
    if s <= 0 or a >= b:
        return 0
    var n = 0
    var i = a
    while i < b:
        n += 1
        i += s
    return n


# =========================
# FloatTensor slicing (1D/2D/3D/4D) + utils
# Requires: FloatTensor has fields _shape: List[Int], _data: List[Float64]
# =========================

fn slice1d_float(t: FloatTensor, start: Int, stop: Int, step: Int = 1) -> FloatTensor:
    var shp = t._shape
    if len(shp) == 0:
        return FloatTensor([0], 0.0)
    var n = shp[0]
    var a_b_s = norm_range_shared(start, stop, step, n)
    var a = a_b_s[0]; var b = a_b_s[1]; var s = a_b_s[2]
    if s <= 0 or a >= b:
        return FloatTensor([0], 0.0)

    var cnt = count_range(a, b, s)
    var out = FloatTensor([cnt], 0.0)
    var strides = compute_strides_shared(shp)
    var dst = 0
    var i = a
    while i < b:
        var src_lin = i * strides[0]
        out._data[dst] = t._data[src_lin]
        dst += 1
        i += s
    return out

fn slice2d_float(t: FloatTensor, r0: Int, r1: Int, c0: Int, c1: Int, rs: Int = 1, cs: Int = 1) -> FloatTensor:
    var shp = t._shape
    if len(shp) < 2:
        return FloatTensor([0, 0], 0.0)

    var R = shp[0]; var C = shp[1]
    var rr = norm_range_shared(r0, r1, rs, R)
    var cc = norm_range_shared(c0, c1, cs, C)
    if rr[2] <= 0 or cc[2] <= 0 or rr[0] >= rr[1] or cc[0] >= cc[1]:
        return FloatTensor([0, 0], 0.0)

    var nr = count_range(rr[0], rr[1], rr[2])
    var nc = count_range(cc[0], cc[1], cc[2])

    var out = FloatTensor([nr, nc], 0.0)
    var strides = compute_strides_shared(shp)   # [C, 1] row-major
    var ri = 0
    var r = rr[0]
    while r < rr[1]:
        var ci = 0
        var c = cc[0]
        while c < cc[1]:
            var src_lin = r * strides[0] + c * strides[1]
            var dst_lin = ri * nc + ci
            out._data[dst_lin] = t._data[src_lin]
            c += cc[2]; ci += 1
        r += rr[2]; ri += 1
    return out

fn slice3d_float(t: FloatTensor, d: Int, r0: Int = 0, r1: Int = 0, c0: Int = 0, c1: Int = 0) -> FloatTensor:
    var shp = t._shape
    if len(shp) < 3:
        return FloatTensor([0, 0], 0.0)

    var D = shp[0]; var H = shp[1]; var W = shp[2]
    var dd = d
    if dd < 0: dd = D + dd
    if dd < 0: dd = 0
    if dd >= D: dd = D - 1

    var rr1 = r1
    if rr1 <= 0: rr1 = H
    var cc1 = c1
    if cc1 <= 0: cc1 = W

    var plane = FloatTensor([H, W], 0.0)
    var strides = compute_strides_shared(shp)   # [H*W, W, 1]
    var y = 0
    while y < H:
        var x = 0
        while x < W:
            var src_lin = dd * strides[0] + y * strides[1] + x * strides[2]
            var dst_lin = y * W + x
            plane._data[dst_lin] = t._data[src_lin]
            x += 1
        y += 1

    return slice2d_float(plane, r0, rr1, c0, cc1, 1, 1)

fn slice4d_float(
    t: FloatTensor,
    a0: Int, a1: Int, b0: Int, b1: Int, c0: Int, c1: Int, d0: Int, d1: Int,
    as_: Int = 1, bs_: Int = 1, cs_: Int = 1, ds_: Int = 1
) -> FloatTensor:
    var shp = t._shape
    if len(shp) < 4:
        return FloatTensor([0, 0, 0, 0], 0.0)

    var A = shp[0]; var B = shp[1]; var C = shp[2]; var D = shp[3]

    var ra = norm_range_shared(a0, a1, as_, A)
    var rb = norm_range_shared(b0, b1, bs_, B)
    var rc = norm_range_shared(c0, c1, cs_, C)
    var rd = norm_range_shared(d0, d1, ds_, D)

    if ra[2] <= 0 or rb[2] <= 0 or rc[2] <= 0 or rd[2] <= 0:
        return FloatTensor([0, 0, 0, 0], 0.0)
    if ra[0] >= ra[1] or rb[0] >= rb[1] or rc[0] >= rc[1] or rd[0] >= rd[1]:
        return FloatTensor([0, 0, 0, 0], 0.0)

    var NA = count_range(ra[0], ra[1], ra[2])
    var NB = count_range(rb[0], rb[1], rb[2])
    var NC = count_range(rc[0], rc[1], rc[2])
    var ND = count_range(rd[0], rd[1], rd[2])

    var out = FloatTensor([NA, NB, NC, ND], 0.0)
    var strides = compute_strides_shared(shp)   # [B*C*D, C*D, D, 1]

    var ai = 0
    var a = ra[0]
    while ai < NA:
        var bi = 0
        var b = rb[0]
        while bi < NB:
            var ci = 0
            var c = rc[0]
            while ci < NC:
                var di = 0
                var d = rd[0]
                while di < ND:
                    var src_lin = a * strides[0] + b * strides[1] + c * strides[2] + d * strides[3]
                    var dst_lin = ((ai * NB + bi) * NC + ci) * ND + di
                    out._data[dst_lin] = t._data[src_lin]
                    di += 1
                    d += rd[2]
                ci += 1
                c += rc[2]
            bi += 1
            b += rb[2]
        ai += 1
        a += ra[2]
    return out

fn head_float(t: FloatTensor, n: Int) -> FloatTensor:
    return slice1d_float(t, 0, n, 1)

fn tail_float(t: FloatTensor, n: Int) -> FloatTensor:
    var shp = t._shape
    if len(shp) == 0:
        return FloatTensor([0], 0.0)
    var N = shp[0]
    var s = N - n
    if s < 0: s = 0
    return slice1d_float(t, s, N, 1)

fn take1d_float(t: FloatTensor, indices: List[Int]) -> FloatTensor:
    var shp = t._shape
    var n0 = 0
    if len(shp) > 0: n0 = shp[0]
    var out = FloatTensor([len(indices)], 0.0)
    var strides = compute_strides_shared(shp)
    var i = 0
    while i < len(indices):
        var idx = clamp_idx_shared(indices[i], 0, (n0 if n0 > 0 else 1))
        if n0 > 0:
            out._data[i] = t._data[idx * strides[0]]
        i += 1
    return out

fn gather2d_float(t: FloatTensor, rows: List[Int], cols: List[Int]) -> FloatTensor:
    var shp = t._shape
    if len(shp) < 2:
        return FloatTensor([0, 0], 0.0)
    var R = shp[0]; var C = shp[1]
    var out = FloatTensor([len(rows), len(cols)], 0.0)
    var strides = compute_strides_shared(shp)
    var i = 0
    while i < len(rows):
        var r = clamp_idx_shared(rows[i], 0, (R if R > 0 else 1))
        var j = 0
        while j < len(cols):
            var c = clamp_idx_shared(cols[j], 0, (C if C > 0 else 1))
            if R > 0 and C > 0:
                var src_lin = r * strides[0] + c * strides[1]
                var dst_lin = i * len(cols) + j
                out._data[dst_lin] = t._data[src_lin]
            j += 1
        i += 1
    return out


# =========================
# IntTensor slicing (1D/2D/3D/4D) + utils
# Requires: IntTensor has fields _shape: List[Int], _data: List[Int]
# =========================

fn slice1d_int(t: IntTensor, start: Int, stop: Int, step: Int = 1) -> IntTensor:
    var shp = t._shape
    if len(shp) == 0:
        return IntTensor([0], 0)
    var n = shp[0]
    var a_b_s = norm_range_shared(start, stop, step, n)
    var a = a_b_s[0]; var b = a_b_s[1]; var s = a_b_s[2]
    if s <= 0 or a >= b:
        return IntTensor([0], 0)

    var cnt = count_range(a, b, s)
    var out = IntTensor([cnt], 0)
    var strides = compute_strides_shared(shp)
    var dst = 0
    var i = a
    while i < b:
        var src_lin = i * strides[0]
        out._data[dst] = t._data[src_lin]
        dst += 1
        i += s
    return out

fn slice2d_int(t: IntTensor, r0: Int, r1: Int, c0: Int, c1: Int, rs: Int = 1, cs: Int = 1) -> IntTensor:
    var shp = t._shape
    if len(shp) < 2:
        return IntTensor([0, 0], 0)

    var R = shp[0]; var C = shp[1]
    var rr = norm_range_shared(r0, r1, rs, R)
    var cc = norm_range_shared(c0, c1, cs, C)
    if rr[2] <= 0 or cc[2] <= 0 or rr[0] >= rr[1] or cc[0] >= cc[1]:
        return IntTensor([0, 0], 0)

    var nr = count_range(rr[0], rr[1], rr[2])
    var nc = count_range(cc[0], cc[1], cc[2])

    var out = IntTensor([nr, nc], 0)
    var strides = compute_strides_shared(shp)
    var ri = 0
    var r = rr[0]
    while r < rr[1]:
        var ci = 0
        var c = cc[0]
        while c < cc[1]:
            var src_lin = r * strides[0] + c * strides[1]
            var dst_lin = ri * nc + ci
            out._data[dst_lin] = t._data[src_lin]
            c += cc[2]; ci += 1
        r += rr[2]; ri += 1
    return out

fn slice3d_int(t: IntTensor, d: Int, r0: Int = 0, r1: Int = 0, c0: Int = 0, c1: Int = 0) -> IntTensor:
    var shp = t._shape
    if len(shp) < 3:
        return IntTensor([0, 0], 0)

    var D = shp[0]; var H = shp[1]; var W = shp[2]
    var dd = d
    if dd < 0: dd = D + dd
    if dd < 0: dd = 0
    if dd >= D: dd = D - 1

    var rr1 = r1
    if rr1 <= 0: rr1 = H
    var cc1 = c1
    if cc1 <= 0: cc1 = W

    var plane = IntTensor([H, W], 0)
    var strides = compute_strides_shared(shp)
    var y = 0
    while y < H:
        var x = 0
        while x < W:
            var src_lin = dd * strides[0] + y * strides[1] + x * strides[2]
            var dst_lin = y * W + x
            plane._data[dst_lin] = t._data[src_lin]
            x += 1
        y += 1

    return slice2d_int(plane, r0, rr1, c0, cc1, 1, 1)

fn slice4d_int(
    t: IntTensor,
    a0: Int, a1: Int, b0: Int, b1: Int, c0: Int, c1: Int, d0: Int, d1: Int,
    as_: Int = 1, bs_: Int = 1, cs_: Int = 1, ds_: Int = 1
) -> IntTensor:
    var shp = t._shape
    if len(shp) < 4:
        return IntTensor([0, 0, 0, 0], 0)

    var A = shp[0]; var B = shp[1]; var C = shp[2]; var D = shp[3]

    var ra = norm_range_shared(a0, a1, as_, A)
    var rb = norm_range_shared(b0, b1, bs_, B)
    var rc = norm_range_shared(c0, c1, cs_, C)
    var rd = norm_range_shared(d0, d1, ds_, D)

    if ra[2] <= 0 or rb[2] <= 0 or rc[2] <= 0 or rd[2] <= 0:
        return IntTensor([0, 0, 0, 0], 0)
    if ra[0] >= ra[1] or rb[0] >= rb[1] or rc[0] >= rc[1] or rd[0] >= rd[1]:
        return IntTensor([0, 0, 0, 0], 0)

    var NA = count_range(ra[0], ra[1], ra[2])
    var NB = count_range(rb[0], rb[1], rb[2])
    var NC = count_range(rc[0], rc[1], rc[2])
    var ND = count_range(rd[0], rd[1], rd[2])

    var out = IntTensor([NA, NB, NC, ND], 0)
    var strides = compute_strides_shared(shp)   # [B*C*D, C*D, D, 1]

    var ai = 0
    var a = ra[0]
    while ai < NA:
        var bi = 0
        var b = rb[0]
        while bi < NB:
            var ci = 0
            var c = rc[0]
            while ci < NC:
                var di = 0
                var d = rd[0]
                while di < ND:
                    var src_lin = a * strides[0] + b * strides[1] + c * strides[2] + d * strides[3]
                    var dst_lin = ((ai * NB + bi) * NC + ci) * ND + di
                    out._data[dst_lin] = t._data[src_lin]
                    di += 1
                    d += rd[2]
                ci += 1
                c += rc[2]
            bi += 1
            b += rb[2]
        ai += 1
        a += ra[2]
    return out

fn head_int(t: IntTensor, n: Int) -> IntTensor:
    return slice1d_int(t, 0, n, 1)

fn tail_int(t: IntTensor, n: Int) -> IntTensor:
    var shp = t._shape
    if len(shp) == 0:
        return IntTensor([0], 0)
    var N = shp[0]
    var s = N - n
    if s < 0: s = 0
    return slice1d_int(t, s, N, 1)

fn take1d_int(t: IntTensor, indices: List[Int]) -> IntTensor:
    var shp = t._shape
    var n0 = 0
    if len(shp) > 0: n0 = shp[0]
    var out = IntTensor([len(indices)], 0)
    var strides = compute_strides_shared(shp)
    var i = 0
    while i < len(indices):
        var idx = clamp_idx_shared(indices[i], 0, (n0 if n0 > 0 else 1))
        if n0 > 0:
            out._data[i] = t._data[idx * strides[0]]
        i += 1
    return out

fn gather2d_int(t: IntTensor, rows: List[Int], cols: List[Int]) -> IntTensor:
    var shp = t._shape
    if len(shp) < 2:
        return IntTensor([0, 0], 0)
    var R = shp[0]; var C = shp[1]
    var out = IntTensor([len(rows), len(cols)], 0)
    var strides = compute_strides_shared(shp)
    var i = 0
    while i < len(rows):
        var r = clamp_idx_shared(rows[i], 0, (R if R > 0 else 1))
        var j = 0
        while j < len(cols):
            var c = clamp_idx_shared(cols[j], 0, (C if C > 0 else 1))
            if R > 0 and C > 0:
                var src_lin = r * strides[0] + c * strides[1]
                var dst_lin = i * len(cols) + j
                out._data[dst_lin] = t._data[src_lin]
            j += 1
        i += 1
    return out


# =========================
# 3D dim0 slices and last-dim plane helpers (exports expected by __init__)
# Int/Float/Bool as required
# =========================

# slice along dim-0: input 3D [B, M, N] -> output 2D [M, N]
fn slice_dim0_int(a: tensor.IntTensor, i: Int) -> tensor.IntTensor:
    var shp = a.shape()
    if len(shp) != 3:
        return tensor.IntTensor([0, 0], 0)
    var B = shp[0]; var M = shp[1]; var N = shp[2]
    if B <= 0 or M <= 0 or N <= 0:
        return tensor.IntTensor([0, 0], 0)

    var ii = clamp_idx_shared(i, 0, B)
    var out = tensor.IntTensor([M, N], 0)
    var j = 0
    while j < M:
        var k = 0
        while k < N:
            out[j, k] = a[ii, j, k]
            k += 1
        j += 1
    return out

fn slice_dim0_float(a: tensor.FloatTensor, i: Int) -> tensor.FloatTensor:
    var shp = a.shape()
    if len(shp) != 3:
        return tensor.FloatTensor([0, 0], 0.0)
    var B = shp[0]; var M = shp[1]; var N = shp[2]
    if B <= 0 or M <= 0 or N <= 0:
        return tensor.FloatTensor([0, 0], 0.0)

    var ii = clamp_idx_shared(i, 0, B)
    var out = tensor.FloatTensor([M, N], 0.0)
    var j = 0
    while j < M:
        var k = 0
        while k < N:
            out[j, k] = a[ii, j, k]
            k += 1
        j += 1
    return out

fn slice_dim0_bool(a: tensor.BoolTensor, i: Int) -> tensor.BoolTensor:
    var shp = a.shape()
    if len(shp) != 3:
        return tensor.BoolTensor([0, 0], False)
    var B = shp[0]; var M = shp[1]; var N = shp[2]
    if B <= 0 or M <= 0 or N <= 0:
        return tensor.BoolTensor([0, 0], False)

    var ii = clamp_idx_shared(i, 0, B)
    var out = tensor.BoolTensor([M, N], False)

    var base = ii * M * N
    var j = 0
    while j < M:
        var k = 0
        while k < N:
            out._data[j * N + k] = a._data[base + j * N + k]
            k += 1
        j += 1
    return out


# read plane on last dim (dim must be 2): input [B,M,N] -> output [B,M]
fn get_last_dim_plane_int(a: tensor.IntTensor, dim: Int, index: Int) -> tensor.IntTensor:
    var shp = a.shape()
    if len(shp) != 3:
        return tensor.IntTensor([0, 0], 0)
    var B = shp[0]; var M = shp[1]; var N = shp[2]
    if B <= 0 or M <= 0 or N <= 0:
        return tensor.IntTensor([0, 0], 0)
    if dim != 2:
        return tensor.IntTensor([0, 0], 0)

    var idx = clamp_idx_shared(index, 0, N)
    var out = tensor.IntTensor([B, M], 0)
    var i = 0
    while i < B:
        var j = 0
        while j < M:
            out[i, j] = a[i, j, idx]
            j += 1
        i += 1
    return out

# write plane on last dim (dim must be 2); rhs is [B,M] or [B,1]
fn assign_last_dim_plane_int(mut a: tensor.IntTensor, dim: Int, index: Int, rhs: tensor.IntTensor) -> tensor.IntTensor:
    var shp = a.shape()
    if len(shp) != 3:
        return a
    var B = shp[0]; var M = shp[1]; var N = shp[2]
    if B <= 0 or M <= 0 or N <= 0:
        return a
    if dim != 2:
        return a

    var idx = clamp_idx_shared(index, 0, N)
    var rsh = rhs.shape()
    if len(rsh) != 2:
        return a
    if rsh[0] != B:
        return a
    if not ((rsh[1] == 1) or (rsh[1] == M)):
        return a

    var i = 0
    while i < B:
        var j = 0
        while j < M:
            var col = 0
            if rsh[1] == 1:
                col = 0
            else:
                col = j
            a[i, j, idx] = rhs[i, col]
            j += 1
        i += 1
    return a


# =========================
# Optional symmetric plane helpers (float) + 4D plane helpers (both types)
# =========================

# float last-dim plane get/assign on [B,M,N]
fn get_last_dim_plane4_int(a: tensor.IntTensor, dim: Int, index: Int) -> tensor.IntTensor:
    var shp = a.shape()
    if len(shp) != 4:
        return tensor.IntTensor([0, 0, 0], 0)
    var A = shp[0]; var B = shp[1]; var C = shp[2]; var D = shp[3]
    if A <= 0 or B <= 0 or C <= 0 or D <= 0:
        return tensor.IntTensor([0, 0, 0], 0)
    if dim != 3:
        return tensor.IntTensor([0, 0, 0], 0)

    var idx = clamp_idx_shared(index, 0, D)
    var out = tensor.IntTensor([A, B, C], 0)
    var a0 = 0
    while a0 < A:
        var b0 = 0
        while b0 < B:
            var c0 = 0
            while c0 < C:
                out[a0, b0, c0] = a[a0, b0, c0, idx]
                c0 += 1
            b0 += 1
        a0 += 1
    return out

fn assign_last_dim_plane4_int(mut a: tensor.IntTensor, dim: Int, index: Int, rhs: tensor.IntTensor) -> tensor.IntTensor:
    var shp = a.shape()
    if len(shp) != 4:
        return a
    var A = shp[0]; var B = shp[1]; var C = shp[2]; var D = shp[3]
    if A <= 0 or B <= 0 or C <= 0 or D <= 0:
        return a
    if dim != 3:
        return a

    var idx = clamp_idx_shared(index, 0, D)
    var rsh = rhs.shape()
    if len(rsh) != 3:
        return a
    if rsh[0] != A or rsh[1] != B:
        return a
    if not ((rsh[2] == 1) or (rsh[2] == C)):
        return a

    var a0 = 0
    while a0 < A:
        var b0 = 0
        while b0 < B:
            var c0 = 0
            while c0 < C:
                var col = 0
                if rsh[2] == 1:
                    col = 0
                else:
                    col = c0
                a[a0, b0, c0, idx] = rhs[a0, b0, col]
                c0 += 1
            b0 += 1
        a0 += 1
    return a

# 4D plane helpers: fix dim==3 (last) to index; return [A,B,C]
fn get_last_dim_plane4_int(a: tensor.IntTensor, dim: Int, index: Int) -> tensor.IntTensor:
    var shp = a.shape()
    if len(shp) != 4:
        return tensor.IntTensor([0, 0, 0], 0)
    var A = shp[0]; var B = shp[1]; var C = shp[2]; var D = shp[3]
    if A <= 0 or B <= 0 or C <= 0 or D <= 0:
        return tensor.IntTensor([0, 0, 0], 0)
    if dim != 3:
        return tensor.IntTensor([0, 0, 0], 0)

    var idx = clamp_idx_shared(index, 0, D)
    var out = tensor.IntTensor([A, B, C], 0)
    var a0 = 0
    while a0 < A:
        var b0 = 0
        while b0 < B:
            var c0 = 0
            while c0 < C:
                out[a0, b0, c0] = a.get4(a0, b0, c0, idx)
                c0 += 1
            b0 += 1
        a0 += 1
    return out

fn assign_last_dim_plane4_int(mut a: tensor.IntTensor, dim: Int, index: Int, rhs: tensor.IntTensor) -> tensor.IntTensor:
    var shp = a.shape()
    if len(shp) != 4:
        return a
    var A = shp[0]; var B = shp[1]; var C = shp[2]; var D = shp[3]
    if A <= 0 or B <= 0 or C <= 0 or D <= 0:
        return a
    if dim != 3:
        return a

    var idx = clamp_idx_shared(index, 0, D)
    var rsh = rhs.shape()
    if len(rsh) != 3:
        return a
    if rsh[0] != A or rsh[1] != B:
        return a
    if not ((rsh[2] == 1) or (rsh[2] == C)):
        return a

    var a0 = 0
    while a0 < A:
        var b0 = 0
        while b0 < B:
            var c0 = 0
            while c0 < C:
                var col = 0
                if rsh[2] == 1:
                    col = 0
                else:
                    col = c0
                a[a0, b0, c0, idx] = rhs[a0, b0, col]
                c0 += 1
            b0 += 1
        a0 += 1
    return a

# float 4D plane helpers
fn get_last_dim_plane4_float(a: tensor.FloatTensor, dim: Int, index: Int) -> tensor.FloatTensor:
    var shp = a.shape()
    if len(shp) != 4:
        return tensor.FloatTensor([0, 0, 0], 0.0)
    var A = shp[0]; var B = shp[1]; var C = shp[2]; var D = shp[3]
    if A <= 0 or B <= 0 or C <= 0 or D <= 0:
        return tensor.FloatTensor([0, 0, 0], 0.0)
    if dim != 3:
        return tensor.FloatTensor([0, 0, 0], 0.0)

    var idx = clamp_idx_shared(index, 0, D)
    var out = tensor.FloatTensor([A, B, C], 0.0)
    var a0 = 0
    while a0 < A:
        var b0 = 0
        while b0 < B:
            var c0 = 0
            while c0 < C:
                out[a0, b0, c0] = a.get4(a0, b0, c0, idx)
                c0 += 1
            b0 += 1
        a0 += 1
    return out

fn assign_last_dim_plane4_float(mut a: tensor.FloatTensor, dim: Int, index: Int, rhs: tensor.FloatTensor) -> tensor.FloatTensor:
    var shp = a.shape()
    if len(shp) != 4:
        return a
    var A = shp[0]; var B = shp[1]; var C = shp[2]; var D = shp[3]
    if A <= 0 or B <= 0 or C <= 0 or D <= 0:
        return a
    if dim != 3:
        return a

    var idx = clamp_idx_shared(index, 0, D)
    var rsh = rhs.shape()
    if len(rsh) != 3:
        return a
    if rsh[0] != A or rsh[1] != B:
        return a
    if not ((rsh[2] == 1) or (rsh[2] == C)):
        return a

    var a0 = 0
    while a0 < A:
        var b0 = 0
        while b0 < B:
            var c0 = 0
            while c0 < C:
                var col = 0
                if rsh[2] == 1:
                    col = 0
                else:
                    col = c0
                a[a0, b0, c0, idx] = rhs[a0, b0, col]
                c0 += 1
            b0 += 1
        a0 += 1
    return a
