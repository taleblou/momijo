# MIT License
# Project: momijo.tensor
# File: src/momijo/tensor/__init__.mojo
# Public API for float/int/bool tensors with dtype factories.
from math import cosh
from math import sinh
from math import tanh
from math import sqrt
from math import log
from math import exp
from math import tan
from math import cos
from math import sin

from momijo.tensor.dtype import DType, int32, float64, bool_
from momijo.tensor.tensorfloat import FloatTensor
from momijo.tensor.tensorint import IntTensor
from momijo.tensor.tensorbool import BoolTensor
from momijo.tensor.printing import to_string_dict, to_string_tensor
from momijo.tensor.indexing import (
    # List helpers
    list_select as select_list,
    list_slice as slice_list,

    # Shared tensor helpers
    compute_strides_shared as compute_strides,
    clamp_idx_shared as clamp_idx,
    norm_range_shared as norm_range,

    # FloatTensor ops
    slice1d_float as _slice1d_float,
    slice2d_float as _slice2d_float,
    slice3d_float as _slice3d_float,
    head_float as _head_float,
    tail_float as _tail_float,
    take1d_float as _take1d_float,
    gather2d_float as _gather2d_float,

    # IntTensor ops
    slice1d_int as _slice1d_int,
    slice2d_int as _slice2d_int,
    slice3d_int as _slice3d_int,
    head_int as _head_int,
    tail_int as _tail_int,
    take1d_int as _take1d_int,
    gather2d_int as _gather2d_int,

    # Dim0 and last-dim plane helpers
    slice_dim0_int as _slice_dim0_int,
    slice_dim0_float as _slice_dim0_float,
    slice_dim0_bool as _slice_dim0_bool,
    get_last_dim_plane_int as _plane_int,
    assign_last_dim_plane_int as _write_plane_int,
)


# --------------------------------------------------------------------
# Small utilities

fn int_to_string(x: Int) -> String:
    return String(x)

fn float_to_string(x: Float64) -> String:
    return String(x)

fn list_to_string(xs: List[Int]) -> String:
    var s = "["
    var i = 0
    while i < len(xs):
        s = s + int_to_string(xs[i])
        if i < len(xs) - 1:
            s = s + ", "
        i += 1
    s = s + "]"
    return s

# --------------------------------------------------------------------
# Unified helpers

fn slice_dim0(a: IntTensor, i: Int) -> IntTensor:
    return _slice_dim0_int(a, i)

fn slice_dim0(a: FloatTensor, i: Int) -> FloatTensor:
    return _slice_dim0_float(a, i)

fn slice_dim0(a: BoolTensor, i: Int) -> BoolTensor:
    return _slice_dim0_bool(a, i)

# ---------------- gather (2D) ----------------
fn gather(a: IntTensor, axis: Int, indices: IntTensor) -> IntTensor:
    var shp = a.shape()
    var R = shp[0]
    var C = shp[1]

    if axis == 0:
        var K = indices.shape()[0]
        var out = IntTensor([K, C], 0)
        var i = 0
        while i < K:
            var r = indices.item(i)
            if r < 0: r = 0
            if r >= R: r = R - 1
            var j = 0
            while j < C:
                out[i, j] = a[r, j]
                j += 1
            i += 1
        return out
    else:
        var K = indices.shape()[0]
        var out = IntTensor([R, K], 0)
        var j2 = 0
        while j2 < K:
            var c = indices.item(j2)
            if c < 0: c = 0
            if c >= C: c = C - 1
            var i = 0
            while i < R:
                out[i, j2] = a[i, c]
                i += 1
            j2 += 1
        return out

fn gather_float(a: FloatTensor, axis: Int, indices: IntTensor) -> FloatTensor:
    var shp = a.shape()
    var R = shp[0]
    var C = shp[1]

    if axis == 0:
        var K = indices.shape()[0]
        var out = FloatTensor([K, C], 0.0)
        var i = 0
        while i < K:
            var r = indices.item(i)
            if r < 0: r = 0
            if r >= R: r = R - 1
            var j = 0
            while j < C:
                out[i, j] = a[r, j]
                j += 1
            i += 1
        return out
    else:
        var K = indices.shape()[0]
        var out = FloatTensor([R, K], 0.0)
        var j2 = 0
        while j2 < K:
            var c = indices.item(j2)
            if c < 0: c = 0
            if c >= C: c = C - 1
            var i = 0
            while i < R:
                out[i, j2] = a[i, c]
                i += 1
            j2 += 1
        return out

fn gather_bool(a: BoolTensor, axis: Int, indices: IntTensor) -> BoolTensor:
    var shp = a.shape()
    var R = shp[0]
    var C = shp[1]

    if axis == 0:
        var K = indices.shape()[0]
        var out = BoolTensor([K, C], False)
        var i = 0
        while i < K:
            var r = indices.item(i)
            if r < 0: r = 0
            if r >= R: r = R - 1
            var j = 0
            while j < C:
                out[i, j] = a[r, j]
                j += 1
            i += 1
        return out
    else:
        var K = indices.shape()[0]
        var out = BoolTensor([R, K], False)
        var j2 = 0
        while j2 < K:
            var c = indices.item(j2)
            if c < 0: c = 0
            if c >= C: c = C - 1
            var i = 0
            while i < R:
                out[i, j2] = a[i, c]
                i += 1
            j2 += 1
        return out

# ---------------- plane / write_plane ----------------
fn plane(a: IntTensor, dim: Int, index: Int) -> IntTensor:
    var d = dim
    if d != 2: d = 2
    return _plane_int(a, d, index)

fn write_plane(mut a: IntTensor, dim: Int, index: Int, rhs: IntTensor) -> IntTensor:
    var d = dim
    if d != 2: d = 2
    return _write_plane_int(a, d, index, rhs)

# ---------------- small numeric helper ----------------
fn float_linspace(start: Float64, stop: Float64, num: Int) -> FloatTensor:
    var out = FloatTensor([num], 0.0)
    if num <= 0:
        return out
    if num == 1:
        out[0] = start
        return out
    var step = (stop - start) / Float64(num - 1)
    var i = 0
    while i < num:
        out[i] = start + step * Float64(i)
        i += 1
    return out

fn tensor_to_string(t: FloatTensor) -> String:
    var s = "FloatTensor(shape=" + list_to_string(t.shape()) + ", data=["
    var limit = 5
    var n = t.len()
    var i = 0
    while i < min(limit, n):
        s = s + float_to_string(t.item(i))
        if i < min(limit, n) - 1:
            s = s + ", "
        i += 1
    if n > limit:
        s = s + ", ..."
    s = s + "])"
    return s

fn tensor_to_string(d: Dict[String, FloatTensor]) -> String:
    var s = "npz{keys=["
    var first = True
    for k in d.keys():
        if not first:
            s = s + ", "
        s = s + k
        first = False
    s = s + "]}"
    return s

# --- Utilities ---
fn _shape_prod(shape: List[Int]) -> Int:
    var p = 1
    var i = 0
    while i < shape.len():
        p *= shape[i]
        i += 1
    return p

fn _compute_strides_row_major(shape: List[Int]) -> List[Int]:
    var n = shape.len()
    var strides = List[Int]()
    strides.resize(n)
    var acc = 1
    var i = n - 1
    while i >= 0:
        strides[i] = acc
        acc *= shape[i]
        i -= 1
    return strides

# --- Public API: stack for 1D FloatTensor list (non-raising, safe) ---
fn stack(inputs: List[FloatTensor], axis: Int = 0) -> FloatTensor:
    # Empty => safe default
    if len(inputs) == 0:
        return FloatTensor.zeros([0, 0])

    # Only 1D vectors allowed
    if len(inputs[0].shape()) != 1:
        return FloatTensor.zeros([0, 0])

    var length0 = inputs[0].shape()[0]
    var i = 1
    while i < len(inputs):
        if len(inputs[i].shape()) != 1:
            return FloatTensor.zeros([0, 0])
        if inputs[i].shape()[0] != length0:
            return FloatTensor.zeros([0, 0])
        i += 1

    # Normalize/validate axis
    var ax = axis
    if ax < 0:
        ax = ax + 2
    if not (ax == 0 or ax == 1):
        return FloatTensor.zeros([0, 0])

    # Output shape
    var k = len(inputs)
    var out_shape = List[Int]()
    if ax == 0:
        # [k, N]
        out_shape.append(k)
        out_shape.append(length0)
    else:
        # [N, k]
        out_shape.append(length0)
        out_shape.append(k)

    # Build flat row-major buffer
    var flat = List[Float64]()
    if ax == 0:
        # rows = inputs[i], each of length N
        var i = 0
        while i < k:
            var r = 0
            while r < length0:
                flat.append(inputs[i].item(r))
                r += 1
            i += 1
    else:
        # rows = over r in [0..N), each row collects inputs[i][r] over i
        var r = 0
        while r < length0:
            var i = 0
            while i < k:
                flat.append(inputs[i].item(r))
                i += 1
            r += 1

    # Materialize without subscript writes
    return FloatTensor.from_flat(flat, out_shape)



# ===== Factory methods for FloatTensor =====
 
fn from_list_float32(data2d: List[List[Float32]]) -> FloatTensor:
    # shape
    var rows = len(data2d)
    var cols = 0
    if rows > 0:
        cols = len(data2d[0])

    # validate (no ragged rows) — use the same assertion style as elsewhere in your codebase
    var r = 0
    while r < rows:
        (len(data2d[r]) == cols), "from_list_float32: ragged rows not allowed"
        if not (len(data2d[r]) == cols):
            # keep control flow valid even in release builds
            return FloatTensor([0, 0], 0.0)
        r += 1

    # materialize (FloatTensor uses Float64 storage)
    var out = FloatTensor([rows, cols], 0.0)
    var i = 0
    r = 0
    while r < rows:
        var c = 0
        while c < cols:
            out._data[i] = Float64(data2d[r][c])
            i += 1
            c += 1
        r += 1
    return out

fn empty(shape: List[Int]) -> FloatTensor:
    return FloatTensor(shape, 0.0)
 
fn full(shape: List[Int], value: Float64) -> FloatTensor:
    var t = FloatTensor.empty(shape)
    var n = len(t._data)
    var i = 0
    while i < n:
        t._data[i] = value
        i += 1
    return t
 
fn full(shape: List[Int], value: Float32) -> FloatTensor:
    return FloatTensor.full(shape, Float64(value))


fn zeros(shape: List[Int]) -> FloatTensor:
    return FloatTensor.full(shape, 0.0)


fn ones(shape: List[Int]) -> FloatTensor:
    return FloatTensor.full(shape, 1.0)


fn zeros_like(x: FloatTensor) -> FloatTensor:
    return FloatTensor.zeros(x._shape)


 
fn arange(start: Int, stop: Int, step: Int = 1) -> FloatTensor:
    step != 0, "arange: step must not be zero"

    var n = 0
    if step > 0 and start < stop:
        n = (stop - start + step - 1) // step
    elif step < 0 and start > stop:
        var neg_step = -step
        n = (start - stop + neg_step - 1) // neg_step
    else:
        n = 0

    var out = FloatTensor([n], 0.0)
    var v = start
    var i = 0
    while i < n:
        out._data[i] = Float64(v)
        v += step
        i += 1
    return out

 
fn linspace(start: Float64, stop: Float64, steps: Int) -> FloatTensor:
    steps >= 2, "linspace: steps must be >= 2"
    var out = FloatTensor([steps], 0.0)
    var i = 0
    var denom = Float64(steps - 1)
    while i < steps:
        var t = Float64(i) / denom
        out._data[i] = start + t * (stop - start)
        i += 1
    return out

 
fn linspace(start: Float32, stop: Float32, steps: Int) -> FloatTensor:
    return FloatTensor.linspace(Float64(start), Float64(stop), steps)


 


fn eye(n: Int) -> FloatTensor:
    var out = FloatTensor.zeros([n, n])
    var i = 0
    while i < n:
        out._data[i * n + i] = 1.0
        i += 1
    return out


 
fn rand(shape: List[Int], seed: Optional[Int] = None) -> FloatTensor:
    # Simple LCG-based RNG; avoids Random.default()
    var out = FloatTensor(shape, 0.0)
    var state: UInt64 = 0x9E3779B97F4A7C15
    if not (seed is None):
        state = UInt64(seed.value())
        if state == 0:
            state = 0x2545F4914F6CDD1D

    var n = len(out._data)
    var i = 0
    while i < n:
        # LCG step
        state = state * 2862933555777941757 + 3037000493
        # Uniform in [0, 1)
        var u = Float64((state >> 11) & 0xFFFFFFFF) / 4294967296.0
        out._data[i] = u
        i += 1
    return out


 
fn randn(shape: List[Int], seed: Optional[Int] = None) -> FloatTensor:
    # Box–Muller with simple LCG; no dependency on Random.default()
    var out = FloatTensor(shape, 0.0)

    var state: UInt64 = 0xA24BAED5CF9A13B7
    if not (seed is None):
        state = UInt64(seed.value())
        if state == 0:
            state = 0xA24BAED5CF9A13B7

    var n = len(out._data)
    var i = 0
    while i < n:
        # u1 in (0,1], u2 in [0,1)
        state = state * 2862933555777941757 + 3037000493
        var u1 = Float64((state >> 11) & 0xFFFFFFFF) / 4294967296.0
        if u1 <= 0.0:
            u1 = 1e-12
        state = state * 2862933555777941757 + 3037000493
        var u2 = Float64((state >> 11) & 0xFFFFFFFF) / 4294967296.0

        var r = sqrt(-2.0 * log(u1))
        var theta = 6.283185307179586 * u2
        out._data[i] = r * cos(theta)
        i += 1
    return out

