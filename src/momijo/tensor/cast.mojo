

from collections.list import List
from momijo.tensor.tensor import Tensor

from momijo.tensor.helpers import numel,compute_row_major_strides



@always_inline
fn _row_major_strides(shape: List[Int]) -> List[Int]:
    var n = len(shape)
    var s = List[Int]()
    s.reserve(n)
    if n == 0:
        return s
    # init with zeros
    var i = 0
    while i < n:
        s.append(0)
        i += 1
    var acc = 1
    var k = n - 1
    while k >= 0:
        s[k] = acc
        acc = acc * shape[k]
        k -= 1
    return s

@always_inline
fn _is_contiguous(shape: List[Int], strides: List[Int]) -> Bool:
    var exp = _row_major_strides(shape)
    var n = len(shape)
    if len(strides) != n:
        return False
    var i = 0
    while i < n:
        if strides[i] != exp[i]:
            return False
        i += 1
    return True

# Walk logical order (row-major) using strides; returns a flat copy in logical order.
fn _logical_flat_copy[T: ImplicitlyCopyable & Copyable & Movable](x: Tensor[T]) -> List[T]:
    var shape = x._shape
    var strides = x._strides
    var data = x._data

    # Fast path: contiguous
    if _is_contiguous(shape, strides):
        return data.copy()

    # General path: stride-walking iterator
    var ndim = len(shape)
    var total = 1
    var d = 0
    while d < ndim:
        total = total * shape[d]
        d += 1

    var out = List[T]()
    out.reserve(total)

    # handle scalars or empty
    if ndim == 0:
        return out

    # index vector and current physical offset
    var idx = List[Int]()
    idx.reserve(ndim)
    var i = 0
    while i < ndim:
        idx.append(0)
        i += 1

    var pos = 0
    var count = 0
    while count < total:
        out.append(data[pos])

        # increment multi-index with carry
        var a = ndim - 1
        while a >= 0:
            idx[a] = idx[a] + 1
            pos = pos + strides[a]
            if idx[a] < shape[a]:
                break
            # carry
            pos = pos - strides[a] * shape[a]
            idx[a] = 0
            a -= 1
        count += 1

    return out


# --- type tag ---

struct TypeTag[U: ImplicitlyCopyable & Copyable & Movable]:
    var from_f64: fn (Float64) -> U
    fn __init__(out self, from_f64: fn (Float64) -> U):
        self.from_f64 = from_f64



@always_inline 
fn from_float64_to[T: ImplicitlyCopyable & Copyable & Movable](x: Float64, f: fn (Float64) -> T) -> T:
    return T(x)

# ===================== Compatibility aliases for mean_with, etc. =====================
# These names are referenced elsewhere (e.g., math.mojo). They map to the concrete
# Float64 -> T converters already defined.

@always_inline
fn f64_to(v: Float64) -> Float64:
    return to_f64_from_f64(v)

@always_inline
fn f64_to_float32(v: Float64) -> Float32:
    return to_f32_from_f64(v)

@always_inline
fn f64_to_int8(v: Float64) -> Int8:
    return to_i8_from_f64(v)

@always_inline
fn f64_to_int16(v: Float64) -> Int16:
    return to_i16_from_f64(v)

@always_inline
fn f64_to_int32(v: Float64) -> Int32:
    return to_i32_from_f64(v)

@always_inline
fn f64_to_int64(v: Float64) -> Int64:
    return to_i64_from_f64(v)

@always_inline
fn f64_to_int(v: Float64) -> Int:
    return to_int_from_f64(v)

@always_inline
fn f64_to_uint8(v: Float64) -> UInt8:
    return to_u8_from_f64(v)

@always_inline
fn f64_to_uint16(v: Float64) -> UInt16:
    return to_u16_from_f64(v)

@always_inline
fn f64_to_uint32(v: Float64) -> UInt32:
    return to_u32_from_f64(v)

@always_inline
fn f64_to_uint64(v: Float64) -> UInt64:
    return to_u64_from_f64(v)


# ===================== Identity =====================

@always_inline
fn id_i8(x: Int8) -> Int8:
    return x

@always_inline
fn id_i16(x: Int16) -> Int16:
    return x

@always_inline
fn id_i32(x: Int32) -> Int32:
    return x

@always_inline
fn id_i64(x: Int64) -> Int64:
    return x

@always_inline
fn id_int(x: Int) -> Int:
    return x

@always_inline
fn id_u8(x: UInt8) -> UInt8:
    return x

@always_inline
fn id_u16(x: UInt16) -> UInt16:
    return x

@always_inline
fn id_u32(x: UInt32) -> UInt32:
    return x

@always_inline
fn id_u64(x: UInt64) -> UInt64:
    return x

@always_inline
fn id_f16(x: Float16) -> Float16:
    return x

@always_inline
fn id_f32(x: Float32) -> Float32:
    return x

@always_inline
fn id_f64(x: Float64) -> Float64:
    return x


# ===================== Int -> {Int, UInt, Float} =====================

@always_inline
fn to_i8_from_int(x: Int) -> Int8:
    return Int8(x)

@always_inline
fn to_i16_from_int(x: Int) -> Int16:
    return Int16(x)

@always_inline
fn to_i32_from_int(x: Int) -> Int32:
    return Int32(x)

@always_inline
fn to_i64_from_int(x: Int) -> Int64:
    return Int64(x)

@always_inline
fn to_u8_from_int(x: Int) -> UInt8:
    return UInt8(x)

@always_inline
fn to_u16_from_int(x: Int) -> UInt16:
    return UInt16(x)

@always_inline
fn to_u32_from_int(x: Int) -> UInt32:
    return UInt32(x)

@always_inline
fn to_u64_from_int(x: Int) -> UInt64:
    return UInt64(x)

@always_inline
fn to_f16_from_int(x: Int) -> Float16:
    return Float16(Float32(x))

@always_inline
fn to_f32_from_int(x: Int) -> Float32:
    return Float32(x)

@always_inline
fn to_f64_from_int(x: Int) -> Float64:
    return Float64(Int64(x))


# ===================== Float64 -> {Int, UInt, Float} =====================

@always_inline
fn to_i8_from_f64(x: Float64) -> Int8:
    return Int8(x)

@always_inline
fn to_i16_from_f64(x: Float64) -> Int16:
    return Int16(x)

@always_inline
fn to_i32_from_f64(x: Float64) -> Int32:
    return Int32(x)

@always_inline
fn to_i64_from_f64(x: Float64) -> Int64:
    return Int64(x)

@always_inline
fn to_int_from_f64(x: Float64) -> Int:
    return Int(x)

@always_inline
fn to_u8_from_f64(x: Float64) -> UInt8:
    return UInt8(x)

@always_inline
fn to_u16_from_f64(x: Float64) -> UInt16:
    return UInt16(x)

@always_inline
fn to_u32_from_f64(x: Float64) -> UInt32:
    return UInt32(x)

@always_inline
fn to_u64_from_f64(x: Float64) -> UInt64:
    return UInt64(x)

@always_inline
fn to_f16_from_f64(x: Float64) -> Float16:
    return Float16(Float32(x))

@always_inline
fn to_f32_from_f64(x: Float64) -> Float32:
    return Float32(x)

@always_inline
fn to_f64_from_f64(x: Float64) -> Float64:
    return x


# ===================== Float32 -> {Int, UInt, Float} =====================

@always_inline
fn to_int_from_f32(x: Float32) -> Int:
    return Int(x)

@always_inline
fn to_i8_from_f32(x: Float32) -> Int8:
    return Int8(x)

@always_inline
fn to_i16_from_f32(x: Float32) -> Int16:
    return Int16(x)

@always_inline
fn to_i32_from_f32(x: Float32) -> Int32:
    return Int32(x)

@always_inline
fn to_i64_from_f32(x: Float32) -> Int64:
    return Int64(x)

@always_inline
fn to_u8_from_f32(x: Float32) -> UInt8:
    return UInt8(x)

@always_inline
fn to_u16_from_f32(x: Float32) -> UInt16:
    return UInt16(x)

@always_inline
fn to_u32_from_f32(x: Float32) -> UInt32:
    return UInt32(x)

@always_inline
fn to_u64_from_f32(x: Float32) -> UInt64:
    return UInt64(x)

@always_inline
fn to_f16_from_f32(x: Float32) -> Float16:
    return Float16(x)

@always_inline
fn to_f32_from_f32(x: Float32) -> Float32:
    return x

@always_inline
fn to_f64_from_f32(x: Float32) -> Float64:
    return Float64(x)


# ===================== Float16 -> {Int, UInt, Float} =====================

@always_inline
fn to_int_from_f16(x: Float16) -> Int:
    return Int(Float32(x))

@always_inline
fn to_i8_from_f16(x: Float16) -> Int8:
    return Int8(Float32(x))

@always_inline
fn to_i16_from_f16(x: Float16) -> Int16:
    return Int16(Float32(x))

@always_inline
fn to_i32_from_f16(x: Float16) -> Int32:
    return Int32(Float32(x))

@always_inline
fn to_i64_from_f16(x: Float16) -> Int64:
    return Int64(Float32(x))

@always_inline
fn to_u8_from_f16(x: Float16) -> UInt8:
    return UInt8(Float32(x))

@always_inline
fn to_u16_from_f16(x: Float16) -> UInt16:
    return UInt16(Float32(x))

@always_inline
fn to_u32_from_f16(x: Float16) -> UInt32:
    return UInt32(Float32(x))

@always_inline
fn to_u64_from_f16(x: Float16) -> UInt64:
    return UInt64(Float32(x))

@always_inline
fn to_f16_from_f16(x: Float16) -> Float16:
    return x

@always_inline
fn to_f32_from_f16(x: Float16) -> Float32:
    return Float32(x)

@always_inline
fn to_f64_from_f16(x: Float16) -> Float64:
    return Float64(Float32(x))


# ===================== {Int, UInt} -> Float64 (fast reducers) =====================

@always_inline
fn to_f64_from_u8(x: UInt8) -> Float64:
    return Float64(UInt64(x))

@always_inline
fn to_f64_from_u16(x: UInt16) -> Float64:
    return Float64(UInt64(x))

@always_inline
fn to_f64_from_u32(x: UInt32) -> Float64:
    return Float64(UInt64(x))

@always_inline
fn to_f64_from_u64(x: UInt64) -> Float64:
    return Float64(x)

@always_inline
fn to_f64_from_i8(x: Int8) -> Float64:
    return Float64(Int64(x))

@always_inline
fn to_f64_from_i16(x: Int16) -> Float64:
    return Float64(Int64(x))

@always_inline
fn to_f64_from_i32(x: Int32) -> Float64:
    return Float64(Int64(x))

@always_inline
fn to_f64_from_i64(x: Int64) -> Float64:
    return Float64(x)


# ===================== Bool helpers (optional) =====================
# ===================== Cast scalar → Bool =====================
# Keep behavior consistent: "zero" ⇒ False, anything else ⇒ True.

@always_inline
fn to_bool_from_bool(x: Bool) -> Bool:
    # Identity for Bool
    return x

@always_inline
fn to_bool_from_f64(x: Float64) -> Bool:
    # Non-zero → True
    return x != 0.0

@always_inline
fn to_bool_from_f32(x: Float32) -> Bool:
    # Non-zero → True
    return x != 0.0

@always_inline
fn to_bool_from_f16(x: Float16) -> Bool:
    # Compare via Float32
    return Float32(x) != 0.0

@always_inline
fn to_bool_from_int(x: Int) -> Bool:
    # Non-zero → True
    return x != 0

@always_inline
fn to_bool_from_i8(x: Int8) -> Bool:
    return x != 0

@always_inline
fn to_bool_from_i16(x: Int16) -> Bool:
    return x != 0

@always_inline
fn to_bool_from_i32(x: Int32) -> Bool:
    return x != 0

@always_inline
fn to_bool_from_i64(x: Int64) -> Bool:
    return x != 0

@always_inline
fn to_bool_from_uint(x: UInt) -> Bool:
    # Unsigned: zero check is enough
    return x != 0

@always_inline
fn to_bool_from_u8(x: UInt8) -> Bool:
    return x != 0

@always_inline
fn to_bool_from_u16(x: UInt16) -> Bool:
    return x != 0

@always_inline
fn to_bool_from_u32(x: UInt32) -> Bool:
    return x != 0

@always_inline
fn to_bool_from_u64(x: UInt64) -> Bool:
    return x != 0

 


# ===================== Overloaded “to_float64_of” (accumulator picker) =====================
 
@always_inline
fn to_float64_of(v: Bool) -> Float64:
    if v: return 1.0
    return 0.0

@always_inline
fn to_float64_of(v: UInt) -> Float64:
    return Float64(UInt64(v))

@always_inline
fn to_float64_of(v: Float64) -> Float64:
    return v

@always_inline
fn to_float64_of(v: Float32) -> Float64:
    return Float64(v)

@always_inline
fn to_float64_of(v: Float16) -> Float64:
    return Float64(Float32(v))

@always_inline
fn to_float64_of(v: Int8) -> Float64:
    return Float64(Int64(v))

@always_inline
fn to_float64_of(v: Int16) -> Float64:
    return Float64(Int64(v))

@always_inline
fn to_float64_of(v: Int32) -> Float64:
    return Float64(Int64(v))

@always_inline
fn to_float64_of(v: Int64) -> Float64:
    return Float64(v)

@always_inline
fn to_float64_of(v: Int) -> Float64:
    return Float64(Int64(v))

@always_inline
fn to_float64_of(v: UInt8) -> Float64:
    return Float64(UInt64(v))

@always_inline
fn to_float64_of(v: UInt16) -> Float64:
    return Float64(UInt64(v))

@always_inline
fn to_float64_of(v: UInt32) -> Float64:
    return Float64(UInt64(v))

@always_inline
fn to_float64_of(v: UInt64) -> Float64:
    return Float64(v)

fn to_float64_of_T[T: ImplicitlyCopyable & Copyable & Movable](x: T) -> Float64:
    return to_float64_of(x)


# ===================== Backwards-compat short names =====================

@always_inline
fn to_float64_from_int(x: Int) -> Float64:
    return to_f64_from_int(x)

@always_inline
fn to_float32_from_int(x: Int) -> Float32:
    return to_f32_from_int(x)

@always_inline
fn to_int_from_float64(x: Float64) -> Int:
    return to_int_from_f64(x)

@always_inline
fn to_int_from_float32(x: Float32) -> Int:
    return to_int_from_f32(x)


 


##################################################################


@always_inline
fn to_float64(x: Tensor[Float64]) -> Tensor[Float64]:
    # Identity: copy as-is.
    var data = x._data.copy()
    var strides = compute_row_major_strides(x._shape)
    return Tensor[Float64](data, x._shape.copy(), strides,x._offset)


@always_inline
fn to_float64(x: Tensor[Float32]) -> Tensor[Float64]:
    # Widen Float32 → Float64.
    var src = x._data.copy()
    var n = len(src)

    var out = List[Float64]()
    out.reserve(n)

    var i = 0
    var lim = (n // 16) * 16
    while i < lim:
        out.append(Float64(src[i    ]))
        out.append(Float64(src[i + 1]))
        out.append(Float64(src[i + 2]))
        out.append(Float64(src[i + 3]))
        out.append(Float64(src[i + 4]))
        out.append(Float64(src[i + 5]))
        out.append(Float64(src[i + 6]))
        out.append(Float64(src[i + 7]))
        out.append(Float64(src[i + 8]))
        out.append(Float64(src[i + 9]))
        out.append(Float64(src[i + 10]))
        out.append(Float64(src[i + 11]))
        out.append(Float64(src[i + 12]))
        out.append(Float64(src[i + 13]))
        out.append(Float64(src[i + 14]))
        out.append(Float64(src[i + 15]))
        i += 16

    while i < n:
        out.append(Float64(src[i]))
        i += 1

    var strides = compute_row_major_strides(x._shape)
    return Tensor[Float64](out, x._shape.copy(), strides,x._offset)


@always_inline
fn to_float64(x: Tensor[Int64]) -> Tensor[Float64]:
    var src = x._data.copy()
    var n = len(src)

    var out = List[Float64]()
    out.reserve(n)

    var i = 0
    var lim = (n // 16) * 16
    while i < lim:
        out.append(Float64(src[i    ]))
        out.append(Float64(src[i + 1]))
        out.append(Float64(src[i + 2]))
        out.append(Float64(src[i + 3]))
        out.append(Float64(src[i + 4]))
        out.append(Float64(src[i + 5]))
        out.append(Float64(src[i + 6]))
        out.append(Float64(src[i + 7]))
        out.append(Float64(src[i + 8]))
        out.append(Float64(src[i + 9]))
        out.append(Float64(src[i + 10]))
        out.append(Float64(src[i + 11]))
        out.append(Float64(src[i + 12]))
        out.append(Float64(src[i + 13]))
        out.append(Float64(src[i + 14]))
        out.append(Float64(src[i + 15]))
        i += 16

    while i < n:
        out.append(Float64(src[i]))
        i += 1

    var strides = compute_row_major_strides(x._shape)
    return Tensor[Float64](out, x._shape.copy(), strides,x._offset)


@always_inline
fn to_float64(x: Tensor[Int32]) -> Tensor[Float64]:
    var src = x._data.copy()
    var n = len(src)

    var out = List[Float64]()
    out.reserve(n)

    var i = 0
    var lim = (n // 16) * 16
    while i < lim:
        out.append(Float64(src[i    ]))
        out.append(Float64(src[i + 1]))
        out.append(Float64(src[i + 2]))
        out.append(Float64(src[i + 3]))
        out.append(Float64(src[i + 4]))
        out.append(Float64(src[i + 5]))
        out.append(Float64(src[i + 6]))
        out.append(Float64(src[i + 7]))
        out.append(Float64(src[i + 8]))
        out.append(Float64(src[i + 9]))
        out.append(Float64(src[i + 10]))
        out.append(Float64(src[i + 11]))
        out.append(Float64(src[i + 12]))
        out.append(Float64(src[i + 13]))
        out.append(Float64(src[i + 14]))
        out.append(Float64(src[i + 15]))
        i += 16

    while i < n:
        out.append(Float64(src[i]))
        i += 1

    var strides = compute_row_major_strides(x._shape)
    return Tensor[Float64](out, x._shape.copy(), strides,x._offset)


@always_inline
fn to_float64(x: Tensor[Int16]) -> Tensor[Float64]:
    var src = x._data.copy()
    var n = len(src)

    var out = List[Float64]()
    out.reserve(n)

    var i = 0
    var lim = (n // 16) * 16
    while i < lim:
        out.append(Float64(src[i    ]))
        out.append(Float64(src[i + 1]))
        out.append(Float64(src[i + 2]))
        out.append(Float64(src[i + 3]))
        out.append(Float64(src[i + 4]))
        out.append(Float64(src[i + 5]))
        out.append(Float64(src[i + 6]))
        out.append(Float64(src[i + 7]))
        out.append(Float64(src[i + 8]))
        out.append(Float64(src[i + 9]))
        out.append(Float64(src[i + 10]))
        out.append(Float64(src[i + 11]))
        out.append(Float64(src[i + 12]))
        out.append(Float64(src[i + 13]))
        out.append(Float64(src[i + 14]))
        out.append(Float64(src[i + 15]))
        i += 16

    while i < n:
        out.append(Float64(src[i]))
        i += 1

    var strides = compute_row_major_strides(x._shape)
    return Tensor[Float64](out, x._shape.copy(), strides,x._offset)


@always_inline
fn to_float64(x: Tensor[Int8]) -> Tensor[Float64]:
    var src = x._data.copy()
    var n = len(src)

    var out = List[Float64]()
    out.reserve(n)

    var i = 0
    var lim = (n // 16) * 16
    while i < lim:
        out.append(Float64(src[i    ]))
        out.append(Float64(src[i + 1]))
        out.append(Float64(src[i + 2]))
        out.append(Float64(src[i + 3]))
        out.append(Float64(src[i + 4]))
        out.append(Float64(src[i + 5]))
        out.append(Float64(src[i + 6]))
        out.append(Float64(src[i + 7]))
        out.append(Float64(src[i + 8]))
        out.append(Float64(src[i + 9]))
        out.append(Float64(src[i + 10]))
        out.append(Float64(src[i + 11]))
        out.append(Float64(src[i + 12]))
        out.append(Float64(src[i + 13]))
        out.append(Float64(src[i + 14]))
        out.append(Float64(src[i + 15]))
        i += 16

    while i < n:
        out.append(Float64(src[i]))
        i += 1

    var strides = compute_row_major_strides(x._shape)
    return Tensor[Float64](out, x._shape.copy(), strides,x._offset)


@always_inline
fn to_float64(x: Tensor[Int]) -> Tensor[Float64]:
    var src = x._data.copy()
    var n = len(src)

    var out = List[Float64]()
    out.reserve(n)

    var i = 0
    var lim = (n // 16) * 16
    while i < lim:
        out.append(Float64(src[i    ]))
        out.append(Float64(src[i + 1]))
        out.append(Float64(src[i + 2]))
        out.append(Float64(src[i + 3]))
        out.append(Float64(src[i + 4]))
        out.append(Float64(src[i + 5]))
        out.append(Float64(src[i + 6]))
        out.append(Float64(src[i + 7]))
        out.append(Float64(src[i + 8]))
        out.append(Float64(src[i + 9]))
        out.append(Float64(src[i + 10]))
        out.append(Float64(src[i + 11]))
        out.append(Float64(src[i + 12]))
        out.append(Float64(src[i + 13]))
        out.append(Float64(src[i + 14]))
        out.append(Float64(src[i + 15]))
        i += 16

    while i < n:
        out.append(Float64(src[i]))
        i += 1

    var strides = compute_row_major_strides(x._shape)
    return Tensor[Float64](out, x._shape.copy(), strides,x._offset)


# Unsigneds
@always_inline
fn to_float64(x: Tensor[UInt64]) -> Tensor[Float64]:
    var src = x._data.copy()
    var n = len(src)

    var out = List[Float64]()
    out.reserve(n)

    var i = 0
    var lim = (n // 16) * 16
    while i < lim:
        out.append(Float64(src[i    ]))
        out.append(Float64(src[i + 1]))
        out.append(Float64(src[i + 2]))
        out.append(Float64(src[i + 3]))
        out.append(Float64(src[i + 4]))
        out.append(Float64(src[i + 5]))
        out.append(Float64(src[i + 6]))
        out.append(Float64(src[i + 7]))
        out.append(Float64(src[i + 8]))
        out.append(Float64(src[i + 9]))
        out.append(Float64(src[i + 10]))
        out.append(Float64(src[i + 11]))
        out.append(Float64(src[i + 12]))
        out.append(Float64(src[i + 13]))
        out.append(Float64(src[i + 14]))
        out.append(Float64(src[i + 15]))
        i += 16

    while i < n:
        out.append(Float64(src[i]))
        i += 1

    var strides = compute_row_major_strides(x._shape)
    return Tensor[Float64](out, x._shape.copy(), strides,x._offset)


@always_inline
fn to_float64(x: Tensor[UInt32]) -> Tensor[Float64]:
    var src = x._data.copy()
    var n = len(src)

    var out = List[Float64]()
    out.reserve(n)

    var i = 0
    var lim = (n // 16) * 16
    while i < lim:
        out.append(Float64(src[i    ]))
        out.append(Float64(src[i + 1]))
        out.append(Float64(src[i + 2]))
        out.append(Float64(src[i + 3]))
        out.append(Float64(src[i + 4]))
        out.append(Float64(src[i + 5]))
        out.append(Float64(src[i + 6]))
        out.append(Float64(src[i + 7]))
        out.append(Float64(src[i + 8]))
        out.append(Float64(src[i + 9]))
        out.append(Float64(src[i + 10]))
        out.append(Float64(src[i + 11]))
        out.append(Float64(src[i + 12]))
        out.append(Float64(src[i + 13]))
        out.append(Float64(src[i + 14]))
        out.append(Float64(src[i + 15]))
        i += 16

    while i < n:
        out.append(Float64(src[i]))
        i += 1

    var strides = compute_row_major_strides(x._shape)
    return Tensor[Float64](out, x._shape.copy(), strides,x._offset)


@always_inline
fn to_float64(x: Tensor[UInt16]) -> Tensor[Float64]:
    var src = x._data.copy()
    var n = len(src)

    var out = List[Float64]()
    out.reserve(n)

    var i = 0
    var lim = (n // 16) * 16
    while i < lim:
        out.append(Float64(src[i    ]))
        out.append(Float64(src[i + 1]))
        out.append(Float64(src[i + 2]))
        out.append(Float64(src[i + 3]))
        out.append(Float64(src[i + 4]))
        out.append(Float64(src[i + 5]))
        out.append(Float64(src[i + 6]))
        out.append(Float64(src[i + 7]))
        out.append(Float64(src[i + 8]))
        out.append(Float64(src[i + 9]))
        out.append(Float64(src[i + 10]))
        out.append(Float64(src[i + 11]))
        out.append(Float64(src[i + 12]))
        out.append(Float64(src[i + 13]))
        out.append(Float64(src[i + 14]))
        out.append(Float64(src[i + 15]))
        i += 16

    while i < n:
        out.append(Float64(src[i]))
        i += 1

    var strides = compute_row_major_strides(x._shape)
    return Tensor[Float64](out, x._shape.copy(), strides,x._offset)


@always_inline
fn to_float64(x: Tensor[UInt8]) -> Tensor[Float64]:
    var src = x._data.copy()
    var n = len(src)

    var out = List[Float64]()
    out.reserve(n)

    var i = 0
    var lim = (n // 16) * 16
    while i < lim:
        out.append(Float64(src[i    ]))
        out.append(Float64(src[i + 1]))
        out.append(Float64(src[i + 2]))
        out.append(Float64(src[i + 3]))
        out.append(Float64(src[i + 4]))
        out.append(Float64(src[i + 5]))
        out.append(Float64(src[i + 6]))
        out.append(Float64(src[i + 7]))
        out.append(Float64(src[i + 8]))
        out.append(Float64(src[i + 9]))
        out.append(Float64(src[i + 10]))
        out.append(Float64(src[i + 11]))
        out.append(Float64(src[i + 12]))
        out.append(Float64(src[i + 13]))
        out.append(Float64(src[i + 14]))
        out.append(Float64(src[i + 15]))
        i += 16

    while i < n:
        out.append(Float64(src[i]))
        i += 1

    var strides = compute_row_major_strides(x._shape)
    return Tensor[Float64](out, x._shape.copy(), strides,x._offset)


# Bool
@always_inline
fn to_float64(x: Tensor[Bool]) -> Tensor[Float64]:
    # False → 0.0, True → 1.0
    var src = x._data.copy()
    var n = len(src)

    var out = List[Float64]()
    out.reserve(n)

    var i = 0
    var lim = (n // 16) * 16
    while i < lim:
        var v0 = 0.0
        if src[i    ]:
            v0 = 1.0
        var v1 = 0.0
        if src[i + 1]:
            v1 = 1.0
        var v2 = 0.0
        if src[i + 2]:
            v2 = 1.0
        var v3 = 0.0
        if src[i + 3]:
            v3 = 1.0
        var v4 = 0.0
        if src[i + 4]:
            v4 = 1.0
        var v5 = 0.0
        if src[i + 5]:
            v5 = 1.0
        var v6 = 0.0
        if src[i + 6]:
            v6 = 1.0
        var v7 = 0.0
        if src[i + 7]:
            v7 = 1.0
        var v8 = 0.0
        if src[i + 8]:
            v8 = 1.0
        var v9 = 0.0
        if src[i + 9]:
            v9 = 1.0
        var v10 = 0.0
        if src[i + 10]:
            v10 = 1.0
        var v11 = 0.0
        if src[i + 11]:
            v11 = 1.0
        var v12 = 0.0
        if src[i + 12]:
            v12 = 1.0
        var v13 = 0.0
        if src[i + 13]:
            v13 = 1.0
        var v14 = 0.0
        if src[i + 14]:
            v14 = 1.0
        var v15 = 0.0
        if src[i + 15]:
            v15 = 1.0

        out.append(v0)
        out.append(v1)
        out.append(v2)
        out.append(v3)
        out.append(v4)
        out.append(v5)
        out.append(v6)
        out.append(v7)
        out.append(v8)
        out.append(v9)
        out.append(v10)
        out.append(v11)
        out.append(v12)
        out.append(v13)
        out.append(v14)
        out.append(v15)
        i += 16

    while i < n:
        var vv = 0.0
        if src[i]:
            vv = 1.0
        out.append(vv)
        i += 1

    var strides = compute_row_major_strides(x._shape)
    return Tensor[Float64](out, x._shape.copy(), strides,x._offset)


# String (no parsing ⇒ zeros)
@always_inline
fn to_float64(x: Tensor[String]) -> Tensor[Float64]:
    var n = len(x._data)

    var out = List[Float64]()
    out.reserve(n)

    var i = 0
    var lim = (n // 16) * 16
    while i < lim:
        out.append(0.0)
        out.append(0.0)
        out.append(0.0)
        out.append(0.0)
        out.append(0.0)
        out.append(0.0)
        out.append(0.0)
        out.append(0.0)
        out.append(0.0)
        out.append(0.0)
        out.append(0.0)
        out.append(0.0)
        out.append(0.0)
        out.append(0.0)
        out.append(0.0)
        out.append(0.0)
        i += 16

    while i < n:
        out.append(0.0)
        i += 1

    var strides = compute_row_major_strides(x._shape)
    return Tensor[Float64](out, x._shape.copy(), strides,x._offset)



##################################################################
 
@always_inline
fn to_float32(x: Tensor[Float32]) -> Tensor[Float32]:
    # Identity: copy as-is.
    var data = x._data.copy()
    var strides = compute_row_major_strides(x._shape)
    return Tensor[Float32](data, x._shape.copy(), strides,x._offset)


@always_inline
fn to_float32(x: Tensor[Float64]) -> Tensor[Float32]:
    # Narrow Float64 → Float32 (may lose precision).
    var src = x._data.copy()
    var n = len(src)

    var out = List[Float32]()
    out.reserve(n)

    var i = 0
    var lim = (n // 16) * 16
    while i < lim:
        out.append(Float32(src[i    ]))
        out.append(Float32(src[i + 1]))
        out.append(Float32(src[i + 2]))
        out.append(Float32(src[i + 3]))
        out.append(Float32(src[i + 4]))
        out.append(Float32(src[i + 5]))
        out.append(Float32(src[i + 6]))
        out.append(Float32(src[i + 7]))
        out.append(Float32(src[i + 8]))
        out.append(Float32(src[i + 9]))
        out.append(Float32(src[i + 10]))
        out.append(Float32(src[i + 11]))
        out.append(Float32(src[i + 12]))
        out.append(Float32(src[i + 13]))
        out.append(Float32(src[i + 14]))
        out.append(Float32(src[i + 15]))
        i += 16

    while i < n:
        out.append(Float32(src[i]))
        i += 1

    var strides = compute_row_major_strides(x._shape)
    return Tensor[Float32](out, x._shape.copy(), strides,x._offset)


@always_inline
fn to_float32(x: Tensor[Int64]) -> Tensor[Float32]:
    # Int64 → Float32 (may lose precision for large magnitudes).
    var src = x._data.copy()
    var n = len(src)

    var out = List[Float32]()
    out.reserve(n)

    var i = 0
    var lim = (n // 16) * 16
    while i < lim:
        out.append(Float32(src[i    ]))
        out.append(Float32(src[i + 1]))
        out.append(Float32(src[i + 2]))
        out.append(Float32(src[i + 3]))
        out.append(Float32(src[i + 4]))
        out.append(Float32(src[i + 5]))
        out.append(Float32(src[i + 6]))
        out.append(Float32(src[i + 7]))
        out.append(Float32(src[i + 8]))
        out.append(Float32(src[i + 9]))
        out.append(Float32(src[i + 10]))
        out.append(Float32(src[i + 11]))
        out.append(Float32(src[i + 12]))
        out.append(Float32(src[i + 13]))
        out.append(Float32(src[i + 14]))
        out.append(Float32(src[i + 15]))
        i += 16

    while i < n:
        out.append(Float32(src[i]))
        i += 1

    var strides = compute_row_major_strides(x._shape)
    return Tensor[Float32](out, x._shape.copy(), strides,x._offset)


@always_inline
fn to_float32(x: Tensor[Int32]) -> Tensor[Float32]:
    var src = x._data.copy()
    var n = len(src)

    var out = List[Float32]()
    out.reserve(n)

    var i = 0
    var lim = (n // 16) * 16
    while i < lim:
        out.append(Float32(src[i    ]))
        out.append(Float32(src[i + 1]))
        out.append(Float32(src[i + 2]))
        out.append(Float32(src[i + 3]))
        out.append(Float32(src[i + 4]))
        out.append(Float32(src[i + 5]))
        out.append(Float32(src[i + 6]))
        out.append(Float32(src[i + 7]))
        out.append(Float32(src[i + 8]))
        out.append(Float32(src[i + 9]))
        out.append(Float32(src[i + 10]))
        out.append(Float32(src[i + 11]))
        out.append(Float32(src[i + 12]))
        out.append(Float32(src[i + 13]))
        out.append(Float32(src[i + 14]))
        out.append(Float32(src[i + 15]))
        i += 16

    while i < n:
        out.append(Float32(src[i]))
        i += 1

    var strides = compute_row_major_strides(x._shape)
    return Tensor[Float32](out, x._shape.copy(), strides,x._offset)


@always_inline
fn to_float32(x: Tensor[Int16]) -> Tensor[Float32]:
    var src = x._data.copy()
    var n = len(src)

    var out = List[Float32]()
    out.reserve(n)

    var i = 0
    var lim = (n // 16) * 16
    while i < lim:
        out.append(Float32(src[i    ]))
        out.append(Float32(src[i + 1]))
        out.append(Float32(src[i + 2]))
        out.append(Float32(src[i + 3]))
        out.append(Float32(src[i + 4]))
        out.append(Float32(src[i + 5]))
        out.append(Float32(src[i + 6]))
        out.append(Float32(src[i + 7]))
        out.append(Float32(src[i + 8]))
        out.append(Float32(src[i + 9]))
        out.append(Float32(src[i + 10]))
        out.append(Float32(src[i + 11]))
        out.append(Float32(src[i + 12]))
        out.append(Float32(src[i + 13]))
        out.append(Float32(src[i + 14]))
        out.append(Float32(src[i + 15]))
        i += 16

    while i < n:
        out.append(Float32(src[i]))
        i += 1

    var strides = compute_row_major_strides(x._shape)
    return Tensor[Float32](out, x._shape.copy(), strides,x._offset)


@always_inline
fn to_float32(x: Tensor[Int8]) -> Tensor[Float32]:
    var src = x._data.copy()
    var n = len(src)

    var out = List[Float32]()
    out.reserve(n)

    var i = 0
    var lim = (n // 16) * 16
    while i < lim:
        out.append(Float32(src[i    ]))
        out.append(Float32(src[i + 1]))
        out.append(Float32(src[i + 2]))
        out.append(Float32(src[i + 3]))
        out.append(Float32(src[i + 4]))
        out.append(Float32(src[i + 5]))
        out.append(Float32(src[i + 6]))
        out.append(Float32(src[i + 7]))
        out.append(Float32(src[i + 8]))
        out.append(Float32(src[i + 9]))
        out.append(Float32(src[i + 10]))
        out.append(Float32(src[i + 11]))
        out.append(Float32(src[i + 12]))
        out.append(Float32(src[i + 13]))
        out.append(Float32(src[i + 14]))
        out.append(Float32(src[i + 15]))
        i += 16

    while i < n:
        out.append(Float32(src[i]))
        i += 1

    var strides = compute_row_major_strides(x._shape)
    return Tensor[Float32](out, x._shape.copy(), strides,x._offset)


@always_inline
fn to_float32(x: Tensor[Int]) -> Tensor[Float32]:
    var src = x._data.copy()
    var n = len(src)

    var out = List[Float32]()
    out.reserve(n)

    var i = 0
    var lim = (n // 16) * 16
    while i < lim:
        out.append(Float32(src[i    ]))
        out.append(Float32(src[i + 1]))
        out.append(Float32(src[i + 2]))
        out.append(Float32(src[i + 3]))
        out.append(Float32(src[i + 4]))
        out.append(Float32(src[i + 5]))
        out.append(Float32(src[i + 6]))
        out.append(Float32(src[i + 7]))
        out.append(Float32(src[i + 8]))
        out.append(Float32(src[i + 9]))
        out.append(Float32(src[i + 10]))
        out.append(Float32(src[i + 11]))
        out.append(Float32(src[i + 12]))
        out.append(Float32(src[i + 13]))
        out.append(Float32(src[i + 14]))
        out.append(Float32(src[i + 15]))
        i += 16

    while i < n:
        out.append(Float32(src[i]))
        i += 1

    var strides = compute_row_major_strides(x._shape)
    return Tensor[Float32](out, x._shape.copy(), strides,x._offset)


@always_inline
fn to_float32(x: Tensor[UInt64]) -> Tensor[Float32]:
    var src = x._data.copy()
    var n = len(src)

    var out = List[Float32]()
    out.reserve(n)

    var i = 0
    var lim = (n // 16) * 16
    while i < lim:
        out.append(Float32(src[i    ]))
        out.append(Float32(src[i + 1]))
        out.append(Float32(src[i + 2]))
        out.append(Float32(src[i + 3]))
        out.append(Float32(src[i + 4]))
        out.append(Float32(src[i + 5]))
        out.append(Float32(src[i + 6]))
        out.append(Float32(src[i + 7]))
        out.append(Float32(src[i + 8]))
        out.append(Float32(src[i + 9]))
        out.append(Float32(src[i + 10]))
        out.append(Float32(src[i + 11]))
        out.append(Float32(src[i + 12]))
        out.append(Float32(src[i + 13]))
        out.append(Float32(src[i + 14]))
        out.append(Float32(src[i + 15]))
        i += 16

    while i < n:
        out.append(Float32(src[i]))
        i += 1

    var strides = compute_row_major_strides(x._shape)
    return Tensor[Float32](out, x._shape.copy(), strides,x._offset)


@always_inline
fn to_float32(x: Tensor[UInt32]) -> Tensor[Float32]:
    var src = x._data.copy()
    var n = len(src)

    var out = List[Float32]()
    out.reserve(n)

    var i = 0
    var lim = (n // 16) * 16
    while i < lim:
        out.append(Float32(src[i    ]))
        out.append(Float32(src[i + 1]))
        out.append(Float32(src[i + 2]))
        out.append(Float32(src[i + 3]))
        out.append(Float32(src[i + 4]))
        out.append(Float32(src[i + 5]))
        out.append(Float32(src[i + 6]))
        out.append(Float32(src[i + 7]))
        out.append(Float32(src[i + 8]))
        out.append(Float32(src[i + 9]))
        out.append(Float32(src[i + 10]))
        out.append(Float32(src[i + 11]))
        out.append(Float32(src[i + 12]))
        out.append(Float32(src[i + 13]))
        out.append(Float32(src[i + 14]))
        out.append(Float32(src[i + 15]))
        i += 16

    while i < n:
        out.append(Float32(src[i]))
        i += 1

    var strides = compute_row_major_strides(x._shape)
    return Tensor[Float32](out, x._shape.copy(), strides,x._offset)


@always_inline
fn to_float32(x: Tensor[UInt16]) -> Tensor[Float32]:
    var src = x._data.copy()
    var n = len(src)

    var out = List[Float32]()
    out.reserve(n)

    var i = 0
    var lim = (n // 16) * 16
    while i < lim:
        out.append(Float32(src[i    ]))
        out.append(Float32(src[i + 1]))
        out.append(Float32(src[i + 2]))
        out.append(Float32(src[i + 3]))
        out.append(Float32(src[i + 4]))
        out.append(Float32(src[i + 5]))
        out.append(Float32(src[i + 6]))
        out.append(Float32(src[i + 7]))
        out.append(Float32(src[i + 8]))
        out.append(Float32(src[i + 9]))
        out.append(Float32(src[i + 10]))
        out.append(Float32(src[i + 11]))
        out.append(Float32(src[i + 12]))
        out.append(Float32(src[i + 13]))
        out.append(Float32(src[i + 14]))
        out.append(Float32(src[i + 15]))
        i += 16

    while i < n:
        out.append(Float32(src[i]))
        i += 1

    var strides = compute_row_major_strides(x._shape)
    return Tensor[Float32](out, x._shape.copy(), strides,x._offset)


@always_inline
fn to_float32(x: Tensor[UInt8]) -> Tensor[Float32]:
    var src = x._data.copy()
    var n = len(src)

    var out = List[Float32]()
    out.reserve(n)

    var i = 0
    var lim = (n // 16) * 16
    while i < lim:
        out.append(Float32(src[i    ]))
        out.append(Float32(src[i + 1]))
        out.append(Float32(src[i + 2]))
        out.append(Float32(src[i + 3]))
        out.append(Float32(src[i + 4]))
        out.append(Float32(src[i + 5]))
        out.append(Float32(src[i + 6]))
        out.append(Float32(src[i + 7]))
        out.append(Float32(src[i + 8]))
        out.append(Float32(src[i + 9]))
        out.append(Float32(src[i + 10]))
        out.append(Float32(src[i + 11]))
        out.append(Float32(src[i + 12]))
        out.append(Float32(src[i + 13]))
        out.append(Float32(src[i + 14]))
        out.append(Float32(src[i + 15]))
        i += 16

    while i < n:
        out.append(Float32(src[i]))
        i += 1

    var strides = compute_row_major_strides(x._shape)
    return Tensor[Float32](out, x._shape.copy(), strides,x._offset)


@always_inline
fn to_float32(x: Tensor[Bool]) -> Tensor[Float32]:
    # False → 0.0f, True → 1.0f
    var src = x._data.copy()
    var n = len(src)

    var out = List[Float32]()
    out.reserve(n)

    var i = 0
    var lim = (n // 16) * 16
    while i < lim:
        var v0 = Float32(0.0)
        if src[i    ]:
            v0 = Float32(1.0)
        var v1 = Float32(0.0)
        if src[i + 1]:
            v1 = Float32(1.0)
        var v2 = Float32(0.0)
        if src[i + 2]:
            v2 = Float32(1.0)
        var v3 = Float32(0.0)
        if src[i + 3]:
            v3 = Float32(1.0)
        var v4 = Float32(0.0)
        if src[i + 4]:
            v4 = Float32(1.0)
        var v5 = Float32(0.0)
        if src[i + 5]:
            v5 = Float32(1.0)
        var v6 = Float32(0.0)
        if src[i + 6]:
            v6 = Float32(1.0)
        var v7 = Float32(0.0)
        if src[i + 7]:
            v7 = Float32(1.0)
        var v8 = Float32(0.0)
        if src[i + 8]:
            v8 = Float32(1.0)
        var v9 = Float32(0.0)
        if src[i + 9]:
            v9 = Float32(1.0)
        var v10 = Float32(0.0)
        if src[i + 10]:
            v10 = Float32(1.0)
        var v11 = Float32(0.0)
        if src[i + 11]:
            v11 = Float32(1.0)
        var v12 = Float32(0.0)
        if src[i + 12]:
            v12 = Float32(1.0)
        var v13 = Float32(0.0)
        if src[i + 13]:
            v13 = Float32(1.0)
        var v14 = Float32(0.0)
        if src[i + 14]:
            v14 = Float32(1.0)
        var v15 = Float32(0.0)
        if src[i + 15]:
            v15 = Float32(1.0)

        out.append(v0)
        out.append(v1)
        out.append(v2)
        out.append(v3)
        out.append(v4)
        out.append(v5)
        out.append(v6)
        out.append(v7)
        out.append(v8)
        out.append(v9)
        out.append(v10)
        out.append(v11)
        out.append(v12)
        out.append(v13)
        out.append(v14)
        out.append(v15)
        i += 16

    while i < n:
        var vv = Float32(0.0)
        if src[i]:
            vv = Float32(1.0)
        out.append(vv)
        i += 1

    var strides = compute_row_major_strides(x._shape)
    return Tensor[Float32](out, x._shape.copy(), strides,x._offset)


@always_inline
fn to_float32(x: Tensor[String]) -> Tensor[Float32]:
    # No parsing here: fill zeros.
    var n = len(x._data)

    var out = List[Float32]()
    out.reserve(n)

    var i = 0
    var lim = (n // 16) * 16
    while i < lim:
        out.append(Float32(0.0))
        out.append(Float32(0.0))
        out.append(Float32(0.0))
        out.append(Float32(0.0))
        out.append(Float32(0.0))
        out.append(Float32(0.0))
        out.append(Float32(0.0))
        out.append(Float32(0.0))
        out.append(Float32(0.0))
        out.append(Float32(0.0))
        out.append(Float32(0.0))
        out.append(Float32(0.0))
        out.append(Float32(0.0))
        out.append(Float32(0.0))
        out.append(Float32(0.0))
        out.append(Float32(0.0))
        i += 16

    while i < n:
        out.append(Float32(0.0))
        i += 1

    var strides = compute_row_major_strides(x._shape)
    return Tensor[Float32](out, x._shape.copy(), strides,x._offset)

# ===================== Cast to Tensor[Int] =====================
@always_inline
fn to_int(x: Tensor[Float64]) -> Tensor[Int]:
    # Convert Float64 tensor to Int by truncating toward zero.
    var src = x._data.copy()
    var n = len(src)

    var out = List[Int]()
    out.reserve(n)

    var i = 0
    var lim = (n // 16) * 16
    while i < lim:
        out.append(Int(src[i    ]))
        out.append(Int(src[i + 1]))
        out.append(Int(src[i + 2]))
        out.append(Int(src[i + 3]))
        out.append(Int(src[i + 4]))
        out.append(Int(src[i + 5]))
        out.append(Int(src[i + 6]))
        out.append(Int(src[i + 7]))
        out.append(Int(src[i + 8]))
        out.append(Int(src[i + 9]))
        out.append(Int(src[i + 10]))
        out.append(Int(src[i + 11]))
        out.append(Int(src[i + 12]))
        out.append(Int(src[i + 13]))
        out.append(Int(src[i + 14]))
        out.append(Int(src[i + 15]))
        i += 16

    while i < n:
        out.append(Int(src[i]))
        i += 1

    var strides = compute_row_major_strides(x._shape)
    return Tensor[Int](out, x._shape.copy(), strides,x._offset)


@always_inline
fn to_int(x: Tensor[Float32]) -> Tensor[Int]:
    # Convert Float32 tensor to Int by truncating toward zero.
    var src = x._data.copy()
    var n = len(src)

    var out = List[Int]()
    out.reserve(n)

    var i = 0
    var lim = (n // 16) * 16
    while i < lim:
        out.append(Int(src[i    ]))
        out.append(Int(src[i + 1]))
        out.append(Int(src[i + 2]))
        out.append(Int(src[i + 3]))
        out.append(Int(src[i + 4]))
        out.append(Int(src[i + 5]))
        out.append(Int(src[i + 6]))
        out.append(Int(src[i + 7]))
        out.append(Int(src[i + 8]))
        out.append(Int(src[i + 9]))
        out.append(Int(src[i + 10]))
        out.append(Int(src[i + 11]))
        out.append(Int(src[i + 12]))
        out.append(Int(src[i + 13]))
        out.append(Int(src[i + 14]))
        out.append(Int(src[i + 15]))
        i += 16

    while i < n:
        out.append(Int(src[i]))
        i += 1

    var strides = compute_row_major_strides(x._shape)
    return Tensor[Int](out, x._shape.copy(), strides,x._offset)


@always_inline
fn to_int(x: Tensor[Int64]) -> Tensor[Int]:
    # Narrow cast to platform Int (may truncate on platforms where Int < 64 bits).
    var src = x._data.copy()
    var n = len(src)

    var out = List[Int]()
    out.reserve(n)

    var i = 0
    var lim = (n // 16) * 16
    while i < lim:
        out.append(Int(src[i    ]))
        out.append(Int(src[i + 1]))
        out.append(Int(src[i + 2]))
        out.append(Int(src[i + 3]))
        out.append(Int(src[i + 4]))
        out.append(Int(src[i + 5]))
        out.append(Int(src[i + 6]))
        out.append(Int(src[i + 7]))
        out.append(Int(src[i + 8]))
        out.append(Int(src[i + 9]))
        out.append(Int(src[i + 10]))
        out.append(Int(src[i + 11]))
        out.append(Int(src[i + 12]))
        out.append(Int(src[i + 13]))
        out.append(Int(src[i + 14]))
        out.append(Int(src[i + 15]))
        i += 16

    while i < n:
        out.append(Int(src[i]))
        i += 1

    var strides = compute_row_major_strides(x._shape)
    return Tensor[Int](out, x._shape.copy(), strides,x._offset)


@always_inline
fn to_int(x: Tensor[Int32]) -> Tensor[Int]:
    var src = x._data.copy()
    var n = len(src)

    var out = List[Int]()
    out.reserve(n)

    var i = 0
    var lim = (n // 16) * 16
    while i < lim:
        out.append(Int(src[i    ]))
        out.append(Int(src[i + 1]))
        out.append(Int(src[i + 2]))
        out.append(Int(src[i + 3]))
        out.append(Int(src[i + 4]))
        out.append(Int(src[i + 5]))
        out.append(Int(src[i + 6]))
        out.append(Int(src[i + 7]))
        out.append(Int(src[i + 8]))
        out.append(Int(src[i + 9]))
        out.append(Int(src[i + 10]))
        out.append(Int(src[i + 11]))
        out.append(Int(src[i + 12]))
        out.append(Int(src[i + 13]))
        out.append(Int(src[i + 14]))
        out.append(Int(src[i + 15]))
        i += 16

    while i < n:
        out.append(Int(src[i]))
        i += 1

    var strides = compute_row_major_strides(x._shape)
    return Tensor[Int](out, x._shape.copy(), strides,x._offset)


@always_inline
fn to_int(x: Tensor[Int16]) -> Tensor[Int]:
    var src = x._data.copy()
    var n = len(src)

    var out = List[Int]()
    out.reserve(n)

    var i = 0
    var lim = (n // 16) * 16
    while i < lim:
        out.append(Int(src[i    ]))
        out.append(Int(src[i + 1]))
        out.append(Int(src[i + 2]))
        out.append(Int(src[i + 3]))
        out.append(Int(src[i + 4]))
        out.append(Int(src[i + 5]))
        out.append(Int(src[i + 6]))
        out.append(Int(src[i + 7]))
        out.append(Int(src[i + 8]))
        out.append(Int(src[i + 9]))
        out.append(Int(src[i + 10]))
        out.append(Int(src[i + 11]))
        out.append(Int(src[i + 12]))
        out.append(Int(src[i + 13]))
        out.append(Int(src[i + 14]))
        out.append(Int(src[i + 15]))
        i += 16

    while i < n:
        out.append(Int(src[i]))
        i += 1

    var strides = compute_row_major_strides(x._shape)
    return Tensor[Int](out, x._shape.copy(), strides,x._offset)


@always_inline
fn to_int(x: Tensor[Int8]) -> Tensor[Int]:
    var src = x._data.copy()
    var n = len(src)

    var out = List[Int]()
    out.reserve(n)

    var i = 0
    var lim = (n // 16) * 16
    while i < lim:
        out.append(Int(src[i    ]))
        out.append(Int(src[i + 1]))
        out.append(Int(src[i + 2]))
        out.append(Int(src[i + 3]))
        out.append(Int(src[i + 4]))
        out.append(Int(src[i + 5]))
        out.append(Int(src[i + 6]))
        out.append(Int(src[i + 7]))
        out.append(Int(src[i + 8]))
        out.append(Int(src[i + 9]))
        out.append(Int(src[i + 10]))
        out.append(Int(src[i + 11]))
        out.append(Int(src[i + 12]))
        out.append(Int(src[i + 13]))
        out.append(Int(src[i + 14]))
        out.append(Int(src[i + 15]))
        i += 16

    while i < n:
        out.append(Int(src[i]))
        i += 1

    var strides = compute_row_major_strides(x._shape)
    return Tensor[Int](out, x._shape.copy(), strides,x._offset)


@always_inline
fn to_int(x: Tensor[Int]) -> Tensor[Int]:
    # Identity-width cast: copy values into a new Int list.
    var src = x._data.copy()
    var n = len(src)

    var out = List[Int]()
    out.reserve(n)

    var i = 0
    var lim = (n // 16) * 16
    while i < lim:
        out.append(src[i    ])
        out.append(src[i + 1])
        out.append(src[i + 2])
        out.append(src[i + 3])
        out.append(src[i + 4])
        out.append(src[i + 5])
        out.append(src[i + 6])
        out.append(src[i + 7])
        out.append(src[i + 8])
        out.append(src[i + 9])
        out.append(src[i + 10])
        out.append(src[i + 11])
        out.append(src[i + 12])
        out.append(src[i + 13])
        out.append(src[i + 14])
        out.append(src[i + 15])
        i += 16

    while i < n:
        out.append(src[i])
        i += 1

    var strides = compute_row_major_strides(x._shape)
    return Tensor[Int](out, x._shape.copy(), strides,x._offset)


# Unsigneds → Int (note: may overflow if source > Int.max on the target platform).
@always_inline
fn to_int(x: Tensor[UInt64]) -> Tensor[Int]:
    var src = x._data.copy()
    var n = len(src)

    var out = List[Int]()
    out.reserve(n)

    var i = 0
    var lim = (n // 16) * 16
    while i < lim:
        out.append(Int(src[i    ]))
        out.append(Int(src[i + 1]))
        out.append(Int(src[i + 2]))
        out.append(Int(src[i + 3]))
        out.append(Int(src[i + 4]))
        out.append(Int(src[i + 5]))
        out.append(Int(src[i + 6]))
        out.append(Int(src[i + 7]))
        out.append(Int(src[i + 8]))
        out.append(Int(src[i + 9]))
        out.append(Int(src[i + 10]))
        out.append(Int(src[i + 11]))
        out.append(Int(src[i + 12]))
        out.append(Int(src[i + 13]))
        out.append(Int(src[i + 14]))
        out.append(Int(src[i + 15]))
        i += 16

    while i < n:
        out.append(Int(src[i]))
        i += 1

    var strides = compute_row_major_strides(x._shape)
    return Tensor[Int](out, x._shape.copy(), strides,x._offset)


@always_inline
fn to_int(x: Tensor[UInt32]) -> Tensor[Int]:
    var src = x._data.copy()
    var n = len(src)

    var out = List[Int]()
    out.reserve(n)

    var i = 0
    var lim = (n // 16) * 16
    while i < lim:
        out.append(Int(src[i    ]))
        out.append(Int(src[i + 1]))
        out.append(Int(src[i + 2]))
        out.append(Int(src[i + 3]))
        out.append(Int(src[i + 4]))
        out.append(Int(src[i + 5]))
        out.append(Int(src[i + 6]))
        out.append(Int(src[i + 7]))
        out.append(Int(src[i + 8]))
        out.append(Int(src[i + 9]))
        out.append(Int(src[i + 10]))
        out.append(Int(src[i + 11]))
        out.append(Int(src[i + 12]))
        out.append(Int(src[i + 13]))
        out.append(Int(src[i + 14]))
        out.append(Int(src[i + 15]))
        i += 16

    while i < n:
        out.append(Int(src[i]))
        i += 1

    var strides = compute_row_major_strides(x._shape)
    return Tensor[Int](out, x._shape.copy(), strides,x._offset)


@always_inline
fn to_int(x: Tensor[UInt16]) -> Tensor[Int]:
    var src = x._data.copy()
    var n = len(src)

    var out = List[Int]()
    out.reserve(n)

    var i = 0
    var lim = (n // 16) * 16
    while i < lim:
        out.append(Int(src[i    ]))
        out.append(Int(src[i + 1]))
        out.append(Int(src[i + 2]))
        out.append(Int(src[i + 3]))
        out.append(Int(src[i + 4]))
        out.append(Int(src[i + 5]))
        out.append(Int(src[i + 6]))
        out.append(Int(src[i + 7]))
        out.append(Int(src[i + 8]))
        out.append(Int(src[i + 9]))
        out.append(Int(src[i + 10]))
        out.append(Int(src[i + 11]))
        out.append(Int(src[i + 12]))
        out.append(Int(src[i + 13]))
        out.append(Int(src[i + 14]))
        out.append(Int(src[i + 15]))
        i += 16

    while i < n:
        out.append(Int(src[i]))
        i += 1

    var strides = compute_row_major_strides(x._shape)
    return Tensor[Int](out, x._shape.copy(), strides,x._offset)


@always_inline
fn to_int(x: Tensor[UInt8]) -> Tensor[Int]:
    var src = x._data.copy()
    var n = len(src)

    var out = List[Int]()
    out.reserve(n)

    var i = 0
    var lim = (n // 16) * 16
    while i < lim:
        out.append(Int(src[i    ]))
        out.append(Int(src[i + 1]))
        out.append(Int(src[i + 2]))
        out.append(Int(src[i + 3]))
        out.append(Int(src[i + 4]))
        out.append(Int(src[i + 5]))
        out.append(Int(src[i + 6]))
        out.append(Int(src[i + 7]))
        out.append(Int(src[i + 8]))
        out.append(Int(src[i + 9]))
        out.append(Int(src[i + 10]))
        out.append(Int(src[i + 11]))
        out.append(Int(src[i + 12]))
        out.append(Int(src[i + 13]))
        out.append(Int(src[i + 14]))
        out.append(Int(src[i + 15]))
        i += 16

    while i < n:
        out.append(Int(src[i]))
        i += 1

    var strides = compute_row_major_strides(x._shape)
    return Tensor[Int](out, x._shape.copy(), strides,x._offset)


# Bool → Int (False → 0, True → 1)
@always_inline
fn to_int(x: Tensor[Bool]) -> Tensor[Int]:
    var src = x._data.copy()
    var n = len(src)

    var out = List[Int]()
    out.reserve(n)

    var i = 0
    var lim = (n // 16) * 16
    while i < lim:
        var v0 = 0
        if src[i    ]:
            v0 = 1
        var v1 = 0
        if src[i + 1]:
            v1 = 1
        var v2 = 0
        if src[i + 2]:
            v2 = 1
        var v3 = 0
        if src[i + 3]:
            v3 = 1
        var v4 = 0
        if src[i + 4]:
            v4 = 1
        var v5 = 0
        if src[i + 5]:
            v5 = 1
        var v6 = 0
        if src[i + 6]:
            v6 = 1
        var v7 = 0
        if src[i + 7]:
            v7 = 1
        var v8 = 0
        if src[i + 8]:
            v8 = 1
        var v9 = 0
        if src[i + 9]:
            v9 = 1
        var v10 = 0
        if src[i + 10]:
            v10 = 1
        var v11 = 0
        if src[i + 11]:
            v11 = 1
        var v12 = 0
        if src[i + 12]:
            v12 = 1
        var v13 = 0
        if src[i + 13]:
            v13 = 1
        var v14 = 0
        if src[i + 14]:
            v14 = 1
        var v15 = 0
        if src[i + 15]:
            v15 = 1

        out.append(v0)
        out.append(v1)
        out.append(v2)
        out.append(v3)
        out.append(v4)
        out.append(v5)
        out.append(v6)
        out.append(v7)
        out.append(v8)
        out.append(v9)
        out.append(v10)
        out.append(v11)
        out.append(v12)
        out.append(v13)
        out.append(v14)
        out.append(v15)
        i += 16

    while i < n:
        var vv = 0
        if src[i]:
            vv = 1
        out.append(vv)
        i += 1

    var strides = compute_row_major_strides(x._shape)
    return Tensor[Int](out, x._shape.copy(), strides,x._offset)


# String → Int (no parsing here: fill zeros)
@always_inline
fn to_int(x: Tensor[String]) -> Tensor[Int]:
    var n = len(x._data)

    var out = List[Int]()
    out.reserve(n)

    var i = 0
    var lim = (n // 16) * 16
    while i < lim:
        out.append(0)
        out.append(0)
        out.append(0)
        out.append(0)
        out.append(0)
        out.append(0)
        out.append(0)
        out.append(0)
        out.append(0)
        out.append(0)
        out.append(0)
        out.append(0)
        out.append(0)
        out.append(0)
        out.append(0)
        out.append(0)
        i += 16

    while i < n:
        out.append(0)
        i += 1

    var strides = compute_row_major_strides(x._shape)
    return Tensor[Int](out, x._shape.copy(), strides,x._offset)

##########################################################################

@always_inline
fn to_int64(x: Tensor[Int64]) -> Tensor[Int64]:
    # Identity-width cast: copy as-is into Int64 list.
    var data = x._data.copy()
    var strides = compute_row_major_strides(x._shape)
    return Tensor[Int64](data, x._shape.copy(), strides,x._offset)


@always_inline
fn to_int64(x: Tensor[Float64]) -> Tensor[Int64]:
    # Truncate toward zero from Float64 to Int64.
    var src = x._data.copy()
    var n = len(src)

    var out = List[Int64]()
    out.reserve(n)

    var i = 0
    var lim = (n // 16) * 16
    while i < lim:
        out.append(Int64(src[i    ]))
        out.append(Int64(src[i + 1]))
        out.append(Int64(src[i + 2]))
        out.append(Int64(src[i + 3]))
        out.append(Int64(src[i + 4]))
        out.append(Int64(src[i + 5]))
        out.append(Int64(src[i + 6]))
        out.append(Int64(src[i + 7]))
        out.append(Int64(src[i + 8]))
        out.append(Int64(src[i + 9]))
        out.append(Int64(src[i + 10]))
        out.append(Int64(src[i + 11]))
        out.append(Int64(src[i + 12]))
        out.append(Int64(src[i + 13]))
        out.append(Int64(src[i + 14]))
        out.append(Int64(src[i + 15]))
        i += 16

    while i < n:
        out.append(Int64(src[i]))
        i += 1

    var strides = compute_row_major_strides(x._shape)
    return Tensor[Int64](out, x._shape.copy(), strides,x._offset)


@always_inline
fn to_int64(x: Tensor[Float32]) -> Tensor[Int64]:
    # Truncate toward zero from Float32 to Int64.
    var src = x._data.copy()
    var n = len(src)

    var out = List[Int64]()
    out.reserve(n)

    var i = 0
    var lim = (n // 16) * 16
    while i < lim:
        out.append(Int64(src[i    ]))
        out.append(Int64(src[i + 1]))
        out.append(Int64(src[i + 2]))
        out.append(Int64(src[i + 3]))
        out.append(Int64(src[i + 4]))
        out.append(Int64(src[i + 5]))
        out.append(Int64(src[i + 6]))
        out.append(Int64(src[i + 7]))
        out.append(Int64(src[i + 8]))
        out.append(Int64(src[i + 9]))
        out.append(Int64(src[i + 10]))
        out.append(Int64(src[i + 11]))
        out.append(Int64(src[i + 12]))
        out.append(Int64(src[i + 13]))
        out.append(Int64(src[i + 14]))
        out.append(Int64(src[i + 15]))
        i += 16

    while i < n:
        out.append(Int64(src[i]))
        i += 1

    var strides = compute_row_major_strides(x._shape)
    return Tensor[Int64](out, x._shape.copy(), strides,x._offset)


@always_inline
fn to_int64(x: Tensor[Int32]) -> Tensor[Int64]:
    # Widen Int32 → Int64.
    var src = x._data.copy()
    var n = len(src)

    var out = List[Int64]()
    out.reserve(n)

    var i = 0
    var lim = (n // 16) * 16
    while i < lim:
        out.append(Int64(src[i    ]))
        out.append(Int64(src[i + 1]))
        out.append(Int64(src[i + 2]))
        out.append(Int64(src[i + 3]))
        out.append(Int64(src[i + 4]))
        out.append(Int64(src[i + 5]))
        out.append(Int64(src[i + 6]))
        out.append(Int64(src[i + 7]))
        out.append(Int64(src[i + 8]))
        out.append(Int64(src[i + 9]))
        out.append(Int64(src[i + 10]))
        out.append(Int64(src[i + 11]))
        out.append(Int64(src[i + 12]))
        out.append(Int64(src[i + 13]))
        out.append(Int64(src[i + 14]))
        out.append(Int64(src[i + 15]))
        i += 16

    while i < n:
        out.append(Int64(src[i]))
        i += 1

    var strides = compute_row_major_strides(x._shape)
    return Tensor[Int64](out, x._shape.copy(), strides,x._offset)


@always_inline
fn to_int64(x: Tensor[Int16]) -> Tensor[Int64]:
    # Widen Int16 → Int64.
    var src = x._data.copy()
    var n = len(src)

    var out = List[Int64]()
    out.reserve(n)

    var i = 0
    var lim = (n // 16) * 16
    while i < lim:
        out.append(Int64(src[i    ]))
        out.append(Int64(src[i + 1]))
        out.append(Int64(src[i + 2]))
        out.append(Int64(src[i + 3]))
        out.append(Int64(src[i + 4]))
        out.append(Int64(src[i + 5]))
        out.append(Int64(src[i + 6]))
        out.append(Int64(src[i + 7]))
        out.append(Int64(src[i + 8]))
        out.append(Int64(src[i + 9]))
        out.append(Int64(src[i + 10]))
        out.append(Int64(src[i + 11]))
        out.append(Int64(src[i + 12]))
        out.append(Int64(src[i + 13]))
        out.append(Int64(src[i + 14]))
        out.append(Int64(src[i + 15]))
        i += 16

    while i < n:
        out.append(Int64(src[i]))
        i += 1

    var strides = compute_row_major_strides(x._shape)
    return Tensor[Int64](out, x._shape.copy(), strides,x._offset)


@always_inline
fn to_int64(x: Tensor[Int8]) -> Tensor[Int64]:
    # Widen Int8 → Int64.
    var src = x._data.copy()
    var n = len(src)

    var out = List[Int64]()
    out.reserve(n)

    var i = 0
    var lim = (n // 16) * 16
    while i < lim:
        out.append(Int64(src[i    ]))
        out.append(Int64(src[i + 1]))
        out.append(Int64(src[i + 2]))
        out.append(Int64(src[i + 3]))
        out.append(Int64(src[i + 4]))
        out.append(Int64(src[i + 5]))
        out.append(Int64(src[i + 6]))
        out.append(Int64(src[i + 7]))
        out.append(Int64(src[i + 8]))
        out.append(Int64(src[i + 9]))
        out.append(Int64(src[i + 10]))
        out.append(Int64(src[i + 11]))
        out.append(Int64(src[i + 12]))
        out.append(Int64(src[i + 13]))
        out.append(Int64(src[i + 14]))
        out.append(Int64(src[i + 15]))
        i += 16

    while i < n:
        out.append(Int64(src[i]))
        i += 1

    var strides = compute_row_major_strides(x._shape)
    return Tensor[Int64](out, x._shape.copy(), strides,x._offset)


@always_inline
fn to_int64(x: Tensor[Int]) -> Tensor[Int64]:
    # Widen platform Int → Int64 (may be no-op if Int == Int64).
    var src = x._data.copy()
    var n = len(src)

    var out = List[Int64]()
    out.reserve(n)

    var i = 0
    var lim = (n // 16) * 16
    while i < lim:
        out.append(Int64(src[i    ]))
        out.append(Int64(src[i + 1]))
        out.append(Int64(src[i + 2]))
        out.append(Int64(src[i + 3]))
        out.append(Int64(src[i + 4]))
        out.append(Int64(src[i + 5]))
        out.append(Int64(src[i + 6]))
        out.append(Int64(src[i + 7]))
        out.append(Int64(src[i + 8]))
        out.append(Int64(src[i + 9]))
        out.append(Int64(src[i + 10]))
        out.append(Int64(src[i + 11]))
        out.append(Int64(src[i + 12]))
        out.append(Int64(src[i + 13]))
        out.append(Int64(src[i + 14]))
        out.append(Int64(src[i + 15]))
        i += 16

    while i < n:
        out.append(Int64(src[i]))
        i += 1

    var strides = compute_row_major_strides(x._shape)
    return Tensor[Int64](out, x._shape.copy(), strides,x._offset)


@always_inline
fn to_int64(x: Tensor[UInt64]) -> Tensor[Int64]:
    # Narrow/reinterpret to signed Int64 (values > Int64.max will wrap/overflow if casting is modular).
    var src = x._data.copy()
    var n = len(src)

    var out = List[Int64]()
    out.reserve(n)

    var i = 0
    var lim = (n // 16) * 16
    while i < lim:
        out.append(Int64(src[i    ]))
        out.append(Int64(src[i + 1]))
        out.append(Int64(src[i + 2]))
        out.append(Int64(src[i + 3]))
        out.append(Int64(src[i + 4]))
        out.append(Int64(src[i + 5]))
        out.append(Int64(src[i + 6]))
        out.append(Int64(src[i + 7]))
        out.append(Int64(src[i + 8]))
        out.append(Int64(src[i + 9]))
        out.append(Int64(src[i + 10]))
        out.append(Int64(src[i + 11]))
        out.append(Int64(src[i + 12]))
        out.append(Int64(src[i + 13]))
        out.append(Int64(src[i + 14]))
        out.append(Int64(src[i + 15]))
        i += 16

    while i < n:
        out.append(Int64(src[i]))
        i += 1

    var strides = compute_row_major_strides(x._shape)
    return Tensor[Int64](out, x._shape.copy(), strides,x._offset)


@always_inline
fn to_int64(x: Tensor[UInt32]) -> Tensor[Int64]:
    # Widen UInt32 → Int64 (may reinterpret sign for large values if cast is modular).
    var src = x._data.copy()
    var n = len(src)

    var out = List[Int64]()
    out.reserve(n)

    var i = 0
    var lim = (n // 16) * 16
    while i < lim:
        out.append(Int64(src[i    ]))
        out.append(Int64(src[i + 1]))
        out.append(Int64(src[i + 2]))
        out.append(Int64(src[i + 3]))
        out.append(Int64(src[i + 4]))
        out.append(Int64(src[i + 5]))
        out.append(Int64(src[i + 6]))
        out.append(Int64(src[i + 7]))
        out.append(Int64(src[i + 8]))
        out.append(Int64(src[i + 9]))
        out.append(Int64(src[i + 10]))
        out.append(Int64(src[i + 11]))
        out.append(Int64(src[i + 12]))
        out.append(Int64(src[i + 13]))
        out.append(Int64(src[i + 14]))
        out.append(Int64(src[i + 15]))
        i += 16

    while i < n:
        out.append(Int64(src[i]))
        i += 1

    var strides = compute_row_major_strides(x._shape)
    return Tensor[Int64](out, x._shape.copy(), strides,x._offset)


@always_inline
fn to_int64(x: Tensor[UInt16]) -> Tensor[Int64]:
    # Widen UInt16 → Int64.
    var src = x._data.copy()
    var n = len(src)

    var out = List[Int64]()
    out.reserve(n)

    var i = 0
    var lim = (n // 16) * 16
    while i < lim:
        out.append(Int64(src[i    ]))
        out.append(Int64(src[i + 1]))
        out.append(Int64(src[i + 2]))
        out.append(Int64(src[i + 3]))
        out.append(Int64(src[i + 4]))
        out.append(Int64(src[i + 5]))
        out.append(Int64(src[i + 6]))
        out.append(Int64(src[i + 7]))
        out.append(Int64(src[i + 8]))
        out.append(Int64(src[i + 9]))
        out.append(Int64(src[i + 10]))
        out.append(Int64(src[i + 11]))
        out.append(Int64(src[i + 12]))
        out.append(Int64(src[i + 13]))
        out.append(Int64(src[i + 14]))
        out.append(Int64(src[i + 15]))
        i += 16

    while i < n:
        out.append(Int64(src[i]))
        i += 1

    var strides = compute_row_major_strides(x._shape)
    return Tensor[Int64](out, x._shape.copy(), strides,x._offset)


@always_inline
fn to_int64(x: Tensor[UInt8]) -> Tensor[Int64]:
    # Widen UInt8 → Int64.
    var src = x._data.copy()
    var n = len(src)

    var out = List[Int64]()
    out.reserve(n)

    var i = 0
    var lim = (n // 16) * 16
    while i < lim:
        out.append(Int64(src[i    ]))
        out.append(Int64(src[i + 1]))
        out.append(Int64(src[i + 2]))
        out.append(Int64(src[i + 3]))
        out.append(Int64(src[i + 4]))
        out.append(Int64(src[i + 5]))
        out.append(Int64(src[i + 6]))
        out.append(Int64(src[i + 7]))
        out.append(Int64(src[i + 8]))
        out.append(Int64(src[i + 9]))
        out.append(Int64(src[i + 10]))
        out.append(Int64(src[i + 11]))
        out.append(Int64(src[i + 12]))
        out.append(Int64(src[i + 13]))
        out.append(Int64(src[i + 14]))
        out.append(Int64(src[i + 15]))
        i += 16

    while i < n:
        out.append(Int64(src[i]))
        i += 1

    var strides = compute_row_major_strides(x._shape)
    return Tensor[Int64](out, x._shape.copy(), strides,x._offset)


@always_inline
fn to_int64(x: Tensor[Bool]) -> Tensor[Int64]:
    # False → 0, True → 1 (as Int64).
    var src = x._data.copy()
    var n = len(src)

    var out = List[Int64]()
    out.reserve(n)

    var i = 0
    var lim = (n // 16) * 16
    while i < lim:
        var v0 = Int64(0)
        if src[i    ]:
            v0 = Int64(1)
        var v1 = Int64(0)
        if src[i + 1]:
            v1 = Int64(1)
        var v2 = Int64(0)
        if src[i + 2]:
            v2 = Int64(1)
        var v3 = Int64(0)
        if src[i + 3]:
            v3 = Int64(1)
        var v4 = Int64(0)
        if src[i + 4]:
            v4 = Int64(1)
        var v5 = Int64(0)
        if src[i + 5]:
            v5 = Int64(1)
        var v6 = Int64(0)
        if src[i + 6]:
            v6 = Int64(1)
        var v7 = Int64(0)
        if src[i + 7]:
            v7 = Int64(1)
        var v8 = Int64(0)
        if src[i + 8]:
            v8 = Int64(1)
        var v9 = Int64(0)
        if src[i + 9]:
            v9 = Int64(1)
        var v10 = Int64(0)
        if src[i + 10]:
            v10 = Int64(1)
        var v11 = Int64(0)
        if src[i + 11]:
            v11 = Int64(1)
        var v12 = Int64(0)
        if src[i + 12]:
            v12 = Int64(1)
        var v13 = Int64(0)
        if src[i + 13]:
            v13 = Int64(1)
        var v14 = Int64(0)
        if src[i + 14]:
            v14 = Int64(1)
        var v15 = Int64(0)
        if src[i + 15]:
            v15 = Int64(1)

        out.append(v0)
        out.append(v1)
        out.append(v2)
        out.append(v3)
        out.append(v4)
        out.append(v5)
        out.append(v6)
        out.append(v7)
        out.append(v8)
        out.append(v9)
        out.append(v10)
        out.append(v11)
        out.append(v12)
        out.append(v13)
        out.append(v14)
        out.append(v15)
        i += 16

    while i < n:
        var vv = Int64(0)
        if src[i]:
            vv = Int64(1)
        out.append(vv)
        i += 1

    var strides = compute_row_major_strides(x._shape)
    return Tensor[Int64](out, x._shape.copy(), strides,x._offset)


@always_inline
fn to_int64(x: Tensor[String]) -> Tensor[Int64]:
    # No parsing here: fill zeros.
    var n = len(x._data)

    var out = List[Int64]()
    out.reserve(n)

    var i = 0
    var lim = (n // 16) * 16
    while i < lim:
        out.append(Int64(0))
        out.append(Int64(0))
        out.append(Int64(0))
        out.append(Int64(0))
        out.append(Int64(0))
        out.append(Int64(0))
        out.append(Int64(0))
        out.append(Int64(0))
        out.append(Int64(0))
        out.append(Int64(0))
        out.append(Int64(0))
        out.append(Int64(0))
        out.append(Int64(0))
        out.append(Int64(0))
        out.append(Int64(0))
        out.append(Int64(0))
        i += 16

    while i < n:
        out.append(Int64(0))
        i += 1

    var strides = compute_row_major_strides(x._shape)
    return Tensor[Int64](out, x._shape.copy(), strides,x._offset)


#######################################################################
@always_inline
fn to_int32(x: Tensor[Int32]) -> Tensor[Int32]:
    # Identity-width cast: copy as-is into Int32 list.
    var data = x._data.copy()
    var strides = compute_row_major_strides(x._shape)
    return Tensor[Int32](data, x._shape.copy(), strides,x._offset)


@always_inline
fn to_int32(x: Tensor[Float64]) -> Tensor[Int32]:
    # Truncate toward zero from Float64 to Int32.
    var src = x._data.copy()
    var n = len(src)

    var out = List[Int32]()
    out.reserve(n)

    var i = 0
    var lim = (n // 16) * 16
    while i < lim:
        out.append(Int32(src[i    ]))
        out.append(Int32(src[i + 1]))
        out.append(Int32(src[i + 2]))
        out.append(Int32(src[i + 3]))
        out.append(Int32(src[i + 4]))
        out.append(Int32(src[i + 5]))
        out.append(Int32(src[i + 6]))
        out.append(Int32(src[i + 7]))
        out.append(Int32(src[i + 8]))
        out.append(Int32(src[i + 9]))
        out.append(Int32(src[i + 10]))
        out.append(Int32(src[i + 11]))
        out.append(Int32(src[i + 12]))
        out.append(Int32(src[i + 13]))
        out.append(Int32(src[i + 14]))
        out.append(Int32(src[i + 15]))
        i += 16

    while i < n:
        out.append(Int32(src[i]))
        i += 1

    var strides = compute_row_major_strides(x._shape)
    return Tensor[Int32](out, x._shape.copy(), strides,x._offset)


@always_inline
fn to_int32(x: Tensor[Float32]) -> Tensor[Int32]:
    # Truncate toward zero from Float32 to Int32.
    var src = x._data.copy()
    var n = len(src)

    var out = List[Int32]()
    out.reserve(n)

    var i = 0
    var lim = (n // 16) * 16
    while i < lim:
        out.append(Int32(src[i    ]))
        out.append(Int32(src[i + 1]))
        out.append(Int32(src[i + 2]))
        out.append(Int32(src[i + 3]))
        out.append(Int32(src[i + 4]))
        out.append(Int32(src[i + 5]))
        out.append(Int32(src[i + 6]))
        out.append(Int32(src[i + 7]))
        out.append(Int32(src[i + 8]))
        out.append(Int32(src[i + 9]))
        out.append(Int32(src[i + 10]))
        out.append(Int32(src[i + 11]))
        out.append(Int32(src[i + 12]))
        out.append(Int32(src[i + 13]))
        out.append(Int32(src[i + 14]))
        out.append(Int32(src[i + 15]))
        i += 16

    while i < n:
        out.append(Int32(src[i]))
        i += 1

    var strides = compute_row_major_strides(x._shape)
    return Tensor[Int32](out, x._shape.copy(), strides,x._offset)


@always_inline
fn to_int32(x: Tensor[Int64]) -> Tensor[Int32]:
    # Narrow Int64 → Int32 (may truncate on platforms where needed).
    var src = x._data.copy()
    var n = len(src)

    var out = List[Int32]()
    out.reserve(n)

    var i = 0
    var lim = (n // 16) * 16
    while i < lim:
        out.append(Int32(src[i    ]))
        out.append(Int32(src[i + 1]))
        out.append(Int32(src[i + 2]))
        out.append(Int32(src[i + 3]))
        out.append(Int32(src[i + 4]))
        out.append(Int32(src[i + 5]))
        out.append(Int32(src[i + 6]))
        out.append(Int32(src[i + 7]))
        out.append(Int32(src[i + 8]))
        out.append(Int32(src[i + 9]))
        out.append(Int32(src[i + 10]))
        out.append(Int32(src[i + 11]))
        out.append(Int32(src[i + 12]))
        out.append(Int32(src[i + 13]))
        out.append(Int32(src[i + 14]))
        out.append(Int32(src[i + 15]))
        i += 16

    while i < n:
        out.append(Int32(src[i]))
        i += 1

    var strides = compute_row_major_strides(x._shape)
    return Tensor[Int32](out, x._shape.copy(), strides,x._offset)


@always_inline
fn to_int32(x: Tensor[Int16]) -> Tensor[Int32]:
    # Widen Int16 → Int32.
    var src = x._data.copy()
    var n = len(src)

    var out = List[Int32]()
    out.reserve(n)

    var i = 0
    var lim = (n // 16) * 16
    while i < lim:
        out.append(Int32(src[i    ]))
        out.append(Int32(src[i + 1]))
        out.append(Int32(src[i + 2]))
        out.append(Int32(src[i + 3]))
        out.append(Int32(src[i + 4]))
        out.append(Int32(src[i + 5]))
        out.append(Int32(src[i + 6]))
        out.append(Int32(src[i + 7]))
        out.append(Int32(src[i + 8]))
        out.append(Int32(src[i + 9]))
        out.append(Int32(src[i + 10]))
        out.append(Int32(src[i + 11]))
        out.append(Int32(src[i + 12]))
        out.append(Int32(src[i + 13]))
        out.append(Int32(src[i + 14]))
        out.append(Int32(src[i + 15]))
        i += 16

    while i < n:
        out.append(Int32(src[i]))
        i += 1

    var strides = compute_row_major_strides(x._shape)
    return Tensor[Int32](out, x._shape.copy(), strides,x._offset)


@always_inline
fn to_int32(x: Tensor[Int8]) -> Tensor[Int32]:
    # Widen Int8 → Int32.
    var src = x._data.copy()
    var n = len(src)

    var out = List[Int32]()
    out.reserve(n)

    var i = 0
    var lim = (n // 16) * 16
    while i < lim:
        out.append(Int32(src[i    ]))
        out.append(Int32(src[i + 1]))
        out.append(Int32(src[i + 2]))
        out.append(Int32(src[i + 3]))
        out.append(Int32(src[i + 4]))
        out.append(Int32(src[i + 5]))
        out.append(Int32(src[i + 6]))
        out.append(Int32(src[i + 7]))
        out.append(Int32(src[i + 8]))
        out.append(Int32(src[i + 9]))
        out.append(Int32(src[i + 10]))
        out.append(Int32(src[i + 11]))
        out.append(Int32(src[i + 12]))
        out.append(Int32(src[i + 13]))
        out.append(Int32(src[i + 14]))
        out.append(Int32(src[i + 15]))
        i += 16

    while i < n:
        out.append(Int32(src[i]))
        i += 1

    var strides = compute_row_major_strides(x._shape)
    return Tensor[Int32](out, x._shape.copy(), strides,x._offset)


@always_inline
fn to_int32(x: Tensor[Int]) -> Tensor[Int32]:
    # Narrow/widen platform Int → Int32 (depending on platform Int width).
    var src = x._data.copy()
    var n = len(src)

    var out = List[Int32]()
    out.reserve(n)

    var i = 0
    var lim = (n // 16) * 16
    while i < lim:
        out.append(Int32(src[i    ]))
        out.append(Int32(src[i + 1]))
        out.append(Int32(src[i + 2]))
        out.append(Int32(src[i + 3]))
        out.append(Int32(src[i + 4]))
        out.append(Int32(src[i + 5]))
        out.append(Int32(src[i + 6]))
        out.append(Int32(src[i + 7]))
        out.append(Int32(src[i + 8]))
        out.append(Int32(src[i + 9]))
        out.append(Int32(src[i + 10]))
        out.append(Int32(src[i + 11]))
        out.append(Int32(src[i + 12]))
        out.append(Int32(src[i + 13]))
        out.append(Int32(src[i + 14]))
        out.append(Int32(src[i + 15]))
        i += 16

    while i < n:
        out.append(Int32(src[i]))
        i += 1

    var strides = compute_row_major_strides(x._shape)
    return Tensor[Int32](out, x._shape.copy(), strides,x._offset)


@always_inline
fn to_int32(x: Tensor[UInt64]) -> Tensor[Int32]:
    # Narrow UInt64 → Int32 (may wrap/overflow if value > Int32.max).
    var src = x._data.copy()
    var n = len(src)

    var out = List[Int32]()
    out.reserve(n)

    var i = 0
    var lim = (n // 16) * 16
    while i < lim:
        out.append(Int32(src[i    ]))
        out.append(Int32(src[i + 1]))
        out.append(Int32(src[i + 2]))
        out.append(Int32(src[i + 3]))
        out.append(Int32(src[i + 4]))
        out.append(Int32(src[i + 5]))
        out.append(Int32(src[i + 6]))
        out.append(Int32(src[i + 7]))
        out.append(Int32(src[i + 8]))
        out.append(Int32(src[i + 9]))
        out.append(Int32(src[i + 10]))
        out.append(Int32(src[i + 11]))
        out.append(Int32(src[i + 12]))
        out.append(Int32(src[i + 13]))
        out.append(Int32(src[i + 14]))
        out.append(Int32(src[i + 15]))
        i += 16

    while i < n:
        out.append(Int32(src[i]))
        i += 1

    var strides = compute_row_major_strides(x._shape)
    return Tensor[Int32](out, x._shape.copy(), strides,x._offset)


@always_inline
fn to_int32(x: Tensor[UInt32]) -> Tensor[Int32]:
    # Identity-width cast if UInt32 fits; note that sign reinterpretation may occur depending on cast semantics.
    var src = x._data.copy()
    var n = len(src)

    var out = List[Int32]()
    out.reserve(n)

    var i = 0
    var lim = (n // 16) * 16
    while i < lim:
        out.append(Int32(src[i    ]))
        out.append(Int32(src[i + 1]))
        out.append(Int32(src[i + 2]))
        out.append(Int32(src[i + 3]))
        out.append(Int32(src[i + 4]))
        out.append(Int32(src[i + 5]))
        out.append(Int32(src[i + 6]))
        out.append(Int32(src[i + 7]))
        out.append(Int32(src[i + 8]))
        out.append(Int32(src[i + 9]))
        out.append(Int32(src[i + 10]))
        out.append(Int32(src[i + 11]))
        out.append(Int32(src[i + 12]))
        out.append(Int32(src[i + 13]))
        out.append(Int32(src[i + 14]))
        out.append(Int32(src[i + 15]))
        i += 16

    while i < n:
        out.append(Int32(src[i]))
        i += 1

    var strides = compute_row_major_strides(x._shape)
    return Tensor[Int32](out, x._shape.copy(), strides,x._offset)


@always_inline
fn to_int32(x: Tensor[UInt16]) -> Tensor[Int32]:
    # Widen UInt16 → Int32.
    var src = x._data.copy()
    var n = len(src)

    var out = List[Int32]()
    out.reserve(n)

    var i = 0
    var lim = (n // 16) * 16
    while i < lim:
        out.append(Int32(src[i    ]))
        out.append(Int32(src[i + 1]))
        out.append(Int32(src[i + 2]))
        out.append(Int32(src[i + 3]))
        out.append(Int32(src[i + 4]))
        out.append(Int32(src[i + 5]))
        out.append(Int32(src[i + 6]))
        out.append(Int32(src[i + 7]))
        out.append(Int32(src[i + 8]))
        out.append(Int32(src[i + 9]))
        out.append(Int32(src[i + 10]))
        out.append(Int32(src[i + 11]))
        out.append(Int32(src[i + 12]))
        out.append(Int32(src[i + 13]))
        out.append(Int32(src[i + 14]))
        out.append(Int32(src[i + 15]))
        i += 16

    while i < n:
        out.append(Int32(src[i]))
        i += 1

    var strides = compute_row_major_strides(x._shape)
    return Tensor[Int32](out, x._shape.copy(), strides,x._offset)


@always_inline
fn to_int32(x: Tensor[UInt8]) -> Tensor[Int32]:
    # Widen UInt8 → Int32.
    var src = x._data.copy()
    var n = len(src)

    var out = List[Int32]()
    out.reserve(n)

    var i = 0
    var lim = (n // 16) * 16
    while i < lim:
        out.append(Int32(src[i    ]))
        out.append(Int32(src[i + 1]))
        out.append(Int32(src[i + 2]))
        out.append(Int32(src[i + 3]))
        out.append(Int32(src[i + 4]))
        out.append(Int32(src[i + 5]))
        out.append(Int32(src[i + 6]))
        out.append(Int32(src[i + 7]))
        out.append(Int32(src[i + 8]))
        out.append(Int32(src[i + 9]))
        out.append(Int32(src[i + 10]))
        out.append(Int32(src[i + 11]))
        out.append(Int32(src[i + 12]))
        out.append(Int32(src[i + 13]))
        out.append(Int32(src[i + 14]))
        out.append(Int32(src[i + 15]))
        i += 16

    while i < n:
        out.append(Int32(src[i]))
        i += 1

    var strides = compute_row_major_strides(x._shape)
    return Tensor[Int32](out, x._shape.copy(), strides,x._offset)


@always_inline
fn to_int32(x: Tensor[Bool]) -> Tensor[Int32]:
    # False → 0, True → 1 (as Int32).
    var src = x._data.copy()
    var n = len(src)

    var out = List[Int32]()
    out.reserve(n)

    var i = 0
    var lim = (n // 16) * 16
    while i < lim:
        var v0 = Int32(0)
        if src[i    ]:
            v0 = Int32(1)
        var v1 = Int32(0)
        if src[i + 1]:
            v1 = Int32(1)
        var v2 = Int32(0)
        if src[i + 2]:
            v2 = Int32(1)
        var v3 = Int32(0)
        if src[i + 3]:
            v3 = Int32(1)
        var v4 = Int32(0)
        if src[i + 4]:
            v4 = Int32(1)
        var v5 = Int32(0)
        if src[i + 5]:
            v5 = Int32(1)
        var v6 = Int32(0)
        if src[i + 6]:
            v6 = Int32(1)
        var v7 = Int32(0)
        if src[i + 7]:
            v7 = Int32(1)
        var v8 = Int32(0)
        if src[i + 8]:
            v8 = Int32(1)
        var v9 = Int32(0)
        if src[i + 9]:
            v9 = Int32(1)
        var v10 = Int32(0)
        if src[i + 10]:
            v10 = Int32(1)
        var v11 = Int32(0)
        if src[i + 11]:
            v11 = Int32(1)
        var v12 = Int32(0)
        if src[i + 12]:
            v12 = Int32(1)
        var v13 = Int32(0)
        if src[i + 13]:
            v13 = Int32(1)
        var v14 = Int32(0)
        if src[i + 14]:
            v14 = Int32(1)
        var v15 = Int32(0)
        if src[i + 15]:
            v15 = Int32(1)

        out.append(v0)
        out.append(v1)
        out.append(v2)
        out.append(v3)
        out.append(v4)
        out.append(v5)
        out.append(v6)
        out.append(v7)
        out.append(v8)
        out.append(v9)
        out.append(v10)
        out.append(v11)
        out.append(v12)
        out.append(v13)
        out.append(v14)
        out.append(v15)
        i += 16

    while i < n:
        var vv = Int32(0)
        if src[i]:
            vv = Int32(1)
        out.append(vv)
        i += 1

    var strides = compute_row_major_strides(x._shape)
    return Tensor[Int32](out, x._shape.copy(), strides,x._offset)


@always_inline
fn to_int32(x: Tensor[String]) -> Tensor[Int32]:
    # No parsing here: fill zeros.
    var n = len(x._data)

    var out = List[Int32]()
    out.reserve(n)

    var i = 0
    var lim = (n // 16) * 16
    while i < lim:
        out.append(Int32(0))
        out.append(Int32(0))
        out.append(Int32(0))
        out.append(Int32(0))
        out.append(Int32(0))
        out.append(Int32(0))
        out.append(Int32(0))
        out.append(Int32(0))
        out.append(Int32(0))
        out.append(Int32(0))
        out.append(Int32(0))
        out.append(Int32(0))
        out.append(Int32(0))
        out.append(Int32(0))
        out.append(Int32(0))
        out.append(Int32(0))
        i += 16

    while i < n:
        out.append(Int32(0))
        i += 1

    var strides = compute_row_major_strides(x._shape)
    return Tensor[Int32](out, x._shape.copy(), strides,x._offset)

######################################################################

@always_inline
fn to_int16(x: Tensor[Int16]) -> Tensor[Int16]:
    # Identity-width cast: copy as-is into Int16 list.
    var data = x._data.copy()
    var strides = compute_row_major_strides(x._shape)
    return Tensor[Int16](data, x._shape.copy(), strides,x._offset)


@always_inline
fn to_int16(x: Tensor[Float64]) -> Tensor[Int16]:
    # Truncate toward zero from Float64 to Int16.
    var src = x._data.copy()
    var n = len(src)

    var out = List[Int16]()
    out.reserve(n)

    var i = 0
    var lim = (n // 16) * 16
    while i < lim:
        out.append(Int16(src[i    ]))
        out.append(Int16(src[i + 1]))
        out.append(Int16(src[i + 2]))
        out.append(Int16(src[i + 3]))
        out.append(Int16(src[i + 4]))
        out.append(Int16(src[i + 5]))
        out.append(Int16(src[i + 6]))
        out.append(Int16(src[i + 7]))
        out.append(Int16(src[i + 8]))
        out.append(Int16(src[i + 9]))
        out.append(Int16(src[i + 10]))
        out.append(Int16(src[i + 11]))
        out.append(Int16(src[i + 12]))
        out.append(Int16(src[i + 13]))
        out.append(Int16(src[i + 14]))
        out.append(Int16(src[i + 15]))
        i += 16

    while i < n:
        out.append(Int16(src[i]))
        i += 1

    var strides = compute_row_major_strides(x._shape)
    return Tensor[Int16](out, x._shape.copy(), strides,x._offset)


@always_inline
fn to_int16(x: Tensor[Float32]) -> Tensor[Int16]:
    # Truncate toward zero from Float32 to Int16.
    var src = x._data.copy()
    var n = len(src)

    var out = List[Int16]()
    out.reserve(n)

    var i = 0
    var lim = (n // 16) * 16
    while i < lim:
        out.append(Int16(src[i    ]))
        out.append(Int16(src[i + 1]))
        out.append(Int16(src[i + 2]))
        out.append(Int16(src[i + 3]))
        out.append(Int16(src[i + 4]))
        out.append(Int16(src[i + 5]))
        out.append(Int16(src[i + 6]))
        out.append(Int16(src[i + 7]))
        out.append(Int16(src[i + 8]))
        out.append(Int16(src[i + 9]))
        out.append(Int16(src[i + 10]))
        out.append(Int16(src[i + 11]))
        out.append(Int16(src[i + 12]))
        out.append(Int16(src[i + 13]))
        out.append(Int16(src[i + 14]))
        out.append(Int16(src[i + 15]))
        i += 16

    while i < n:
        out.append(Int16(src[i]))
        i += 1

    var strides = compute_row_major_strides(x._shape)
    return Tensor[Int16](out, x._shape.copy(), strides,x._offset)


@always_inline
fn to_int16(x: Tensor[Int64]) -> Tensor[Int16]:
    # Narrow Int64 → Int16 (may truncate).
    var src = x._data.copy()
    var n = len(src)

    var out = List[Int16]()
    out.reserve(n)

    var i = 0
    var lim = (n // 16) * 16
    while i < lim:
        out.append(Int16(src[i    ]))
        out.append(Int16(src[i + 1]))
        out.append(Int16(src[i + 2]))
        out.append(Int16(src[i + 3]))
        out.append(Int16(src[i + 4]))
        out.append(Int16(src[i + 5]))
        out.append(Int16(src[i + 6]))
        out.append(Int16(src[i + 7]))
        out.append(Int16(src[i + 8]))
        out.append(Int16(src[i + 9]))
        out.append(Int16(src[i + 10]))
        out.append(Int16(src[i + 11]))
        out.append(Int16(src[i + 12]))
        out.append(Int16(src[i + 13]))
        out.append(Int16(src[i + 14]))
        out.append(Int16(src[i + 15]))
        i += 16

    while i < n:
        out.append(Int16(src[i]))
        i += 1

    var strides = compute_row_major_strides(x._shape)
    return Tensor[Int16](out, x._shape.copy(), strides,x._offset)


@always_inline
fn to_int16(x: Tensor[Int32]) -> Tensor[Int16]:
    # Narrow Int32 → Int16 (may truncate).
    var src = x._data.copy()
    var n = len(src)

    var out = List[Int16]()
    out.reserve(n)

    var i = 0
    var lim = (n // 16) * 16
    while i < lim:
        out.append(Int16(src[i    ]))
        out.append(Int16(src[i + 1]))
        out.append(Int16(src[i + 2]))
        out.append(Int16(src[i + 3]))
        out.append(Int16(src[i + 4]))
        out.append(Int16(src[i + 5]))
        out.append(Int16(src[i + 6]))
        out.append(Int16(src[i + 7]))
        out.append(Int16(src[i + 8]))
        out.append(Int16(src[i + 9]))
        out.append(Int16(src[i + 10]))
        out.append(Int16(src[i + 11]))
        out.append(Int16(src[i + 12]))
        out.append(Int16(src[i + 13]))
        out.append(Int16(src[i + 14]))
        out.append(Int16(src[i + 15]))
        i += 16

    while i < n:
        out.append(Int16(src[i]))
        i += 1

    var strides = compute_row_major_strides(x._shape)
    return Tensor[Int16](out, x._shape.copy(), strides,x._offset)


@always_inline
fn to_int16(x: Tensor[Int8]) -> Tensor[Int16]:
    # Widen Int8 → Int16.
    var src = x._data.copy()
    var n = len(src)

    var out = List[Int16]()
    out.reserve(n)

    var i = 0
    var lim = (n // 16) * 16
    while i < lim:
        out.append(Int16(src[i    ]))
        out.append(Int16(src[i + 1]))
        out.append(Int16(src[i + 2]))
        out.append(Int16(src[i + 3]))
        out.append(Int16(src[i + 4]))
        out.append(Int16(src[i + 5]))
        out.append(Int16(src[i + 6]))
        out.append(Int16(src[i + 7]))
        out.append(Int16(src[i + 8]))
        out.append(Int16(src[i + 9]))
        out.append(Int16(src[i + 10]))
        out.append(Int16(src[i + 11]))
        out.append(Int16(src[i + 12]))
        out.append(Int16(src[i + 13]))
        out.append(Int16(src[i + 14]))
        out.append(Int16(src[i + 15]))
        i += 16

    while i < n:
        out.append(Int16(src[i]))
        i += 1

    var strides = compute_row_major_strides(x._shape)
    return Tensor[Int16](out, x._shape.copy(), strides,x._offset)


@always_inline
fn to_int16(x: Tensor[Int]) -> Tensor[Int16]:
    # Narrow/widen platform Int → Int16.
    var src = x._data.copy()
    var n = len(src)

    var out = List[Int16]()
    out.reserve(n)

    var i = 0
    var lim = (n // 16) * 16
    while i < lim:
        out.append(Int16(src[i    ]))
        out.append(Int16(src[i + 1]))
        out.append(Int16(src[i + 2]))
        out.append(Int16(src[i + 3]))
        out.append(Int16(src[i + 4]))
        out.append(Int16(src[i + 5]))
        out.append(Int16(src[i + 6]))
        out.append(Int16(src[i + 7]))
        out.append(Int16(src[i + 8]))
        out.append(Int16(src[i + 9]))
        out.append(Int16(src[i + 10]))
        out.append(Int16(src[i + 11]))
        out.append(Int16(src[i + 12]))
        out.append(Int16(src[i + 13]))
        out.append(Int16(src[i + 14]))
        out.append(Int16(src[i + 15]))
        i += 16

    while i < n:
        out.append(Int16(src[i]))
        i += 1

    var strides = compute_row_major_strides(x._shape)
    return Tensor[Int16](out, x._shape.copy(), strides,x._offset)


@always_inline
fn to_int16(x: Tensor[UInt64]) -> Tensor[Int16]:
    # Narrow UInt64 → Int16 (may wrap/overflow).
    var src = x._data.copy()
    var n = len(src)

    var out = List[Int16]()
    out.reserve(n)

    var i = 0
    var lim = (n // 16) * 16
    while i < lim:
        out.append(Int16(src[i    ]))
        out.append(Int16(src[i + 1]))
        out.append(Int16(src[i + 2]))
        out.append(Int16(src[i + 3]))
        out.append(Int16(src[i + 4]))
        out.append(Int16(src[i + 5]))
        out.append(Int16(src[i + 6]))
        out.append(Int16(src[i + 7]))
        out.append(Int16(src[i + 8]))
        out.append(Int16(src[i + 9]))
        out.append(Int16(src[i + 10]))
        out.append(Int16(src[i + 11]))
        out.append(Int16(src[i + 12]))
        out.append(Int16(src[i + 13]))
        out.append(Int16(src[i + 14]))
        out.append(Int16(src[i + 15]))
        i += 16

    while i < n:
        out.append(Int16(src[i]))
        i += 1

    var strides = compute_row_major_strides(x._shape)
    return Tensor[Int16](out, x._shape.copy(), strides,x._offset)


@always_inline
fn to_int16(x: Tensor[UInt32]) -> Tensor[Int16]:
    # Narrow UInt32 → Int16 (may wrap/overflow).
    var src = x._data.copy()
    var n = len(src)

    var out = List[Int16]()
    out.reserve(n)

    var i = 0
    var lim = (n // 16) * 16
    while i < lim:
        out.append(Int16(src[i    ]))
        out.append(Int16(src[i + 1]))
        out.append(Int16(src[i + 2]))
        out.append(Int16(src[i + 3]))
        out.append(Int16(src[i + 4]))
        out.append(Int16(src[i + 5]))
        out.append(Int16(src[i + 6]))
        out.append(Int16(src[i + 7]))
        out.append(Int16(src[i + 8]))
        out.append(Int16(src[i + 9]))
        out.append(Int16(src[i + 10]))
        out.append(Int16(src[i + 11]))
        out.append(Int16(src[i + 12]))
        out.append(Int16(src[i + 13]))
        out.append(Int16(src[i + 14]))
        out.append(Int16(src[i + 15]))
        i += 16

    while i < n:
        out.append(Int16(src[i]))
        i += 1

    var strides = compute_row_major_strides(x._shape)
    return Tensor[Int16](out, x._shape.copy(), strides,x._offset)


@always_inline
fn to_int16(x: Tensor[UInt16]) -> Tensor[Int16]:
    # Identity-width cast if already UInt16-sized, but signedness changes; values > Int16.max may wrap.
    var src = x._data.copy()
    var n = len(src)

    var out = List[Int16]()
    out.reserve(n)

    var i = 0
    var lim = (n // 16) * 16
    while i < lim:
        out.append(Int16(src[i    ]))
        out.append(Int16(src[i + 1]))
        out.append(Int16(src[i + 2]))
        out.append(Int16(src[i + 3]))
        out.append(Int16(src[i + 4]))
        out.append(Int16(src[i + 5]))
        out.append(Int16(src[i + 6]))
        out.append(Int16(src[i + 7]))
        out.append(Int16(src[i + 8]))
        out.append(Int16(src[i + 9]))
        out.append(Int16(src[i + 10]))
        out.append(Int16(src[i + 11]))
        out.append(Int16(src[i + 12]))
        out.append(Int16(src[i + 13]))
        out.append(Int16(src[i + 14]))
        out.append(Int16(src[i + 15]))
        i += 16

    while i < n:
        out.append(Int16(src[i]))
        i += 1

    var strides = compute_row_major_strides(x._shape)
    return Tensor[Int16](out, x._shape.copy(), strides,x._offset)


@always_inline
fn to_int16(x: Tensor[UInt8]) -> Tensor[Int16]:
    # Widen UInt8 → Int16.
    var src = x._data.copy()
    var n = len(src)

    var out = List[Int16]()
    out.reserve(n)

    var i = 0
    var lim = (n // 16) * 16
    while i < lim:
        out.append(Int16(src[i    ]))
        out.append(Int16(src[i + 1]))
        out.append(Int16(src[i + 2]))
        out.append(Int16(src[i + 3]))
        out.append(Int16(src[i + 4]))
        out.append(Int16(src[i + 5]))
        out.append(Int16(src[i + 6]))
        out.append(Int16(src[i + 7]))
        out.append(Int16(src[i + 8]))
        out.append(Int16(src[i + 9]))
        out.append(Int16(src[i + 10]))
        out.append(Int16(src[i + 11]))
        out.append(Int16(src[i + 12]))
        out.append(Int16(src[i + 13]))
        out.append(Int16(src[i + 14]))
        out.append(Int16(src[i + 15]))
        i += 16

    while i < n:
        out.append(Int16(src[i]))
        i += 1

    var strides = compute_row_major_strides(x._shape)
    return Tensor[Int16](out, x._shape.copy(), strides,x._offset)


@always_inline
fn to_int16(x: Tensor[Bool]) -> Tensor[Int16]:
    # False → 0, True → 1 (as Int16).
    var src = x._data.copy()
    var n = len(src)

    var out = List[Int16]()
    out.reserve(n)

    var i = 0
    var lim = (n // 16) * 16
    while i < lim:
        var v0 = Int16(0)
        if src[i    ]:
            v0 = Int16(1)
        var v1 = Int16(0)
        if src[i + 1]:
            v1 = Int16(1)
        var v2 = Int16(0)
        if src[i + 2]:
            v2 = Int16(1)
        var v3 = Int16(0)
        if src[i + 3]:
            v3 = Int16(1)
        var v4 = Int16(0)
        if src[i + 4]:
            v4 = Int16(1)
        var v5 = Int16(0)
        if src[i + 5]:
            v5 = Int16(1)
        var v6 = Int16(0)
        if src[i + 6]:
            v6 = Int16(1)
        var v7 = Int16(0)
        if src[i + 7]:
            v7 = Int16(1)
        var v8 = Int16(0)
        if src[i + 8]:
            v8 = Int16(1)
        var v9 = Int16(0)
        if src[i + 9]:
            v9 = Int16(1)
        var v10 = Int16(0)
        if src[i + 10]:
            v10 = Int16(1)
        var v11 = Int16(0)
        if src[i + 11]:
            v11 = Int16(1)
        var v12 = Int16(0)
        if src[i + 12]:
            v12 = Int16(1)
        var v13 = Int16(0)
        if src[i + 13]:
            v13 = Int16(1)
        var v14 = Int16(0)
        if src[i + 14]:
            v14 = Int16(1)
        var v15 = Int16(0)
        if src[i + 15]:
            v15 = Int16(1)

        out.append(v0)
        out.append(v1)
        out.append(v2)
        out.append(v3)
        out.append(v4)
        out.append(v5)
        out.append(v6)
        out.append(v7)
        out.append(v8)
        out.append(v9)
        out.append(v10)
        out.append(v11)
        out.append(v12)
        out.append(v13)
        out.append(v14)
        out.append(v15)
        i += 16

    while i < n:
        var vv = Int16(0)
        if src[i]:
            vv = Int16(1)
        out.append(vv)
        i += 1

    var strides = compute_row_major_strides(x._shape)
    return Tensor[Int16](out, x._shape.copy(), strides,x._offset)


@always_inline
fn to_int16(x: Tensor[String]) -> Tensor[Int16]:
    # No parsing here: fill zeros.
    var n = len(x._data)

    var out = List[Int16]()
    out.reserve(n)

    var i = 0
    var lim = (n // 16) * 16
    while i < lim:
        out.append(Int16(0))
        out.append(Int16(0))
        out.append(Int16(0))
        out.append(Int16(0))
        out.append(Int16(0))
        out.append(Int16(0))
        out.append(Int16(0))
        out.append(Int16(0))
        out.append(Int16(0))
        out.append(Int16(0))
        out.append(Int16(0))
        out.append(Int16(0))
        out.append(Int16(0))
        out.append(Int16(0))
        out.append(Int16(0))
        out.append(Int16(0))
        i += 16

    while i < n:
        out.append(Int16(0))
        i += 1

    var strides = compute_row_major_strides(x._shape)
    return Tensor[Int16](out, x._shape.copy(), strides,x._offset)


##########################################################################

@always_inline
fn to_int8(x: Tensor[Int8]) -> Tensor[Int8]:
    # Identity-width cast: copy as-is into Int8 list.
    var data = x._data.copy()
    var strides = compute_row_major_strides(x._shape)
    return Tensor[Int8](data, x._shape.copy(), strides,x._offset)


@always_inline
fn to_int8(x: Tensor[Float64]) -> Tensor[Int8]:
    # Truncate toward zero from Float64 to Int8 (may overflow/clip depending on cast semantics).
    var src = x._data.copy()
    var n = len(src)

    var out = List[Int8]()
    out.reserve(n)

    var i = 0
    var lim = (n // 16) * 16
    while i < lim:
        out.append(Int8(src[i    ]))
        out.append(Int8(src[i + 1]))
        out.append(Int8(src[i + 2]))
        out.append(Int8(src[i + 3]))
        out.append(Int8(src[i + 4]))
        out.append(Int8(src[i + 5]))
        out.append(Int8(src[i + 6]))
        out.append(Int8(src[i + 7]))
        out.append(Int8(src[i + 8]))
        out.append(Int8(src[i + 9]))
        out.append(Int8(src[i + 10]))
        out.append(Int8(src[i + 11]))
        out.append(Int8(src[i + 12]))
        out.append(Int8(src[i + 13]))
        out.append(Int8(src[i + 14]))
        out.append(Int8(src[i + 15]))
        i += 16

    while i < n:
        out.append(Int8(src[i]))
        i += 1

    var strides = compute_row_major_strides(x._shape)
    return Tensor[Int8](out, x._shape.copy(), strides,x._offset)


@always_inline
fn to_int8(x: Tensor[Float32]) -> Tensor[Int8]:
    # Truncate toward zero from Float32 to Int8.
    var src = x._data.copy()
    var n = len(src)

    var out = List[Int8]()
    out.reserve(n)

    var i = 0
    var lim = (n // 16) * 16
    while i < lim:
        out.append(Int8(src[i    ]))
        out.append(Int8(src[i + 1]))
        out.append(Int8(src[i + 2]))
        out.append(Int8(src[i + 3]))
        out.append(Int8(src[i + 4]))
        out.append(Int8(src[i + 5]))
        out.append(Int8(src[i + 6]))
        out.append(Int8(src[i + 7]))
        out.append(Int8(src[i + 8]))
        out.append(Int8(src[i + 9]))
        out.append(Int8(src[i + 10]))
        out.append(Int8(src[i + 11]))
        out.append(Int8(src[i + 12]))
        out.append(Int8(src[i + 13]))
        out.append(Int8(src[i + 14]))
        out.append(Int8(src[i + 15]))
        i += 16

    while i < n:
        out.append(Int8(src[i]))
        i += 1

    var strides = compute_row_major_strides(x._shape)
    return Tensor[Int8](out, x._shape.copy(), strides,x._offset)


@always_inline
fn to_int8(x: Tensor[Int64]) -> Tensor[Int8]:
    # Narrow Int64 → Int8 (may truncate).
    var src = x._data.copy()
    var n = len(src)

    var out = List[Int8]()
    out.reserve(n)

    var i = 0
    var lim = (n // 16) * 16
    while i < lim:
        out.append(Int8(src[i    ]))
        out.append(Int8(src[i + 1]))
        out.append(Int8(src[i + 2]))
        out.append(Int8(src[i + 3]))
        out.append(Int8(src[i + 4]))
        out.append(Int8(src[i + 5]))
        out.append(Int8(src[i + 6]))
        out.append(Int8(src[i + 7]))
        out.append(Int8(src[i + 8]))
        out.append(Int8(src[i + 9]))
        out.append(Int8(src[i + 10]))
        out.append(Int8(src[i + 11]))
        out.append(Int8(src[i + 12]))
        out.append(Int8(src[i + 13]))
        out.append(Int8(src[i + 14]))
        out.append(Int8(src[i + 15]))
        i += 16

    while i < n:
        out.append(Int8(src[i]))
        i += 1

    var strides = compute_row_major_strides(x._shape)
    return Tensor[Int8](out, x._shape.copy(), strides,x._offset)


@always_inline
fn to_int8(x: Tensor[Int32]) -> Tensor[Int8]:
    # Narrow Int32 → Int8 (may truncate).
    var src = x._data.copy()
    var n = len(src)

    var out = List[Int8]()
    out.reserve(n)

    var i = 0
    var lim = (n // 16) * 16
    while i < lim:
        out.append(Int8(src[i    ]))
        out.append(Int8(src[i + 1]))
        out.append(Int8(src[i + 2]))
        out.append(Int8(src[i + 3]))
        out.append(Int8(src[i + 4]))
        out.append(Int8(src[i + 5]))
        out.append(Int8(src[i + 6]))
        out.append(Int8(src[i + 7]))
        out.append(Int8(src[i + 8]))
        out.append(Int8(src[i + 9]))
        out.append(Int8(src[i + 10]))
        out.append(Int8(src[i + 11]))
        out.append(Int8(src[i + 12]))
        out.append(Int8(src[i + 13]))
        out.append(Int8(src[i + 14]))
        out.append(Int8(src[i + 15]))
        i += 16

    while i < n:
        out.append(Int8(src[i]))
        i += 1

    var strides = compute_row_major_strides(x._shape)
    return Tensor[Int8](out, x._shape.copy(), strides,x._offset)


@always_inline
fn to_int8(x: Tensor[Int16]) -> Tensor[Int8]:
    # Narrow Int16 → Int8 (may truncate).
    var src = x._data.copy()
    var n = len(src)

    var out = List[Int8]()
    out.reserve(n)

    var i = 0
    var lim = (n // 16) * 16
    while i < lim:
        out.append(Int8(src[i    ]))
        out.append(Int8(src[i + 1]))
        out.append(Int8(src[i + 2]))
        out.append(Int8(src[i + 3]))
        out.append(Int8(src[i + 4]))
        out.append(Int8(src[i + 5]))
        out.append(Int8(src[i + 6]))
        out.append(Int8(src[i + 7]))
        out.append(Int8(src[i + 8]))
        out.append(Int8(src[i + 9]))
        out.append(Int8(src[i + 10]))
        out.append(Int8(src[i + 11]))
        out.append(Int8(src[i + 12]))
        out.append(Int8(src[i + 13]))
        out.append(Int8(src[i + 14]))
        out.append(Int8(src[i + 15]))
        i += 16

    while i < n:
        out.append(Int8(src[i]))
        i += 1

    var strides = compute_row_major_strides(x._shape)
    return Tensor[Int8](out, x._shape.copy(), strides,x._offset)


@always_inline
fn to_int8(x: Tensor[Int]) -> Tensor[Int8]:
    # Narrow/widen platform Int → Int8 (may truncate).
    var src = x._data.copy()
    var n = len(src)

    var out = List[Int8]()
    out.reserve(n)

    var i = 0
    var lim = (n // 16) * 16
    while i < lim:
        out.append(Int8(src[i    ]))
        out.append(Int8(src[i + 1]))
        out.append(Int8(src[i + 2]))
        out.append(Int8(src[i + 3]))
        out.append(Int8(src[i + 4]))
        out.append(Int8(src[i + 5]))
        out.append(Int8(src[i + 6]))
        out.append(Int8(src[i + 7]))
        out.append(Int8(src[i + 8]))
        out.append(Int8(src[i + 9]))
        out.append(Int8(src[i + 10]))
        out.append(Int8(src[i + 11]))
        out.append(Int8(src[i + 12]))
        out.append(Int8(src[i + 13]))
        out.append(Int8(src[i + 14]))
        out.append(Int8(src[i + 15]))
        i += 16

    while i < n:
        out.append(Int8(src[i]))
        i += 1

    var strides = compute_row_major_strides(x._shape)
    return Tensor[Int8](out, x._shape.copy(), strides,x._offset)


@always_inline
fn to_int8(x: Tensor[UInt64]) -> Tensor[Int8]:
    # Narrow UInt64 → Int8 (may wrap/overflow).
    var src = x._data.copy()
    var n = len(src)

    var out = List[Int8]()
    out.reserve(n)

    var i = 0
    var lim = (n // 16) * 16
    while i < lim:
        out.append(Int8(src[i    ]))
        out.append(Int8(src[i + 1]))
        out.append(Int8(src[i + 2]))
        out.append(Int8(src[i + 3]))
        out.append(Int8(src[i + 4]))
        out.append(Int8(src[i + 5]))
        out.append(Int8(src[i + 6]))
        out.append(Int8(src[i + 7]))
        out.append(Int8(src[i + 8]))
        out.append(Int8(src[i + 9]))
        out.append(Int8(src[i + 10]))
        out.append(Int8(src[i + 11]))
        out.append(Int8(src[i + 12]))
        out.append(Int8(src[i + 13]))
        out.append(Int8(src[i + 14]))
        out.append(Int8(src[i + 15]))
        i += 16

    while i < n:
        out.append(Int8(src[i]))
        i += 1

    var strides = compute_row_major_strides(x._shape)
    return Tensor[Int8](out, x._shape.copy(), strides,x._offset)


@always_inline
fn to_int8(x: Tensor[UInt32]) -> Tensor[Int8]:
    # Narrow UInt32 → Int8 (may wrap/overflow).
    var src = x._data.copy()
    var n = len(src)

    var out = List[Int8]()
    out.reserve(n)

    var i = 0
    var lim = (n // 16) * 16
    while i < lim:
        out.append(Int8(src[i    ]))
        out.append(Int8(src[i + 1]))
        out.append(Int8(src[i + 2]))
        out.append(Int8(src[i + 3]))
        out.append(Int8(src[i + 4]))
        out.append(Int8(src[i + 5]))
        out.append(Int8(src[i + 6]))
        out.append(Int8(src[i + 7]))
        out.append(Int8(src[i + 8]))
        out.append(Int8(src[i + 9]))
        out.append(Int8(src[i + 10]))
        out.append(Int8(src[i + 11]))
        out.append(Int8(src[i + 12]))
        out.append(Int8(src[i + 13]))
        out.append(Int8(src[i + 14]))
        out.append(Int8(src[i + 15]))
        i += 16

    while i < n:
        out.append(Int8(src[i]))
        i += 1

    var strides = compute_row_major_strides(x._shape)
    return Tensor[Int8](out, x._shape.copy(), strides,x._offset)


@always_inline
fn to_int8(x: Tensor[UInt16]) -> Tensor[Int8]:
    # Narrow UInt16 → Int8 (may wrap/overflow).
    var src = x._data.copy()
    var n = len(src)

    var out = List[Int8]()
    out.reserve(n)

    var i = 0
    var lim = (n // 16) * 16
    while i < lim:
        out.append(Int8(src[i    ]))
        out.append(Int8(src[i + 1]))
        out.append(Int8(src[i + 2]))
        out.append(Int8(src[i + 3]))
        out.append(Int8(src[i + 4]))
        out.append(Int8(src[i + 5]))
        out.append(Int8(src[i + 6]))
        out.append(Int8(src[i + 7]))
        out.append(Int8(src[i + 8]))
        out.append(Int8(src[i + 9]))
        out.append(Int8(src[i + 10]))
        out.append(Int8(src[i + 11]))
        out.append(Int8(src[i + 12]))
        out.append(Int8(src[i + 13]))
        out.append(Int8(src[i + 14]))
        out.append(Int8(src[i + 15]))
        i += 16

    while i < n:
        out.append(Int8(src[i]))
        i += 1

    var strides = compute_row_major_strides(x._shape)
    return Tensor[Int8](out, x._shape.copy(), strides,x._offset)


@always_inline
fn to_int8(x: Tensor[UInt8]) -> Tensor[Int8]:
    # Identity-width cast: copy UInt8 to Int8 (signedness changes; values > 127 may wrap).
    var data = x._data.copy()
    var n = len(data)

    var out = List[Int8]()
    out.reserve(n)

    var i = 0
    var lim = (n // 16) * 16
    while i < lim:
        out.append(Int8(data[i    ]))
        out.append(Int8(data[i + 1]))
        out.append(Int8(data[i + 2]))
        out.append(Int8(data[i + 3]))
        out.append(Int8(data[i + 4]))
        out.append(Int8(data[i + 5]))
        out.append(Int8(data[i + 6]))
        out.append(Int8(data[i + 7]))
        out.append(Int8(data[i + 8]))
        out.append(Int8(data[i + 9]))
        out.append(Int8(data[i + 10]))
        out.append(Int8(data[i + 11]))
        out.append(Int8(data[i + 12]))
        out.append(Int8(data[i + 13]))
        out.append(Int8(data[i + 14]))
        out.append(Int8(data[i + 15]))
        i += 16

    while i < n:
        out.append(Int8(data[i]))
        i += 1

    var strides = compute_row_major_strides(x._shape)
    return Tensor[Int8](out, x._shape.copy(), strides,x._offset)


@always_inline
fn to_int8(x: Tensor[Bool]) -> Tensor[Int8]:
    # False → 0, True → 1 (as Int8).
    var src = x._data.copy()
    var n = len(src)

    var out = List[Int8]()
    out.reserve(n)

    var i = 0
    var lim = (n // 16) * 16
    while i < lim:
        var v0 = Int8(0)
        if src[i    ]:
            v0 = Int8(1)
        var v1 = Int8(0)
        if src[i + 1]:
            v1 = Int8(1)
        var v2 = Int8(0)
        if src[i + 2]:
            v2 = Int8(1)
        var v3 = Int8(0)
        if src[i + 3]:
            v3 = Int8(1)
        var v4 = Int8(0)
        if src[i + 4]:
            v4 = Int8(1)
        var v5 = Int8(0)
        if src[i + 5]:
            v5 = Int8(1)
        var v6 = Int8(0)
        if src[i + 6]:
            v6 = Int8(1)
        var v7 = Int8(0)
        if src[i + 7]:
            v7 = Int8(1)
        var v8 = Int8(0)
        if src[i + 8]:
            v8 = Int8(1)
        var v9 = Int8(0)
        if src[i + 9]:
            v9 = Int8(1)
        var v10 = Int8(0)
        if src[i + 10]:
            v10 = Int8(1)
        var v11 = Int8(0)
        if src[i + 11]:
            v11 = Int8(1)
        var v12 = Int8(0)
        if src[i + 12]:
            v12 = Int8(1)
        var v13 = Int8(0)
        if src[i + 13]:
            v13 = Int8(1)
        var v14 = Int8(0)
        if src[i + 14]:
            v14 = Int8(1)
        var v15 = Int8(0)
        if src[i + 15]:
            v15 = Int8(1)

        out.append(v0)
        out.append(v1)
        out.append(v2)
        out.append(v3)
        out.append(v4)
        out.append(v5)
        out.append(v6)
        out.append(v7)
        out.append(v8)
        out.append(v9)
        out.append(v10)
        out.append(v11)
        out.append(v12)
        out.append(v13)
        out.append(v14)
        out.append(v15)
        i += 16

    while i < n:
        var vv = Int8(0)
        if src[i]:
            vv = Int8(1)
        out.append(vv)
        i += 1

    var strides = compute_row_major_strides(x._shape)
    return Tensor[Int8](out, x._shape.copy(), strides,x._offset)


@always_inline
fn to_int8(x: Tensor[String]) -> Tensor[Int8]:
    # No parsing here: fill zeros.
    var n = len(x._data)

    var out = List[Int8]()
    out.reserve(n)

    var i = 0
    var lim = (n // 16) * 16
    while i < lim:
        out.append(Int8(0))
        out.append(Int8(0))
        out.append(Int8(0))
        out.append(Int8(0))
        out.append(Int8(0))
        out.append(Int8(0))
        out.append(Int8(0))
        out.append(Int8(0))
        out.append(Int8(0))
        out.append(Int8(0))
        out.append(Int8(0))
        out.append(Int8(0))
        out.append(Int8(0))
        out.append(Int8(0))
        out.append(Int8(0))
        out.append(Int8(0))
        i += 16

    while i < n:
        out.append(Int8(0))
        i += 1

    var strides = compute_row_major_strides(x._shape)
    return Tensor[Int8](out, x._shape.copy(), strides,x._offset)



#############################################################################

# @always_inline
# fn to_string(x: Tensor[String]) -> Tensor[String]:
#     # Identity-width cast: copy as-is into String list.
#     var data = x._data.copy()
#     var strides = compute_row_major_strides(x._shape)
#     return Tensor[String](data, x._shape.copy(), strides,x._offset)


# @always_inline
# fn to_string(x: Tensor[Float64]) -> Tensor[String]:
#     # Format Float64 elements to String with default String() conversion.
#     var src = x._data.copy()
#     var n = len(src)

#     var out = List[String]()
#     out.reserve(n)

#     var i = 0
#     var lim = (n // 8) * 8
#     while i < lim:
#         out.append(String(src[i    ]))
#         out.append(String(src[i + 1]))
#         out.append(String(src[i + 2]))
#         out.append(String(src[i + 3]))
#         out.append(String(src[i + 4]))
#         out.append(String(src[i + 5]))
#         out.append(String(src[i + 6]))
#         out.append(String(src[i + 7]))
#         i += 8

#     while i < n:
#         out.append(String(src[i]))
#         i += 1

#     var strides = compute_row_major_strides(x._shape)
#     return Tensor[String](out, x._shape.copy(), strides,x._offset)


# @always_inline
# fn to_string(x: Tensor[Float32]) -> Tensor[String]:
#     # Format Float32 elements to String.
#     var src = x._data.copy()
#     var n = len(src)

#     var out = List[String]()
#     out.reserve(n)

#     var i = 0
#     var lim = (n // 8) * 8
#     while i < lim:
#         out.append(String(src[i    ]))
#         out.append(String(src[i + 1]))
#         out.append(String(src[i + 2]))
#         out.append(String(src[i + 3]))
#         out.append(String(src[i + 4]))
#         out.append(String(src[i + 5]))
#         out.append(String(src[i + 6]))
#         out.append(String(src[i + 7]))
#         i += 8

#     while i < n:
#         out.append(String(src[i]))
#         i += 1

#     var strides = compute_row_major_strides(x._shape)
#     return Tensor[String](out, x._shape.copy(), strides,x._offset)


# @always_inline
# fn to_string(x: Tensor[Int64]) -> Tensor[String]:
#     # Format Int64 elements to String.
#     var src = x._data.copy()
#     var n = len(src)

#     var out = List[String]()
#     out.reserve(n)

#     var i = 0
#     var lim = (n // 8) * 8
#     while i < lim:
#         out.append(String(src[i    ]))
#         out.append(String(src[i + 1]))
#         out.append(String(src[i + 2]))
#         out.append(String(src[i + 3]))
#         out.append(String(src[i + 4]))
#         out.append(String(src[i + 5]))
#         out.append(String(src[i + 6]))
#         out.append(String(src[i + 7]))
#         i += 8

#     while i < n:
#         out.append(String(src[i]))
#         i += 1

#     var strides = compute_row_major_strides(x._shape)
#     return Tensor[String](out, x._shape.copy(), strides,x._offset)


# @always_inline
# fn to_string(x: Tensor[Int32]) -> Tensor[String]:
#     # Format Int32 elements to String.
#     var src = x._data.copy()
#     var n = len(src)

#     var out = List[String]()
#     out.reserve(n)

#     var i = 0
#     var lim = (n // 8) * 8
#     while i < lim:
#         out.append(String(src[i    ]))
#         out.append(String(src[i + 1]))
#         out.append(String(src[i + 2]))
#         out.append(String(src[i + 3]))
#         out.append(String(src[i + 4]))
#         out.append(String(src[i + 5]))
#         out.append(String(src[i + 6]))
#         out.append(String(src[i + 7]))
#         i += 8

#     while i < n:
#         out.append(String(src[i]))
#         i += 1

#     var strides = compute_row_major_strides(x._shape)
#     return Tensor[String](out, x._shape.copy(), strides,x._offset)


# @always_inline
# fn to_string(x: Tensor[Int16]) -> Tensor[String]:
#     # Format Int16 elements to String.
#     var src = x._data.copy()
#     var n = len(src)

#     var out = List[String]()
#     out.reserve(n)

#     var i = 0
#     var lim = (n // 8) * 8
#     while i < lim:
#         out.append(String(src[i    ]))
#         out.append(String(src[i + 1]))
#         out.append(String(src[i + 2]))
#         out.append(String(src[i + 3]))
#         out.append(String(src[i + 4]))
#         out.append(String(src[i + 5]))
#         out.append(String(src[i + 6]))
#         out.append(String(src[i + 7]))
#         i += 8

#     while i < n:
#         out.append(String(src[i]))
#         i += 1

#     var strides = compute_row_major_strides(x._shape)
#     return Tensor[String](out, x._shape.copy(), strides,x._offset)


# @always_inline
# fn to_string(x: Tensor[Int8]) -> Tensor[String]:
#     # Format Int8 elements to String.
#     var src = x._data.copy()
#     var n = len(src)

#     var out = List[String]()
#     out.reserve(n)

#     var i = 0
#     var lim = (n // 8) * 8
#     while i < lim:
#         out.append(String(src[i    ]))
#         out.append(String(src[i + 1]))
#         out.append(String(src[i + 2]))
#         out.append(String(src[i + 3]))
#         out.append(String(src[i + 4]))
#         out.append(String(src[i + 5]))
#         out.append(String(src[i + 6]))
#         out.append(String(src[i + 7]))
#         i += 8

#     while i < n:
#         out.append(String(src[i]))
#         i += 1

#     var strides = compute_row_major_strides(x._shape)
#     return Tensor[String](out, x._shape.copy(), strides,x._offset)


# @always_inline
# fn to_string(x: Tensor[Int]) -> Tensor[String]:
#     # Format platform Int elements to String.
#     var src = x._data.copy()
#     var n = len(src)

#     var out = List[String]()
#     out.reserve(n)

#     var i = 0
#     var lim = (n // 8) * 8
#     while i < lim:
#         out.append(String(src[i    ]))
#         out.append(String(src[i + 1]))
#         out.append(String(src[i + 2]))
#         out.append(String(src[i + 3]))
#         out.append(String(src[i + 4]))
#         out.append(String(src[i + 5]))
#         out.append(String(src[i + 6]))
#         out.append(String(src[i + 7]))
#         i += 8

#     while i < n:
#         out.append(String(src[i]))
#         i += 1

#     var strides = compute_row_major_strides(x._shape)
#     return Tensor[String](out, x._shape.copy(), strides,x._offset)


# @always_inline
# fn to_string(x: Tensor[UInt64]) -> Tensor[String]:
#     # Format UInt64 elements to String.
#     var src = x._data.copy()
#     var n = len(src)

#     var out = List[String]()
#     out.reserve(n)

#     var i = 0
#     var lim = (n // 8) * 8
#     while i < lim:
#         out.append(String(src[i    ]))
#         out.append(String(src[i + 1]))
#         out.append(String(src[i + 2]))
#         out.append(String(src[i + 3]))
#         out.append(String(src[i + 4]))
#         out.append(String(src[i + 5]))
#         out.append(String(src[i + 6]))
#         out.append(String(src[i + 7]))
#         i += 8

#     while i < n:
#         out.append(String(src[i]))
#         i += 1

#     var strides = compute_row_major_strides(x._shape)
#     return Tensor[String](out, x._shape.copy(), strides,x._offset)


# @always_inline
# fn to_string(x: Tensor[UInt32]) -> Tensor[String]:
#     # Format UInt32 elements to String.
#     var src = x._data.copy()
#     var n = len(src)

#     var out = List[String]()
#     out.reserve(n)

#     var i = 0
#     var lim = (n // 8) * 8
#     while i < lim:
#         out.append(String(src[i    ]))
#         out.append(String(src[i + 1]))
#         out.append(String(src[i + 2]))
#         out.append(String(src[i + 3]))
#         out.append(String(src[i + 4]))
#         out.append(String(src[i + 5]))
#         out.append(String(src[i + 6]))
#         out.append(String(src[i + 7]))
#         i += 8

#     while i < n:
#         out.append(String(src[i]))
#         i += 1

#     var strides = compute_row_major_strides(x._shape)
#     return Tensor[String](out, x._shape.copy(), strides,x._offset)


# @always_inline
# fn to_string(x: Tensor[UInt16]) -> Tensor[String]:
#     # Format UInt16 elements to String.
#     var src = x._data.copy()
#     var n = len(src)

#     var out = List[String]()
#     out.reserve(n)

#     var i = 0
#     var lim = (n // 8) * 8
#     while i < lim:
#         out.append(String(src[i    ]))
#         out.append(String(src[i + 1]))
#         out.append(String(src[i + 2]))
#         out.append(String(src[i + 3]))
#         out.append(String(src[i + 4]))
#         out.append(String(src[i + 5]))
#         out.append(String(src[i + 6]))
#         out.append(String(src[i + 7]))
#         i += 8

#     while i < n:
#         out.append(String(src[i]))
#         i += 1

#     var strides = compute_row_major_strides(x._shape)
#     return Tensor[String](out, x._shape.copy(), strides,x._offset)


# @always_inline
# fn to_string(x: Tensor[UInt8]) -> Tensor[String]:
#     # Format UInt8 elements to String.
#     var src = x._data.copy()
#     var n = len(src)

#     var out = List[String]()
#     out.reserve(n)

#     var i = 0
#     var lim = (n // 8) * 8
#     while i < lim:
#         out.append(String(src[i    ]))
#         out.append(String(src[i + 1]))
#         out.append(String(src[i + 2]))
#         out.append(String(src[i + 3]))
#         out.append(String(src[i + 4]))
#         out.append(String(src[i + 5]))
#         out.append(String(src[i + 6]))
#         out.append(String(src[i + 7]))
#         i += 8

#     while i < n:
#         out.append(String(src[i]))
#         i += 1

#     var strides = compute_row_major_strides(x._shape)
#     return Tensor[String](out, x._shape.copy(), strides,x._offset)


# @always_inline
# fn to_string(x: Tensor[Bool]) -> Tensor[String]:
#     # False → "0", True → "1".
#     var src = x._data.copy()
#     var n = len(src)

#     var out = List[String]()
#     out.reserve(n)

#     var i = 0
#     var lim = (n // 8) * 8
#     while i < lim:
#         var s0 = String("0")
#         if src[i    ]:
#             s0 = String("1")
#         var s1 = String("0")
#         if src[i + 1]:
#             s1 = String("1")
#         var s2 = String("0")
#         if src[i + 2]:
#             s2 = String("1")
#         var s3 = String("0")
#         if src[i + 3]:
#             s3 = String("1")
#         var s4 = String("0")
#         if src[i + 4]:
#             s4 = String("1")
#         var s5 = String("0")
#         if src[i + 5]:
#             s5 = String("1")
#         var s6 = String("0")
#         if src[i + 6]:
#             s6 = String("1")
#         var s7 = String("0")
#         if src[i + 7]:
#             s7 = String("1")

#         out.append(s0)
#         out.append(s1)
#         out.append(s2)
#         out.append(s3)
#         out.append(s4)
#         out.append(s5)
#         out.append(s6)
#         out.append(s7)
#         i += 8

#     while i < n:
#         var s = String("0")
#         if src[i]:
#             s = String("1")
#         out.append(s)
#         i += 1

#     var strides = compute_row_major_strides(x._shape)
#     return Tensor[String](out, x._shape.copy(), strides,x._offset)



###########################################################################
@always_inline
fn to_bool(x: Tensor[Bool]) -> Tensor[Bool]:
    # Identity conversion: copy as-is.
    var data = x._data.copy()
    var strides = compute_row_major_strides(x._shape)
    return Tensor[Bool](data, x._shape.copy(), strides,x._offset)


@always_inline
fn to_bool(x: Tensor[Float64]) -> Tensor[Bool]:
    var src = x._data.copy()
    var n = len(src)

    var out = List[Bool]()
    out.reserve(n)

    var i = 0
    var lim = (n // 16) * 16
    while i < lim:
        out.append(src[i    ] != 0.0)
        out.append(src[i + 1] != 0.0)
        out.append(src[i + 2] != 0.0)
        out.append(src[i + 3] != 0.0)
        out.append(src[i + 4] != 0.0)
        out.append(src[i + 5] != 0.0)
        out.append(src[i + 6] != 0.0)
        out.append(src[i + 7] != 0.0)
        out.append(src[i + 8] != 0.0)
        out.append(src[i + 9] != 0.0)
        out.append(src[i + 10] != 0.0)
        out.append(src[i + 11] != 0.0)
        out.append(src[i + 12] != 0.0)
        out.append(src[i + 13] != 0.0)
        out.append(src[i + 14] != 0.0)
        out.append(src[i + 15] != 0.0)
        i += 16

    while i < n:
        out.append(src[i] != 0.0)
        i += 1

    var strides = compute_row_major_strides(x._shape)
    return Tensor[Bool](out, x._shape.copy(), strides,x._offset)


@always_inline
fn to_bool(x: Tensor[Float32]) -> Tensor[Bool]:
    var src = x._data.copy()
    var n = len(src)

    var out = List[Bool]()
    out.reserve(n)

    var i = 0
    var lim = (n // 16) * 16
    while i < lim:
        out.append(src[i    ] != 0.0)
        out.append(src[i + 1] != 0.0)
        out.append(src[i + 2] != 0.0)
        out.append(src[i + 3] != 0.0)
        out.append(src[i + 4] != 0.0)
        out.append(src[i + 5] != 0.0)
        out.append(src[i + 6] != 0.0)
        out.append(src[i + 7] != 0.0)
        out.append(src[i + 8] != 0.0)
        out.append(src[i + 9] != 0.0)
        out.append(src[i + 10] != 0.0)
        out.append(src[i + 11] != 0.0)
        out.append(src[i + 12] != 0.0)
        out.append(src[i + 13] != 0.0)
        out.append(src[i + 14] != 0.0)
        out.append(src[i + 15] != 0.0)
        i += 16

    while i < n:
        out.append(src[i] != 0.0)
        i += 1

    var strides = compute_row_major_strides(x._shape)
    return Tensor[Bool](out, x._shape.copy(), strides,x._offset)


@always_inline
fn to_bool(x: Tensor[Int64]) -> Tensor[Bool]:
    var src = x._data.copy()
    var n = len(src)

    var out = List[Bool]()
    out.reserve(n)

    var i = 0
    var lim = (n // 16) * 16
    while i < lim:
        out.append(src[i    ] != 0)
        out.append(src[i + 1] != 0)
        out.append(src[i + 2] != 0)
        out.append(src[i + 3] != 0)
        out.append(src[i + 4] != 0)
        out.append(src[i + 5] != 0)
        out.append(src[i + 6] != 0)
        out.append(src[i + 7] != 0)
        out.append(src[i + 8] != 0)
        out.append(src[i + 9] != 0)
        out.append(src[i + 10] != 0)
        out.append(src[i + 11] != 0)
        out.append(src[i + 12] != 0)
        out.append(src[i + 13] != 0)
        out.append(src[i + 14] != 0)
        out.append(src[i + 15] != 0)
        i += 16

    while i < n:
        out.append(src[i] != 0)
        i += 1

    var strides = compute_row_major_strides(x._shape)
    return Tensor[Bool](out, x._shape.copy(), strides,x._offset)


@always_inline
fn to_bool(x: Tensor[Int32]) -> Tensor[Bool]:
    var src = x._data.copy()
    var n = len(src)

    var out = List[Bool]()
    out.reserve(n)

    var i = 0
    var lim = (n // 16) * 16
    while i < lim:
        out.append(src[i    ] != 0)
        out.append(src[i + 1] != 0)
        out.append(src[i + 2] != 0)
        out.append(src[i + 3] != 0)
        out.append(src[i + 4] != 0)
        out.append(src[i + 5] != 0)
        out.append(src[i + 6] != 0)
        out.append(src[i + 7] != 0)
        out.append(src[i + 8] != 0)
        out.append(src[i + 9] != 0)
        out.append(src[i + 10] != 0)
        out.append(src[i + 11] != 0)
        out.append(src[i + 12] != 0)
        out.append(src[i + 13] != 0)
        out.append(src[i + 14] != 0)
        out.append(src[i + 15] != 0)
        i += 16

    while i < n:
        out.append(src[i] != 0)
        i += 1

    var strides = compute_row_major_strides(x._shape)
    return Tensor[Bool](out, x._shape.copy(), strides,x._offset)


@always_inline
fn to_bool(x: Tensor[Int16]) -> Tensor[Bool]:
    var src = x._data.copy()
    var n = len(src)

    var out = List[Bool]()
    out.reserve(n)

    var i = 0
    var lim = (n // 16) * 16
    while i < lim:
        out.append(src[i    ] != 0)
        out.append(src[i + 1] != 0)
        out.append(src[i + 2] != 0)
        out.append(src[i + 3] != 0)
        out.append(src[i + 4] != 0)
        out.append(src[i + 5] != 0)
        out.append(src[i + 6] != 0)
        out.append(src[i + 7] != 0)
        out.append(src[i + 8] != 0)
        out.append(src[i + 9] != 0)
        out.append(src[i + 10] != 0)
        out.append(src[i + 11] != 0)
        out.append(src[i + 12] != 0)
        out.append(src[i + 13] != 0)
        out.append(src[i + 14] != 0)
        out.append(src[i + 15] != 0)
        i += 16

    while i < n:
        out.append(src[i] != 0)
        i += 1

    var strides = compute_row_major_strides(x._shape)
    return Tensor[Bool](out, x._shape.copy(), strides,x._offset)


@always_inline
fn to_bool(x: Tensor[Int8]) -> Tensor[Bool]:
    var src = x._data.copy()
    var n = len(src)

    var out = List[Bool]()
    out.reserve(n)

    var i = 0
    var lim = (n // 16) * 16
    while i < lim:
        out.append(src[i    ] != 0)
        out.append(src[i + 1] != 0)
        out.append(src[i + 2] != 0)
        out.append(src[i + 3] != 0)
        out.append(src[i + 4] != 0)
        out.append(src[i + 5] != 0)
        out.append(src[i + 6] != 0)
        out.append(src[i + 7] != 0)
        out.append(src[i + 8] != 0)
        out.append(src[i + 9] != 0)
        out.append(src[i + 10] != 0)
        out.append(src[i + 11] != 0)
        out.append(src[i + 12] != 0)
        out.append(src[i + 13] != 0)
        out.append(src[i + 14] != 0)
        out.append(src[i + 15] != 0)
        i += 16

    while i < n:
        out.append(src[i] != 0)
        i += 1

    var strides = compute_row_major_strides(x._shape)
    return Tensor[Bool](out, x._shape.copy(), strides,x._offset)


@always_inline
fn to_bool(x: Tensor[Int]) -> Tensor[Bool]:
    var src = x._data.copy()
    var n = len(src)

    var out = List[Bool]()
    out.reserve(n)

    var i = 0
    var lim = (n // 16) * 16
    while i < lim:
        out.append(src[i    ] != 0)
        out.append(src[i + 1] != 0)
        out.append(src[i + 2] != 0)
        out.append(src[i + 3] != 0)
        out.append(src[i + 4] != 0)
        out.append(src[i + 5] != 0)
        out.append(src[i + 6] != 0)
        out.append(src[i + 7] != 0)
        out.append(src[i + 8] != 0)
        out.append(src[i + 9] != 0)
        out.append(src[i + 10] != 0)
        out.append(src[i + 11] != 0)
        out.append(src[i + 12] != 0)
        out.append(src[i + 13] != 0)
        out.append(src[i + 14] != 0)
        out.append(src[i + 15] != 0)
        i += 16

    while i < n:
        out.append(src[i] != 0)
        i += 1

    var strides = compute_row_major_strides(x._shape)
    return Tensor[Bool](out, x._shape.copy(), strides,x._offset)


@always_inline
fn to_bool(x: Tensor[UInt64]) -> Tensor[Bool]:
    var src = x._data.copy()
    var n = len(src)

    var out = List[Bool]()
    out.reserve(n)

    var i = 0
    var lim = (n // 16) * 16
    while i < lim:
        out.append(src[i    ] != 0)
        out.append(src[i + 1] != 0)
        out.append(src[i + 2] != 0)
        out.append(src[i + 3] != 0)
        out.append(src[i + 4] != 0)
        out.append(src[i + 5] != 0)
        out.append(src[i + 6] != 0)
        out.append(src[i + 7] != 0)
        out.append(src[i + 8] != 0)
        out.append(src[i + 9] != 0)
        out.append(src[i + 10] != 0)
        out.append(src[i + 11] != 0)
        out.append(src[i + 12] != 0)
        out.append(src[i + 13] != 0)
        out.append(src[i + 14] != 0)
        out.append(src[i + 15] != 0)
        i += 16

    while i < n:
        out.append(src[i] != 0)
        i += 1

    var strides = compute_row_major_strides(x._shape)
    return Tensor[Bool](out, x._shape.copy(), strides,x._offset)


@always_inline
fn to_bool(x: Tensor[UInt32]) -> Tensor[Bool]:
    var src = x._data.copy()
    var n = len(src)

    var out = List[Bool]()
    out.reserve(n)

    var i = 0
    var lim = (n // 16) * 16
    while i < lim:
        out.append(src[i    ] != 0)
        out.append(src[i + 1] != 0)
        out.append(src[i + 2] != 0)
        out.append(src[i + 3] != 0)
        out.append(src[i + 4] != 0)
        out.append(src[i + 5] != 0)
        out.append(src[i + 6] != 0)
        out.append(src[i + 7] != 0)
        out.append(src[i + 8] != 0)
        out.append(src[i + 9] != 0)
        out.append(src[i + 10] != 0)
        out.append(src[i + 11] != 0)
        out.append(src[i + 12] != 0)
        out.append(src[i + 13] != 0)
        out.append(src[i + 14] != 0)
        out.append(src[i + 15] != 0)
        i += 16

    while i < n:
        out.append(src[i] != 0)
        i += 1

    var strides = compute_row_major_strides(x._shape)
    return Tensor[Bool](out, x._shape.copy(), strides,x._offset)


@always_inline
fn to_bool(x: Tensor[UInt16]) -> Tensor[Bool]:
    var src = x._data.copy()
    var n = len(src)

    var out = List[Bool]()
    out.reserve(n)

    var i = 0
    var lim = (n // 16) * 16
    while i < lim:
        out.append(src[i    ] != 0)
        out.append(src[i + 1] != 0)
        out.append(src[i + 2] != 0)
        out.append(src[i + 3] != 0)
        out.append(src[i + 4] != 0)
        out.append(src[i + 5] != 0)
        out.append(src[i + 6] != 0)
        out.append(src[i + 7] != 0)
        out.append(src[i + 8] != 0)
        out.append(src[i + 9] != 0)
        out.append(src[i + 10] != 0)
        out.append(src[i + 11] != 0)
        out.append(src[i + 12] != 0)
        out.append(src[i + 13] != 0)
        out.append(src[i + 14] != 0)
        out.append(src[i + 15] != 0)
        i += 16

    while i < n:
        out.append(src[i] != 0)
        i += 1

    var strides = compute_row_major_strides(x._shape)
    return Tensor[Bool](out, x._shape.copy(), strides,x._offset)


@always_inline
fn to_bool(x: Tensor[UInt8]) -> Tensor[Bool]:
    var src = x._data.copy()
    var n = len(src)

    var out = List[Bool]()
    out.reserve(n)

    var i = 0
    var lim = (n // 16) * 16
    while i < lim:
        out.append(src[i    ] != 0)
        out.append(src[i + 1] != 0)
        out.append(src[i + 2] != 0)
        out.append(src[i + 3] != 0)
        out.append(src[i + 4] != 0)
        out.append(src[i + 5] != 0)
        out.append(src[i + 6] != 0)
        out.append(src[i + 7] != 0)
        out.append(src[i + 8] != 0)
        out.append(src[i + 9] != 0)
        out.append(src[i + 10] != 0)
        out.append(src[i + 11] != 0)
        out.append(src[i + 12] != 0)
        out.append(src[i + 13] != 0)
        out.append(src[i + 14] != 0)
        out.append(src[i + 15] != 0)
        i += 16

    while i < n:
        out.append(src[i] != 0)
        i += 1

    var strides = compute_row_major_strides(x._shape)
    return Tensor[Bool](out, x._shape.copy(), strides,x._offset)


@always_inline
fn to_bool(x: Tensor[String]) -> Tensor[Bool]:
    # No parsing here: all strings map to False.
    var n = len(x._data)

    var out = List[Bool]()
    out.reserve(n)

    var i = 0
    var lim = (n // 16) * 16
    while i < lim:
        out.append(False)
        out.append(False)
        out.append(False)
        out.append(False)
        out.append(False)
        out.append(False)
        out.append(False)
        out.append(False)
        out.append(False)
        out.append(False)
        out.append(False)
        out.append(False)
        out.append(False)
        out.append(False)
        out.append(False)
        out.append(False)
        i += 16

    while i < n:
        out.append(False)
        i += 1

    var strides = compute_row_major_strides(x._shape)
    return Tensor[Bool](out, x._shape.copy(), strides,x._offset)




# --- put these helpers near your Tensor[T] definition (top-level) ---

 
fn tensor_core_print_nd[U: ImplicitlyCopyable & Copyable & Movable](
    x: Tensor[U],
    fmt: fn (U) -> String,
    dim: Int,
    base: Int
) -> String:
    # Print along axis `dim`, starting from linear offset `base` using strides.
    var s = String("")
    var d = x._shape[dim]

    # Empty length on this axis → nothing between brackets
    if d <= 0:
        return s

    if dim == len(x._shape) - 1:
        # Last axis: 1D row
        var j = 0
        var step = x._strides[dim]
        while j < d:
            if j > 0: s = s + ", "
            s = s + fmt(x._data[base + j * step])
            j = j + 1
        return s

    # Higher dims: recurse
    var i = 0
    var step2 = x._strides[dim]
    while i < d:
        if i > 0: s = s + ", "
        s = s + "["
        s = s + tensor_core_print_nd[U](x, fmt, dim + 1, base + i * step2)
        s = s + "]"
        i = i + 1
    return s


@always_inline
fn tensor_core_print[U: ImplicitlyCopyable & Copyable & Movable](
    x: Tensor[U],
    fmt: fn (U) -> String
) -> String:
    var s = String("[Tensor shape=")
    s = s + x._shape.__str__()
    s = s + ", n=" + numel(x._shape).__str__()
    s = s + ", data="

    var ndim = len(x._shape)

    # 0-D tensor (scalar view): print the scalar at offset.
    if ndim == 0:
        s = s + "[" + fmt(x._data[x._offset]) + "]]"
        return s

    # Any axis with zero length → empty
    var z = 0
    var has_zero = False
    while z < ndim:
        if x._shape[z] == 0:
            has_zero = True
            break
        z = z + 1
    if has_zero:
        s = s + "[]]"
        return s

    # N-D: recurse from base = x._offset (critical!)
    s = s + "["
    s = s + tensor_core_print_nd[U](x, fmt, 0, x._offset)
    s = s + "]]"
    return s



    
@always_inline
fn _fmt_int(x: Int) -> String:
    return x.__str__()


@always_inline
fn _fmt_i16(x: Int16) -> String:
    return x.__str__()

@always_inline
fn _fmt_i32(x: Int32) -> String:
    return x.__str__()

@always_inline
fn _fmt_i64(x: Int64) -> String:
    return x.__str__()

@always_inline
fn _fmt_f32(x: Float32) -> String:
    return x.__str__()

@always_inline
fn _fmt_f64(x: Float64) -> String:
    return x.__str__()

@always_inline
fn _fmt_bool(x: Bool) -> String:
    return x.__str__()

@always_inline
fn _fmt_str(x: String) -> String:
    return x

@always_inline
fn _fmt_obj_U[U: ImplicitlyCopyable & Copyable & Movable](x: U) -> String:
    return String("<obj>")

# ---- per-dtype overloads on Tensor[•] ----

fn to_string(x: Tensor[Int]) -> String:
    return tensor_core_print[Int](x, _fmt_int)

fn to_string(x: Tensor[Int16]) -> String:
    return tensor_core_print[Int16](x, _fmt_i16)

fn to_string(x: Tensor[Int32]) -> String:
    return tensor_core_print[Int32](x, _fmt_i32)

fn to_string(x: Tensor[Int64]) -> String:
    return tensor_core_print[Int64](x, _fmt_i64)

fn to_string(x: Tensor[Float32]) -> String:
    return tensor_core_print[Float32](x, _fmt_f32)

fn to_string(x: Tensor[Float64]) -> String:
    return tensor_core_print[Float64](x, _fmt_f64)

fn to_string(x: Tensor[Bool]) -> String:
    return tensor_core_print[Bool](x, _fmt_bool)

fn to_string(x: Tensor[String]) -> String:
    return tensor_core_print[String](x, _fmt_str)

 
fn to_string[U: ImplicitlyCopyable & Copyable & Movable](x: Tensor[U]) -> String:
    return tensor_core_print[U](x, _fmt_obj_U[U])


##############################################################################################33
@always_inline
fn val_f64(xs: List[Float64], i: Int) -> Float64:
    return xs[i]

@always_inline
fn val_f64(xs: List[Int], i: Int) -> Float64:
    return Float64(xs[i])

 


 
@always_inline 
fn to_f64_f64(x: Float64) -> Float64: 
    return x
@always_inline 
fn to_f64_f32(x: Float32) -> Float64: 
    return Float64(x)

@always_inline 
fn to_f64_i8 (x: Int8 ) -> Float64: 
    return Float64(Int64(x))
@always_inline 
fn to_f64_i16(x: Int16) -> Float64: 
    return Float64(Int64(x))
@always_inline 
fn to_f64_i32(x: Int32) -> Float64: 
    return Float64(Int64(x))
@always_inline 
fn to_f64_i64(x: Int64) -> Float64: 
    return Float64(x)
@always_inline 
fn to_f64_int(x: Int  ) -> Float64: 
    return Float64(Int64(x))

@always_inline 
fn to_f64_u8 (x: UInt8 ) -> Float64: 
    return Float64(UInt64(x))
@always_inline 
fn to_f64_u16(x: UInt16) -> Float64: 
    return Float64(UInt64(x))
@always_inline 
fn to_f64_u32(x: UInt32) -> Float64: 
    return Float64(UInt64(x))
@always_inline 
fn to_f64_u64(x: UInt64) -> Float64: 
    return Float64(x)

fn astype_f64_from_int(t: Tensor[Int]) -> Tensor[Float64]:
    var flat = List[Float64]()
    flat.reserve(len(t._data))
    var i = 0
    while i < len(t._data):
        flat.append(Float64(t._data[i]))
        i = i + 1

    var sh = List[Int]()
    sh.reserve(len(t._shape))
    var k = 0
    while k < len(t._shape):
        sh.append(t._shape[k])
        k = k + 1

    return Tensor[Float64](sh, flat)


fn astype_f64_from_f32(t: Tensor[Float32]) -> Tensor[Float64]:
    var flat = List[Float64]()
    flat.reserve(len(t._data))
    var i = 0
    while i < len(t._data):
        flat.append(Float64(t._data[i]))
        i = i + 1

    var sh = List[Int]()       
    sh.reserve(len(t._shape))
    var k = 0
    while k < len(t._shape):
        sh.append(t._shape[k])
        k = k + 1

    return Tensor[Float64](sh, flat)

@always_inline 
fn opt_i_to_f32(o: Optional[Int])      -> Optional[Float32]:
    if o is None: return None
    return Optional[Float32](Float32(o.value()))

@always_inline 
fn opt_i_to_f64(o: Optional[Int])      -> Optional[Float64]:
    if o is None: return None
    return Optional[Float64](Float64(o.value()))

@always_inline 
fn opt_f32_to_f64(o: Optional[Float32]) -> Optional[Float64]:
    if o is None: return None
    return Optional[Float64](Float64(o.value()))

@always_inline 
fn some_i(x: Int) -> Optional[Int]: 
    return Optional[Int](x)
@always_inline 
fn some_f32(x: Float32) -> Optional[Float32]: 
    return Optional[Float32](x)
@always_inline 
fn some_f64(x: Float64) -> Optional[Float64]: 
    return Optional[Float64](x)