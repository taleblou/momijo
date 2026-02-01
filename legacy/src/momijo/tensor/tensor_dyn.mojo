# MIT License
# SPDX-License-Identifier: MIT
# Project:      Momijo
# Module:       momijo.tensor.anytensor
# File:         src/momijo/tensor/anytensor.mojo
#
# Description:
#   Tagged, type-erased tensor: AnyBuffer + AnyTensor
#   - Stores numeric/boolean buffers with a runtime dtype tag.
#   - Zero/full builders, deep copy, contiguous check.
#   - Generic import/export to Tensor[T] with fast paths.
#   - In-place and out-of-place dtype casting.
#
# Authors:      Morteza Taleblou & Mitra Daneshmand
# Website:      https://taleblou.ir/
# Repository:   https://github.com/taleblou/momijo

from collections.list import List
from momijo.tensor.tensor import Tensor
from momijo.tensor.dtype import (
    DType,
    BOOL, INT8, INT16, INT32, INT64,
    UINT8, UINT16, UINT32, UINT64,
    FLOAT16, BF16, FLOAT32, FLOAT64
)
from momijo.tensor.helpers import (
    copy_list_int, row_major_strides, numel
)

# --------------------------- tag helpers (Int) ---------------------------

@always_inline
fn _TAG_BOOL()  -> Int:  return Int(BOOL().tag.code)
@always_inline
fn _TAG_I8()    -> Int:  return Int(INT8().tag.code)
@always_inline
fn _TAG_I16()   -> Int:  return Int(INT16().tag.code)
@always_inline
fn _TAG_I32()   -> Int:  return Int(INT32().tag.code)
@always_inline
fn _TAG_I64()   -> Int:  return Int(INT64().tag.code)
@always_inline
fn _TAG_U8()    -> Int:  return Int(UINT8().tag.code)
@always_inline
fn _TAG_U16()   -> Int:  return Int(UINT16().tag.code)
@always_inline
fn _TAG_U32()   -> Int:  return Int(UINT32().tag.code)
@always_inline
fn _TAG_U64()   -> Int:  return Int(UINT64().tag.code)
@always_inline
fn _TAG_F16()   -> Int:  return Int(FLOAT16().tag.code)
@always_inline
fn _TAG_BF16()  -> Int:  return Int(BF16().tag.code)
@always_inline
fn _TAG_F32()   -> Int:  return Int(FLOAT32().tag.code)
@always_inline
fn _TAG_F64()   -> Int:  return Int(FLOAT64().tag.code)

# -------------------------- AnyBuffer (tagged storage) --------------------------

struct AnyBuffer(Copyable, Movable):
    var tag:  Int
    var f64:  List[Float64]
    var f32:  List[Float32]
    var f16:  List[Float16]
    var bf16: List[BFloat16]
    var i64:  List[Int64]
    var i32:  List[Int32]
    var i16:  List[Int16]
    var i8:   List[Int8]
    var u64:  List[UInt64]
    var u32:  List[UInt32]
    var u16:  List[UInt16]
    var u8:   List[UInt8]
    var b:    List[UInt8]   # bool stored as 0/1

    fn __init__(out self):
        self.tag  = _TAG_U8()
        self.f64  = List[Float64]()
        self.f32  = List[Float32]()
        self.f16  = List[Float16]()
        self.bf16 = List[BFloat16]()
        self.i64  = List[Int64]()
        self.i32  = List[Int32]()
        self.i16  = List[Int16]()
        self.i8   = List[Int8]()
        self.u64  = List[UInt64]()
        self.u32  = List[UInt32]()
        self.u16  = List[UInt16]()
        self.u8   = List[UInt8]()
        self.b    = List[UInt8]()

    @always_inline
    fn length(self) -> Int:
        var t = self.tag
        if t == _TAG_F64():   return len(self.f64)
        if t == _TAG_F32():   return len(self.f32)
        if t == _TAG_F16():   return len(self.f16)
        if t == _TAG_BF16():  return len(self.bf16)
        if t == _TAG_I64():   return len(self.i64)
        if t == _TAG_I32():   return len(self.i32)
        if t == _TAG_I16():   return len(self.i16)
        if t == _TAG_I8():    return len(self.i8)
        if t == _TAG_U64():   return len(self.u64)
        if t == _TAG_U32():   return len(self.u32)
        if t == _TAG_U16():   return len(self.u16)
        if t == _TAG_U8():    return len(self.u8)
        return len(self.b)

    @always_inline
    fn reserve(mut self, n: Int, tag: Int):
        if tag == _TAG_F64():   self.f64.reserve(n);   return
        if tag == _TAG_F32():   self.f32.reserve(n);   return
        if tag == _TAG_F16():   self.f16.reserve(n);   return
        if tag == _TAG_BF16():  self.bf16.reserve(n);  return
        if tag == _TAG_I64():   self.i64.reserve(n);   return
        if tag == _TAG_I32():   self.i32.reserve(n);   return
        if tag == _TAG_I16():   self.i16.reserve(n);   return
        if tag == _TAG_I8():    self.i8.reserve(n);    return
        if tag == _TAG_U64():   self.u64.reserve(n);   return
        if tag == _TAG_U32():   self.u32.reserve(n);   return
        if tag == _TAG_U16():   self.u16.reserve(n);   return
        if tag == _TAG_U8():    self.u8.reserve(n);    return
        self.b.reserve(n)

    # Read as Float64 (generic math path)
    @always_inline
    fn get_f64(self, i: Int) -> Float64:
        var t = self.tag
        if t == _TAG_F64():   return self.f64[i]
        if t == _TAG_F32():   return Float64(self.f32[i])
        if t == _TAG_F16():   return Float64(self.f16[i])
        if t == _TAG_BF16():  return Float64(self.bf16[i])
        if t == _TAG_I64():   return Float64(self.i64[i])
        if t == _TAG_I32():   return Float64(self.i32[i])
        if t == _TAG_I16():   return Float64(self.i16[i])
        if t == _TAG_I8():    return Float64(self.i8[i])
        if t == _TAG_U64():   return Float64(self.u64[i])
        if t == _TAG_U32():   return Float64(self.u32[i])
        if t == _TAG_U16():   return Float64(self.u16[i])
        if t == _TAG_U8():    return Float64(self.u8[i])
        return Float64(self.b[i])

    # Append casting from Float64
    @always_inline
    fn append_cast_from_f64(mut self, x: Float64, out_tag: Int):
        if out_tag == _TAG_F64():   self.f64.append(x);               return
        if out_tag == _TAG_F32():   self.f32.append(Float32(x));      return
        if out_tag == _TAG_F16():   self.f16.append(Float16(x));      return
        if out_tag == _TAG_BF16():  self.bf16.append(BFloat16(x));    return
        if out_tag == _TAG_I64():   self.i64.append(Int64(x));        return
        if out_tag == _TAG_I32():   self.i32.append(Int32(x));        return
        if out_tag == _TAG_I16():   self.i16.append(Int16(x));        return
        if out_tag == _TAG_I8():    self.i8.append(Int8(x));          return
        if out_tag == _TAG_U64():   self.u64.append(UInt64(x));       return
        if out_tag == _TAG_U32():   self.u32.append(UInt32(x));       return
        if out_tag == _TAG_U16():   self.u16.append(UInt16(x));       return
        if out_tag == _TAG_U8():    self.u8.append(UInt8(x));         return
        var v = UInt8(0)
        if x != 0.0:
            v = 1
        self.b.append(v)

# -------------------------------- AnyTensor --------------------------------

struct AnyTensor(Copyable, Movable):
    var _buf: AnyBuffer
    var _shape: List[Int]
    var _strides: List[Int]
    var _dtype: DType

    fn __init__(out self):
        self._buf = AnyBuffer()
        self._shape = List[Int]()
        self._strides = List[Int]()
        self._dtype = UINT8()

    @always_inline
    fn dtype(self) -> DType:
        return self._dtype

    @always_inline
    fn shape(self) -> List[Int]:
        return copy_list_int(self._shape)

    @always_inline
    fn strides(self) -> List[Int]:
        return copy_list_int(self._strides)

    @always_inline
    fn numel(self) -> Int:
        return numel(self._shape)

    # -------------------------- builders --------------------------

    fn zeros_like_dtype_shape(mut self, dt: DType, shp: List[Int]):
        self._dtype = dt
        self._shape = copy_list_int(shp)
        self._strides = row_major_strides(self._shape)
        self._buf = AnyBuffer()
        var n = numel(self._shape)
        var tag = Int(dt.tag.code)
        self._buf.tag = tag
        self._buf.reserve(n, tag)
        self.__append_repeat_value_as_f64(mut self._buf, tag, 0.0, n)

    fn full_like_dtype_shape(mut self, dt: DType, shp: List[Int], value_as_f64: Float64):
        self._dtype = dt
        self._shape = copy_list_int(shp)
        self._strides = row_major_strides(self._shape)
        self._buf = AnyBuffer()
        var n = numel(self._shape)
        var tag = Int(dt.tag.code)
        self._buf.tag = tag
        self._buf.reserve(n, tag)
        self.__append_repeat_value_as_f64(mut self._buf, tag, value_as_f64, n)

    # ----------------------- optimized repeat append -----------------------

    @always_inline
    @staticmethod
    fn __append_repeat_value_as_f64(mut buf: AnyBuffer, tag: Int, v: Float64, n: Int):
        var i = 0
        if tag == _TAG_F64():
            var lim = (n // 8) * 8
            while i < lim:
                buf.f64.append(v); buf.f64.append(v); buf.f64.append(v); buf.f64.append(v)
                buf.f64.append(v); buf.f64.append(v); buf.f64.append(v); buf.f64.append(v)
                i += 8
            while i < n:
                buf.f64.append(v)
                i += 1
            return
        if tag == _TAG_F32():
            var t = Float32(v)
            buf.f32.reserve(n)
            while i < n:
                buf.f32.append(t)
                i += 1
            return
        if tag == _TAG_F16():
            var t16 = Float16(v)
            buf.f16.reserve(n)
            while i < n:
                buf.f16.append(t16)
                i += 1
            return
        if tag == _TAG_BF16():
            var tb = BFloat16(v)
            buf.bf16.reserve(n)
            while i < n:
                buf.bf16.append(tb)
                i += 1
            return
        if tag == _TAG_I64():
            var t64 = Int64(v)
            buf.i64.reserve(n)
            while i < n:
                buf.i64.append(t64)
                i += 1
            return
        if tag == _TAG_I32():
            var t32 = Int32(v)
            buf.i32.reserve(n)
            while i < n:
                buf.i32.append(t32)
                i += 1
            return
        if tag == _TAG_I16():
            var tI16 = Int16(v)
            buf.i16.reserve(n)
            while i < n:
                buf.i16.append(tI16)
                i += 1
            return
        if tag == _TAG_I8():
            var t8 = Int8(v)
            buf.i8.reserve(n)
            while i < n:
                buf.i8.append(t8)
                i += 1
            return
        if tag == _TAG_U64():
            var u64 = UInt64(v)
            buf.u64.reserve(n)
            while i < n:
                buf.u64.append(u64)
                i += 1
            return
        if tag == _TAG_U32():
            var u32 = UInt32(v)
            buf.u32.reserve(n)
            while i < n:
                buf.u32.append(u32)
                i += 1
            return
        if tag == _TAG_U16():
            var u16 = UInt16(v)
            buf.u16.reserve(n)
            while i < n:
                buf.u16.append(u16)
                i += 1
            return
        if tag == _TAG_U8():
            var u8 = UInt8(v)
            buf.u8.reserve(n)
            while i < n:
                buf.u8.append(u8)
                i += 1
            return
        var b = UInt8(0)
        if v != 0.0:
            b = 1
        buf.b.reserve(n)
        while i < n:
            buf.b.append(b)
            i += 1

    # --------------------------- deep copy --------------------------------

    fn copy(self) -> AnyTensor:
        var z = AnyTensor()
        z._dtype = self._dtype
        z._shape = copy_list_int(self._shape)
        z._strides = row_major_strides(z._shape)
        z._buf.tag = self._buf.tag
        var n = self._buf.length()
        self.__append_same_tag_copy(mut z._buf, self._buf, n)
        return z

    @always_inline
    @staticmethod
    fn __append_same_tag_copy(mut dst: AnyBuffer, src: AnyBuffer, n: Int):
        var t = src.tag
        var i = 0
        if t == _TAG_F64():
            dst.f64.reserve(n)
            var lim = (n // 8) * 8
            while i < lim:
                dst.f64.append(src.f64[i    ]); dst.f64.append(src.f64[i + 1])
                dst.f64.append(src.f64[i + 2]); dst.f64.append(src.f64[i + 3])
                dst.f64.append(src.f64[i + 4]); dst.f64.append(src.f64[i + 5])
                dst.f64.append(src.f64[i + 6]); dst.f64.append(src.f64[i + 7])
                i += 8
            while i < n:
                dst.f64.append(src.f64[i])
                i += 1
            return
        if t == _TAG_F32():
            dst.f32.reserve(n)
            while i < n:
                dst.f32.append(src.f32[i])
                i += 1
            return
        if t == _TAG_F16():
            dst.f16.reserve(n)
            while i < n:
                dst.f16.append(src.f16[i])
                i += 1
            return
        if t == _TAG_BF16():
            dst.bf16.reserve(n)
            while i < n:
                dst.bf16.append(src.bf16[i])
                i += 1
            return
        if t == _TAG_I64():
            dst.i64.reserve(n)
            while i < n:
                dst.i64.append(src.i64[i])
                i += 1
            return
        if t == _TAG_I32():
            dst.i32.reserve(n)
            while i < n:
                dst.i32.append(src.i32[i])
                i += 1
            return
        if t == _TAG_I16():
            dst.i16.reserve(n)
            while i < n:
                dst.i16.append(src.i16[i])
                i += 1
            return
        if t == _TAG_I8():
            dst.i8.reserve(n)
            while i < n:
                dst.i8.append(src.i8[i])
                i += 1
            return
        if t == _TAG_U64():
            dst.u64.reserve(n)
            while i < n:
                dst.u64.append(src.u64[i])
                i += 1
            return
        if t == _TAG_U32():
            dst.u32.reserve(n)
            while i < n:
                dst.u32.append(src.u32[i])
                i += 1
            return
        if t == _TAG_U16():
            dst.u16.reserve(n)
            while i < n:
                dst.u16.append(src.u16[i])
                i += 1
            return
        if t == _TAG_U8():
            dst.u8.reserve(n)
            while i < n:
                dst.u8.append(src.u8[i])
                i += 1
            return
        dst.b.reserve(n)
        while i < n:
            dst.b.append(src.b[i])
            i += 1

    # ------------------------- contiguity check ---------------------------

    @always_inline
    fn is_row_major_contiguous(self) -> Bool:
        var want = row_major_strides(self._shape)
        var n = len(want)
        if len(self._strides) != n:
            return False
        var i = 0
        while i < n:
            if self._strides[i] != want[i]:
                return False
            i += 1
        return True

    # --------------------------- generic interop ---------------------------

    fn __from_tensor_generic[T: ImplicitlyCopyable & Copyable & Movable](
        mut self,
        t: Tensor[T],
        tag_target: Int,
        to_f64: fn(T) -> Float64
    ):
        self._dtype = self.__dtype_from_tag(tag_target)
        self._shape = copy_list_int(t.shape())
        self._strides = row_major_strides(self._shape)
        self._buf = AnyBuffer()
        self._buf.tag = tag_target
        var n = len(t._data)
        self._buf.reserve(n, tag_target)
        var i = 0
        while i < n:
            self._buf.append_cast_from_f64(to_f64(t._data[i]), tag_target)
            i += 1

    fn __to_tensor_generic[T: ImplicitlyCopyable & Copyable & Movable](
        self,
        tag_out: Int,
        from_f64: fn(Float64) -> T
    ) -> Tensor[T]:
        var n = self._buf.length()
        var out = List[T]()
        out.reserve(n)
        if self._buf.tag == tag_out:
            self.__direct_copy_list_for_tag[T](mut out, self._buf, n, tag_out)
            return Tensor[T](out, self._shape)
        var i = 0
        while i < n:
            out.append(from_f64(self._buf.get_f64(i)))
            i += 1
        return Tensor[T](out, self._shape)

    @always_inline
    @staticmethod
    fn __direct_copy_list_for_tag[T](mut out: List[T], src: AnyBuffer, n: Int, tag: Int):
        var i = 0
        if tag == _TAG_F64():
            while i < n:
                out.append(T(src.f64[i]))
                i += 1
            return
        if tag == _TAG_F32():
            while i < n:
                out.append(T(src.f32[i]))
                i += 1
            return
        if tag == _TAG_F16():
            while i < n:
                out.append(T(src.f16[i]))
                i += 1
            return
        if tag == _TAG_BF16():
            while i < n:
                out.append(T(src.bf16[i]))
                i += 1
            return
        if tag == _TAG_I64():
            while i < n:
                out.append(T(src.i64[i]))
                i += 1
            return
        if tag == _TAG_I32():
            while i < n:
                out.append(T(src.i32[i]))
                i += 1
            return
        if tag == _TAG_I16():
            while i < n:
                out.append(T(src.i16[i]))
                i += 1
            return
        if tag == _TAG_I8():
            while i < n:
                out.append(T(src.i8[i]))
                i += 1
            return
        if tag == _TAG_U64():
            while i < n:
                out.append(T(src.u64[i]))
                i += 1
            return
        if tag == _TAG_U32():
            while i < n:
                out.append(T(src.u32[i]))
                i += 1
            return
        if tag == _TAG_U16():
            while i < n:
                out.append(T(src.u16[i]))
                i += 1
            return
        if tag == _TAG_U8():
            while i < n:
                out.append(T(src.u8[i]))
                i += 1
            return
        while i < n:
            out.append(T(src.b[i]))
            i += 1

    @always_inline
    @staticmethod
    fn __dtype_from_tag(tag: Int) -> DType:
        if tag == _TAG_F64():   return FLOAT64()
        if tag == _TAG_F32():   return FLOAT32()
        if tag == _TAG_F16():   return FLOAT16()
        if tag == _TAG_BF16():  return BF16()
        if tag == _TAG_I64():   return INT64()
        if tag == _TAG_I32():   return INT32()
        if tag == _TAG_I16():   return INT16()
        if tag == _TAG_I8():    return INT8()
        if tag == _TAG_U64():   return UINT64()
        if tag == _TAG_U32():   return UINT32()
        if tag == _TAG_U16():   return UINT16()
        if tag == _TAG_U8():    return UINT8()
        return BOOL()

    # ---------------------------- wrappers: FROM ----------------------------

    @always_inline
    fn from_tensor_f64(mut self, t: Tensor[Float64]):
        self.__from_tensor_generic[Float64](t, _TAG_F64(), __id_f64)

    @always_inline
    fn from_tensor_f32(mut self, t: Tensor[Float32]):
        self.__from_tensor_generic[Float32](t, _TAG_F32(), __f32_to_f64)

    @always_inline
    fn from_tensor_f16(mut self, t: Tensor[Float16]):
        self.__from_tensor_generic[Float16](t, _TAG_F16(), __f16_to_f64)

    @always_inline
    fn from_tensor_bf16(mut self, t: Tensor[BFloat16]):
        self.__from_tensor_generic[BFloat16](t, _TAG_BF16(), __bf16_to_f64)

    @always_inline
    fn from_tensor_i64(mut self, t: Tensor[Int64]):
        self.__from_tensor_generic[Int64](t, _TAG_I64(), __i64_to_f64)

    @always_inline
    fn from_tensor_i32(mut self, t: Tensor[Int32]):
        self.__from_tensor_generic[Int32](t, _TAG_I32(), __i32_to_f64)

    @always_inline
    fn from_tensor_i16(mut self, t: Tensor[Int16]):
        self.__from_tensor_generic[Int16](t, _TAG_I16(), __i16_to_f64)

    @always_inline
    fn from_tensor_i8(mut self, t: Tensor[Int8]):
        self.__from_tensor_generic[Int8](t, _TAG_I8(), __i8_to_f64)

    @always_inline
    fn from_tensor_u64(mut self, t: Tensor[UInt64]):
        self.__from_tensor_generic[UInt64](t, _TAG_U64(), __u64_to_f64)

    @always_inline
    fn from_tensor_u32(mut self, t: Tensor[UInt32]):
        self.__from_tensor_generic[UInt32](t, _TAG_U32(), __u32_to_f64)

    @always_inline
    fn from_tensor_u16(mut self, t: Tensor[UInt16]):
        self.__from_tensor_generic[UInt16](t, _TAG_U16(), __u16_to_f64)

    @always_inline
    fn from_tensor_u8(mut self, t: Tensor[UInt8]):
        self.__from_tensor_generic[UInt8](t, _TAG_U8(), __u8_to_f64)

    @always_inline
    fn from_tensor_bool(mut self, t: Tensor[UInt8]):
        self.__from_tensor_generic[UInt8](t, _TAG_BOOL(), __u8nz_to_f64)

    # ----------------------------- wrappers: TO -----------------------------

    @always_inline
    fn to_tensor_f64(self) -> Tensor[Float64]:
        return self.__to_tensor_generic[Float64](_TAG_F64(), __id_f64)

    @always_inline
    fn to_tensor_f32(self) -> Tensor[Float32]:
        return self.__to_tensor_generic[Float32](_TAG_F32(), __f64_to_f32)

    @always_inline
    fn to_tensor_f16(self) -> Tensor[Float16]:
        return self.__to_tensor_generic[Float16](_TAG_F16(), __f64_to_f16)

    @always_inline
    fn to_tensor_bf16(self) -> Tensor[BFloat16]:
        return self.__to_tensor_generic[BFloat16](_TAG_BF16(), __f64_to_bf16)

    @always_inline
    fn to_tensor_i64(self) -> Tensor[Int64]:
        return self.__to_tensor_generic[Int64](_TAG_I64(), __f64_to_i64)

    @always_inline
    fn to_tensor_i32(self) -> Tensor[Int32]:
        return self.__to_tensor_generic[Int32](_TAG_I32(), __f64_to_i32)

    @always_inline
    fn to_tensor_i16(self) -> Tensor[Int16]:
        return self.__to_tensor_generic[Int16](_TAG_I16(), __f64_to_i16)

    @always_inline
    fn to_tensor_i8(self) -> Tensor[Int8]:
        return self.__to_tensor_generic[Int8](_TAG_I8(), __f64_to_i8)

    @always_inline
    fn to_tensor_u64(self) -> Tensor[UInt64]:
        return self.__to_tensor_generic[UInt64](_TAG_U64(), __f64_to_u64)

    @always_inline
    fn to_tensor_u32(self) -> Tensor[UInt32]:
        return self.__to_tensor_generic[UInt32](_TAG_U32(), __f64_to_u32)

    @always_inline
    fn to_tensor_u16(self) -> Tensor[UInt16]:
        return self.__to_tensor_generic[UInt16](_TAG_U16(), __f64_to_u16)

    @always_inline
    fn to_tensor_u8(self) -> Tensor[UInt8]:
        return self.__to_tensor_generic[UInt8](_TAG_U8(), __f64_to_u8)

    @always_inline
    fn to_tensor_bool(self) -> Tensor[UInt8]:
        return self.__to_tensor_generic[UInt8](_TAG_BOOL(), __f64_to_bool_u8)

    # ----------------------------- casting API -----------------------------

    fn astype(self, dt: DType) -> AnyTensor:
        var out = AnyTensor()
        out._dtype = dt
        out._shape = copy_list_int(self._shape)
        out._strides = row_major_strides(out._shape)
        out._buf = AnyBuffer()
        var n = self._buf.length()
        var tag = Int(dt.tag.code)
        out._buf.tag = tag
        out._buf.reserve(n, tag)
        if tag == self._buf.tag:
            self.__append_same_tag_copy(mut out._buf, self._buf, n)
            return out
        var i = 0
        while i < n:
            out._buf.append_cast_from_f64(self._buf.get_f64(i), tag)
            i += 1
        return out

    fn astype_inplace(mut self, dt: DType):
        var new_tag = Int(dt.tag.code)
        if new_tag == self._buf.tag and dt.tag.code == self._dtype.tag.code:
            self._dtype = dt
            return
        var tmp = self.astype(dt)
        self._dtype = tmp._dtype
        self._shape = tmp._shape
        self._strides = tmp._strides
        self._buf = tmp._buf

# ---------------------------- converters (fn) ----------------------------

@always_inline
fn __id_f64(x: Float64) -> Float64: return x
@always_inline
fn __f32_to_f64(x: Float32) -> Float64: return Float64(x)
@always_inline
fn __f16_to_f64(x: Float16) -> Float64: return Float64(x)
@always_inline
fn __bf16_to_f64(x: BFloat16) -> Float64: return Float64(x)
@always_inline
fn __i64_to_f64(x: Int64) -> Float64: return Float64(x)
@always_inline
fn __i32_to_f64(x: Int32) -> Float64: return Float64(x)
@always_inline
fn __i16_to_f64(x: Int16) -> Float64: return Float64(x)
@always_inline
fn __i8_to_f64(x: Int8) -> Float64: return Float64(x)
@always_inline
fn __u64_to_f64(x: UInt64) -> Float64: return Float64(x)
@always_inline
fn __u32_to_f64(x: UInt32) -> Float64: return Float64(x)
@always_inline
fn __u16_to_f64(x: UInt16) -> Float64: return Float64(x)
@always_inline
fn __u8_to_f64(x: UInt8) -> Float64: return Float64(x)
@always_inline
fn __u8nz_to_f64(x: UInt8) -> Float64:
    var v = 0.0
    if x != 0:
        v = 1.0
    return v

@always_inline
fn __f64_to_f32(x: Float64) -> Float32: return Float32(x)
@always_inline
fn __f64_to_f16(x: Float64) -> Float16: return Float16(x)
@always_inline
fn __f64_to_bf16(x: Float64) -> BFloat16: return BFloat16(x)
@always_inline
fn __f64_to_i64(x: Float64) -> Int64: return Int64(x)
@always_inline
fn __f64_to_i32(x: Float64) -> Int32: return Int32(x)
@always_inline
fn __f64_to_i16(x: Float64) -> Int16: return Int16(x)
@always_inline
fn __f64_to_i8(x: Float64) -> Int8: return Int8(x)
@always_inline
fn __f64_to_u64(x: Float64) -> UInt64: return UInt64(x)
@always_inline
fn __f64_to_u32(x: Float64) -> UInt32: return UInt32(x)
@always_inline
fn __f64_to_u16(x: Float64) -> UInt16: return UInt16(x)
@always_inline
fn __f64_to_u8(x: Float64) -> UInt8: return UInt8(x)
@always_inline
fn __f64_to_bool_u8(x: Float64) -> UInt8:
    var v = UInt8(0)
    if x != 0.0:
        v = 1
    return v



# ------------------------------ TensorView[T] ------------------------------

struct TensorView[T: ImplicitlyCopyable & Copyable & Movable]:
    var _data: List[T]
    var _shape: List[Int]
    var _strides: List[Int]
    var _offset: Int

    fn __init__(out self, base: Tensor[T]):
        self._data = base._data
        self._shape = base._shape
        self._strides = base._strides
        self._offset = 0

    fn shape(self) -> List[Int]:
        return copy_list(self._shape)

    # Materialize to dense Tensor with same shape
    fn to_tensor(self) -> Tensor[T]:
        var n = numel(self._shape)
        var rm = row_major_strides(self._shape)

        # Fast path: contiguous view (any offset)
        var is_contig = lists_equal_int(self._strides, rm)
        if is_contig:
            var data = List[T]()
            data.reserve(n)
            append_block_unrolled16(data, self._data, self._offset, n)
            return Tensor[T](data, self._shape)

        # Generic path
        var data2 = List[T]()
        data2.reserve(n)
        var idx = List[Int]()
        var i = 0
        var r = len(self._shape)
        while i < n:
            unravel_index(i, self._shape, idx)
            var li = self._offset
            var d = 0
            while d < r:
                li = li + idx[d] * self._strides[d]
                d += 1
            data2.append(self._data[li])
            i += 1
        return Tensor[T](data2, self._shape)


