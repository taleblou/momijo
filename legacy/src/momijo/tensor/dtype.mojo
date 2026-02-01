# # MIT License
# # SPDX-License-Identifier: MIT
# # Project:      Momijo
# # Module:       momijo.tensor.dtype
# # File:         src/momijo/tensor/dtype.mojo
# # Authors:      Morteza Taleblou & Mitra Daneshmand
# # Website:      https://taleblou.ir/
# # Repository:   https://github.com/taleblou/momijo
# #
# # Ultra-lean & fast dtype layer for Momijo Tensor[T] + AnyTensor.

# from collections.list import List

# # ---------- Integral codes (compact Int32) ----------
# @always_inline fn __BOOL()     -> Int32: return 1
# @always_inline fn __INT8()     -> Int32: return 2
# @always_inline fn __INT16()    -> Int32: return 3
# @always_inline fn __INT32()    -> Int32: return 4
# @always_inline fn __INT64()    -> Int32: return 5
# @always_inline fn __UINT8()    -> Int32: return 6
# @always_inline fn __UINT16()   -> Int32: return 7
# @always_inline fn __UINT32()   -> Int32: return 8
# @always_inline fn __UINT64()   -> Int32: return 9
# @always_inline fn __FLOAT16()  -> Int32: return 10
# @always_inline fn __BFLOAT16() -> Int32: return 11
# @always_inline fn __FLOAT32()  -> Int32: return 12
# @always_inline fn __FLOAT64()  -> Int32: return 13

# # ---------- Classification ----------
# @always_inline
# fn is_float_code(t: Int32) -> Bool:
#     return t >= __FLOAT16() and t <= __FLOAT64()

# @always_inline
# fn __bitwidth_for_code(t: Int32) -> Int:
#     if t == __BOOL():     return 8
#     if t == __INT8():     return 8
#     if t == __UINT8():    return 8
#     if t == __INT16():    return 16
#     if t == __UINT16():   return 16
#     if t == __INT32():    return 32
#     if t == __UINT32():   return 32
#     if t == __INT64():    return 64
#     if t == __UINT64():   return 64
#     if t == __FLOAT16():  return 16
#     if t == __BFLOAT16(): return 16
#     if t == __FLOAT32():  return 32
#     return 64  # __FLOAT64()

# @always_inline
# fn itemsize_for_code(t: Int32) -> Int:
#     return __bitwidth_for_code(t) // 8

# # ---------- DTypeTag ----------
# struct DTypeTag(Copyable, Movable, EqualityComparable):
#     var code: Int32

#     @always_inline
#     fn __init__(out self, code: Int32):
#         self.code = code

#     @always_inline fn __eq__(self, other: DTypeTag) -> Bool: return self.code == other.code
#     @always_inline fn __ne__(self, other: DTypeTag) -> Bool: return self.code != other.code

#     @staticmethod @always_inline fn BOOL()     -> DTypeTag: return DTypeTag(__BOOL())
#     @staticmethod @always_inline fn INT8()     -> DTypeTag: return DTypeTag(__INT8())
#     @staticmethod @always_inline fn INT16()    -> DTypeTag: return DTypeTag(__INT16())
#     @staticmethod @always_inline fn INT32()    -> DTypeTag: return DTypeTag(__INT32())
#     @staticmethod @always_inline fn INT64()    -> DTypeTag: return DTypeTag(__INT64())
#     @staticmethod @always_inline fn UINT8()    -> DTypeTag: return DTypeTag(__UINT8())
#     @staticmethod @always_inline fn UINT16()   -> DTypeTag: return DTypeTag(__UINT16())
#     @staticmethod @always_inline fn UINT32()   -> DTypeTag: return DTypeTag(__UINT32())
#     @staticmethod @always_inline fn UINT64()   -> DTypeTag: return DTypeTag(__UINT64())
#     @staticmethod @always_inline fn FLOAT16()  -> DTypeTag: return DTypeTag(__FLOAT16())
#     @staticmethod @always_inline fn BFLOAT16() -> DTypeTag: return DTypeTag(__BFLOAT16())
#     @staticmethod @always_inline fn FLOAT32()  -> DTypeTag: return DTypeTag(__FLOAT32())
#     @staticmethod @always_inline fn FLOAT64()  -> DTypeTag: return DTypeTag(__FLOAT64())

# # ---------- DType ----------
# struct DType(Copyable, Movable, EqualityComparable):
#     var tag: DTypeTag
#     var itemsize: Int   # bytes

#     @always_inline
#     fn __init__(out self, tag: DTypeTag, itemsize: Int):
#         self.tag = tag
#         var b = itemsize
#         if b == 0:
#             b = itemsize_for_code(tag.code)
#         self.itemsize = b

#     @always_inline fn __eq__(self, other: DType) -> Bool:
#         return (self.tag.code == other.tag.code) and (self.itemsize == other.itemsize)

#     @always_inline fn __ne__(self, other: DType) -> Bool:
#         return not self.__eq__(other)

#     # Predicates
#     @always_inline fn is_bool(self)     -> Bool: return self.tag.code == __BOOL()
#     @always_inline fn is_unsigned(self) -> Bool:
#         var t = self.tag.code
#         return (t == __UINT8()) or (t == __UINT16()) or (t == __UINT32()) or (t == __UINT64())
#     @always_inline fn is_signed(self)   -> Bool:
#         var t = self.tag.code
#         return (t == __INT8()) or (t == __INT16()) or (t == __INT32()) or (t == __INT64())
#     @always_inline fn is_integer(self)  -> Bool:
#         return self.is_bool() or self.is_unsigned() or self.is_signed()
#     @always_inline fn is_float(self)    -> Bool: return is_float_code(self.tag.code)
#     @always_inline fn is_float32(self)  -> Bool: return self.tag.code == __FLOAT32()
#     @always_inline fn is_float64(self)  -> Bool: return self.tag.code == __FLOAT64()
#     @always_inline fn nbits(self)       -> Int:  return __bitwidth_for_code(self.tag.code)
#     @always_inline fn canonical_itemsize(self) -> Int: return itemsize_for_code(self.tag.code)
#     @always_inline fn nbytes(self)      -> Int:  return self.itemsize

#     @always_inline
#     fn ravel(self) -> String:
#         var t = self.tag.code
#         if t == __BOOL():     return String("bool")
#         if t == __INT8():     return String("int8")
#         if t == __INT16():    return String("int16")
#         if t == __INT32():    return String("int32")
#         if t == __INT64():    return String("int64")
#         if t == __UINT8():    return String("uint8")
#         if t == __UINT16():   return String("uint16")
#         if t == __UINT32():   return String("uint32")
#         if t == __UINT64():   return String("uint64")
#         if t == __FLOAT16():  return String("float16")
#         if t == __BFLOAT16(): return String("bfloat16")
#         if t == __FLOAT32():  return String("float32")
#         return String("float64")

#     @always_inline
#     fn __str__(self) -> String:
#         return self.ravel()

# # ---------- Fast factories (UPPERCASE) ----------
# @always_inline fn BOOL()    -> DType: return DType(DTypeTag(__BOOL()),     1)
# @always_inline fn INT8()    -> DType: return DType(DTypeTag(__INT8()),     1)
# @always_inline fn INT16()   -> DType: return DType(DTypeTag(__INT16()),    2)
# @always_inline fn INT32()   -> DType: return DType(DTypeTag(__INT32()),    4)
# @always_inline fn INT64()   -> DType: return DType(DTypeTag(__INT64()),    8)
# @always_inline fn UINT8()   -> DType: return DType(DTypeTag(__UINT8()),    1)
# @always_inline fn UINT16()  -> DType: return DType(DTypeTag(__UINT16()),   2)
# @always_inline fn UINT32()  -> DType: return DType(DTypeTag(__UINT32()),   4)
# @always_inline fn UINT64()  -> DType: return DType(DTypeTag(__UINT64()),   8)
# @always_inline fn FLOAT16() -> DType: return DType(DTypeTag(__FLOAT16()),  2)
# @always_inline fn BF16()    -> DType: return DType(DTypeTag(__BFLOAT16()), 2)
# @always_inline fn FLOAT32() -> DType: return DType(DTypeTag(__FLOAT32()),  4)
# @always_inline fn FLOAT64() -> DType: return DType(DTypeTag(__FLOAT64()),  8)

# # ---------- Lowercase aliases ----------
# @always_inline fn bool_()   -> DType: return BOOL()
# @always_inline fn int8()    -> DType: return INT8()
# @always_inline fn int16()   -> DType: return INT16()
# @always_inline fn int32()   -> DType: return INT32()
# @always_inline fn int64()   -> DType: return INT64()
# @always_inline fn uint8()   -> DType: return UINT8()
# @always_inline fn uint16()  -> DType: return UINT16()
# @always_inline fn uint32()  -> DType: return UINT32()
# @always_inline fn uint64()  -> DType: return UINT64()
# @always_inline fn float16() -> DType: return FLOAT16()
# @always_inline fn bf16()    -> DType: return BF16()
# @always_inline fn float32() -> DType: return FLOAT32()
# @always_inline fn float64() -> DType: return FLOAT64()

# # ---------- Predicates as free functions ----------
# @always_inline
# fn is_int(tag: DTypeTag) -> Bool:
#     var t = tag.code
#     if is_float_code(t): return False
#     if t == __BOOL():   return True
#     if t == __INT8():   return True
#     if t == __INT16():  return True
#     if t == __INT32():  return True
#     if t == __INT64():  return True
#     if t == __UINT8():  return True
#     if t == __UINT16(): return True
#     if t == __UINT32(): return True
#     if t == __UINT64(): return True
#     return False

# # ---------- Registry / defaults ----------
# @always_inline
# fn all_dtypes() -> List[DType]:
#     var xs = List[DType]()
#     xs.reserve(13)
#     xs.append(BOOL())
#     xs.append(UINT8())
#     xs.append(INT8())
#     xs.append(INT16())
#     xs.append(INT32())
#     xs.append(INT64())
#     xs.append(UINT16())
#     xs.append(UINT32())
#     xs.append(UINT64())
#     xs.append(FLOAT16())
#     xs.append(BF16())
#     xs.append(FLOAT32())
#     xs.append(FLOAT64())
#     return xs

# @always_inline
# fn default_dtype() -> DType:
#     return FLOAT32()

# # ---------- AnyTensor shims ----------
# @always_inline fn _DT_FLOAT64() -> Int: return Int(__FLOAT64())
# @always_inline fn _DT_FLOAT32() -> Int: return Int(__FLOAT32())
# @always_inline fn _DT_INT64()   -> Int: return Int(__INT64())
# @always_inline fn _DT_INT32()   -> Int: return Int(__INT32())
# @always_inline fn _DT_UINT8()   -> Int: return Int(__UINT8())

# @always_inline fn dtype_of_float64() -> DType: return FLOAT64()
# @always_inline fn dtype_of_float32() -> DType: return FLOAT32()
# @always_inline fn dtype_of_int64()   -> DType: return INT64()
# @always_inline fn dtype_of_int32()   -> DType: return INT32()
# @always_inline fn dtype_of_uint8()   -> DType: return UINT8()

# # ---------- Promotion lattice ----------
# # bool < u8 < i8 < i16 < u16 < i32 < u32 < i64 < u64 < f16 ~ bf16 < f32 < f64
# @always_inline
# fn __rank_code(t: Int32) -> Int32:
#     if t == __BOOL():     return 0
#     if t == __UINT8():    return 1
#     if t == __INT8():     return 2
#     if t == __INT16():    return 3
#     if t == __UINT16():   return 4
#     if t == __INT32():    return 5
#     if t == __UINT32():   return 6
#     if t == __INT64():    return 7
#     if t == __UINT64():   return 8
#     if t == __FLOAT16():  return 9
#     if t == __BFLOAT16(): return 9
#     if t == __FLOAT32():  return 10
#     return 11  # __FLOAT64()

# @always_inline
# fn __from_code_fast(t: Int32) -> DType:
#     if t == __BOOL():     return BOOL()
#     if t == __UINT8():    return UINT8()
#     if t == __INT8():     return INT8()
#     if t == __INT16():    return INT16()
#     if t == __UINT16():   return UINT16()
#     if t == __INT32():    return INT32()
#     if t == __UINT32():   return UINT32()
#     if t == __INT64():    return INT64()
#     if t == __UINT64():   return UINT64()
#     if t == __FLOAT16():  return FLOAT16()
#     if t == __BFLOAT16(): return BF16()
#     if t == __FLOAT32():  return FLOAT32()
#     return FLOAT64()

# @always_inline
# fn promote_dtype(a: DType, b: DType) -> DType:
#     var ta = a.tag.code
#     var tb = b.tag.code
#     if is_float_code(ta) or is_float_code(tb):
#         if (ta == __FLOAT64()) or (tb == __FLOAT64()): return FLOAT64()
#         if (ta == __FLOAT32()) or (tb == __FLOAT32()): return FLOAT32()
#         return __from_code_fast(__FLOAT16())  # f16 / bf16 tier
#     var ra = __rank_code(ta)
#     var rb = __rank_code(tb)
#     return a if ra >= rb else b

# # ---------- Casting helpers ----------
# # mode: 0 = unsafe (allow narrowing), 1 = safe (only widening/float-up)
# @always_inline
# fn can_cast(src: DType, dst: DType, mode: Int) -> Bool:
#     if mode == 0: return True
#     var s = src.tag.code
#     var d = dst.tag.code
#     if s == d: return True
#     if not is_float_code(s):
#         if is_float_code(d): return True
#         return __rank_code(d) >= __rank_code(s)
#     if is_float_code(d):
#         if (s == __FLOAT16()) or (s == __BFLOAT16()): return True
#         if s == __FLOAT32(): return (d == __FLOAT32()) or (d == __FLOAT64())
#         return d == __FLOAT64()
#     return False

# # ---------- Conversions ----------
# @always_inline
# fn to_code(dt: DType) -> Int32:
#     return dt.tag.code

# @always_inline
# fn from_code(code: Int32) -> DType:
#     # Falls back to FLOAT64 if unknown (defensive).
#     return __from_code_fast(code)

# @always_inline
# fn from_name(name: String) -> DType:
#     # Lowercase matching; unknown -> FLOAT64
#     var s = name.lower()
#     if s == "bool":       return BOOL()
#     if s == "uint8":      return UINT8()
#     if s == "int8":       return INT8()
#     if s == "int16":      return INT16()
#     if s == "uint16":     return UINT16()
#     if s == "int32":      return INT32()
#     if s == "uint32":     return UINT32()
#     if s == "int64":      return INT64()
#     if s == "uint64":     return UINT64()
#     if s == "float16":    return FLOAT16()
#     if (s == "bf16") or (s == "bfloat16"): return BF16()
#     if (s == "float32") or (s == "f32"):   return FLOAT32()
#     return FLOAT64()
