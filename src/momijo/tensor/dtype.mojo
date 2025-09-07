# MIT License
# Copyright (c) 2025 Morteza Talebou and Mitra Daneshmand
# Project: momijo  |  Source: https://github.com/taleblou/momijo
# This file is part of the Momijo project. See the LICENSE file at the repository root.
# Momijo 
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Morteza Talebou and Mitra Daneshmand
# Website: https://taleblou.ir/
# Repository: https://github.com/taleblou/momijo
#
# Project: momijo.tensor
# File: src/momijo/tensor/dtype.mojo

fn __module_name__() -> String:
    return String("momijo/tensor/dtype.mojo")
fn __self_test__() -> Bool:
    return True

# ---------- Small helpers ----------
fn argmax_index(xs: List[Float64]) -> Int:
    if len(xs) == 0: return -1
    var best = xs[0]
    var idx = 0
    var i = 1
    while i < len(xs):
        if xs[i] > best:
            best = xs[i]
            idx = i
        i += 1
    return idx
fn argmin_index(xs: List[Float64]) -> Int:
    if len(xs) == 0: return -1
    var best = xs[0]
    var idx = 0
    var i = 1
    while i < len(xs):
        if xs[i] < best:
            best = xs[i]
            idx = i
        i += 1
    return idx

# ---------- DTypeTag ----------
struct DTypeTag(Copyable, Movable, EqualityComparable):
    # Codes: 1..13 (bool..float64)
    var code: Int32
fn __init__(out self, code: Int32) -> None:
        self.code = code
fn __copyinit__(out self, other: Self) -> None:
        self.code = other.code

    @always_inline
fn __eq__(self, other: DTypeTag) -> Bool:
        return self.code == other.code

    @always_inline
fn __ne__(self, other: DTypeTag) -> Bool:
        return self.code != other.code

    # Static constructors (decorator must be on its own line)
    @staticmethod
fn BOOL() -> DTypeTag:
        return DTypeTag(Int32(1))

    @staticmethod
fn INT8() -> DTypeTag:
        return DTypeTag(Int32(2))

    @staticmethod
fn INT16() -> DTypeTag:
        return DTypeTag(Int32(3))

    @staticmethod
fn INT32() -> DTypeTag:
        return DTypeTag(Int32(4))

    @staticmethod
fn INT64() -> DTypeTag:
        return DTypeTag(Int32(5))

    @staticmethod
fn UINT8() -> DTypeTag:
        return DTypeTag(Int32(6))

    @staticmethod
fn UINT16() -> DTypeTag:
        return DTypeTag(Int32(7))

    @staticmethod
fn UINT32() -> DTypeTag:
        return DTypeTag(Int32(8))

    @staticmethod
fn UINT64() -> DTypeTag:
        return DTypeTag(Int32(9))

    @staticmethod
fn FLOAT16() -> DTypeTag:
        return DTypeTag(Int32(10))

    @staticmethod
fn BFLOAT16() -> DTypeTag:
        return DTypeTag(Int32(11))

    @staticmethod
fn FLOAT32() -> DTypeTag:
        return DTypeTag(Int32(12))

    @staticmethod
fn FLOAT64() -> DTypeTag:
        return DTypeTag(Int32(13))

# ---------- Tag-to-size helpers ----------
fn _nbits_for_tag(tag: DTypeTag) -> Int:
    var t: Int32 = tag.code
    if t == 1:   return 1    # bool
    if t == 2:   return 8    # int8
    if t == 3:   return 16   # int16
    if t == 4:   return 32   # int32
    if t == 5:   return 64   # int64
    if t == 6:   return 8    # uint8
    if t == 7:   return 16   # uint16
    if t == 8:   return 32   # uint32
    if t == 9:   return 64   # uint64
    if t == 10:  return 16   # float16
    if t == 11:  return 16   # bfloat16
    if t == 12:  return 32   # float32
    # t == 13
    return 64                 # float64
fn _itemsize_for_tag(tag: DTypeTag) -> Int:
    var b: Int = _nbits_for_tag(tag)
    if b <= 8:   return 1
    if b <= 16:  return 2
    if b <= 32:  return 4
    return 8

# ---------- DType ----------
struct DType(Copyable, Movable, EqualityComparable):
    var tag: DTypeTag
    var itemsize: Int
fn __init__(out self, tag: DTypeTag, itemsize: Int) -> None:
        self.tag = tag
        self.itemsize = itemsize
fn __copyinit__(out self, other: Self) -> None:
        self.tag = other.tag
        self.itemsize = other.itemsize

    @always_inline
fn __eq__(self, other: DType) -> Bool:
        return (self.tag == other.tag) and (self.itemsize == other.itemsize)

    @always_inline
fn __ne__(self, other: DType) -> Bool:
        return not self.__eq__(other)

    # Predicates
fn is_bool(self) -> Bool:
        return self.tag == DTypeTag.BOOL()
fn is_unsigned(self) -> Bool:
        var t: Int32 = self.tag.code
        return (t == 6) or (t == 7) or (t == 8) or (t == 9)
fn is_signed(self) -> Bool:
        var t: Int32 = self.tag.code
        return (t == 2) or (t == 3) or (t == 4) or (t == 5)
fn is_integer(self) -> Bool:
        return self.is_unsigned() or self.is_signed() or self.is_bool()
fn is_float(self) -> Bool:
        var t: Int32 = self.tag.code
        return (t == 10) or (t == 11) or (t == 12) or (t == 13)
fn is_float32(self) -> Bool:
        return self.tag == DTypeTag.FLOAT32()
fn is_float64(self) -> Bool:
        return self.tag == DTypeTag.FLOAT64()

    # Sizes
fn nbits(self) -> Int:
        return _nbits_for_tag(self.tag)
fn canonical_itemsize(self) -> Int:
        return _itemsize_for_tag(self.tag)

    # String
fn to_string(self) -> String:
        var t: Int32 = self.tag.code
        if t == 1:  return String("bool")
        if t == 2:  return String("int8")
        if t == 3:  return String("int16")
        if t == 4:  return String("int32")
        if t == 5:  return String("int64")
        if t == 6:  return String("uint8")
        if t == 7:  return String("uint16")
        if t == 8:  return String("uint32")
        if t == 9:  return String("uint64")
        if t == 10: return String("float16")
        if t == 11: return String("bfloat16")
        if t == 12: return String("float32")
        return String("float64")

# ---------- DType factories (free functions) ----------
fn BOOL()    -> DType: return DType(DTypeTag.BOOL(),     _itemsize_for_tag(DTypeTag.BOOL()))
fn INT8()    -> DType: return DType(DTypeTag.INT8(),     _itemsize_for_tag(DTypeTag.INT8()))
fn INT16()   -> DType: return DType(DTypeTag.INT16(),    _itemsize_for_tag(DTypeTag.INT16()))
fn INT32()   -> DType: return DType(DTypeTag.INT32(),    _itemsize_for_tag(DTypeTag.INT32()))
fn INT64()   -> DType: return DType(DTypeTag.INT64(),    _itemsize_for_tag(DTypeTag.INT64()))
fn UINT8()   -> DType: return DType(DTypeTag.UINT8(),    _itemsize_for_tag(DTypeTag.UINT8()))
fn UINT16()  -> DType: return DType(DTypeTag.UINT16(),   _itemsize_for_tag(DTypeTag.UINT16()))
fn UINT32()  -> DType: return DType(DTypeTag.UINT32(),   _itemsize_for_tag(DTypeTag.UINT32()))
fn UINT64()  -> DType: return DType(DTypeTag.UINT64(),   _itemsize_for_tag(DTypeTag.UINT64()))
fn FLOAT16() -> DType: return DType(DTypeTag.FLOAT16(),  _itemsize_for_tag(DTypeTag.FLOAT16()))
fn BF16()    -> DType: return DType(DTypeTag.BFLOAT16(), _itemsize_for_tag(DTypeTag.BFLOAT16()))
fn FLOAT32() -> DType: return DType(DTypeTag.FLOAT32(),  _itemsize_for_tag(DTypeTag.FLOAT32()))
fn FLOAT64() -> DType: return DType(DTypeTag.FLOAT64(),  _itemsize_for_tag(DTypeTag.FLOAT64()))
fn is_floating(dt: DType) -> Bool:
    return dt.is_float()

# ---------- Registry / parsing ----------
fn all_dtypes() -> List[DType]:
    var xs = List[DType]()
    xs.append(BOOL())
    xs.append(UINT8())
    xs.append(INT8())
    xs.append(INT16())
    xs.append(INT32())
    xs.append(INT64())
    xs.append(UINT16())
    xs.append(UINT32())
    xs.append(UINT64())
    xs.append(FLOAT16())
    xs.append(BF16())
    xs.append(FLOAT32())
    xs.append(FLOAT64())
    return xs
fn default_dtype() -> DType:
    return FLOAT32()

# simple multi-case matching without byte indexing
fn dtype_from_string(text: String) -> (Bool, DType):
    if text == String("bool") or text == String("BOOL"):
        return (True, BOOL())

    if (text == String("uint8")) or (text == String("UINT8")) or
       (text == String("u8"))    or (text == String("U8")):
        return (True, UINT8())

    if (text == String("int8")) or (text == String("INT8")) or
       (text == String("i8"))   or (text == String("I8")):
        return (True, INT8())

    if (text == String("int16")) or (text == String("INT16")) or
       (text == String("i16"))   or (text == String("I16")):
        return (True, INT16())

    if (text == String("int32")) or (text == String("INT32")) or
       (text == String("i32"))   or (text == String("I32")):
        return (True, INT32())

    if (text == String("int64")) or (text == String("INT64")) or
       (text == String("i64"))   or (text == String("I64")):
        return (True, INT64())

    if (text == String("uint16")) or (text == String("UINT16")) or
       (text == String("u16"))    or (text == String("U16")):
        return (True, UINT16())

    if (text == String("uint32")) or (text == String("UINT32")) or
       (text == String("u32"))    or (text == String("U32")):
        return (True, UINT32())

    if (text == String("uint64")) or (text == String("UINT64")) or
       (text == String("u64"))    or (text == String("U64")):
        return (True, UINT64())

    if (text == String("float16")) or (text == String("FLOAT16")) or
       (text == String("f16"))     or (text == String("F16")):
        return (True, FLOAT16())

    if (text == String("bfloat16")) or (text == String("BFLOAT16")) or
       (text == String("bf16"))     or (text == String("BF16")):
        return (True, BF16())

    if (text == String("float32")) or (text == String("FLOAT32")) or
       (text == String("f32"))     or (text == String("F32")):
        return (True, FLOAT32())

    if (text == String("float64")) or (text == String("FLOAT64")) or
       (text == String("f64"))     or (text == String("F64")) or
       (text == String("double"))  or (text == String("DOUBLE")):
        return (True, FLOAT64())

    return (False, FLOAT32())

# ---------- Casting / promotion ----------
# mode: 0=unsafe, 1=safe, 2=lossless
fn _is_lossless_int_to_int(src: DType, dst: DType) -> Bool:
    if not src.is_integer() or not dst.is_integer():
        return False
    var s_bits: Int = src.nbits()
    var d_bits: Int = dst.nbits()
    if d_bits < s_bits:
        return False
    if src.is_unsigned() and dst.is_signed() and d_bits == s_bits:
        return False
    return True
fn _is_lossless_float_to_float(src: DType, dst: DType) -> Bool:
    if not src.is_float() or not dst.is_float():
        return False
    return dst.nbits() >= src.nbits()
fn _is_lossless_bool_to_numeric(dst: DType) -> Bool:
    return dst.is_integer() or dst.is_float()
fn can_cast(src: DType, dst: DType, mode: Int) -> Bool:
    if src == dst:
        return True

    if mode == 2:
        if src.is_bool():
            return _is_lossless_bool_to_numeric(dst)
        if src.is_integer() and dst.is_integer():
            return _is_lossless_int_to_int(src, dst)
        if src.is_float() and dst.is_float():
            return _is_lossless_float_to_float(src, dst)
        return False

    if mode == 1:
        if src.is_bool():
            return _is_lossless_bool_to_numeric(dst)
        if src.is_integer() and dst.is_integer():
            var s_bits: Int = src.nbits()
            var d_bits: Int = dst.nbits()
            if d_bits < s_bits:
                return False
            if src.is_unsigned() and dst.is_signed() and d_bits == s_bits:
                return False
            return True
        if src.is_integer() and dst.is_float():
            return False
        if src.is_float() and dst.is_float():
            return dst.nbits() >= src.nbits()
        if src.is_float() and dst.is_integer():
            return False
        return False

    # mode == 0 (unsafe)
    if src.is_bool():
        return dst.is_integer() or dst.is_float()
    if src.is_integer() and (dst.is_integer() or dst.is_float() or dst.is_bool()):
        return True
    if src.is_float() and (dst.is_float() or dst.is_integer() or dst.is_bool()):
        return True
    return False
fn promote(a: DType, b: DType) -> DType:
    if a == b:
        return a
    # floats dominate ints
    if a.is_float() or b.is_float():
        if a.is_float64() or b.is_float64():
            return FLOAT64()
        return FLOAT32()

    # integers
    var aw: Int = a.nbits()
    var bw: Int = b.nbits()
    var width: Int = aw
    if bw > width: width = bw

    var final_signed: Bool = a.is_signed() or b.is_signed()
    if (a.is_unsigned() and b.is_signed()) or (a.is_signed() and b.is_unsigned()):
        if aw == bw:
            if width <= 8:       width = 32
            elif width <= 32:    width = 64
            else:                width = 64
        final_signed = True

    if final_signed:
        if width <= 8:   return INT8()
        if width <= 32:  return INT32()
        return INT64()

    # unsigned-only fallback
    if width <= 8:   return UINT8()
    if width <= 16:  return UINT16()
    if width <= 32:  return UINT32()
    return UINT64()