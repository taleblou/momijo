# MIT License
# Copyright (c) 2025 Morteza Talebou and Mitra Daneshmand
# Project: momijo  |  Source: https://github.com/taleblou/momijo
# This file is part of the Momijo project. See the LICENSE file at the repository root.
# Momijo
# SPDX-License-Identifier: MIT
# Website: https://taleblou.ir/
# Repository: https://github.com/taleblou/momijo
#
# Project: momijo.tensor
# File: src/momijo/tensor/dtype.mojo


# ---------- DTypeTag ----------
struct DTypeTag(Copyable, Movable, EqualityComparable):
    # Codes: 1..13 (bool..float64)
    var code: Int32

    fn __init__(out self, code: Int32):
        self.code = code

    fn __copyinit__(out self, other: Self):
        self.code = other.code

    @always_inline
    fn __eq__(self, other: DTypeTag) -> Bool:
        return self.code == other.code

    @always_inline
    fn __ne__(self, other: DTypeTag) -> Bool:
        return self.code != other.code

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

    fn __init__(out self, tag: DTypeTag, itemsize: Int):
        self.tag = tag
        self.itemsize = itemsize

    fn __copyinit__(out self, other: Self):
        self.tag = other.tag
        self.itemsize = other.itemsize

    @always_inline
    fn __eq__(self, other: DType) -> Bool:
        return (self.tag == other.tag) and (self.itemsize == other.itemsize)

    @always_inline
    fn __ne__(self, other: DType) -> Bool:
        return not self.__eq__(other)

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

    fn nbits(self) -> Int:
        return _nbits_for_tag(self.tag)

    fn canonical_itemsize(self) -> Int:
        return _itemsize_for_tag(self.tag)

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

# ---------- DType factories (UPPERCASE) ----------
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

# ---------- DType factories (lowercase aliases expected by tests) ----------
fn bool_()   -> DType: return BOOL()
fn int8()    -> DType: return INT8()
fn int16()   -> DType: return INT16()
fn int32()   -> DType: return INT32()
fn int64()   -> DType: return INT64()
fn uint8()   -> DType: return UINT8()
fn uint16()  -> DType: return UINT16()
fn uint32()  -> DType: return UINT32()
fn uint64()  -> DType: return UINT64()
fn float16() -> DType: return FLOAT16()
fn bf16()    -> DType: return BF16()
fn float32() -> DType: return FLOAT32()
fn float64() -> DType: return FLOAT64()

# ---------- Predicates used by tensor.mojo as free functions ----------
fn is_int(tag: DTypeTag) -> Bool:
    var dt = DType(tag, _itemsize_for_tag(tag))
    return dt.is_integer()

fn is_float(tag: DTypeTag) -> Bool:
    var dt = DType(tag, _itemsize_for_tag(tag))
    return dt.is_float()

# ---------- Simple registry ----------
fn all_dtypes() -> List[DType]:
    var xs = List[DType]()
    xs.append(BOOL()); xs.append(UINT8()); xs.append(INT8()); xs.append(INT16()); xs.append(INT32()); xs.append(INT64())
    xs.append(UINT16()); xs.append(UINT32()); xs.append(UINT64())
    xs.append(FLOAT16()); xs.append(BF16()); xs.append(FLOAT32()); xs.append(FLOAT64())
    return xs

fn default_dtype() -> DType:
    return FLOAT32()

# ---------- Casting/promote (stubs – can be expanded later) ----------
fn can_cast(src: DType, dst: DType, mode: Int) -> Bool:
    if src == dst: return True
    return True

fn promote(a: DType, b: DType) -> DType:
    if a.is_float() or b.is_float():
        if a.is_float64() or b.is_float64(): return FLOAT64()
        return FLOAT32()
    if a.nbits() >= b.nbits(): return a
    return b
