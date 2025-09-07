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
# Project: momijo.core
# File: src/momijo/core/traits.mojo

from builtin.dtype import DType
from momijo.core.shape import Shape

trait PrettyPrintable:
fn to_string(self) -> String

trait Describable:
fn description(self) -> String

trait Hashable64:
fn hash64(self) -> UInt64

trait Cloneable:
fn clone(self) -> Self

# -------------------------
# Size, indexing, sequences, iteration
# -------------------------

trait SizedLike:
fn size(self) -> Int

trait IndexGet[T]:
fn get(self, index: Int) -> T

trait IndexSet[T]:
fn set(self, index: Int, value: T) -> None

trait Sequence[T]:
fn size(self) -> Int
fn get(self, index: Int) -> T

trait MutableSequence[T]:
fn size(self) -> Int
fn get(self, index: Int) -> T
fn set(self, index: Int, value: T) -> None
fn append(self, value: T) -> None
fn clear(self) -> None

trait Iterable[T]:
fn iter(self) -> List[T]

trait RandomAccessIterator[T]:
fn has_next(self) -> Bool
fn next(self) -> T
fn reset(self) -> None

# -------------------------
# Numeric & ordering
# -------------------------

trait Numeric[T]:
fn zero(self) -> T
fn one(self) -> T
fn add(self, rhs: T) -> T
fn sub(self, rhs: T) -> T
fn mul(self, rhs: T) -> T
fn div(self, rhs: T) -> T

trait Orderable:
    # Return -1 if self<other, 0 if equal, +1 if self>other
fn compare(self, other: Self) -> Int

# -------------------------
# Tensor-like geometry traits
# -------------------------

trait TensorLike:
fn shape(self) -> Shape
fn dtype(self) -> DType
fn device(self) -> String
fn size(self) -> Int

trait GradCarrier:
fn requires_grad(self) -> Bool
fn has_grad(self) -> Bool

# -------------------------
# Adapters / Helper structs
# -------------------------

# A thin wrapper that turns a List[T] into a MutableSequence[T] & Iterable[T].
struct SeqList[T: Copyable & Movable](Copyable, Movable, EqualityComparable):
    var _data: List[T]
fn __init__(out self self, data: List[T] = List[T]()):
        self._data = data

    # MutableSequence[T]
fn size(self) -> Int:
        return len(self._data)
fn get(self, index: Int) -> T:
        return self._data[index]
fn set(self, index: Int, value: T) -> None:
        self._data[index] = value
fn append(self, value: T) -> None:
        self._data.append(value)
fn clear(self) -> None:
        self._data = List[T]()

    # Iterable[T]
fn iter(self) -> List[T]:
        # returns a shallow copy for safe iteration
        var out = List[T]()
        var i = 0
        while i < len(self._data):
            out.append(self._data[i])
            i += 1
        return out

    # PrettyPrintable
fn to_string(self) -> String:
        var s = "SeqList[" + String(len(self._data)) + "]{"
        var i = 0
        while i < len(self._data):
            if i > 0: s = s + ", "
            # Best-effort element stringification; assume String(T) exists
            s = s + String(self._data[i])
            i += 1
        s = s + "}"
        return s

# A simple random-access iterator over a List[T].
struct ListIterator[T: Copyable & Movable](Copyable, Movable, EqualityComparable):
    var _data: List[T]
    var _idx: Int
fn __init__(out self self, data: List[T] = List[T]()):
        self._data = data
        self._idx = 0
fn has_next(self) -> Bool:
        return self._idx < len(self._data)
fn next(self) -> T:
        var v = self._data[self._idx]
        self._idx = self._idx + 1
        return v
fn reset(self) -> None:
        self._idx = 0

# Adapter to provide TensorLike metadata without coupling to a concrete Tensor type.
@fieldwise_init
struct TensorMeta(Copyable, Movable):
fn __copyinit__(out self, other: Self) -> None:
        self = other

    var _shape: Shape
    var _dtype: DType
    var _device: String
fn __init__(out self self, shape: Shape = Shape(dims=[]), dtype: DType = DType.f32(), device: String = "cpu"):
        self._shape = shape
        self._dtype = dtype
        self._device = device
fn shape(self) -> Shape: return self._shape
fn dtype(self) -> DType: return self._dtype
fn device(self) -> String: return self._device
fn size(self) -> Int: return self._shape.count()
fn to_string(self) -> String:
        return "TensorMeta{shape=" + self._shape.to_string() + ", dtype=" + self._dtype.to_string() + ", device=" + self._device + "}"

# -------------------------
# Free helpers
# -------------------------

@staticmethod
fn print_pretty(x: PrettyPrintable) -> None:
    print(x.to_string())

@staticmethod
fn order_lt[T: Copyable & Movable](a: T, b: T) -> Bool: return a.compare(b) < 0
@staticmethod
fn order_le[T: Copyable & Movable](a: T, b: T) -> Bool: return a.compare(b) <= 0
@staticmethod
fn order_gt[T: Copyable & Movable](a: T, b: T) -> Bool: return a.compare(b) > 0
@staticmethod
fn order_ge[T: Copyable & Movable](a: T, b: T) -> Bool: return a.compare(b) >= 0

@staticmethod
fn numeric_add[T: Copyable & Movable]](x: T, y: T) -> T: return x.add(y)
@staticmethod
fn numeric_sub[T: Copyable & Movable]](x: T, y: T) -> T: return x.sub(y)
@staticmethod
fn numeric_mul[T: Copyable & Movable]](x: T, y: T) -> T: return x.mul(y)
@staticmethod
fn numeric_div[T: Copyable & Movable]](x: T, y: T) -> T: return x.div(y)