# MIT License
# SPDX-License-Identifier: MIT
# Project:      Momijo
# Module:       learn.data.dataset
# File:         src/momijo/learn/data/dataset.mojo
#
# Description:
#   Dataset utilities for Momijo Learn.
#   - Dataset[T]: generic indexed dataset with bound-checked access and list-like ops.
#   - TupleDataset[A,B]: paired (X, y) dataset for supervised learning.
#   - IterableDataset[T] + IndexIter: lightweight index-based iterator pattern
#     designed to integrate with DataLoader/Sampler layers (shuffling happens elsewhere).
#   - IndexAdapter[B,T]: adapts any base B that provides __len__/__getitem__ to a T stream.
#   - IndexSliceDataset[T,B]: a contiguous slice view over a base.
#   - IndexListDataset[T,B]: an arbitrary-indices view over a base.
#   - ZipDataset[A,B,BA,BB]: zips two bases into (A,B) items; length = min(lenA,lenB).
#   - EnumerateDataset[T,B]: wraps a base as (index, item) pairs.
#
# Notes:
#   - Types T, A, B must be Copyable & Movable for safe value semantics.
#   - No globals; constructors via `fn __init__(out self, ...)`.
#   - Backend-agnostic (no tensor dependencies).
#   - All __getitem__ in views clamp out-of-range indices and return a value (no Optional).
#   - Use get(...) when a default value fallback is preferred.
#
# Minimal expected base API for adapters:
#   fn __len__(self) -> Int
#   fn __getitem__(self, idx: Int) -> TLike
#
# Example usage (slice view):
#   var base = Dataset[Int].from_list([1,2,3,4,5])
#   var ad   = IndexAdapter[Dataset[Int], Int](base)
#   var sv   = IndexSliceDataset[Int, Dataset[Int]](ad, 1, 3, fn() -> Int: return 0)  # [2,3,4]
#
# Example usage (list view + random split):
#   var (idxA, idxB) = make_random_index_partition(10, 0.8, 123)
#   var lvA = make_index_list[Int, Dataset[Int]](base, idxA, fn() -> Int: return 0)
#   var lvB = make_index_list[Int, Dataset[Int]](base, idxB, fn() -> Int: return 0)

from collections.list import List

from momijo.learn.data.image import ImageFolderDataset
from momijo.learn.data.pair import Pair
# -----------------------------------------------------------------------------
# Generic indexed dataset
# -----------------------------------------------------------------------------

struct Dataset[T: Copyable & Movable](Copyable, Movable):
    var items: List[T]

    fn __init__(out self):
        self.items = List[T]()

    fn __copyinit__(out self, other: Self):
        self.items = other.items.copy()

    @staticmethod
    fn from_list(items: List[T]) -> Dataset[T]:
        var ds = Dataset[T]()
        ds.items = items.copy()
        return ds.copy()

    fn with_items(out self, items: List[T]):
        self.items = items

    fn __len__(self) -> Int:
        return len(self.items)

    fn is_empty(self) -> Bool:
        return len(self.items) == 0

    fn front(self) -> T:
        var n = len(self.items)
        if n == 0:
            return T()
        return self.items[0]

    fn back(self) -> T:
        var n = len(self.items)
        if n == 0:
            return T()
        return self.items[n - 1]

    fn __getitem__(self, idx: Int) -> T:
        # Clamp index into valid range and return a copy.
        var n = len(self.items)
        if n == 0:
            return T()
        var i = idx
        if i < 0: i = 0
        if i >= n: i = n - 1
        return self.items[i].copy()

    fn get(self, idx: Int, default: T) -> T:
        if idx < 0:
            return default
        var n = len(self.items)
        if idx >= n:
            return default
        return self.items[idx]

    fn set(mut self, idx: Int, value: T):
        var n = len(self.items)
        if n == 0:
            return
        var i = idx
        if i < 0 or i >= n:
            return
        self.items[i] = value

    fn append(mut self, value: T):
        self.items.append(value.copy())

    fn extend(mut self, values: List[T]):
        var i = 0
        var n = len(values)
        while i < n:
            self.items.append(values[i])
            i = i + 1

    fn insert(mut self, idx: Int, value: T):
        var n = len(self.items)
        var pos = idx
        if pos < 0: pos = 0
        if pos > n: pos = n
        if n == 0:
            self.items.append(value)
            return
        var last = self.items[n - 1]
        self.items.append(last)
        var j = n - 1
        while j > pos:
            self.items[j] = self.items[j - 1]
            j = j - 1
        self.items[pos] = value

    fn pop(mut self) -> T:
        var n = len(self.items)
        if n == 0:
            return T()
        var v = self.items[n - 1]
        var out = List[T]()
        var i = 0
        while i < n - 1:
            out.append(self.items[i])
            i = i + 1
        self.items = out
        return v

    fn clear(mut self):
        self.items = List[T]()

    fn to_list(self) -> List[T]:
        var out_list = List[T]()
        var i = 0
        var n = len(self.items)
        while i < n:
            out_list.append(self.items[i])
            i = i + 1
        return out_list

    fn slice(self, start: Int, stop: Int) -> Dataset[T]:
        var n = len(self.items)
        var s = start
        var e = stop
        if s < 0: s = 0
        if e < s: e = s
        if e > n: e = n
        var out_items = List[T]()
        var i = s
        while i < e:
            out_items.append(self.items[i])
            i = i + 1
        return Dataset[T].from_list(out_items)

    fn map[U: Copyable & Movable](self, f) -> Dataset[U]:
        var out_items = List[U]()
        var i = 0
        var n = len(self.items)
        while i < n:
            out_items.append(f(self.items[i]))
            i = i + 1
        return Dataset[U].from_list(out_items)

    fn filter(self, pred) -> Dataset[T]:
        var out_items = List[T]()
        var i = 0
        var n = len(self.items)
        while i < n:
            var x = self.items[i]
            if pred(x):
                out_items.append(x)
            i = i + 1
        return Dataset[T].from_list(out_items)

    fn concat(self, other: Dataset[T]) -> Dataset[T]:
        var out_items = List[T]()
        var n0 = len(self.items)
        var i = 0
        while i < n0:
            out_items.append(self.items[i])
            i = i + 1
        var n1 = len(other.items)
        var j = 0
        while j < n1:
            out_items.append(other.items[j])
            j = j + 1
        return Dataset[T].from_list(out_items)


# -----------------------------------------------------------------------------
# Simple tuple/pair dataset for supervised learning (X, y)
# -----------------------------------------------------------------------------
struct TupleDataset[A: Copyable & Movable, B: Copyable & Movable](Copyable, Movable):
    var x: List[A]
    var y: List[B]

    fn __init__(out self):
        self.x = List[A]()
        self.y = List[B]()

    fn __copyinit__(out self, other: Self):
        self.x = other.x
        self.y = other.y

    @staticmethod
    fn from_lists(x: List[A], y: List[B]) -> TupleDataset[A, B]:
        var nx = len(x)
        var ny = len(y)
        var m = (nx if nx < ny else ny)
        var xx = List[A]()
        var yy = List[B]()
        var i = 0
        while i < m:
            xx.append(x[i])
            yy.append(y[i])
            i = i + 1
        var ds = TupleDataset[A, B]()
        ds.x = xx
        ds.y = yy
        return ds

    fn with_lists(out self, x: List[A], y: List[B]):
        var nx = len(x)
        var ny = len(y)
        var m = (nx if nx < ny else ny)
        self.x = List[A]()
        self.y = List[B]()
        var i = 0
        while i < m:
            self.x.append(x[i])
            self.y.append(y[i])
            i = i + 1

    fn __len__(self) -> Int:
        return len(self.x)

    fn is_empty(self) -> Bool:
        return len(self.x) == 0

    fn __getitem__(self, idx: Int) -> (A, B):
        var n = len(self.x)
        if n == 0:
            return (A(), B())
        var i = idx
        if i < 0: i = 0
        if i >= n: i = n - 1
        return (self.x[i], self.y[i])

    fn get(self, idx: Int, default_a: A, default_b: B) -> (A, B):
        if idx < 0:
            return (default_a, default_b)
        var n = len(self.x)
        if idx >= n:
            return (default_a, default_b)
        return (self.x[idx], self.y[idx])

    fn append(mut self, a: A, b: B):
        self.x.append(a)
        self.y.append(b)

    fn to_lists(self) -> (List[A], List[B]):
        var cx = List[A]()
        var cy = List[B]()
        var i = 0
        var n = len(self.x)
        while i < n:
            cx.append(self.x[i])
            cy.append(self.y[i])
            i = i + 1
        return (cx, cy)

    fn to_dataset(self) -> Dataset[(A, B)]:
        var out = List[(A, B)]()
        var n = len(self.x)
        var i = 0
        while i < n:
            out.append((self.x[i], self.y[i]))
            i = i + 1
        return Dataset[(A, B)].from_list(out)


# -----------------------------------------------------------------------------
# Iterator and iterable wrapper
# -----------------------------------------------------------------------------

struct IndexIter(Copyable, Movable):
    var i: Int
    var n: Int

    fn __init__(out self, n: Int):
        self.i = 0
        self.n = n

    fn __copyinit__(out self, other: Self):
        self.i = other.i
        self.n = other.n

    fn reset(mut self):
        self.i = 0

    fn done(self) -> Bool:
        return self.i >= self.n

struct IterableDataset[T: Copyable & Movable](Copyable, Movable):
    var data: Dataset[T]

    fn __init__(out self):
        self.data = Dataset[T]()

    fn __copyinit__(out self, other: Self):
        self.data = other.data

    fn with_dataset(out self, data: Dataset[T]):
        self.data = data

    fn __len__(self) -> Int:
        return self.data.__len__()

    fn _make_index_iter(self) -> IndexIter:
        return IndexIter(self.__len__())

    fn next_item(self, mut it: IndexIter) -> Optional[T]:
        if it.done():
            return None
        var idx = it.i
        it.i = it.i + 1
        return Optional[T](self.data.__getitem__(idx))


# -----------------------------------------------------------------------------
# Adapters and views
# -----------------------------------------------------------------------------

struct IndexAdapter[B: Copyable & Movable, T: Copyable & Movable](Copyable, Movable):
    var base: B
    var _len: fn(B) -> Int
    var _get: fn(B, Int) -> T

    fn __init__(out self, base: B, len_fn: fn(B) -> Int, get_fn: fn(B, Int) -> T):
        self.base = base.copy()
        self._len = len_fn
        self._get = get_fn

    fn __copyinit__(out self, other: Self):
        self.base = other.base.copy()
        self._len = other._len
        self._get = other._get

    fn __len__(self) -> Int:
        return self._len(self.base)

    fn __getitem__(self, idx: Int) -> T:
        return self._get(self.base, idx)

    # --- سازنده‌های راحت برای انواع رایج ---
    @staticmethod
    fn from_dataset(base: Dataset[T]) -> IndexAdapter[Dataset[T], T]:
        fn _l(ds: Dataset[T]) -> Int: return ds.__len__()
        fn _g(ds: Dataset[T], i: Int) -> T: return ds.__getitem__(i)
        return IndexAdapter[Dataset[T], T](base, _l, _g)

    @staticmethod
    fn from_image_folder(base: ImageFolderDataset) -> IndexAdapter[ImageFolderDataset, Pair]:
        fn _l2(ds: ImageFolderDataset) -> Int: return ds.__len__()
        fn _g2(ds: ImageFolderDataset, i: Int) -> Pair: return ds.__getitem__(i)
        return IndexAdapter[ImageFolderDataset, Pair](base, _l2, _g2)


struct IndexSliceDataset[
    T: Copyable & Movable,
    B: Copyable & Movable
](Copyable, Movable):
    var base: IndexAdapter[B, T]
    var start: Int
    var length: Int
    var mk_empty: fn() -> T

    fn __init__(out self, base: IndexAdapter[B, T], start: Int, length: Int, mk_empty: fn() -> T):
        self.base = base.copy()
        self.start = (start if start >= 0 else 0)
        self.length = (length if length >= 0 else 0)
        self.mk_empty = mk_empty

    fn __copyinit__(out self, other: Self):
        self.base = other.base.copy()
        self.start = other.start
        self.length = other.length
        self.mk_empty = other.mk_empty

    fn __len__(self) -> Int:
        return self.length

    fn __getitem__(self, idx: Int) -> T:
        var n = self.length
        if n == 0:
            return self.mk_empty()
        var i = idx
        if i < 0: i = 0
        if i >= n: i = n - 1
        return self.base.__getitem__(self.start + i)


struct IndexListDataset[
    T: Copyable & Movable,
    B: Copyable & Movable
](Copyable, Movable):
    var base: IndexAdapter[B, T]
    var indices: List[Int]
    var mk_empty: fn() -> T

    fn __init__(out self, base: IndexAdapter[B, T], indices: List[Int], mk_empty: fn() -> T):
        self.base = base.copy()
        self.indices = indices.copy()
        self.mk_empty = mk_empty

    fn __copyinit__(out self, other: Self):
        self.base = other.base
        self.indices = other.indices.copy()
        self.mk_empty = other.mk_empty

    fn __len__(self) -> Int:
        return len(self.indices)

    fn __getitem__(self, idx: Int) -> T:
        var n = len(self.indices)
        if n == 0:
            return self.mk_empty()
        var i = idx
        if i < 0: i = 0
        if i >= n: i = n - 1
        var j = self.indices[i]
        return self.base.__getitem__(j)


struct ZipDataset[
    A: Copyable & Movable,
    B: Copyable & Movable,
    BA: Copyable & Movable,  # base for A
    BB: Copyable & Movable   # base for B
](Copyable, Movable):
    var left:  IndexAdapter[BA, A]
    var right: IndexAdapter[BB, B]

    fn __init__(out self, left: IndexAdapter[BA, A], right: IndexAdapter[BB, B]):
        self.left = left
        self.right = right

    fn __copyinit__(out self, other: Self):
        self.left = other.left
        self.right = other.right

    fn __len__(self) -> Int:
        var nl = self.left.__len__()
        var nr = self.right.__len__()
        return (nl if nl < nr else nr)

    fn __getitem__(self, idx: Int) -> (A, B):
        var n = self.__len__()
        if n == 0:
            return (A(), B())
        var i = idx
        if i < 0: i = 0
        if i >= n: i = n - 1
        return (self.left.__getitem__(i), self.right.__getitem__(i))


struct EnumerateDataset[
    T: Copyable & Movable,
    B: Copyable & Movable
](Copyable, Movable):
    var base: IndexAdapter[B, T]

    fn __init__(out self, base: IndexAdapter[B, T]):
        self.base = base.copy()

    fn __copyinit__(out self, other: Self):
        self.base = other.base

    fn __len__(self) -> Int:
        return self.base.__len__()

    fn __getitem__(self, idx: Int) -> (Int, T):
        var n = self.base.__len__()
        if n == 0:
            return (0, self.base.__getitem__(0))  # returns default via clamping
        var i = idx
        if i < 0: i = 0
        if i >= n: i = n - 1
        return (i, self.base.__getitem__(i))


# -----------------------------------------------------------------------------
# Factory helpers
# -----------------------------------------------------------------------------

fn make_index_slice[
    T: Copyable & Movable,
    B: Copyable & Movable
](base: B, start: Int, length: Int, mk_empty: fn() -> T) -> IndexSliceDataset[T, B]:
    var a = IndexAdapter[B, T](base)
    return IndexSliceDataset[T, B](a, start, length, mk_empty)

fn make_index_list[
    T: Copyable & Movable,
    B: Copyable & Movable
](base: B, indices: List[Int], mk_empty: fn() -> T) -> IndexListDataset[T, B]:
    var a = IndexAdapter[B, T](base)
    return IndexListDataset[T, B](a, indices, mk_empty)

fn make_zip[
    A: Copyable & Movable,
    B: Copyable & Movable,
    BA: Copyable & Movable,
    BB: Copyable & Movable
](left: BA, right: BB) -> ZipDataset[A, B, BA, BB]:
    var la = IndexAdapter[BA, A](left)
    var rb = IndexAdapter[BB, B](right)
    return ZipDataset[A, B, BA, BB](la, rb)

fn make_enumerate[
    T: Copyable & Movable,
    B: Copyable & Movable
](base: B) -> EnumerateDataset[T, B]:
    var a = IndexAdapter[B, T](base)
    return EnumerateDataset[T, B](a)


# -----------------------------------------------------------------------------
# Index utilities (no RNG dependency for core; simple LCG here if needed)
# -----------------------------------------------------------------------------

fn clamp_index(i: Int, n: Int) -> Int:
    if n <= 0:
        return 0
    var j = i
    if j < 0: j = 0
    if j >= n: j = n - 1
    return j

fn make_arange(n: Int) -> List[Int]:
    var out = List[Int]()
    var i = 0
    while i < n:
        out.append(i)
        i = i + 1
    return out.copy()

# Simple LCG for reproducible shuffles when no RNG module is available.
struct LCG32(Copyable, Movable):
    var state: UInt32

    fn __init__(out self, seed: UInt64):
        self.state = UInt32(seed & 0xFFFFFFFF)

    fn next_u32(mut self) -> UInt32:
        self.state = self.state * 1664525 + 1013904223
        return self.state

    fn randint(mut self, lo: Int, hi: Int) -> Int:
        # [lo, hi)
        var span = (hi - lo)
        if span <= 0:
            return lo
        var r = Int(self.next_u32() & 0x7FFFFFFF)
        return lo + (r % span)

fn shuffle_inplace(mut idxs: List[Int], seed: UInt64):
    var n = len(idxs)
    var rng = LCG32(seed)
    var i = n - 1
    while i > 0:
        var j = rng.randint(0, i + 1)
        var tmp = idxs[i]
        idxs[i] = idxs[j]
        idxs[j] = tmp
        i = i - 1

fn make_random_index_partition(n: Int, train_ratio: Float32, seed: UInt64) -> (List[Int], List[Int]):
    # Returns (train_indices, val_indices)
    var idxs = make_arange(n)
    shuffle_inplace(mut idxs, seed)
    var k = Int(Float32(n) * train_ratio)
    if k < 0: k = 0
    if k > n: k = n
    var a = List[Int]()
    var b = List[Int]()
    var i = 0
    while i < n:
        if i < k:
            a.append(idxs[i])
        else:
            b.append(idxs[i])
        i = i + 1
    return (a, b)
