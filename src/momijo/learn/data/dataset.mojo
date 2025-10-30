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
#
# Author(s):    Morteza Taleblou & Mitra Daneshmand
# Website:      https://taleblou.ir/
# Repository:   https://github.com/taleblou/momijo
#
# Notes:
#   - Traits: T, A, B must be Copyable & Movable for safe value semantics.
#   - Assertions are used for bound checks; replace with exceptions where needed.
#   - Iterator usage pattern:
#       var it = ids._make_index_iter()
#       while True:
#           var opt = ids.next_item(mut it)
#           if opt is None: break
#           var item = opt.value()
#   - No globals; var-only; constructors via `fn __init__(out self, ...)`.
#   - This module is backend-agnostic and does not depend on tensor backends.

from collections.list import List

# -----------------------------------------------------------------------------
# Generic indexed dataset
# -----------------------------------------------------------------------------

struct Dataset[T: Copyable & Movable]:
    var items: List[T]

    fn __init__(out self):
        self.items = List[T]()

    @staticmethod
    fn from_list(items: List[T]) -> Dataset[T]:
        var ds = Dataset[T]()
        ds.items = items
        return ds

    fn with_items(out self, items: List[T]):
        # Construct from an existing list (ownership transfer by value).
        self.items = items

    fn __len__(self) -> Int:
        return Int(self.items.size())

    fn is_empty(self) -> Bool:
        return Int(self.items.size()) == 0

    fn __getitem__(self, idx: Int) -> T:
        # Bound-checked indexing.
        assert(idx >= 0)
        var n = Int(self.items.size())
        assert(idx < n)
        return self.items[idx]

    fn get(self, idx: Int, default: T) -> T:
        # Safe accessor with same-type default.
        if idx < 0:
            return default
        var n = Int(self.items.size())
        if idx >= n:
            return default
        return self.items[idx]

    fn set(mut self, idx: Int, value: T):
        # Bound-checked mutation.
        assert(idx >= 0)
        var n = Int(self.items.size())
        assert(idx < n)
        self.items[idx] = value

    fn append(mut self, value: T):
        self.items.push_back(value)

    fn extend(mut self, values: List[T]):
        # Append all items from values.
        var i = 0
        var n = Int(values.size())
        while i < n:
            self.items.push_back(values[i])
            i = i + 1

    fn insert(mut self, idx: Int, value: T):
        # Insert by shifting tail to the right (O(n)); idx is clamped to [0..len].
        var n = Int(self.items.size())
        var pos = idx
        if pos < 0:
            pos = 0
        if pos > n:
            pos = n
        # Append a duplicate of the last or value itself to grow, then shift.
        if n == 0:
            self.items.push_back(value)
            return
        var last = self.items[n - 1]
        self.items.push_back(last)
        var j = n - 1
        while j > pos:
            self.items[j] = self.items[j - 1]
            j = j - 1
        self.items[pos] = value

    fn pop(mut self) -> T:
        # Remove and return the last element; asserts non-empty.
        var n = Int(self.items.size())
        assert(n > 0)
        var v = self.items[n - 1]
        # Rebuild without the last item (List lacks pop_back in some builds).
        var out = List[T]()
        var i = 0
        while i < n - 1:
            out.push_back(self.items[i])
            i = i + 1
        self.items = out
        return v

    fn clear(mut self):
        # Drop all elements (list is reusable).
        self.items = List[T]()

    fn to_list(self) -> List[T]:
        # Return a shallow copy of items.
        var out_list = List[T]()
        var i = 0
        var n = Int(self.items.size())
        while i < n:
            out_list.push_back(self.items[i])
            i = i + 1
        return out_list

    fn slice(self, start: Int, stop: Int) -> Dataset[T]:
        # Safe slice with clamping; stop is exclusive.
        var n = Int(self.items.size())
        var s = start
        var e = stop
        if s < 0:
            s = 0
        if e < s:
            e = s
        if e > n:
            e = n
        var out_items = List[T]()
        var i = s
        while i < e:
            out_items.push_back(self.items[i])
            i = i + 1
        return Dataset[T].from_list(out_items)

    fn map[U: Copyable & Movable](self, f) -> Dataset[U]:
        # Apply a pure mapping function T -> U.
        var out_items = List[U]()
        var i = 0
        var n = Int(self.items.size())
        while i < n:
            out_items.push_back(f(self.items[i]))
            i = i + 1
        return Dataset[U].from_list(out_items)

    fn filter(self, pred) -> Dataset[T]:
        # Keep items where pred(T) -> Bool.
        var out_items = List[T]()
        var i = 0
        var n = Int(self.items.size())
        while i < n:
            var x = self.items[i]
            if pred(x):
                out_items.push_back(x)
            i = i + 1
        return Dataset[T].from_list(out_items)

    fn concat(self, other: Dataset[T]) -> Dataset[T]:
        # Concatenate self and other into a new dataset.
        var out_items = List[T]()
        var n0 = Int(self.items.size())
        var i = 0
        while i < n0:
            out_items.push_back(self.items[i])
            i = i + 1
        var n1 = Int(other.items.size())
        var j = 0
        while j < n1:
            out_items.push_back(other.items[j])
            j = j + 1
        return Dataset[T].from_list(out_items)


# -----------------------------------------------------------------------------
# Simple tuple/pair dataset for supervised learning (X, y)
# -----------------------------------------------------------------------------

struct TupleDataset[A: Copyable & Movable, B: Copyable & Movable]:
    var x: List[A]
    var y: List[B]

    fn __init__(out self):
        self.x = List[A]()
        self.y = List[B]()

    @staticmethod
    fn from_lists(x: List[A], y: List[B]) -> TupleDataset[A, B]:
        assert(Int(x.size()) == Int(y.size()))
        var ds = TupleDataset[A, B]()
        ds.x = x
        ds.y = y
        return ds

    fn with_lists(out self, x: List[A], y: List[B]):
        assert(Int(x.size()) == Int(y.size()))
        self.x = x
        self.y = y

    fn __len__(self) -> Int:
        return Int(self.x.size())

    fn is_empty(self) -> Bool:
        return Int(self.x.size()) == 0

    fn __getitem__(self, idx: Int) -> (A, B):
        # Bound-checked pair access.
        assert(idx >= 0)
        var n = Int(self.x.size())
        assert(idx < n)
        return (self.x[idx], self.y[idx])

    fn get(self, idx: Int, default_a: A, default_b: B) -> (A, B):
        # Safe accessor with defaults.
        if idx < 0:
            return (default_a, default_b)
        var n = Int(self.x.size())
        if idx >= n:
            return (default_a, default_b)
        return (self.x[idx], self.y[idx])

    fn append(mut self, a: A, b: B):
        self.x.push_back(a)
        self.y.push_back(b)

    fn to_lists(self) -> (List[A], List[B]):
        # Shallow copies.
        var cx = List[A]()
        var cy = List[B]()
        var i = 0
        var n = Int(self.x.size())
        while i < n:
            cx.push_back(self.x[i])
            cy.push_back(self.y[i])
            i = i + 1
        return (cx, cy)

    fn to_dataset(self) -> Dataset[(A, B)]:
        # Materialize as a Dataset of tuples (A,B).
        var out = List[(A, B)]()
        var n = Int(self.x.size())
        var i = 0
        while i < n:
            out.push_back((self.x[i], self.y[i]))
            i = i + 1
        return Dataset[(A, B)].from_list(out)


# -----------------------------------------------------------------------------
# Iterable wrapper and index iterator
# -----------------------------------------------------------------------------

struct IndexIter:
    var i: Int
    var n: Int

    fn __init__(out self, n: Int):
        self.i = 0
        self.n = n

    fn reset(mut self):
        self.i = 0

    fn done(self) -> Bool:
        return self.i >= self.n


# Iteration over any Dataset[T] by index. This does not shuffle; shuffling and
# sampling strategies belong to Sampler and DataLoader layers.
struct IterableDataset[T: Copyable & Movable]:
    var data: Dataset[T]

    fn __init__(out self):
        self.data = Dataset[T]()

    fn with_dataset(out self, data: Dataset[T]):
        self.data = data

    fn __len__(self) -> Int:
        return self.data.__len__()

    # Create a fresh index iterator covering [0, len).
    fn _make_index_iter(self) -> IndexIter:
        return IndexIter(self.__len__())

    # Return next item if available, else Optional None.
    # Caller pattern:
    #   var it = ds._make_index_iter()
    #   while True:
    #       var opt = ds.next_item(mut it)
    #       if opt is None: break
    #       var item = opt.value()
    fn next_item(self, mut it: IndexIter) -> Optional[T]:
        if it.done():
            return None
        var idx = it.i
        it.i = it.i + 1
        return Optional[T](self.data.__getitem__(idx))
