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
# Notes:
#   - Types T, A, B must be Copyable & Movable for safe value semantics.
#   - Assertions are used for bound checks; replace with exceptions where needed.
#   - Iterator usage pattern:
#       var it = ids._make_index_iter()
#       while True:
#           var opt = ids.next_item(mut it)
#           if opt is None: break
#           var item = opt.value()
#   - No globals; var-only; constructors via `fn __init__(out self, ...)`.
#   - Backend-agnostic (no tensor dependencies).

from collections.list import List

# -----------------------------------------------------------------------------
# Generic indexed dataset
# -----------------------------------------------------------------------------

struct Dataset[T: Copyable & Movable](Copyable, Movable):
    var items: List[T]

    fn __init__(out self):
        self.items = List[T]()

    fn __copyinit__(out self, other: Self):
        # Shallow copy of the underlying List[T] (value-semantics).
        self.items = other.items.copy()

    @staticmethod
    fn from_list(items: List[T]) -> Dataset[T]:
        var ds = Dataset[T]()
        ds.items = items.copy()
        return ds.copy()

    fn with_items(out self, items: List[T]):
        # Construct from an existing list (ownership transfer by value).
        self.items = items

    fn __len__(self) -> Int:
        return len(self.items)

    fn is_empty(self) -> Bool:
        return len(self.items) == 0

        
    fn __getitem__(self, idx: Int) -> Optional[T]:
        var n = len(self.items)
        if n == 0:
            return None
        var i = idx
        if i < 0: i = 0
        if i >= n: i = n - 1
        return self.items[i].copy()

    fn get(self, idx: Int, default: T) -> T:
        # Safe accessor with same-type default.
        if idx < 0:
            return default
        var n = len(self.items)
        if idx >= n:
            return default
        return self.items[idx]

    fn set(mut self, idx: Int, value: T):
        # Bound-checked mutation without assert: ignore if out-of-range.
        var n = len(self.items)
        if n == 0:
            return
        var i = idx
        if i < 0 or i >= n:
            # Clamp-to-range alternative:
            # if i < 0: i = 0
            # if i >= n: i = n - 1
            return
        self.items[i] = value

    fn append(mut self, value: T):
        self.items.append(value.copy())

    fn extend(mut self, values: List[T]):
        # Append all items from values.
        var i = 0
        var n = len(values)
        while i < n:
            self.items.append(values[i])
            i = i + 1

    fn insert(mut self, idx: Int, value: T):
        # Insert by shifting tail to the right (O(n)); idx is clamped to [0..len].
        var n = len(self.items)
        var pos = idx
        if pos < 0:
            pos = 0
        if pos > n:
            pos = n
        # Append a duplicate of the last or value itself to grow, then shift.
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
        # Remove and return the last element; safe without assert.
        var n = len(self.items)
        if n == 0:
            # Fallback: return default T()
            var zero = T()
            return zero
        var v = self.items[n - 1]
        # Rebuild without the last item (List may lack pop_back).
        var out = List[T]()
        var i = 0
        while i < n - 1:
            out.append(self.items[i])
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
        var n = len(self.items)
        while i < n:
            out_list.append(self.items[i])
            i = i + 1
        return out_list

    fn slice(self, start: Int, stop: Int) -> Dataset[T]:
        # Safe slice with clamping; stop is exclusive.
        var n = len(self.items)
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
            out_items.append(self.items[i])
            i = i + 1
        return Dataset[T].from_list(out_items)

    fn map[U: Copyable & Movable](self, f) -> Dataset[U]:
        # Apply a pure mapping function T -> U.
        var out_items = List[U]()
        var i = 0
        var n = len(self.items)
        while i < n:
            out_items.append(f(self.items[i]))
            i = i + 1
        return Dataset[U].from_list(out_items)

    fn filter(self, pred) -> Dataset[T]:
        # Keep items where pred(T) -> Bool.
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
        # Concatenate self and other into a new dataset.
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
        # if aligned lengths without assert: trim to min length
        var nx = len(x)
        var ny = len(y)
        var m = nx if nx < ny else ny

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
        # Align by trimming to min length
        var nx = len(x)
        var ny = len(y)
        var m = nx if nx < ny else ny

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
        # Safe, bound-checked access without assert.
        var n = len(self.x)
        if n == 0:
            # Fallback if empty: default-construct a pair
            var a0 = A()
            var b0 = B()
            return (a0, b0)

        var i = idx
        if i < 0: i = 0
        if i >= n: i = n - 1
        return (self.x[i], self.y[i])

    fn get(self, idx: Int, default_a: A, default_b: B) -> (A, B):
        # Safe accessor with defaults.
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
        # Shallow copies.
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
        # Materialize as a Dataset of tuples (A,B).
        var out = List[(A, B)]()
        var n = len(self.x)
        var i = 0
        while i < n:
            out.append((self.x[i], self.y[i]))
            i = i + 1
        return Dataset[(A, B)].from_list(out)


# -----------------------------------------------------------------------------
# Iterable wrapper and index iterator
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

# Iteration over any Dataset[T] by index. This does not shuffle; shuffling and
# sampling strategies belong to Sampler and DataLoader layers.
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
