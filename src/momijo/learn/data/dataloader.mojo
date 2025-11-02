# MIT License
# SPDX-License-Identifier: MIT
# Project: momijo
# File: src/momijo/learn/data/dataloader.mojo
# Description: Deterministic-shuffle DataLoader (index batches) for Dataset[T].

from collections.list import List
from momijo.learn.data.dataset import Dataset  # generic Dataset[T]

# -----------------------------------------------------------------------------
# RNG (LCG) for deterministic shuffling
# -----------------------------------------------------------------------------
struct RNG32:
    var state: UInt32

    fn __init__(out self, seed: UInt64):
        var mixed = seed ^ (seed >> UInt64(32))
        var s32 = UInt32(mixed & UInt64(0xFFFFFFFF))
        if s32 == UInt32(0):
            s32 = UInt32(0x6D2B79F5)
        self.state = s32

    fn next_u32(mut self) -> UInt32:
        var x = UInt64(self.state)
        var a = UInt64(1664525)
        var c = UInt64(1013904223)
        x = (a * x + c) & UInt64(0xFFFFFFFF)
        self.state = UInt32(x)
        return self.state

    fn randint(mut self, low: Int, high_inclusive: Int) -> Int:
        var lo = low
        var hi = high_inclusive
        if hi < lo:
            var tmp = lo; lo = hi; hi = tmp
        var span = hi - lo + 1
        if span <= 1:
            return lo
        var r = Int(self.next_u32() % UInt32(span))
        return lo + r

# -----------------------------------------------------------------------------
# Options
# -----------------------------------------------------------------------------
struct DataLoaderOptions:
    var batch_size: Int
    var shuffle: Bool
    var drop_last: Bool
    var seed: UInt64
    var collate_id: Int  # 0: indices only

    fn __init__(
        out self,
        batch_size: Int = 32,
        shuffle: Bool = True,
        drop_last: Bool = False,
        seed: UInt64 = 42,
        collate_id: Int = 0
    ):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.collate_id = collate_id

# -----------------------------------------------------------------------------
# Batch index iterator
# -----------------------------------------------------------------------------
struct IndexIter:
    var order: List[Int]
    var pos: Int
    var batch_size: Int
    var drop_last: Bool

    fn __init__(out self, order: List[Int], batch_size: Int, drop_last: Bool):
        self.order = order.copy()
        self.pos = 0
        self.batch_size = batch_size
        self.drop_last = drop_last

# -----------------------------------------------------------------------------
# DataLoader[T]: yields List[Int] indices per batch
# -----------------------------------------------------------------------------
struct DataLoader[T: Copyable & Movable]:
    var dataset: Dataset[T]
    var opts: DataLoaderOptions
    var length: Int   # number of batches in one epoch

    fn __init__(
        out self,
        dataset: Dataset[T],
        batch_size: Int = 32,
        shuffle: Bool = True,
        drop_last: Bool = False,
        seed: UInt64 = 42,
        collate_id: Int = 0
    ):
        self.dataset = dataset.copy()
        self.opts = DataLoaderOptions(batch_size, shuffle, drop_last, seed, collate_id)
        self.length = _compute_num_batches(dataset.__len__(), batch_size, drop_last)

    fn with_options(out self, dataset: Dataset[T], opts: DataLoaderOptions):
        self.dataset = dataset
        self.opts = opts
        self.length = _compute_num_batches(dataset.__len__(), opts.batch_size, opts.drop_last)

    fn __len__(self) -> Int:
        return self.length

    fn _make_index_iter(self) -> IndexIter:
        var n = self.dataset.__len__()
        var order = _arange(n)
        if self.opts.shuffle and n > 1:
            var rng = RNG32(self.opts.seed)
            _fisher_yates_shuffle( order, rng)
        return IndexIter(order, self.opts.batch_size, self.opts.drop_last)

    fn next_batch(self, mut it: IndexIter) -> List[Int]:
        var out = List[Int]()
        var n = len(it.order)
        if it.pos >= n:
            return out.copy()

        var remaining = n - it.pos
        if remaining < it.batch_size and it.drop_last:
            it.pos = n
            return out.copy()

        var take = it.batch_size
        if remaining < take:
            take = remaining

        var i = 0
        while i < take:
            out.append(it.order[it.pos + i])
            i = i + 1

        it.pos = it.pos + take
        return out.copy()

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
fn _compute_num_batches(n_items: Int, batch_size: Int, drop_last: Bool) -> Int:
    if batch_size <= 0:
        return 0
    var full = n_items // batch_size
    var rem = n_items - full * batch_size
    if rem != 0 and not drop_last:
        return full + 1
    return full

fn _arange(n: Int) -> List[Int]:
    var xs = List[Int]()
    var i = 0
    while i < n:
        xs.append(i)
        i = i + 1
    return xs.copy()

fn _fisher_yates_shuffle(mut xs: List[Int], mut rng: RNG32):
    var n = len(xs)
    if n <= 1:
        return
    var i = n - 1
    while i > 0:
        var j = rng.randint(0, i)
        var tmp = xs[i]
        xs[i] = xs[j]
        xs[j] = tmp
        i = i - 1



fn gather_and_collate[U: Copyable & Movable](
    self,
    mut it: IndexIter,
    fetch: fn(idx: Int) -> T,
    collate: fn(List[T]) -> U
) -> (Bool, U):
    var idxs = self.next_batch(mut it)   # List[Int]
    if len(idxs) == 0:
        # No more batches; fabricate an empty batch via collate on an empty list
        var empty = List[T]()
        return (False, collate(empty))

    var batch = List[T]()
    var i = 0
    var m = len(idxs)
    while i < m:
        batch.append(fetch(idxs[i]))
        i = i + 1

    var out = collate(batch)
    return (True, out)