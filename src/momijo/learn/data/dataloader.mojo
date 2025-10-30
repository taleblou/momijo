# MIT License
# SPDX-License-Identifier: MIT
# Project:      Momijo
# Module:       learn.data.dataloader
# File:         src/momijo/learn/data/dataloader.mojo
#
# Description:
#   DataLoader with reproducible shuffling and batch iteration.
#   Backend-agnostic and minimal-API-compatible with Momijo standards:
#     - Options: batch_size, shuffle, drop_last, seed, collate_id
#     - Iterator pattern: _make_index_iter() + next_batch(mut it)
#     - No globals; var-only; English-only comments.
#
# Author(s):    Morteza Taleblou & Mitra Daneshmand
# Website:      https://taleblou.ir/
# Repository:   https://github.com/taleblou/momijo

from collections.list import List
from momijo.learn.data.dataset import Dataset

# -----------------------------------------------------------------------------
# Utilities: simple 32-bit LCG RNG for deterministic shuffling
#   - We perform the LCG update in 64-bit space and then mask to 32 bits
#     to avoid undefined overflow behavior.
# -----------------------------------------------------------------------------
struct RNG32:
    var state: UInt32

    fn __init__(out self, seed: UInt64):
        # Mix the 64-bit seed down to 32-bit (xorshift-like).
        var mixed = seed ^ (seed >> UInt64(32))
        var s32 = UInt32(mixed & UInt64(0xFFFFFFFF))
        if s32 == UInt32(0):
            s32 = UInt32(0x6D2B79F5)  # non-zero default
        self.state = s32

    fn next_u32(mut self) -> UInt32:
        # Numerical Recipes LCG: x = (a*x + c) mod 2^32
        # Do the math in 64-bit and then mask back to 32-bit.
        var x = UInt64(self.state)
        var a = UInt64(1664525)
        var c = UInt64(1013904223)
        x = (a * x + c) & UInt64(0xFFFFFFFF)
        self.state = UInt32(x)
        return self.state

    fn randint(mut self, low: Int, high_inclusive: Int) -> Int:
        # Return integer in [low, high_inclusive]
        var lo = low
        var hi = high_inclusive
        if hi < lo:
            # swap if caller passed inverted bounds
            var tmp = lo; lo = hi; hi = tmp
        var span = hi - lo + 1
        if span <= 1:
            return lo
        var r = Int(self.next_u32() % UInt32(span))
        return lo + r


# -----------------------------------------------------------------------------
# Options for DataLoader
# -----------------------------------------------------------------------------
struct DataLoaderOptions:
    var batch_size: Int
    var shuffle: Bool
    var drop_last: Bool
    var seed: UInt64
    var collate_id: Int  # 0: indices only (default); future IDs can map to strategies

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
# Iterator over batch indices
# -----------------------------------------------------------------------------
struct IndexIter:
    var order: List[Int]
    var pos: Int
    var batch_size: Int
    var drop_last: Bool

    fn __init__(out self, order: List[Int], batch_size: Int, drop_last: Bool):
        self.order = order
        self.pos = 0
        self.batch_size = batch_size
        self.drop_last = drop_last


# -----------------------------------------------------------------------------
# DataLoader: produces batches of indices (List[Int]) deterministically.
#   - __len__() returns number of batches for one epoch
#   - _make_index_iter() initializes per-epoch index order
#   - next_batch(mut it) advances iterator and returns List[Int] (may be empty)
# -----------------------------------------------------------------------------
struct DataLoader:
    var dataset: Dataset
    var opts: DataLoaderOptions
    var length: Int  # number of batches in one epoch

    fn __init__(
        out self,
        dataset: Dataset,
        batch_size: Int = 32,
        shuffle: Bool = True,
        drop_last: Bool = False,
        seed: UInt64 = 42,
        collate_id: Int = 0
    ):
        self.dataset = dataset
        self.opts = DataLoaderOptions(batch_size, shuffle, drop_last, seed, collate_id)
        self.length = _compute_num_batches(dataset.__len__(), batch_size, drop_last)

    fn with_options(out self, dataset: Dataset, opts: DataLoaderOptions):
        self.dataset = dataset
        self.opts = opts
        self.length = _compute_num_batches(dataset.__len__(), opts.batch_size, opts.drop_last)

    fn __len__(self) -> Int:
        return self.length

    # Create an index iterator for one full epoch.
    fn _make_index_iter(self) -> IndexIter:
        var n = self.dataset.__len__()
        var order = _arange(n)
        if self.opts.shuffle and n > 1:
            var rng = RNG32(self.opts.seed)
            _fisher_yates_shuffle(mut order, mut rng)
        return IndexIter(order, self.opts.batch_size, self.opts.drop_last)

    # Advance the iterator and return the next batch of indices.
    # Returns an empty List[Int] when iteration ends (or when drop_last prunes tail).
    fn next_batch(self, mut it: IndexIter) -> List[Int]:
        var out = List[Int]()
        var n = Int(it.order.size())
        if it.pos >= n:
            return out  # exhausted

        var remaining = n - it.pos
        if remaining < it.batch_size and it.drop_last:
            it.pos = n  # consume and end
            return out  # empty due to drop_last

        var take = it.batch_size
        if remaining < take:
            take = remaining

        var i = 0
        while i < take:
            out.push_back(it.order[it.pos + i])
            i = i + 1

        it.pos = it.pos + take
        return out

    # Usage pattern:
    # var it = dl._make_index_iter()
    # var batch = dl.next_batch(mut it)
    # while batch.size() > 0:
    #     # fetch items by indices in `batch` at a higher layer
    #     batch = dl.next_batch(mut it)


# -----------------------------------------------------------------------------
# Helpers (internal)
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
        xs.push_back(i)
        i = i + 1
    return xs

fn _fisher_yates_shuffle(mut xs: List[Int], mut rng: RNG32):
    var n = Int(xs.size())
    if n <= 1:
        return
    var i = n - 1
    while i > 0:
        var j = rng.randint(0, i)
        var tmp = xs[i]
        xs[i] = xs[j]
        xs[j] = tmp
        i = i - 1
