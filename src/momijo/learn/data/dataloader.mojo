# MIT License
# SPDX-License-Identifier: MIT
# Project: momijo
# File: src/momijo/learn/data/dataloader.mojo
# Description: Deterministic-shuffle DataLoader (index batches) for generic indexable bases.

from collections.list import List
from pathlib import Path

# Common sample type (X, y) for vision tasks
from momijo.learn.data.pair import Pair

# Core dataset views/adapters from Momijo Learn
from momijo.learn.data.dataset import IndexAdapter,shuffle_inplace
from momijo.learn.data.dataset import IndexSliceDataset,make_arange
from momijo.learn.data.dataset import make_index_slice

# Vision dataset and transforms (consumed here)
from momijo.learn.data.image import ImageFolderDataset
from momijo.vision.transforms.transforms import build_transforms

# -----------------------------------------------------------------------------
# RNG (LCG) for deterministic shuffling
# -----------------------------------------------------------------------------
struct RNG32(Copyable, Movable):
    var state: UInt32

    fn __init__(out self, seed: UInt64):
        var mixed = seed ^ (seed >> UInt64(32))
        var s32 = UInt32(mixed & UInt64(0xFFFFFFFF))
        if s32 == UInt32(0):
            s32 = UInt32(0x6D2B79F5)
        self.state = s32

    fn next_u32(mut self) -> UInt32:
        # LCG parameters (Numerical Recipes)
        self.state = self.state * 1664525 + 1013904223
        return self.state

    fn randint(mut self, low: Int, high_inclusive: Int) -> Int:
        var lo = low
        var hi = high_inclusive
        if hi < lo:
            var tmp = lo; lo = hi; hi = tmp
        var span = hi - lo + 1
        if span <= 1:
            return lo
        var r = Int(self.next_u32() & 0x7FFFFFFF)
        return lo + (r % span)

# -----------------------------------------------------------------------------
# Options
# -----------------------------------------------------------------------------
struct DataLoaderOptions(Copyable, Movable):
    var batch_size: Int
    var shuffle: Bool
    var drop_last: Bool
    var seed: UInt64
    var collate_id: Int
    var num_workers: Int

    fn __init__(
        out self,
        batch_size: Int = 32,
        shuffle: Bool = True,
        drop_last: Bool = False,
        seed: UInt64 = 42,
        collate_id: Int = 0,
        num_workers: Int = 0
    ):
        self.batch_size  = batch_size
        self.shuffle     = shuffle
        self.drop_last   = drop_last
        self.seed        = seed
        self.collate_id  = collate_id
        self.num_workers = num_workers

# -----------------------------------------------------------------------------
# Batch index iterator
# -----------------------------------------------------------------------------
struct IndexIter(Copyable, Movable):
    var order: List[Int]
    var pos: Int
    var batch_size: Int
    var drop_last: Bool

    fn __init__(out self, order: List[Int], batch_size: Int, drop_last: Bool):
        self.order = order.copy()
        self.pos = 0
        self.batch_size = (batch_size if batch_size > 0 else 1)
        self.drop_last = drop_last

    fn __copyinit__(out self, other: Self):
        self.order = other.order.copy()
        self.pos = other.pos
        self.batch_size = other.batch_size
        self.drop_last = other.drop_last

    fn remaining(self) -> Int:
        var n = len(self.order)
        if self.pos >= n: return 0
        return n - self.pos

# -----------------------------------------------------------------------------
# Iterator over DataLoader[B,T] that yields index lists
# -----------------------------------------------------------------------------
struct BatchIter[B: Copyable & Movable, T: Copyable & Movable](Copyable, Movable):
    var loader: DataLoader[B, T]
    var it: IndexIter

    fn __init__(out self, loader: DataLoader[B, T]):
        self.loader = loader.copy()
        self.it = self.loader._make_index_iter()

    fn __copyinit__(out self, other: Self):
        self.loader = other.loader.copy()
        self.it = other.it.copy()

    fn __iter__(mut self) -> Self:
        return self

    fn __next__(mut self) -> Optional[List[Int]]:
        var idxs = self.loader.next_batch(mut self.it)
        if len(idxs) == 0:
            return None
        return Some(idxs.copy())

# -----------------------------------------------------------------------------
# DataLoader[B,T]: yields List[Int] indices per batch
# B must provide: __len__() -> Int, __getitem__(idx: Int) -> T
# -----------------------------------------------------------------------------
struct DataLoader[B: Copyable & Movable, T: Copyable & Movable](Copyable, Movable):
    var base: IndexSliceDataset[T, B]
    var opts: DataLoaderOptions
    var length: Int   # number of batches in one epoch

    fn __init__(
        out self,
        base: IndexSliceDataset[T, B],
        batch_size: Int = 32,
        shuffle: Bool = True,
        drop_last: Bool = False,
        seed: UInt64 = 42,
        collate_id: Int = 0,
        num_workers: Int = 0
    ):
        self.base = base.copy()
        self.opts = DataLoaderOptions(batch_size, shuffle, drop_last, seed, collate_id, num_workers)
        self.length = _compute_num_batches(base.__len__(), self.opts.batch_size, self.opts.drop_last)

    fn __init__(out self, base: IndexSliceDataset[T, B], opts: DataLoaderOptions):
        self.base = base.copy()
        self.opts = opts.copy()
        self.length = _compute_num_batches(base.__len__(), self.opts.batch_size, self.opts.drop_last)

    fn __copyinit__(out self, other: Self):
        self.base   = other.base.copy()
        self.opts   = other.opts.copy()
        self.length = other.length

    fn __iter__(self) -> BatchIter[B, T]:
        return BatchIter[B, T](self.copy())

    fn __len__(self) -> Int:
        return self.length

    fn with_options(out self, base: IndexSliceDataset[T, B], opts: DataLoaderOptions):
        self.base = base.copy()
        self.opts = opts
        self.length = _compute_num_batches(base.__len__(), self.opts.batch_size, self.opts.drop_last)

    fn set_seed(mut self, seed: UInt64):
        self.opts.seed = seed

    fn set_shuffle(mut self, flag: Bool):
        self.opts.shuffle = flag

    fn set_batch_size(mut self, bs: Int):
        self.opts.batch_size = (bs if bs > 0 else 1)
        self.length = _compute_num_batches(self.base.__len__(), self.opts.batch_size, self.opts.drop_last)

    fn set_drop_last(mut self, flag: Bool):
        self.opts.drop_last = flag
        self.length = _compute_num_batches(self.base.__len__(), self.opts.batch_size, self.opts.drop_last)

    fn _make_index_iter(self) -> IndexIter:
        var n = self.base.__len__()
        var order = make_arange(n)                      # was: _arange
        if self.opts.shuffle and n > 1:
            shuffle_inplace(order, self.opts.seed)  # was: RNG32 + _fisher_yates_shuffle
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

    # Fetch items for given indices into a flat list (no collation)
    fn gather(self, idxs: List[Int]) -> List[T]:
        var batch = List[T]()
        var i = 0
        var m = len(idxs)
        while i < m:
            batch.append(self.base.__getitem__(idxs[i]))
            i = i + 1
        return batch.copy()

    # Iterate one epoch and apply a user-provided function `consume` to each batch of T.
    fn for_each_batch(self, consume: fn(List[T]) -> Void):
        var it = self._make_index_iter()
        while True:
            var idxs = self.next_batch(mut it)
            if len(idxs) == 0:
                break
            var batch = self.gather(idxs)
            consume(batch)

    # Fetch + collate via user-provided collate function
    fn gather_and_collate[U: Copyable & Movable](
        self,
        mut it: IndexIter,
        collate: fn(List[T]) -> U
    ) -> (Bool, U):
        var idxs = self.next_batch(mut it)
        if len(idxs) == 0:
            var empty = List[T]()
            return (False, collate(empty))
        var batch = self.gather(idxs)
        var out = collate(batch)
        return (True, out)




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

# -----------------------------------------------------------------------------
# Convenience: build loaders for an image-folder classification dataset
# -----------------------------------------------------------------------------
fn _mk_empty_pair() -> Pair:
    return Pair.empty()

fn make_dataloaders_from_png(
    root: String,
    img_size: Int = 28,
    batch_size: Int = 128,
    val_split: Float32 = 0.1,
    workers: Int = 0,
    augment: Bool = False
) -> (DataLoader[ImageFolderDataset, Pair],
      DataLoader[ImageFolderDataset, Pair],
      DataLoader[ImageFolderDataset, Pair],Int):

    var train_dir = root + String("/train")
    var test_dir  = root + String("/test")

    var train_tf = build_transforms(img_size, augment)
    var test_tf  = build_transforms(img_size, False)

    var train_full = ImageFolderDataset(train_dir, train_tf)
    var test_ds    = ImageFolderDataset(test_dir,  test_tf)
    var num_classes=train_full.num_classes()

    var total = train_full.__len__()
    var vlen  = Int(Float32(total) * val_split)
    if vlen < 0: vlen = 0
    if vlen > total: vlen = total
    var tlen  = total - vlen

    # Slice views (need mk_empty factory)
    var train_ds = IndexSliceDataset[Pair, ImageFolderDataset](
        IndexAdapter[ImageFolderDataset, Pair].from_image_folder(train_full),
        0, tlen, _mk_empty_pair
    )
    var val_ds   = IndexSliceDataset[Pair, ImageFolderDataset](
        IndexAdapter[ImageFolderDataset, Pair].from_image_folder(train_full),
        tlen, vlen, _mk_empty_pair
    )
    var test_view = IndexSliceDataset[Pair, ImageFolderDataset](
        IndexAdapter[ImageFolderDataset, Pair].from_image_folder(test_ds),
        0, test_ds.__len__(), _mk_empty_pair
    )

    var opt_tr = DataLoaderOptions(batch_size, True,  False, 42, 0, workers)
    var opt_va = DataLoaderOptions(batch_size, False, False, 42, 0, workers)
    var opt_te = DataLoaderOptions(batch_size, False, False, 42, 0, workers)

    var train_loader = DataLoader[ImageFolderDataset, Pair](train_ds, opt_tr)
    var val_loader   = DataLoader[ImageFolderDataset, Pair](val_ds,   opt_va)
    var test_loader  = DataLoader[ImageFolderDataset, Pair](test_view, opt_te)

    return (train_loader.copy(), val_loader.copy(), test_loader.copy(),num_classes)
