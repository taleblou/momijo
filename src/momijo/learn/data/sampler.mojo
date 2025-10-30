# MIT License
# SPDX-License-Identifier: MIT
# Project:      Momijo
# Module:       learn.data.sampler
# File:         src/momijo/learn/data/sampler.mojo
#
# Description:
#   Data samplers for epoch-wise index generation.
#   - RandomSampler: produces a per-epoch permutation of [0..size-1] with
#     deterministic shuffling from (seed + epoch).
#   - DistributedSampler: partitions the global shuffled index set across
#     (world_size, rank) with optional drop_last behavior.
#   Backend-agnostic and returning plain index lists (List[Int]).
#
# Author(s):    Morteza Taleblou & Mitra Daneshmand
# Website:      https://taleblou.ir/
# Repository:   https://github.com/taleblou/momijo
#
# Notes:
#   - Types: RandomSampler, DistributedSampler, _LcgRng
#   - Key fns: epoch_indices(epoch), set_seed(seed), set_shuffle(flag)
#   - Deterministic shuffle via a simple LCG RNG (no external deps)

from collections.list import List

# -----------------------------------------------------------------------------
# Minimal deterministic RNG (LCG) for shuffling (backend-agnostic)
# -----------------------------------------------------------------------------
struct _LcgRng:
    var state: Int

    fn __init__(out self, seed: Int):
        var s = seed
        if s <= 0:
            s = 1  # avoid zero lock
        self.state = s

    fn next_u32(mut self) -> Int:
        # LCG with Numerical Recipes constants: a=1664525, c=1013904223, m=2^32
        self.state = (self.state * 1664525 + 1013904223)
        # mask to 32-bit unsigned range
        var v = self.state & 0xFFFFFFFF
        return v

    fn randrange(mut self, n: Int) -> Int:
        # returns x in [0, n) assuming n > 0
        if n <= 0:
            return 0
        var x = self.next_u32()
        return x % n


# -----------------------------------------------------------------------------
# RandomSampler
# -----------------------------------------------------------------------------
struct RandomSampler:
    var size: Int
    var shuffle: Bool
    var seed: Int

    fn __init__(out self, size: Int, shuffle: Bool = True, seed: Int = 42):
        assert(size >= 0)
        self.size = size
        self.shuffle = shuffle
        self.seed = seed

    fn set_seed(mut self, seed: Int):
        self.seed = seed

    fn set_shuffle(mut self, shuffle: Bool):
        self.shuffle = shuffle

    # Create a fresh epoch index order. Deterministic w.r.t. (seed + epoch).
    fn epoch_indices(self, epoch: Int = 0) -> List[Int]:
        var n = self.size
        var idx = List[Int]()
        var i = 0
        while i < n:
            idx.append(i)
            i = i + 1

        if self.shuffle and n > 1:
            var rng = _LcgRng(self.seed + epoch + 1)
            # Fisher–Yates
            var j = n - 1
            while j > 0:
                var k = rng.randrange(j + 1)  # 0..j
                # swap idx[j], idx[k]
                var tmp = idx[j]
                idx[j] = idx[k]
                idx[k] = tmp
                j = j - 1
        return idx


# -----------------------------------------------------------------------------
# DistributedSampler
# -----------------------------------------------------------------------------
# Partitions the (optionally shuffled) global index set across replicas.
# Mirrors PyTorch-like logic: pad when drop_last == False to make all replicas equal length.
struct DistributedSampler:
    var size: Int
    var world_size: Int
    var rank: Int
    var shuffle: Bool
    var drop_last: Bool
    var seed: Int

    fn __init__(
        out self,
        size: Int,
        world_size: Int,
        rank: Int,
        shuffle: Bool = True,
        drop_last: Bool = False,
        seed: Int = 42
    ):
        assert(size >= 0)
        assert(world_size > 0)
        assert(rank >= 0 and rank < world_size)
        self.size = size
        self.world_size = world_size
        self.rank = rank
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed

    fn set_seed(mut self, seed: Int):
        self.seed = seed

    fn set_shuffle(mut self, shuffle: Bool):
        self.shuffle = shuffle

    fn set_rank(mut self, rank: Int):
        assert(rank >= 0 and rank < self.world_size)
        self.rank = rank

    fn set_world_size(mut self, world_size: Int):
        assert(world_size > 0)
        self.world_size = world_size
        # caller should ensure rank stays valid

    # Return the indices assigned to this replica for a given epoch.
    fn epoch_indices(self, epoch: Int = 0) -> List[Int]:
        var n = self.size

        # Build base [0..n-1]
        var base = List[Int]()
        var i = 0
        while i < n:
            base.append(i)
            i = i + 1

        # Optional shuffle (deterministic per epoch)
        if self.shuffle and n > 1:
            var rng = _LcgRng(self.seed + epoch + 1)
            var j = n - 1
            while j > 0:
                var k = rng.randrange(j + 1)
                var tmp = base[j]
                base[j] = base[k]
                base[k] = tmp
                j = j - 1

        var total_size = n
        var num_samples = 0

        if self.drop_last:
            # exact partition (tail dropped)
            num_samples = n // self.world_size
            total_size = num_samples * self.world_size
            # truncate base if needed
            while len(base) > total_size:
                # pop from end
                base.pop()
        else:
            # pad to make divisible by world_size (repeat from start)
            var rem = 0
            if self.world_size != 0:
                rem = n % self.world_size
            if rem != 0:
                var pad = self.world_size - rem
                var p = 0
                while p < pad:
                    if n == 0:
                        # nothing to pad from; keep list empty
                        break
                    base.append(base[p % n])
                    p = p + 1
                total_size = len(base)
            else:
                total_size = n
            num_samples = 0
            if self.world_size != 0:
                num_samples = total_size // self.world_size

        # Slice for this rank: take every world_size-th element starting at rank
        var out = List[Int]()
        var taken = 0
        var pos = self.rank
        while taken < num_samples:
            out.append(base[pos])
            pos = pos + self.world_size
            taken = taken + 1

        return out

# -----------------------------------------------------------------------------
# (Optional) Lightweight self-check helpers for development.
# Keep these as utilities; examples/tests should live outside this module.
# -----------------------------------------------------------------------------
fn _assert_dist_shapes_equal(world_size: Int, size: Int, drop_last: Bool) -> Bool:
    var ok = True
    var e = 0
    while e < 2:
        var r = 0
        var lens = List[Int]()
        while r < world_size:
            var ds = DistributedSampler(size, world_size, r, True, drop_last, 123)
            var li = ds.epoch_indices(e)
            lens.append(len(li))
            r = r + 1
        # all lens equal
        var j = 1
        while j < len(lens):
            if lens[j] != lens[0]:
                ok = False
            j = j + 1
        e = e + 1
    return ok
