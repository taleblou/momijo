# MIT License
# SPDX-License-Identifier: MIT
# Project:      Momijo
# Module:       learn.data.pipeline
# File:         src/momijo/learn/data/pipeline.mojo
#
# Description:
#   Data pipeline utilities (tf.data-like) for batching, shuffling, prefetch hints,
#   and repetition over a Dataset. The pipeline yields batches of sample indices;
#   consumers fetch samples from the underlying Dataset to avoid generic element
#   type issues.
#
# Author(s):    Morteza Taleblou & Mitra Daneshmand
# Website:      https://taleblou.ir/
# Repository:   https://github.com/taleblou/momijo
#
# Notes:
#   - Types: Pipeline, PipelineIter
#   - Key fns: Pipeline.batch(), .shuffle(), .prefetch(), .repeat(), .build_iter()
#   - Returns index batches (List[Int]); consumer maps indices → samples.
#   - Deterministic pseudo-shuffle (no RNG dependency yet). Replace later with
#     real RNG-based Fisher–Yates once momijo.utils.randomness exposes a PRNG.

from collections.list import List
from momijo.learn.data.dataset import Dataset

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

# Build a simple 0..n-1 index list
fn _arange(n: Int) -> List[Int]:
    var idxs = List[Int]()
    var i = 0
    while i < n:
        idxs.append(i)
        i = i + 1
    return idxs

# Lightweight deterministic "pseudo-shuffle" that permutes indices by a step.
# This avoids RNG dependencies at this stage. Later, replace with a PRNG-based
# Fisher–Yates once randomness utilities are ready.
fn _pseudo_permute(indices: List[Int], buffer_hint: Int) -> List[Int]:
    var n = len(indices)
    if n <= 1:
        return indices

    # Choose a step that is not a trivial divisor of n; keep it simple.
    var step = (buffer_hint % n) + 1
    if step == 1:
        # Nudge step to avoid identity for common small n
        step = (n % 7) + 1
        if step == 1:
            step = 2

    var out = List[Int]()
    out.reserve(n)

    var i = 0
    while i < n:
        # i * step modulo n
        var j = (i * step) % n
        out.append(indices[j])
        i = i + 1
    return out

# -----------------------------------------------------------------------------
# Iterator yielding batches of indices according to the pipeline config
# -----------------------------------------------------------------------------

struct PipelineIter:
    var dataset: Dataset
    var order: List[Int]
    var pos: Int
    var batch_size: Int
    var drop_last: Bool
    var prefetch_n: Int
    var total_repeats: Int         # -1 means infinite
    var current_repeat: Int

    fn __init__(
        out self,
        dataset: Dataset,
        order: List[Int],
        batch_size: Int,
        drop_last: Bool,
        prefetch_n: Int,
        repeats: Int
    ):
        self.dataset = dataset
        self.order = order
        self.pos = 0
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.prefetch_n = prefetch_n
        self.total_repeats = repeats
        self.current_repeat = 0

    fn _has_more_epoch(self) -> Bool:
        if self.total_repeats < 0:
            return True
        return self.current_repeat < self.total_repeats

    fn _advance_epoch(mut self):
        self.current_repeat = self.current_repeat + 1
        self.pos = 0

    # Return whether any further batch can be produced.
    fn has_next(self) -> Bool:
        if self.pos < len(self.order):
            return True
        # Current epoch consumed; check if we have more repeats.
        if self._has_more_epoch():
            return True
        return False

    # Produce the next batch of indices. If no more data, returns an empty list.
    fn next_index_batch(mut self) -> List[Int]:
        var empty = List[Int]()
        var n = len(self.order)

        # If current epoch ended, move to next (if allowed)
        if self.pos >= n:
            if not self._has_more_epoch():
                return empty
            self._advance_epoch()

        # Compute slice bounds
        var start = self.pos
        var end = start + self.batch_size
        if end > n:
            if self.drop_last:
                # Skip incomplete tail
                self._advance_epoch()
                return self.next_index_batch()
            end = n

        # Slice [start, end)
        var batch = List[Int]()
        var i = start
        while i < end:
            batch.append(self.order[i])
            i = i + 1

        self.pos = end
        return batch

    fn __str__(self) -> String:
        var s = "PipelineIter(pos="
        s = s + String(self.pos)
        s = s + ", batch_size=" + String(self.batch_size)
        s = s + ", drop_last=" + String(self.drop_last)
        s = s + ", prefetch_n=" + String(self.prefetch_n)
        s = s + ", repeat " + String(self.current_repeat) + "/" + String(self.total_repeats)
        s = s + ", len=" + String(len(self.order)) + ")"
        return s

# -----------------------------------------------------------------------------
# Public pipeline that configures iteration and yields index batches
# -----------------------------------------------------------------------------

struct Pipeline:
    var dataset: Dataset
    var _batch_size: Int
    var _drop_last: Bool
    var _shuffle_buf: Int
    var _prefetch_n: Int
    var _repeats: Int                # -1 for infinite, else number of epochs

    fn __init__(out self, dataset: Dataset):
        self.dataset = dataset
        self._batch_size = 1
        self._drop_last = False
        self._shuffle_buf = 0
        self._prefetch_n = 0
        self._repeats = 1

    # Set the batch size and drop_last policy.
    fn batch(mut self, size: Int, drop_last: Bool = False) -> Pipeline:
        if size < 1:
            self._batch_size = 1
        else:
            self._batch_size = size
        self._drop_last = drop_last
        return self

    # Enable deterministic pseudo-shuffle. 'buffer' is a hint that affects the permutation.
    fn shuffle(mut self, buffer: Int) -> Pipeline:
        if buffer < 0:
            self._shuffle_buf = 0
        else:
            self._shuffle_buf = buffer
        return self

    # Prefetch hint (not enforced yet; kept for future async prefetch workers).
    fn prefetch(mut self, n: Int) -> Pipeline:
        if n < 0:
            self._prefetch_n = 0
        else:
            self._prefetch_n = n
        return self

    # Repeat the dataset for 'count' epochs; use -1 for infinite.
    fn repeat(mut self, count: Int) -> Pipeline:
        self._repeats = count
        return self

    # Build an iterator over index batches according to the pipeline configuration.
    fn build_iter(self) -> PipelineIter:
        var length = self.dataset.__len__()
        var order = _arange(length)
        if self._shuffle_buf > 0 and length > 1:
            order = _pseudo_permute(order, self._shuffle_buf)
        return PipelineIter(
            self.dataset,
            order,
            self._batch_size,
            self._drop_last,
            self._prefetch_n,
            self._repeats
        )

    fn __str__(self) -> String:
        var s = "Pipeline(batch_size="
        s = s + String(self._batch_size)
        s = s + ", drop_last=" + String(self._drop_last)
        s = s + ", shuffle_buf=" + String(self._shuffle_buf)
        s = s + ", prefetch_n=" + String(self._prefetch_n)
        s = s + ", repeats=" + String(self._repeats) + ")"
        return s
