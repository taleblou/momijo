# MIT License
# SPDX-License-Identifier: MIT
# Project:      Momijo
# Module:       learn.data.collate
# File:         src/momijo/learn/data/collate.mojo
#
# Description:
#   Common collate helpers for variable-length sequences.
#   - pad_int_sequences: pad List[List[Int]] to a rectangular 2D Tensor[Int].
#   - stack_int_labels: convert List[Int] labels to a 1D Tensor[Int].
#
# Notes:
#   - English-only comments (project rule).
#   - No globals; var-only; strict constructors.

from collections.list import List
from momijo.tensor import tensor

fn pad_int_sequences(
    seqs: List[List[Int]],
    pad_value: Int = 0
) -> tensor.Tensor[Int]:
    var n = len(seqs)
    var maxlen = 0
    var i = 0
    while i < n:
        var li = len(seqs[i])
        if li > maxlen: maxlen = li
        i = i + 1

    var xs = List[List[Int]]()
    i = 0
    while i < n:
        var row = List[Int]()
        var j = 0
        var li = len(seqs[i])
        while j < li:
            row.append(seqs[i][j])
            j = j + 1
        while j < maxlen:
            row.append(pad_value)
            j = j + 1
        xs.append(row.copy())
        i = i + 1

    return tensor.Tensor(xs)   # shape [n, maxlen], Int

fn stack_int_labels(labels: List[Int]) -> tensor.Tensor[Int]:
    return tensor.Tensor(labels)  # shape [n]



fn collate_pairs_varlen(batch: List[Pair]) -> (tensor.Tensor[Int], tensor.Tensor[Int]):
    var seqs = List[List[Int]]()
    var labs  = List[Int]()
    var i = 0
    var n = len(batch)
    while i < n:
        seqs.append(batch[i].x)
        labs.append(batch[i].y)
        i = i + 1
    var xb = pad_int_sequences(seqs, 0)
    var yb = stack_int_labels(labs)
    return (xb.copy(), yb.copy())



# Generic collate for variable-length sequences using extractors.
fn collate_varlen_by[T: Copyable & Movable](
    batch: List[T],
    get_seq: fn(T) -> List[Int],
    get_label: fn(T) -> Int
) -> (tensor.Tensor[Int], tensor.Tensor[Int]):
    var n = len(batch)
    var seqs = List[List[Int]]()
    var labs = List[Int]()
    var i = 0
    while i < n:
        seqs.append(get_seq(batch[i]))
        labs.append(get_label(batch[i]))
        i = i + 1
    var xb = pad_int_sequences(seqs, 0)   # [B, maxlen]
    var yb = stack_int_labels(labs)       # [B]
    return (xb.copy(), yb.copy())