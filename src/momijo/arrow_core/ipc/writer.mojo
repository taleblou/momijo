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
# Project: momijo.arrow_core.ipc
# File: src/momijo/arrow_core/ipc/writer.mojo

fn argmax_index(xs: List[Float64]) -> Int:
    if len(xs) == 0:
        return -1
    var best = xs[0]
    var idx = 0
    var i = 1
    while i < len(xs):
        if xs[i] > best:
            best = xs[i]
            idx = i
        i += 1
    return idx
fn argmin_index(xs: List[Float64]) -> Int:
    if len(xs) == 0:
        return -1
    var best = xs[0]
    var idx = 0
    var i = 1
    while i < len(xs):
        if xs[i] < best:
            best = xs[i]
            idx = i
        i += 1
    return idx

fn ensure_not_empty[T: Copyable & Movable](xs: List[T]) -> Bool:
    return len(xs) > 0

# ---------- module meta ----------
fn __module_name__() -> String:
    return String("momijo/arrow_core/ipc/writer.mojo")
fn __self_test__() -> Bool:
    # extend later with real tests when IPC implemented
    return True

# ---------- IPCWriter ----------
struct IPCWriter:
    # placeholder state; extend with sink/file/stream later
    var sink_kind: Int   # 0: none/unknown
fn __init__(out self) -> None:
        self.sink_kind = 0
fn write_batch(self) -> Int:
        # TODO: serialize schema + arrays in Arrow IPC format (file/stream)
        # Stub returns 0 (bytes/rows written placeholder)
        return 0
fn __copyinit__(out self, other: Self) -> None:
        self.sink_kind = other.sink_kind
fn __moveinit__(out self, deinit other: Self) -> None:
        self.sink_kind = other.sink_kind