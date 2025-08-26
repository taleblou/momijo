# MIT License
# Copyright (c) 2025 Morteza Talebou and Mitra Daneshmand
# Project: momijo  |  Source: https://github.com/taleblou/momijo
# This file is part of the Momijo project. See the LICENSE file at the repository root.
# Momijo
# SPDX-License-Identifier: MIT
# Website: https://taleblou.ir/
# Repository: https://github.com/taleblou/momijo
#
# Project: momijo.arrow_core.ipc
# File: momijo/arrow_core/ipc/reader.mojo

# ---------- small helpers (kept for smoke tests / parity) ----------
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
    return String("momijo/arrow_core/ipc/reader.mojo")

fn __self_test__() -> Bool:
    # extend later with real tests when IPC implemented
    return True

# ---------- IPCReader ----------
struct IPCReader:
    # lightweight stub; extend with stream/file state as needed
    var source_kind: Int  # 0: none/unknown (placeholder)

    fn __init__(out self):
        self.source_kind = 0

    fn read_next_batch(self) -> Int:
        # TODO: implement Arrow IPC reading (file/stream) + message/metadata parsing
        # Stub returns 0 (no rows)
        return 0
