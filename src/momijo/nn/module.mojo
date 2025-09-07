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
# Project: momijo.nn
# File: src/momijo/nn/module.mojo

from momijo.core.error import module
from momijo.dataframe.helpers import m
from momijo.visual.stats.stats import argmax, argmin
from pathlib import Path
from pathlib.path import Path

struct Module:
    var _training: Bool
fn __init__(out self, training: Bool = True) -> None:
        self._training = training
fn train_mode(mut self) -> None:
        # Switch this module into training mode.
        self._training = True
fn eval_mode(mut self) -> None:
        # Switch this module into eval/inference mode.
        self._training = False
fn is_training(self) -> Bool:
        return self._training
fn __copyinit__(out self, other: Self) -> None:
        self._training = other._training
fn __moveinit__(out self, deinit other: Self) -> None:
        self._training = other._training
# --- Utility: ensure_not_empty ---
fn ensure_not_empty[T: Copyable & Movable](xs: List[T]) -> Bool:
    return len(xs) > 0

# --- Argmax / Argmin (generic over Comparable) ---
fn argmax_index[T: Comparable & Copyable & Movable](xs: List[T]) -> Int:
    var n = len(xs)
    if n == 0:
        return -1
    var best = xs[0]
    var idx = 0
    var i = 1
    while i < n:
        if xs[i] > best:
            best = xs[i]
            idx = i
        i += 1
    return idx

fn argmin_index[T: Comparable & Copyable & Movable](xs: List[T]) -> Int:
    var n = len(xs)
    if n == 0:
        return -1
    var best = xs[0]
    var idx = 0
    var i = 1
    while i < n:
        if xs[i] < best:
            best = xs[i]
            idx = i
        i += 1
    return idx

# --- Minimal smoke test ---
fn _self_test() -> Bool:
    var ok = True

    # Module mode flips
    var m = Module()
    ok = ok and m.is_training()
    m.eval_mode()
    ok = ok and not m.is_training()
    m.train_mode()
    ok = ok and m.is_training()

    # ensure_not_empty
    var xs = List[Int]()
    ok = ok and not ensure_not_empty(xs)
    xs.push(3)
    ok = ok and ensure_not_empty(xs)

    # argmax/argmin
    var ys = List[Int](); ys.push(5); ys.push(2); ys.push(9); ys.push(1)
    ok = ok and (argmax_index(ys) == 2)
    ok = ok and (argmin_index(ys) == 3)

    var zs = List[Float64](); zs.push(1.5); zs.push(3.25); zs.push(-7.0)
    ok = ok and (argmax_index(zs) == 1)
    ok = ok and (argmin_index(zs) == 2)

    return ok