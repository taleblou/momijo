# Project:      Momijo
# Module:       src.momijo.nn.parameter
# File:         parameter.mojo
# Path:         src/momijo/nn/parameter.mojo
#
# Description:  Neural-network utilities for Momijo integrating with tensors,
#               optimizers, and training/evaluation loops.
#
# Author(s):    Morteza Taleblou & Mitra Daneshmand
# Website:      https://taleblou.ir/
# Repository:   https://github.com/taleblou/momijo
#
# License:      MIT License
# SPDX-License-Identifier: MIT
# Copyright:    (c) 2025 Morteza Taleblou & Mitra Daneshmand
#
# Notes:
#   - Structs: Parameter
#   - Key functions: __init__, data, set_data, requires_grad, set_requires_grad, has_grad, grad, set_grad ...
#   - Uses generic functions/types with explicit trait bounds.


from momijo.autograd.hook import call
from momijo.core.error import module
from momijo.utils.result import g
from pathlib import Path
from pathlib.path import Path

struct Parameter[T: Copyable & Movable]:
    var _data: T
    var _requires_grad: Bool
    var _has_grad: Bool
    var _grad: T
fn __init__(out self, data: T, requires_grad: Bool = True) -> None:
        self._data = data
        self._requires_grad = requires_grad
        self._has_grad = False
        # Initialize grad storage to a copy of data as a placeholder shape
        self._grad = data

    # --- Data access ---
fn data(self) -> T:
        return self._data
fn set_data(mut self, data: T) -> None:
        self._data = data

    # --- Autograd flags ---
fn requires_grad(self) -> Bool:
        return self._requires_grad
fn set_requires_grad(mut self, b: Bool) -> None:
        self._requires_grad = b
        if not b:
            # Drop any existing grad tracking
            self._has_grad = False

    # --- Gradient management ---
fn has_grad(self) -> Bool:
        return self._has_grad
fn grad(self, fallback: T) -> T:
        # Return stored gradient if present; otherwise return the provided fallback.
        if self._has_grad:
            return self._grad
        return fallback
fn set_grad(mut self, g: T) -> None:
        if self._requires_grad:
            self._grad = g
            self._has_grad = True
fn zero_grad(mut self) -> None:
        # Do not mutate `_grad` contents to avoid type-specific "zeros". Just drop the flag.
        self._has_grad = False
fn detach_(mut self) -> None:
        # Stop tracking grads for this parameter, similar to tensor.detach_ semantics.
        self._requires_grad = False
        self._has_grad = False

    # Convenience: assign value with optional "no grad" flag
fn assign_(mut self, value: T, track_grad: Bool = True) -> None:
        self._data = value
        self._requires_grad = track_grad
        self._has_grad = False

# --- Small helpers over a list of parameters ---
fn parameters_count[T: Copyable & Movable](params: List[Parameter[T]], only_trainable: Bool = True) -> Int:
    var c = 0
    for p in params:
        if only_trainable:
            if p.requires_grad():
                c += 1
        else:
            c += 1
    return c

fn zero_grad_all[T: Copyable & Movable](mut params: List[Parameter[T]]):
    # Note: List[Parameter[T]] is value-copied when passed; provide guidance for usage.
    # Users should call p.zero_grad() per parameter or manage in owning module.
    for i in range(len(params)):
        var p = params[i]
        p.zero_grad()
        params[i] = p

# --- Smoke tests ---
fn _self_test() -> Bool:
    var ok = True

    # Scalar Float64 parameter
    var p = Parameter[Float64](1.5, True)
    ok = ok and p.requires_grad()
    ok = ok and (p.data() == 1.5)

    p.set_grad(0.3)
    ok = ok and p.has_grad()
    ok = ok and (p.grad(0.0) == 0.3)

    p.zero_grad()
    ok = ok and not p.has_grad()
    ok = ok and (p.grad(-1.0) == -1.0)

    p.detach_()
    ok = ok and not p.requires_grad()

    # List parameter
    var v = List[Float64](); v.push(1.0); v.push(2.0)
    var q = Parameter[List[Float64]](v, True)
    ok = ok and q.requires_grad()

    var g = List[Float64](); g.push(0.1); g.push(0.2)
    q.set_grad(g)
    ok = ok and q.has_grad()
    var gf = List[Float64](); gf.push(-9.0)  # fallback
    var gr = q.grad(gf)
    ok = ok and (len(gr) == 2)

    # Helpers
    var ps = List[Parameter[Float64]](); ps.push(p)
    var cnt = parameters_count(ps, only_trainable=True)
    ok = ok and (cnt == 0)  # p was detached
    var cnt_all = parameters_count(ps, only_trainable=False)
    ok = ok and (cnt_all == 1)

    return ok