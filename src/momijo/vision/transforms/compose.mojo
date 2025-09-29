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
# Project: momijo.vision.transforms
# File: momijo/vision/transforms/compose.mojo
 
 
@fieldwise_init
struct Compose[T: Copyable & Movable]:
    var _fns: List[fn(T) -> T]

    fn __init__(out self, fns: List[fn(T) -> T]):
        self._fns = fns

    # Apply the pipeline to x, returning the transformed result.
    fn __call__(self, x: T) -> T:
        var y = x
        var i = 0
        var n = len(self._fns)
        while i < n:
            y = self._fns[i](y)
            i += 1
        return y

    # Pipeline management ------------------------------------------------------

    fn push(mut self, f: fn(T) -> T):
        self._fns.append(f)

    fn extend(mut self, fs: List[fn(T) -> T]):
        var i = 0
        var n = len(fs)
        while i < n:
            self._fns.append(fs[i])
            i += 1

    fn len(self) -> Int:
        return len(self._fns)

    fn is_empty(self) -> Bool:
        return len(self._fns) == 0

    fn clear(mut self):
        self._fns = List[fn(T) -> T]()

    fn at(self, idx: Int) -> fn(T) -> T:
        return self._fns[idx]

    # Constructors -------------------------------------------------------------

    @staticmethod
    fn from_funcs(fs: List[fn(T) -> T]) -> Compose[T]:
        return Compose[T](fs)

# Convenience helper for immediate composition without constructing explicitly
fn compose_apply[T: Copyable & Movable](x: T, fs: List[fn(T) -> T]) -> T:
    var p = Compose[T](fs)
    return p(x)
