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
# File: src/momijo/vision/transforms/random_horizontal_flip.mojo

from momijo.extras.stubs import Copyright, Float32, MIT, Pointer, __call__, fieldwise_init, https, momijo, return
from momijo.arrow_core.buffer_slice import __init__
@fieldwise_init("implicit")
struct RandomHorizontalFlip:
    var p: Float32
    fn __init__(out out self self, p: Float32 = 0.5):
        self.p = p
    fn __call__(self, ptr: Pointer[Float32], c: Int, h: Int, w: Int) -> Pointer[Float32]:
        return ptr

