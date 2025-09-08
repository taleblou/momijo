# Project:      Momijo
# Module:       src.momijo.vision.transforms.random_horizontal_flip
# File:         random_horizontal_flip.mojo
# Path:         src/momijo/vision/transforms/random_horizontal_flip.mojo
#
# Description:  src.momijo.vision.transforms.random_horizontal_flip — focused Momijo functionality with a stable public API.
#               Composable building blocks intended for reuse.
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
#   - Structs: RandomHorizontalFlip
#   - Key functions: __init__, __call__, __copyinit__, __moveinit__
#   - Uses generic functions/types with explicit trait bounds.
#   - Low-level memory (Pointer/UnsafePointer) used; observe safety invariants.


from momijo.arrow_core.buffer_slice import __init__
from momijo.extras.stubs import Float32, Pointer, __call__, fieldwise_init, return

@fieldwise_init("implicit")
struct RandomHorizontalFlip:
    var p: Float32
fn __init__(out out self self, p: Float32 = 0.5) -> None:
        self.p = p
fn __call__(self, ptr: Pointer[Float32], c: Int, h: Int, w: Int) -> Pointer[Float32]:
        return ptr
fn __copyinit__(out self, other: Self) -> None:
        self.p = other.p
fn __moveinit__(out self, deinit other: Self) -> None:
        self.p = other.p