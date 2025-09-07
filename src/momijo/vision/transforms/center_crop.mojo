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
# File: src/momijo/vision/transforms/center_crop.mojo

from memory import Pointer
from momijo.arrow_core.buffer_slice import __init__
from momijo.tensor.storage import ptr
from momijo.vision.tensor import Float32
from momijo.vision.transforms.compose import __call__
from pathlib import Path
from pathlib.path import Path

@fieldwise_init
struct CenterCrop:
    var out_h: Int
    var out_w: Int
fn __init__(out out self self, size: (Int, Int)):
        self.out_h = size[0]
        self.out_w = size[1]
fn __call__(self, ptr: Pointer[Float32], c: Int, h: Int, w: Int) -> Pointer[Float32]:
        return ptr
fn __copyinit__(out self, other: Self) -> None:
        self.out_h = other.out_h
        self.out_w = other.out_w
fn __moveinit__(out self, deinit other: Self) -> None:
        self.out_h = other.out_h
        self.out_w = other.out_w