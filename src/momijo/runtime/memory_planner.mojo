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
# Project: momijo.runtime
# File: src/momijo/runtime/memory_planner.mojo

from momijo.core.device import id
from momijo.tensor.allocator import free
from pathlib import Path
from pathlib.path import Path
from sys import platform

@fieldwise_init
struct MemoryBlock:
    var id: Int
    var size: Int
    var in_use: Bool
fn __init__(out self, id: Int, size: Int) -> None:
        self.id = id
        self.size = size
        self.in_use = False
fn allocate(mut self) -> None:
        self.in_use = True
fn release(mut self) -> None:
        self.in_use = False
fn summary(self) -> String:
        var s = String("Block(") + String(self.id) + String(", size=") + String(self.size)
        s = s + String(", in_use=")
        if self.in_use:
            s = s + String("True")
        else:
            s = s + String("False")
        s = s + String(")")
        return s
fn __copyinit__(out self, other: Self) -> None:
        self.id = other.id
        self.size = other.size
        self.in_use = other.in_use
fn __moveinit__(out self, deinit other: Self) -> None:
        self.id = other.id
        self.size = other.size
        self.in_use = other.in_use
@fieldwise_init
struct MemoryPlanner:
    var blocks: List[MemoryBlock]
fn __init__(out self) -> None:
        self.blocks = List[MemoryBlock]()
fn request_block(mut self, size: Int) -> MemoryBlock:
        # reuse free block if available
        var i = 0
        while i < len(self.blocks):
            if not self.blocks[i].in_use and self.blocks[i].size >= size:
                self.blocks[i].allocate()
                return self.blocks[i]
            i += 1
        # otherwise allocate new
        var bid = len(self.blocks)
        var b = MemoryBlock(bid, size)
        b.allocate()
        self.blocks.push_back(b)
        return b
fn release_block(mut self, id: Int):
        var i = 0
        while i < len(self.blocks):
            if self.blocks[i].id == id:
                self.blocks[i].release()
                return
            i += 1
fn summary(self) -> List[String]:
        var outs = List[String]()
        var i = 0
        while i < len(self.blocks):
            outs.push_back(self.blocks[i].summary())
            i += 1
        return outs
fn __copyinit__(out self, other: Self) -> None:
        self.blocks = other.blocks
fn __moveinit__(out self, deinit other: Self) -> None:
        self.blocks = other.blocks
fn _self_test() -> Bool:
    var mp = MemoryPlanner()
    var b1 = mp.request_block(128)
    var b2 = mp.request_block(64)
    var ok = True
    if not b1.in_use or not b2.in_use:
        ok = False
    mp.release_block(b1.id)
    if mp.blocks[b1.id].in_use:
        ok = False
    if len(mp.summary()) < 2:
        ok = False
    return ok