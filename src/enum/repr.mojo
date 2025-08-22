# MIT License
# Copyright (c) 2025 Morteza Talebou and Mitra Daneshmand
# Project: momijo  |  Source: https://github.com/taleblou/momijo
# This file is part of the Momijo project. See the LICENSE file at the repository root.
struct EnumRepr(ExplicitlyCopyable, Movable):
    var strategy: Int
    fn __init__(out self, strategy: Int):
        self.strategy = strategy
    fn __copyinit__(out self, other: Self):
        self.strategy = other.strategy