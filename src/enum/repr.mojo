# MIT License
# Copyright (c) 2025 Morteza Talebou (https://taleblou.ir/)
# Module: momijo.enum.repr

struct EnumRepr(ExplicitlyCopyable, Movable):
    var strategy: Int
    fn __init__(out self, strategy: Int):
        self.strategy = strategy
    fn __copyinit__(out self, other: Self):
        self.strategy = other.strategy
