# MIT License
# Copyright (c) 2025 Morteza Talebou (https://taleblou.ir/)
# Module: momijo.enum.enum

struct EnumValue(ExplicitlyCopyable, Movable):
    var tag: Int
    var w0: UInt64
    var w1: UInt64
    var w2: UInt64
    var w3: UInt64

    fn __init__(out self, tag: Int, w0: Int = 0, w1: Int = 0, w2: Int = 0, w3: Int = 0):
        self.tag = tag
        self.w0 = UInt64(w0)
        self.w1 = UInt64(w1)
        self.w2 = UInt64(w2)
        self.w3 = UInt64(w3)

    fn __copyinit__(out self, other: Self):
        self.tag = other.tag
        self.w0 = other.w0
        self.w1 = other.w1
        self.w2 = other.w2
        self.w3 = other.w3
