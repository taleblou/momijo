# MIT License
# Copyright (c) 2025 Morteza Talebou (https://taleblou.ir/)
# Module: momijo.enum.match (package)

struct Case(ExplicitlyCopyable, Movable):
    var tag: Int
    var result: Int
    fn __init__(out self, tag: Int, result: Int):
        self.tag = tag
        self.result = result
    fn __copyinit__(out self, other: Self):
        self.tag = other.tag
        self.result = other.result

struct RangeCase(ExplicitlyCopyable, Movable):
    var start: Int
    var end: Int
    var result: Int
    fn __init__(out self, start: Int, end: Int, result: Int):
        self.start = start
        self.end = end
        self.result = result
    fn __copyinit__(out self, other: Self):
        self.start = other.start
        self.end = other.end
        self.result = other.result
