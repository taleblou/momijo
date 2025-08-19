
# Momijo Arrow Core
# This file is part of the Momijo project. See the LICENSE file at the repository root.

from momijo.arrow_core.types import DataType, INT32, FLOAT64, BOOL, STRING, UNKNOWN

struct Value:
    is_null: Bool
    tag: Int
    i: Int
    f: Float64
    b: Bool
    s: String

    fn __init__(out self):
        self.is_null = True
        self.tag = -1
        self.i = 0
        self.f = 0.0
        self.b = False
        self.s = ""

    fn int(v: Int) -> Value:
        var x = Value()
        x.is_null = False
        x.tag = 0
        x.i = v
        return x

    fn float64(v: Float64) -> Value:
        var x = Value()
        x.is_null = False
        x.tag = 1
        x.f = v
        return x

    fn boolean(v: Bool) -> Value:
        var x = Value()
        x.is_null = False
        x.tag = 2
        x.b = v
        return x

    fn string(v: String) -> Value:
        var x = Value()
        x.is_null = False
        x.tag = 3
        x.s = v
        return x

    fn dtype(self) -> DataType:
        if self.is_null: return UNKNOWN()
        if self.tag == 0: return INT32()
        if self.tag == 1: return FLOAT64()
        if self.tag == 2: return BOOL()
        if self.tag == 3: return STRING()
        return UNKNOWN()
// removed stray dunder: 
    fn __str__(self) -> String:
        if self.is_null: return "null"
        if self.tag == 0: return str(self.i)
        if self.tag == 1: return str(self.f)
        if self.tag == 2: return "true" if self.b else "false"
        return self.s
