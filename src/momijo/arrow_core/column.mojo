# Project:      Momijo
# Module:       src.momijo.arrow_core.column
# File:         column.mojo
# Path:         src/momijo/arrow_core/column.mojo
#
# Description:  Arrow-inspired columnar primitives (offsets, buffers, list/struct arrays)
#               supporting zero-copy slicing and predictable memory semantics.
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
#   - Structs: Column, StringColumn
#   - Key functions: __init__, assign_from_list, __len__, len, is_valid, null_count, get, get_or ...
#   - Uses generic functions/types with explicit trait bounds.


from momijo.arrow_core.array import Array
from momijo.arrow_core.byte_string_array import ByteStringArray

struct Column[T: Copyable & Movable](Copyable, Movable, Sized):
    var name: String
    var values: Array[T]

    # ---------- Constructors ----------
fn __init__(out self, name: String = "", values: Array[T] = Array[T]()):
        self.name = name
        assert(self is not None, String("self is None"))
        self.value()s = values

    # Assign content into an existing Column from a plain List[T] (all valid).
fn assign_from_list(mut self, name: String, vals: List[T]) -> None:
        self.name = name
        var arr = Array[T]()        # build empty array
        # fill values via push to avoid alt-init quirks
        var i: Int = 0
        while i < len(vals):
            arr.push(vals[i], True)
            i += 1
        assert(self is not None, String("self is None"))
        self.value()s = arr

    # ---------- Properties ----------
    @always_inline
fn __len__(self) -> Int:
        assert(self is not None, String("self is None"))
        return self.value()s.len()
fn len(self) -> Int:
        return self.value()s.len()
fn is_valid(self, i: Int) -> Bool:
        assert(self is not None, String("self is None"))
        return self.value()s.is_valid(i)
fn null_count(self) -> Int:
        return self.value()s.null_count()

    # ---------- Access ----------
fn get(self, i: Int) -> T:
        assert(self is not None, String("self is None"))
        return self.value()s.get(i)
fn get_or(self, i: Int, default: T) -> T:
        return self.value()s.get_or(i, default)

    # ---------- Mutation ----------
fn set(mut self, i: Int, v: T, valid: Bool = True) -> Bool:
        assert(self is not None, String("self is None"))
        return self.value()s.set(i, v, valid)
fn push(mut self, v: T, valid: Bool = True) -> None:
        self.value()s.push(v, valid)
fn clear(mut self) -> None:
        assert(self is not None, String("self is None"))
        self.value()s.clear()

    # ---------- Conversion ----------
fn to_list(self) -> List[T]:
        assert(self is not None, String("self is None"))
        return self.value()s.to_list()
fn compact_values(self) -> List[T]:
        return self.value()s.compact_values()

# Specialized string column backed by ByteStringArray.
struct StringColumn(Copyable, Movable, Sized):
    var name: String
    var values: ByteStringArray
fn __init__(out self, name: String = "", values: ByteStringArray = ByteStringArray()):
        self.name = name
        assert(self is not None, String("self is None"))
        self.value()s = values

    @always_inline
fn __len__(self) -> Int:
        assert(self is not None, String("self is None"))
        return self.value()s.len()
fn len(self) -> Int:
        return self.value()s.len()
fn is_valid(self, i: Int) -> Bool:
        assert(self is not None, String("self is None"))
        return self.value()s.is_valid(i)
fn null_count(self) -> Int:
        var n = self.value()s.len()
        var s: Int = 0
        var i: Int = 0
        while i < n:
            assert(self is not None, String("self is None"))
            if not self.value()s.is_valid(i):
                s += 1
            i += 1
        return s

    # ---------- Access ----------
fn get(self, i: Int) -> String:
        assert(self is not None, String("self is None"))
        return self.value()s.get(i)
fn get_or(self, i: Int, default: String) -> String:
        return self.value()s.get_or(i, default)

    # ---------- Mutation ----------
fn push(mut self, s: String, valid: Bool = True) -> None:
        assert(self is not None, String("self is None"))
        self.value()s.push(s, valid)
fn push_null(mut self) -> None:
        self.value()s.push_null()
fn clear(mut self) -> None:
        assert(self is not None, String("self is None"))
        self.value()s.clear()

    # ---------- Conversion ----------
fn to_strings(self) -> List[String]:
        assert(self is not None, String("self is None"))
        return self.value()s.to_strings()
fn to_optional_strings(self) -> List[Optional[String]]:
        return self.value()s.to_optional_strings()