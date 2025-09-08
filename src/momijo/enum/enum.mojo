# Project:      Momijo
# Module:       src.momijo.enum.enum
# File:         enum.mojo
# Path:         src/momijo/enum/enum.mojo
#
# Description:  src.momijo.enum.enum — focused Momijo functionality with a stable public API.
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
#   - Structs: EnumValue
#   - Traits: Enum
#   - Key functions: __init__, __copyinit__, name, is_valid, clamp, parse, equals, enum_names_expected ...
#   - Uses generic functions/types with explicit trait bounds.


from momijo.enum.abi import enum_index_clamp, enum_index_of, enum_index_valid, enum_name_at
from momijo.enum.diagnostics import enum_check_index, enum_debug_dump, enum_expected_one_of, enum_mismatch, enum_parse_error

trait Enum:
    pass

# A typed handle to a value of an enum E, represented by its index.
# Names must be supplied (typically via a module-level function) when constructing
# or performing conversions. We avoid storing the names inside to keep the value tiny.
struct EnumValue[E: Enum](Copyable, Movable, Sized):
    var index: Int
fn __init__(out self, index: Int = 0) -> None:
        self.index = index
fn __copyinit__(out self, other: Self) -> None:
        self.index = other.index
fn name(self, names: List[String]) -> String:
        return enum_name_at(names, self.index)
fn is_valid(self, names: List[String]) -> Bool:
        return enum_index_valid(names, self.index)
fn clamp(mut self, names: List[String]) -> None:
        self.index = enum_index_clamp(names, self.index)
fn parse(out self, token: String, names: List[String], case_sensitive: Bool = True) -> Bool:
        var idx = enum_index_of(names, token, case_sensitive)
        if idx < 0:
            self.index = -1
            return False
        self.index = idx
        return True
fn equals(self, other: EnumValue[E]) -> Bool:
        return self.index == other.index

# Pretty helpers (forwarders)
fn enum_names_expected(enum_name: String, names: List[String]) -> String:
    return enum_expected_one_of(enum_name, names)
fn enum_dump(enum_name: String, names: List[String]) -> String:
    return enum_debug_dump(enum_name, names)
fn enum_validate_index(enum_name: String, i: Int, count: Int) -> (Bool, String):
    return enum_check_index(enum_name, i, count)
fn enum_parse_error_msg(enum_name: String, token: String) -> String:
    return enum_parse_error(enum_name, token)
fn enum_mismatch_msg(enum_name: String, got: String, expected: String) -> String:
    return enum_mismatch(enum_name, got, expected)