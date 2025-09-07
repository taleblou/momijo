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
# Project: momijo.arrow_core
# File: src/momijo/arrow_core/poly_column.mojo

from momijo.arrow_core.types import UNKNOWN
from momijo.nn.parameter import data
from momijo.tensor.broadcast import valid
from momijo.tensor.errors import Unknown
from pathlib import Path
from pathlib.path import Path

fn __module_name__() -> String:
    return String("momijo/arrow_core/poly_column.mojo")
fn __self_test__() -> Bool:
    var p = PolyColumn(String("x"))
    return p.len() == 0 and p.type_name() == String("Unknown")

# ------------------------------------------------------------
# Tag helpers (Int32-based). 0=UNKNOWN, 1=INT, 2=FLOAT64, 3=STRING
# ------------------------------------------------------------
fn POLYTAG_UNKNOWN() -> Int32:  return 0
fn POLYTAG_INT()     -> Int32:  return 1
fn POLYTAG_F64()     -> Int32:  return 2
fn POLYTAG_STR()     -> Int32:  return 3
fn poly_tag_name(tag: Int32) -> String:
    if tag == POLYTAG_INT():      return String("Int")
    elif tag == POLYTAG_F64():    return String("Float64")
    elif tag == POLYTAG_STR():    return String("String")
    else:                         return String("Unknown")

# ------------------------------------------------------------
# PolyColumn (list-backed, stable API)
# ------------------------------------------------------------
struct PolyColumn(Copyable, Movable, Sized):
    var name: String
    var tag: Int32            # see tag helpers above
    var int_data: List[Int]
    var f64_data: List[Float64]
    var str_data: List[String]

    # Constructors
fn __init__(out self, name: String = String("")):
        self.name = name
        self.tag = POLYTAG_UNKNOWN()
        self.int_data = List[Int]()
        self.f64_data = List[Float64]()
        self.str_data = List[String]()

    # "from_*" should mutate an existing instance (no 'out self' here)
fn from_int_array(mut self, name: String, arr: List[Int]) -> None:
        self.name = name
        self.tag = POLYTAG_INT()
        self.int_data = arr
        self.f64_data = List[Float64]()
        self.str_data = List[String]()
fn from_f64_array(mut self, name: String, arr: List[Float64]) -> None:
        self.name = name
        self.tag = POLYTAG_F64()
        self.f64_data = arr
        self.int_data = List[Int]()
        self.str_data = List[String]()
fn from_strings(mut self, name: String, arr: List[String]) -> None:
        self.name = name
        self.tag = POLYTAG_STR()
        self.str_data = arr
        self.int_data = List[Int]()
        self.f64_data = List[Float64]()

    # Properties
fn __len__(self) -> Int:
        if self.tag == POLYTAG_INT():
            return len(self.int_data)
        elif self.tag == POLYTAG_F64():
            return len(self.f64_data)
        elif self.tag == POLYTAG_STR():
            return len(self.str_data)
        else:
            return 0
fn len(self) -> Int:
        return self.__len__()
fn type_name(self) -> String:
        return poly_tag_name(self.tag)

    # Accessors
fn get_int(self, i: Int) -> Int:
        if self.tag != POLYTAG_INT(): return 0
        if i < 0 or i >= len(self.int_data): return 0
        return self.int_data[i]
fn get_f64(self, i: Int) -> Float64:
        if self.tag != POLYTAG_F64(): return 0.0
        if i < 0 or i >= len(self.f64_data): return 0.0
        return self.f64_data[i]
fn get_string(self, i: Int) -> String:
        if self.tag != POLYTAG_STR(): return String("")
        if i < 0 or i >= len(self.str_data): return String("")
        return self.str_data[i]

    # Mutation helpers
fn push_int(mut self, v: Int, valid: Bool = True) -> None:
        if not valid: return
        if self.tag == POLYTAG_INT():
            self.int_data.append(v)
fn push_f64(mut self, v: Float64, valid: Bool = True) -> None:
        if not valid: return
        if self.tag == POLYTAG_F64():
            self.f64_data.append(v)
fn push_string(mut self, s: String, valid: Bool = True) -> None:
        if not valid: return
        if self.tag == POLYTAG_STR():
            self.str_data.append(s)

    # Conversion
fn to_list_int(self) -> List[Int]:
        if self.tag == POLYTAG_INT(): return self.int_data
        return List[Int]()
fn to_list_f64(self) -> List[Float64]:
        if self.tag == POLYTAG_F64(): return self.f64_data
        return List[Float64]()
fn to_list_strings(self) -> List[String]:
        if self.tag == POLYTAG_STR(): return self.str_data
        return List[String]()