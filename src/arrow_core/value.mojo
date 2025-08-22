# MIT License
# Copyright (c) 2025 Morteza Talebou and Mitra Daneshmand
# Project: momijo  |  Source: https://github.com/taleblou/momijo
# This file is part of the Momijo project. See the LICENSE file at the repository root.
# Momijo 
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Morteza Taleblou and Mitra Daneshmand
# Website: https://taleblou.ir/
# Repository: https://github.com/taleblou/momijo
#
# Project: momijo.arrow_core
# File: momijo/arrow_core/value.mojo
#
# This file is part of the Momijo project.
# See the LICENSE file at the repository root for license information. 

from momijo.arrow_core.types import ArrowType, arrow_type_name
from momijo.enum import Enum

struct ValueTag: Enum:
    INT
    FLOAT64
    STRING
    BOOL
    NULL

struct Value(Copyable, Movable, Sized):
    var tag: ValueTag
    var int_val: Int
    var f64_val: Float64
    var str_val: String
    var bool_val: Bool

    # ---------- Constructors ----------

    fn __init__(out self):
        self.tag = ValueTag.NULL
        self.int_val = 0
        self.f64_val = 0.0
        self.str_val = ""
        self.bool_val = False

    fn from_int(out self, v: Int):
        self.tag = ValueTag.INT
        self.int_val = v
        self.f64_val = 0.0
        self.str_val = ""
        self.bool_val = False

    fn from_f64(out self, v: Float64):
        self.tag = ValueTag.FLOAT64
        self.int_val = 0
        self.f64_val = v
        self.str_val = ""
        self.bool_val = False

    fn from_string(out self, v: String):
        self.tag = ValueTag.STRING
        self.int_val = 0
        self.f64_val = 0.0
        self.str_val = v
        self.bool_val = False

    fn from_bool(out self, v: Bool):
        self.tag = ValueTag.BOOL
        self.int_val = 0
        self.f64_val = 0.0
        self.str_val = ""
        self.bool_val = v

    fn null(out self):
        self.tag = ValueTag.NULL
        self.int_val = 0
        self.f64_val = 0.0
        self.str_val = ""
        self.bool_val = False

    # ---------- Accessors ----------

    fn as_int(self) -> Int:
        return self.int_val if self.tag == ValueTag.INT else 0

    fn as_f64(self) -> Float64:
        return self.f64_val if self.tag == ValueTag.FLOAT64 else 0.0

    fn as_string(self) -> String:
        return self.str_val if self.tag == ValueTag.STRING else ""

    fn as_bool(self) -> Bool:
        return self.bool_val if self.tag == ValueTag.BOOL else False

    fn is_null(self) -> Bool:
        return self.tag == ValueTag.NULL

    fn type_name(self) -> String:
        match self.tag:
            case .INT: return "Int"
            case .FLOAT64: return "Float64"
            case .STRING: return "String"
            case .BOOL: return "Bool"
            case .NULL: return "Null"
