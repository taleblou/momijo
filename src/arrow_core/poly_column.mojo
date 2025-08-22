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
# File: momijo/arrow_core/poly_column.mojo
#
# This file is part of the Momijo project.
# See the LICENSE file at the repository root for license information. 

from momijo.arrow_core.array import Array
from momijo.arrow_core.byte_string_array import ByteStringArray
from momijo.arrow_core.column import Column, StringColumn

enum PolyColumnTag:
    INT
    FLOAT64
    STRING
    UNKNOWN

struct PolyColumn(Copyable, Movable, Sized):
    var name: String
    var tag: PolyColumnTag
    var int_col: Optional[Column[Int]]
    var f64_col: Optional[Column[Float64]]
    var str_col: Optional[StringColumn]

    # ---------- Constructors ----------

    fn __init__(out self, name: String = ""):
        self.name = name
        self.tag = PolyColumnTag.UNKNOWN
        self.int_col = Optional[Column[Int]]()
        self.f64_col = Optional[Column[Float64]]()
        self.str_col = Optional[StringColumn]()

    fn from_int_array(out self, name: String, arr: Array[Int]):
        self.name = name
        self.tag = PolyColumnTag.INT
        var col: Column[Int]
        col.__init__(name, arr)
        self.int_col = Optional[Column[Int]](col)
        self.f64_col = Optional[Column[Float64]]()
        self.str_col = Optional[StringColumn]()

    fn from_f64_array(out self, name: String, arr: Array[Float64]):
        self.name = name
        self.tag = PolyColumnTag.FLOAT64
        var col: Column[Float64]
        col.__init__(name, arr)
        self.f64_col = Optional[Column[Float64]](col)
        self.int_col = Optional[Column[Int]]()
        self.str_col = Optional[StringColumn]()

    fn from_strings(out self, name: String, arr: ByteStringArray):
        self.name = name
        self.tag = PolyColumnTag.STRING
        var scol: StringColumn
        scol.__init__(name, arr)
        self.str_col = Optional[StringColumn](scol)
        self.int_col = Optional[Column[Int]]()
        self.f64_col = Optional[Column[Float64]]()

    # ---------- Properties ----------

    fn __len__(self) -> Int:
        match self.tag:
            case .INT: return self.int_col.value().len() if self.int_col.has_value() else 0
            case .FLOAT64: return self.f64_col.value().len() if self.f64_col.has_value() else 0
            case .STRING: return self.str_col.value().len() if self.str_col.has_value() else 0
            case .UNKNOWN: return 0

    fn len(self) -> Int:
        return self.__len__()

    fn type_name(self) -> String:
        match self.tag:
            case .INT: return "Int"
            case .FLOAT64: return "Float64"
            case .STRING: return "String"
            case .UNKNOWN: return "Unknown"

    # ---------- Accessors (generic) ----------

    fn get_int(self, i: Int) -> Int:
        if self.tag != PolyColumnTag.INT or not self.int_col.has_value():
            return 0
        return self.int_col.value().get_or(i, 0)

    fn get_f64(self, i: Int) -> Float64:
        if self.tag != PolyColumnTag.FLOAT64 or not self.f64_col.has_value():
            return 0.0
        return self.f64_col.value().get_or(i, 0.0)

    fn get_string(self, i: Int) -> String:
        if self.tag != PolyColumnTag.STRING or not self.str_col.has_value():
            return ""
        return self.str_col.value().get_or(i, "")

    # ---------- Mutation helpers ----------

    fn push_int(mut self, v: Int, valid: Bool = True):
        if self.tag == PolyColumnTag.INT and self.int_col.has_value():
            var c = self.int_col.value()
            c.push(v, valid)
            self.int_col = Optional[Column[Int]](c)

    fn push_f64(mut self, v: Float64, valid: Bool = True):
        if self.tag == PolyColumnTag.FLOAT64 and self.f64_col.has_value():
            var c = self.f64_col.value()
            c.push(v, valid)
            self.f64_col = Optional[Column[Float64]](c)

    fn push_string(mut self, s: String, valid: Bool = True):
        if self.tag == PolyColumnTag.STRING and self.str_col.has_value():
            var c = self.str_col.value()
            c.push(s, valid)
            self.str_col = Optional[StringColumn](c)

    # ---------- Conversion ----------

    fn to_list_int(self) -> List[Int]:
        if self.tag == PolyColumnTag.INT and self.int_col.has_value():
            return self.int_col.value().to_list()
        return List[Int]()

    fn to_list_f64(self) -> List[Float64]:
        if self.tag == PolyColumnTag.FLOAT64 and self.f64_col.has_value():
            return self.f64_col.value().to_list()
        return List[Float64]()

    fn to_list_strings(self) -> List[String]:
        if self.tag == PolyColumnTag.STRING and self.str_col.has_value():
            return self.str_col.value().to_strings()
        return List[String]()