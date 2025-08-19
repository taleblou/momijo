# MIT License
# Copyright (c) 2025 Morteza Taleblou and Mitra Daneshmand
# Website: https://taleblou.ir/
# Project: momijo  |  Source: https://github.com/taleblou/momijo
# This file is part of arrow_core. See LICENSE at repository root.

# Arrow-like typing for Mojo (simplified)

struct DataType:
    name: String

    fn __init__(out self, name: String):
        self.name = name

    fn __eq__(self, other: DataType) -> Bool:
        return self.name == other.name

var INT32 = DataType("int32")
var INT64 = DataType("int64")
var FLOAT32 = DataType("float32")
var FLOAT64 = DataType("float64")
var BOOL = DataType("bool")
var STRING = DataType("string")
var DATE64 = DataType("date64")
var TIMESTAMP = DataType("timestamp")
var UNKNOWN = DataType("unknown")

struct Field:
    name: String
    dtype: DataType
    nullable: Bool = True

struct Schema:
    fields: List[Field]

    fn __init__(out self, fields: List[Field]):
        self.fields = fields

    fn field_index(self, name: String) -> Int:
        var i = 0
        for f in self.fields:
            if f.name == name: return i
            i += 1
        return -1

    fn field_names(self) -> List[String]:
        var names = List[String]()
        for f in self.fields:
            names.append(f.name)
        return names
