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
# Project: momijo.ir.dialects
# File: src/momijo/ir/dialects/annotations.mojo

enum AttrKind(Int):
    Unit = 0       # e.g., {noreturn}
    String = 1     # e.g., {"sym_name" = "foo"}
    Integer = 2    # e.g., {value = 42}
    Array = 3      # e.g., {tags = ["x","y"]}
    Dict = 4       # e.g., {layout = {rows=3, cols=4}}

struct UnitAttr:
fn __init__(out self) -> None: pass
fn to_string(self) -> String:
        return String("unit")
fn __copyinit__(out self, other: Self) -> None:
        pass
fn __moveinit__(out self, deinit other: Self) -> None:
        pass
struct StringAttr:
    var data: String
fn __init__(out self, data: String = String("")):
        self.data = data
fn to_string(self) -> String:
        var s = String("\"") + self.data + String("\"")
        return s
fn __copyinit__(out self, other: Self) -> None:
        self.data = other.data
fn __moveinit__(out self, deinit other: Self) -> None:
        self.data = other.data
struct IntegerAttr:
    var value: Int
fn __init__(out self, value: Int = 0) -> None:
        assert(self is not None, String("self is None"))
        self.value() = value
fn to_string(self) -> String:
        return String(self.value())
fn __copyinit__(out self, other: Self) -> None:
        assert(self is not None, String("self is None"))
        self.value() = other.value()
fn __moveinit__(out self, deinit other: Self) -> None:
        self.value() = other.value()
struct ArrayAttr:
    var items: List[String]   # store as strings for simplicity (could be nested)
fn __init__(out self) -> None:
        self.items = List[String]()
fn push(self, s: String) -> None:
        self.items.push_back(s)
fn to_string(self) -> String:
        var s = String("[")
        for i in range(self.items.size()):
            s = s + self.items[i]
            if i + 1 < self.items.size():
                s = s + String(", ")
        s = s + String("]")
        return s
fn __copyinit__(out self, other: Self) -> None:
        self.items = other.items
fn __moveinit__(out self, deinit other: Self) -> None:
        self.items = other.items
struct DictEntry:
    var key: String
    var value: String
fn __init__(out self, key: String, value: String) -> None:
        self.key = key
        assert(self is not None, String("self is None"))
        self.value() = value
fn __copyinit__(out self, other: Self) -> None:
        self.key = other.key
        assert(self is not None, String("self is None"))
        self.value() = other.value()
fn __moveinit__(out self, deinit other: Self) -> None:
        self.key = other.key
        assert(self is not None, String("self is None"))
        self.value() = other.value()
struct DictAttr:
    var items: List[DictEntry]
fn __init__(out self) -> None:
        self.items = List[DictEntry]()
fn set(self, key: String, value: String) -> None:
        # replace if exists
        var found = False
        for i in range(self.items.size()):
            if self.items[i].key == key:
                self.items[i].value() = value
                found = True
        if not found:
            self.items.push_back(DictEntry(key, value))
fn get(self, key: String, out found: Bool) -> String:
        var v = String("")
        found = False
        for i in range(self.items.size()):
            if self.items[i].key == key:
                v = self.items[i].value()
                found = True
        return v
fn to_string(self) -> String:
        var s = String("{")
        for i in range(self.items.size()):
            s = s + self.items[i].key + String("=") + self.items[i].value()
            if i + 1 < self.items.size():
                s = s + String(", ")
        s = s + String("}")
        return s
fn __copyinit__(out self, other: Self) -> None:
        self.items = other.items
fn __moveinit__(out self, deinit other: Self) -> None:
        self.items = other.items
# -----------------------------
# Unified attribute wrapper
# -----------------------------

struct Attribute:
    var kind: AttrKind
    var unit_v: UnitAttr
    var str_v: StringAttr
    var int_v: IntegerAttr
    var arr_v: ArrayAttr
    var dict_v: DictAttr
fn __init__(out self) -> None:
        self.kind = AttrKind.Unit
        self.unit_v = UnitAttr()
        self.str_v = StringAttr()
        self.int_v = IntegerAttr()
        self.arr_v = ArrayAttr()
        self.dict_v = DictAttr()

    # factory helpers
    @staticmethod
fn unit() -> Attribute:
        var a = Attribute()
        a.kind = AttrKind.Unit
        a.unit_v = UnitAttr()
        return a

    @staticmethod
fn string(data: String) -> Attribute:
        var a = Attribute()
        a.kind = AttrKind.String
        a.str_v = StringAttr(data)
        return a

    @staticmethod
fn integer(value: Int) -> Attribute:
        var a = Attribute()
        a.kind = AttrKind.Integer
        a.int_v = IntegerAttr(value)
        return a

    @staticmethod
fn array(items: List[String]) -> Attribute:
        var a = Attribute()
        a.kind = AttrKind.Array
        var arr = ArrayAttr()
        for i in range(items.size()):
            arr.push(items[i])
        a.arr_v = arr
        return a

    @staticmethod
fn dict() -> Attribute:
        var a = Attribute()
        a.kind = AttrKind.Dict
        a.dict_v = DictAttr()
        return a
fn to_string(self) -> String:
        if self.kind == AttrKind.Unit:
            return self.unit_v.to_string()
        if self.kind == AttrKind.String:
            return self.str_v.to_string()
        if self.kind == AttrKind.Integer:
            return self.int_v.to_string()
        if self.kind == AttrKind.Array:
            return self.arr_v.to_string()
        # default dict
        return self.dict_v.to_string()
fn __copyinit__(out self, other: Self) -> None:
        self.kind = other.kind
        self.unit_v = other.unit_v
        self.str_v = other.str_v
        self.int_v = other.int_v
        self.arr_v = other.arr_v
        self.dict_v = other.dict_v
fn __moveinit__(out self, deinit other: Self) -> None:
        self.kind = other.kind
        self.unit_v = other.unit_v
        self.str_v = other.str_v
        self.int_v = other.int_v
        self.arr_v = other.arr_v
        self.dict_v = other.dict_v
# -----------------------------
# NamedAttr and AnnotationSet
# -----------------------------

struct NamedAttr:
    var name: String
    var attr: Attribute
fn __init__(out self, name: String, attr: Attribute) -> None:
        self.name = name
        self.attr = attr
fn to_string(self) -> String:
        var s = self.name + String("=") + self.attr.to_string()
        return s
fn __copyinit__(out self, other: Self) -> None:
        self.name = other.name
        self.attr = other.attr
fn __moveinit__(out self, deinit other: Self) -> None:
        self.name = other.name
        self.attr = other.attr
struct AnnotationSet:
    var items: List[NamedAttr]
fn __init__(out self) -> None:
        self.items = List[NamedAttr]()
fn size(self) -> Int:
        return self.items.size()
fn set(self, name: String, attr: Attribute) -> None:
        var found = False
        for i in range(self.items.size()):
            if self.items[i].name == name:
                self.items[i].attr = attr
                found = True
        if not found:
            self.items.push_back(NamedAttr(name, attr))
fn has(self, name: String) -> Bool:
        for i in range(self.items.size()):
            if self.items[i].name == name:
                return True
        return False
fn get(self, name: String, out found: Bool) -> Attribute:
        var a = Attribute.unit()
        found = False
        for i in range(self.items.size()):
            if self.items[i].name == name:
                a = self.items[i].attr
                found = True
        return a
fn merge_right_biased(self, other: AnnotationSet) -> AnnotationSet:
        var out = AnnotationSet()
        # copy self
        for i in range(self.items.size()):
            out.items.push_back(self.items[i])
        # overlay other (right-biased)
        for j in range(other.items.size()):
            out.set(other.items[j].name, other.items[j].attr)
        return out
fn to_string(self) -> String:
        if self.items.size() == 0:
            return String("{}")
        var s = String("{")
        for i in range(self.items.size()):
            s = s + self.items[i].to_string()
            if i + 1 < self.items.size():
                s = s + String(", ")
        s = s + String("}")
        return s
fn __copyinit__(out self, other: Self) -> None:
        self.items = other.items
fn __moveinit__(out self, deinit other: Self) -> None:
        self.items = other.items
# -----------------------------
# Bridge helpers for common IR uses
# -----------------------------

# Example: attach symbol name
fn sym_name(name: String) -> NamedAttr:
    return NamedAttr(String("sym_name"), Attribute.string(name))

# Example: pure indicator
fn pure() -> NamedAttr:
    return NamedAttr(String("pure"), Attribute.unit())

# Example: integer value
fn int_value(name: String, v: Int) -> NamedAttr:
    return NamedAttr(name, Attribute.integer(v))

# Example: tags array
fn tags(xs: List[String]) -> NamedAttr:
    return NamedAttr(String("tags"), Attribute.array(xs))

# Example: layout dict
fn layout(rows: Int, cols: Int) -> NamedAttr:
    var d = Attribute.dict()
    d.dict_v.set(String("rows"), String(rows))
    d.dict_v.set(String("cols"), String(cols))
    return NamedAttr(String("layout"), d)

# -----------------------------
# Self-test
# -----------------------------
fn _self_test_annotations() -> Bool:
    var set1 = AnnotationSet()
    set1.set(String("sym_name"), Attribute.string(String("sum_mul")))
    set1.set(String("pure"), Attribute.unit())
    set1.set(String("value"), Attribute.integer(42))

    var arr = List[String]()
    arr.push_back(String("\"x\""))
    arr.push_back(String("\"y\""))
    set1.set(String("tags"), Attribute.array(arr))

    var d = Attribute.dict()
    d.dict_v.set(String("rows"), String(3))
    d.dict_v.set(String("cols"), String(4))
    set1.set(String("layout"), d)

    var txt = set1.to_string()
    var ok = True
    if txt.find(String("sym_name=\"sum_mul\"")) < 0: ok = False
    if txt.find(String("pure=unit")) < 0: ok = False
    if txt.find(String("value=42")) < 0: ok = False
    if txt.find(String("tags=[\"x\", \"y\"]")) < 0: ok = False
    if txt.find(String("layout={rows=3, cols=4}")) < 0: ok = False

    # merge test
    var set2 = AnnotationSet()
    set2.set(String("value"), Attribute.integer(7))  # override
    set2.set(String("newkey"), Attribute.string(String("v")))
    var merged = set1.merge_right_biased(set2)
    var m = merged.to_string()
    if m.find(String("value=7")) < 0: ok = False
    if m.find(String("newkey=\"v\"")) < 0: ok = False

    if ok:
        print(String("OK"))
    else:
        print(String("FAIL"))
    return ok