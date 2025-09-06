# ============================================================================
# MIT License
# Copyright (c) 2025 Morteza Talebou and Mitra Daneshmand
# Project: momijo  |  Source: https://github.com/taleblou/momijo
# This file is part of the Momijo project. See the LICENSE file at the repository root.
# Momijo
# SPDX-License-Identifier: MIT
# Website: https://taleblou.ir/
# Repository: https://github.com/taleblou/momijo
#
# Project: momijo.ir
# File: src/momijo/ir/printer.mojo
# Description: Pretty-printer for Momijo IR with configurable options.
# Notes:
#   - No globals, no 'export'; only 'var' (no 'let').
#   - Constructors: fn __init__(out self, ...)
#   - This file intentionally avoids external imports to reduce coupling.
#   - Includes a tiny fallback IR model so the printer is self-testable.
# ============================================================================

# -----------------------------
# Fallback IR "view" (for self-test)
# If you already have real IR types, you can remove this block and import them.
# Expected minimal interfaces:
#   Module { name: String, functions: List[Function] }
#   Function { name: String, arg_types: List[TypeDesc], ret_types: List[TypeDesc], blocks: List[Block] }
#   Block { name: String, args: List[Value], ops: List[Op] }
#   Op { name: String, results: List[Result], operands: List[Operand], attrs: List[String], loc: Location }
#   Value { id: Int, typ: TypeDesc }
#   TypeDesc { name: String }
#   Location { filename: String, line: Int, col: Int }
# -----------------------------

struct Location:
    var filename: String
    var line: Int
    var col: Int
    fn __init__(out self, filename: String = String("<unknown>"), line: Int = 0, col: Int = 0):
        self.filename = filename
        self.line = line
        self.col = col

struct TypeDesc:
    var name: String
    fn __init__(out self, name: String): self.name = name

@register_passable
struct Value:
    var id: Int
    var typ: TypeDesc
    fn __init__(out self, id: Int, typ: TypeDesc):
        self.id = id
        self.typ = typ

struct Result:
    var value: Value
    fn __init__(out self, value: Value): self.value = value

struct Operand:
    var value: Value
    fn __init__(out self, value: Value): self.value = value

struct Op:
    var name: String
    var results: List[Result]
    var operands: List[Operand]
    var attrs: List[String]
    var loc: Location
    fn __init__(out self, name: String, loc: Location):
        self.name = name
        self.results = List[Result]()
        self.operands = List[Operand]()
        self.attrs = List[String]()
        self.loc = loc

struct Block:
    var name: String
    var args: List[Value]
    var ops: List[Op]
    fn __init__(out self, name: String):
        self.name = name
        self.args = List[Value]()
        self.ops = List[Op]()

struct Function:
    var name: String
    var arg_types: List[TypeDesc]
    var ret_types: List[TypeDesc]
    var blocks: List[Block]
    fn __init__(out self, name: String):
        self.name = name
        self.arg_types = List[TypeDesc]()
        self.ret_types = List[TypeDesc]()
        self.blocks = List[Block]()

struct Module:
    var name: String
    var functions: List[Function]
    fn __init__(out self, name: String = String("module")):
        self.name = name
        self.functions = List[Function]()

# -----------------------------
# Printer
# -----------------------------

struct PrintOptions:
    var indent_size: Int
    var show_attrs: Bool
    var show_locs: Bool

    fn __init__(out self, indent_size: Int = 2, show_attrs: Bool = True, show_locs: Bool = True):
        self.indent_size = indent_size
        self.show_attrs = show_attrs
        self.show_locs = show_locs

struct Printer:
    var opt: PrintOptions

    fn __init__(out self, opt: PrintOptions = PrintOptions()):
        self.opt = opt

    fn _indent(self, level: Int) -> String:
        var s = String("")
        var spaces = self.opt.indent_size * level
        var i = 0
        while i < spaces:
            s = s + String(" ")
            i = i + 1
        return s

    fn _type_list(self, tys: List[TypeDesc]) -> String:
        var s = String("")
        for i in range(tys.size()):
            s = s + tys[i].name
            if i + 1 < tys.size():
                s = s + String(", ")
        return s

    fn _value(self, v: Value) -> String:
        return String("%") + String(v.id) + String(":") + v.typ.name

    fn _values_csv(self, vals: List[Value]) -> String:
        var s = String("")
        for i in range(vals.size()):
            s = s + self._value(vals[i])
            if i + 1 < vals.size():
                s = s + String(", ")
        return s

    fn print_module(self, m: Module) -> String:
        var s = String("module @") + m.name + String(" {\n")
        for i in range(m.functions.size()):
            s = s + self.print_function(m.functions[i], 1) + String("\n")
        s = s + String("}\n")
        return s

    fn print_function(self, f: Function, level: Int) -> String:
        var s = self._indent(level)
        s = s + String("func @") + f.name + String(" (") + self._type_list(f.arg_types) + String(") -> (") + self._type_list(f.ret_types) + String(") {\n")
        for i in range(f.blocks.size()):
            s = s + self.print_block(f.blocks[i], level + 1)
        s = s + self._indent(level) + String("}\n")
        return s

    fn print_block(self, b: Block, level: Int) -> String:
        var s = self._indent(level)
        s = s + String("^") + b.name + String("(") + self._values_csv(b.args) + String("):\n")
        for i in range(b.ops.size()):
            s = s + self.print_op(b.ops[i], level + 1) + String("\n")
        return s

    fn print_op(self, op: Op, level: Int) -> String:
        var s = self._indent(level)
        if op.results.size() > 0:
            for i in range(op.results.size()):
                s = s + self._value(op.results[i].value)
                if i + 1 < op.results.size():
                    s = s + String(", ")
                else:
                    s = s + String(" = ")
        s = s + op.name + String("(")
        for i in range(op.operands.size()):
            s = s + self._value(op.operands[i].value)
            if i + 1 < op.operands.size():
                s = s + String(", ")
        s = s + String(")")
        if self.opt.show_attrs and op.attrs.size() > 0:
            s = s + String(" {")
            for i in range(op.attrs.size()):
                s = s + op.attrs[i]
                if i + 1 < op.attrs.size():
                    s = s + String(", ")
            s = s + String("}")
        if self.opt.show_locs:
            s = s + String("  // ") + op.loc.filename + String(":") + String(op.loc.line) + String(":") + String(op.loc.col)
        return s

# -----------------------------
# Self-test: build a tiny module and print it
# -----------------------------

fn _self_test_printer() -> Bool:
    var m = Module(String("demo"))
    var f = Function(String("sum_mul"))
    f.arg_types.push_back(TypeDesc(String("i64")))
    f.arg_types.push_back(TypeDesc(String("i64")))
    f.ret_types.push_back(TypeDesc(String("i64")))

    var entry = Block(String("entry"))
    var a = Value(0, TypeDesc(String("i64")))
    var b = Value(1, TypeDesc(String("i64")))
    entry.args.push_back(a)
    entry.args.push_back(b)

    var op0 = Op(String("add.i64"), Location(String("demo.mojo"), 1, 1))
    op0.results.push_back(Result(Value(2, TypeDesc(String("i64")))))
    op0.operands.push_back(Operand(a))
    op0.operands.push_back(Operand(b))

    var op1 = Op(String("const.i64"), Location(String("demo.mojo"), 1, 2))
    op1.results.push_back(Result(Value(3, TypeDesc(String("i64")))))
    op1.attrs.push_back(String("value=2"))

    var op2 = Op(String("mul.i64"), Location(String("demo.mojo"), 1, 3))
    op2.results.push_back(Result(Value(4, TypeDesc(String("i64")))))
    op2.operands.push_back(Operand(Value(2, TypeDesc(String("i64")))))
    op2.operands.push_back(Operand(Value(3, TypeDesc(String("i64")))))

    var ret = Op(String("return"), Location(String("demo.mojo"), 1, 4))
    ret.operands.push_back(Operand(Value(4, TypeDesc(String("i64")))))

    entry.ops.push_back(op0)
    entry.ops.push_back(op1)
    entry.ops.push_back(op2)
    entry.ops.push_back(ret)

    f.blocks.push_back(entry)
    m.functions.push_back(f)

    var pr = Printer(PrintOptions(2, True, True))
    var txt = pr.print_module(m)

    var ok = True
    if txt.find(String("module @demo")) < 0: ok = False
    if txt.find(String("func @sum_mul")) < 0: ok = False
    if txt.find(String("^entry(%0:i64, %1:i64)")) < 0: ok = False
    if txt.find(String("add.i64")) < 0: ok = False
    if txt.find(String("return")) < 0: ok = False

    if ok:
        # Avoid printing complex structs; print a scalar string marker.
        print(String("OK"))
    else:
        print(String("FAIL"))
    return ok
 
