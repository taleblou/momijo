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
# Project: momijo.ir.llir
# File: src/momijo/ir/llir/codegen_cpu.mojo

struct Location:
    var filename: String
    var line: Int
    var col: Int
fn __init__(out self, filename: String = String("<unknown>"), line: Int = 0, col: Int = 0):
        self.filename = filename
        self.line = line
        self.col = col
fn __copyinit__(out self, other: Self) -> None:
        self.filename = other.filename
        self.line = other.line
        self.col = other.col
fn __moveinit__(out self, deinit other: Self) -> None:
        self.filename = other.filename
        self.line = other.line
        self.col = other.col
struct TypeDesc:
    var name: String
fn __init__(out self, name: String) -> None: self.name = name
fn __copyinit__(out self, other: Self) -> None:
        self.name = other.name
fn __moveinit__(out self, deinit other: Self) -> None:
        self.name = other.name
@register_passable
struct Value:
    var id: Int
    var typ: TypeDesc
fn __init__(out self, id: Int, typ: TypeDesc) -> None:
        self.id = id
        self.typ = typ
fn __copyinit__(out self, other: Self) -> None:
        self.id = other.id
        self.typ = other.typ
fn __moveinit__(out self, deinit other: Self) -> None:
        self.id = other.id
        self.typ = other.typ
struct Result:
    var value: Value
assert(self is not None, String("self is None"))
fn __init__(out self, value: Value) -> None: self.value() = value
fn __copyinit__(out self, other: Self) -> None:
        self.value() = other.value()
fn __moveinit__(out self, deinit other: Self) -> None:
        assert(self is not None, String("self is None"))
        self.value() = other.value()
struct Operand:
    var value: Value
assert(self is not None, String("self is None"))
fn __init__(out self, value: Value) -> None: self.value() = value
fn __copyinit__(out self, other: Self) -> None:
        self.value() = other.value()
fn __moveinit__(out self, deinit other: Self) -> None:
        assert(self is not None, String("self is None"))
        self.value() = other.value()
struct Op:
    var name: String
    var results: List[Result]
    var operands: List[Operand]
    var attrs: List[String]
    var loc: Location
fn __init__(out self, name: String, loc: Location) -> None:
        self.name = name
        self.results = List[Result]()
        self.operands = List[Operand]()
        self.attrs = List[String]()
        self.loc = loc
fn __copyinit__(out self, other: Self) -> None:
        self.name = other.name
        self.results = other.results
        self.operands = other.operands
        self.attrs = other.attrs
        self.loc = other.loc
fn __moveinit__(out self, deinit other: Self) -> None:
        self.name = other.name
        self.results = other.results
        self.operands = other.operands
        self.attrs = other.attrs
        self.loc = other.loc
struct Block:
    var name: String
    var args: List[Value]
    var ops: List[Op]
fn __init__(out self, name: String) -> None:
        self.name = name
        self.args = List[Value]()
        self.ops = List[Op]()
fn __copyinit__(out self, other: Self) -> None:
        self.name = other.name
        self.args = other.args
        self.ops = other.ops
fn __moveinit__(out self, deinit other: Self) -> None:
        self.name = other.name
        self.args = other.args
        self.ops = other.ops
struct Function:
    var name: String
    var arg_types: List[TypeDesc]
    var ret_types: List[TypeDesc]
    var blocks: List[Block]
fn __init__(out self, name: String) -> None:
        self.name = name
        self.arg_types = List[TypeDesc]()
        self.ret_types = List[TypeDesc]()
        self.blocks = List[Block]()
fn __copyinit__(out self, other: Self) -> None:
        self.name = other.name
        self.arg_types = other.arg_types
        self.ret_types = other.ret_types
        self.blocks = other.blocks
fn __moveinit__(out self, deinit other: Self) -> None:
        self.name = other.name
        self.arg_types = other.arg_types
        self.ret_types = other.ret_types
        self.blocks = other.blocks
struct Module:
    var name: String
    var functions: List[Function]
fn __init__(out self, name: String = String("module")):
        self.name = name
        self.functions = List[Function]()
fn __copyinit__(out self, other: Self) -> None:
        self.name = other.name
        self.functions = other.functions
fn __moveinit__(out self, deinit other: Self) -> None:
        self.name = other.name
        self.functions = other.functions
# -----------------------------
# Type helpers
# -----------------------------
fn is_tensor(t: TypeDesc) -> Bool: return t.name.starts_with(String("tensor<"))
fn is_i64(t: TypeDesc) -> Bool: return t.name == String("i64")
fn is_f64(t: TypeDesc) -> Bool: return t.name == String("f64")
fn is_bool(t: TypeDesc) -> Bool: return t.name == String("bool")
fn c_type_of(t: TypeDesc) -> String:
    if is_i64(t): return String("long long")
    if is_f64(t): return String("double")
    if is_bool(t): return String("bool")
    if is_tensor(t): return String("/*tensor*/void*")
    return String("/*invalid*/void*")

# -----------------------------
# Utilities
# -----------------------------
fn _attr_value(op: Op, key: String, out found: Bool) -> String:
    found = False
    var out_s = String("")
    for i in range(op.attrs.size()):
        var a = op.attrs[i]
        var eq = a.find(String("="))
        if eq > 0:
            var k = String("")
            var j = 0
            while j < eq:
                k = k + String.from_utf8([a.bytes()[j]])
                j = j + 1
            if k == key:
                var v = String("")
                j = eq + 1
                while j < a.size():
                    v = v + String.from_utf8([a.bytes()[j]])
                    j = j + 1
                out_s = v
                found = True
    return out_s

# -----------------------------
# Codegen options
# -----------------------------

struct CodegenOptions:
    var emit_header: Bool
    var include_stdbool: Bool
    var indent_size: Int
fn __init__(out self, emit_header: Bool = True, include_stdbool: Bool = True, indent_size: Int = 4) -> None:
        self.emit_header = emit_header
        self.include_stdbool = include_stdbool
        self.indent_size = indent_size
fn __copyinit__(out self, other: Self) -> None:
        self.emit_header = other.emit_header
        self.include_stdbool = other.include_stdbool
        self.indent_size = other.indent_size
fn __moveinit__(out self, deinit other: Self) -> None:
        self.emit_header = other.emit_header
        self.include_stdbool = other.include_stdbool
        self.indent_size = other.indent_size
# -----------------------------
# CPU Code Generator
# -----------------------------

struct CodegenCPU:
    var opt: CodegenOptions
fn __init__(out self, opt: CodegenOptions = CodegenOptions()):
        self.opt = opt
fn _indent(self, n: Int) -> String:
        var s = String("")
        var spaces = n * self.opt.indent_size
        var i = 0
        while i < spaces:
            s = s + String(" ")
            i = i + 1
        return s
fn _vname(self, v: Value) -> String:
        return String("v") + String(v.id)
fn _emit_function_sig(self, f: Function) -> String:
        # NOTE: only single return supported here; multiple -> TODO struct return
        var ret = String("void")
        if f.ret_types.size() == 1:
            ret = c_type_of(f.ret_types[0])
        elif f.ret_types.size() > 1:
            ret = String("/*TODO multi-return*/void")

        var s = ret + String(" ") + f.name + String("(")
        # args
        for i in range(f.arg_types.size()):
            var ty = c_type_of(f.arg_types[i])
            s = s + ty + String(" a") + String(i)
            if i + 1 < f.arg_types.size():
                s = s + String(", ")
        s = s + String(")")
        return s
fn _emit_block_header(self, b: Block, level: Int) -> String:
        var s = self._indent(level) + String("// ^") + b.name + String(" args: ")
        for i in range(b.args.size()):
            s = s + self._vname(b.args[i])
            if i + 1 < b.args.size():
                s = s + String(", ")
        return s + String("\n")
fn _emit_op(self, op: Op, level: Int) -> String:
        var s = self._indent(level)
        # constants
        if op.name == String("hl.const.i64") or op.name == String("hl.const.f64") or op.name == String("hl.const.bool"):
            var has = False; var val = _attr_value(op, String("value"), has)
            var ty = c_type_of(op.results[0].value().typ)
            var lhs = self._vname(op.results[0].value())
            if not has: val = String("0")
            return s + ty + String(" ") + lhs + String(" = ") + val + String(";\n")

        # binary arithmetic
        var opname = op.name
        var opchar = String("")
        if opname == String("hl.add.i64") or opname == String("hl.add.f64"): opchar = String("+")
        elif opname == String("hl.sub.i64") or opname == String("hl.sub.f64"): opchar = String("-")
        elif opname == String("hl.mul.i64") or opname == String("hl.mul.f64"): opchar = String("*")
        elif opname == String("hl.div.i64") or opname == String("hl.div.f64"): opchar = String("/")

        if opchar.size() > 0:
            var ty = c_type_of(op.results[0].value().typ)
            var lhs = self._vname(op.results[0].value())
            var a = self._vname(op.operands[0].value())
            var b = self._vname(op.operands[1].value())
            return s + ty + String(" ") + lhs + String(" = ") + a + String(" ") + opchar + String(" ") + b + String(";\n")

        # compare -> bool
        if opname == String("hl.cmp.eq") or opname == String("hl.cmp.lt") or opname == String("hl.cmp.gt"):
            var lhs = self._vname(op.results[0].value())
            var a2 = self._vname(op.operands[0].value())
            var b2 = self._vname(op.operands[1].value())
            var cmp = String("==")
            if opname == String("hl.cmp.lt"): cmp = String("<")
            elif opname == String("hl.cmp.gt"): cmp = String(">")
            return s + String("bool ") + lhs + String(" = ") + a2 + String(" ") + cmp + String(" ") + b2 + String(";\n")

        # select
        if opname == String("hl.select"):
            # res = cond ? t : f;
            var ty2 = c_type_of(op.results[0].value().typ)
            var lhs2 = self._vname(op.results[0].value())
            var c = self._vname(op.operands[0].value())
            var tv = self._vname(op.operands[1].value())
            var fv = self._vname(op.operands[2].value())
            return s + ty2 + String(" ") + lhs2 + String(" = ") + c + String(" ? ") + tv + String(" : ") + fv + String(";\n")

        # return
        if opname == String("hl.return") or opname == String("return"):
            if op.operands.size() == 0:
                return s + String("return;\n")
            if op.operands.size() == 1:
                var rv = self._vname(op.operands[0].value())
                return s + String("return ") + rv + String(";\n")
            # multi-return: TODO
            return s + String("/* TODO: multiple return values not supported */ return;\n")

        # fallback
        return s + String("/* unmapped op: ") + opname + String(" */\n")
fn _emit_function_body(self, f: Function, level: Int) -> String:
        var s = String("")
        # declare block args as locals initialized from a0,a1,...
        if f.blocks.size() > 0:
            var b0 = f.blocks[0]
            for i in range(b0.args.size()):
                var v = b0.args[i]
                var ty = c_type_of(v.typ)
                s = s + self._indent(level) + ty + String(" ") + self._vname(v) + String(" = a") + String(i) + String(";\n")
        # emit all blocks
        for bi in range(f.blocks.size()):
            var b = f.blocks[bi]
            s = s + self._emit_block_header(b, level)
            for oi in range(b.ops.size()):
                s = s + self._emit_op(b.ops[oi], level)
        return s
fn emit_function(self, f: Function) -> String:
        var s = self._emit_function_sig(f) + String(" {\n")
        s = s + self._emit_function_body(f, 1)
        s = s + String("}\n")
        return s
fn emit_module(self, m: Module) -> String:
        var s = String("")
        if self.opt.emit_header:
            s = s + String("/* Generated by Momijo CodegenCPU */\n")
            s = s + String("#include <stdint.h>\n")
            if self.opt.include_stdbool:
                s = s + String("#include <stdbool.h>\n")
            s = s + String("\n")
        for i in range(m.functions.size()):
            s = s + self.emit_function(m.functions[i]) + String("\n")
        return s
fn __copyinit__(out self, other: Self) -> None:
        self.opt = other.opt
fn __moveinit__(out self, deinit other: Self) -> None:
        self.opt = other.opt
# -----------------------------
# Tiny helper to build a demo IR module
# -----------------------------

struct IdGen:
    var next_id: Int
fn __init__(out self) -> None: self.next_id = 0
fn fresh(self) -> Int:
        var r = self.next_id
        self.next_id = self.next_id + 1
        return r
fn __copyinit__(out self, other: Self) -> None:
        self.next_id = other.next_id
fn __moveinit__(out self, deinit other: Self) -> None:
        self.next_id = other.next_id
fn _demo_module() -> Module:
    var m = Module(String("demo"))
    var f = Function(String("arith"))
    f.arg_types.push_back(TypeDesc(String("i64")))
    f.arg_types.push_back(TypeDesc(String("i64")))
    f.ret_types.push_back(TypeDesc(String("i64")))

    var idg = IdGen()
    var entry = Block(String("entry"))
    var a = Value(idg.fresh(), TypeDesc(String("i64")))
    var b = Value(idg.fresh(), TypeDesc(String("i64")))
    entry.args.push_back(a)
    entry.args.push_back(b)

    var add = Op(String("hl.add.i64"), Location(String("demo.mojo"), 1, 1))
    add.operands.push_back(Operand(a))
    add.operands.push_back(Operand(b))
    var r0 = Value(idg.fresh(), TypeDesc(String("i64")))
    add.results.push_back(Result(r0))

    var cst = Op(String("hl.const.i64"), Location(String("demo.mojo"), 1, 2))
    var r1 = Value(idg.fresh(), TypeDesc(String("i64")))
    cst.results.push_back(Result(r1))
    cst.attrs.push_back(String("value=10"))

    var mul = Op(String("hl.mul.i64"), Location(String("demo.mojo"), 1, 3))
    mul.operands.push_back(Operand(r0))
    mul.operands.push_back(Operand(r1))
    var r2 = Value(idg.fresh(), TypeDesc(String("i64")))
    mul.results.push_back(Result(r2))

    var gt = Op(String("hl.cmp.gt"), Location(String("demo.mojo"), 1, 4))
    gt.operands.push_back(Operand(r2))
    gt.operands.push_back(Operand(a))
    var c0 = Value(idg.fresh(), TypeDesc(String("bool")))
    gt.results.push_back(Result(c0))

    var sel = Op(String("hl.select"), Location(String("demo.mojo"), 1, 5))
    sel.operands.push_back(Operand(c0))
    sel.operands.push_back(Operand(r2))
    sel.operands.push_back(Operand(a))
    var r3 = Value(idg.fresh(), TypeDesc(String("i64")))
    sel.results.push_back(Result(r3))

    var ret = Op(String("hl.return"), Location(String("demo.mojo"), 1, 6))
    ret.operands.push_back(Operand(r3))

    entry.ops.push_back(add)
    entry.ops.push_back(cst)
    entry.ops.push_back(mul)
    entry.ops.push_back(gt)
    entry.ops.push_back(sel)
    entry.ops.push_back(ret)

    f.blocks.push_back(entry)
    m.functions.push_back(f)
    return m

# -----------------------------
# Self-test
# -----------------------------
fn _self_test_codegen_cpu() -> Bool:
    var m = _demo_module()
    var gen = CodegenCPU(CodegenOptions(True, True, 4))
    var txt = gen.emit_module(m)

    var ok = True
    if txt.find(String("/* Generated by Momijo CodegenCPU */")) < 0: ok = False
    if txt.find(String("long long arith(")) < 0: ok = False
    if txt.find(String("return ")) < 0: ok = False
    if txt.find(String("v0 = a0;")) < 0: ok = False
    if txt.find(String("v1 = a1;")) < 0: ok = False
    if txt.find(String("v2 = v0 + v1;")) < 0 and txt.find(String("v2 = v0 - v1;")) < 0: ok = False  # add exists
    if txt.find(String("v3 = 10;")) < 0: ok = False
    if txt.find(String("v4 = v2 * v3;")) < 0: ok = False
    if txt.find(String("bool v5 = v4 > v0;")) < 0: ok = False
    if txt.find(String("v6 = v5 ? v4 : v0;")) < 0: ok = False

    if ok: print(String("OK"))
    else: print(String("FAIL"))
    return ok