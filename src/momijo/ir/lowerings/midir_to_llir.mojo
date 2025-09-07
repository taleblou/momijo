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
# Project: momijo.ir.lowerings
# File: src/momijo/ir/lowerings/midir_to_llir.mojo

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
# Helpers
# -----------------------------
fn is_tensor(t: TypeDesc) -> Bool: return t.name.starts_with(String("tensor<"))
fn is_i64(t: TypeDesc) -> Bool: return t.name == String("i64")
fn is_f64(t: TypeDesc) -> Bool: return t.name == String("f64")
fn is_bool(t: TypeDesc) -> Bool: return t.name == String("bool")
fn is_numeric(t: TypeDesc) -> Bool: return is_i64(t) or is_f64(t)

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
# -----------------------------
# Attribute helpers
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
# Lowering state
# -----------------------------

struct LoweringCtx:
    var idg: IdGen
    var vmap: Dict[Int, Value]  # midir Value.id -> llir Value
fn __init__(out self) -> None:
        self.idg = IdGen()
        self.vmap = Dict[Int, Value]()
fn map_value(self, v: Value) -> Value:
        if self.vmap.contains_key(v.id):
            return self.vmap[v.id]
        var nv = Value(self.idg.fresh(), v.typ)  # preserve type
        self.vmap[v.id] = nv
        return nv
fn __copyinit__(out self, other: Self) -> None:
        self.idg = other.idg
        self.vmap = other.vmap
fn __moveinit__(out self, deinit other: Self) -> None:
        self.idg = other.idg
        self.vmap = other.vmap
# -----------------------------

# -----------------------------
fn _map_bin(name: String) -> String:
    if name == String("mid.add"): return String("ll.add")
    if name == String("mid.sub"): return String("ll.sub")
    if name == String("mid.mul"): return String("ll.mul")
    if name == String("mid.div"): return String("ll.div")
    return String("")
fn _map_const(name: String) -> String:
    if name == String("mid.constant.i64"): return String("ll.const.i64")
    if name == String("mid.constant.f64"): return String("ll.const.f64")
    if name == String("mid.constant.bool"): return String("ll.const.bool")
    return String("")

fn _map_cmp_and_pred(op: Op, out target: String, out ok: Bool) -> None:
    ok = False
    var has = False
    var pred = _attr_value(op, String("pred"), has)
    if not has: return
    if pred == String("EQ"): target = String("ll.cmp.eq"); ok = True; return
    if pred == String("LT"): target = String("ll.cmp.lt"); ok = True; return
    if pred == String("GT"): target = String("ll.cmp.gt"); ok = True; return
    # unknown predicate
    target = String("ll.cmp.unknown")

# -----------------------------
# Lowerer
# -----------------------------

struct LowerMIDIRToLLIR:
    var ctx: LoweringCtx
fn __init__(out self) -> None:
        self.ctx = LoweringCtx()
fn lower_module(self, src: Module) -> Module:
        var dst = Module(src.name + String("_ll"))
        for fi in range(src.functions.size()):
            dst.functions.push_back(self.lower_function(src.functions[fi]))
        return dst
fn lower_function(self, f: Function) -> Function:
        var nf = Function(f.name)
        for i in range(f.arg_types.size()): nf.arg_types.push_back(f.arg_types[i])
        for i in range(f.ret_types.size()): nf.ret_types.push_back(f.ret_types[i])

        for bi in range(f.blocks.size()):
            var b = f.blocks[bi]
            var nb = Block(b.name)
            for ai in range(b.args.size()):
                var a = b.args[ai]
                var na = self.ctx.map_value(a)
                nb.args.push_back(na)
            for oi in range(b.ops.size()):
                var op = b.ops[oi]
                nb.ops.push_back(self.lower_op(op))
            nf.blocks.push_back(nb)
        return nf
fn _lower_results(self, op: Op, nop: Op) -> None:
        for i in range(op.results.size()):
            var rv = op.results[i].value()
            var new_v = self.ctx.map_value(rv)
            nop.results.push_back(Result(new_v))
fn _copy_operands(self, op: Op, nop: Op) -> None:
        for i in range(op.operands.size()):
            var ov = op.operands[i].value()
            var mv = self.ctx.map_value(ov)
            nop.operands.push_back(Operand(mv))
fn _copy_attrs(self, op: Op, nop: Op) -> None:
        for ai in range(op.attrs.size()):
            nop.attrs.push_back(op.attrs[ai])
fn lower_op(self, op: Op) -> Op:
        # constants
        var cn = _map_const(op.name)
        if cn.size() > 0:
            var nop = Op(cn, op.loc)
            self._lower_results(op, nop)
            # keep attributes (e.g., value=...)
            self._copy_attrs(op, nop)
            return nop

        # binary numeric
        var bn = _map_bin(op.name)
        if bn.size() > 0:
            var nop2 = Op(bn, op.loc)
            self._lower_results(op, nop2)
            self._copy_operands(op, nop2)
            self._copy_attrs(op, nop2)
            return nop2

        # compare
        var tgt = String("")
        var okp = False
        _map_cmp_and_pred(op, tgt, okp)
        if okp:
            var nop3 = Op(tgt, op.loc)
            self._lower_results(op, nop3)
            self._copy_operands(op, nop3)
            # drop pred attribute (encoded in name now)
            return nop3

        # select
        if op.name == String("mid.select"):
            var nop4 = Op(String("ll.select"), op.loc)
            self._lower_results(op, nop4)
            self._copy_operands(op, nop4)
            return nop4

        # ret
        if op.name == String("mid.ret"):
            var nop5 = Op(String("ll.ret"), op.loc)
            self._copy_operands(op, nop5)
            return nop5

        # pass-through: if already "mid." then rename to "ll."
        if op.name.starts_with(String("mid.")):
            var rest = String("")
            var i = 4
            while i < op.name.size():
                rest = rest + String.from_utf8([op.name.bytes()[i]])
                i = i + 1
            var nop6 = Op(String("ll.") + rest, op.loc)
            self._lower_results(op, nop6)
            self._copy_operands(op, nop6)
            self._copy_attrs(op, nop6)
            return nop6

        # otherwise clone
        var nop7 = Op(op.name, op.loc)
        self._lower_results(op, nop7)
        self._copy_operands(op, nop7)
        self._copy_attrs(op, nop7)
        return nop7
fn __copyinit__(out self, other: Self) -> None:
        self.ctx = other.ctx
fn __moveinit__(out self, deinit other: Self) -> None:
        self.ctx = other.ctx
# -----------------------------
# Tiny printer
# -----------------------------

struct Printer:
fn _value(self, v: Value) -> String:
        return String("%") + String(v.id) + String(":") + v.typ.name
fn print(self, m: Module) -> String:
        var s = String("module @") + m.name + String(" {\n")
        for i in range(m.functions.size()):
            var f = m.functions[i]
            s = s + String("  func @") + f.name + String("(")
            for ai in range(f.arg_types.size()):
                s = s + f.arg_types[ai].name
                if ai + 1 < f.arg_types.size(): s = s + String(", ")
            s = s + String(") -> (")
            for ri in range(f.ret_types.size()):
                s = s + f.ret_types[ri].name
                if ri + 1 < f.ret_types.size(): s = s + String(", ")
            s = s + String(") {\n")
            for bi in range(f.blocks.size()):
                var b = f.blocks[bi]
                s = s + String("    ^") + b.name + String("(")
                for aa in range(b.args.size()):
                    s = s + self._value(b.args[aa])
                    if aa + 1 < b.args.size(): s = s + String(", ")
                s = s + String("):\n")
                for oi in range(b.ops.size()):
                    var op = b.ops[oi]
                    s = s + String("      ") + op.name
                    if op.operands.size() > 0:
                        s = s + String("(")
                        for pi in range(op.operands.size()):
                            s = s + self._value(op.operands[pi].value())
                            if pi + 1 < op.operands.size(): s = s + String(", ")
                        s = s + String(")")
                    if op.results.size() > 0:
                        s = s + String(" -> ")
                        for ri2 in range(op.results.size()):
                            s = s + self._value(op.results[ri2].value())
                            if ri2 + 1 < op.results.size(): s = s + String(", ")
                    if op.attrs.size() > 0:
                        s = s + String(" {")
                        for ai in range(op.attrs.size()):
                            s = s + op.attrs[ai]
                            if ai + 1 < op.attrs.size(): s = s + String(", ")
                        s = s + String("}")
                    s = s + String("\n")
            s = s + String("  }\n")
        s = s + String("}\n")
        return s
fn __init__(out self, ) -> None:
        pass
fn __copyinit__(out self, other: Self) -> None:
        pass
fn __moveinit__(out self, deinit other: Self) -> None:
        pass
# -----------------------------
# Self-test
# -----------------------------
fn _make_demo_midir() -> Module:
    var m = Module(String("demo_midir"))
    var f = Function(String("arith"))
    f.arg_types.push_back(TypeDesc(String("i64")))
    f.arg_types.push_back(TypeDesc(String("i64")))
    f.ret_types.push_back(TypeDesc(String("i64")))

    var idg = IdGen()
    var b = Block(String("entry"))
    var a0 = Value(idg.fresh(), TypeDesc(String("i64")))
    var a1 = Value(idg.fresh(), TypeDesc(String("i64")))
    b.args.push_back(a0)
    b.args.push_back(a1)

    var add = Op(String("mid.add"), Location(String("demo.mojo"), 1, 1))
    add.operands.push_back(Operand(a0))
    add.operands.push_back(Operand(a1))
    var r0 = Value(idg.fresh(), TypeDesc(String("i64")))
    add.results.push_back(Result(r0))

    var cst = Op(String("mid.constant.i64"), Location(String("demo.mojo"), 1, 2))
    var r1 = Value(idg.fresh(), TypeDesc(String("i64")))
    cst.results.push_back(Result(r1))
    cst.attrs.push_back(String("value=5"))

    var cmp = Op(String("mid.cmp"), Location(String("demo.mojo"), 1, 3))
    cmp.operands.push_back(Operand(r0))
    cmp.operands.push_back(Operand(r1))
    var c0 = Value(idg.fresh(), TypeDesc(String("bool")))
    cmp.results.push_back(Result(c0))
    cmp.attrs.push_back(String("pred=GT"))

    var sel = Op(String("mid.select"), Location(String("demo.mojo"), 1, 4))
    sel.operands.push_back(Operand(c0))
    sel.operands.push_back(Operand(r0))
    sel.operands.push_back(Operand(a0))
    var r2 = Value(idg.fresh(), TypeDesc(String("i64")))
    sel.results.push_back(Result(r2))

    var ret = Op(String("mid.ret"), Location(String("demo.mojo"), 1, 5))
    ret.operands.push_back(Operand(r2))

    b.ops.push_back(add)
    b.ops.push_back(cst)
    b.ops.push_back(cmp)
    b.ops.push_back(sel)
    b.ops.push_back(ret)

    f.blocks.push_back(b)
    m.functions.push_back(f)
    return m
fn _self_test_midir_to_llir() -> Bool:
    var src = _make_demo_midir()
    var lower = LowerMIDIRToLLIR()
    var dst = lower.lower_module(src)

    var pr = Printer()
    var s2 = pr.print(dst)

    var ok = True
    if s2.find(String("ll.add")) < 0: ok = False
    if s2.find(String("ll.const.i64")) < 0: ok = False
    if s2.find(String("ll.cmp.gt")) < 0: ok = False
    if s2.find(String("ll.select")) < 0: ok = False
    if s2.find(String("ll.ret")) < 0: ok = False
    if ok: print(String("OK"))
    else: print(String("FAIL"))
    return ok