# Project:      Momijo
# Module:       src.momijo.ir.lowerings.hlir_to_midir
# File:         hlir_to_midir.mojo
# Path:         src/momijo/ir/lowerings/hlir_to_midir.mojo
#
# Description:  src.momijo.ir.lowerings.hlir_to_midir — focused Momijo functionality with a stable public API.
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
#   - Structs: Location, TypeDesc, Value, Result, Operand, Op, Block, Function
#   - Key functions: __init__, __copyinit__, __moveinit__, __init__, __copyinit__, __moveinit__, __init__, __copyinit__ ...
#   - GPU/device utilities present; validate backend assumptions.


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

# -----------------------------
fn _map_bin(name: String) -> String:
    if name == String("hl.add.i64") or name == String("hl.add.f64"): return String("mid.add")
    if name == String("hl.sub.i64") or name == String("hl.sub.f64"): return String("mid.sub")
    if name == String("hl.mul.i64") or name == String("hl.mul.f64"): return String("mid.mul")
    if name == String("hl.div.i64") or name == String("hl.div.f64"): return String("mid.div")
    return String("")
fn _map_const(name: String) -> String:
    if name == String("hl.const.i64"): return String("mid.constant.i64")
    if name == String("hl.const.f64"): return String("mid.constant.f64")
    if name == String("hl.const.bool"): return String("mid.constant.bool")
    return String("")
fn _map_cmp(name: String, out pred: String, out ok: Bool) -> String:
    ok = True
    if name == String("hl.cmp.eq"): pred = String("EQ"); return String("mid.cmp")
    if name == String("hl.cmp.lt"): pred = String("LT"); return String("mid.cmp")
    if name == String("hl.cmp.gt"): pred = String("GT"); return String("mid.cmp")
    ok = False
    pred = String("")
    return String("")

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
    var vmap: Dict[Int, Value]  # hlir Value.id -> midir Value
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
# Lowerer
# -----------------------------

struct LowerHLIRToMIDIR:
    var ctx: LoweringCtx
fn __init__(out self) -> None:
        self.ctx = LoweringCtx()
fn lower_module(self, src: Module) -> Module:
        var dst = Module(src.name + String("_mid"))
        for fi in range(src.functions.size()):
            dst.functions.push_back(self.lower_function(src.functions[fi]))
        return dst
fn lower_function(self, f: Function) -> Function:
        var nf = Function(f.name)
        # copy signature
        for i in range(f.arg_types.size()):
            nf.arg_types.push_back(f.arg_types[i])
        for i in range(f.ret_types.size()):
            nf.ret_types.push_back(f.ret_types[i])

        # blocks
        for bi in range(f.blocks.size()):
            var b = f.blocks[bi]
            var nb = Block(b.name)
            # map block args
            for ai in range(b.args.size()):
                var a = b.args[ai]
                var na = self.ctx.map_value(a)
                nb.args.push_back(na)
            # lower ops
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
fn lower_op(self, op: Op) -> Op:
        # constants
        var cn = _map_const(op.name)
        if cn.size() > 0:
            var nop = Op(cn, op.loc)
            self._lower_results(op, nop)
            # keep value attr intact if present
            var has = False
            var v = _attr_value(op, String("value"), has)
            if has:
                nop.attrs.push_back(String("value=") + v)
            return nop

        # binary numeric
        var bn = _map_bin(op.name)
        if bn.size() > 0:
            var nop2 = Op(bn, op.loc)
            self._lower_results(op, nop2)
            self._copy_operands(op, nop2)
            return nop2

        var pred = String("")
        var okp = False
        var cnm = _map_cmp(op.name, pred, okp)
        if okp:
            var nop3 = Op(cnm, op.loc)
            self._lower_results(op, nop3)
            self._copy_operands(op, nop3)
            nop3.attrs.push_back(String("pred=") + pred)
            return nop3

        # select
        if op.name == String("hl.select"):
            var nop4 = Op(String("mid.select"), op.loc)
            self._lower_results(op, nop4)
            self._copy_operands(op, nop4)
            return nop4

        # return
        if op.name == String("hl.return") or op.name == String("return"):
            var nop5 = Op(String("mid.ret"), op.loc)
            self._copy_operands(op, nop5)
            return nop5

        # pass through unknowns with prefix swap if they already "hl."
        if op.name.starts_with(String("hl.")):
            var rest = String("")
            var i = 3
            while i < op.name.size():
                rest = rest + String.from_utf8([op.name.bytes()[i]])
                i = i + 1
            var nop6 = Op(String("mid.") + rest, op.loc)
            self._lower_results(op, nop6)
            self._copy_operands(op, nop6)
            # copy attrs
            for ai in range(op.attrs.size()):
                nop6.attrs.push_back(op.attrs[ai])
            return nop6

        # otherwise clone (no rename)
        var nop7 = Op(op.name, op.loc)
        self._lower_results(op, nop7)
        self._copy_operands(op, nop7)
        for ai in range(op.attrs.size()):
            nop7.attrs.push_back(op.attrs[ai])
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
fn _make_demo_hlir() -> Module:
    var m = Module(String("demo_hlir"))
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

    var add = Op(String("hl.add.i64"), Location(String("demo.mojo"), 1, 1))
    add.operands.push_back(Operand(a0))
    add.operands.push_back(Operand(a1))
    var r0 = Value(idg.fresh(), TypeDesc(String("i64")))
    add.results.push_back(Result(r0))

    var cst = Op(String("hl.const.i64"), Location(String("demo.mojo"), 1, 2))
    var r1 = Value(idg.fresh(), TypeDesc(String("i64")))
    cst.results.push_back(Result(r1))
    cst.attrs.push_back(String("value=5"))

    var cmp = Op(String("hl.cmp.gt"), Location(String("demo.mojo"), 1, 3))
    cmp.operands.push_back(Operand(r0))
    cmp.operands.push_back(Operand(r1))
    var c0 = Value(idg.fresh(), TypeDesc(String("bool")))
    cmp.results.push_back(Result(c0))

    var sel = Op(String("hl.select"), Location(String("demo.mojo"), 1, 4))
    sel.operands.push_back(Operand(c0))
    sel.operands.push_back(Operand(r0))
    sel.operands.push_back(Operand(a0))
    var r2 = Value(idg.fresh(), TypeDesc(String("i64")))
    sel.results.push_back(Result(r2))

    var ret = Op(String("hl.return"), Location(String("demo.mojo"), 1, 5))
    ret.operands.push_back(Operand(r2))

    b.ops.push_back(add)
    b.ops.push_back(cst)
    b.ops.push_back(cmp)
    b.ops.push_back(sel)
    b.ops.push_back(ret)

    f.blocks.push_back(b)
    m.functions.push_back(f)
    return m
fn _self_test_hlir_to_midir() -> Bool:
    var src = _make_demo_hlir()
    var lower = LowerHLIRToMIDIR()
    var dst = lower.lower_module(src)

    var pr = Printer()
    var s1 = pr.print(src)
    var s2 = pr.print(dst)

    var ok = True
    if s2.find(String("mid.add")) < 0: ok = False
    if s2.find(String("mid.constant.i64")) < 0: ok = False
    if s2.find(String("mid.cmp")) < 0: ok = False
    if s2.find(String("pred=GT")) < 0: ok = False
    if s2.find(String("mid.select")) < 0: ok = False
    if s2.find(String("mid.ret")) < 0: ok = False
    if ok: print(String("OK"))
    else: print(String("FAIL"))
    return ok