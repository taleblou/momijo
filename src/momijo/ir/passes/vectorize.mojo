# Project:      Momijo
# Module:       src.momijo.ir.passes.vectorize
# File:         vectorize.mojo
# Path:         src/momijo/ir/passes/vectorize.mojo
#
# Description:  src.momijo.ir.passes.vectorize — focused Momijo functionality with a stable public API.
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
fn _has_attr(op: Op, key: String) -> Bool:
    var has = False; var _ = _attr_value(op, key, has); return has
fn _set_attr(op: Op, key: String, val: String) -> Op:
    # append new key=value; simple implementation (not deduping)
    op.attrs.push_back(key + String("=") + val)
    return op
fn _replace_or_add_attr(op: Op, key: String, val: String) -> Op:
    var replaced = False
    var new_attrs = List[String]()
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
                new_attrs.push_back(key + String("=") + val)
                replaced = True
            else:
                new_attrs.push_back(a)
        else:
            new_attrs.push_back(a)
    if not replaced:
        new_attrs.push_back(key + String("=") + val)
    op.attrs = new_attrs
    return op
fn _parse_i64(s: String, out ok: Bool) -> Int:
    ok = False
    if s.size() == 0: return 0
    var sign = 1
    var i = 0
    if s.bytes()[0] == 45: sign = -1; i = 1
    var num = 0
    while i < s.size():
        var ch = s.bytes()[i]
        if ch < 48 or ch > 57: return 0
        num = num * 10 + Int(ch - 48)
        i = i + 1
    ok = True
    return sign * num
fn _vec_kind(name: String, out base: String, out dtype: String, out ok: Bool) -> None:
    ok = False
    base = String(""); dtype = String("")
    if not name.starts_with(String("vec.")): return
    var i = name.find(String("."))
    if i < 0: return
    var j = name.find(String("."), i+1)
    if j < 0: return
    base = String("")
    var k = i + 1
    while k < j:
        base = base + String.from_utf8([name.bytes()[k]]); k = k + 1
    dtype = String("")
    k = j + 1
    while k < name.size():
        dtype = dtype + String.from_utf8([name.bytes()[k]]); k = k + 1
    if base == String("add") or base == String("sub") or base == String("mul") or base == String("div") or base == String("fused"):
        if dtype == String("i64") or dtype == String("f64"): ok = True
fn _is_vec_target(op: Op) -> Bool:
    var b = String(""); var d = String(""); var ok = False
    _vec_kind(op.name, b, d, ok)
    return ok

# -----------------------------
# Vectorization pass
# -----------------------------

struct VectorizeOptions:
    var preferred_width: Int
fn __init__(out self) -> None:
        self.preferred_width = 4
fn __copyinit__(out self, other: Self) -> None:
        self.preferred_width = other.preferred_width
fn __moveinit__(out self, deinit other: Self) -> None:
        self.preferred_width = other.preferred_width
struct Vectorize:
    var opt: VectorizeOptions
fn __init__(out self, opt: VectorizeOptions = VectorizeOptions()):
        self.opt = opt
fn _choose_width(self, dtype: String, N: Int) -> Int:

        if dtype == String("i64") or dtype == String("f64"):
            if N >= self.opt.preferred_width: return self.opt.preferred_width
        return 1
fn _clone_op_with_new_N(self, op: Op, newN: Int) -> Op:
        var c = Op(op.name, op.loc)
        # copy operands/results
        for i in range(op.operands.size()): c.operands.push_back(op.operands[i])
        for i in range(op.results.size()): c.results.push_back(op.results[i])
        # copy attrs but replace N
        var hasN = False; var _ = _attr_value(op, String("N"), hasN)
        if hasN:
            c = _replace_or_add_attr(c, String("N"), String(newN))
        else:
            c = _set_attr(c, String("N"), String(newN))
        # copy the rest naively
        for i in range(op.attrs.size()):
            var a = op.attrs[i]
            if a.starts_with(String("N=")): continue
            c.attrs.push_back(a)
        return c
fn _split_for_tail(self, b: Block, idx: Int, vecW: Int, dtype: String, N: Int) -> None:
        if vecW <= 1: 
            # set vector=1 and done
            var op = b.ops[idx]
            if not _has_attr(op, String("vector")):
                op = _set_attr(op, String("vector"), String(1))
            b.ops[idx] = op
            var note = Op(String("vectorize.note"), op.loc)
            note.attrs.push_back(String("split=false"))
            note.attrs.push_back(String("width=1"))
            b.ops.insert(idx+1, note)
            return

        var mainN = (N / vecW) * vecW
        var tailN = N - mainN

        if tailN == 0:
            # just annotate
            var op0 = b.ops[idx]
            op0 = _replace_or_add_attr(op0, String("vector"), String(vecW))
            b.ops[idx] = op0
            var note0 = Op(String("vectorize.note"), op0.loc)
            note0.attrs.push_back(String("split=false"))
            note0.attrs.push_back(String("width=") + String(vecW))
            b.ops.insert(idx+1, note0)
            return

        # Create main op
        var main = self._clone_op_with_new_N(b.ops[idx], mainN)
        main = _replace_or_add_attr(main, String("vector"), String(vecW))
        main.attrs.push_back(String("tail=false"))

        # Create tail op (scalar)
        var tail = self._clone_op_with_new_N(b.ops[idx], tailN)
        tail = _replace_or_add_attr(tail, String("vector"), String(1))
        tail.attrs.push_back(String("offset=") + String(mainN))
        tail.attrs.push_back(String("tail=true"))

        # Replace current op by main, then insert tail and note
        b.ops[idx] = main
        b.ops.insert(idx+1, tail)

        var note = Op(String("vectorize.note"), main.loc)
        note.attrs.push_back(String("split=true"))
        note.attrs.push_back(String("width=") + String(vecW))
        note.attrs.push_back(String("tail=") + String(tailN))
        b.ops.insert(idx+2, note)
fn run_on_block(self, b: Block) -> Block:
        var i = 0
        while i < b.ops.size():
            var op = b.ops[i]
            if _is_vec_target(op):
                # parse attrs
                var hasN = False; var sN = _attr_value(op, String("N"), hasN)
                if hasN:
                    var ok = False; var N = _parse_i64(sN, ok)
                    if ok and N > 0:
                        var base = String(""); var dtype = String(""); var ok2 = False
                        _vec_kind(op.name, base, dtype, ok2)
                        if ok2:
                            # respect existing vector attr
                            var hasV = False; var Vtxt = _attr_value(op, String("vector"), hasV)
                            var vecW = 0
                            if hasV:
                                var okv = False; vecW = _parse_i64(Vtxt, okv); if not okv: vecW = 1
                            else:
                                vecW = self._choose_width(dtype, N)
                            self._split_for_tail(b, i, vecW, dtype, N)
                            # move i past inserted note
                            i = i + 3 if (N % (vecW if vecW>0 else 1) != 0 and vecW > 1) else i + 2
                            continue
            i = i + 1
        return b
fn run_on_function(self, f: Function) -> Function:
        for bi in range(f.blocks.size()):
            f.blocks[bi] = self.run_on_block(f.blocks[bi])
        return f
fn run_on_module(self, m: Module) -> Module:
        var out = Module(m.name + String("_vect"))
        for fi in range(m.functions.size()):
            var f = m.functions[fi]
            var nf = Function(f.name)
            for ai in range(f.arg_types.size()): nf.arg_types.push_back(f.arg_types[ai])
            for ri in range(f.ret_types.size()): nf.ret_types.push_back(f.ret_types[ri])
            for bi in range(f.blocks.size()): nf.blocks.push_back(f.blocks[bi])
            out.functions.push_back(self.run_on_function(nf))
        return out
fn __copyinit__(out self, other: Self) -> None:
        self.opt = other.opt
fn __moveinit__(out self, deinit other: Self) -> None:
        self.opt = other.opt
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
fn _buffer_i64_1d(n: Int) -> TypeDesc: return TypeDesc(String("buffer<i64,[") + String(n) + String("]>"))

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
    var m = Module(String("demo_vect"))
    var f = Function(String("main"))
    var b = Block(String("entry"))
    var idg = IdGen()

    # buffers
    var outb = Value(idg.fresh(), _buffer_i64_1d(32))
    var a = Value(idg.fresh(), _buffer_i64_1d(32))
    var c = Value(idg.fresh(), _buffer_i64_1d(32))
    b.args.push_back(outb); b.args.push_back(a); b.args.push_back(c)

    var add1 = Op(String("vec.add.i64"), Location(String("d"), 1, 1))
    add1.operands.push_back(Operand(outb)); add1.operands.push_back(Operand(a)); add1.operands.push_back(Operand(c))
    add1.attrs.push_back(String("N=16"))
    b.ops.push_back(add1)

    var out2 = Value(idg.fresh(), _buffer_i64_1d(32))
    var a2 = Value(idg.fresh(), _buffer_i64_1d(32))
    var b2 = Value(idg.fresh(), _buffer_i64_1d(32))
    var c2 = Value(idg.fresh(), _buffer_i64_1d(32))
    b.args.push_back(out2); b.args.push_back(a2); b.args.push_back(b2); b.args.push_back(c2)
    var fused = Op(String("vec.fused.i64"), Location(String("d"), 1, 2))
    fused.operands.push_back(Operand(out2)); fused.operands.push_back(Operand(a2)); fused.operands.push_back(Operand(b2)); fused.operands.push_back(Operand(c2))
    fused.attrs.push_back(String("N=18")); fused.attrs.push_back(String("arity=3")); fused.attrs.push_back(String("expr_rpn=x0 x1 add x2 mul"))
    b.ops.push_back(fused)

    var ret = Op(String("ll.ret"), Location(String("d"), 1, 3)); b.ops.push_back(ret)

    f.blocks.push_back(b); m.functions.push_back(f)
    return m
fn _self_test_vectorize() -> Bool:
    var m = _demo_module()
    var opt = VectorizeOptions()
    opt.preferred_width = 4
    var pass = Vectorize(opt)
    var out = pass.run_on_module(m)

    var pr = Printer()
    var txt = pr.print(out)

    var ok = True
    # Case 1: expect vector=4 and split=false note
    if txt.find(String("vec.add.i64")) < 0: ok = False
    if txt.find(String("N=16")) < 0: ok = False
    if txt.find(String("vector=4")) < 0: ok = False
    if txt.find(String("vectorize.note {split=false, width=4}")) < 0:
        if txt.find(String("split=false")) < 0 or txt.find(String("width=4")) < 0: ok = False

    # Case 2: expect two vec.fused.i64 with N=16(vector=4,tail=false) and N=2(vector=1,offset=16,tail=true)
    var pos_fused = txt.find(String("vec.fused.i64"))
    if pos_fused < 0: ok = False
    var pos_second = txt.find(String("vec.fused.i64"), pos_fused + 1)
    if pos_second < 0: ok = False
    if txt.find(String("N=16")) < 0 or txt.find(String("N=2")) < 0: ok = False
    if txt.find(String("offset=16")) < 0: ok = False
    if txt.find(String("tail=true")) < 0 or txt.find(String("tail=false")) < 0: ok = False

    if ok: print(String("OK"))
    else: print(String("FAIL"))
    return ok