# Project:      Momijo
# Module:       src.momijo.ir.passes.memory_coalesce
# File:         memory_coalesce.mojo
# Path:         src/momijo/ir/passes/memory_coalesce.mojo
#
# Description:  src.momijo.ir.passes.memory_coalesce — focused Momijo functionality with a stable public API.
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
# Parse buffer type like buffer<i64,[...]>
struct ShapeInfo:
    var dtype: String
fn __init__(out self) -> None: self.dtype = String("")
fn __copyinit__(out self, other: Self) -> None:
        self.dtype = other.dtype
fn __moveinit__(out self, deinit other: Self) -> None:
        self.dtype = other.dtype
fn _parse_buffer_dtype(t: TypeDesc, out info: ShapeInfo, out ok: Bool) -> None:
    ok = False
    info = ShapeInfo()
    var s = t.name
    if not s.starts_with(String("buffer<")) and not s.starts_with(String("tensor<")): return
    var lt = s.find(String("<")); if lt < 0: return
    var comma = s.find(String(","), lt+1); if comma < 0: return
    var dt = String("")
    var i = lt + 1
    while i < comma:
        dt = dt + String.from_utf8([s.bytes()[i]]); i = i + 1
    info.dtype = dt
    ok = True

# -----------------------------
# Memory coalescer
# -----------------------------

struct MemoryCoalesce:
fn _find_const_value(self, b: Block, val: Value, out found: Bool) -> Int:
        # Search for an op producing 'val' with name ll.const.i64 {value=...}
        found = False
        var i = 0
        while i < b.ops.size():
            var op = b.ops[i]
            if op.name == String("ll.const.i64"):
                for r in range(op.results.size()):
                    if op.results[r].value().id == val.id:
                        var has = False; var s = _attr_value(op, String("value"), has)
                        if has:
                            var ok = False; var v = _parse_i64(s, ok)
                            if ok:
                                found = True
                                return v
            i = i + 1
        return 0
fn _try_coalesce_copy_loop(self, b: Block, for_idx: Int) -> Bool:
        # Pattern match: loop.for, buf.load, buf.store, loop.end
        if for_idx + 3 >= b.ops.size(): return False
        var op_for = b.ops[for_idx]
        if op_for.name != String("loop.for"): return False
        var op1 = b.ops[for_idx + 1]
        var op2 = b.ops[for_idx + 2]
        var op_end = b.ops[for_idx + 3]
        if op_end.name != String("loop.end"): return False
        if op1.name != String("buf.load"): return False
        if op2.name != String("buf.store"): return False

        # loop.for operands: (lo, hi, step); result: iv
        if op_for.operands.size() < 3 or op_for.results.size() < 1: return False
        var lo = op_for.operands[0].value()
        var hi = op_for.operands[1].value()
        var step = op_for.operands[2].value()
        var iv = op_for.results[0].value()

        # load(src, iv)
        if op1.operands.size() < 2: return False
        var src = op1.operands[0].value()
        var idx1 = op1.operands[1].value()
        if idx1.id != iv.id: return False

        # store(val, dst, iv)
        if op2.operands.size() < 3: return False
        var v = op2.operands[0].value()
        var dst = op2.operands[1].value()
        var idx2 = op2.operands[2].value()
        if idx2.id != iv.id: return False

        # Ensure 'v' is the result of op1
        if op1.results.size() == 0 or op1.results[0].value().id != v.id: return False

        # Extract integer lo, hi, step
        var has_lo = False; var lo_i = self._find_const_value(b, lo, has_lo)
        var has_hi = False; var hi_i = self._find_const_value(b, hi, has_hi)
        var has_st = False; var st_i = self._find_const_value(b, step, has_st)
        if not has_lo or not has_hi or not has_st: return False
        if st_i != 1: return False
        if hi_i < lo_i: return False
        var length = hi_i - lo_i

        # Build memcpy op
        var memcpy = Op(String("buf.memcpy"), op_for.loc)
        memcpy.operands.push_back(Operand(dst))
        memcpy.operands.push_back(Operand(src))
        memcpy.attrs.push_back(String("len=") + String(length))
        memcpy.attrs.push_back(String("coalesce=true"))

        # element dtype
        var info = ShapeInfo(); var ok = False
        _parse_buffer_dtype(src.typ, info, ok)
        if ok and info.dtype.size() > 0:
            memcpy.attrs.push_back(String("elem=") + info.dtype)

        # width heuristic
        if length % 4 == 0:
            memcpy.attrs.push_back(String("coalesce.width=4"))
        else:
            memcpy.attrs.push_back(String("coalesce.width=1"))

        # Replace range [for_idx .. for_idx+3] with memcpy + note
        var note = Op(String("coalesce.removed_loop"), op_end.loc)
        note.attrs.push_back(String("from=loop.for"))
        note.attrs.push_back(String("len=") + String(length))

        var new_ops = List[Op]()
        for i in range(b.ops.size()):
            if i == for_idx:
                new_ops.push_back(memcpy)
                new_ops.push_back(note)
            elif i > for_idx and i <= for_idx + 3:
                # skip load/store/end
                continue
            else:
                new_ops.push_back(b.ops[i])
        b.ops = new_ops
        return True
fn run_on_block(self, b: Block) -> Block:
        var i = 0
        while i < b.ops.size():
            if b.ops[i].name == String("loop.for"):
                # try to coalesce; if success, do not advance i (block changed)
                var ok = self._try_coalesce_copy_loop(b, i)
                if ok:
                    # after replacement, next op after memcpy is note; move past it
                    i = i + 2
                    continue
            i = i + 1
        return b
fn run_on_function(self, f: Function) -> Function:
        for bi in range(f.blocks.size()):
            f.blocks[bi] = self.run_on_block(f.blocks[bi])
        return f
fn run_on_module(self, m: Module) -> Module:
        var out = Module(m.name + String("_coalesce"))
        for fi in range(m.functions.size()):
            var f = m.functions[fi]
            var nf = Function(f.name)
            for ai in range(f.arg_types.size()): nf.arg_types.push_back(f.arg_types[ai])
            for ri in range(f.ret_types.size()): nf.ret_types.push_back(f.ret_types[ri])
            for bi in range(f.blocks.size()): nf.blocks.push_back(f.blocks[bi])
            out.functions.push_back(self.run_on_function(nf))
        return out
fn __init__(out self, ) -> None:
        pass
fn __copyinit__(out self, other: Self) -> None:
        pass
fn __moveinit__(out self, deinit other: Self) -> None:
        pass
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
fn _buffer_i64_1d(n: Int) -> TypeDesc:
    return TypeDesc(String("buffer<i64,[") + String(n) + String("]>"))

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
fn _demo_copy_loop(N: Int) -> Module:
    var m = Module(String("demo_copy"))
    var f = Function(String("copy"))
    var b = Block(String("entry"))
    var idg = IdGen()

    var src = Value(idg.fresh(), _buffer_i64_1d(N))
    var dst = Value(idg.fresh(), _buffer_i64_1d(N))
    b.args.push_back(src); b.args.push_back(dst)

    var c0 = Op(String("ll.const.i64"), Location(String("d"), 1, 1))
    var v0 = Value(idg.fresh(), TypeDesc(String("i64"))); c0.results.push_back(Result(v0)); c0.attrs.push_back(String("value=0"))
    var cN = Op(String("ll.const.i64"), Location(String("d"), 1, 2))
    var vN = Value(idg.fresh(), TypeDesc(String("i64"))); cN.results.push_back(Result(vN)); cN.attrs.push_back(String("value=") + String(N))
    var c1 = Op(String("ll.const.i64"), Location(String("d"), 1, 3))
    var v1 = Value(idg.fresh(), TypeDesc(String("i64"))); c1.results.push_back(Result(v1)); c1.attrs.push_back(String("value=1"))

    var loop = Op(String("loop.for"), Location(String("d"), 1, 4))
    var iv = Value(idg.fresh(), TypeDesc(String("i64"))); loop.results.push_back(Result(iv))
    loop.operands.push_back(Operand(v0)); loop.operands.push_back(Operand(vN)); loop.operands.push_back(Operand(v1))

    var ld = Op(String("buf.load"), Location(String("d"), 1, 5))
    var x = Value(idg.fresh(), TypeDesc(String("i64"))); ld.results.push_back(Result(x))
    ld.operands.push_back(Operand(src)); ld.operands.push_back(Operand(iv))

    var st = Op(String("buf.store"), Location(String("d"), 1, 6))
    st.operands.push_back(Operand(x)); st.operands.push_back(Operand(dst)); st.operands.push_back(Operand(iv))

    var end = Op(String("loop.end"), Location(String("d"), 1, 7))

    var ret = Op(String("ll.ret"), Location(String("d"), 1, 8))

    b.ops.push_back(c0); b.ops.push_back(cN); b.ops.push_back(c1)
    b.ops.push_back(loop); b.ops.push_back(ld); b.ops.push_back(st); b.ops.push_back(end)
    b.ops.push_back(ret)

    f.blocks.push_back(b); m.functions.push_back(f)
    return m
fn _demo_other_loop(N: Int) -> Module:
    var m = Module(String("demo_other"))
    var f = Function(String("scale"))
    var b = Block(String("entry"))
    var idg = IdGen()

    var a = Value(idg.fresh(), _buffer_i64_1d(N))
    var bbuf = Value(idg.fresh(), _buffer_i64_1d(N))
    b.args.push_back(a); b.args.push_back(bbuf)

    var c0 = Op(String("ll.const.i64"), Location(String("d"), 2, 1))
    var v0 = Value(idg.fresh(), TypeDesc(String("i64"))); c0.results.push_back(Result(v0)); c0.attrs.push_back(String("value=0"))
    var cN = Op(String("ll.const.i64"), Location(String("d"), 2, 2))
    var vN = Value(idg.fresh(), TypeDesc(String("i64"))); cN.results.push_back(Result(vN)); cN.attrs.push_back(String("value=") + String(N))
    var c1 = Op(String("ll.const.i64"), Location(String("d"), 2, 3))
    var v1 = Value(idg.fresh(), TypeDesc(String("i64"))); c1.results.push_back(Result(v1)); c1.attrs.push_back(String("value=1"))

    var loop = Op(String("loop.for"), Location(String("d"), 2, 4))
    var iv = Value(idg.fresh(), TypeDesc(String("i64"))); loop.results.push_back(Result(iv))
    loop.operands.push_back(Operand(v0)); loop.operands.push_back(Operand(vN)); loop.operands.push_back(Operand(v1))

    var ld = Op(String("buf.load"), Location(String("d"), 2, 5))
    var x = Value(idg.fresh(), TypeDesc(String("i64"))); ld.results.push_back(Result(x))
    ld.operands.push_back(Operand(a)); ld.operands.push_back(Operand(iv))

    var c2 = Op(String("ll.const.i64"), Location(String("d"), 2, 6))
    var v2 = Value(idg.fresh(), TypeDesc(String("i64"))); c2.results.push_back(Result(v2)); c2.attrs.push_back(String("value=2"))

    var mul = Op(String("ll.mul"), Location(String("d"), 2, 7))
    var xm = Value(idg.fresh(), TypeDesc(String("i64"))); mul.results.push_back(Result(xm))
    mul.operands.push_back(Operand(x)); mul.operands.push_back(Operand(v2))

    var st = Op(String("buf.store"), Location(String("d"), 2, 8))
    st.operands.push_back(Operand(xm)); st.operands.push_back(Operand(bbuf)); st.operands.push_back(Operand(iv))

    var end = Op(String("loop.end"), Location(String("d"), 2, 9))

    var ret = Op(String("ll.ret"), Location(String("d"), 2, 10))

    b.ops.push_back(c0); b.ops.push_back(cN); b.ops.push_back(c1)
    b.ops.push_back(loop); b.ops.push_back(ld); b.ops.push_back(c2); b.ops.push_back(mul); b.ops.push_back(st); b.ops.push_back(end)
    b.ops.push_back(ret)

    f.blocks.push_back(b); m.functions.push_back(f)
    return m
fn _self_test_coalesce() -> Bool:
    var m1 = _demo_copy_loop(16)
    var m2 = _demo_other_loop(16)

    var pass = MemoryCoalesce()
    var out1 = pass.run_on_module(m1)
    var out2 = pass.run_on_module(m2)

    var pr = Printer()
    var t1 = pr.print(out1)
    var t2 = pr.print(out2)

    var ok = True
    # for copy loop: expect buf.memcpy and no loop.for
    if t1.find(String("buf.memcpy")) < 0: ok = False
    if t1.find(String("loop.for")) >= 0: ok = False
    if t1.find(String("len=16")) < 0: ok = False
    if t1.find(String("coalesce.width=4")) < 0: ok = False

    # for multiply loop: should NOT coalesce
    if t2.find(String("loop.for")) < 0: ok = False
    if t2.find(String("buf.memcpy")) >= 0: ok = False

    if ok: print(String("OK"))
    else: print(String("FAIL"))
    return ok