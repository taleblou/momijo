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
# Project: momijo.ir.midir
# File: src/momijo/ir/midir/loop_nest.mojo

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
fn is_buffer(t: TypeDesc) -> Bool: return t.name.starts_with(String("buffer<"))
fn is_i64(t: TypeDesc) -> Bool: return t.name == String("i64")
fn is_f64(t: TypeDesc) -> Bool: return t.name == String("f64")

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
# Tiny builders
# -----------------------------

struct Builders:
    var idg: IdGen
fn __init__(out self) -> None: self.idg = IdGen()
fn const_i64(self, b: Block, value: Int, loc: Location) -> Value:
        var v = Value(self.idg.fresh(), TypeDesc(String("i64")))
        var op = Op(String("ll.const.i64"), loc)
        op.results.push_back(Result(v))
        op.attrs.push_back(String("value=") + String(value))
        b.ops.push_back(op)
        return v
fn loop_for(self, b: Block, lo: Value, hi: Value, step: Value, loc: Location) -> Value:
        var iv = Value(self.idg.fresh(), TypeDesc(String("i64")))
        var loop = Op(String("loop.for"), loc)
        loop.operands.push_back(Operand(lo))
        loop.operands.push_back(Operand(hi))
        loop.operands.push_back(Operand(step))
        loop.results.push_back(Result(iv))
        b.ops.push_back(loop)
        return iv
fn loop_end(self, b: Block, loc: Location) -> None:
        var e = Op(String("loop.end"), loc)
        b.ops.push_back(e)
fn add_scalar(self, b: Block, a: Value, c: Value, is_f64_: Bool, loc: Location) -> Value:
        var v = Value(self.idg.fresh(), a.typ)
        var op = Op(is_f64_ ? String("ll.add") : String("ll.add"), loc)  # same name; types carry meaning
        op.operands.push_back(Operand(a))
        op.operands.push_back(Operand(c))
        op.results.push_back(Result(v))
        b.ops.push_back(op)
        return v
fn mul_scalar(self, b: Block, a: Value, c: Value, is_f64_: Bool, loc: Location) -> Value:
        var v = Value(self.idg.fresh(), a.typ)
        var op = Op(is_f64_ ? String("ll.mul") : String("ll.mul"), loc)
        op.operands.push_back(Operand(a))
        op.operands.push_back(Operand(c))
        op.results.push_back(Result(v))
        b.ops.push_back(op)
        return v
fn add_i64(self, b: Block, a: Value, c: Value, loc: Location) -> Value:
        var v = Value(self.idg.fresh(), TypeDesc(String("i64")))
        var op = Op(String("ll.add"), loc)
        op.operands.push_back(Operand(a))
        op.operands.push_back(Operand(c))
        op.results.push_back(Result(v))
        b.ops.push_back(op)
        return v
fn load(self, b: Block, buf: Value, idx: Value, elem_t: TypeDesc, loc: Location) -> Value:
        var v = Value(self.idg.fresh(), elem_t)
        var op = Op(String("buf.load"), loc)
        op.operands.push_back(Operand(buf))
        op.operands.push_back(Operand(idx))
        op.results.push_back(Result(v))
        b.ops.push_back(op)
        return v
fn store(self, b: Block, val: Value, buf: Value, idx: Value, loc: Location) -> None:
        var op = Op(String("buf.store"), loc)
        op.operands.push_back(Operand(val))
        op.operands.push_back(Operand(buf))
        op.operands.push_back(Operand(idx))
        b.ops.push_back(op)
fn __copyinit__(out self, other: Self) -> None:
        self.idg = other.idg
fn __moveinit__(out self, deinit other: Self) -> None:
        self.idg = other.idg
# -----------------------------

# -----------------------------

struct LoopNestLowering:
    var idg: IdGen
    var bld: Builders
fn __init__(out self) -> None:
        self.idg = IdGen()
        self.bld = Builders()
fn _int_attr(self, op: Op, key: String, default_val: Int, out ok: Bool) -> Int:
        var has = False
        var s = _attr_value(op, key, has)
        ok = has
        if not has: return default_val
        # naive parse int
        var sign = 1
        var i = 0
        if s.size() > 0 and s.bytes()[0] == 45:  # '-'
            sign = -1
            i = 1
        var num = 0
        while i < s.size():
            var d = s.bytes()[i] - UInt8(48)
            num = num * 10 + Int(d)
            i = i + 1
        return sign * num
fn _replace_vec_add_or_mul(self, f: Function, b: Block, idx_op: Int, is_add: Bool, elem_t: TypeDesc) -> None:
        var op = b.ops[idx_op]
        # Expect operands: out, a, b
        if op.operands.size() < 3: return
        var out_buf = op.operands[0].value()
        var a_buf = op.operands[1].value()
        var b_buf = op.operands[2].value()
        if not is_buffer(out_buf.typ) or not is_buffer(a_buf.typ) or not is_buffer(b_buf.typ):
            return

        var okN = False
        var N = self._int_attr(op, String("N"), 0, okN)
        if not okN or N <= 0: return

        var okU = False
        var unroll = self._int_attr(op, String("unroll"), 1, okU)
        if unroll <= 0: unroll = 1

        var loc = op.loc
        # Build constants
        var c0 = self.bld.const_i64(b, 0, loc)
        var cN = self.bld.const_i64(b, N, loc)
        var c1 = self.bld.const_i64(b, unroll, loc)  # step := unroll (assumes N%unroll==0)

        # Build loop header
        var iv = self.bld.loop_for(b, c0, cN, c1, loc)

        # Emit unrolled body
        var u = 0
        while u < unroll:
            var off = self.bld.const_i64(b, u, loc) if u > 0 else Value(-1, TypeDesc(String("i64")))
            var idx = iv
            if u > 0:
                idx = self.bld.add_i64(b, iv, off, loc)

            var ax = self.bld.load(b, a_buf, idx, elem_t, loc)
            var bx = self.bld.load(b, b_buf, idx, elem_t, loc)
            var cx = Value(-1, elem_t)
            if is_add:
                cx = self.bld.add_scalar(b, ax, bx, elem_t.name == String("f64"), loc)
            else:
                cx = self.bld.mul_scalar(b, ax, bx, elem_t.name == String("f64"), loc)
            self.bld.store(b, cx, out_buf, idx, loc)

            u = u + 1

        # Close loop
        self.bld.loop_end(b, loc)

        # Remove the original vec.* op (replace with a comment op for trace)
        var note = Op(String("loop.lowered"), loc)
        note.attrs.push_back(String("from=") + op.name)
        note.attrs.push_back(String("N=") + String(N))
        if unroll > 1:
            note.attrs.push_back(String("unroll=") + String(unroll))
        # Replace in-place: set placeholder and clear body? Simpler: mutate original op into note
        b.ops[idx_op] = note
fn run_on_function(self, f: Function) -> Function:
        for bi in range(f.blocks.size()):
            var b = f.blocks[bi]
            var i = 0
            while i < b.ops.size():
                var name = b.ops[i].name
                if name == String("vec.add.i64"):
                    self._replace_vec_add_or_mul(f, b, i, True, TypeDesc(String("i64")))
                elif name == String("vec.add.f64"):
                    self._replace_vec_add_or_mul(f, b, i, True, TypeDesc(String("f64")))
                elif name == String("vec.mul.i64"):
                    self._replace_vec_add_or_mul(f, b, i, False, TypeDesc(String("i64")))
                elif name == String("vec.mul.f64"):
                    self._replace_vec_add_or_mul(f, b, i, False, TypeDesc(String("f64")))
                i = i + 1
        return f
fn run_on_module(self, m: Module) -> Module:
        var out = Module(m.name + String("_loops"))
        for fi in range(m.functions.size()):
            var f = m.functions[fi]
            # shallow clone
            var nf = Function(f.name)
            for ai in range(f.arg_types.size()): nf.arg_types.push_back(f.arg_types[ai])
            for ri in range(f.ret_types.size()): nf.ret_types.push_back(f.ret_types[ri])
            for bi in range(f.blocks.size()):
                var b = f.blocks[bi]
                var nb = Block(b.name)
                for aa in range(b.args.size()): nb.args.push_back(b.args[aa])
                for oi in range(b.ops.size()): nb.ops.push_back(b.ops[oi])
                nf.blocks.push_back(nb)
            var lf = self.run_on_function(nf)
            out.functions.push_back(lf)
        return out
fn __copyinit__(out self, other: Self) -> None:
        self.idg = other.idg
        self.bld = other.bld
fn __moveinit__(out self, deinit other: Self) -> None:
        self.idg = other.idg
        self.bld = other.bld
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
fn _buffer_i64_1d() -> TypeDesc: return TypeDesc(String("buffer<i64,[?]>"))
fn _buffer_f64_1d() -> TypeDesc: return TypeDesc(String("buffer<f64,[?]>"))

struct DemoIR:
    var idg: IdGen
fn __init__(out self) -> None: self.idg = IdGen()
fn make_vec_add(self, N: Int, unroll: Int = 1) -> Module:
        var m = Module(String("demo_vec_add"))
        var f = Function(String("main"))
        f.arg_types.push_back(_buffer_i64_1d())
        f.arg_types.push_back(_buffer_i64_1d())
        f.arg_types.push_back(_buffer_i64_1d())
        var b = Block(String("entry"))
        var outb = Value(self.idg.fresh(), _buffer_i64_1d())
        var a = Value(self.idg.fresh(), _buffer_i64_1d())
        var c = Value(self.idg.fresh(), _buffer_i64_1d())
        b.args.push_back(outb); b.args.push_back(a); b.args.push_back(c)
        var vec = Op(String("vec.add.i64"), Location(String("demo.mojo"), 1, 1))
        vec.operands.push_back(Operand(outb))
        vec.operands.push_back(Operand(a))
        vec.operands.push_back(Operand(c))
        vec.attrs.push_back(String("N=") + String(N))
        if unroll > 1: vec.attrs.push_back(String("unroll=") + String(unroll))
        b.ops.push_back(vec)
        var ret = Op(String("ll.ret"), Location(String("demo.mojo"), 1, 2))
        b.ops.push_back(ret)
        f.blocks.push_back(b)
        m.functions.push_back(f)
        return m
fn __copyinit__(out self, other: Self) -> None:
        self.idg = other.idg
fn __moveinit__(out self, deinit other: Self) -> None:
        self.idg = other.idg
fn _self_test_loop_nest() -> Bool:
    var demo = DemoIR()
    var src = demo.make_vec_add(8, 2)
    var lower = LoopNestLowering()
    var dst = lower.run_on_module(src)
    var pr = Printer()
    var text = pr.print(dst)

    var ok = True
    if text.find(String("loop.for")) < 0: ok = False
    if text.find(String("buf.load")) < 0: ok = False
    if text.find(String("buf.store")) < 0: ok = False
    if text.find(String("ll.add")) < 0: ok = False
    if text.find(String("loop.end")) < 0: ok = False
    if text.find(String("unroll=2")) < 0: ok = False  # note op present

    if ok: print(String("OK"))
    else: print(String("FAIL"))
    return ok