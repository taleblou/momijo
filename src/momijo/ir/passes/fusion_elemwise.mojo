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
# Project: momijo.ir.passes
# File: src/momijo/ir/passes/fusion_elemwise.mojo

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
fn _same_attr(op1: Op, op2: Op, key: String) -> Bool:
    var h1 = False; var v1 = _attr_value(op1, key, h1)
    var h2 = False; var v2 = _attr_value(op2, key, h2)
    if not h1 and not h2: return True     # treat missing as compatible
    if h1 and h2 and v1 == v2: return True
    return False
fn _vec_kind(name: String, out base: String, out dtype: String, out ok: Bool) -> None:
    ok = False
    base = String(""); dtype = String("")
    # expect "vec.<op>.<ty>"
    if not name.starts_with(String("vec.")): return
    # parse
    var dot1 = name.find(String("."))          # after "vec"
    if dot1 < 0: return
    var dot2 = name.find(String("."), dot1+1)
    if dot2 < 0: return
    # base between dot1+1 .. dot2-1
    var b = String("")
    var i = dot1 + 1
    while i < dot2:
        b = b + String.from_utf8([name.bytes()[i]])
        i = i + 1
    # dtype after dot2+1 .. end
    var d = String("")
    i = dot2 + 1
    while i < name.size():
        d = d + String.from_utf8([name.bytes()[i]])
        i = i + 1
    # supported
    if b == String("add") or b == String("sub") or b == String("mul") or b == String("div"):
        if d == String("i64") or d == String("f64"):
            base = b; dtype = d; ok = True; return
fn _is_vec_bin(op: Op, out base: String, out dtype: String) -> Bool:
    var ok = False
    _vec_kind(op.name, base, dtype, ok)
    return ok

# -----------------------------
# Use counts (of Values as operands)
# -----------------------------

struct UseCounts:
    var uses: Dict[Int, Int]
fn __init__(out self) -> None:
        self.uses = Dict[Int, Int]()
fn add_use(self, vid: Int) -> None:
        var c = 0
        if self.uses.contains_key(vid): c = self.uses[vid]
        self.uses[vid] = c + 1
fn get(self, vid: Int) -> Int:
        if not self.uses.contains_key(vid): return 0
        return self.uses[vid]
fn __copyinit__(out self, other: Self) -> None:
        self.uses = other.uses
fn __moveinit__(out self, deinit other: Self) -> None:
        self.uses = other.uses
fn _compute_block_use_counts(b: Block) -> UseCounts:
    var uc = UseCounts()
    for i in range(b.ops.size()):
        var op = b.ops[i]
        for j in range(op.operands.size()):
            uc.add_use(op.operands[j].value().id)
    return uc

# -----------------------------
# Fusion pass
# -----------------------------

struct ElemwiseFusion:
fn _fuse_pair(self, b: Block, prod_idx: Int, cons_idx: Int, prod_base: String, cons_base: String, dtype: String, temp_buf: Value) -> None:
        var prod = b.ops[prod_idx]
        var cons = b.ops[cons_idx]

        # Producer: vec.<OP1>.<dtype>(tmp, a, b)
        if prod.operands.size() < 3: return
        var out_tmp = prod.operands[0].value()
        var a = prod.operands[1].value()
        var b0 = prod.operands[2].value()

        # Consumer: vec.<OP2>.<dtype>(out, tmp, c)  or (out, c, tmp)
        if cons.operands.size() < 3: return
        var out_buf = cons.operands[0].value()
        var c1 = cons.operands[1].value()
        var c2 = cons.operands[2].value()

        var left_tmp = (c1.id == temp_buf.id)
        var right_tmp = (c2.id == temp_buf.id)
        if not (left_tmp or right_tmp): return

        # shape/unroll compatibility
        if not _same_attr(prod, cons, String("N")): return
        if not _same_attr(prod, cons, String("unroll")): return

        var N_attr_has = False; var N_attr = _attr_value(cons, String("N"), N_attr_has)
        var U_attr_has = False; var U_attr = _attr_value(cons, String("unroll"), U_attr_has)

        # Build fused op: vec.fused.<dtype>(out, <inputs...>) {N=?, [unroll], arity=?, expr_rpn="..."}
        var fused = Op(String("vec.fused.") + dtype, cons.loc)
        fused.operands.push_back(Operand(out_buf))

        # inputs in order: a, b0, (other input of consumer)
        var inputs = List[Value]()
        inputs.push_back(a); inputs.push_back(b0)
        var other = left_tmp ? c2 : c1
        inputs.push_back(other)

        # attach as operands after out
        for i in range(inputs.size()):
            fused.operands.push_back(Operand(inputs[i]))

        # attrs
        if N_attr_has: fused.attrs.push_back(String("N=") + N_attr)
        if U_attr_has: fused.attrs.push_back(String("unroll=") + U_attr)
        fused.attrs.push_back(String("arity=3"))

        # expr as RPN: "x0 x1 OP1 x2 OP2" or swapped if tmp on right? we already always place other as x2
        var rpn = String("x0 x1 ") + prod_base + String(" x2 ") + cons_base
        fused.attrs.push_back(String("expr_rpn=") + rpn)

        # Emit note in place of producer
        var note = Op(String("fusion.removed"), prod.loc)
        note.attrs.push_back(String("from=") + prod.name)

        # Replace ops
        b.ops[prod_idx] = note
        b.ops[cons_idx] = fused
fn run_on_block(self, b: Block) -> Block:
        var uc = _compute_block_use_counts(b)

        var i = 0
        while i < b.ops.size():
            var cons = b.ops[i]
            var base2 = String(""); var dtype = String("")
            if _is_vec_bin(cons, base2, dtype):
                # check its two inputs for temp buffer produced by previous vec.* op
                # We'll scan backwards to find a vec.* writing to that buffer.
                if cons.operands.size() >= 3:
                    var out_buf = cons.operands[0].value()
                    var in1 = cons.operands[1].value()
                    var in2 = cons.operands[2].value()

                    # candidate temp is the operand with use-count == 1 (only here)
                    var candidate = Value(-1, TypeDesc(String("?")))
                    var has_cand = False
                    if uc.get(in1.id) == 1: candidate = in1; has_cand = True
                    elif uc.get(in2.id) == 1: candidate = in2; has_cand = True

                    if has_cand:
                        # find producer writing to candidate as its 'out'
                        var j = i - 1
                        while j >= 0:
                            var prod = b.ops[j]
                            var base1 = String(""); var dt = String("")
                            if _is_vec_bin(prod, base1, dt) and dt == dtype:
                                if prod.operands.size() >= 3 and prod.operands[0].value().id == candidate.id:
                                    # Found chain; fuse them
                                    self._fuse_pair(b, j, i, base1, base2, dtype, candidate)
                                    # after replacement, keep scanning (but avoid double-fusing the same consumer)
                                    break
                            j = j - 1
            i = i + 1
        return b
fn run_on_function(self, f: Function) -> Function:
        for bi in range(f.blocks.size()):
            f.blocks[bi] = self.run_on_block(f.blocks[bi])
        return f
fn run_on_module(self, m: Module) -> Module:
        var out = Module(m.name + String("_fused"))
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
fn _buffer_i64_1d() -> TypeDesc: return TypeDesc(String("buffer<i64,[?]>"))
fn _buffer_f64_1d() -> TypeDesc: return TypeDesc(String("buffer<f64,[?]>"))

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
    var m = Module(String("demo_fuse"))
    var f = Function(String("main"))
    var b = Block(String("entry"))
    var idg = IdGen()

    # buffers: out, a, b, c, tmp
    var outb = Value(idg.fresh(), _buffer_i64_1d())
    var a = Value(idg.fresh(), _buffer_i64_1d())
    var bb = Value(idg.fresh(), _buffer_i64_1d())
    var c = Value(idg.fresh(), _buffer_i64_1d())
    var tmp = Value(idg.fresh(), _buffer_i64_1d())
    b.args.push_back(outb); b.args.push_back(a); b.args.push_back(bb); b.args.push_back(c); b.args.push_back(tmp)

    # vec.add.i64(tmp, a, b) {N=8}
    var add = Op(String("vec.add.i64"), Location(String("d"), 1, 1))
    add.operands.push_back(Operand(tmp)); add.operands.push_back(Operand(a)); add.operands.push_back(Operand(bb))
    add.attrs.push_back(String("N=8"))
    b.ops.push_back(add)

    # vec.mul.i64(out, tmp, c) {N=8}
    var mul = Op(String("vec.mul.i64"), Location(String("d"), 1, 2))
    mul.operands.push_back(Operand(outb)); mul.operands.push_back(Operand(tmp)); mul.operands.push_back(Operand(c))
    mul.attrs.push_back(String("N=8"))
    b.ops.push_back(mul)

    var ret = Op(String("ll.ret"), Location(String("d"), 1, 3))
    b.ops.push_back(ret)

    f.blocks.push_back(b)
    m.functions.push_back(f)
    return m
fn _self_test_fusion() -> Bool:
    var m = _demo_module()
    var fusion = ElemwiseFusion()
    var out = fusion.run_on_module(m)

    var pr = Printer()
    var txt = pr.print(out)

    var ok = True
    if txt.find(String("vec.fused.i64")) < 0: ok = False
    if txt.find(String("expr_rpn=x0 x1 add x2 mul")) < 0: ok = False
    if txt.find(String("fusion.removed {from=vec.add.i64}")) < 0: ok = False
    # consumer replaced, so "vec.mul.i64" should be gone
    if txt.find(String("vec.mul.i64")) >= 0: ok = False

    if ok: print(String("OK"))
    else: print(String("FAIL"))
    return ok