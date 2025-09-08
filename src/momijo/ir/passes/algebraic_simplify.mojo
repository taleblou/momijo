# Project:      Momijo
# Module:       src.momijo.ir.passes.algebraic_simplify
# File:         algebraic_simplify.mojo
# Path:         src/momijo/ir/passes/algebraic_simplify.mojo
#
# Description:  src.momijo.ir.passes.algebraic_simplify — focused Momijo functionality with a stable public API.
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
fn is_i64(t: TypeDesc) -> Bool: return t.name == String("i64")
fn is_f64(t: TypeDesc) -> Bool: return t.name == String("f64")
fn is_bool(t: TypeDesc) -> Bool: return t.name == String("bool")
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
fn _parse_bool(s: String, out ok: Bool) -> Bool:
    ok = True
    if s == String("true") or s == String("1"): return True
    if s == String("false") or s == String("0"): return False
    ok = False
    return False

# -----------------------------
# Constant/alias DB
# -----------------------------

enum ConstKind(Int):
    None_ = 0
    I64 = 1
    F64 = 2
    Bool = 3

struct ConstInfo:
    var kind: ConstKind
    var s: String   # textual
fn __init__(out self) -> None: self.kind = ConstKind.None_; self.s = String("")
fn __copyinit__(out self, other: Self) -> None:
        self.kind = other.kind
        self.s = other.s
fn __moveinit__(out self, deinit other: Self) -> None:
        self.kind = other.kind
        self.s = other.s
struct SimplifyCtx:
    var const_of: Dict[Int, ConstInfo]  # Value.id -> const info
    var alias_to: Dict[Int, Int]        # Value.id -> representative id
fn __init__(out self) -> None:
        self.const_of = Dict[Int, ConstInfo]()
        self.alias_to = Dict[Int, Int]()
fn find_rep(self, vid: Int) -> Int:
        if not self.alias_to.contains_key(vid): return vid
        # path compress
        var r = self.alias_to[vid]
        var root = self.find_rep(r)
        self.alias_to[vid] = root
        return root
fn set_alias(self, from_id: Int, to_id: Int) -> None:
        var r = self.find_rep(to_id)
        self.alias_to[from_id] = r
fn set_const(self, vid: Int, kind: ConstKind, s: String) -> None:
        var ci = ConstInfo()
        ci.kind = kind; ci.s = s
        self.const_of[self.find_rep(vid)] = ci
fn get_const(self, vid: Int, out found: Bool) -> ConstInfo:
        var r = self.find_rep(vid)
        if self.const_of.contains_key(r):
            found = True
            return self.const_of[r]
        found = False
        var z = ConstInfo()
        return z
fn __copyinit__(out self, other: Self) -> None:
        self.const_of = other.const_of
        self.alias_to = other.alias_to
fn __moveinit__(out self, deinit other: Self) -> None:
        self.const_of = other.const_of
        self.alias_to = other.alias_to
# -----------------------------
# Dialect helpers
# -----------------------------
fn _is_add(name: String) -> Bool:
    return name == String("ll.add") or name == String("mid.add") or name == String("hl.add.i64") or name == String("hl.add.f64")
fn _is_sub(name: String) -> Bool:
    return name == String("ll.sub") or name == String("mid.sub") or name == String("hl.sub.i64") or name == String("hl.sub.f64")
fn _is_mul(name: String) -> Bool:
    return name == String("ll.mul") or name == String("mid.mul") or name == String("hl.mul.i64") or name == String("hl.mul.f64")
fn _is_div(name: String) -> Bool:
    return name == String("ll.div") or name == String("mid.div") or name == String("hl.div.i64") or name == String("hl.div.f64")
fn _is_cmp_eq(name: String) -> Bool:
    return name == String("ll.cmp.eq") or name == String("hl.cmp.eq") or name == String("mid.cmp.eq")
fn _is_cmp_lt(name: String) -> Bool:
    return name == String("ll.cmp.lt") or name == String("hl.cmp.lt") or name == String("mid.cmp.lt")
fn _is_cmp_gt(name: String) -> Bool:
    return name == String("ll.cmp.gt") or name == String("hl.cmp.gt") or name == String("mid.cmp.gt")
fn _is_mid_cmp(name: String) -> Bool:
    return name == String("mid.cmp")
fn _is_select(name: String) -> Bool:
    return name == String("ll.select") or name == String("mid.select") or name == String("hl.select")

# -----------------------------
# Rewriter
# -----------------------------

struct AlgebraicSimplify:
    var ctx: SimplifyCtx
fn __init__(out self) -> None:
        self.ctx = SimplifyCtx()
fn _materialize_const_op(self, kind: ConstKind, s: String, loc: Location) -> Op:
        var opname = String("ll.const.i64")
        if kind == ConstKind.I64: opname = String("ll.const.i64")
        elif kind == ConstKind.F64: opname = String("ll.const.f64")
        elif kind == ConstKind.Bool: opname = String("ll.const.bool")
        var op = Op(opname, loc)
        var v = Value(-1, TypeDesc(kind == ConstKind.I64 ? String("i64") : (kind == ConstKind.F64 ? String("f64") : String("bool"))))
        op.results.push_back(Result(v))
        op.attrs.push_back(String("value=") + s)
        return op
fn _replace_with_const(self, b: Block, idx: Int, kind: ConstKind, s: String) -> None:
        var old = b.ops[idx]
        # mutate: turn into ll.const.* and keep result id/type
        if old.results.size() == 0: return
        var resv = old.results[0].value()
        old.name = (kind == ConstKind.I64 ? String("ll.const.i64") : (kind == ConstKind.F64 ? String("ll.const.f64") : String("ll.const.bool")))
        old.attrs = List[String]()
        old.attrs.push_back(String("value=") + s)
        old.operands = List[Operand]()
        # ensure type name matches
        resv.typ = (kind == ConstKind.I64 ? TypeDesc(String("i64")) : (kind == ConstKind.F64 ? TypeDesc(String("f64")) : TypeDesc(String("bool"))))
        old.results[0] = Result(resv)
        b.ops[idx] = old
        self.ctx.set_const(resv.id, kind, s)
fn _mark_removed(self, b: Block, idx: Int, from_name: String, rep: Value) -> None:
        var old = b.ops[idx]
        var note = Op(String("simp.removed"), old.loc)
        note.attrs.push_back(String("from=") + from_name)
        # keep result to help debug
        if old.results.size() > 0:
            note.results.push_back(old.results[0])
        b.ops[idx] = note
        if old.results.size() > 0:
            self.ctx.set_alias(old.results[0].value().id, rep.id)
fn _binop_fold_i64(self, name: String, a: Int, b: Int, out ok: Bool) -> Int:
        ok = True
        if _is_add(name): return a + b
        if _is_sub(name): return a - b
        if _is_mul(name): return a * b
        if _is_div(name):
            if b == 0: ok = False; return 0
            return a / b
        ok = False
        return 0
fn _try_simplify_binop(self, b: Block, idx: Int) -> Bool:
        var op = b.ops[idx]
        if op.operands.size() < 2 or op.results.size() == 0: return False
        if not (_is_add(op.name) or _is_sub(op.name) or _is_mul(op.name) or _is_div(op.name)): return False

        # resolve reps
        var lhs = op.operands[0].value()
        var rhs = op.operands[1].value()
        lhs.id = self.ctx.find_rep(lhs.id)
        rhs.id = self.ctx.find_rep(rhs.id)

        # const folding (i64)
        var hasL = False; var cL = self.ctx.get_const(lhs.id, hasL)
        var hasR = False; var cR = self.ctx.get_const(rhs.id, hasR)
        if hasL and hasR and cL.kind == ConstKind.I64 and cR.kind == ConstKind.I64:
            var ok = False
            var a = _parse_i64(cL.s, ok); if not ok: return False
            var b2 = _parse_i64(cR.s, ok); if not ok: return False
            var res = 0
            var ok2 = False
            res = self._binop_fold_i64(op.name, a, b2, ok2)
            if ok2:
                self._replace_with_const(b, idx, ConstKind.I64, String(res))
                return True

        # identities (only consider i64 types by name or via known const)
        var zero = ConstInfo(); zero.kind = ConstKind.I64; zero.s = String("0")
        var one = ConstInfo();  one.kind = ConstKind.I64;  one.s = String("1")

        # recognize literal zero/one via const map or via immediate const ops encountered earlier
        var lhs_is0 = hasL and cL.kind == ConstKind.I64 and cL.s == String("0")
        var rhs_is0 = hasR and cR.kind == ConstKind.I64 and cR.s == String("0")
        var lhs_is1 = hasL and cL.kind == ConstKind.I64 and cL.s == String("1")
        var rhs_is1 = hasR and cR.kind == ConstKind.I64 and cR.s == String("1")

        if _is_add(op.name):
            if lhs_is0:
                self._mark_removed(b, idx, op.name, rhs); return True
            if rhs_is0:
                self._mark_removed(b, idx, op.name, lhs); return True
        elif _is_sub(op.name):
            if rhs_is0:
                self._mark_removed(b, idx, op.name, lhs); return True
        elif _is_mul(op.name):
            if lhs_is0 or rhs_is0:
                self._replace_with_const(b, idx, ConstKind.I64, String(0)); return True
            if lhs_is1:
                self._mark_removed(b, idx, op.name, rhs); return True
            if rhs_is1:
                self._mark_removed(b, idx, op.name, lhs); return True
        elif _is_div(op.name):
            if rhs_is1:
                self._mark_removed(b, idx, op.name, lhs); return True
            if lhs_is0:
                self._replace_with_const(b, idx, ConstKind.I64, String(0)); return True

        return False
fn _try_simplify_cmp(self, b: Block, idx: Int) -> Bool:
        var op = b.ops[idx]
        var is_cmp = _is_cmp_eq(op.name) or _is_cmp_lt(op.name) or _is_cmp_gt(op.name) or _is_mid_cmp(op.name)
        if not is_cmp: return False
        if op.operands.size() < 2 or op.results.size() == 0: return False

        var a = op.operands[0].value(); a.id = self.ctx.find_rep(a.id)
        var c = op.operands[1].value(); c.id = self.ctx.find_rep(c.id)

        var name = op.name
        if _is_mid_cmp(name):
            var has = False; var p = _attr_value(op, String("pred"), has)
            if has and p == String("EQ"): name = String("ll.cmp.eq")
            elif has and p == String("LT"): name = String("ll.cmp.lt")
            elif has and p == String("GT"): name = String("ll.cmp.gt")
            else: name = String("ll.cmp.eq")

        # x ? x shortcuts
        if a.id == c.id:
            if _is_cmp_eq(name):
                self._replace_with_const(b, idx, ConstKind.Bool, String("true")); return True
            if _is_cmp_lt(name) or _is_cmp_gt(name):
                self._replace_with_const(b, idx, ConstKind.Bool, String("false")); return True

        # const folding (i64 only for now)
        var hasA = False; var ca = self.ctx.get_const(a.id, hasA)
        var hasB = False; var cb = self.ctx.get_const(c.id, hasB)
        if hasA and hasB and ca.kind == ConstKind.I64 and cb.kind == ConstKind.I64:
            var ok = False
            var x = _parse_i64(ca.s, ok); if not ok: return False
            var y = _parse_i64(cb.s, ok); if not ok: return False
            var resb = False
            if _is_cmp_eq(name): resb = (x == y)
            elif _is_cmp_lt(name): resb = (x < y)
            elif _is_cmp_gt(name): resb = (x > y)
            self._replace_with_const(b, idx, ConstKind.Bool, resb ? String("true") : String("false"))
            return True

        return False
fn _try_simplify_select(self, b: Block, idx: Int) -> Bool:
        var op = b.ops[idx]
        if not _is_select(op.name): return False
        if op.operands.size() < 3 or op.results.size() == 0: return False
        var cond = op.operands[0].value(); cond.id = self.ctx.find_rep(cond.id)
        var tv = op.operands[1].value();  tv.id = self.ctx.find_rep(tv.id)
        var fv = op.operands[2].value();  fv.id = self.ctx.find_rep(fv.id)

        # select(c, x, x) -> x
        if tv.id == fv.id:
            self._mark_removed(b, idx, op.name, tv); return True

        # if cond is const
        var has = False; var cc = self.ctx.get_const(cond.id, has)
        if has and cc.kind == ConstKind.Bool:
            if cc.s == String("true"):
                self._mark_removed(b, idx, op.name, tv); return True
            if cc.s == String("false"):
                self._mark_removed(b, idx, op.name, fv); return True

        return False
fn _see_constant_def(self, op: Op) -> None:
        if op.results.size() == 0: return
        var resv = op.results[0].value()
        var has = False
        if op.name == String("ll.const.i64") or op.name == String("mid.constant.i64") or op.name == String("hl.const.i64"):
            var v = _attr_value(op, String("value"), has)
            if has: self.ctx.set_const(resv.id, ConstKind.I64, v)
        elif op.name == String("ll.const.f64") or op.name == String("mid.constant.f64") or op.name == String("hl.const.f64"):
            var v2 = _attr_value(op, String("value"), has)
            if has: self.ctx.set_const(resv.id, ConstKind.F64, v2)
        elif op.name == String("ll.const.bool") or op.name == String("mid.constant.bool") or op.name == String("hl.const.bool"):
            var v3 = _attr_value(op, String("value"), has)
            if has: self.ctx.set_const(resv.id, ConstKind.Bool, v3)
fn run_on_block(self, b: Block) -> Block:
        var i = 0
        while i < b.ops.size():
            # rewrite operands to reps (aliases)
            var op = b.ops[i]
            for oi in range(op.operands.size()):
                var ov = op.operands[oi].value()
                ov.id = self.ctx.find_rep(ov.id)
                op.operands[oi] = Operand(ov)
            b.ops[i] = op

            # record constants seen
            self._see_constant_def(op)

            # try folds in priority order
            if self._try_simplify_binop(b, i) or self._try_simplify_cmp(b, i) or self._try_simplify_select(b, i):
                # if simplified, the op at index i may have changed; also constants/aliases updated
                pass

            i = i + 1
        return b
fn run_on_function(self, f: Function) -> Function:
        for bi in range(f.blocks.size()):
            f.blocks[bi] = self.run_on_block(f.blocks[bi])
        return f
fn run_on_module(self, m: Module) -> Module:
        var out = Module(m.name + String("_simp"))
        for fi in range(m.functions.size()):
            # shallow clone + rewrite
            var f = m.functions[fi]
            var nf = Function(f.name)
            for ai in range(f.arg_types.size()): nf.arg_types.push_back(f.arg_types[ai])
            for ri in range(f.ret_types.size()): nf.ret_types.push_back(f.ret_types[ri])
            for bi in range(f.blocks.size()): nf.blocks.push_back(f.blocks[bi])
            out.functions.push_back(self.run_on_function(nf))
        return out
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
    var m = Module(String("demo_simp"))
    var f = Function(String("main"))
    var b = Block(String("entry"))
    var idg = IdGen()

    # args
    var x = Value(idg.fresh(), TypeDesc(String("i64")))
    var y = Value(idg.fresh(), TypeDesc(String("i64")))
    b.args.push_back(x); b.args.push_back(y)

    # consts
    var c3 = Op(String("ll.const.i64"), Location(String("d"), 1, 1))
    var v3 = Value(idg.fresh(), TypeDesc(String("i64"))); c3.results.push_back(Result(v3)); c3.attrs.push_back(String("value=3"))
    var c5 = Op(String("ll.const.i64"), Location(String("d"), 1, 2))
    var v5 = Value(idg.fresh(), TypeDesc(String("i64"))); c5.results.push_back(Result(v5)); c5.attrs.push_back(String("value=5"))
    var c0 = Op(String("ll.const.i64"), Location(String("d"), 1, 3))
    var v0 = Value(idg.fresh(), TypeDesc(String("i64"))); c0.results.push_back(Result(v0)); c0.attrs.push_back(String("value=0"))
    var c1 = Op(String("ll.const.i64"), Location(String("d"), 1, 4))
    var v1 = Value(idg.fresh(), TypeDesc(String("i64"))); c1.results.push_back(Result(v1)); c1.attrs.push_back(String("value=1"))

    # add(v3,v5) -> 8 (const fold)
    var addc = Op(String("ll.add"), Location(String("d"), 1, 5))
    var vaddc = Value(idg.fresh(), TypeDesc(String("i64")))
    addc.operands.push_back(Operand(v3)); addc.operands.push_back(Operand(v5)); addc.results.push_back(Result(vaddc))

    # mul(x,0) -> 0
    var mulz = Op(String("ll.mul"), Location(String("d"), 1, 6))
    var vmulz = Value(idg.fresh(), TypeDesc(String("i64")))
    mulz.operands.push_back(Operand(x)); mulz.operands.push_back(Operand(v0)); mulz.results.push_back(Result(vmulz))

    # div(3,1) -> 3 (alias to 3)
    var div1 = Op(String("ll.div"), Location(String("d"), 1, 7))
    var vdiv1 = Value(idg.fresh(), TypeDesc(String("i64")))
    div1.operands.push_back(Operand(v3)); div1.operands.push_back(Operand(v1)); div1.results.push_back(Result(vdiv1))

    # add(y,0) -> y (removed)
    var add0 = Op(String("ll.add"), Location(String("d"), 1, 8))
    var vadd0 = Value(idg.fresh(), TypeDesc(String("i64")))
    add0.operands.push_back(Operand(y)); add0.operands.push_back(Operand(v0)); add0.results.push_back(Result(vadd0))

    # cmp.eq(5,5) -> true
    var cmpeq = Op(String("ll.cmp.eq"), Location(String("d"), 1, 9))
    var vcmp = Value(idg.fresh(), TypeDesc(String("bool")))
    cmpeq.operands.push_back(Operand(v5)); cmpeq.operands.push_back(Operand(v5)); cmpeq.results.push_back(Result(vcmp))

    # select(true, v3, v5) -> v3
    var ctrue = Op(String("ll.const.bool"), Location(String("d"), 1, 10))
    var vtrue = Value(idg.fresh(), TypeDesc(String("bool"))); ctrue.results.push_back(Result(vtrue)); ctrue.attrs.push_back(String("value=true"))
    var sel = Op(String("ll.select"), Location(String("d"), 1, 11))
    var vsel = Value(idg.fresh(), TypeDesc(String("i64")))
    sel.operands.push_back(Operand(vtrue)); sel.operands.push_back(Operand(v3)); sel.operands.push_back(Operand(v5)); sel.results.push_back(Result(vsel))

    b.ops.push_back(c3); b.ops.push_back(c5); b.ops.push_back(c0); b.ops.push_back(c1)
    b.ops.push_back(addc); b.ops.push_back(mulz); b.ops.push_back(div1); b.ops.push_back(add0); b.ops.push_back(cmpeq); b.ops.push_back(ctrue); b.ops.push_back(sel)

    var ret = Op(String("ll.ret"), Location(String("d"), 1, 12))
    ret.operands.push_back(Operand(vsel))
    b.ops.push_back(ret)

    f.blocks.push_back(b)
    m.functions.push_back(f)
    return m
fn _self_test_algebraic_simplify() -> Bool:
    var m = _demo_module()
    var simp = AlgebraicSimplify()
    var out = simp.run_on_module(m)

    var pr = Printer()
    var txt = pr.print(out)

    var ok = True
    if txt.find(String("ll.const.i64 {value=8}")) < 0: ok = False
    if txt.find(String("simp.removed")) < 0: ok = False
    if txt.find(String("ll.const.bool {value=true}")) < 0: ok = False

    if ok: print(String("OK"))
    else: print(String("FAIL"))
    return ok