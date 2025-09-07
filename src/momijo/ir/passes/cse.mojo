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
# File: src/momijo/ir/passes/cse.mojo

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
fn _normalize_attrs(attrs: List[String]) -> String:
    # canonical textual form: "k=v;k2=v2;..."
    var keys = List[String]()
    var map = Dict[String, String]()
    for i in range(attrs.size()):
        var a = attrs[i]
        var eq = a.find(String("="))
        if eq > 0:
            var k = String("")
            var v = String("")
            var j = 0
            while j < eq:
                k = k + String.from_utf8([a.bytes()[j]])
                j = j + 1
            j = eq + 1
            while j < a.size():
                v = v + String.from_utf8([a.bytes()[j]])
                j = j + 1
            map[k] = v
            keys.push_back(k)
        else:
            # treat as flag
            map[a] = String("true")
            keys.push_back(a)
    # sort keys (selection sort for minimal env)
    var sorted = List[String]()
    while keys.size() > 0:
        var best = 0
        var bests = keys[0]
        var i = 1
        while i < keys.size():
            if keys[i] < bests:
                best = i
                bests = keys[i]
            i = i + 1
        sorted.push_back(bests)
        var tmp = List[String]()
        for j in range(keys.size()):
            if j != best: tmp.push_back(keys[j])
        keys = tmp
    # build string
    var out = String("")
    for i in range(sorted.size()):
        var k2 = sorted[i]
        out = out + k2 + String("=") + map[k2]
        if i + 1 < sorted.size(): out = out + String(";")
    return out
fn _is_commutative(name: String, attrs: List[String]) -> Bool:
    if name == String("ll.add") or name == String("mid.add") or name == String("hl.add.i64") or name == String("hl.add.f64"):
        return True
    if name == String("ll.mul") or name == String("mid.mul") or name == String("hl.mul.i64") or name == String("hl.mul.f64"):
        return True
    if name == String("ll.and") or name == String("ll.or"):
        return True
    if name == String("ll.cmp.eq") or name == String("hl.cmp.eq"):
        return True
    # mid.cmp {pred=EQ} is commutative
    if name == String("mid.cmp"):
        var has = False; var p = _attr_value(Op(name, Location()), String("pred"), has)  # this dummy can't see attrs; handle below
        # We'll assume non-commutative here; special handling in _key_for_op
        return False
    return False
fn _is_effectful(name: String) -> Bool:
    # Store, return, loops, runtime calls, calls, IO
    if name == String("ll.ret") or name == String("mid.ret") or name == String("hl.return"): return True
    if name.starts_with(String("buf.store")): return True
    if name.starts_with(String("rt.")): return True
    if name.starts_with(String("loop.")): return True
    if name.starts_with(String("call")): return True
    if name.starts_with(String("gpu.") or name.starts_with(String("mps."))): return True
    return False

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
# Alias map
# -----------------------------

struct AliasMap:
    var parent: Dict[Int, Int]
fn __init__(out self) -> None: self.parent = Dict[Int, Int]()
fn find(self, x: Int) -> Int:
        if not self.parent.contains_key(x): return x
        var r = self.parent[x]
        var root = self.find(r)
        self.parent[x] = root
        return root
fn set_alias(self, a: Int, b: Int) -> None:
        var rb = self.find(b)
        self.parent[a] = rb
fn __copyinit__(out self, other: Self) -> None:
        self.parent = other.parent
fn __moveinit__(out self, deinit other: Self) -> None:
        self.parent = other.parent
# -----------------------------
# CSE core
# -----------------------------

struct CSE:
    var alias: AliasMap
fn __init__(out self) -> None:
        self.alias = AliasMap()
fn _rep(self, v: Value) -> Int:
        return self.alias.find(v.id)
fn _normalize_operands(self, op: Op, out ids: List[Int], out comm: Bool) -> None:
        ids = List[Int]()
        var commut = _is_commutative(op.name, op.attrs)
        # mid.cmp special check for pred
        if op.name == String("mid.cmp"):
            var has = False; var p = _attr_value(op, String("pred"), has)
            if has and p == String("EQ"):
                commut = True
        for i in range(op.operands.size()):
            var id = self._rep(op.operands[i].value())
            ids.push_back(id)
        if commut and ids.size() >= 2:
            # sort ids
            var sorted = List[Int]()
            while ids.size() > 0:
                var best = 0
                var bestv = ids[0]
                var j = 1
                while j < ids.size():
                    if ids[j] < bestv:
                        best = j
                        bestv = ids[j]
                    j = j + 1
                sorted.push_back(bestv)
                var tmp = List[Int]()
                for k in range(ids.size()):
                    if k != best: tmp.push_back(ids[k])
                ids = tmp
            # reassign
            for t in range(sorted.size()):
                ids.push_back(sorted[t])
        comm = commut
fn _key_for_op(self, op: Op) -> String:
        # single-result only (multi-result: skip CSE)
        if op.results.size() != 1: return String("")
        if _is_effectful(op.name): return String("")
        # build key
        var ids = List[Int](); var comm = False
        self._normalize_operands(op, ids, comm)
        var key = String(op.name) + String("|")
        key = key + _normalize_attrs(op.attrs) + String("|")
        key = key + String(comm ? "C" : "N") + String("|")
        for i in range(ids.size()):
            key = key + String(ids[i])
            if i + 1 < ids.size(): key = key + String(",")
        key = key + String("|rtype=") + op.results[0].value().typ.name
        return key
fn run_on_block(self, b: Block) -> Block:
        var memo = Dict[String, Int]()       # key -> representative value id
        var i = 0
        while i < b.ops.size():
            var op = b.ops[i]
            # rewrite operands to reps
            for oi in range(op.operands.size()):
                var vv = op.operands[oi].value()
                vv.id = self.alias.find(vv.id)
                op.operands[oi] = Operand(vv)
            b.ops[i] = op

            var key = self._key_for_op(op)
            if key.size() == 0:
                # Not CSEable; but record constants to help later
                i = i + 1
                continue

            if memo.contains_key(key):
                # alias this op's result to representative
                var rep_id = memo[key]
                if op.results.size() > 0:
                    var resv = op.results[0].value()
                    self.alias.set_alias(resv.id, rep_id)
                # replace op with a note
                var note = Op(String("cse.removed"), op.loc)
                note.attrs.push_back(String("from=") + op.name)
                note.attrs.push_back(String("alias=%") + String(rep_id))
                # keep the original result handle for debugging
                if op.results.size() > 0:
                    note.results.push_back(op.results[0])
                b.ops[i] = note
            else:
                # First occurrence: install in memo
                if op.results.size() > 0:
                    var resv = op.results[0].value()
                    memo[key] = self.alias.find(resv.id)
                # keep as is
            i = i + 1
        return b
fn run_on_function(self, f: Function) -> Function:
        for bi in range(f.blocks.size()):
            f.blocks[bi] = self.run_on_block(f.blocks[bi])
        return f
fn run_on_module(self, m: Module) -> Module:
        var out = Module(m.name + String("_cse"))
        for fi in range(m.functions.size()):
            var f = m.functions[fi]
            var nf = Function(f.name)
            for ai in range(f.arg_types.size()): nf.arg_types.push_back(f.arg_types[ai])
            for ri in range(f.ret_types.size()): nf.ret_types.push_back(f.ret_types[ri])
            for bi in range(f.blocks.size()): nf.blocks.push_back(f.blocks[bi])
            out.functions.push_back(self.run_on_function(nf))
        return out
fn __copyinit__(out self, other: Self) -> None:
        self.alias = other.alias
fn __moveinit__(out self, deinit other: Self) -> None:
        self.alias = other.alias
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
    var m = Module(String("demo_cse"))
    var f = Function(String("main"))
    var b = Block(String("entry"))
    var idg = IdGen()

    # Constants 2 and 3 (duplicate 2 will be CSE'd)
    var c2a = Op(String("ll.const.i64"), Location(String("d"), 1, 1))
    var v2a = Value(idg.fresh(), TypeDesc(String("i64"))); c2a.results.push_back(Result(v2a)); c2a.attrs.push_back(String("value=2"))
    var c3 = Op(String("ll.const.i64"), Location(String("d"), 1, 2))
    var v3 = Value(idg.fresh(), TypeDesc(String("i64"))); c3.results.push_back(Result(v3)); c3.attrs.push_back(String("value=3"))
    var c2b = Op(String("ll.const.i64"), Location(String("d"), 1, 3))
    var v2b = Value(idg.fresh(), TypeDesc(String("i64"))); c2b.results.push_back(Result(v2b)); c2b.attrs.push_back(String("value=2"))

    # add(2,3) twice and reversed
    var add1 = Op(String("ll.add"), Location(String("d"), 1, 4))
    var va1 = Value(idg.fresh(), TypeDesc(String("i64"))); add1.operands.push_back(Operand(v2a)); add1.operands.push_back(Operand(v3)); add1.results.push_back(Result(va1))
    var add2 = Op(String("ll.add"), Location(String("d"), 1, 5))
    var va2 = Value(idg.fresh(), TypeDesc(String("i64"))); add2.operands.push_back(Operand(v2a)); add2.operands.push_back(Operand(v3)); add2.results.push_back(Result(va2))
    var add3 = Op(String("ll.add"), Location(String("d"), 1, 6))
    var va3 = Value(idg.fresh(), TypeDesc(String("i64"))); add3.operands.push_back(Operand(v3)); add3.operands.push_back(Operand(v2a)); add3.results.push_back(Result(va3))

    # sub(3,2) not equal to sub(2,3)
    var sub1 = Op(String("ll.sub"), Location(String("d"), 1, 7))
    var vs1 = Value(idg.fresh(), TypeDesc(String("i64"))); sub1.operands.push_back(Operand(v3)); sub1.operands.push_back(Operand(v2a)); sub1.results.push_back(Result(vs1))
    var sub2 = Op(String("ll.sub"), Location(String("d"), 1, 8))
    var vs2 = Value(idg.fresh(), TypeDesc(String("i64"))); sub2.operands.push_back(Operand(v2a)); sub2.operands.push_back(Operand(v3)); sub2.results.push_back(Result(vs2))

    # effectful store prevents elimination
    var outb = Value(idg.fresh(), TypeDesc(String("buffer<i64,[?]>")))
    var idx = Value(idg.fresh(), TypeDesc(String("i64")))
    var st1 = Op(String("buf.store"), Location(String("d"), 1, 9))
    st1.operands.push_back(Operand(va1)); st1.operands.push_back(Operand(outb)); st1.operands.push_back(Operand(idx))

    var ret = Op(String("ll.ret"), Location(String("d"), 1, 10))
    ret.operands.push_back(Operand(va1))

    b.ops.push_back(c2a); b.ops.push_back(c3); b.ops.push_back(c2b)
    b.ops.push_back(add1); b.ops.push_back(add2); b.ops.push_back(add3)
    b.ops.push_back(sub1); b.ops.push_back(sub2)
    b.ops.push_back(st1)
    b.ops.push_back(ret)
    f.blocks.push_back(b)
    m.functions.push_back(f)
    return m
fn _self_test_cse() -> Bool:
    var m = _demo_module()
    var cse = CSE()
    var out = cse.run_on_module(m)

    var pr = Printer()
    var txt = pr.print(out)

    var ok = True
    # duplicate const folded to notes
    if txt.find(String("cse.removed {from=ll.const.i64")) < 0: ok = False
    # add duplicates removed
    if txt.find(String("cse.removed {from=ll.add")) < 0: ok = False
    # ensure store is still present
    if txt.find(String("buf.store")) < 0: ok = False

    if ok: print(String("OK"))
    else: print(String("FAIL"))
    return ok