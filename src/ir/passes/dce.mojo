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
# File: src/momijo/ir/dce.mojo
# Description: Dead Code Elimination (DCE) pass for Momijo IR (per-block, SSA).
# Notes:
#   - Self-contained fallback IR; no globals, no 'export'; only 'var' (no 'let').
#   - Constructors use: fn __init__(out self, ...)
#   - Removes ops that (a) have no side effects and (b) all results are unused.
#   - Updates use-counts while removing, repeats to a fixed point.
#   - Treats 'cse.removed' / 'simp.removed' as effect-free notes; removes them if unused.
#   - Keeps effectful ops: returns, stores, loops, runtime calls, calls, gpu/mps/io.
#   - Includes a tiny printer and a self-test (prints OK).
# ============================================================================

# -----------------------------
# Minimal IR model
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
# Utilities
# -----------------------------

fn _is_effectful(name: String) -> Bool:
    if name == String("ll.ret") or name == String("mid.ret") or name == String("hl.return") or name == String("return"):
        return True
    if name.starts_with(String("buf.store")): return True
    if name.starts_with(String("rt.")): return True
    if name.starts_with(String("loop.")): return True
    if name.starts_with(String("call")): return True
    if name.starts_with(String("gpu.")) or name.starts_with(String("mps.")): return True
    if name.starts_with(String("io.")): return True
    return False

fn _note_kind(name: String) -> Bool:
    return name == String("cse.removed") or name == String("simp.removed")

# -----------------------------
# Use-count analysis
# -----------------------------

struct UseCounts:
    var uses: Dict[Int, Int]
    fn __init__(out self):
        self.uses = Dict[Int, Int]()

    fn add_use(self, vid: Int) -> None:
        var c = 0
        if self.uses.contains_key(vid): c = self.uses[vid]
        self.uses[vid] = c + 1

    fn dec_use(self, vid: Int) -> None:
        if not self.uses.contains_key(vid): return
        var c = self.uses[vid] - 1
        if c < 0: c = 0
        self.uses[vid] = c

    fn get(self, vid: Int) -> Int:
        if not self.uses.contains_key(vid): return 0
        return self.uses[vid]

fn _compute_block_use_counts(b: Block) -> UseCounts:
    var uc = UseCounts()
    for oi in range(b.ops.size()):
        var op = b.ops[oi]
        for pi in range(op.operands.size()):
            uc.add_use(op.operands[pi].value.id)
    return uc

# -----------------------------
# DCE core (per-block fixed-point)
# -----------------------------

struct DCE:
    fn _remove_indices(self, b: Block, to_remove: List[Int], uc: UseCounts) -> Block:
        # Remove ops at indices listed in ascending order (already expected sorted).
        # While removing, decrement use-counts for their operands.
        # Build new list skipping removed ones.
        var rm = Dict[Int, Bool]()
        for i in range(to_remove.size()):
            rm[to_remove[i]] = True
        var new_ops = List[Op]()
        for i in range(b.ops.size()):
            if rm.contains_key(i):
                var op = b.ops[i]
                for pi in range(op.operands.size()):
                    uc.dec_use(op.operands[pi].value.id)
                # if op had results, they had zero uses by premise
                continue
            new_ops.push_back(b.ops[i])
        b.ops = new_ops
        return b

    fn run_on_block(self, b: Block) -> Block:
        var changed = True
        var blk = b
        while changed:
            changed = False
            var uc = _compute_block_use_counts(blk)
            var dead_idxs = List[Int]()

            # mark for removal
            for i in range(blk.ops.size()):
                var op = blk.ops[i]
                # never remove effectful ops
                if _is_effectful(op.name): continue

                # special-case notes: remove if result unused OR they have no results
                if _note_kind(op.name):
                    if op.results.size() == 0:
                        dead_idxs.push_back(i)
                        continue
                    var all_unused_note = True
                    for ri in range(op.results.size()):
                        if uc.get(op.results[ri].value.id) > 0:
                            all_unused_note = False; break
                    if all_unused_note:
                        dead_idxs.push_back(i)
                    continue

                # if op has no results and is effect-free, it's dead
                if op.results.size() == 0:
                    dead_idxs.push_back(i)
                    continue

                # all results unused?
                var all_unused = True
                for ri in range(op.results.size()):
                    if uc.get(op.results[ri].value.id) > 0:
                        all_unused = False; break
                if all_unused:
                    dead_idxs.push_back(i)

            if dead_idxs.size() > 0:
                # sort indices ascending (selection sort for minimal env)
                var sorted = List[Int]()
                var tmp = dead_idxs
                while tmp.size() > 0:
                    var best = 0
                    var bestv = tmp[0]
                    var j = 1
                    while j < tmp.size():
                        if tmp[j] < bestv:
                            best = j
                            bestv = tmp[j]
                        j = j + 1
                    sorted.push_back(bestv)
                    var rest = List[Int]()
                    for k in range(tmp.size()):
                        if k != best: rest.push_back(tmp[k])
                    tmp = rest

                blk = self._remove_indices(blk, sorted, uc)
                changed = True

        return blk

    fn run_on_function(self, f: Function) -> Function:
        for bi in range(f.blocks.size()):
            f.blocks[bi] = self.run_on_block(f.blocks[bi])
        return f

    fn run_on_module(self, m: Module) -> Module:
        var out = Module(m.name + String("_dce"))
        for fi in range(m.functions.size()):
            var f = m.functions[fi]
            var nf = Function(f.name)
            for ai in range(f.arg_types.size()): nf.arg_types.push_back(f.arg_types[ai])
            for ri in range(f.ret_types.size()): nf.ret_types.push_back(f.ret_types[ri])
            for bi in range(f.blocks.size()): nf.blocks.push_back(f.blocks[bi])
            out.functions.push_back(self.run_on_function(nf))
        return out

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
                            s = s + self._value(op.operands[pi].value)
                            if pi + 1 < op.operands.size(): s = s + String(", ")
                        s = s + String(")")
                    if op.results.size() > 0:
                        s = s + String(" -> ")
                        for ri2 in range(op.results.size()):
                            s = s + self._value(op.results[ri2].value)
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

# -----------------------------
# Self-test
# -----------------------------

struct IdGen:
    var next_id: Int
    fn __init__(out self): self.next_id = 0
    fn fresh(self) -> Int:
        var r = self.next_id
        self.next_id = self.next_id + 1
        return r

fn _demo_module() -> Module:
    var m = Module(String("demo_dce"))
    var f = Function(String("main"))
    var b = Block(String("entry"))
    var idg = IdGen()

    # const 1, 2, and a dead const 99
    var c1 = Op(String("ll.const.i64"), Location(String("d"), 1, 1))
    var v1 = Value(idg.fresh(), TypeDesc(String("i64"))); c1.results.push_back(Result(v1)); c1.attrs.push_back(String("value=1"))
    var c2 = Op(String("ll.const.i64"), Location(String("d"), 1, 2))
    var v2 = Value(idg.fresh(), TypeDesc(String("i64"))); c2.results.push_back(Result(v2)); c2.attrs.push_back(String("value=2"))
    var cdead = Op(String("ll.const.i64"), Location(String("d"), 1, 3))
    var vdead = Value(idg.fresh(), TypeDesc(String("i64"))); cdead.results.push_back(Result(vdead)); cdead.attrs.push_back(String("value=99"))

    # add used only by store (effectful) → must be kept
    var add1 = Op(String("ll.add"), Location(String("d"), 1, 4))
    var vadd1 = Value(idg.fresh(), TypeDesc(String("i64"))); add1.operands.push_back(Operand(v1)); add1.operands.push_back(Operand(v2)); add1.results.push_back(Result(vadd1))

    var outb = Value(idg.fresh(), TypeDesc(String("buffer<i64,[?]>")))
    var idx = Value(idg.fresh(), TypeDesc(String("i64")))
    var st = Op(String("buf.store"), Location(String("d"), 1, 5))
    st.operands.push_back(Operand(vadd1)); st.operands.push_back(Operand(outb)); st.operands.push_back(Operand(idx))

    # redundant add (unused) → should be removed
    var add_dead = Op(String("ll.add"), Location(String("d"), 1, 6))
    var vadd_dead = Value(idg.fresh(), TypeDesc(String("i64"))); add_dead.operands.push_back(Operand(v1)); add_dead.operands.push_back(Operand(v2)); add_dead.results.push_back(Result(vadd_dead))

    # an unused note op
    var note = Op(String("cse.removed"), Location(String("d"), 1, 7))
    var vnote = Value(idg.fresh(), TypeDesc(String("i64"))); note.results.push_back(Result(vnote))

    # ret uses v1 so c1 must remain
    var ret = Op(String("ll.ret"), Location(String("d"), 1, 8))
    ret.operands.push_back(Operand(v1))

    b.ops.push_back(c1); b.ops.push_back(c2); b.ops.push_back(cdead)
    b.ops.push_back(add1); b.ops.push_back(st)
    b.ops.push_back(add_dead); b.ops.push_back(note)
    b.ops.push_back(ret)

    f.blocks.push_back(b)
    m.functions.push_back(f)
    return m

fn _self_test_dce() -> Bool:
    var m = _demo_module()
    var dce = DCE()
    var out = dce.run_on_module(m)

    var pr = Printer()
    var txt = pr.print(out)

    var ok = True
    # dead const 99 removed
    if txt.find(String("value=99")) >= 0: ok = False
    # dead add removed
    if txt.find(String("ll.add"), txt.find(String("buf.store"))) >= 0 and txt.find(String("ll.add"), txt.find(String("buf.store"))) < txt.find(String("ll.ret")):
        # There is at least one add; ensure only the first (used by store) remains by checking count
        # crude check: "ll.add" should appear only once
        var first = txt.find(String("ll.add"))
        var second = txt.find(String("ll.add"), first + 1)
        if second >= 0: ok = False
    # note removed
    if txt.find(String("cse.removed")) >= 0: ok = False
    # store and ret remain
    if txt.find(String("buf.store")) < 0: ok = False
    if txt.find(String("ll.ret")) < 0: ok = False

    if ok: print(String("OK"))
    else: print(String("FAIL"))
    return ok
 
