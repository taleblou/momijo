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
# File: src/momijo/ir/tiling.mojo
# Description: Loop tiling pass (per-block). Converts a simple 1-D loop.for into
#              a two-level tiled nest. Keeps IR self-contained and only rewrites
#              the loop bounds; body uses of the original IV are remapped to the
#              inner IV. Step must be 1 (for now). Adds a small note:
#              `tiling.note {tile=<int>}` after transformation.
# Notes:
#   - Self-contained fallback IR; no globals, no 'export'; only 'var' (no 'let').
#   - Constructors use: fn __init__(out self, ...)
#   - Pattern:
#         %iv = loop.for(%lo,%hi,%one)
#           BODY(%iv)
#         loop.end
#     →
#         %tstep = const tile
#         %iv0 = loop.for(%lo,%hi,%tstep)              // outer
#           %end = min(%hi, add(%iv0, %tstep))         // clamp tail
#           %iv = loop.for(%iv0,%end,%one)             // inner
#             BODY(%iv)                                // iv remapped
#           loop.end
#         loop.end
#   - Compatible with later passes (schedule, memory_coalesce, ...).
#   - Includes tiny printer and a self-test (prints OK).
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

struct IdGen:
    var next_id: Int
    fn __init__(out self): self.next_id = 0
    fn fresh(self) -> Int:
        var r = self.next_id
        self.next_id = self.next_id + 1
        return r

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

# find int value of ll.const.i64 producing given Value id
fn _const_i64_of(b: Block, v: Value, out found: Bool) -> Int:
    found = False
    for i in range(b.ops.size()):
        var op = b.ops[i]
        if op.name == String("ll.const.i64") and op.results.size() > 0 and op.results[0].value.id == v.id:
            var has = False; var s = _attr_value(op, String("value"), has)
            if has:
                var ok = False; var val = _parse_i64(s, ok)
                if ok: found = True; return val
    return 0

fn _i64_const(value: Int, loc: Location, out op: Op, out v: Value) -> None:
    op = Op(String("ll.const.i64"), loc)
    v = Value(-1, TypeDesc(String("i64")))
    op.results.push_back(Result(v))
    op.attrs.push_back(String("value=") + String(value))

# Find the matching loop.end for a loop.for at index 'i' (handles nesting)
fn _find_loop_end(b: Block, start_idx: Int) -> Int:
    var depth = 0
    for i in range(start_idx, b.ops.size()):
        var n = b.ops[i].name
        if n == String("loop.for"):
            depth = depth + 1
        elif n == String("loop.end"):
            depth = depth - 1
            if depth == 0:
                return i
    return -1

# -----------------------------
# Tiling pass
# -----------------------------

struct TilingOptions:
    var tile_size: Int
    fn __init__(out self):
        self.tile_size = 16

struct Tiling:
    var opt: TilingOptions
    fn __init__(out self, opt: TilingOptions = TilingOptions()):
        self.opt = opt

    fn _tile_simple_for(self, b: Block, for_idx: Int) -> Bool:
        # Preconditions
        var op_for = b.ops[for_idx]
        if op_for.name != String("loop.for"): return False
        if op_for.operands.size() < 3 or op_for.results.size() < 1: return False

        var end_idx = _find_loop_end(b, for_idx)
        if end_idx < 0: return False

        var lo = op_for.operands[0].value
        var hi = op_for.operands[1].value
        var step = op_for.operands[2].value
        var iv = op_for.results[0].value

        # Require step == 1 (constant)
        var has1 = False; var step_val = _const_i64_of(b, step, has1)
        if not has1 or step_val != 1: return False

        # Build new sequence:
        var new_ops = List[Op]()

        # const tile
        var cT = Op(String("ll.const.i64"), op_for.loc)
        var vT = Value(-1, TypeDesc(String("i64")))
        cT.results.push_back(Result(vT))
        cT.attrs.push_back(String("value=") + String(self.opt.tile_size))
        new_ops.push_back(cT)

        # outer loop: for (lo, hi, tile)
        var outer = Op(String("loop.for"), op_for.loc)
        var iv0 = Value(-1, TypeDesc(String("i64"))); outer.results.push_back(Result(iv0))
        outer.operands.push_back(Operand(lo)); outer.operands.push_back(Operand(hi)); outer.operands.push_back(Operand(vT))
        new_ops.push_back(outer)

        # end bound for inner: add(iv0, T) ; min(hi, ...)
        var add = Op(String("ll.add"), op_for.loc)
        var v_add = Value(-1, TypeDesc(String("i64"))); add.results.push_back(Result(v_add))
        add.operands.push_back(Operand(iv0)); add.operands.push_back(Operand(vT))
        new_ops.push_back(add)

        var vmin = Op(String("ll.min"), op_for.loc)
        var v_end = Value(-1, TypeDesc(String("i64"))); vmin.results.push_back(Result(v_end))
        vmin.operands.push_back(Operand(hi)); vmin.operands.push_back(Operand(v_add))
        new_ops.push_back(vmin)

        # inner loop: for (iv0, end, 1)
        var c1 = Op(String("ll.const.i64"), op_for.loc)
        var v1 = Value(-1, TypeDesc(String("i64"))); c1.results.push_back(Result(v1)); c1.attrs.push_back(String("value=1"))
        new_ops.push_back(c1)

        var inner = Op(String("loop.for"), op_for.loc)
        var iv_inner = Value(-1, TypeDesc(String("i64"))); inner.results.push_back(Result(iv_inner))
        inner.operands.push_back(Operand(iv0)); inner.operands.push_back(Operand(v_end)); inner.operands.push_back(Operand(v1))
        new_ops.push_back(inner)

        # body: copy original body with iv replaced -> iv_inner
        for i in range(for_idx + 1, end_idx):
            var body_op = b.ops[i]
            # rewrite operands
            for oi in range(body_op.operands.size()):
                var ov = body_op.operands[oi].value
                if ov.id == iv.id:
                    ov = iv_inner
                body_op.operands[oi] = Operand(ov)
            new_ops.push_back(body_op)

        # inner end
        var inner_end = Op(String("loop.end"), b.ops[end_idx].loc)
        new_ops.push_back(inner_end)

        # outer end
        var outer_end = Op(String("loop.end"), b.ops[end_idx].loc)
        new_ops.push_back(outer_end)

        # note
        var note = Op(String("tiling.note"), b.ops[end_idx].loc)
        note.attrs.push_back(String("tile=") + String(self.opt.tile_size))
        new_ops.push_back(note)

        # splice into block
        var out_ops = List[Op]()
        for i in range(b.ops.size()):
            if i == for_idx:
                # insert new seq
                for j in range(new_ops.size()): out_ops.push_back(new_ops[j])
            elif i > for_idx and i <= end_idx:
                # drop original for-body-end
                continue
            else:
                out_ops.push_back(b.ops[i])
        b.ops = out_ops
        return True

    fn run_on_block(self, b: Block) -> Block:
        var i = 0
        while i < b.ops.size():
            if b.ops[i].name == String("loop.for"):
                var changed = self._tile_simple_for(b, i)
                if changed:
                    # move past the sequence we just inserted (heuristic jump)
                    i = i + 7  # constT, outer, add, min, const1, inner, (body...), inner_end, outer_end, note → jump enough
                    continue
            i = i + 1
        return b

    fn run_on_function(self, f: Function) -> Function:
        for bi in range(f.blocks.size()):
            f.blocks[bi] = self.run_on_block(f.blocks[bi])
        return f

    fn run_on_module(self, m: Module) -> Module:
        var out = Module(m.name + String("_tiled"))
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

fn _buffer_i64_1d(n: Int) -> TypeDesc:
    return TypeDesc(String("buffer<i64,[") + String(n) + String("]>"))

struct IdGen:
    var next_id: Int
    fn __init__(out self): self.next_id = 0
    fn fresh(self) -> Int:
        var r = self.next_id
        self.next_id = self.next_id + 1
        return r

fn _demo_loop(N: Int) -> Module:
    var m = Module(String("demo_tile"))
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

fn _self_test_tiling() -> Bool:
    var m = _demo_loop(64)
    var opt = TilingOptions()
    opt.tile_size = 16
    var pass = Tiling(opt)
    var out = pass.run_on_module(m)

    var pr = Printer()
    var txt = pr.print(out)

    var ok = True
    # Heuristic checks
    # At least two loop.for (outer + inner)
    var first = txt.find(String("loop.for"))
    var second = txt.find(String("loop.for"), first + 1)
    if first < 0 or second < 0: ok = False
    # tile constant and note
    if txt.find(String("value=16")) < 0: ok = False
    if txt.find(String("tiling.note {tile=16}")) < 0: ok = False
    # min/add present for tail handling
    if txt.find(String("ll.add")) < 0: ok = False
    if txt.find(String("ll.min")) < 0: ok = False
    # original single for should be replaced; there should be at least one loop.end more
    var end1 = txt.find(String("loop.end"))
    var end2 = txt.find(String("loop.end"), end1 + 1)
    if end1 < 0 or end2 < 0: ok = False

    if ok: print(String("OK"))
    else: print(String("FAIL"))
    return ok
 
