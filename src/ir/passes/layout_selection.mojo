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
# File: src/momijo/ir/layout_selection.mojo
# Description: Heuristic memory-layout selection for buffers/tensors.
#              Annotates buffer values with layout metadata (layout tag, strides,
#              vector width, alignment). For buf.alloc ops, attributes are added
#              directly; for block arguments, a 'layout.hint' op is inserted.
# Notes:
#   - Self-contained fallback IR; no globals, no 'export'; only 'var' (no 'let').
#   - Constructors use: fn __init__(out self, ...)
#   - Recognizes shapes from type strings like: buffer<i64,[N,H,W,C]>
#   - Heuristics:
#       * If function contains any '*conv2d*' op → prefer 'nhwc' for 4-D buffers,
#         otherwise 'row-major' (last-dimension contiguous).
#       * Respect existing 'layout='/'strides=' on buf.alloc (don't override).
#       * Vector width: if last dimension known and divisible by 4 → 4 else 1.
#       * Alignment: 64 bytes for i64/f64 buffers, otherwise 16.
#   - Includes tiny printer & a self-test (prints OK).
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
# Helpers
# -----------------------------

fn is_buffer(t: TypeDesc) -> Bool: return t.name.starts_with(String("buffer<"))
fn is_tensor(t: TypeDesc) -> Bool: return t.name.starts_with(String("tensor<"))
fn is_i64(t: TypeDesc) -> Bool: return t.name == String("i64")
fn is_f64(t: TypeDesc) -> Bool: return t.name == String("f64")

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

struct ShapeInfo:
    var dtype: String
    var dims: List[Int]   # -1 means unknown
    fn __init__(out self):
        self.dtype = String("")
        self.dims = List[Int]()

fn _parse_int_token(tok: String, out ok: Bool) -> Int:
    ok = False
    if tok.size() == 0: return 0
    if tok == String("?"):
        ok = True; return -1
    var sign = 1
    var i = 0
    if tok.bytes()[0] == 45: sign = -1; i = 1
    var num = 0
    while i < tok.size():
        var ch = tok.bytes()[i]
        if ch < 48 or ch > 57: return 0
        num = num * 10 + Int(ch - 48)
        i = i + 1
    ok = True
    return sign * num

fn _parse_buffer_type(t: TypeDesc, out info: ShapeInfo, out ok: Bool) -> None:
    ok = False
    info = ShapeInfo()
    var s = t.name
    if not s.starts_with(String("buffer<")) and not s.starts_with(String("tensor<")): return
    # buffer<DT,[...]> OR tensor<DT,[...]>
    # find first '<' and first ',' and '[' and ']'
    var lt = s.find(String("<")); if lt < 0: return
    var comma = s.find(String(","), lt+1); if comma < 0: return
    var lbr = s.find(String("["), comma+1); if lbr < 0: return
    var rbr = s.find(String("]"), lbr+1); if rbr < 0: return

    # dtype
    var dt = String("")
    var i = lt + 1
    while i < comma:
        dt = dt + String.from_utf8([s.bytes()[i]]); i = i + 1
    # dims list
    var dims = List[Int]()
    var tok = String("")
    i = lbr + 1
    while i < rbr:
        var ch = s.bytes()[i]
        if ch == 44:  # ','
            var ok1 = False; var v = _parse_int_token(tok, ok1)
            if ok1: dims.push_back(v)
            tok = String("")
        else:
            tok = tok + String.from_utf8([ch])
        i = i + 1
    if tok.size() > 0:
        var ok2 = False; var v2 = _parse_int_token(tok, ok2)
        if ok2: dims.push_back(v2)

    info.dtype = dt
    info.dims = dims
    ok = True
    return

fn _compute_row_major_strides(dims: List[Int]) -> List[Int]:
    var n = dims.size()
    var strides = List[Int]()
    # init with 0s
    for i in range(n): strides.push_back(0)
    var prod = 1
    var i = n - 1
    while i >= 0:
        strides[i] = prod if dims[i] >= 0 else -1  # -1 stride when unknown later sizes
        if dims[i] >= 0 and prod >= 0:
            prod = prod * dims[i]
        else:
            prod = -1
        i = i - 1
    return strides

fn _ints_to_bracket_string(vals: List[Int]) -> String:
    var s = String("[")
    for i in range(vals.size()):
        var vi = vals[i]
        if vi < 0: s = s + String("?")
        else: s = s + String(vi)
        if i + 1 < vals.size(): s = s + String(",")
    s = s + String("]")
    return s

# -----------------------------
# Layout selection
# -----------------------------

struct LayoutPlan:
    var tag: String        # "row-major" or "nhwc"
    var strides: List[Int] # -1 for unknown
    var vector: Int
    var align: Int
    fn __init__(out self):
        self.tag = String("row-major")
        self.strides = List[Int]()
        self.vector = 1
        self.align = 16

struct LayoutSelection:
    fn _prefer_nhwc_in_function(self, f: Function) -> Bool:
        for bi in range(f.blocks.size()):
            var b = f.blocks[bi]
            for oi in range(b.ops.size()):
                var n = b.ops[oi].name
                if n.find(String("conv2d")) >= 0: return True
        return False

    fn _choose_plan(self, typ: TypeDesc, prefer_nhwc: Bool) -> LayoutPlan:
        var info = ShapeInfo(); var ok = False
        _parse_buffer_type(typ, info, ok)
        var plan = LayoutPlan()
        plan.align = (info.dtype == String("i64") or info.dtype == String("f64")) ? 64 : 16
        var strides = _compute_row_major_strides(info.dims)
        plan.strides = strides
        # tag
        if prefer_nhwc and info.dims.size() == 4:
            plan.tag = String("nhwc")
        else:
            plan.tag = String("row-major")
        # vector width
        if info.dims.size() > 0:
            var last = info.dims[info.dims.size()-1]
            if last >= 0 and (last % 4 == 0):
                plan.vector = 4
            else:
                plan.vector = 1
        return plan

    fn _annotate_alloc(self, op: Op, plan: LayoutPlan) -> Op:
        # don't override if already present
        if not _has_attr(op, String("layout")):
            op.attrs.push_back(String("layout=") + plan.tag)
        if not _has_attr(op, String("strides")) and plan.strides.size() > 0:
            op.attrs.push_back(String("strides=") + _ints_to_bracket_string(plan.strides))
        if not _has_attr(op, String("vector")):
            op.attrs.push_back(String("vector=") + String(plan.vector))
        if not _has_attr(op, String("align")):
            op.attrs.push_back(String("align=") + String(plan.align))
        return op

    fn _make_hint(self, v: Value, plan: LayoutPlan, loc: Location) -> Op:
        var op = Op(String("layout.hint"), loc)
        op.operands.push_back(Operand(v))
        op.attrs.push_back(String("layout=") + plan.tag)
        if plan.strides.size() > 0: op.attrs.push_back(String("strides=") + _ints_to_bracket_string(plan.strides))
        op.attrs.push_back(String("vector=") + String(plan.vector))
        op.attrs.push_back(String("align=") + String(plan.align))
        return op

    fn run_on_function(self, f: Function) -> Function:
        var prefer_nhwc = self._prefer_nhwc_in_function(f)
        for bi in range(f.blocks.size()):
            var b = f.blocks[bi]
            # hints for block args that are buffers
            var insert_at = 0
            for ai in range(b.args.size()):
                var a = b.args[ai]
                if is_buffer(a.typ) or is_tensor(a.typ):
                    var plan = self._choose_plan(a.typ, prefer_nhwc)
                    var hint = self._make_hint(a, plan, Location(String(f.name), 0, 0))
                    b.ops.insert(insert_at, hint)
                    insert_at = insert_at + 1
            # annotate buf.alloc
            for oi in range(b.ops.size()):
                var op = b.ops[oi]
                if op.name == String("buf.alloc") and op.results.size() > 0:
                    var v = op.results[0].value
                    var plan2 = self._choose_plan(v.typ, prefer_nhwc)
                    b.ops[oi] = self._annotate_alloc(op, plan2)
            f.blocks[bi] = b
        return f

    fn run_on_module(self, m: Module) -> Module:
        var out = Module(m.name + String("_layout"))
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

fn _buffer_i64_4d() -> TypeDesc: return TypeDesc(String("buffer<i64,[2,3,4,8]>"))
fn _buffer_f64_2d() -> TypeDesc: return TypeDesc(String("buffer<f64,[64,128]>"))

fn _demo_module() -> Module:
    var m = Module(String("demo_layout"))
    var f = Function(String("main"))
    var b = Block(String("entry"))
    var idg = IdGen()

    # Args: out, a
    var outb = Value(idg.fresh(), _buffer_i64_4d())
    var a = Value(idg.fresh(), _buffer_i64_4d())
    b.args.push_back(outb); b.args.push_back(a)

    # alloc tmp : buffer<f64,[64,128]>
    var tmp = Value(idg.fresh(), _buffer_f64_2d())
    var alloc = Op(String("buf.alloc"), Location(String("demo.mojo"), 1, 1))
    alloc.results.push_back(Result(tmp))
    b.ops.push_back(alloc)

    # a conv2d op to trigger NHWC preference
    var conv = Op(String("ll.conv2d"), Location(String("demo.mojo"), 1, 2))
    b.ops.push_back(conv)

    var ret = Op(String("ll.ret"), Location(String("demo.mojo"), 1, 3))
    b.ops.push_back(ret)

    f.blocks.push_back(b)
    m.functions.push_back(f)
    return m

fn _self_test_layout() -> Bool:
    var m = _demo_module()
    var pass = LayoutSelection()
    var out = pass.run_on_module(m)

    var pr = Printer()
    var txt = pr.print(out)

    var ok = True
    # expect layout.hint with nhwc + strides for 4D arg
    if txt.find(String("layout.hint")) < 0: ok = False
    if txt.find(String("layout=nhwc")) < 0: ok = False
    if txt.find(String("strides=[96,32,8,1]")) < 0: ok = False
    # expect buf.alloc annotated (row-major for 2D)
    if txt.find(String("buf.alloc")) < 0: ok = False
    if txt.find(String("layout=row-major")) < 0: ok = False
    if txt.find(String("strides=[128,1]")) < 0: ok = False
    if ok: print(String("OK"))
    else: print(String("FAIL"))
    return ok

 
