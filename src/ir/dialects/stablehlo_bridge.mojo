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
# File: src/momijo/ir/stablehlo_bridge.mojo
# Description: Bridge from Momijo HLIR/core-IR to a StableHLO-like textual form.
# Notes:
#   - Self-contained fallback IR model + simple printer for StableHLO text.
#   - Only 'var' (no 'let'), no globals, no 'export'.
#   - Constructors: fn __init__(out self, ...)
#   - This is a "compatible textual form" (not claiming full StableHLO coverage).
# ============================================================================

# -----------------------------
# Minimal IR model (fallback)
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
# Type helpers (scalar + tensor encode/decode)
# -----------------------------

fn is_tensor(t: TypeDesc) -> Bool:
    return t.name.starts_with(String("tensor<"))

fn is_i64(t: TypeDesc) -> Bool: return t.name == String("i64")
fn is_f64(t: TypeDesc) -> Bool: return t.name == String("f64")
fn is_bool(t: TypeDesc) -> Bool: return t.name == String("bool")

enum DType(Int):
    Invalid = 0
    I64 = 1
    F64 = 2
    Bool = 3

fn dtype_of_scalar(t: TypeDesc) -> DType:
    if is_i64(t): return DType.I64
    if is_f64(t): return DType.F64
    if is_bool(t): return DType.Bool
    return DType.Invalid

fn _parse_between(s: String, start_ch: UInt8, end_ch: UInt8, out ok: Bool) -> String:
    var i = 0
    var depth = 0
    var start_idx = -1
    var end_idx = -1
    ok = False
    while i < s.size():
        var ch = s.bytes()[i]
        if ch == start_ch:
            if depth == 0: start_idx = i + 1
            depth = depth + 1
        elif ch == end_ch:
            depth = depth - 1
            if depth == 0:
                end_idx = i
                ok = True
                break
        i = i + 1
    if not ok or start_idx < 0 or end_idx < 0 or end_idx <= start_idx:
        return String("")
    var out_s = String("")
    var j = start_idx
    while j < end_idx:
        out_s = out_s + String.from_utf8([s.bytes()[j]])
        j = j + 1
    return out_s

struct Shape:
    var dims: List[Int]
    fn __init__(out self): self.dims = List[Int]()
    fn rank(self) -> Int: return self.dims.size()

fn tensor_unpack(t: TypeDesc, out ok: Bool, out out_dt: DType, out out_sh: Shape) -> None:
    ok = False
    out_dt = DType.Invalid
    out_sh = Shape()
    if not is_tensor(t): return
    var inner = _parse_between(t.name, 60, 62, ok)  # '<','>'
    if not ok: return
    var comma = inner.find(String(","))
    if comma < 0: return

    var dts = String("")
    var i = 0
    while i < comma:
        dts = dts + String.from_utf8([inner.bytes()[i]])
        i = i + 1
    dts = dts.strip()

    var rest = String("")
    i = comma + 1
    while i < inner.size():
        rest = rest + String.from_utf8([inner.bytes()[i]])
        i = i + 1
    rest = rest.strip()

    if dts == String("i64"): out_dt = DType.I64
    elif dts == String("f64"): out_dt = DType.F64
    elif dts == String("bool"): out_dt = DType.Bool
    else: out_dt = DType.Invalid

    var ok2 = False
    var ss = _parse_between(rest, 91, 93, ok2)  # '[',']'
    if not ok2: return

    var cur = String("")
    for k in range(ss.size() + 1):
        var is_end = k == ss.size()
        var ch = UInt8(0)
        if not is_end: ch = ss.bytes()[k]
        if is_end or ch == 44:  # ','
            var tok = cur.strip()
            if tok.size() > 0:
                var sign = 1
                var idx = 0
                if tok.size() > 0 and tok.bytes()[0] == 45:  # '-'
                    sign = -1
                    idx = 1
                var num = 0
                while idx < tok.size():
                    var d = tok.bytes()[idx] - UInt8(48)
                    num = num * 10 + Int(d)
                    idx = idx + 1
                out_sh.dims.push_back(sign * num)
            cur = String("")
        else:
            cur = cur + String.from_utf8([ch])
    ok = True

# -----------------------------
# StableHLO textual helpers
# -----------------------------

fn _hlo_scalar_type(dt: DType) -> String:
    if dt == DType.I64: return String("i64")
    if dt == DType.F64: return String("f64")
    if dt == DType.Bool: return String("i1")  # bool as i1
    return String("invalid")

fn _hlo_tensor_type(dt: DType, sh: Shape) -> String:
    var s = String("tensor<") + _hlo_scalar_type(dt) + String("[")
    for i in range(sh.dims.size()):
        var d = sh.dims[i]
        if d < 0: s = s + String("?")
        else: s = s + String(d)
        if i + 1 < sh.dims.size(): s = s + String("x")
    s = s + String("]>")
    return s

fn _hlo_from_typedesc(t: TypeDesc) -> String:
    if is_tensor(t):
        var ok = False; var dt = DType.Invalid; var sh = Shape()
        tensor_unpack(t, ok, dt, sh)
        if not ok: return String("tensor<invalid[]>")
        return _hlo_tensor_type(dt, sh)
    # scalars are printed as 0-d tensors
    var sdt = dtype_of_scalar(t)
    return String("tensor<") + _hlo_scalar_type(sdt) + String("[]>")

# -----------------------------
# Op mapping
# -----------------------------

fn _attr_value(op: Op, key: String, out found: Bool) -> String:
    found = False
    var out_s = String("")
    for i in range(op.attrs.size()):
        # expect "key=VALUE"
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

fn _map_bin_arith_name(op: String) -> String:
    if op == String("hl.add.i64") or op == String("hl.add.f64"): return String("stablehlo.add")
    if op == String("hl.sub.i64") or op == String("hl.sub.f64"): return String("stablehlo.subtract")
    if op == String("hl.mul.i64") or op == String("hl.mul.f64"): return String("stablehlo.multiply")
    if op == String("hl.div.i64") or op == String("hl.div.f64"): return String("stablehlo.divide")
    return String("stablehlo.unknown")

fn _map_cmp_dir(op: String, out found: Bool) -> String:
    found = True
    if op == String("hl.cmp.eq"): return String("EQ")
    if op == String("hl.cmp.lt"): return String("LT")
    if op == String("hl.cmp.gt"): return String("GT")
    found = False
    return String("")

# -----------------------------
# Bridge
# -----------------------------

struct StableHloBridge:
    fn _value_name(self, v: Value) -> String:
        return String("%") + String(v.id)

    fn _fmt_operands(self, op: Op) -> String:
        var s = String("")
        for i in range(op.operands.size()):
            s = s + self._value_name(op.operands[i].value)
            if i + 1 < op.operands.size(): s = s + String(", ")
        return s

    fn _fmt_results(self, op: Op) -> String:
        var s = String("")
        for i in range(op.results.size()):
            s = s + self._value_name(op.results[i].value)
            if i + 1 < op.results.size(): s = s + String(", ")
        return s

    fn _map_op(self, op: Op) -> String:
        # Constants
        if op.name == String("hl.const.i64") or op.name == String("hl.const.f64") or op.name == String("hl.const.bool"):
            var has = False
            var val = _attr_value(op, String("value"), has)
            if not has: val = String("?")
            var ty = _hlo_from_typedesc(op.results[0].value.typ)
            var lhs = self._fmt_results(op)
            return lhs + String(" = stablehlo.constant(") + val + String(") : ") + ty

        # Binary arithmetic
        var ar = _map_bin_arith_name(op.name)
        if ar != String("stablehlo.unknown"):
            var ty = _hlo_from_typedesc(op.results[0].value.typ)
            var lhs = self._fmt_results(op)
            var rhs = self._fmt_operands(op)
            return lhs + String(" = ") + ar + String("(") + rhs + String(") : ") + ty

        # Compare
        var has_dir = False
        var dir = _map_cmp_dir(op.name, has_dir)
        if has_dir:
            var ty = _hlo_from_typedesc(op.results[0].value.typ)
            var lhs = self._fmt_results(op)
            var rhs = self._fmt_operands(op)
            return lhs + String(" = stablehlo.compare(") + rhs + String(") {comparison_direction=") + dir + String("} : ") + ty

        # Select
        if op.name == String("hl.select"):
            var ty = _hlo_from_typedesc(op.results[0].value.typ)
            var lhs = self._fmt_results(op)
            var rhs = self._fmt_operands(op)
            return lhs + String(" = stablehlo.select(") + rhs + String(") : ") + ty

        # Return
        if op.name == String("hl.return") or op.name == String("return"):
            var rhs2 = self._fmt_operands(op)
            return String("stablehlo.return ") + rhs2

        # Fallback
        var s = String("// unmapped op: ") + op.name
        return s

    fn _print_function(self, f: Function) -> String:
        var s = String("  func @") + f.name + String("(")
        for i in range(f.arg_types.size()):
            s = s + _hlo_from_typedesc(f.arg_types[i])
            if i + 1 < f.arg_types.size(): s = s + String(", ")
        s = s + String(") -> (")
        for i in range(f.ret_types.size()):
            s = s + _hlo_from_typedesc(f.ret_types[i])
            if i + 1 < f.ret_types.size(): s = s + String(", ")
        s = s + String(") {\n")

        for bi in range(f.blocks.size()):
            var b = f.blocks[bi]
            s = s + String("    ^") + b.name + String("(")
            for ai in range(b.args.size()):
                s = s + _hlo_from_typedesc(b.args[ai].typ)
                if ai + 1 < b.args.size(): s = s + String(", ")
            s = s + String("):\n")
            for oi in range(b.ops.size()):
                s = s + String("      ") + self._map_op(b.ops[oi]) + String("\n")

        s = s + String("  }\n")
        return s

    fn to_stablehlo_text(self, m: Module) -> String:
        var s = String("stablehlo.module @") + m.name + String(" {\n")
        for i in range(m.functions.size()):
            s = s + self._print_function(m.functions[i])
        s = s + String("}\n")
        return s

# -----------------------------
# Self-test: build tiny HLIR-ish module and bridge it
# -----------------------------

struct IdGen:
    var next_id: Int
    fn __init__(out self): self.next_id = 0
    fn fresh(self) -> Int:
        var r = self.next_id
        self.next_id = self.next_id + 1
        return r

fn _demo_module() -> Module:
    var m = Module(String("demo"))
    var f = Function(String("arith"))
    f.arg_types.push_back(TypeDesc(String("i64")))
    f.arg_types.push_back(TypeDesc(String("i64")))
    f.ret_types.push_back(TypeDesc(String("i64")))

    var idg = IdGen()
    var entry = Block(String("entry"))
    # block args
    var a = Value(idg.fresh(), TypeDesc(String("i64")))
    var b = Value(idg.fresh(), TypeDesc(String("i64")))
    entry.args.push_back(a)
    entry.args.push_back(b)

    var add = Op(String("hl.add.i64"), Location(String("demo.mojo"), 1, 1))
    add.operands.push_back(Operand(a))
    add.operands.push_back(Operand(b))
    var r0 = Value(idg.fresh(), TypeDesc(String("i64")))
    add.results.push_back(Result(r0))

    var cst = Op(String("hl.const.i64"), Location(String("demo.mojo"), 1, 2))
    var r1 = Value(idg.fresh(), TypeDesc(String("i64")))
    cst.results.push_back(Result(r1))
    cst.attrs.push_back(String("value=10"))

    var mul = Op(String("hl.mul.i64"), Location(String("demo.mojo"), 1, 3))
    mul.operands.push_back(Operand(r0))
    mul.operands.push_back(Operand(r1))
    var r2 = Value(idg.fresh(), TypeDesc(String("i64")))
    mul.results.push_back(Result(r2))

    var ret = Op(String("hl.return"), Location(String("demo.mojo"), 1, 4))
    ret.operands.push_back(Operand(r2))

    entry.ops.push_back(add)
    entry.ops.push_back(cst)
    entry.ops.push_back(mul)
    entry.ops.push_back(ret)

    f.blocks.push_back(entry)
    m.functions.push_back(f)
    return m

fn _self_test_stablehlo_bridge() -> Bool:
    var m = _demo_module()
    var br = StableHloBridge()
    var txt = br.to_stablehlo_text(m)

    var ok = True
    if txt.find(String("stablehlo.module @demo")) < 0: ok = False
    if txt.find(String("func @arith")) < 0: ok = False
    if txt.find(String("stablehlo.add")) < 0: ok = False
    if txt.find(String("stablehlo.multiply")) < 0: ok = False
    if txt.find(String("stablehlo.constant")) < 0: ok = False
    if txt.find(String("stablehlo.return")) < 0: ok = False

    if ok: print(String("OK"))
    else: print(String("FAIL"))
    return ok
 
