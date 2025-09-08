# Project:      Momijo
# Module:       src.momijo.ir.llir.codegen_rocm
# File:         codegen_rocm.mojo
# Path:         src/momijo/ir/llir/codegen_rocm.mojo
#
# Description:  src.momijo.ir.llir.codegen_rocm — focused Momijo functionality with a stable public API.
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
# Type helpers (tensor encode/decode)
# -----------------------------
fn is_tensor(t: TypeDesc) -> Bool: return t.name.starts_with(String("tensor<"))
fn is_i64(t: TypeDesc) -> Bool: return t.name == String("i64")
fn is_f64(t: TypeDesc) -> Bool: return t.name == String("f64")
fn is_bool(t: TypeDesc) -> Bool: return t.name == String("bool")

enum DType(Int):
    Invalid = 0
    I64 = 1
    F64 = 2
    Bool = 3

struct Shape:
    var dims: List[Int]
fn __init__(out self) -> None: self.dims = List[Int]()
fn rank(self) -> Int: return self.dims.size()
fn __copyinit__(out self, other: Self) -> None:
        self.dims = other.dims
fn __moveinit__(out self, deinit other: Self) -> None:
        self.dims = other.dims
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
fn hip_scalar_type(dt: DType) -> String:
    if dt == DType.I64: return String("long long")
    if dt == DType.F64: return String("double")
    if dt == DType.Bool: return String("bool")
    return String("/*invalid*/void*")
fn hip_type_of(t: TypeDesc) -> String:
    if is_tensor(t):
        var ok = False; var dt = DType.Invalid; var sh = Shape()
        tensor_unpack(t, ok, dt, sh)
        if not ok: return String("/*tensor*/void*")
        return hip_scalar_type(dt) + String("*")
    if is_i64(t): return String("long long")
    if is_f64(t): return String("double")
    if is_bool(t): return String("bool")
    return String("/*invalid*/void*")
fn elem_hip_type(t: TypeDesc) -> String:
    if is_tensor(t):
        var ok = False; var dt = DType.Invalid; var sh = Shape()
        tensor_unpack(t, ok, dt, sh)
        if not ok: return String("/*invalid*/void*")
        return hip_scalar_type(dt)
    return hip_type_of(t)

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

# -----------------------------
# Codegen options
# -----------------------------

struct CodegenOptions:
    var emit_headers: Bool
    var block_dim_x: Int
    var kernel_suffix: String
    var launcher_prefix: String
fn __init__(out self,
                emit_headers: Bool = True,
                block_dim_x: Int = 256,
                kernel_suffix: String = String("_kernel"),
                launcher_prefix: String = String("run_")):
        self.emit_headers = emit_headers
        self.block_dim_x = block_dim_x
        self.kernel_suffix = kernel_suffix
        self.launcher_prefix = launcher_prefix
fn __copyinit__(out self, other: Self) -> None:
        self.emit_headers = other.emit_headers
        self.block_dim_x = other.block_dim_x
        self.kernel_suffix = other.kernel_suffix
        self.launcher_prefix = other.launcher_prefix
fn __moveinit__(out self, deinit other: Self) -> None:
        self.emit_headers = other.emit_headers
        self.block_dim_x = other.block_dim_x
        self.kernel_suffix = other.kernel_suffix
        self.launcher_prefix = other.launcher_prefix
# -----------------------------
# HIP Code Generator
# -----------------------------

struct CodegenROCm:
    var opt: CodegenOptions
fn __init__(out self, opt: CodegenOptions = CodegenOptions()):
        self.opt = opt
fn _vname(self, v: Value) -> String:
        return String("v") + String(v.id)
fn _argname(self, i: Int) -> String:
        return String("a") + String(i)
fn _emit_headers(self) -> String:
        var s = String("")
        if self.opt.emit_headers:
            s = s + String("/* Generated by Momijo CodegenROCm (HIP) */\n")
            s = s + String("#include <hip/hip_runtime.h>\n")
            s = s + String("#include <stdint.h>\n")
            s = s + String("#include <stdbool.h>\n\n")
        return s
fn _kernel_sig(self, fname: String, out_type: TypeDesc, args: List[TypeDesc], out needs_N: Bool) -> String:
        # __global__ void fname_kernel(OUT_T* out, const T0* a0, ..., int N?)
        var s = String("__global__ void ") + fname + self.opt.kernel_suffix + String("(")
        var ok = False; var dt = DType.Invalid; var sh = Shape()
        var out_elem = String("void")
        var out_is_tensor = is_tensor(out_type)
        if out_is_tensor:
            tensor_unpack(out_type, ok, dt, sh)
            out_elem = hip_scalar_type(dt)
        else:
            out_elem = hip_type_of(out_type)
        s = s + out_elem + String("* out")

        for i in range(args.size()):
            s = s + String(", ")
            var t = args[i]
            var ty = hip_type_of(t)
            if is_tensor(t):
                s = s + String("const ") + ty + String(" ") + self._argname(i)
            else:
                s = s + ty + String(" ") + self._argname(i)

        needs_N = False
        if out_is_tensor:
            if sh.rank() == 0 or sh.dims.size() == 0 or sh.dims[0] < 0:
                needs_N = True
        if needs_N:
            s = s + String(", int N")
        s = s + String(")")
        return s
fn _launcher_sig(self, fname: String, out_type: TypeDesc, args: List[TypeDesc], needs_N: Bool) -> String:
        # void run_fname(OUT_T* out, const T0* a0, ..., int N)
        var s = String("void ") + self.opt.launcher_prefix + fname + String("(")
        var ok = False; var dt = DType.Invalid; var sh = Shape()
        var out_elem = String("void")
        if is_tensor(out_type):
            tensor_unpack(out_type, ok, dt, sh)
            out_elem = hip_scalar_type(dt)
        else:
            out_elem = hip_type_of(out_type)
        s = s + out_elem + String("* out")
        for i in range(args.size()):
            s = s + String(", ")
            var t = args[i]
            var ty = hip_type_of(t)
            if is_tensor(t):
                s = s + String("const ") + ty + String(" ") + self._argname(i)
            else:
                s = s + ty + String(" ") + self._argname(i)
        if needs_N:
            s = s + String(", int N")
        s = s + String(")")
        return s
fn _idx_line(self) -> String:
        return String("  int idx = blockIdx.x * blockDim.x + threadIdx.x;")
fn _guard_open(self) -> String:
        return String("  if (idx < N) {")
fn _guard_close(self) -> String:
        return String("  }")
fn _expr_for_arg(self, v: Value, entry: Block, args: List[TypeDesc]) -> String:
        for i in range(entry.args.size()):
            if entry.args[i].id == v.id:
                if is_tensor(args[i]):
                    return self._argname(i) + String("[idx]")
                else:
                    return self._argname(i)
        return self._vname(v)
fn _emit_op_kernel(self, op: Op, entry: Block, args: List[TypeDesc]) -> String:
        var s = String("")
        var name = op.name
        # constants
        if name == String("hl.const.i64") or name == String("hl.const.f64") or name == String("hl.const.bool"):
            var has = False; var val = _attr_value(op, String("value"), has)
            if not has: val = String("0")
            var ty = elem_hip_type(op.results[0].value().typ)
            var lhs = self._vname(op.results[0].value())
            return String("  ") + ty + String(" ") + lhs + String(" = ") + val + String(";\n")

        # binary arithmetic
        var opch = String("")
        if name == String("hl.add.i64") or name == String("hl.add.f64"): opch = String("+")
        elif name == String("hl.sub.i64") or name == String("hl.sub.f64"): opch = String("-")
        elif name == String("hl.mul.i64") or name == String("hl.mul.f64"): opch = String("*")
        elif name == String("hl.div.i64") or name == String("hl.div.f64"): opch = String("/")

        if opch.size() > 0:
            var ty2 = elem_hip_type(op.results[0].value().typ)
            var lhs2 = self._vname(op.results[0].value())
            var a = self._expr_for_arg(op.operands[0].value(), entry, args)
            var b = self._expr_for_arg(op.operands[1].value(), entry, args)
            return String("  ") + ty2 + String(" ") + lhs2 + String(" = ") + a + String(" ") + opch + String(" ") + b + String(";\n")

        # compare -> bool
        if name == String("hl.cmp.eq") or name == String("hl.cmp.lt") or name == String("hl.cmp.gt"):
            var lhs3 = self._vname(op.results[0].value())
            var a2 = self._expr_for_arg(op.operands[0].value(), entry, args)
            var b2 = self._expr_for_arg(op.operands[1].value(), entry, args)
            var cmp = String("==")
            if name == String("hl.cmp.lt"): cmp = String("<")
            elif name == String("hl.cmp.gt"): cmp = String(">")
            return String("  bool ") + lhs3 + String(" = ") + a2 + String(" ") + cmp + String(" ") + b2 + String(";\n")

        # select
        if name == String("hl.select"):
            var ty3 = elem_hip_type(op.results[0].value().typ)
            var lhs4 = self._vname(op.results[0].value())
            var c = self._expr_for_arg(op.operands[0].value(), entry, args)
            var tv = self._expr_for_arg(op.operands[1].value(), entry, args)
            var fv = self._expr_for_arg(op.operands[2].value(), entry, args)
            return String("  ") + ty3 + String(" ") + lhs4 + String(" = ") + c + String(" ? ") + tv + String(" : ") + fv + String(";\n")

        # return handled in epilogue
        return String("")
fn emit_function(self, f: Function) -> String:
        # Only support single return tensor for kernel path
        if f.ret_types.size() != 1:
            return String("/* Unsupported: functions with ") + String(f.ret_types.size()) + String(" returns */\n")
        var ret_t = f.ret_types[0]
        if not is_tensor(ret_t):
            return String("/* Unsupported: return type is not a tensor; HIP backend expects tensor out */\n")

        if f.blocks.size() == 0:
            return String("/* Error: function has no blocks */\n")
        var entry = f.blocks[0]

        var needs_N = False
        var ksig = self._kernel_sig(f.name, ret_t, f.arg_types, needs_N)
        var lsig = self._launcher_sig(f.name, ret_t, f.arg_types, needs_N)

        var ok = False; var dt = DType.Invalid; var sh = Shape()
        tensor_unpack(ret_t, ok, dt, sh)
        var Nexpr = String("N")
        if ok and sh.rank() > 0 and sh.dims.size() > 0 and sh.dims[0] >= 0:
            Nexpr = String(String(sh.dims[0]))

        var s = String("")
        s = s + self._emit_headers()

        # Kernel
        s = s + ksig + String(" {\n")
        s = s + self._idx_line() + String("\n")
        s = s + self._guard_open() + String("\n")

        var ret_expr = String("")
        for oi in range(entry.ops.size()):
            var op = entry.ops[oi]
            if op.name == String("hl.return") or op.name == String("return"):
                if op.operands.size() > 0:
                    ret_expr = self._expr_for_arg(op.operands[0].value(), entry, f.arg_types)
                continue
            s = s + self._emit_op_kernel(op, entry, f.arg_types)

        if ret_expr.size() == 0:
            ret_expr = String("0")
        s = s + String("    out[idx] = ") + ret_expr + String(";\n")
        s = s + self._guard_close() + String("\n")
        s = s + String("}\n\n")

        # Launcher
        s = s + lsig + String(" {\n")
        s = s + String("  int block = ") + String(self.opt.block_dim_x) + String(";\n")
        s = s + String("  int N = ") + Nexpr + String(";\n")
        s = s + String("  int grid = (N + block - 1) / block;\n")
        s = s + String("  hipLaunchKernelGGL(") + f.name + self.opt.kernel_suffix + String(", dim3(grid), dim3(block), 0, 0, ")
        s = s + String("out")
        for i in range(f.arg_types.size()):
            s = s + String(", ") + self._argname(i)
        s = s + String(", N);\n")
        s = s + String("}\n\n")

        return s
fn emit_module(self, m: Module) -> String:
        var s = String("")
        s = s + self._emit_headers()
        for i in range(m.functions.size()):
            s = s + self.emit_function(m.functions[i])
        return s
fn __copyinit__(out self, other: Self) -> None:
        self.opt = other.opt
fn __moveinit__(out self, deinit other: Self) -> None:
        self.opt = other.opt
# -----------------------------
# Tiny demo IR module for self-test
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
fn _tensor_i64_1d(n: Int) -> TypeDesc:
    var sh = String("[") + String(n) + String("]")
    return TypeDesc(String("tensor<i64,") + sh + String(">"))
fn _demo_module() -> Module:
    var m = Module(String("demo"))
    var f = Function(String("vec_arith"))
    # args: two tensors i64[1024], one scalar i64
    f.arg_types.push_back(_tensor_i64_1d(1024))
    f.arg_types.push_back(_tensor_i64_1d(1024))
    f.arg_types.push_back(TypeDesc(String("i64")))
    f.ret_types.push_back(_tensor_i64_1d(1024))

    var idg = IdGen()
    var entry = Block(String("entry"))
    var a0 = Value(idg.fresh(), f.arg_types[0])
    var a1 = Value(idg.fresh(), f.arg_types[1])
    var s0 = Value(idg.fresh(), f.arg_types[2])
    entry.args.push_back(a0)
    entry.args.push_back(a1)
    entry.args.push_back(s0)

    var add = Op(String("hl.add.i64"), Location(String("demo.mojo"), 1, 1))
    add.operands.push_back(Operand(a0))
    add.operands.push_back(Operand(a1))
    var r0 = Value(idg.fresh(), TypeDesc(String("i64")))
    add.results.push_back(Result(r0))

    var cst = Op(String("hl.const.i64"), Location(String("demo.mojo"), 1, 2))
    var r1 = Value(idg.fresh(), TypeDesc(String("i64")))
    cst.results.push_back(Result(r1))
    cst.attrs.push_back(String("value=5"))

    var mul = Op(String("hl.mul.i64"), Location(String("demo.mojo"), 1, 3))
    mul.operands.push_back(Operand(r0))
    mul.operands.push_back(Operand(s0))  # scalar arg
    var r2 = Value(idg.fresh(), TypeDesc(String("i64")))
    mul.results.push_back(Result(r2))

    var gt = Op(String("hl.cmp.gt"), Location(String("demo.mojo"), 1, 4))
    gt.operands.push_back(Operand(r2))
    gt.operands.push_back(Operand(r1))  # compare with const 5
    var c0 = Value(idg.fresh(), TypeDesc(String("bool")))
    gt.results.push_back(Result(c0))

    var sel = Op(String("hl.select"), Location(String("demo.mojo"), 1, 5))
    sel.operands.push_back(Operand(c0))
    sel.operands.push_back(Operand(r2))
    sel.operands.push_back(Operand(r0))
    var r3 = Value(idg.fresh(), TypeDesc(String("i64")))
    sel.results.push_back(Result(r3))

    var ret = Op(String("hl.return"), Location(String("demo.mojo"), 1, 6))
    ret.operands.push_back(Operand(r3))

    entry.ops.push_back(add)
    entry.ops.push_back(cst)
    entry.ops.push_back(mul)
    entry.ops.push_back(gt)
    entry.ops.push_back(sel)
    entry.ops.push_back(ret)

    f.blocks.push_back(entry)
    m.functions.push_back(f)
    return m

# -----------------------------
# Self-test
# -----------------------------
fn _self_test_codegen_rocm() -> Bool:
    var m = _demo_module()
    var gen = CodegenROCm(CodegenOptions(True, 256, String("_kernel"), String("run_")))
    var txt = gen.emit_module(m)

    var ok = True
    if txt.find(String("#include <hip/hip_runtime.h>")) < 0: ok = False
    if txt.find(String("__global__ void vec_arith_kernel")) < 0: ok = False
    if txt.find(String("hipLaunchKernelGGL")) < 0: ok = False
    if txt.find(String("int grid = (N + block - 1) / block;")) < 0: ok = False
    if txt.find(String("out[idx] =")) < 0: ok = False

    if ok: print(String("OK"))
    else: print(String("FAIL"))
    return ok