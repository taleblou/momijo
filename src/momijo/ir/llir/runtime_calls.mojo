# Project:      Momijo
# Module:       src.momijo.ir.llir.runtime_calls
# File:         runtime_calls.mojo
# Path:         src/momijo/ir/llir/runtime_calls.mojo
#
# Description:  src.momijo.ir.llir.runtime_calls — focused Momijo functionality with a stable public API.
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
# Runtime op names (rt.*) and attributes
# -----------------------------
# Signatures are conceptual; codegen backends decide the actual mapping.
#
#   - rt.printf.i64    (%x:i64)             : ()           attrs: {fmt?}
#   - rt.printf.f64    (%x:f64)             : ()
#   - rt.printf.bool   (%x:bool)            : ()
#   - rt.println.str   ()                   : ()           attrs: {text="<...>"}
#   - rt.assert        (%cond:bool)         : ()           attrs: {msg="<...>"}
#   - rt.time.now_ms   ()                   : (i64)
#   - rt.memset.i64    (%dst:tensor<i64,[N]>, %val:i64) : ()
#   - rt.memcpy.i64    (%dst:tensor<i64,[N]>, %src:tensor<i64,[N]>) : ()
#   - rt.rand.fill.i64 (%dst:tensor<i64,[N]>, attrs: {seed?=...})   : ()

# -----------------------------
# Feature collector
# -----------------------------

struct RuntimeUse:
    var print_i64: Bool
    var print_f64: Bool
    var print_bool: Bool
    var print_str: Bool
    var assert_: Bool
    var time_now_ms: Bool
    var memset_i64: Bool
    var memcpy_i64: Bool
    var randfill_i64: Bool
fn __init__(out self) -> None:
        self.print_i64 = False
        self.print_f64 = False
        self.print_bool = False
        self.print_str = False
        self.assert_ = False
        self.time_now_ms = False
        self.memset_i64 = False
        self.memcpy_i64 = False
        self.randfill_i64 = False
fn __copyinit__(out self, other: Self) -> None:
        self.print_i64 = other.print_i64
        self.print_f64 = other.print_f64
        self.print_bool = other.print_bool
        self.print_str = other.print_str
        self.assert_ = other.assert_
        self.time_now_ms = other.time_now_ms
        self.memset_i64 = other.memset_i64
        self.memcpy_i64 = other.memcpy_i64
        self.randfill_i64 = other.randfill_i64
fn __moveinit__(out self, deinit other: Self) -> None:
        self.print_i64 = other.print_i64
        self.print_f64 = other.print_f64
        self.print_bool = other.print_bool
        self.print_str = other.print_str
        self.assert_ = other.assert_
        self.time_now_ms = other.time_now_ms
        self.memset_i64 = other.memset_i64
        self.memcpy_i64 = other.memcpy_i64
        self.randfill_i64 = other.randfill_i64
struct RuntimeCollector:
fn collect(self, m: Module) -> RuntimeUse:
        var use = RuntimeUse()
        for fi in range(m.functions.size()):
            var f = m.functions[fi]
            for bi in range(f.blocks.size()):
                var b = f.blocks[bi]
                for oi in range(b.ops.size()):
                    var op = b.ops[oi]
                    if op.name == String("rt.printf.i64"): use.print_i64 = True
                    elif op.name == String("rt.printf.f64"): use.print_f64 = True
                    elif op.name == String("rt.printf.bool"): use.print_bool = True
                    elif op.name == String("rt.println.str"): use.print_str = True
                    elif op.name == String("rt.assert"): use.assert_ = True
                    elif op.name == String("rt.time.now_ms"): use.time_now_ms = True
                    elif op.name == String("rt.memset.i64"): use.memset_i64 = True
                    elif op.name == String("rt.memcpy.i64"): use.memcpy_i64 = True
                    elif op.name == String("rt.rand.fill.i64"): use.randfill_i64 = True
        return use
fn __init__(out self, ) -> None:
        pass
fn __copyinit__(out self, other: Self) -> None:
        pass
fn __moveinit__(out self, deinit other: Self) -> None:
        pass
# -----------------------------
# Tiny attribute helper
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
# CPU (C) stub emitter
# -----------------------------

struct CpuRuntimeEmitter:
fn emit_header(self, use: RuntimeUse) -> String:
        var s = String("/* Momijo CPU Runtime (header+impl inline) */\n")
        s = s + String("#include <stdint.h>\n#include <stdbool.h>\n#include <stdio.h>\n#include <stdlib.h>\n#include <time.h>\n\n")
        if use.print_i64:
            s = s + String("static inline void mjrt_print_i64(long long x){ printf(\"%lld\", x); }\n")
        if use.print_f64:
            s = s + String("static inline void mjrt_print_f64(double x){ printf(\"%g\", x); }\n")
        if use.print_bool:
            s = s + String("static inline void mjrt_print_bool(bool x){ fputs(x?\"true\":\"false\", stdout); }\n")
        if use.print_str:
            s = s + String("static inline void mjrt_println_str(const char* s){ puts(s?s:\"\"); }\n")
        if use.assert_:
            s = s + String("static inline void mjrt_assert(bool cond, const char* msg){ if(!cond){ fprintf(stderr, \"ASSERT: %s\\n\", msg?msg:\"\"); abort(); } }\n")
        if use.time_now_ms:
            s = s + String("static inline long long mjrt_time_now_ms(){ struct timespec ts; clock_gettime(CLOCK_REALTIME, &ts); return (long long)ts.tv_sec*1000LL + (long long)(ts.tv_nsec/1000000LL); }\n")
        if use.memset_i64:
            s = s + String("static inline void mjrt_memset_i64(long long* dst, long long value, int N){ for(int i=0;i<N;++i) dst[i]=value; }\n")
        if use.memcpy_i64:
            s = s + String("static inline void mjrt_memcpy_i64(long long* dst, const long long* src, int N){ for(int i=0;i<N;++i) dst[i]=src[i]; }\n")
        if use.randfill_i64:
(            s = s + String("static inline void mjrt_randfill_i64(long long* dst, int N, unsigned long long seed){ unsigned long long x=seed?seed:88172645463393265ULL; for(int i=0;i<N;++i){ x ^= x << UInt8(7); x ^= x >> UInt8(9); dst[i]=(long long)(x & 0x7FFFFFFFFFFFFFFFULL); } }\n")) & UInt8(0xFF)
        s = s + String("\n")
        return s
fn __init__(out self, ) -> None:
        pass
fn __copyinit__(out self, other: Self) -> None:
        pass
fn __moveinit__(out self, deinit other: Self) -> None:
        pass
# -----------------------------
# CUDA/HIP/MSL helper emitters (comment stubs)
# -----------------------------

struct CudaRuntimeEmitter:
fn emit_notes(self, use: RuntimeUse) -> String:
        var s = String("/* Momijo CUDA Runtime (notes)\n")
        s = s + String(" - Printing on device is not included; prefer host-side printf.\n")
        if use.memset_i64: s = s + String(" - Provide device memset kernels or use cudaMemsetAsync for bytes.\n")
        if use.memcpy_i64: s = s + String(" - Host-side cudaMemcpy for transfers.\n")
        s = s + String("*/\n\n")
        return s
fn __init__(out self, ) -> None:
        pass
fn __copyinit__(out self, other: Self) -> None:
        pass
fn __moveinit__(out self, deinit other: Self) -> None:
        pass
struct HipRuntimeEmitter:
fn emit_notes(self, use: RuntimeUse) -> String:
        var s = String("/* Momijo HIP Runtime (notes)\n")
        s = s + String(" - Printing on device is not included; prefer host-side I/O.\n")
        if use.memset_i64: s = s + String(" - Provide device memset kernels or hipMemset for bytes.\n")
        if use.memcpy_i64: s = s + String(" - Host-side hipMemcpy for transfers.\n")
        s = s + String("*/\n\n")
        return s
fn __init__(out self, ) -> None:
        pass
fn __copyinit__(out self, other: Self) -> None:
        pass
fn __moveinit__(out self, deinit other: Self) -> None:
        pass
struct MpsRuntimeEmitter:
fn emit_notes(self, use: RuntimeUse) -> String:
        var s = String("// Momijo Metal/MPS Runtime (notes)\n")
        s = s + String("// - Printing is host-side; pass data back or use debug capture.\n\n")
        return s
fn __init__(out self, ) -> None:
        pass
fn __copyinit__(out self, other: Self) -> None:
        pass
fn __moveinit__(out self, deinit other: Self) -> None:
        pass
# -----------------------------
# IR builders for rt.* ops
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
struct RtBuilder:
    var idg: IdGen
fn __init__(out self) -> None:
        self.idg = IdGen()
fn printf_i64(self, b: Block, x: Value, loc: Location = Location()) -> Op:
        var op = Op(String("rt.printf.i64"), loc)
        op.operands.push_back(Operand(x))
        b.ops.push_back(op)
        return op
fn printf_f64(self, b: Block, x: Value, loc: Location = Location()) -> Op:
        var op = Op(String("rt.printf.f64"), loc)
        op.operands.push_back(Operand(x))
        b.ops.push_back(op)
        return op
fn printf_bool(self, b: Block, x: Value, loc: Location = Location()) -> Op:
        var op = Op(String("rt.printf.bool"), loc)
        op.operands.push_back(Operand(x))
        b.ops.push_back(op)
        return op
fn println_str(self, b: Block, text: String, loc: Location = Location()) -> Op:
        var op = Op(String("rt.println.str"), loc)
        op.attrs.push_back(String("text=") + text)
        b.ops.push_back(op)
        return op
fn assert(self, b: Block, cond: Value, msg: String = String(""), loc: Location = Location()) -> Op:
        var op = Op(String("rt.assert"), loc)
        op.operands.push_back(Operand(cond))
        if msg.size() > 0:
            op.attrs.push_back(String("msg=") + msg)
        b.ops.push_back(op)
        return op
fn time_now_ms(self, b: Block, loc: Location = Location()) -> Value:
        var v = Value(self.idg.fresh(), TypeDesc(String("i64")))
        var op = Op(String("rt.time.now_ms"), loc)
        op.results.push_back(Result(v))
        b.ops.push_back(op)
        return v
fn __copyinit__(out self, other: Self) -> None:
        self.idg = other.idg
fn __moveinit__(out self, deinit other: Self) -> None:
        self.idg = other.idg
# -----------------------------
# Pretty printer (tiny): shows rt.* ops
# -----------------------------

struct Printer:
fn _value(self, v: Value) -> String:
        return String("%") + String(v.id) + String(":") + v.typ.name
fn print(self, m: Module) -> String:
        var s = String("module @") + m.name + String(" {\n")
        for i in range(m.functions.size()):
            var f = m.functions[i]
            s = s + String("  func @") + f.name + String("() -> () {\n")
            for bi in range(f.blocks.size()):
                var b = f.blocks[bi]
                s = s + String("    ^") + b.name + String("():\n")
                for oi in range(b.ops.size()):
                    var op = b.ops[oi]
                    s = s + String("      ") + op.name
                    if op.operands.size() > 0:
                        s = s + String("(")
                        for pi in range(op.operands.size()):
                            s = s + self._value(op.operands[pi].value())
                            if pi + 1 < op.operands.size():
                                s = s + String(", ")
                        s = s + String(")")
                    if op.results.size() > 0:
                        s = s + String(" -> ")
                        for ri in range(op.results.size()):
                            s = s + self._value(op.results[ri].value())
                            if ri + 1 < op.results.size():
                                s = s + String(", ")
                    if op.attrs.size() > 0:
                        s = s + String(" {")
                        for ai in range(op.attrs.size()):
                            s = s + op.attrs[ai]
                            if ai + 1 < op.attrs.size():
                                s = s + String(", ")
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
# Self-test: build a module with runtime ops, collect features, emit stubs.
# -----------------------------
fn _self_test_runtime_calls() -> Bool:
    var m = Module(String("demo_rt"))
    var f = Function(String("main"))
    var b = Block(String("entry"))

    var rtb = RtBuilder()
    var idg = IdGen()

    # values to feed into rt ops
    var v_i = Value(idg.fresh(), TypeDesc(String("i64")))
    var v_f = Value(idg.fresh(), TypeDesc(String("f64")))
    var v_c = Value(idg.fresh(), TypeDesc(String("bool")))
    b.args.push_back(v_i)
    b.args.push_back(v_f)
    b.args.push_back(v_c)

    rtb.printf_i64(b, v_i)
    rtb.printf_f64(b, v_f)
    rtb.printf_bool(b, v_c)
    rtb.println_str(b, String("\"hello\""))
    rtb.assert(b, v_c, String("\"cond failed\""))
    var t = rtb.time_now_ms(b)
    # use 't' somehow (print it)
    rtb.printf_i64(b, t)

    f.blocks.push_back(b)
    m.functions.push_back(f)

    var pr = Printer()
    var txt_ir = pr.print(m)

    var col = RuntimeCollector()
    var use = col.collect(m)

    var cpu = CpuRuntimeEmitter()
    var hdr = cpu.emit_header(use)

    var ok = True
    if txt_ir.find(String("rt.printf.i64")) < 0: ok = False
    if txt_ir.find(String("rt.assert")) < 0: ok = False
    if txt_ir.find(String("rt.time.now_ms")) < 0: ok = False
    if hdr.find(String("mjrt_print_i64")) < 0: ok = False
    if hdr.find(String("mjrt_assert")) < 0: ok = False
    if hdr.find(String("mjrt_time_now_ms")) < 0: ok = False

    if ok: print(String("OK"))
    else: print(String("FAIL"))
    return ok