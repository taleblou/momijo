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
# Project: momijo.ir.dialects
# File: src/momijo/ir/dialects/hlir_ops.mojo

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
fn to_string(self) -> String: return self.name
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
struct BuilderCore:
    var module: Module
    var current_func: Function
    var current_block: Block
    var idgen: IdGen
fn __init__(out self, module_name: String = String("main")):
        self.module = Module(module_name)
        self.current_func = Function(String(""))
        self.current_block = Block(String("entry"))
        self.idgen = IdGen()
fn create_function(self, name: String, args: List[TypeDesc], rets: List[TypeDesc]) -> None:
        var f = Function(name)
        for i in range(args.size()): f.arg_types.push_back(args[i])
        for i in range(rets.size()): f.ret_types.push_back(rets[i])
        self.current_func = f
        self.current_block = Block(String("entry"))
        for i in range(args.size()):
            var v = Value(self.idgen.fresh(), args[i])
            self.current_block.args.push_back(v)
fn end_function(self) -> None:
        self.current_func.blocks.push_back(self.current_block)
        self.module.functions.push_back(self.current_func)
fn new_block(self, name: String) -> None:
        self.current_func.blocks.push_back(self.current_block)
        self.current_block = Block(name)
fn __copyinit__(out self, other: Self) -> None:
        self.module = other.module
        self.current_func = other.current_func
        self.current_block = other.current_block
        self.idgen = other.idgen
fn __moveinit__(out self, deinit other: Self) -> None:
        self.module = other.module
        self.current_func = other.current_func
        self.current_block = other.current_block
        self.idgen = other.idgen
# -----------------------------
# HLIR: Types & utilities
# -----------------------------
fn t_i64() -> TypeDesc: return TypeDesc(String("i64"))
fn t_f64() -> TypeDesc: return TypeDesc(String("f64"))
fn t_bool() -> TypeDesc: return TypeDesc(String("bool"))
fn is_i64(t: TypeDesc) -> Bool: return t.name == String("i64")
fn is_f64(t: TypeDesc) -> Bool: return t.name == String("f64")
fn is_bool(t: TypeDesc) -> Bool: return t.name == String("bool")
fn is_same_numeric(a: TypeDesc, b: TypeDesc) -> Bool:
    if is_i64(a) and is_i64(b): return True
    if is_f64(a) and is_f64(b): return True
    return False

# -----------------------------
# HLIR emit helpers (typed)
# -----------------------------

struct EmitResult:
    var ok: Bool
    var value: Value
    var message: String
fn __init__(out self, ok: Bool, value: Value, message: String = String("")):
        self.ok = ok
        assert(self is not None, String("self is None"))
        self.value() = value
        self.message = message
fn __copyinit__(out self, other: Self) -> None:
        self.ok = other.ok
        assert(self is not None, String("self is None"))
        self.value() = other.value()
        self.message = other.message
fn __moveinit__(out self, deinit other: Self) -> None:
        self.ok = other.ok
        assert(self is not None, String("self is None"))
        self.value() = other.value()
        self.message = other.message
struct BuilderHL:
    var b: BuilderCore
fn __init__(out self, module_name: String = String("main")):
        self.b = BuilderCore(module_name)
fn _res(self, id: Int, t: TypeDesc) -> Value:
        return Value(id, t)

    # --- constants ---
fn const_i64(self, v: Int, loc: Location = Location()) -> EmitResult:
        var rid = self.b.idgen.fresh()
        var res = self._res(rid, t_i64())
        var op = Op(String("hl.const.i64"), loc)
        op.results.push_back(Result(res))
        op.attrs.push_back(String("value=") + String(v))
        self.b.current_block.ops.push_back(op)
        return EmitResult(True, res)
fn const_f64(self, v: Float64, loc: Location = Location()) -> EmitResult:
        var rid = self.b.idgen.fresh()
        var res = self._res(rid, t_f64())
        var op = Op(String("hl.const.f64"), loc)
        op.results.push_back(Result(res))
        op.attrs.push_back(String("value=") + String(v))
        self.b.current_block.ops.push_back(op)
        return EmitResult(True, res)
fn const_bool(self, v: Bool, loc: Location = Location()) -> EmitResult:
        var rid = self.b.idgen.fresh()
        var res = self._res(rid, t_bool())
        var op = Op(String("hl.const.bool"), loc)
        op.results.push_back(Result(res))
        op.attrs.push_back(String("value=") + (String("true") if v else String("false")))
        self.b.current_block.ops.push_back(op)
        return EmitResult(True, res)

    # --- binary numeric (same-type) ---
fn _emit_bin_num(self, opname: String, lhs: Value, rhs: Value, loc: Location) -> EmitResult:
        if not is_same_numeric(lhs.typ, rhs.typ):
            return EmitResult(False, Value(-1, TypeDesc(String("invalid"))), String("Type mismatch for ") + opname)
        var rid = self.b.idgen.fresh()
        var res = self._res(rid, lhs.typ)
        var op = Op(opname, loc)
        op.results.push_back(Result(res))
        op.operands.push_back(Operand(lhs))
        op.operands.push_back(Operand(rhs))
        self.b.current_block.ops.push_back(op)
        return EmitResult(True, res)
fn add(self, lhs: Value, rhs: Value, loc: Location = Location()) -> EmitResult:
        # op name depends on type
        var name = String("hl.add.")
        if is_i64(lhs.typ): name = name + String("i64")
        elif is_f64(lhs.typ): name = name + String("f64")
        else: return EmitResult(False, Value(-1, TypeDesc(String("invalid"))), String("Unsupported type for add"))
        return self._emit_bin_num(name, lhs, rhs, loc)
fn sub(self, lhs: Value, rhs: Value, loc: Location = Location()) -> EmitResult:
        var name = String("hl.sub.")
        if is_i64(lhs.typ): name = name + String("i64")
        elif is_f64(lhs.typ): name = name + String("f64")
        else: return EmitResult(False, Value(-1, TypeDesc(String("invalid"))), String("Unsupported type for sub"))
        return self._emit_bin_num(name, lhs, rhs, loc)
fn mul(self, lhs: Value, rhs: Value, loc: Location = Location()) -> EmitResult:
        var name = String("hl.mul.")
        if is_i64(lhs.typ): name = name + String("i64")
        elif is_f64(lhs.typ): name = name + String("f64")
        else: return EmitResult(False, Value(-1, TypeDesc(String("invalid"))), String("Unsupported type for mul"))
        return self._emit_bin_num(name, lhs, rhs, loc)
fn div(self, lhs: Value, rhs: Value, loc: Location = Location()) -> EmitResult:
        var name = String("hl.div.")
        if is_i64(lhs.typ): name = name + String("i64")
        elif is_f64(lhs.typ): name = name + String("f64")
        else: return EmitResult(False, Value(-1, TypeDesc(String("invalid"))), String("Unsupported type for div"))
        return self._emit_bin_num(name, lhs, rhs, loc)

    # --- comparisons -> bool ---
fn _emit_cmp(self, opname: String, lhs: Value, rhs: Value, loc: Location) -> EmitResult:
        if not is_same_numeric(lhs.typ, rhs.typ):
            return EmitResult(False, Value(-1, TypeDesc(String("invalid"))), String("Type mismatch for ") + opname)
        var rid = self.b.idgen.fresh()
        var res = self._res(rid, t_bool())
        var op = Op(opname, loc)
        op.results.push_back(Result(res))
        op.operands.push_back(Operand(lhs))
        op.operands.push_back(Operand(rhs))
        self.b.current_block.ops.push_back(op)
        return EmitResult(True, res)
fn cmp_eq(self, lhs: Value, rhs: Value, loc: Location = Location()) -> EmitResult:
        return self._emit_cmp(String("hl.cmp.eq"), lhs, rhs, loc)
fn cmp_lt(self, lhs: Value, rhs: Value, loc: Location = Location()) -> EmitResult:
        return self._emit_cmp(String("hl.cmp.lt"), lhs, rhs, loc)
fn cmp_gt(self, lhs: Value, rhs: Value, loc: Location = Location()) -> EmitResult:
        return self._emit_cmp(String("hl.cmp.gt"), lhs, rhs, loc)

    # --- select(cond, t, f) ---
fn select(self, cond: Value, tval: Value, fval: Value, loc: Location = Location()) -> EmitResult:
        if not is_bool(cond.typ):
            return EmitResult(False, Value(-1, TypeDesc(String("invalid"))), String("Condition for select must be bool"))
        if not is_same_numeric(tval.typ, fval.typ):
            return EmitResult(False, Value(-1, TypeDesc(String("invalid"))), String("Arms of select must have same numeric type"))
        var rid = self.b.idgen.fresh()
        var res = self._res(rid, tval.typ)
        var op = Op(String("hl.select"), loc)
        op.results.push_back(Result(res))
        op.operands.push_back(Operand(cond))
        op.operands.push_back(Operand(tval))
        op.operands.push_back(Operand(fval))
        self.b.current_block.ops.push_back(op)
        return EmitResult(True, res)

    # --- return ---
fn ret(self, values: List[Value], loc: Location = Location()) -> None:
        var op = Op(String("hl.return"), loc)
        for i in range(values.size()):
            op.operands.push_back(Operand(values[i]))
        self.b.current_block.ops.push_back(op)
fn __copyinit__(out self, other: Self) -> None:
        self.b = other.b
fn __moveinit__(out self, deinit other: Self) -> None:
        self.b = other.b
# -----------------------------
# Pretty-printer (tiny)
# -----------------------------

struct Printer:
fn _value(self, v: Value) -> String:
        return String("%") + String(v.id) + String(":") + v.typ.name
fn _values_csv(self, vals: List[Value]) -> String:
        var s = String("")
        for i in range(vals.size()):
            s = s + self._value(vals[i])
            if i + 1 < vals.size(): s = s + String(", ")
        return s
fn print(self, m: Module) -> String:
        var s = String("module @") + m.name + String(" {\n")
        for i in range(m.functions.size()):
            var f = m.functions[i]
            s = s + String("  func @") + f.name + String(" (")
            for a in range(f.arg_types.size()):
                s = s + f.arg_types[a].name
                if a + 1 < f.arg_types.size(): s = s + String(", ")
            s = s + String(") -> (")
            for r in range(f.ret_types.size()):
                s = s + f.ret_types[r].name
                if r + 1 < f.ret_types.size(): s = s + String(", ")
            s = s + String(") {\n")

            for bi in range(f.blocks.size()):
                var b = f.blocks[bi]
                s = s + String("    ^") + b.name + String("(") + self._values_csv(b.args) + String("):\n")
                for oi in range(b.ops.size()):
                    var op = b.ops[oi]
                    s = s + String("      ")
                    if op.results.size() > 0:
                        for ri in range(op.results.size()):
                            s = s + self._value(op.results[ri].value())
                            if ri + 1 < op.results.size(): s = s + String(", ")
                            else: s = s + String(" = ")
                    s = s + op.name + String("(")
                    for pi in range(op.operands.size()):
                        s = s + self._value(op.operands[pi].value())
                        if pi + 1 < op.operands.size(): s = s + String(", ")
                    s = s + String(")")
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
# Self-test (build a small HLIR function)
# -----------------------------
fn _self_test_hlir_ops() -> Bool:
    var hl = BuilderHL(String("demo"))
    var args = List[TypeDesc]()
    args.push_back(t_i64())
    args.push_back(t_i64())
    var rets = List[TypeDesc]()
    rets.push_back(t_i64())
    hl.b.create_function(String("arith"), args, rets)

    var a = hl.b.current_block.args[0]
    var b = hl.b.current_block.args[1]

    var s1 = hl.add(a, b, Location(String("arith.mojo"), 1, 1))
    if not s1.ok: return False
    var c2 = hl.const_i64(10, Location(String("arith.mojo"), 1, 2))
    if not c2.ok: return False
    assert(s1 is not None, String("s1 is None"))
    var s2 = hl.mul(s1.value(), c2.value(), Location(String("arith.mojo"), 1, 3))
    if not s2.ok: return False

    assert(s2 is not None, String("s2 is None"))
    var cmp = hl.cmp_gt(s2.value(), a, Location(String("arith.mojo"), 1, 4))
    if not cmp.ok: return False
    assert(cmp is not None, String("cmp is None"))
    var sel = hl.select(cmp.value(), s2.value(), a, Location(String("arith.mojo"), 1, 5))
    if not sel.ok: return False

    var outs = List[Value]()
    assert(sel is not None, String("sel is None"))
    outs.push_back(sel.value())
    hl.ret(outs, Location(String("arith.mojo"), 1, 6))
    hl.b.end_function()

    var p = Printer()
    var txt = p.print(hl.b.module)

    var ok = True
    if txt.find(String("module @demo")) < 0: ok = False
    if txt.find(String("func @arith")) < 0: ok = False
    if txt.find(String("hl.add.i64")) < 0: ok = False
    if txt.find(String("hl.mul.i64")) < 0: ok = False
    if txt.find(String("hl.cmp.gt")) < 0: ok = False
    if txt.find(String("hl.select")) < 0: ok = False
    if txt.find(String("^entry(%0:i64, %1:i64)")) < 0: ok = False

    if ok: print(String("OK"))
    else: print(String("FAIL"))
    return ok