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
# Project: momijo.ir
# File: src/momijo/ir/builder.mojo

struct Location:
    var filename: String
    var line: Int
    var col: Int
fn __init__(out self, filename: String = String("<unknown>"), line: Int = 0, col: Int = 0):
        self.filename = filename
        self.line = line
        self.col = col
fn to_string(self) -> String:
        var s = String(self.filename) + String(":") + String(self.line) + String(":") + String(self.col)
        return s
fn __copyinit__(out self, other: Self) -> None:
        self.filename = other.filename
        self.line = other.line
        self.col = other.col
fn __moveinit__(out self, deinit other: Self) -> None:
        self.filename = other.filename
        self.line = other.line
        self.col = other.col
struct TypeDesc:
    # Textual type descriptor (e.g., "i64", "f64", "bool")
    var name: String
fn __init__(out self, name: String) -> None:
        self.name = name
fn to_string(self) -> String:
        return self.name
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
fn to_string(self) -> String:
        var s = String("%") + String(self.id) + String(":") + self.typ.to_string()
        return s
fn __copyinit__(out self, other: Self) -> None:
        self.id = other.id
        self.typ = other.typ
fn __moveinit__(out self, deinit other: Self) -> None:
        self.id = other.id
        self.typ = other.typ
struct Operand:
    var value: Value
fn __init__(out self, value: Value) -> None:
        assert(self is not None, String("self is None"))
        self.value() = value
fn to_string(self) -> String:
        return self.value().to_string()
fn __copyinit__(out self, other: Self) -> None:
        assert(self is not None, String("self is None"))
        self.value() = other.value()
fn __moveinit__(out self, deinit other: Self) -> None:
        self.value() = other.value()
struct Result:
    var value: Value
fn __init__(out self, value: Value) -> None:
        assert(self is not None, String("self is None"))
        self.value() = value
fn to_string(self) -> String:
        return self.value().to_string()
fn __copyinit__(out self, other: Self) -> None:
        assert(self is not None, String("self is None"))
        self.value() = other.value()
fn __moveinit__(out self, deinit other: Self) -> None:
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
fn add_result(self, r: Result) -> None:
        self.results.push_back(r)
fn add_operand(self, o: Operand) -> None:
        self.operands.push_back(o)
fn add_attr(self, a: String) -> None:
        self.attrs.push_back(a)
fn to_string(self) -> String:
        var s = String("  ")
        if self.results.size() > 0:
            for i in range(self.results.size()):
                s = s + self.results[i].to_string()
                if i + 1 < self.results.size():
                    s = s + String(", ")
                else:
                    s = s + String(" = ")
        s = s + self.name + String("(")
        for i in range(self.operands.size()):
            s = s + self.operands[i].to_string()
            if i + 1 < self.operands.size():
                s = s + String(", ")
        s = s + String(")")
        if self.attrs.size() > 0:
            s = s + String(" {")
            for i in range(self.attrs.size()):
                s = s + self.attrs[i]
                if i + 1 < self.attrs.size():
                    s = s + String(", ")
            s = s + String("}")
        s = s + String("  // ") + self.loc.to_string()
        return s
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
fn add_arg(self, v: Value) -> None:
        self.args.push_back(v)
fn append_op(self, op: Op) -> None:
        self.ops.push_back(op)
fn to_string(self) -> String:
        var s = String("^") + self.name + String("(")
        for i in range(self.args.size()):
            s = s + self.args[i].to_string()
            if i + 1 < self.args.size():
                s = s + String(", ")
        s = s + String("):\n")
        for i in range(self.ops.size()):
            s = s + self.ops[i].to_string() + String("\n")
        return s
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
fn add_arg_type(self, t: TypeDesc) -> None:
        self.arg_types.push_back(t)
fn add_ret_type(self, t: TypeDesc) -> None:
        self.ret_types.push_back(t)
fn append_block(self, b: Block) -> None:
        self.blocks.push_back(b)
fn signature_string(self) -> String:
        var s = String("(")
        for i in range(self.arg_types.size()):
            s = s + self.arg_types[i].to_string()
            if i + 1 < self.arg_types.size():
                s = s + String(", ")
        s = s + String(") -> (")
        for i in range(self.ret_types.size()):
            s = s + self.ret_types[i].to_string()
            if i + 1 < self.ret_types.size():
                s = s + String(", ")
        s = s + String(")")
        return s
fn to_string(self) -> String:
        var s = String("func @") + self.name + String(" ") + self.signature_string() + String(" {\n")
        for i in range(self.blocks.size()):
            s = s + self.blocks[i].to_string()
        s = s + String("}\n")
        return s
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
fn append_function(self, f: Function) -> None:
        self.functions.push_back(f)
fn to_string(self) -> String:
        var s = String("module @") + self.name + String(" {\n")
        for i in range(self.functions.size()):
            s = s + self.functions[i].to_string() + String("\n")
        s = s + String("}\n")
        return s
fn __copyinit__(out self, other: Self) -> None:
        self.name = other.name
        self.functions = other.functions
fn __moveinit__(out self, deinit other: Self) -> None:
        self.name = other.name
        self.functions = other.functions
# -----------------------------
# IR Builder
# -----------------------------

struct IdGen:
    var next_id: Int
fn __init__(out self) -> None:
        self.next_id = 0
fn fresh(self) -> Int:
        var r = self.next_id
        self.next_id = self.next_id + 1
        return r
fn __copyinit__(out self, other: Self) -> None:
        self.next_id = other.next_id
fn __moveinit__(out self, deinit other: Self) -> None:
        self.next_id = other.next_id
struct Builder:
    var module: Module
    var current_func: Function
    var current_block: Block
    var idgen: IdGen
fn __init__(out self, module_name: String = String("main")):
        self.module = Module(module_name)
        self.current_func = Function(String(""))
        self.current_block = Block(String("entry"))  # default, replaced on first function create
        self.idgen = IdGen()

    # --- Module / Function / Block management ---
fn create_function(self, name: String, args: List[TypeDesc], rets: List[TypeDesc]) -> None:
        var f = Function(name)
        for i in range(args.size()):
            f.add_arg_type(args[i])
        for i in range(rets.size()):
            f.add_ret_type(rets[i])
        self.current_func = f
        self.current_block = Block(String("entry"))
        # Materialize block args for function arguments
        for i in range(args.size()):
            var v = Value(self.idgen.fresh(), args[i])
            self.current_block.add_arg(v)
fn end_function(self) -> None:
        self.current_func.append_block(self.current_block)
        self.module.append_function(self.current_func)
fn new_block(self, name: String) -> None:
        # Push current block and start a new one
        self.current_func.append_block(self.current_block)
        self.current_block = Block(name)

    # --- Typed helpers ---
fn t_i64(self) -> TypeDesc:
        return TypeDesc(String("i64"))
fn t_f64(self) -> TypeDesc:
        return TypeDesc(String("f64"))
fn t_bool(self) -> TypeDesc:
        return TypeDesc(String("bool"))

    # --- Emit operations (toy dialect) ---
fn emit_const_i64(self, val: Int, loc: Location = Location()) -> Value:
        var rid = self.idgen.fresh()
        var res = Value(rid, self.t_i64())
        var op = Op(String("const.i64"), loc)
        op.add_result(Result(res))
        op.add_attr(String("value=") + String(val))
        self.current_block.append_op(op)
        return res
fn emit_add_i64(self, lhs: Value, rhs: Value, loc: Location = Location()) -> Value:
        var rid = self.idgen.fresh()
        var res = Value(rid, self.t_i64())
        var op = Op(String("add.i64"), loc)
        op.add_result(Result(res))
        op.add_operand(Operand(lhs))
        op.add_operand(Operand(rhs))
        self.current_block.append_op(op)
        return res
fn emit_mul_i64(self, lhs: Value, rhs: Value, loc: Location = Location()) -> Value:
        var rid = self.idgen.fresh()
        var res = Value(rid, self.t_i64())
        var op = Op(String("mul.i64"), loc)
        op.add_result(Result(res))
        op.add_operand(Operand(lhs))
        op.add_operand(Operand(rhs))
        self.current_block.append_op(op)
        return res
fn emit_ret(self, values: List[Value], loc: Location = Location()) -> None:
        var op = Op(String("return"), loc)
        for i in range(values.size()):
            op.add_operand(Operand(values[i]))
        self.current_block.append_op(op)

    # --- Finalize ---
fn module_ir(self) -> String:
        return self.module.to_string()
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
# Minimal smoke tests
# -----------------------------
fn _self_test_builder() -> Bool:
    var b = Builder(String("demo"))
    var args = List[TypeDesc]()
    args.push_back(TypeDesc(String("i64")))
    args.push_back(TypeDesc(String("i64")))
    var rets = List[TypeDesc]()
    rets.push_back(TypeDesc(String("i64")))

    b.create_function(String("sum_mul"), args, rets)

    # Block args: ^entry(%0:i64, %1:i64)
    # Emit: c = a + b; d = c * 2; return d
    # We need to fetch the two block args (first two values of block args)
    var a = b.current_block.args[0]
    var c = b.current_block.args[1]

    var tmp = b.emit_add_i64(a, c, Location(String("demo.mojo"), 1, 1))
    var two = b.emit_const_i64(2, Location(String("demo.mojo"), 1, 2))
    var out = b.emit_mul_i64(tmp, two, Location(String("demo.mojo"), 1, 3))
    var outs = List[Value]()
    outs.push_back(out)
    b.emit_ret(outs, Location(String("demo.mojo"), 1, 4))
    b.end_function()

    var ir = b.module_ir()
    # sanity: the IR must contain basic markers
    var ok = True
    if ir.find(String("module @demo")) < 0:
        ok = False
    if ir.find(String("func @sum_mul")) < 0:
        ok = False
    if ir.find(String("add.i64")) < 0:
        ok = False
    if ir.find(String("return")) < 0:
        ok = False
    return ok