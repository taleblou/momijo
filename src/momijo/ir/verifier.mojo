# Project:      Momijo
# Module:       src.momijo.ir.verifier
# File:         verifier.mojo
# Path:         src/momijo/ir/verifier.mojo
#
# Description:  src.momijo.ir.verifier — focused Momijo functionality with a stable public API.
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
# Verifier
# -----------------------------

struct VerifyOptions:
    var allow_empty_blocks: Bool
    var require_entry_first: Bool
    var enforce_value_id_monotonic: Bool
fn __init__(out self,
                allow_empty_blocks: Bool = False,
                require_entry_first: Bool = True,
                enforce_value_id_monotonic: Bool = False) -> None:
        self.allow_empty_blocks = allow_empty_blocks
        self.require_entry_first = require_entry_first
        self.enforce_value_id_monotonic = enforce_value_id_monotonic
fn __copyinit__(out self, other: Self) -> None:
        self.allow_empty_blocks = other.allow_empty_blocks
        self.require_entry_first = other.require_entry_first
        self.enforce_value_id_monotonic = other.enforce_value_id_monotonic
fn __moveinit__(out self, deinit other: Self) -> None:
        self.allow_empty_blocks = other.allow_empty_blocks
        self.require_entry_first = other.require_entry_first
        self.enforce_value_id_monotonic = other.enforce_value_id_monotonic
struct VerifyError:
    var message: String
    var loc: Location
fn __init__(out self, message: String, loc: Location = Location()):
        self.message = message
        self.loc = loc
fn __copyinit__(out self, other: Self) -> None:
        self.message = other.message
        self.loc = other.loc
fn __moveinit__(out self, deinit other: Self) -> None:
        self.message = other.message
        self.loc = other.loc
struct VerifyReport:
    var ok: Bool
    var errors: List[VerifyError]
fn __init__(out self) -> None:
        self.ok = True
        self.errors = List[VerifyError]()
fn add_error(self, msg: String, loc: Location = Location()) -> None:
        self.ok = False
        self.errors.push_back(VerifyError(msg, loc))
fn summary(self) -> String:
        if self.ok:
            return String("OK")
        var s = String("FAIL (") + String(self.errors.size()) + String(" errors)")
        return s
fn __copyinit__(out self, other: Self) -> None:
        self.ok = other.ok
        self.errors = other.errors
fn __moveinit__(out self, deinit other: Self) -> None:
        self.ok = other.ok
        self.errors = other.errors
struct Verifier:
    var opt: VerifyOptions
fn __init__(out self, opt: VerifyOptions = VerifyOptions()):
        self.opt = opt
fn _is_i64(self, t: TypeDesc) -> Bool:
        return t.name == String("i64")
fn _is_bool(self, t: TypeDesc) -> Bool:
        return t.name == String("bool")
fn _result_values(self, op: Op) -> List[Value]:
        var xs = List[Value]()
        for i in range(op.results.size()):
            xs.push_back(op.results[i].value())
        return xs
fn _operand_values(self, op: Op) -> List[Value]:
        var xs = List[Value]()
        for i in range(op.operands.size()):
            xs.push_back(op.operands[i].value())
        return xs
fn verify_module(self, m: Module) -> VerifyReport:
        var rep = VerifyReport()

        # unique function names
        var seen = Dict[String, Int]()
        for i in range(m.functions.size()):
            var name = m.functions[i].name
            if seen.contains_key(name):
                rep.add_error(String("Duplicate function: ") + name)
            else:
                seen[name] = 1

        # verify each function
        for i in range(m.functions.size()):
            self._verify_function(m.functions[i], rep)
        return rep
fn _verify_function(self, f: Function, rep: VerifyReport) -> None:
        # entry block checks
        if f.blocks.size() == 0:
            rep.add_error(String("Function has no blocks: ") + f.name)
            return

        if self.opt.require_entry_first and f.blocks[0].name != String("entry"):
            rep.add_error(String("First block must be ^entry in function: ") + f.name)

        # argument arity/types = block-arg types
        var argn = f.arg_types.size()
        var bargn = f.blocks[0].args.size()
        if argn != bargn:
            rep.add_error(String("Entry block arg count mismatch in function: ") + f.name)

        var k = 0
        while k < argn and k < bargn:
            if f.blocks[0].args[k].typ.name != f.arg_types[k].name:
                rep.add_error(String("Type mismatch for entry arg #") + String(k) + String(" in function: ") + f.name)
            k = k + 1

        # value table of available SSA values per block (args + prior op results)
        for bi in range(f.blocks.size()):
            var b = f.blocks[bi]
            if (not self.opt.allow_empty_blocks) and b.ops.size() == 0:
                rep.add_error(String("Empty block not allowed: ^") + b.name)

            # track seen ids if enforcing monotonic IDs
            var last_id = -1
            var avail = Dict[Int, TypeDesc]()
            # block args available
            for ai in range(b.args.size()):
                var v = b.args[ai]
                avail[v.id] = v.typ
                if self.opt.enforce_value_id_monotonic and v.id <= last_id:
                    rep.add_error(String("Non-monotonic value id in block args of ^") + b.name)
                if v.id > last_id: last_id = v.id

            # all ops
            for oi in range(b.ops.size()):
                var op = b.ops[oi]

                # Operand must refer to available values
                for pi in range(op.operands.size()):
                    var vv = op.operands[pi].value()
                    if not avail.contains_key(vv.id):
                        rep.add_error(String("Use-before-def value %") + String(vv.id) + String(" in ^") + b.name, op.loc)
                    else:
                        # type must match recorded
                        var t = avail[vv.id]
                        if t.name != vv.typ.name:
                            rep.add_error(String("Type inconsistency for operand %") + String(vv.id), op.loc)

                # Dialect-specific type rules
                self._verify_op_types(op, rep)

                # Register produced results as available
                for ri in range(op.results.size()):
                    var rv = op.results[ri].value()
                    if avail.contains_key(rv.id):
                        rep.add_error(String("Redefinition of value %") + String(rv.id) + String(" in ^") + b.name, op.loc)
                    avail[rv.id] = rv.typ
                    if self.opt.enforce_value_id_monotonic and rv.id <= last_id:
                        rep.add_error(String("Non-monotonic value id in results in ^") + b.name, op.loc)
                    if rv.id > last_id: last_id = rv.id

                # Return validation vs function return types (last op only by convention)
                if op.name == String("return"):
                    var vals = List[Value]()
                    for vi in range(op.operands.size()):
                        vals.push_back(op.operands[vi].value())
                    if vals.size() != f.ret_types.size():
                        rep.add_error(String("Return arity mismatch in function: ") + f.name, op.loc)
                    else:
                        for ri in range(vals.size()):
                            if vals[ri].typ.name != f.ret_types[ri].name:
                                rep.add_error(String("Return type mismatch at index ") + String(ri) + String(" in function: ") + f.name, op.loc)
fn _verify_op_types(self, op: Op, rep: VerifyReport) -> None:
        # const.i64: 1 result: i64, must have attr "value=" prefix
        if op.name == String("const.i64"):
            if op.results.size() != 1:
                rep.add_error(String("const.i64 must produce 1 result"), op.loc)
            else:
                if not self._is_i64(op.results[0].value().typ):
                    rep.add_error(String("const.i64 result must be i64"), op.loc)
            # check attribute carries value
            var has_val = False
            for i in range(op.attrs.size()):
                if op.attrs[i].starts_with(String("value=")):
                    has_val = True
            if not has_val:
                rep.add_error(String("const.i64 missing value attribute"), op.loc)

        # add.i64 / mul.i64: 2 operands i64 -> 1 result i64
        if op.name == String("add.i64") or op.name == String("mul.i64"):
            if op.operands.size() != 2:
                rep.add_error(op.name + String(" must have 2 operands"), op.loc)
            else:
                if not self._is_i64(op.operands[0].value().typ) or not self._is_i64(op.operands[1].value().typ):
                    rep.add_error(op.name + String(" operands must be i64"), op.loc)
            if op.results.size() != 1:
                rep.add_error(op.name + String(" must produce 1 result"), op.loc)
            else:
                if not self._is_i64(op.results[0].value().typ):
                    rep.add_error(op.name + String(" result must be i64"), op.loc)

        # return: any arity allowed; types checked at function level
fn __copyinit__(out self, other: Self) -> None:
        self.opt = other.opt
fn __moveinit__(out self, deinit other: Self) -> None:
        self.opt = other.opt
# -----------------------------
# Self-test
# -----------------------------
fn _build_demo_module(ok_case: Bool) -> Module:
    var m = Module(String("demo"))
    var f = Function(String("sum_mul"))
    f.arg_types.push_back(TypeDesc(String("i64")))
    f.arg_types.push_back(TypeDesc(String("i64")))
    f.ret_types.push_back(TypeDesc(String("i64")))

    var entry = Block(String("entry"))
    var a = Value(0, TypeDesc(String("i64")))
    var b = Value(1, TypeDesc(String("i64")))
    entry.args.push_back(a)
    entry.args.push_back(b)

    var add = Op(String("add.i64"), Location(String("demo.mojo"), 1, 1))
    add.operands.push_back(Operand(a))
    add.operands.push_back(Operand(b))
    add.results.push_back(Result(Value(2, TypeDesc(String("i64")))))

    var cst = Op(String("const.i64"), Location(String("demo.mojo"), 1, 2))
    cst.results.push_back(Result(Value(3, TypeDesc(String("i64")))))
    cst.attrs.push_back(String("value=2"))

    var mul = Op(String("mul.i64"), Location(String("demo.mojo"), 1, 3))
    mul.operands.push_back(Operand(Value(2, TypeDesc(String("i64")))))
    mul.operands.push_back(Operand(Value(3, TypeDesc(String("i64")))))
    mul.results.push_back(Result(Value(4, TypeDesc(String("i64")))))

    var ret = Op(String("return"), Location(String("demo.mojo"), 1, 4))
    ret.operands.push_back(Operand(Value(4, TypeDesc(String("i64")))))

    entry.ops.push_back(add)
    entry.ops.push_back(cst)
    entry.ops.push_back(mul)
    entry.ops.push_back(ret)

    f.blocks.push_back(entry)
    m.functions.push_back(f)

    if not ok_case:
        # introduce a deliberate error: wrong result type for add
        f.blocks[0].ops[0].results[0].value().typ = TypeDesc(String("bool"))
    return m
fn _self_test_verifier() -> Bool:
    var v = Verifier(VerifyOptions(False, True, True))

    var m_ok = _build_demo_module(True)
    var rep_ok = v.verify_module(m_ok)
    var ok1 = rep_ok.ok

    var m_bad = _build_demo_module(False)
    var rep_bad = v.verify_module(m_bad)
    var ok2 = not rep_bad.ok

    var ok = ok1 and ok2
    if ok:
        print(String("OK"))
    else:
        print(String("FAIL"))
    return ok