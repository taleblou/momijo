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
# Project: momijo.ir.midir
# File: src/momijo/ir/midir/bufferization.mojo

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

# -----------------------------
fn is_tensor(t: TypeDesc) -> Bool: return t.name.starts_with(String("tensor<"))
fn is_buffer(t: TypeDesc) -> Bool: return t.name.starts_with(String("buffer<"))
fn tensor_to_buffer_type(t: TypeDesc) -> TypeDesc:
    if not is_tensor(t): return t
    # Replace prefix "tensor<" with "buffer<"
    var s = t.name
    var out = String("buffer<")
    var i = 7  # len("tensor<")
    while i < s.size():
        out = out + String.from_utf8([s.bytes()[i]])
        i = i + 1
    return TypeDesc(out)
fn buffer_to_tensor_type(t: TypeDesc) -> TypeDesc:
    if not is_buffer(t): return t
    var s = t.name
    var out = String("tensor<")
    var i = 7  # len("buffer<")
    while i < s.size():
        out = out + String.from_utf8([s.bytes()[i]])
        i = i + 1
    return TypeDesc(out)

# -----------------------------
# Id generator
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
# -----------------------------
# Bufferization pass
# -----------------------------

struct Bufferization:
    var idg: IdGen
fn __init__(out self) -> None:
        self.idg = IdGen()
fn _insert_allocs_for_rets(self, f: Function, b0: Block) -> List[Value]:
        # For each tensor return type, create a buf.alloc at function entry.
        var outs = List[Value]()
        for ri in range(f.ret_types.size()):
            var rt = f.ret_types[ri]
            if is_tensor(rt):
                var bt = tensor_to_buffer_type(rt)
                var v = Value(self.idg.fresh(), bt)
                var op = Op(String("buf.alloc"), Location(String(f.name), 0, 0))
                op.results.push_back(Result(v))
                # Optionally attach shape as attr copied from type string
                op.attrs.push_back(String("like=") + rt.name)
                # Insert at beginning
                b0.ops.insert(0, op)
                outs.push_back(v)
            else:
                # Non-tensor rets are not allocated
                var dummy = Value(-1, rt)
                outs.push_back(dummy)
        return outs
fn _rewrite_return_to_buffers(self, f: Function, b0: Block, out_bufs: List[Value]) -> None:
        # Find return op and rewrite tensor returns into stores+buffer returns.
        for oi in range(b0.ops.size()):
            var op = b0.ops[oi]
            if op.name == String("ll.ret") or op.name == String("mid.ret") or op.name == String("hl.return") or op.name == String("return"):
                # Build new operand list
                var new_ops = List[Operand]()
                var idx = 0
                for pi in range(op.operands.size()):
                    var v = op.operands[pi].value()
                    var rt = f.ret_types[idx]
                    if is_tensor(rt):
                        # store tensor value into pre-alloc buffer, then return the buffer
                        var bufv = out_bufs[idx]
                        var st = Op(String("buf.store_tensor"), op.loc)
                        st.operands.push_back(Operand(v))
                        st.operands.push_back(Operand(bufv))
                        # place store right before return
                        b0.ops.insert(oi, st)
                        new_ops.push_back(Operand(bufv))
                    else:
                        # pass-through scalar
                        new_ops.push_back(Operand(v))
                    idx = idx + 1
                # mutate return op name to ll.ret (normalized)
                op.name = String("ll.ret")
                # replace operands
                op.operands = new_ops
                # NOTE: do not change results
                break
fn run_on_function(self, f: Function) -> Function:
        if f.blocks.size() == 0:
            return f
        # Only handle single-entry block for this minimal pass
        var b0 = f.blocks[0]

        # Allocate output buffers for tensor returns
        var out_bufs = self._insert_allocs_for_rets(f, b0)

        # Rewrite return to return buffers instead of tensors
        self._rewrite_return_to_buffers(f, b0, out_bufs)

        for ri in range(f.ret_types.size()):
            var rt = f.ret_types[ri]
            if is_tensor(rt):
                f.ret_types[ri] = tensor_to_buffer_type(rt)

        return f
fn run_on_module(self, m: Module) -> Module:
        var out = Module(m.name + String("_buf"))
        for fi in range(m.functions.size()):
            var f = m.functions[fi]
            # clone skeleton
            var nf = Function(f.name)
            for ai in range(f.arg_types.size()): nf.arg_types.push_back(f.arg_types[ai])
            for ri in range(f.ret_types.size()): nf.ret_types.push_back(f.ret_types[ri])
            # clone blocks/ops shallowly (we'll mutate nf in-place)
            for bi in range(f.blocks.size()):
                var b = f.blocks[bi]
                var nb = Block(b.name)
                for aa in range(b.args.size()): nb.args.push_back(b.args[aa])
                for oi in range(b.ops.size()): nb.ops.push_back(b.ops[oi])
                nf.blocks.push_back(nb)
            # bufferize
            var bf = self.run_on_function(nf)
            out.functions.push_back(bf)
        return out
fn __copyinit__(out self, other: Self) -> None:
        self.idg = other.idg
fn __moveinit__(out self, deinit other: Self) -> None:
        self.idg = other.idg
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
fn _tensor_i64_1d(n: Int) -> TypeDesc:
    return TypeDesc(String("tensor<i64,[") + String(n) + String("]>"))
fn _make_demo_llir() -> Module:
    var m = Module(String("demo_llir"))
    var f = Function(String("make_vec"))
    # returns tensor<i64,[1024]>
    f.ret_types.push_back(_tensor_i64_1d(1024))

    var idg = IdGen()
    var b = Block(String("entry"))
    # fake producer of a tensor
    var v_t = Value(idg.fresh(), _tensor_i64_1d(1024))
    var make = Op(String("ll.make.tensor"), Location(String("demo.mojo"), 1, 1))
    make.results.push_back(Result(v_t))

    var ret = Op(String("ll.ret"), Location(String("demo.mojo"), 1, 2))
    ret.operands.push_back(Operand(v_t))

    b.ops.push_back(make)
    b.ops.push_back(ret)

    f.blocks.push_back(b)
    m.functions.push_back(f)
    return m
fn _self_test_bufferization() -> Bool:
    var src = _make_demo_llir()
    var buf = Bufferization()
    var dst = buf.run_on_module(src)

    var pr = Printer()
    var s2 = pr.print(dst)

    var ok = True
    if s2.find(String("buf.alloc")) < 0: ok = False
    if s2.find(String("buf.store_tensor")) < 0: ok = False
    if s2.find(String("buffer<i64,[1024]>")) < 0: ok = False
    if s2.find(String("ll.ret")) < 0: ok = False
    if ok: print(String("OK"))
    else: print(String("FAIL"))
    return ok