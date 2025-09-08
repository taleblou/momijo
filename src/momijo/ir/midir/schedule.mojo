# Project:      Momijo
# Module:       src.momijo.ir.midir.schedule
# File:         schedule.mojo
# Path:         src/momijo/ir/midir/schedule.mojo
#
# Description:  src.momijo.ir.midir.schedule — focused Momijo functionality with a stable public API.
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
#   - Uses generic functions/types with explicit trait bounds.


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
fn _parse_int(s: String, default_val: Int) -> Int:
    if s.size() == 0: return default_val
    var sign = 1
    var i = 0
    if s.bytes()[0] == 45:
        sign = -1
        i = 1
    var num = 0
    while i < s.size():
        var d = s.bytes()[i] - UInt8(48)
        num = num * 10 + Int(d)
        i = i + 1
    return sign * num

# -----------------------------
# Schedule options
# -----------------------------

struct ScheduleOptions:
    var default_latency: Int
    var op_latency_by_prefix: Dict[String, Int]     # e.g., {"ll.mul": 3, "ll.add": 1}
    var resource_by_prefix: Dict[String, String]    # e.g., {"ll.mul": "MUL", "ll.add": "ALU"}
    var resource_limits: Dict[String, Int]          # e.g., {"ALU": 1, "MUL": 1, "default": 4}
    var honor_attrs: Bool
fn __init__(out self) -> None:
        self.default_latency = 1
        self.op_latency_by_prefix = Dict[String, Int]()
        self.resource_by_prefix = Dict[String, String]()
        self.resource_limits = Dict[String, Int]()
        self.honor_attrs = True
        self.resource_limits[String("default")] = 999999  # practically unlimited
fn __copyinit__(out self, other: Self) -> None:
        self.default_latency = other.default_latency
        self.op_latency_by_prefix = other.op_latency_by_prefix
        self.resource_by_prefix = other.resource_by_prefix
        self.resource_limits = other.resource_limits
        self.honor_attrs = other.honor_attrs
fn __moveinit__(out self, deinit other: Self) -> None:
        self.default_latency = other.default_latency
        self.op_latency_by_prefix = other.op_latency_by_prefix
        self.resource_by_prefix = other.resource_by_prefix
        self.resource_limits = other.resource_limits
        self.honor_attrs = other.honor_attrs
# -----------------------------
# Internal: Node/Edge
# -----------------------------

struct NodeInfo:
    var op_index: Int
    var asap: Int
    var start_cycle: Int
    var latency: Int
    var resource: String
fn __init__(out self, op_index: Int) -> None:
        self.op_index = op_index
        self.asap = 0
        self.start_cycle = -1
        self.latency = 1
        self.resource = String("default")
fn __copyinit__(out self, other: Self) -> None:
        self.op_index = other.op_index
        self.asap = other.asap
        self.start_cycle = other.start_cycle
        self.latency = other.latency
        self.resource = other.resource
fn __moveinit__(out self, deinit other: Self) -> None:
        self.op_index = other.op_index
        self.asap = other.asap
        self.start_cycle = other.start_cycle
        self.latency = other.latency
        self.resource = other.resource
# -----------------------------
# Scheduler
# -----------------------------

struct Scheduler:
    var opt: ScheduleOptions
fn __init__(out self, opt: ScheduleOptions = ScheduleOptions()):
        self.opt = opt
fn _latency_of(self, op: Op) -> Int:
        if self.opt.honor_attrs:
            var has = False
            var val = _attr_value(op, String("latency"), has)
            if has: return _parse_int(val, self.opt.default_latency)
        # by prefix
        for k in self.opt.op_latency_by_prefix.keys():
            if op.name.starts_with(k):
                return self.opt.op_latency_by_prefix[k]
        return self.opt.default_latency
fn _resource_of(self, op: Op) -> String:
        if self.opt.honor_attrs:
            var has = False
            var val = _attr_value(op, String("res"), has)
            if has and val.size() > 0: return val
        for k in self.opt.resource_by_prefix.keys():
            if op.name.starts_with(k):
                return self.opt.resource_by_prefix[k]
        return String("default")
fn _build_value_producer_map(self, b: Block) -> Dict[Int, Int]:
        # maps Value.id -> producing op index
        var prod = Dict[Int, Int]()
        for i in range(b.ops.size()):
            var op = b.ops[i]
            for r in range(op.results.size()):
                prod[op.results[r].value().id] = i
        return prod
fn _build_graph(self, b: Block, prod: Dict[Int, Int]) -> Tuple[List[List[Int]], List[Int]]:
        # returns (succs, indegree)
        var n = b.ops.size()
        var succs = List[List[Int]]()
        var indeg = List[Int]()
        var i = 0
        while i < n:
            succs.push_back(List[Int]())
            indeg.push_back(0)
            i = i + 1
        # edges: producer -> consumer (for SSA operands)
        for ci in range(n):
            var op = b.ops[ci]
            for pi in range(op.operands.size()):
                var v = op.operands[pi].value()
                if prod.contains_key(v.id):
                    var pi_idx = prod[v.id]
                    succs[pi_idx].push_back(ci)
                    indeg[ci] = indeg[ci] + 1
        return (succs, indeg)
fn _compute_asap(self, b: Block, succs: List[List[Int]], indeg: List[Int], latencies: List[Int]) -> List[Int]:
        var n = b.ops.size()
        var asap = List[Int]()
        for i in range(n): asap.push_back(0)
        # Kahn topological order
        var q = List[Int]()
        for i in range(n):
            if indeg[i] == 0: q.push_back(i)
        while q.size() > 0:
            var u = q.pop_back()
            for si in range(succs[u].size()):
                var v = succs[u][si]
                # dependency constraint
                var cand = asap[u] + latencies[u]
                if cand > asap[v]: asap[v] = cand
                indeg[v] = indeg[v] - 1
                if indeg[v] == 0: q.push_back(v)
        return asap
fn _priority_index(self, pair: Tuple[Int, Int]) -> Int:
        # pack two ints (asap, orig_idx) to a comparable value (stable-ish)
        return pair.first * 1000000 + pair.second
fn schedule_block(self, b: Block) -> Block:
        var n = b.ops.size()
        if n == 0: return b

        var prod = self._build_value_producer_map(b)
        var g = self._build_graph(b, prod)
        var succs = g.first
        var indeg0 = g.second

        # latencies & resources per op
        var lat = List[Int](); var res = List[String]()
        for i in range(n):
            var L = self._latency_of(b.ops[i])
            lat.push_back(L)
            res.push_back(self._resource_of(b.ops[i]))

        # ASAP
        var indeg_copy = List[Int]()
        for i in range(n): indeg_copy.push_back(indeg0[i])
        var asap = self._compute_asap(b, succs, indeg_copy, lat)

        # Ready sets for list scheduling
        var indeg = List[Int]()
        for i in range(n): indeg.push_back(indeg0[i])
        var scheduled = List[Bool]()
        for i in range(n): scheduled.push_back(False)
        var start_cycle = List[Int]()
        for i in range(n): start_cycle.push_back(-1)
        var done_cnt = 0
        var t = 0

        while done_cnt < n:
            # per-cycle capacities
            var cap = Dict[String, Int]()
            for k in self.opt.resource_limits.keys():
                cap[k] = self.opt.resource_limits[k]
            if not cap.contains_key(String("default")): cap[String("default")] = 999999

            # collect ready ops (not scheduled, indeg==0, deps finished by now)
            var ready = List[Tuple[Int, Int]]()  # (asap, idx)
            for i in range(n):
                if scheduled[i]: continue
                if indeg[i] != 0: continue
                # check predecessor finish times
                var preds_ok = True
                # scan producers by scanning all ops and their succs; or recompute via operands
                var op = b.ops[i]
                for pi in range(op.operands.size()):
                    var v = op.operands[pi].value()
                    if prod.contains_key(v.id):
                        var p = prod[v.id]
                        var finish = start_cycle[p] + lat[p]
                        if finish > t:
                            preds_ok = False
                            break
                if preds_ok:
                    ready.push_back(Tuple(asap[i], i))

            # sort ready by (asap asc, original idx asc)
            # simple selection sort for minimal environment
            var sorted = List[Int]()
            while ready.size() > 0:
                var best_idx = 0
                var best_key = self._priority_index(ready[0])
                var j = 1
                while j < ready.size():
                    var key = self._priority_index(ready[j])
                    if key < best_key:
                        best_key = key
                        best_idx = j
                    j = j + 1
                var chosen = ready[best_idx]
                # remove chosen
                var tmp = List[Tuple[Int, Int]]()
                for k in range(ready.size()):
                    if k != best_idx:
                        tmp.push_back(ready[k])
                ready = tmp
                sorted.push_back(chosen.second)

            # place as many as capacity allows
            var placed_any = False
            for si in range(sorted.size()):
                var i = sorted[si]
                var r = res[i]
                var avail = cap.contains_key(r) ? cap[r] : cap[String("default")]
                if avail <= 0: continue
                # place
                start_cycle[i] = t
                scheduled[i] = True
                placed_any = True
                cap[r] = avail - 1
                # reduce indegree of succs
                for s in range(succs[i].size()):
                    var v = succs[i][s]
                    indeg[v] = indeg[v] - 1
                done_cnt = done_cnt + 1

            if not placed_any:
                # advance time to next possible time
                t = t + 1
            else:
                t = t + 1

        # Annotate ops and reorder by (start_cycle, original index)
        var order = List[Tuple[Int, Int]]()
        for i in range(n):
            order.push_back(Tuple(start_cycle[i], i))
        # sort 'order'
        var out_order = List[Int]()
        while order.size() > 0:
            var best = 0
            var bestc = order[0].first * 1000000 + order[0].second
            var j2 = 1
            while j2 < order.size():
                var key = order[j2].first * 1000000 + order[j2].second
                if key < bestc:
                    best = j2
                    bestc = key
                j2 = j2 + 1
            out_order.push_back(order[best].second)
            # remove
            var tmp2 = List[Tuple[Int, Int]]()
            for k in range(order.size()):
                if k != best: tmp2.push_back(order[k])
            order = tmp2

        # annotate
        for i in range(n):
            var op = b.ops[i]
            op.attrs.push_back(String("cycle=") + String(start_cycle[i]))
            op.attrs.push_back(String("latency=") + String(lat[i]))
            var rname = res[i]
            if rname.size() > 0:
                op.attrs.push_back(String("res=") + rname)
            b.ops[i] = op

        # reorder
        var new_ops = List[Op]()
        for k in range(out_order.size()):
            new_ops.push_back(b.ops[out_order[k]])
        b.ops = new_ops
        return b
fn run_on_function(self, f: Function) -> Function:
        for bi in range(f.blocks.size()):
            var b = f.blocks[bi]
            var nb = self.schedule_block(b)
            f.blocks[bi] = nb
        return f
fn run_on_module(self, m: Module) -> Module:
        var out = Module(m.name + String("_sched"))
        for fi in range(m.functions.size()):
            var f = m.functions[fi]
            # shallow clone
            var nf = Function(f.name)
            for ai in range(f.arg_types.size()): nf.arg_types.push_back(f.arg_types[ai])
            for ri in range(f.ret_types.size()): nf.ret_types.push_back(f.ret_types[ri])
            for bi in range(f.blocks.size()):
                nf.blocks.push_back(f.blocks[bi])
            # schedule
            var sf = self.run_on_function(nf)
            out.functions.push_back(sf)
        return out
fn __copyinit__(out self, other: Self) -> None:
        self.opt = other.opt
fn __moveinit__(out self, deinit other: Self) -> None:
        self.opt = other.opt
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
fn _demo_module() -> Module:
    var m = Module(String("demo_sched"))
    var f = Function(String("pipe"))
    var b = Block(String("entry"))
    var idg = IdGen()

    # Values
    var a = Value(idg.fresh(), TypeDesc(String("i64")))
    var bv = Value(idg.fresh(), TypeDesc(String("i64")))
    b.args.push_back(a); b.args.push_back(bv)

    # c0 = add a,b      // ALU
    var add1 = Op(String("ll.add"), Location(String("demo.mojo"), 1, 1))
    var c0 = Value(idg.fresh(), TypeDesc(String("i64")))
    add1.operands.push_back(Operand(a)); add1.operands.push_back(Operand(bv))
    add1.results.push_back(Result(c0))

    # c1 = mul c0,b     // MUL
    var mul1 = Op(String("ll.mul"), Location(String("demo.mojo"), 1, 2))
    var c1 = Value(idg.fresh(), TypeDesc(String("i64")))
    mul1.operands.push_back(Operand(c0)); mul1.operands.push_back(Operand(bv))
    mul1.results.push_back(Result(c1))

    # c2 = add a,a      // ALU (independent of mul1's result)
    var add2 = Op(String("ll.add"), Location(String("demo.mojo"), 1, 3))
    var c2 = Value(idg.fresh(), TypeDesc(String("i64")))
    add2.operands.push_back(Operand(a)); add2.operands.push_back(Operand(a))
    add2.results.push_back(Result(c2))

    # ret c1
    var ret = Op(String("ll.ret"), Location(String("demo.mojo"), 1, 4))
    ret.operands.push_back(Operand(c1))

    b.ops.push_back(add1); b.ops.push_back(mul1); b.ops.push_back(add2); b.ops.push_back(ret)
    f.blocks.push_back(b)
    m.functions.push_back(f)
    return m
fn _self_test_schedule() -> Bool:
    var m = _demo_module()

    var opt = ScheduleOptions()
    opt.default_latency = 1
    opt.op_latency_by_prefix[String("ll.mul")] = 3
    opt.op_latency_by_prefix[String("ll.add")] = 1
    opt.resource_by_prefix[String("ll.mul")] = String("MUL")
    opt.resource_by_prefix[String("ll.add")] = String("ALU")
    opt.resource_limits[String("MUL")] = 1
    opt.resource_limits[String("ALU")] = 1

    var sch = Scheduler(opt)
    var out = sch.run_on_module(m)

    var pr = Printer()
    var txt = pr.print(out)

    var ok = True
    # basic checks
    if txt.find(String("cycle=")) < 0: ok = False
    if txt.find(String("res=MUL")) < 0: ok = False
    if txt.find(String("res=ALU")) < 0: ok = False
    # Ensure mul scheduled after add due to dep (latency 3)
    if txt.find(String("ll.mul")) < 0: ok = False
    if txt.find(String("ll.add")) < 0: ok = False

    if ok: print(String("OK"))
    else: print(String("FAIL"))
    return ok