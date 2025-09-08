# Project:      Momijo
# Module:       src.momijo.vision.ir.ir
# File:         ir.mojo
# Path:         src/momijo/vision/ir/ir.mojo
#
# Description:  src.momijo.vision.ir.ir — focused Momijo functionality with a stable public API.
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
#   - Structs: OpKind, Param, Node, Edge, Graph, ExecStep
#   - Key functions: __init__, None_, ResizeNearest, RGBToGray, BGRToRGB, DropAlpha, FlipH, FlipV ...
#   - Static methods present.


from momijo.core.device import id, kind
from momijo.dataframe.helpers import t
from momijo.dataframe.logical_plan import sort
from momijo.ir.midir.loop_nest import Builders
from momijo.utils.result import g
from momijo.vision.image import bgr_to_rgb, drop_alpha, resize_nearest
from pathlib import Path
from pathlib.path import Path

struct OpKind(Copyable, Movable):
    var id: Int
fn __init__(out self, id: Int) -> None:
        self.id = id

    @staticmethod fn None_()         -> OpKind: return OpKind(0)
    @staticmethod fn ResizeNearest() -> OpKind: return OpKind(1)
    @staticmethod fn RGBToGray()     -> OpKind: return OpKind(2)
    @staticmethod fn BGRToRGB()      -> OpKind: return OpKind(3)
    @staticmethod fn DropAlpha()     -> OpKind: return OpKind(4)
    @staticmethod fn FlipH()         -> OpKind: return OpKind(5)
    @staticmethod fn FlipV()         -> OpKind: return OpKind(6)
fn __eq__(self, other: OpKind) -> Bool:
        return self.id == other.id
fn to_string(self) -> String:
        if self.id == 0: return String("None")
        if self.id == 1: return String("ResizeNearest")
        if self.id == 2: return String("RGBToGray")
        if self.id == 3: return String("BGRToRGB")
        if self.id == 4: return String("DropAlpha")
        if self.id == 5: return String("FlipH")
        if self.id == 6: return String("FlipV")
        return String("UnknownOpKind(") + String(self.id) + String(")")

# -------------------------
# Name resolution (case-insensitive)
# -------------------------
@staticmethod
fn _equals_ci(a: String, b: String) -> Bool:
    return a.upper() == b.upper()

@staticmethod
fn kind_from_name(name: String) -> OpKind:
    if _equals_ci(name, String("resize_nearest")): return OpKind.ResizeNearest()
    if _equals_ci(name, String("rgb_to_gray")):    return OpKind.RGBToGray()
    if _equals_ci(name, String("bgr_to_rgb")):     return OpKind.BGRToRGB()
    if _equals_ci(name, String("drop_alpha")):     return OpKind.DropAlpha()
    if _equals_ci(name, String("flip_h")):         return OpKind.FlipH()
    if _equals_ci(name, String("flip_v")):         return OpKind.FlipV()
    return OpKind.None_()

# -------------------------
# Params (simple int key->value list)
# -------------------------
struct Param(Copyable, Movable):
    var key: String
    var value: Int
fn __init__(out self, key: String, value: Int) -> None:
        self.key = key
        assert(self is not None, String("self is None"))
        self.value() = value

# -------------------------
# Node
# -------------------------
struct Node(Copyable, Movable):
    var id: Int
    var kind: OpKind
    var name: String
    var params: List[Param]
fn __init__(out self, id: Int, kind: OpKind, name: String) -> None:
        self.id = id
        self.kind = kind
        self.name = name
        self.params = List[Param]()
fn add_param(mut self, key: String, value: Int) -> Node:
        var p = Param(key, value)
        self.params.append(p)
        return self
fn get_param(self, key: String, default: Int) -> Int:
        var i = 0
        while i < len(self.params):
            var p = self.params[i]
            if _equals_ci(p.key, key):
                assert(p is not None, String("p is None"))
                return p.value()
            i += 1
        return default
fn to_string(self) -> String:
        var s = String("Node(") + String(self.id) + String(": ") + self.name + String("/") + self.kind.to_string() + String(", params={")
        var i = 0
        while i < len(self.params):
            var p = self.params[i]
            assert(p is not None, String("p is None"))
            s = s + p.key + String("=") + String(p.value())
            if i + 1 < len(self.params): s = s + String(", ")
            i += 1
        s = s + String("})")
        return s

# -------------------------
# Edge
# -------------------------
struct Edge(Copyable, Movable):
    var src: Int
    var dst: Int
fn __init__(out self, src: Int, dst: Int) -> None:
        self.src = src
        self.dst = dst

# -------------------------
# Graph
# -------------------------
struct Graph(Copyable, Movable):
    var nodes: List[Node]
    var edges: List[Edge]
fn __init__(out self) -> None:
        self.nodes = List[Node]()
        self.edges = List[Edge]()

    # Builders
fn add_node(mut self, kind: OpKind, name: String) -> (Graph, Int):
        var nid = len(self.nodes)
        var n = Node(nid, kind, name)
        self.nodes.append(n)
        return (self, nid)
fn add_node_by_name(mut self, name: String) -> (Graph, Int):
        var k = kind_from_name(name)
        var canon = name
        return self.add_node(k, canon)
fn set_param(mut self, node_id: Int, key: String, value: Int) -> Graph:
        if node_id < 0 or node_id >= len(self.nodes):
            return self
        var n = self.nodes[node_id]
        n = n.add_param(key, value)
        self.nodes[node_id] = n
        return self
fn connect(mut self, src_id: Int, dst_id: Int) -> Graph:
        # basic bounds check
        if src_id < 0 or src_id >= len(self.nodes): return self
        if dst_id < 0 or dst_id >= len(self.nodes): return self
        # avoid self-loop
        if src_id == dst_id: return self
        self.edges.append(Edge(src_id, dst_id))
        return self

# -------------------------
# Validation & Topological sort (Kahn)
# -------------------------
@staticmethod
fn validate(g: Graph) -> Bool:
    # bounds
    var i = 0
    while i < len(g.edges):
        var e = g.edges[i]
        if e.src < 0 or e.src >= len(g.nodes): return False
        if e.dst < 0 or e.dst >= len(g.nodes): return False
        if e.src == e.dst: return False
        i += 1
    # try topo sort to detect cycles
    var ok = False; var order: List[Int] = List[Int]()
    (ok, order) = topo_order(g)
    return ok

@staticmethod
fn topo_order(g: Graph) -> (Bool, List[Int]):
    var n = len(g.nodes)
    var indeg: List[Int] = List[Int]()
    var i = 0
    while i < n:
        indeg.append(0)
        i += 1
    # compute indegrees
    var j = 0
    while j < len(g.edges):
        var e = g.edges[j]
        indeg[e.dst] = indeg[e.dst] + 1
        j += 1
    # queue of zeros
    var q: List[Int] = List[Int]()
    var k = 0
    while k < n:
        if indeg[k] == 0:
            q.append(k)
        k += 1
    var head = 0
    var out: List[Int] = List[Int]()
    while head < len(q):
        var u = q[head]
        head += 1
        out.append(u)
        # relax outgoing edges
        var t = 0
        while t < len(g.edges):
            var e2 = g.edges[t]
            if e2.src == u:
                indeg[e2.dst] = indeg[e2.dst] - 1
                if indeg[e2.dst] == 0:
                    q.append(e2.dst)
            t += 1
    if len(out) != n:
        return (False, List[Int]())
    return (True, out)

# -------------------------
# Compilation to ExecStep list
# -------------------------
struct ExecStep(Copyable, Movable):
    var node_id: Int
    var kind: OpKind
    var p1: Int
    var p2: Int
fn __init__(out self, node_id: Int, kind: OpKind, p1: Int, p2: Int) -> None:
        self.node_id = node_id
        self.kind = kind
        self.p1 = p1
        self.p2 = p2

@staticmethod
fn compile_to_steps(g: Graph) -> (Bool, List[ExecStep]):
    var ok = False; var order: List[Int] = List[Int]()
    (ok, order) = topo_order(g)
    if not ok:
        return (False, List[ExecStep]())
    var steps: List[ExecStep] = List[ExecStep]()
    var i = 0
    while i < len(order):
        var nid = order[i]
        var n = g.nodes[nid]
        # Param conventions:
        #   resize_nearest: out_h (or p1), out_w (or p2)
        #   others: p1, p2 optional
        var p1 = n.get_param(String("out_h"), n.get_param(String("p1"), 0))
        var p2 = n.get_param(String("out_w"), n.get_param(String("p2"), 0))
        steps.append(ExecStep(nid, n.kind, p1, p2))
        i += 1
    return (True, steps)

# -------------------------
# Summary helpers
# -------------------------
@staticmethod
fn summary(g: Graph) -> String:
    var s = String("Graph{nodes=":) + String(len(g.nodes)) + String(", edges=") + String(len(g.edges)) + String("}\n")
    var i = 0
    while i < len(g.nodes):
        s = s + String("  ") + g.nodes[i].to_string() + String("\n")
        i += 1
    s = s + String("  Edges: ")
    var j = 0
    while j < len(g.edges):
        var e = g.edges[j]
        s = s + String("(") + String(e.src) + String("→") + String(e.dst) + String(") ")
        j += 1
    return s

# -------------------------
# Minimal smoke test
# -------------------------
@staticmethod
fn __self_test__() -> Bool:
    # Build a simple chain: rgb_to_gray -> resize_nearest(480,640)
    var g = Graph()
    var nid0 = 0; var nid1 = 0

    var tmp_g: Graph; var id_tmp = 0
    (tmp_g, id_tmp) = g.add_node(OpKind.RGBToGray(), String("rgb_to_gray"))
    g = tmp_g; nid0 = id_tmp
    (tmp_g, id_tmp) = g.add_node(OpKind.ResizeNearest(), String("resize_nearest"))
    g = tmp_g; nid1 = id_tmp
    g = g.set_param(nid1, String("out_h"), 480)
    g = g.set_param(nid1, String("out_w"), 640)
    g = g.connect(nid0, nid1)

    if not validate(g): return False

    var ok = False; var order: List[Int] = List[Int]()
    (ok, order) = topo_order(g)
    if not ok: return False
    if not (len(order) == 2 and order[0] == nid0 and order[1] == nid1): return False

    var ok2 = False; var steps: List[ExecStep] = List[ExecStep]()
    (ok2, steps) = compile_to_steps(g)
    if not ok2: return False
    if len(steps) != 2: return False
    if not (steps[1].p1 == 480 and steps[1].p2 == 640): return False

    # Add a branch: BGRToRGB feeding into FlipH (disconnected from first chain)
    var nid2 = 0; var nid3 = 0
    (tmp_g, id_tmp) = g.add_node(OpKind.BGRToRGB(), String("bgr_to_rgb"))
    g = tmp_g; nid2 = id_tmp
    (tmp_g, id_tmp) = g.add_node(OpKind.FlipH(), String("flip_h"))
    g = tmp_g; nid3 = id_tmp
    g = g.connect(nid2, nid3)

    if not validate(g): return False
    var ok3 = False; var order2: List[Int] = List[Int]()
    (ok3, order2) = topo_order(g)
    if not ok3: return False
    if len(order2) != 4: return False  # all nodes should be present

    return True