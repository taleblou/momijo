# MIT License
# Copyright (c) 2025 Morteza Talebou and Mitra Daneshmand
# Project: momijo  |  Source: https://github.com/taleblou/momijo
# This file is part of the Momijo project. See the LICENSE file at the repository root.
# Momijo 
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Morteza Talebou and Mitra Daneshmand
# Website: https://taleblou.ir/
# Repository: https://github.com/taleblou/momijo
 
from momijo.vision.dtypes import DType

# -----------------------------------------------------------------------------
# NodeId
# -----------------------------------------------------------------------------
@fieldwise_init("implicit")
struct NodeId:
    var _id: Int

    fn __init__(out self, id: Int):
        self._id = id

    fn id(self) -> Int:
        return self._id

# -----------------------------------------------------------------------------
# OpKind
# -----------------------------------------------------------------------------
@value
enum OpKind(Int):
    ResizeNearest    = 0
    ConvertRGBToGray = 1

# -----------------------------------------------------------------------------
# OpNode
# -----------------------------------------------------------------------------
@fieldwise_init
struct OpNode:
    var _kind: OpKind
    var _oh: Int
    var _ow: Int

    fn __init__(out self, kind: OpKind, oh: Int, ow: Int):
        self._kind = kind
        self._oh = oh
        self._ow = ow

    # Accessors (useful for IR clients that prefer methods)
    fn kind(self) -> OpKind: return self._kind
    fn oh(self) -> Int:      return self._oh
    fn ow(self) -> Int:      return self._ow

# -----------------------------------------------------------------------------
# Graph
# -----------------------------------------------------------------------------
@fieldwise_init("implicit")
struct Graph:
    var _nodes: List[OpNode]

    fn __init__(out self):
        self._nodes = List[OpNode]()

    fn __copyinit__(out self, other: Self):
        # Deep copy node list
        var copied = List[OpNode]()
        var i = 0
        var n = len(other._nodes)
        while i < n:
            copied.append(other._nodes[i])
            i += 1
        self._nodes = copied

    # Basic API
    fn add_node(mut self, node: OpNode):
        self._nodes.append(node)

    fn node_count(self) -> Int:
        return len(self._nodes)

    fn node_at(self, i: Int) -> OpNode:
        return self._nodes[i]

    fn nodes(self) -> List[OpNode]: 
        var out = List[OpNode]()
        var i = 0
        var n = len(self._nodes)
        while i < n:
            out.append(self._nodes[i])
            i += 1
        return out.copy()

    fn clear(mut self):
        self._nodes = List[OpNode]()

    fn set_nodes(mut self, nodes: List[OpNode]):
        # Replace the node list (copies elements)
        var out = List[OpNode]()
        var i = 0
        var n = len(nodes)
        while i < n:
            out.append(nodes[i])
            i += 1
        self._nodes = out

    # Convenience helpers
    fn append_resize(mut self, oh: Int, ow: Int):
        self.add_node(OpNode(OpKind.ResizeNearest, oh, ow))

    fn append_rgb_to_gray(mut self):
        self.add_node(OpNode(OpKind.ConvertRGBToGray, 0, 0))

    fn last_index_of_resize(self) -> Int:
        var i = len(self._nodes) - 1
        while i >= 0:
            if self._nodes[i]._kind == OpKind.ResizeNearest:
                return i
            i -= 1
        return -1

# --- Free-function wrapper (if external code expects this symbol) ---
fn Graph_add_node(mut x: Graph, node: OpNode):
    x.add_node(node)
