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
# Project: momijo.vision.ir
# File: momijo/vision/ir/fusion.mojo
 
 

from momijo.vision.ir.ir import Graph, OpKind, OpNode

# -----------------------------------------------------------------------------
# Small compatibility helpers (field vs. accessor style)
# If your OpNode exposes kind()/oh()/ow() methods, switch the implementations below.
# -----------------------------------------------------------------------------
fn _node_kind(n: OpNode) -> OpKind:
    return n._kind        # or: return n.kind()

fn _node_oh(n: OpNode) -> Int:
    return n._oh          # or: return n.oh()

fn _node_ow(n: OpNode) -> Int:
    return n._ow          # or: return n.ow()

fn _make_resize(oh: Int, ow: Int) -> OpNode:
    # Adjust this ctor to your real OpNode constructor if different.
    return OpNode(OpKind.ResizeNearest, oh, ow)

# In some IR versions you may need: g.set_nodes(nodes: List[OpNode])
fn _set_nodes(mut g: Graph, nodes: List[OpNode]):
    # Prefer a setter if available; otherwise assign to the backing field.
    g._nodes = nodes

# -----------------------------------------------------------------------------
# Pass: fuse_resizes
# -----------------------------------------------------------------------------
fn fuse_resizes(mut g: Graph):
    var count = g.node_count()
    if count == 0:
        return

    # Pass 1: find the last resize and its target dims
    var last_idx: Int = -1
    var last_oh: Int = -1
    var last_ow: Int = -1

    var i: Int = 0
    while i < count:
        var n = g.node_at(i)
        if _node_kind(n) == OpKind.ResizeNearest:
            last_idx = i
            last_oh = _node_oh(n)
            last_ow = _node_ow(n)
        i += 1

    if last_idx < 0:
         return

    # Pass 2: rebuild node list
     #           ResizeNearest to (last_oh, last_ow); skip earlier resizes.
    var new_nodes = List[OpNode]()
    var j: Int = 0
    while j < count:
        var n = g.node_at(j)
        if _node_kind(n) == OpKind.ResizeNearest:
            if j == last_idx:
                new_nodes.append(_make_resize(last_oh, last_ow))
            # else: drop
        else:
            new_nodes.append(n)
        j += 1

    _set_nodes(g, new_nodes)

# -----------------------------------------------------------------------------
# Pass: simplify_convert_color
# -----------------------------------------------------------------------------
fn simplify_convert_color(mut g: Graph):
    var count = g.node_count()
    if count == 0:
        return

    var new_nodes = List[OpNode]()
    var prev_was_gray: Bool = False

    var i: Int = 0
    while i < count:
        var n = g.node_at(i)
        var k = _node_kind(n)

        if k == OpKind.ConvertRGBToGray:
            # Collapse consecutive rgb->gray into a single one.
            if not prev_was_gray:
                new_nodes.append(n)
                prev_was_gray = True
            # else: drop this redundant conversion
        else:
            new_nodes.append(n)
            prev_was_gray = False
        i += 1

    _set_nodes(g, new_nodes)
