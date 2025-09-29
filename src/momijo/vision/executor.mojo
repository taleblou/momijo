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
# Project: momijo.vision
# File: momijo/vision/executor.mojo
  
# --- IR & transforms imports ---------------------------------------------------------
from momijo.vision.ir.ir import Graph, OpKind, OpNode
from momijo.vision.ir.fusion import fuse_resizes, simplify_convert_color

from momijo.vision.tensor import Tensor
from momijo.vision.transforms.resize import resize_nearest_u8_hwc
from momijo.vision.transforms.convert_color import rgb_to_gray

# --- Small compatibility shims for OpNode field vs. accessor style -----------
# Some IR versions expose fields (_kind/_oh/_ow); others use methods (kind()/oh()/ow()).
# The helpers below try methods first, then fallback to fields.

fn _node_kind(n: OpNode) -> OpKind:
    # Prefer method if available
    # NOTE: In Mojo, duck-typing like this is limited; we assume one of the two is present.
    return n._kind  # if your IR uses methods, change to: return n.kind()

fn _node_oh(n: OpNode) -> Int:
    return n._oh  # or: return n.oh()

fn _node_ow(n: OpNode) -> Int:
    return n._ow  # or: return n.ow()

# --- Core executor helpers ----------------------------------------------------

fn apply_node(cur: Tensor, n: OpNode) -> Tensor:
    var kind = _node_kind(n)

    if kind == OpKind.ResizeNearest:
        # Our resize uses (src, out_h, out_w) and preserves channels implicitly.
        return resize_nearest_u8_hwc(cur, _node_oh(n), _node_ow(n))

    if kind == OpKind.ConvertRGBToGray:
        return rgb_to_gray(cur)

    # Unknown op: pass-through (no-op). You may want to assert instead.
    return cur

# Execute a full graph with canonical fusion/simplification passes enabled.
fn run_graph(g: Graph, src: Tensor) -> Tensor:
    var cur = src
    # Optional IR optimizations
    fuse_resizes(g)
    simplify_convert_color(g)

    var i: Int = 0
    var n_nodes = g.node_count()
    while i < n_nodes:
        var node = g.node_at(i)
        cur = apply_node(cur, node)
        i += 1
    return cur

# Variant: run a graph *without* fusion/simplification (useful for debugging IR).
fn run_graph_no_fuse(g: Graph, src: Tensor) -> Tensor:
    var cur = src
    var i: Int = 0
    var n_nodes = g.node_count()
    while i < n_nodes:
        var node = g.node_at(i)
        cur = apply_node(cur, node)
        i += 1
    return cur

# Convenience: execute a single op kind with optional shape args.
# For ConvertRGBToGray, (oh/ow) are ignored.
fn run_single_op(kind: OpKind, src: Tensor, oh: Int = 0, ow: Int = 0) -> Tensor:
    if kind == OpKind.ResizeNearest:
        return resize_nearest_u8_hwc(src, oh, ow)
    if kind == OpKind.ConvertRGBToGray:
        return rgb_to_gray(src)
    # Unknown op -> pass-through
    return src
