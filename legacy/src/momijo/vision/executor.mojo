# MIT License
# SPDX-License-Identifier: MIT
# Project: momijo.vision
# File: src/momijo/vision/executor.mojo
# Description: Lightweight IR executor with optional fusion passes.

# --- IR & transforms imports -------------------------------------------------
from momijo.vision.ir.ir import Graph, OpKind, OpNode
from momijo.vision.ir.fusion import fuse_resizes, simplify_convert_color

from momijo.vision.tensor import Tensor
from momijo.vision.transforms.resize import resize_nearest_u8_hwc
from momijo.vision.transforms.convert_color import rgb_to_gray

# --- Private helpers ---------------------------------------------------------
# Some IR variants expose methods (kind()/oh()/ow()), others expose fields (_kind/_oh/_ow).
# The shim below tries methods first (if compiled in that IR), otherwise falls back to fields.

@always_inline
fn _node_kind(n: OpNode) -> OpKind:
    # Prefer accessor if available in your IR build.
    # Uncomment the next line if OpNode.kind() exists in your IR:
    # return n.kind()
    # Fallback: field (common in simpler IR structs)
    return n._kind

@always_inline
fn _node_oh(n: OpNode) -> Int:
    # return n.oh()    # if available
    return n._oh

@always_inline
fn _node_ow(n: OpNode) -> Int:
    # return n.ow()    # if available
    return n._ow

# if the tensor is packed HWC/UInt8. If already in desired form:
# - when copy_if_needed=True, return a *clone* (defensive copy)
# - when copy_if_needed=False, return a shallow copy of the same storage
@always_inline
fn _ensure_packed_hwc_u8(t: Tensor, copy_if_needed: Bool = False) -> Tensor:
    if t.is_contiguous_hwc_u8():
        # Defensive strategy: for ops that mutate, call with copy_if_needed=True
        return copy_if_needed ? t.clone() : t.copy()
    return t.copy_to_packed_hwc()

# Dispatch a single node on the current tensor.
# Pass-through for unknown ops.
fn _apply_node(cur: Tensor, n: OpNode) -> Tensor:
    var kind = _node_kind(n)

    # --- ResizeNearest (HWC/UInt8 only) ---
    if kind == OpKind.ResizeNearest:
        # if input layout/dtype expectations
        var src = _ensure_packed_hwc_u8(cur, False)
        var oh = _node_oh(n)
        var ow = _node_ow(n)
        # resize preserves channels
        return resize_nearest_u8_hwc(src, oh, ow)

    # --- ConvertRGBToGray (expects 3 channels, HWC/UInt8) ---
    if kind == OpKind.ConvertRGBToGray:
        var src = _ensure_packed_hwc_u8(cur, False)
        if src.channels() == 3:
            return rgb_to_gray(src)
        # If channels are not 3, pass-through (fail-soft)
        return cur.copy()

    # Unknown op: pass-through (no-op)
    return cur.copy()

# --- Public API --------------------------------------------------------------

# Execute a full graph with canonical fusion/simplification passes.
fn run_graph(g: Graph, src: Tensor) -> Tensor:
    # Apply optional IR optimizations (no-throw; mutate g in place)
    fuse_resizes(g)
    simplify_convert_color(g)

    var cur = src.copy()
    var i = 0
    # Allow dynamic graphs but the passes above finalize structure, so a fixed count is fine
    var n_nodes = g.node_count()
    while i < n_nodes:
        var node = g.node_at(i)
        cur = _apply_node(cur, node)
        i += 1
    return cur

# Variant: run a graph without any fusion/simplification (useful for IR debugging).
fn run_graph_no_fuse(g: Graph, src: Tensor) -> Tensor:
    var cur = src.copy()
    var i = 0
    var n_nodes = g.node_count()
    while i < n_nodes:
        var node = g.node_at(i)
        cur = _apply_node(cur, node)
        i += 1
    return cur

# Convenience: execute a single op kind with optional shape args.
# For ConvertRGBToGray, (oh/ow) are ignored.
fn run_single_op(kind: OpKind, src: Tensor, oh: Int = 0, ow: Int = 0) -> Tensor:
    if kind == OpKind.ResizeNearest:
        var inp = _ensure_packed_hwc_u8(src, False)
        return resize_nearest_u8_hwc(inp, oh, ow)
    if kind == OpKind.ConvertRGBToGray:
        var inp = _ensure_packed_hwc_u8(src, False)
        if inp.channels() == 3:
            return rgb_to_gray(inp)
        return src.copy()
    # Unknown op → pass-through
    return src.copy()
