# Project:      Momijo
# Module:       src.momijo.vision.backend.registry
# File:         registry.mojo
# Path:         src/momijo/vision/backend/registry.mojo
#
# Description:  src.momijo.vision.backend.registry — focused Momijo functionality with a stable public API.
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
#   - Structs: OpKind, ImplState, OpSpec, Node, Registry
#   - Key functions: __init__, None_, ResizeNearest, RGBToGray, BGRToRGB, DropAlpha, FlipH, FlipV ...
#   - Static methods present.


struct OpKind(Copyable, Movable):
    var id: Int
fn __init__(out self, id: Int) -> None:
        self.id = id

    @staticmethod
fn None_() -> OpKind:
        return OpKind(0)

    @staticmethod
fn ResizeNearest() -> OpKind:
        return OpKind(1)

    @staticmethod
fn RGBToGray() -> OpKind:
        return OpKind(2)

    @staticmethod
fn BGRToRGB() -> OpKind:
        return OpKind(3)

    @staticmethod
fn DropAlpha() -> OpKind:
        return OpKind(4)

    @staticmethod
fn FlipH() -> OpKind:
        return OpKind(5)

    @staticmethod
fn FlipV() -> OpKind:
        return OpKind(6)

    # reserve: 7.. for future ops (ResizeBilinear, Brightness, Contrast, ...)
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
# ImplState tag (for documentation)
# -------------------------
struct ImplState(Copyable, Movable):
    var id: Int
fn __init__(out self, id: Int) -> None:
        self.id = id

    @staticmethod
fn Available() -> ImplState:
        return ImplState(1)

    @staticmethod
fn Planned() -> ImplState:
        return ImplState(2)
fn __eq__(self, other: ImplState) -> Bool:
        return self.id == other.id
fn to_string(self) -> String:
        if self.id == 1: return String("Available")
        if self.id == 2: return String("Planned")
        return String("UnknownState")

# -------------------------
# OpSpec + Node
# -------------------------
struct OpSpec(Copyable, Movable):
    var kind: OpKind
    var name: String
    var num_params: Int     # number of required integer params (0..2)
    var impl: ImplState
    var doc: String
fn __init__(out self, kind: OpKind, name: String, num_params: Int, impl: ImplState, doc: String) -> None:
        self.kind = kind
        self.name = name
        self.num_params = num_params
        self.impl = impl
        self.doc = doc

struct Node(Copyable, Movable):
    var kind: OpKind
    var p1: Int
    var p2: Int
fn __init__(out self, kind: OpKind, p1: Int, p2: Int) -> None:
        self.kind = kind
        self.p1 = p1
        self.p2 = p2

# -------------------------
# Registry
# -------------------------
struct Registry(Copyable, Movable):
    var ops: List[OpSpec]
fn __init__(out self) -> None:
        self.ops = List[OpSpec]()
        # Built-ins (aligned with current vision executor & image utilities)
        self = add_op(self, OpKind.ResizeNearest(), String("resize_nearest"), 2, ImplState.Available(),
                      String("Resize using nearest-neighbor. Params: out_h, out_w"))
        self = add_op(self, OpKind.RGBToGray(), String("rgb_to_gray"), 0, ImplState.Available(),
                      String("Convert RGB/BGR to single-channel grayscale (luma approx)."))
        self = add_op(self, OpKind.BGRToRGB(), String("bgr_to_rgb"), 0, ImplState.Available(),
                      String("Swap channels B<->R for 3-channel images."))
        self = add_op(self, OpKind.DropAlpha(), String("drop_alpha"), 0, ImplState.Available(),
                      String("Remove alpha channel from RGBA/BGRA/ARGB images."))
        self = add_op(self, OpKind.FlipH(), String("flip_h"), 0, ImplState.Available(),
                      String("Horizontal flip (mirror left-right)."))
        self = add_op(self, OpKind.FlipV(), String("flip_v"), 0, ImplState.Available(),
                      String("Vertical flip (mirror top-bottom)."))
fn add_op(mut self, kind: OpKind, name: String, num_params: Int, impl: ImplState, doc: String) -> Registry:
        var spec = OpSpec(kind, name, num_params, impl, doc)
        self.ops.append(spec)
        return self

# -------------------------
# Lookup helpers
# -------------------------
@staticmethod
fn _equals_ci(a: String, b: String) -> Bool:
    # case-insensitive compare via upper()
    return a.upper() == b.upper()

@staticmethod
fn find_spec(reg: Registry, name: String) -> (Bool, OpSpec):
    var i = 0
    while i < len(reg.ops):
        var sp = reg.ops[i]
        if _equals_ci(sp.name, name):
            return (True, sp)
        i += 1
    # not found: return a dummy spec
    return (False, OpSpec(OpKind.None_(), String(""), 0, ImplState.Planned(), String("")))

@staticmethod
fn has_op(reg: Registry, name: String) -> Bool:
    var (ok, _) = find_spec(reg, name)
    return ok

@staticmethod
fn get_kind(reg: Registry, name: String) -> OpKind:
    var (ok, sp) = find_spec(reg, name)
    if ok: return sp.kind
    return OpKind.None_()

@staticmethod
fn get_num_params(reg: Registry, name: String) -> Int:
    var (ok, sp) = find_spec(reg, name)
    if ok: return sp.num_params
    return 0

@staticmethod
fn make_node(reg: Registry, name: String, p1: Int, p2: Int) -> Node:
    var (ok, sp) = find_spec(reg, name)
    if not ok:
        return Node(OpKind.None_(), 0, 0)
    # basic arity check (excess params kept but ignored by executors that don't use them)
    if sp.num_params == 0:
        return Node(sp.kind, 0, 0)
    if sp.num_params == 1:
        return Node(sp.kind, p1, 0)
    return Node(sp.kind, p1, p2)

# -------------------------
# Listing & docs
# -------------------------
@staticmethod
fn list_ops(reg: Registry) -> List[String]:
    var out: List[String] = List[String]()
    var i = 0
    while i < len(reg.ops):
        out.append(reg.ops[i].name)
        i += 1
    return out

@staticmethod
fn describe(reg: Registry, name: String) -> String:
    var (ok, sp) = find_spec(reg, name)
    if not ok:
        return String("(not found)")
    var s = sp.name + String(": kind=") + sp.kind.to_string() + String(", params=") + String(sp.num_params) + String(", impl=") + sp.impl.to_string() + String("\n  ") + sp.doc
    return s

# -------------------------
# Minimal smoke test
# -------------------------
@staticmethod
fn __self_test__() -> Bool:
    var r = Registry()
    # Lookup
    if not has_op(r, String("rgb_to_gray")): return False
    if get_kind(r, String("flip_h")).id != OpKind.FlipH().id: return False
    if get_num_params(r, String("resize_nearest")) != 2: return False

    # Node building
    var n0 = make_node(r, String("rgb_to_gray"), 123, 456)
    if not (n0.kind == OpKind.RGBToGray() and n0.p1 == 0 and n0.p2 == 0): return False
    var n1 = make_node(r, String("resize_nearest"), 480, 640)
    if not (n1.kind == OpKind.ResizeNearest() and n1.p1 == 480 and n1.p2 == 640): return False
    var n2 = make_node(r, String("unknown"), 1, 2)
    if not (n2.kind == OpKind.None_()): return False

    # List & describe sanity
    var names = list_ops(r)
    if len(names) < 3: return False
    var desc = describe(r, String("bgr_to_rgb"))
    if len(desc) == 0: return False
    return True