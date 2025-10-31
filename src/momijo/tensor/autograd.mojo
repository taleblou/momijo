# MIT License
# SPDX-License-Identifier: MIT
# Project:      Momijo
# Module:       momijo.tensor.autograd
# File:         src/momijo/tensor/autograd.mojo
#
# Reverse-mode autograd (Float64) without globals/consts.
# - Stable permute fallback via moveaxis tracking
# - Axis-reduction backward respects keepdims and reshapes grad properly
# - Avoid *_like constructors (use zeros/full with explicit shapes)
# - Copy semantics handled to satisfy ImplicitlyCopyable rules
# - no_grad_begin/end + no_grad_do

from momijo.tensor import tensor
from collections.list import List 

@always_inline
fn TAG_LEAF() -> Int: return 0
@always_inline
fn TAG_ADD() -> Int: return 1
@always_inline
fn TAG_MUL() -> Int: return 2
@always_inline
fn TAG_ADDS() -> Int: return 3
@always_inline
fn TAG_MULS() -> Int: return 4
@always_inline
fn TAG_SUM_ALL() -> Int: return 5
@always_inline
fn TAG_MEAN_ALL() -> Int: return 6
@always_inline
fn TAG_RESHAPE() -> Int: return 7
@always_inline
fn TAG_PERMUTE() -> Int: return 8
@always_inline
fn TAG_SUM_AX() -> Int: return 9
@always_inline
fn TAG_MEAN_AX() -> Int: return 10

# Stable permute fallback using moveaxis and axis bookkeeping.
fn _permute_with_moveaxis(x: tensor.Tensor[Float64], order: List[Int]) -> tensor.Tensor[Float64]:
    var out = x.copy()
    var r = len(x._shape)
    var axes = List[Int]()
    var i = 0
    while i < r:
        axes.append(i); i = i + 1
    var k = 0
    while k < r:
        var target_axis = order[k]
        var pos = 0
        while pos < r:
            if axes[pos] == target_axis: break
            pos = pos + 1
        if pos != k:
            out = tensor.moveaxis(out, pos, k)
            var moved = axes[pos]
            var new_axes = List[Int]()
            var j = 0
            while j < len(axes):
                if j != pos: new_axes.append(axes[j])
                j = j + 1
            var ins = List[Int]()
            var u = 0
            var inserted = False
            while u < len(new_axes):
                if u == k and not inserted:
                    ins.append(moved); inserted = True
                ins.append(new_axes[u]); u = u + 1
            if not inserted: ins.append(moved)
            axes = ins.copy()
        k = k + 1
    return out.copy()

struct Tape:
    var values:  List[tensor.Tensor[Float64]]
    var grads:   List[tensor.Tensor[Float64]]
    var op_tags: List[Int]
    var parents: List[List[Int]]
    var scalars: List[Float64]
    var shapes:  List[List[Int]]
    var orders:  List[List[Int]]
    var axes:    List[Int]
    var keep:    List[Bool]

    fn __init__(out self):
        self.values  = List[tensor.Tensor[Float64]]()
        self.grads   = List[tensor.Tensor[Float64]]()
        self.op_tags = List[Int]()
        self.parents = List[List[Int]]()
        self.scalars = List[Float64]()
        self.shapes  = List[List[Int]]()
        self.orders  = List[List[Int]]()
        self.axes    = List[Int]()
        self.keep    = List[Bool]()

    fn _push_common(mut self,
                    val: tensor.Tensor[Float64],
                    op: Int, p0: Int, p1: Int,
                    s: Float64,
                    shp: List[Int],
                    ord: List[Int],
                    ax: Int, kd: Bool) -> Int:
        self.values.append(val.copy())
        self.grads.append(tensor.zeros(val._shape.copy()))
        self.op_tags.append(op)
        var ps = List[Int](); ps.append(p0); ps.append(p1)
        self.parents.append(ps.copy())
        self.scalars.append(s)
        self.shapes.append(shp.copy())
        self.orders.append(ord.copy())
        self.axes.append(ax)
        self.keep.append(kd)
        return len(self.values) - 1

    fn add_leaf(mut self, x: tensor.Tensor[Float64]) -> Int:
        var empty_i = List[Int]()
        return self._push_common(x, TAG_LEAF(), -1, -1, 0.0, empty_i, empty_i, -1, False)

    fn add_unary(mut self, val: tensor.Tensor[Float64], op: Int, p: Int,
                 s: Float64, shp: List[Int], ord: List[Int], ax: Int, kd: Bool) -> Int:
        return self._push_common(val, op, p, -1, s, shp, ord, ax, kd)

    fn add_binary(mut self, val: tensor.Tensor[Float64], op: Int, p0: Int, p1: Int) -> Int:
        var empty_i = List[Int]()
        return self._push_common(val, op, p0, p1, 0.0, empty_i, empty_i, -1, False)

struct GradContext:
    var tape: Tape
    var no_grad_depth: Int

    fn __init__(out self):
        self.tape = Tape()
        self.no_grad_depth = 0

    @always_inline
    fn grad_enabled(self) -> Bool:
        return self.no_grad_depth == 0

@always_inline
fn no_grad_begin(mut ctx: GradContext) -> None:
    ctx.no_grad_depth = ctx.no_grad_depth + 1

@always_inline
fn no_grad_end(mut ctx: GradContext) -> None:
    if ctx.no_grad_depth > 0:
        ctx.no_grad_depth = ctx.no_grad_depth - 1

fn no_grad_do(mut ctx: GradContext, f: fn() -> None) -> None:
    no_grad_begin(ctx); try: f() finally: no_grad_end(ctx)

struct GradTensor:
    var id: Int
    var requires_grad: Bool

    fn __init__(out self, id: Int, requires_grad: Bool):
        self.id = id; self.requires_grad = requires_grad

    @staticmethod
    fn from_tensor(mut ctx: GradContext, x: tensor.Tensor[Float64], requires_grad: Bool = False) -> GradTensor:
        var nid = ctx.tape.add_leaf(x)
        return GradTensor(nid, requires_grad and ctx.grad_enabled())

    @staticmethod
    fn from_randn(mut ctx: GradContext, shape: List[Int], requires_grad: Bool = False) -> GradTensor:
        var x = tensor.randn(shape)
        var nid = ctx.tape.add_leaf(x)
        return GradTensor(nid, requires_grad and ctx.grad_enabled())

    fn value(self, ctx: GradContext) -> tensor.Tensor[Float64]:
        return ctx.tape.values[self.id].copy()

    fn grad(self, ctx: GradContext) -> tensor.Tensor[Float64]:
        return ctx.tape.grads[self.id].copy()

    # Binary ops
    fn add(self, mut ctx: GradContext, other: GradTensor) -> GradTensor:
        var out = ctx.tape.values[self.id].add(ctx.tape.values[other.id])
        var track = (self.requires_grad or other.requires_grad) and ctx.grad_enabled()
        var nid = (ctx.tape.add_binary(out, TAG_ADD(), self.id, other.id) if track else ctx.tape.add_leaf(out))
        return GradTensor(nid, track)

    fn mul(self, mut ctx: GradContext, other: GradTensor) -> GradTensor:
        var out = ctx.tape.values[self.id].mul(ctx.tape.values[other.id])
        var track = (self.requires_grad or other.requires_grad) and ctx.grad_enabled()
        var nid = (ctx.tape.add_binary(out, TAG_MUL(), self.id, other.id) if track else ctx.tape.add_leaf(out))
        return GradTensor(nid, track)

    # Scalar ops
    fn add_scalar(self, mut ctx: GradContext, s: Float64) -> GradTensor:
        var out = ctx.tape.values[self.id].add_scalar(s)
        var track = self.requires_grad and ctx.grad_enabled()
        var nid = (ctx.tape.add_unary(out, TAG_ADDS(), self.id, s, List[Int](), List[Int](), -1, False)
                   if track else ctx.tape.add_leaf(out))
        return GradTensor(nid, track)

    fn mul_scalar(self, mut ctx: GradContext, s: Float64) -> GradTensor:
        var out = ctx.tape.values[self.id].mul_scalar(s)
        var track = self.requires_grad and ctx.grad_enabled()
        var nid = (ctx.tape.add_unary(out, TAG_MULS(), self.id, s, List[Int](), List[Int](), -1, False)
                   if track else ctx.tape.add_leaf(out))
        return GradTensor(nid, track)

    # Reductions (all elements)
    fn sum(self, mut ctx: GradContext) -> GradTensor:
        var out = ctx.tape.values[self.id].sum()
        var track = self.requires_grad and ctx.grad_enabled()
        var nid = (ctx.tape.add_unary(out, TAG_SUM_ALL(), self.id, 0.0, List[Int](), List[Int](), -1, False)
                   if track else ctx.tape.add_leaf(out))
        return GradTensor(nid, track)

    fn mean(self, mut ctx: GradContext) -> GradTensor:
        var out = ctx.tape.values[self.id].mean()
        var track = self.requires_grad and ctx.grad_enabled()
        var nid = (ctx.tape.add_unary(out, TAG_MEAN_ALL(), self.id, 0.0, List[Int](), List[Int](), -1, False)
                   if track else ctx.tape.add_leaf(out))
        return GradTensor(nid, track)

    # Reductions (axis-aware)
    fn sum_axis(self, mut ctx: GradContext, axis: Int, keepdims: Bool = False) -> GradTensor:
        var v = ctx.tape.values[self.id].copy()
        var out = tensor.sum(v, Optional[Int](axis), keepdims)
        var track = self.requires_grad and ctx.grad_enabled()
        var nid = (ctx.tape.add_unary(out, TAG_SUM_AX(), self.id, 0.0, v._shape.copy(), List[Int](), axis, keepdims)
                   if track else ctx.tape.add_leaf(out))
        return GradTensor(nid, track)

    fn mean_axis(self, mut ctx: GradContext, axis: Int, keepdims: Bool = False) -> GradTensor:
        var v = ctx.tape.values[self.id].copy()
        var out = tensor.mean(v, Optional[Int](axis), keepdims)
        var track = self.requires_grad and ctx.grad_enabled()
        var nid = (ctx.tape.add_unary(out, TAG_MEAN_AX(), self.id, 0.0, v._shape.copy(), List[Int](), axis, keepdims)
                   if track else ctx.tape.add_leaf(out))
        return GradTensor(nid, track)

    # Shape ops
    fn reshape(self, mut ctx: GradContext, new_shape: List[Int]) -> GradTensor:
        var v = ctx.tape.values[self.id].copy()
        var yv = tensor.reshape(v, new_shape)
        var track = self.requires_grad and ctx.grad_enabled()
        var nid = (ctx.tape.add_unary(yv, TAG_RESHAPE(), self.id, 0.0, v._shape.copy(), List[Int](), -1, False)
                   if track else ctx.tape.add_leaf(yv))
        return GradTensor(nid, track)

    fn permute(self, mut ctx: GradContext, order: List[Int]) -> GradTensor:
        var v = ctx.tape.values[self.id].copy()
        # Prefer native permute if available:
        # var yv = v.permute(order)
        var yv = _permute_with_moveaxis(v, order)
        var track = self.requires_grad and ctx.grad_enabled()
        var nid = (ctx.tape.add_unary(yv, TAG_PERMUTE(), self.id, 0.0, v._shape.copy(), order.copy(), -1, False)
                   if track else ctx.tape.add_leaf(yv))
        return GradTensor(nid, track)

    # Backward
    fn backward(self, mut ctx: GradContext) -> None:
        ctx.tape.grads[self.id] = tensor.zeros(ctx.tape.values[self.id]._shape.copy()).add_scalar(1.0)

        var stack = List[Int](); stack.append(self.id)

        while len(stack) > 0:
            var top_id = stack[len(stack) - 1]
            _ = stack.pop()

            var op = ctx.tape.op_tags[top_id]
            var g_out = ctx.tape.grads[top_id].copy()
            var p0 = ctx.tape.parents[top_id][0]
            var p1 = ctx.tape.parents[top_id][1]

            if op == TAG_LEAF():
                continue

            elif op == TAG_ADD():
                if p0 >= 0:
                    ctx.tape.grads[p0] = ctx.tape.grads[p0].add(g_out)
                    stack.append(p0)
                if p1 >= 0:
                    ctx.tape.grads[p1] = ctx.tape.grads[p1].add(g_out)
                    stack.append(p1)

            elif op == TAG_MUL():
                if p0 >= 0:
                    var gb = g_out.mul(ctx.tape.values[p1])
                    ctx.tape.grads[p0] = ctx.tape.grads[p0].add(gb)
                    stack.append(p0)
                if p1 >= 0:
                    var ga = g_out.mul(ctx.tape.values[p0])
                    ctx.tape.grads[p1] = ctx.tape.grads[p1].add(ga)
                    stack.append(p1)

            elif op == TAG_ADDS():
                if p0 >= 0:
                    ctx.tape.grads[p0] = ctx.tape.grads[p0].add(g_out)
                    stack.append(p0)

            elif op == TAG_MULS():
                if p0 >= 0:
                    var s = ctx.tape.scalars[top_id]
                    ctx.tape.grads[p0] = ctx.tape.grads[p0].add(g_out.mul_scalar(s))
                    stack.append(p0)

            elif op == TAG_SUM_ALL():
                if p0 >= 0:
                    var s = tensor.item(g_out)
                    var gx = tensor.full(ctx.tape.values[p0]._shape.copy(), s)
                    ctx.tape.grads[p0] = ctx.tape.grads[p0].add(gx)
                    stack.append(p0)

            elif op == TAG_MEAN_ALL():
                if p0 >= 0:
                    var n = tensor.numel(ctx.tape.values[p0]._shape)
                    var s = tensor.item(g_out) / Float64(n)
                    var gx = tensor.full(ctx.tape.values[p0]._shape.copy(), s)
                    ctx.tape.grads[p0] = ctx.tape.grads[p0].add(gx)
                    stack.append(p0)

            elif op == TAG_RESHAPE():
                if p0 >= 0:
                    var orig = ctx.tape.shapes[top_id].copy()
                    var gin = tensor.reshape(g_out, orig)
                    ctx.tape.grads[p0] = ctx.tape.grads[p0].add(gin)
                    stack.append(p0)

            elif op == TAG_PERMUTE():
                if p0 >= 0:
                    var ord = ctx.tape.orders[top_id].copy()
                    var r = len(ord)
                    var inv = List[Int](); var t = 0
                    while t < r: inv.append(0); t = t + 1
                    var k = 0
                    while k < r: inv[ord[k]] = k; k = k + 1
                    # Prefer native: var gin = g_out.permute(inv)
                    var gin = _permute_with_moveaxis(g_out, inv)
                    ctx.tape.grads[p0] = ctx.tape.grads[p0].add(gin)
                    stack.append(p0)

            elif op == TAG_SUM_AX():
                if p0 >= 0:
                    var shp = ctx.tape.shapes[top_id].copy()
                    var ax = ctx.tape.axes[top_id]
                    var kd = ctx.tape.keep[top_id]
                    var gout_adj = g_out.copy()
                    if not kd:
                        var out_shape = List[Int]()
                        var i = 0
                        while i < len(shp):
                            if i == ax: out_shape.append(1)
                            else: out_shape.append(shp[i])
                            i = i + 1
                        gout_adj = tensor.reshape(g_out, out_shape)
                    var g_in = tensor.zeros(ctx.tape.values[p0]._shape.copy()).add(gout_adj)
                    ctx.tape.grads[p0] = ctx.tape.grads[p0].add(g_in)
                    stack.append(p0)

            elif op == TAG_MEAN_AX():
                if p0 >= 0:
                    var shp = ctx.tape.shapes[top_id].copy()
                    var ax = ctx.tape.axes[top_id]
                    var kd = ctx.tape.keep[top_id]
                    var dim = 1
                    if ax >= 0 and ax < len(shp): dim = shp[ax]
                    var gout_scaled = g_out.mul_scalar(1.0 / Float64(dim))
                    var gout_adj2 = gout_scaled.copy()
                    if not kd:
                        var out_shape2 = List[Int]()
                        var i2 = 0
                        while i2 < len(shp):
                            if i2 == ax: out_shape2.append(1)
                            else: out_shape2.append(shp[i2])
                            i2 = i2 + 1
                        gout_adj2 = tensor.reshape(gout_scaled, out_shape2)
                    var g_in2 = tensor.zeros(ctx.tape.values[p0]._shape.copy()).add(gout_adj2)
                    ctx.tape.grads[p0] = ctx.tape.grads[p0].add(g_in2)
                    stack.append(p0)

        # Final pass: coerce grad shapes to value shapes
        var idx = 0
        while idx < len(ctx.tape.values):
            var vsh = ctx.tape.values[idx]._shape.copy()
            var gsh = ctx.tape.grads[idx]._shape.copy()
            var same = True
            if len(vsh) != len(gsh):
                same = False
            else:
                var t2 = 0
                while t2 < len(vsh):
                    if vsh[t2] != gsh[t2]:
                        same = False; break
                    t2 = t2 + 1
            if not same:
                ctx.tape.grads[idx] = tensor.zeros(vsh).add(ctx.tape.grads[idx])
            idx = idx + 1
