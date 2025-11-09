# MIT License
# SPDX-License-Identifier: MIT
# Project:      Momijo
# Module:       momijo.tensor.autograd
# File:         src/momijo/tensor/autograd.mojo
#
# Description:
#   This file follows Momijo coding standards: var-only, no globals, explicit imports,
#   English-only comments, and correct constructor signatures. See CONTRIBUTING.md.



from momijo.tensor import tensor
from collections.list import List

# ---- Shared autograd utilities ----
@always_inline
fn needs_grad(flag_a: Bool, flag_b: Bool = False) -> Bool:
    # Returns True if any input requires gradients.
    return flag_a or flag_b

# Broadcasting helper: reshape/expand 'x' gradient to 'shape' and sum-reduce where needed.
fn broadcast_to_shape(x: tensor.Tensor[Float32], shape: List[Int]) -> tensor.Tensor[Float32]:
    var sx = x.shape()
    if sx == shape:
        return x.copy()
    # If x is scalar, expand to target
    if len(sx) == 0:
        return x.reshape(shape)
    # Right-align shapes and sum-reduce axes of length-1 where target has larger dims
    var dx = len(sx)
    var dy = len(shape)
    var out = x.copy()
    if dx < dy:
        # prepend ones then reshape
        var newshape = List[Int]()
        var pad = dy - dx
        var i = 0
        while i < pad:
            newshape.append(1)
            i = i + 1
        var j = 0
        while j < dx:
            newshape.append(sx[j])
            j = j + 1
        out = out.reshape(newshape)
        sx = out.shape()
        dx = len(sx)
    # sum-reduce on axes where needed
    var k = 0
    while k < dy:
        if sx[k] != shape[k]:
            # reduce this axis
            out = out.sum(axis=k)
        k = k + 1
    # finally reshape to target
    return out.reshape(shape)

# Shape helper: returns input's shape as a new list (defensive copy)
@always_inline
fn shape_like(x: tensor.Tensor[Float32]) -> List[Int]:
    var s = x.shape()
    var out = List[Int]()
    var i = 0
    while i < len(s):
        out.append(s[i])
        i = i + 1
    return out

# No-grad helper: detach a tensor value (stops gradient flow)
fn stop_gradient(x: GradTensor) -> GradTensor:
    var y = x.copy()
    y._requires_grad = False
    y._node_id = -1
    y._parents = List[Int]()
    return y


# ---------- Stage 10: value-only reduce max over an axis (no grad needed) ----------
fn _value_max_axis(x: tensor.Tensor[Float32], axis: Int, keepdims: Bool) -> tensor.Tensor[Float32]:
    var n = x._shape[axis]
    var acc = tensor.index_axis(x, axis, 0)
    var i = 1
    while i < n:
        var sl = tensor.index_axis(x, axis, i)
        var mask = sl.gt(acc)
        acc = tensor.where(mask, sl, acc)
        i = i + 1
    if keepdims:
        acc = tensor.unsqueeze(acc, axis)
    return acc

# ----------------------------- Op tags ----------------------------------------
@always_inline
fn TAG_LEAF()         -> Int: return 0
@always_inline
fn TAG_ADD()          -> Int: return 1
@always_inline
fn TAG_MUL()          -> Int: return 2
@always_inline
fn TAG_ADDS()         -> Int: return 3
@always_inline
fn TAG_MULS()         -> Int: return 4
@always_inline
fn TAG_SUM_ALL()      -> Int: return 5
@always_inline
fn TAG_MEAN_ALL()     -> Int: return 6
@always_inline
fn TAG_RESHAPE()      -> Int: return 7
@always_inline
fn TAG_PERMUTE()      -> Int: return 8
@always_inline
fn TAG_SUM_AX()       -> Int: return 9
@always_inline
fn TAG_MEAN_AX()      -> Int: return 10
@always_inline
fn TAG_NEG()          -> Int: return 11
@always_inline
fn TAG_SUB()          -> Int: return 12
@always_inline
fn TAG_DIVS()         -> Int: return 13     # divide by scalar
@always_inline
fn TAG_SQUARE()       -> Int: return 14
@always_inline
fn TAG_POWS()         -> Int: return 15     # power by scalar
@always_inline
fn TAG_TRANSPOSE2D()  -> Int: return 16
@always_inline
fn TAG_FLATTEN()      -> Int: return 17
@always_inline
fn TAG_MATMUL2D()     -> Int: return 18     # 2D matrix multiply
@always_inline
fn TAG_TENSORDOT()   -> Int: return 51     # tensordot(axis)
@always_inline
fn TAG_MAXS()         -> Int: return 19     # maximum(x, c)
@always_inline
fn TAG_MINS()         -> Int: return 20     # minimum(x, c)
@always_inline
fn TAG_RELU()         -> Int: return 21     # relu(x) = max(x,0)
@always_inline
fn TAG_LRELU()        -> Int: return 22
@always_inline
fn TAG_GATHER()      -> Int: return 24     # gather(axis, index)
@always_inline
fn TAG_INDEX_AXIS()  -> Int: return 25     # index_axis(axis, idx)
@always_inline
fn TAG_TAKE()        -> Int: return 26     # take(js, axis)
@always_inline
fn TAG_SCATTER()     -> Int: return 27     # scatter(axis, index, src)
@always_inline
fn TAG_MASKED_SELECT() -> Int: return 28   # masked_select(mask)
@always_inline
fn TAG_MASKED_SELECT_ELEM() -> Int: return 29   # masked_select elementwise (same-shape)
@always_inline
fn TAG_WHERE()       -> Int: return 90     # where(mask, a, b)
@always_inline
fn TAG_SQRT()        -> Int: return 60     # sqrt(x)
@always_inline
fn TAG_EXP()        -> Int: return 61     # exp(x)
@always_inline
fn TAG_LOG()        -> Int: return 62     # log(x)
@always_inline
fn TAG_TANH()        -> Int: return 63     # tanh(x)
@always_inline
fn TAG_SIGMOID()        -> Int: return 64     # sigmoid(x)
@always_inline
fn TAG_ABS()        -> Int: return 65     # abs(x)
@always_inline
fn TAG_SIGN()        -> Int: return 66     # sign(x) (zero-grad)
@always_inline
fn TAG_FLOOR()        -> Int: return 67     # floor(x) (zero-grad)
@always_inline
fn TAG_CEIL()        -> Int: return 68     # ceil(x) (zero-grad)
@always_inline
fn TAG_ROUND()        -> Int: return 69     # round(x) (zero-grad)
@always_inline
fn TAG_LOG1P()        -> Int: return 70     # log1p(x)
@always_inline
fn TAG_EXPM1()        -> Int: return 71     # expm1(x)
@always_inline
fn TAG_STD()         -> Int: return 80     # std over axis or all dims (biased, eps)
@always_inline
fn TAG_CLAMP()        -> Int: return 72     # clamp(x, min, max)
@always_inline
fn TAG_REPEAT()      -> Int: return 23     # repeat(reps)
@always_inline
fn TAG_DIV()      -> Int: return 30     #
@always_inline
fn TAG_CLIP() -> Int: return 31
@always_inline
fn TAG_CLAMP_DISABLED() -> Int: return 32
@always_inline
fn TAG_CLAMP_DISA() -> Int: return TAG_CLIP()
# leaky_relu(x,a)

# ----------------------------- Small helpers ---------------------------------
@always_inline
fn _apply_permute(x: tensor.Tensor[Float32], order: List[Int]) -> tensor.Tensor[Float32]:
    return tensor.permute(x, order)

@always_inline
fn _invert_perm(order: List[Int]) -> List[Int]:
    var r = len(order)
    var inv = List[Int]()
    var i = 0
    while i < r: inv.append(0); i = i + 1
    i = 0
    while i < r:
        inv[order[i]] = i
        i = i + 1
    return inv.copy()

# ----------------------------- Tape ------------------------------------------
struct Tape:
    var values:  List[tensor.Tensor[Float32]]
    var grads:   List[tensor.Tensor[Float32]]
    var op_tags: List[Int]
    var parents: List[List[Int]]
    var scalars: List[Float32]      # single scalar parameter (e.g., muls/divs/pows/maxs/mins/leaky slope)
    var shapes:  List[List[Int]]    # original shapes for reshape/reductions/flatten
    var orders:  List[List[Int]]    # for permute/transpose2d
    var axes:    List[Int]          # for axis-reductions
    var keep:    List[Bool]         # keepdims flag for reductions

    fn __init__(out self):
        self.values  = List[tensor.Tensor[Float32]]()
        self.grads   = List[tensor.Tensor[Float32]]()
        self.op_tags = List[Int]()
        self.parents = List[List[Int]]()
        self.scalars = List[Float32]()
        self.shapes  = List[List[Int]]()
        self.orders  = List[List[Int]]()
        self.axes    = List[Int]()
        self.keep    = List[Bool]()

    fn _push_common(
        mut self,
        val: tensor.Tensor[Float32],
        op: Int, p0: Int, p1: Int,
        s: Float32,
        shp: List[Int],
        ord: List[Int],
        ax: Int, kd: Bool
    ) -> Int:
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

    fn add_leaf(mut self, x: tensor.Tensor[Float32]) -> Int:
        var empty_i = List[Int]()
        return self._push_common(x, TAG_LEAF(), -1, -1, 0.0, empty_i, empty_i, -1, False)

    fn add_unary(
        mut self,
        val: tensor.Tensor[Float32],
        op: Int, p: Int,
        s: Float32, shp: List[Int], ord: List[Int],
        ax: Int, kd: Bool
    ) -> Int:
        return self._push_common(val, op, p, -1, s, shp, ord, ax, kd)

    fn add_binary(mut self, val: tensor.Tensor[Float32], op: Int, p0: Int, p1: Int) -> Int:
        var empty_i = List[Int]()
        return self._push_common(val, op, p0, p1, 0.0, empty_i, empty_i, -1, False)

# ----------------------------- Grad context ----------------------------------
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
    no_grad_begin(ctx)
    try: f()
    finally: no_grad_end(ctx)

# ----------------------------- GradTensor ------------------------------------
struct GradTensor(Copyable, Movable):
    var id: Int
    var requires_grad: Bool

    fn __init__(out self, id: Int, requires_grad: Bool):
        self.id = id
        self.requires_grad = requires_grad

    fn __copyinit__(out self, other: Self):
        self.id = other.id
        self.requires_grad = other.requires_grad

    @staticmethod
    fn from_tensor(mut ctx: GradContext, x: tensor.Tensor[Float32], requires_grad: Bool = False) -> GradTensor:
        var nid = ctx.tape.add_leaf(x)
        return GradTensor(nid, requires_grad and ctx.grad_enabled())

    @staticmethod
    fn from_randn(mut ctx: GradContext, shape: List[Int], requires_grad: Bool = False) -> GradTensor:
        var x = tensor.randn(shape)
        var nid = ctx.tape.add_leaf(x)
        return GradTensor(nid, requires_grad and ctx.grad_enabled())

    # Access current tensor value from the active context
    fn value(self, mut ctx: GradContext) -> tensor.Tensor[Float32]:
        return ctx.tape.values[self.id].copy()
    # Shape as a plain List[Int]
    fn shape(self, mut ctx: GradContext) -> List[Int]:
        return ctx.tape.values[self.id]._shape.copy()

    # Number of elements (product of shape)
    fn numel(self, mut ctx: GradContext) -> Int:
        var s = ctx.tape.values[self.id]._shape
        var n = 1
        var i = 0
        while i < len(s):
            n = n * s[i]
            i = i + 1
        return n

    # Read a scalar value; safe only when numel()==1
    fn item(self, mut ctx: GradContext) -> Float32:
        var v = ctx.tape.values[self.id].copy()
        # Optional safety check (comment out if asserts are undesired)
        # assert(len(v._data) == 1)
        return v._data[0]

    # Convert to flat List[Float32] (read-only snapshot)
    fn to_list(self, mut ctx: GradContext) -> List[Float32]:
        var out = List[Float32]()
        var v = ctx.tape.values[self.id].copy()
        var n = len(v._data)
        var i = 0
        while i < n:
            out.append(v._data[i])
            i = i + 1
        return out.copy()

    # Convenience: get an element by flat index (read-only)
    fn at_flat(self, mut ctx: GradContext, i: Int) -> Float32:
        var v = ctx.tape.values[self.id]
        # Optional bound check:
        # assert(i >= 0 and i < len(v._data))
        return v._data[i]

    fn grad(self, ctx: GradContext) -> tensor.Tensor[Float32]:
        return ctx.tape.grads[self.id].copy()

    # ---------- Binary elementwise ----------
    fn add(self, mut ctx: GradContext, other: GradTensor) -> GradTensor:
        var out = ctx.tape.values[self.id].add(ctx.tape.values[other.id])
        var track = (self.requires_grad or other.requires_grad) and ctx.grad_enabled()
        var nid = (ctx.tape.add_binary(out, TAG_ADD(), self.id, other.id)
                   if track else ctx.tape.add_leaf(out))
        return GradTensor(nid, track)

    fn mul(self, mut ctx: GradContext, other: GradTensor) -> GradTensor:
        var out = ctx.tape.values[self.id].mul(ctx.tape.values[other.id])
        var track = (self.requires_grad or other.requires_grad) and ctx.grad_enabled()
        var nid = (ctx.tape.add_binary(out, TAG_MUL(), self.id, other.id)
                   if track else ctx.tape.add_leaf(out))
        return GradTensor(nid, track)

    fn sub(self, mut ctx: GradContext, other: GradTensor) -> GradTensor:
        var neg_y = other.mul_scalar(ctx, -1.0)
        return self.add(ctx, neg_y)

    fn add(self, mut ctx: GradContext, other: tensor.Tensor[Float32]) -> GradTensor:
        var a   = ctx.tape.values[self.id].copy()                 # Tensor[Float32]
        var out = tensor.add_t(a, other)                     # elementwise add
        var track = self.requires_grad and ctx.grad_enabled()
        if not track:
            var lid = ctx.tape.add_leaf(out)
            return GradTensor(lid, False)
        var oid = ctx.tape.add_leaf(other)                 # stash RHS value (no grad)
        var nid = ctx.tape.add_binary(out, TAG_ADD(), self.id, oid)
        return GradTensor(nid, True)

    fn mul(self, mut ctx: GradContext, other: tensor.Tensor[Float32]) -> GradTensor:
        var a   = ctx.tape.values[self.id].copy()
        var out = tensor.mul_t(a, other)                     # elementwise mul
        var track = self.requires_grad and ctx.grad_enabled()
        if not track:
            var lid = ctx.tape.add_leaf(out)
            return GradTensor(lid, False)
        var oid = ctx.tape.add_leaf(other)
        var nid = ctx.tape.add_binary(out, TAG_MUL(), self.id, oid)
        return GradTensor(nid, True)

    fn sub(self, mut ctx: GradContext, other: tensor.Tensor[Float32]) -> GradTensor:
        var a   = ctx.tape.values[self.id].copy()
        # out = a - other  == a + (-other)
        var neg_other = other.mul_scalar(-1.0)
        var out = a.add(neg_other)
        var track = self.requires_grad and ctx.grad_enabled()
        if not track:
            var lid = ctx.tape.add_leaf(out)
            return GradTensor(lid, False)
        var oid = ctx.tape.add_leaf(other)                 # stash original RHS (optional: stash neg_other instead)
        var nid = ctx.tape.add_binary(out, TAG_SUB(), self.id, oid)   # if you don't have TAG_SUB(), use TAG_ADD()
        return GradTensor(nid, True)
    # ---------- Scalar elementwise ----------
    fn add_scalar(self, mut ctx: GradContext, s: Float32) -> GradTensor:
        var out = ctx.tape.values[self.id].add_scalar(s)
        var track = self.requires_grad and ctx.grad_enabled()
        var nid = (ctx.tape.add_unary(out, TAG_ADDS(), self.id, s, List[Int](), List[Int](), -1, False)
                   if track else ctx.tape.add_leaf(out))
        return GradTensor(nid, track)

    fn add_scalar(self, mut ctx: GradContext, s: GradTensor) -> GradTensor:
        var a = ctx.tape.values[self.id] .copy()
        var b = ctx.tape.values[s.id] .copy()
        var out = a.add( b)
        var track = (self.requires_grad or s.requires_grad) and ctx.grad_enabled()
        if not track:
            var lid = ctx.tape.add_leaf(out)
            return GradTensor(lid, False)
        var nid = ctx.tape.add_binary(out, TAG_ADD(), self.id, s.id)
        return GradTensor(nid, True)

    fn mul_scalar(self, mut ctx: GradContext, s: Float32) -> GradTensor:
        var out = ctx.tape.values[self.id].mul_scalar(s)
        var track = self.requires_grad and ctx.grad_enabled()
        var nid = (ctx.tape.add_unary(out, TAG_MULS(), self.id, s, List[Int](), List[Int](), -1, False)
                   if track else ctx.tape.add_leaf(out))
        return GradTensor(nid, track)

    fn mul_scalar(self, mut ctx: GradContext, s: GradTensor) -> GradTensor:
        var a = ctx.tape.values[self.id].copy()
        var b = ctx.tape.values[s.id].copy()
        var out = a.mul(b)
        var track = (self.requires_grad or s.requires_grad) and ctx.grad_enabled()
        if not track:
            var lid = ctx.tape.add_leaf(out)
            return GradTensor(lid, False)
        var nid = ctx.tape.add_binary(out, TAG_MUL(), self.id, s.id)
        return GradTensor(nid, True)

    fn div_scalar(self, mut ctx: GradContext, s: Float32) -> GradTensor:
        var inv = 1.0 / s
        var out = ctx.tape.values[self.id].mul_scalar(inv)
        var track = self.requires_grad and ctx.grad_enabled()
        var nid = (ctx.tape.add_unary(out, TAG_DIVS(), self.id, s, List[Int](), List[Int](), -1, False)
                   if track else ctx.tape.add_leaf(out))
        return GradTensor(nid, track)

    fn div_scalar(self, mut ctx: GradContext, s: GradTensor) -> GradTensor:
        var a = ctx.tape.values[self.id].copy()
        var b = ctx.tape.values[s.id].copy()

        # a / b
        var out = a.mul( tensor.reciprocal(b))

        # var out = tensor.mul(a, tensor.pow_scalar(b, -1.0))

        var track = (self.requires_grad or s.requires_grad) and ctx.grad_enabled()
        if not track:
            var lid = ctx.tape.add_leaf(out)
            return GradTensor(lid, False)

        var nid = ctx.tape.add_binary(out, TAG_DIV(), self.id, s.id)
        return GradTensor(nid, True)


    fn add_scalar(self, mut ctx: GradContext, s: tensor.Tensor[Float32]) -> GradTensor:
        var a   = ctx.tape.values[self.id] .copy()
        var out = a.add( s)
        var track = self.requires_grad and ctx.grad_enabled()
        if not track:
            var lid = ctx.tape.add_leaf(out)
            return GradTensor(lid, False)
        var sid = ctx.tape.add_leaf(s)                         # stash RHS (no grad)
        var nid = ctx.tape.add_binary(out, TAG_ADD(), self.id, sid)
        return GradTensor(nid, True)

    fn mul_scalar(self, mut ctx: GradContext, s: tensor.Tensor[Float32]) -> GradTensor:
        var a   = ctx.tape.values[self.id] .copy()
        var out = a.mul( s)
        var track = self.requires_grad and ctx.grad_enabled()
        if not track:
            var lid = ctx.tape.add_leaf(out)
            return GradTensor(lid, False)
        var sid = ctx.tape.add_leaf(s)
        var nid = ctx.tape.add_binary(out, TAG_MUL(), self.id, sid)
        return GradTensor(nid, True)

    fn div_scalar(self, mut ctx: GradContext, s: tensor.Tensor[Float32]) -> GradTensor:
        var a   = ctx.tape.values[self.id].copy()
        var out = a.mul(tensor.reciprocal(s))   # a / s

        var track = self.requires_grad and ctx.grad_enabled()
        if not track:
            var lid = ctx.tape.add_leaf(out)
            return GradTensor(lid, False)

        var sid = ctx.tape.add_leaf(s)
        var nid = ctx.tape.add_binary(out, TAG_DIV(), self.id, sid)
        return GradTensor(nid, True)


    fn neg(self, mut ctx: GradContext) -> GradTensor:
        var out = ctx.tape.values[self.id].mul_scalar(-1.0)
        var track = self.requires_grad and ctx.grad_enabled()
        var nid = (ctx.tape.add_unary(out, TAG_NEG(), self.id, -1.0, List[Int](), List[Int](), -1, False)
                   if track else ctx.tape.add_leaf(out))
        return GradTensor(nid, track)

    fn square(self, mut ctx: GradContext) -> GradTensor:
        var v = ctx.tape.values[self.id].copy()
        var out = v.mul(v)
        var track = self.requires_grad and ctx.grad_enabled()
        var nid = (ctx.tape.add_unary(out, TAG_SQUARE(), self.id, 0.0, List[Int](), List[Int](), -1, False)
                   if track else ctx.tape.add_leaf(out))
        return GradTensor(nid, track)

    fn pow_scalar(self, mut ctx: GradContext, p: Float32) -> GradTensor:
        if p == 2.0:
            return self.square(ctx)
        elif p == 1.0:
            return GradTensor(self.id, self.requires_grad)
        elif p == 0.0:
            var one = tensor.zeros(ctx.tape.values[self.id]._shape.copy()).add_scalar(1.0)
            var as_leaf = ctx.tape.add_leaf(one)
            return GradTensor(as_leaf, False)
        else:
            # Best-effort integer p>2 by iterative mul to build forward, then tag as POWS
            var ip = Int(p)
            if Float32(ip) == p and ip > 2:
                var out_g = GradTensor(self.id, self.requires_grad)
                var k = 1
                while k < ip:
                    out_g = out_g.mul(ctx, GradTensor(self.id, self.requires_grad))
                    k = k + 1
                var out_t = ctx.tape.values[out_g.id]
                var nid = (ctx.tape.add_unary(out_t, TAG_POWS(), self.id, p, List[Int](), List[Int](), -1, False)
                           if (self.requires_grad and ctx.grad_enabled()) else ctx.tape.add_leaf(out_t))
                return GradTensor(nid, self.requires_grad and ctx.grad_enabled())
            else:
                # Fallback to square if exp/log not in core.
                return self.square(ctx)


    fn pow_scalar(self, mut ctx: GradContext, p: GradTensor) -> GradTensor:
        # read scalar value from p
        var pt = ctx.tape.values[p.id].copy()
        var pv = (pt._data[0] if len(pt._data) > 0 else 1.0)   # default to 1.0 if somehow empty

        # Fast paths
        if pv == 2.0:
            return self.square(ctx)
        elif pv == 1.0:
            # identity (preserve requires_grad)
            return GradTensor(self.id, self.requires_grad)
        elif pv == 0.0:
            var one = tensor.zeros(ctx.tape.values[self.id]._shape.copy()).add_scalar(1.0)
            var lid = ctx.tape.add_leaf(one)
            return GradTensor(lid, False)

        # Integer exponent > 2 → repeated multiply to build the graph
        var ip = Int(pv)
        if Float32(ip) == pv and ip > 2:
            var out_g = GradTensor(self.id, self.requires_grad)
            var k = 1
            while k < ip:
                out_g = out_g.mul(ctx, GradTensor(self.id, self.requires_grad))
                k = k + 1
            # Tag as POWS for bookkeeping (optional)
            var out_t = ctx.tape.values[out_g.id]
            var track = self.requires_grad and ctx.grad_enabled()
            var nid = (ctx.tape.add_unary(out_t, TAG_POWS(), self.id, pv, List[Int](), List[Int](), -1, False)
                    if track else ctx.tape.add_leaf(out_t))
            return GradTensor(nid, track)

        # General scalar exponent (non-integer): use exp(p * log(x)), requires x > 0
        # Build p as a constant GradTensor scalar for mul
        var pconst = GradTensor.from_tensor(ctx, tensor.zeros(List[Int]([1])).add_scalar(pv), False)
        # y = exp(p * log(x))
        # NOTE: log requires positive inputs; if your data may be non-positive, guard/clip before.
        var y = self.log(ctx).mul(ctx, pconst).exp(ctx)
        return y

    fn pow_scalar(self, mut ctx: GradContext, p: tensor.Tensor[Float32]) -> GradTensor:
        # read scalar value
        var pv = 1.0
        if len(p._data) > 0:
            pv = p._data[0]

        # Fast paths (no need to create extra nodes)
        if pv == 2.0:
            return self.square(ctx)
        if pv == 1.0:
            return GradTensor(self.id, self.requires_grad)
        if pv == 0.0:
            var one = tensor.zeros(ctx.tape.values[self.id]._shape.copy()).add_scalar(1.0)
            var lid = ctx.tape.add_leaf(one)
            return GradTensor(lid, False)

        # Integer exponent > 2 → build graph by repeated multiply
        var ip = Int(pv)
        if Float32(ip) == pv and ip > 2:
            var out_g = GradTensor(self.id, self.requires_grad)
            var k = 1
            while k < ip:
                out_g = out_g.mul(ctx, GradTensor(self.id, self.requires_grad))
                k = k + 1
            return out_g

        # General exponent (non-integer): y = exp( pv * log(clamp(x)) )
        # Guard domain of log (x > 0)
        var x_safe = self.clip(ctx, 1e-12, 1.0e300)
        # multiply by constant scalar provided as Tensor[Float32]
        var pconst = tensor.zeros(List[Int]([1])).add_scalar(pv)     # scalar tensor
        var y = x_safe.log(ctx).mul_scalar(ctx, pconst).exp(ctx)
        return y
    # ---------- Reductions ----------

    # sum_all: reduce over all elements → scalar (Float32)
    fn sum_all(self, mut ctx: GradContext) -> GradTensor:
        var x = ctx.tape.values[self.id].copy()
        var y = tensor.sum(x)  # all-reduce to scalar
        var track = self.requires_grad and ctx.grad_enabled()
        if not track:
            var lid = ctx.tape.add_leaf(y)
            return GradTensor(lid, False)
        var nid = ctx.tape.add_unary(
            y, TAG_SUM_ALL(), self.id,
            0.0, x._shape.copy(), List[Int](), -1, False
        )
        return GradTensor(nid, True)

    # mean_all: mean over all elements = sum_all / numel
    fn mean_all(self, mut ctx: GradContext) -> GradTensor:
        var s = self.sum_all(ctx)
        # compute numel from input shape
        var v = ctx.tape.values[self.id]
        var n = 1
        var i = 0
        while i < len(v._shape):
            n = n * v._shape[i]
            i = i + 1
        var nf = Float32(n)
        var invn_t = tensor.zeros(List[Int]([1])).add_scalar(1.0 / nf)
        var invn = GradTensor.from_tensor(ctx, invn_t, False)
        return s.mul(ctx, invn)

    # Aliases (common API)
    fn sum(self, mut ctx: GradContext) -> GradTensor:
        return self.sum_all(ctx)

    fn mean(self, mut ctx: GradContext) -> GradTensor:
        return self.mean_all(ctx)

    # sum_axis: reduce over a given axis (keepdims optional)
    fn sum_axis(self, mut ctx: GradContext, axis: Int, keepdims: Bool = False) -> GradTensor:
        var v = ctx.tape.values[self.id].copy()
        # If your tensor API uses sum(v, Optional[Int](axis), keepdims) keep this; else swap to tensor.sum_axis(v, axis, keepdims)
        var out = tensor.sum(v, Optional[Int](axis), keepdims)
        var track = self.requires_grad and ctx.grad_enabled()
        if not track:
            var lid = ctx.tape.add_leaf(out)
            return GradTensor(lid, False)
        var nid = ctx.tape.add_unary(
            out, TAG_SUM_AX(), self.id,
            0.0, v._shape.copy(), List[Int](), axis, keepdims
        )
        return GradTensor(nid, True)

    # mean_axis: reduce mean over a given axis (keepdims optional)
    fn mean_axis(self, mut ctx: GradContext, axis: Int, keepdims: Bool = False) -> GradTensor:
        var v = ctx.tape.values[self.id].copy()
        # If your tensor API uses mean(v, Optional[Int](axis), keepdims) keep this; else swap to tensor.mean_axis(v, axis, keepdims)
        var out = tensor.mean(v, Optional[Int](axis), keepdims)
        var track = self.requires_grad and ctx.grad_enabled()
        if not track:
            var lid = ctx.tape.add_leaf(out)
            return GradTensor(lid, False)
        var nid = ctx.tape.add_unary(
            out, TAG_MEAN_AX(), self.id,
            0.0, v._shape.copy(), List[Int](), axis, keepdims
        )
        return GradTensor(nid, True)

    # ---------- std (biased, with eps) over axis or all dims ----------
    fn std(self, mut ctx: GradContext, axis: Int = -1, keepdims: Bool = False, eps: Float32 = 1e-12) -> GradTensor:
        var x = ctx.tape.values[self.id].copy()
        var y = tensor.zeros(List[Int]([1])).add_scalar(0.0)   # init

        if axis < 0:
            # all-dims
            var m  = tensor.mean(x)                   # scalar
            var xm = x.sub(m)
            var sq = xm.mul(xm)
            var v  = tensor.mean(sq)                  # scalar
            y = v.add_scalar(eps).sqrt()
        else:
            # axis-wise
            var m  = x.mean( axis)  # keepdims=True to broadcast safely
            var xm = x.sub(m)
            var sq = xm.mul(xm)
            var v  = sq.mean( axis)
            y = v.add_scalar(eps).sqrt()

        var track = self.requires_grad and ctx.grad_enabled()
        if not track:
            var lid = ctx.tape.add_leaf(y)
            return GradTensor(lid, False)

        var nid = ctx.tape.add_unary(
            y, TAG_STD(), self.id,
            eps, x._shape.copy(), List[Int](), axis, keepdims
        )
        return GradTensor(nid, True)

    # ---------- Shape ops ----------

    fn reshape(self, mut ctx: GradContext, new_shape: List[Int]) -> GradTensor:
        var v = ctx.tape.values[self.id].copy()
        var yv = tensor.reshape(v, new_shape)
        var track = self.requires_grad and ctx.grad_enabled()
        if not track:
            var lid = ctx.tape.add_leaf(yv)
            return GradTensor(lid, False)
        var nid = ctx.tape.add_unary(
            yv, TAG_RESHAPE(), self.id,
            0.0, v._shape.copy(), List[Int](), -1, False
        )
        return GradTensor(nid, True)

    fn permute(self, mut ctx: GradContext, order: List[Int]) -> GradTensor:
        var v = ctx.tape.values[self.id].copy()
        var yv = tensor.permute(v, order)
        var track = self.requires_grad and ctx.grad_enabled()
        if not track:
            var lid = ctx.tape.add_leaf(yv)
            return GradTensor(lid, False)
        # Store order in orders[nid] to help backward
        var nid = ctx.tape.add_unary(
            yv, TAG_PERMUTE(), self.id,
            0.0, v._shape.copy(), order.copy(), -1, False
        )
        return GradTensor(nid, True)

    # ---------- Stage 3: Indexing ops (gather / index_axis / take) ----------

    # gather along axis with integer index tensor (indices are non-differentiable)
    fn gather(self, mut ctx: GradContext, axis: Int, index: tensor.Tensor[Int]) -> GradTensor:
        var x = ctx.tape.values[self.id].copy()
        var y = tensor.gather(x, axis, index)
        var track = self.requires_grad and ctx.grad_enabled()
        if not track:
            var lid = ctx.tape.add_leaf(y)
            return GradTensor(lid, False)
        # Store index as a leaf Float32 tensor; cast back to Int in backward.
        var idx_f  = tensor.to_Float32(index)
        var idx_id = ctx.tape.add_leaf(idx_f)       # p1-like leaf (no grad)
        var nid    = ctx.tape.add_binary(y, TAG_GATHER(), self.id, idx_id)
        ctx.tape.axes[nid] = axis
        return GradTensor(nid, True)

    # index_axis: select single index on axis -> dimension is removed
    fn index_axis(self, mut ctx: GradContext, axis: Int, idx: Int) -> GradTensor:
        var x = ctx.tape.values[self.id].copy()
        var y = tensor.index_axis(x, axis, idx)
        var track = self.requires_grad and ctx.grad_enabled()
        if not track:
            var lid = ctx.tape.add_leaf(y)
            return GradTensor(lid, False)
        # store original shape, axis, and scalar idx
        var nid = ctx.tape.add_unary(
            y, TAG_INDEX_AXIS(), self.id,
            Float32(idx), x._shape.copy(), List[Int](), axis, False
        )
        return GradTensor(nid, True)

    # take: select a list of indices along 'axis' (indices are non-differentiable)
    fn take(self, mut ctx: GradContext, js: List[Int], axis: Int = 0) -> GradTensor:
        var x = ctx.tape.values[self.id].copy()
        var y = x.take(js, axis)
        var track = self.requires_grad and ctx.grad_enabled()
        if not track:
            var lid = ctx.tape.add_leaf(y)
            return GradTensor(lid, False)
        # keep original shape + store the picked indices in orders[nid] and the axis
        var nid = ctx.tape.add_unary(
            y, TAG_TAKE(), self.id,
            0.0, x._shape.copy(), js.copy(), axis, False
        )
        return GradTensor(nid, True)



    fn where(self: GradTensor, mut ctx: GradContext, mask: tensor.Tensor[Int], other: GradTensor) -> GradTensor:
        return where(ctx, mask, self, other)

    fn select(self: GradTensor, mut ctx: GradContext, mask: tensor.Tensor[Int], other: GradTensor) -> GradTensor:
        return where(ctx, mask, self, other)

    # scatter: y = scatter(base=self, axis, index, src)

    fn scatter(self, mut ctx: GradContext, axis: Int, index: tensor.Tensor[Int], src: GradTensor) -> GradTensor:
        var base = ctx.tape.values[self.id].copy()
        var s    = ctx.tape.values[src.id].copy()

        #   1) base.scatter(axis, index, s)
        #   2) tensor.scatter(base, axis, index, s)

        var y = base.scatter(axis, index, s)

        var track = (self.requires_grad or src.requires_grad) and ctx.grad_enabled()
        if not track:
            var lid = ctx.tape.add_leaf(y)
            return GradTensor(lid, False)


        var idx_f  = tensor.to_Float32(index)
        var idx_id = ctx.tape.add_leaf(idx_f)  # no grad for indices


        var nid = ctx.tape.add_binary(y, TAG_SCATTER(), self.id, src.id)
        ctx.tape.axes[nid] = axis

        var stash = List[Int]()
        stash.append(idx_id)           # orders[nid][0] = idِ leafِ
        ctx.tape.orders[nid] = stash

        return GradTensor(nid, True)


    # masked_select: treat as non-differentiable selection (compressed output). Returns leaf.
    # ---------- masked_select with elementwise support ----------

    fn masked_select(self, mut ctx: GradContext, mask: tensor.Tensor[Int]) -> GradTensor:
        var x = ctx.tape.values[self.id].copy()
        var y = x.masked_select(mask)
        # If output shape equals input shape, treat as elementwise mask ⇒ differentiable via multiply.
        var elem = (len(y._shape) == len(x._shape))
        if elem:
            var same = True
            var i = 0
            while i < len(x._shape):
                if x._shape[i] != y._shape[i]:
                    same = False
                    break
                i = i + 1
            elem = same
        if not elem:
            # compressed output ⇒ non-differentiable leaf
            var lid = ctx.tape.add_leaf(y)
            return GradTensor(lid, False)
        # elementwise path: store mask as leaf (Float32) and backprop g * (mask>0)
        var track = self.requires_grad and ctx.grad_enabled()
        if not track:
            var lid2 = ctx.tape.add_leaf(y)
            return GradTensor(lid2, False)
        var mask_f = tensor.to_Float32(mask)
        var mid = ctx.tape.add_leaf(mask_f)
        var nid = ctx.tape.add_binary(y, TAG_MASKED_SELECT_ELEM(), self.id, mid)
        return GradTensor(nid, True)

    fn squeeze(self, mut ctx: GradContext, axis: Int) -> GradTensor:
        var v = ctx.tape.values[self.id].copy()
        var yv = tensor.squeeze(v, axis)
        var track = self.requires_grad and ctx.grad_enabled()
        var nid = (ctx.tape.add_unary(yv, TAG_RESHAPE(), self.id, 0.0, v._shape.copy(), List[Int](), -1, False)
                   if track else ctx.tape.add_leaf(yv))
        return GradTensor(nid, track)

    fn unsqueeze(self, mut ctx: GradContext, axis: Int) -> GradTensor:
        var v = ctx.tape.values[self.id].copy()
        var yv = tensor.unsqueeze(v, axis)
        var track = self.requires_grad and ctx.grad_enabled()
        var nid = (ctx.tape.add_unary(yv, TAG_RESHAPE(), self.id, 0.0, v._shape.copy(), List[Int](), -1, False)
                   if track else ctx.tape.add_leaf(yv))
        return GradTensor(nid, track)

    fn repeat(self, mut ctx: GradContext, reps: List[Int]) -> GradTensor:
        var v = ctx.tape.values[self.id].copy()
        var yv = v.repeat(reps)
        var track = self.requires_grad and ctx.grad_enabled()
        var nid = (ctx.tape.add_unary(yv, TAG_REPEAT(), self.id, 0.0, v._shape.copy(), reps.copy(), -1, False)
                   if track else ctx.tape.add_leaf(yv))
        return GradTensor(nid, track)

    fn flatten(self, mut ctx: GradContext) -> GradTensor:
        var v = ctx.tape.values[self.id]
        var n = tensor.numel(v._shape)
        var new_shape = List[Int](); new_shape.append(n)
        var yv = tensor.reshape(v, new_shape)
        var track = self.requires_grad and ctx.grad_enabled()
        var nid = (ctx.tape.add_unary(yv, TAG_FLATTEN(), self.id, 0.0, v._shape.copy(), List[Int](), -1, False)
                   if track else ctx.tape.add_leaf(yv))
        return GradTensor(nid, track)

    fn transpose2d(self, mut ctx: GradContext) -> GradTensor:
        var v = ctx.tape.values[self.id]
        var r = len(v._shape)
        if r == 2:
            var ord = List[Int](); ord.append(1); ord.append(0)
            var yv = _apply_permute(v, ord)
            var track = self.requires_grad and ctx.grad_enabled()
            var nid = (ctx.tape.add_unary(yv, TAG_TRANSPOSE2D(), self.id, 0.0, v._shape.copy(), ord.copy(), -1, False)
                       if track else ctx.tape.add_leaf(yv))
            return GradTensor(nid, track)
        else:
            return self.reshape(ctx, v._shape.copy())


    # Optional: pure 2D matmul
    fn matmul2d(self, mut ctx: GradContext, other: GradTensor) -> GradTensor:
        var a = ctx.tape.values[self.id]
        var b = ctx.tape.values[other.id]
        var y = (tensor.matmul2d(a, b) if has_matmul2d() else tensor.matmul(a, b))
        var track = (self.requires_grad or other.requires_grad) and ctx.grad_enabled()
        if not track:
            var lid = ctx.tape.add_leaf(y)
            return GradTensor(lid, False)
        var nid = ctx.tape.add_binary(y, TAG_MATMUL2D(), self.id, other.id)
        return GradTensor(nid, True)


    @always_inline
    fn has_matmul2d() -> Bool:
        return True

    # ---------- Shape ops ----------

    fn permute(self, mut ctx: GradContext, order: List[Int]) -> GradTensor:
        var v = ctx.tape.values[self.id].copy()
        var yv = tensor.permute(v, order)          # _apply_permute
        var track = self.requires_grad and ctx.grad_enabled()
        if not track:
            var lid = ctx.tape.add_leaf(yv)
            return GradTensor(lid, False)
        var nid = ctx.tape.add_unary(
            yv, TAG_PERMUTE(), self.id,
            0.0, v._shape.copy(), order.copy(), -1, False
        )
        return GradTensor(nid, True)

    # ---------- Unary Math ----------

    fn expm1(self, mut ctx: GradContext) -> GradTensor:
        var v = ctx.tape.values[self.id].copy()
        var y = tensor.expm1(v)
        var track = self.requires_grad and ctx.grad_enabled()
        if not track:
            var lid = ctx.tape.add_leaf(y)
            return GradTensor(lid, False)
        var nid = ctx.tape.add_unary(
            y, TAG_EXPM1(), self.id,
            0.0, v._shape.copy(), List[Int](), -1, False
        )
        return GradTensor(nid, True)

    # ---------- Clipping ----------

    fn clip(self, mut ctx: GradContext, lo: Float32, hi: Float32) -> GradTensor:
        var l = lo
        var h = hi
        if l > h:
            var tmp = l
            l = h
            h = tmp
        # clip(x, lo, hi) = min(max(x, lo), hi)
        var g1 = self.maximum_scalar(ctx, l)
        var g2 = g1.minimum_scalar(ctx, h)
        return g2

    fn clamp(self, mut ctx: GradContext, lo: Float32, hi: Float32) -> GradTensor:
        return self.clip(ctx, lo, hi)


    # General matmul that defers to kernel matmul (supports batched/broadcasted)
    fn matmul(self, mut ctx: GradContext, other: GradTensor) -> GradTensor:
        var a = ctx.tape.values[self.id] .copy()
        var b = ctx.tape.values[other.id] .copy()
        var y = tensor.matmul(a, b)
        var track = (self.requires_grad or other.requires_grad) and ctx.grad_enabled()
        var nid = (ctx.tape.add_binary(y, TAG_MATMUL2D(), self.id, other.id)
                   if track else ctx.tape.add_leaf(y))
        return GradTensor(nid, track)

    fn matmul(self, mut ctx: GradContext, other: tensor.Tensor[Float32]) -> GradTensor:
        var a = ctx.tape.values[self.id].copy()         # Tensor[Float32]
        var y = tensor.matmul(a, other)          # uses your kernel's matmul (2D/batched)

        # track only if self requires grad (other is a plain Tensor with no grad)
        var track = self.requires_grad and ctx.grad_enabled()
        if not track:
            var lid = ctx.tape.add_leaf(y)
            return GradTensor(lid, False)

        # Store `other` as a leaf so backward can read its value; it receives no gradient.
        var bid = ctx.tape.add_leaf(other)
        var nid = ctx.tape.add_binary(y, TAG_MATMUL2D(), self.id, bid)
        return GradTensor(nid, True)

    # Tensordot using core op; axis = number of contracted dims (last of A, first of B)
    fn tensordot(self, mut ctx: GradContext, other: GradTensor, axis: Int = 1) -> GradTensor:
        var a = ctx.tape.values[self.id].copy()
        var b = ctx.tape.values[other.id].copy()
        var y = tensor.tensordot(a, b, axis)
        var track = (self.requires_grad or other.requires_grad) and ctx.grad_enabled()
        if not track:
            var lid = ctx.tape.add_leaf(y)
            return GradTensor(lid, False)
        var nid = ctx.tape.add_binary(y, TAG_TENSORDOT(), self.id, other.id)
        ctx.tape.axes[nid] = axis
        return GradTensor(nid, True)

    # ---------- Piecewise family ----------
    fn maximum_scalar(self, mut ctx: GradContext, c: Float32) -> GradTensor:
        # y = max(x, c)    (forward via core op if exists or compose)
        var x = ctx.tape.values[self.id] .copy()
        # Prefer direct kernel op if available; otherwise compose: max(x,c) = where(x>=c, x, c)
        # Here we simply use kernel's maximum with scalar for clarity:
        var y = x.maximum_scalar(c)
        var track = self.requires_grad and ctx.grad_enabled()
        var nid = (ctx.tape.add_unary(y, TAG_MAXS(), self.id, c, List[Int](), List[Int](), -1, False)
                   if track else ctx.tape.add_leaf(y))
        return GradTensor(nid, track)

    fn minimum_scalar(self, mut ctx: GradContext, c: Float32) -> GradTensor:
        # y = min(x, c)
        var x = ctx.tape.values[self.id].copy()
        var y = x.minimum_scalar( c)
        var track = self.requires_grad and ctx.grad_enabled()
        var nid = (ctx.tape.add_unary(y, TAG_MINS(), self.id, c, List[Int](), List[Int](), -1, False)
                   if track else ctx.tape.add_leaf(y))
        return GradTensor(nid, track)

    fn relu(self, mut ctx: GradContext) -> GradTensor:
        # relu(x) = max(x, 0)
        var x = ctx.tape.values[self.id].copy()
        var y = x.maximum_scalar( 0.0)
        var track = self.requires_grad and ctx.grad_enabled()
        var nid = (ctx.tape.add_unary(y, TAG_RELU(), self.id, 0.0, List[Int](), List[Int](), -1, False)
                   if track else ctx.tape.add_leaf(y))
        return GradTensor(nid, track)

    fn leaky_relu(self, mut ctx: GradContext, alpha: Float32 = 0.01) -> GradTensor:
        # leaky_relu(x,a) = relu(x) + a * min(x,0)
        var x = ctx.tape.values[self.id].copy()
        var pos = x.maximum_scalar( 0.0)
        var neg = x.minimum_scalar( 0.0).mul_scalar(alpha)
        var y = pos.add(neg)
        var track = self.requires_grad and ctx.grad_enabled()
        var nid = (ctx.tape.add_unary(y, TAG_LRELU(), self.id, alpha, List[Int](), List[Int](), -1, False)
                   if track else ctx.tape.add_leaf(y))
        return GradTensor(nid, track)


    # ---------- Stage 7: Type/Cast & I/O helpers ----------
    # Note: GradTensor stores Float32 tensors internally. Casts to non-Float32
    #       return raw Tensor[...] values (non-differentiable). Float→Float32 is identity.

    # Identity cast (already Float32); keeps gradient tracking unchanged.
    fn to_Float32(self, mut ctx: GradContext) -> GradTensor:
        return GradTensor(self.id, self.requires_grad)

    # Return a non-differentiable Int tensor view (value cast)
    fn to_int_value(self, mut ctx: GradContext) -> tensor.Tensor[Int]:
        var x = ctx.tape.values[self.id].copy()
        return to_int(x)

    # Return a non-differentiable Float32 tensor
    fn to_float32_value(self, mut ctx: GradContext) -> tensor.Tensor[Float32]:
        var x = ctx.tape.values[self.id].copy()
        return to_float32(x)

    # Export helpers (non-differentiable):
    fn to_list(self, mut ctx: GradContext) -> List[Float32]:
        var v = List[Float32]()
        var x = ctx.tape.values[self.id].copy()
        var n = len(x._data)
        var i = 0
        while i < n:
            v.append(x._data[i])
            i = i + 1
        return v

    fn to_string(self, mut ctx: GradContext) -> String:
        var x = ctx.tape.values[self.id].copy()
        return x.__str__()



    # ---------- Stage 4: Unary math ops ----------
    fn sqrt(self, mut ctx: GradContext) -> GradTensor:
        var x = ctx.tape.values[self.id].copy()
        var y = x.sqrt()
        var track = self.requires_grad and ctx.grad_enabled()
        var nid = (ctx.tape.add_unary(y, TAG_SQRT(), self.id, 0.0, List[Int](), List[Int](), -1, False)
                   if track else ctx.tape.add_leaf(y))
        return GradTensor(nid, track)

    fn exp(self, mut ctx: GradContext) -> GradTensor:
        var x = ctx.tape.values[self.id].copy()
        var y = x.exp()
        var track = self.requires_grad and ctx.grad_enabled()
        var nid = (ctx.tape.add_unary(y, TAG_EXP(), self.id, 0.0, List[Int](), List[Int](), -1, False)
                   if track else ctx.tape.add_leaf(y))
        return GradTensor(nid, track)

    fn log(self, mut ctx: GradContext) -> GradTensor:
        var x = ctx.tape.values[self.id].copy()
        var y = x.log()
        var track = self.requires_grad and ctx.grad_enabled()
        var nid = (ctx.tape.add_unary(y, TAG_LOG(), self.id, 0.0, List[Int](), List[Int](), -1, False)
                   if track else ctx.tape.add_leaf(y))
        return GradTensor(nid, track)

    fn tanh(self, mut ctx: GradContext) -> GradTensor:
        var x = ctx.tape.values[self.id].copy()
        var y = x.tanh()
        var track = self.requires_grad and ctx.grad_enabled()
        var nid = (ctx.tape.add_unary(y, TAG_TANH(), self.id, 0.0, List[Int](), List[Int](), -1, False)
                   if track else ctx.tape.add_leaf(y))
        return GradTensor(nid, track)

    fn sigmoid(self, mut ctx: GradContext) -> GradTensor:
        var x = ctx.tape.values[self.id].copy()
        var y = x.sigmoid()
        var track = self.requires_grad and ctx.grad_enabled()
        var nid = (ctx.tape.add_unary(y, TAG_SIGMOID(), self.id, 0.0, List[Int](), List[Int](), -1, False)
                   if track else ctx.tape.add_leaf(y))
        return GradTensor(nid, track)

    fn abs(self, mut ctx: GradContext) -> GradTensor:
        var x = ctx.tape.values[self.id].copy()
        var y = x.abs()
        var track = self.requires_grad and ctx.grad_enabled()
        var nid = (ctx.tape.add_unary(y, TAG_ABS(), self.id, 0.0, List[Int](), List[Int](), -1, False)
                   if track else ctx.tape.add_leaf(y))
        return GradTensor(nid, track)

    fn sign(self, mut ctx: GradContext) -> GradTensor:
        var x = ctx.tape.values[self.id].copy()
        var y = x.sign()
        # zero-grad op
        var nid = ctx.tape.add_leaf(y)
        return GradTensor(nid, False)

    fn floor(self, mut ctx: GradContext) -> GradTensor:
        var x = ctx.tape.values[self.id].copy()
        var y = x.floor()
        var nid = ctx.tape.add_leaf(y)
        return GradTensor(nid, False)

    fn ceil(self, mut ctx: GradContext) -> GradTensor:
        var x = ctx.tape.values[self.id].copy()
        var y = x.ceil()
        var nid = ctx.tape.add_leaf(y)
        return GradTensor(nid, False)

    fn round(self, mut ctx: GradContext) -> GradTensor:
        var x = ctx.tape.values[self.id].copy()
        var y = x.round()
        var nid = ctx.tape.add_leaf(y)
        return GradTensor(nid, False)

    fn log1p(self, mut ctx: GradContext) -> GradTensor:
        var x = ctx.tape.values[self.id]
        var y = x.log1p()
        var track = self.requires_grad and ctx.grad_enabled()
        var nid = (ctx.tape.add_unary(y, TAG_LOG1P(), self.id, 0.0, List[Int](), List[Int](), -1, False)
                   if track else ctx.tape.add_leaf(y))
        return GradTensor(nid, track)


    # ---------- Stage 9: softmax / log_softmax (axis-wise) ----------
    # Note: for numerical stability ideally subtract max over axis; if max_axis is unavailable,
    #       we use a simpler (less-stable) formulation.

    # ---------- Stage 10: Stable softmax (subtract max) ----------
    fn softmax(self, mut ctx: GradContext, axis: Int) -> GradTensor:
        var x = ctx.tape.values[self.id].copy()
        var m = _value_max_axis(x, axis, True)
        var mgt = GradTensor.from_tensor(ctx, m, False)
        var xm  = self.sub(ctx, mgt)
        var ex  = xm.exp(ctx)
        var den = ex.sum_axis(ctx, axis, True)
        return ex.div(ctx, den)


    fn logsumexp(self, mut ctx: GradContext, axis: Int, keepdims: Bool = False) -> GradTensor:
        var x = ctx.tape.values[self.id].copy()
        var m = _value_max_axis(x, axis, keepdims)
        var mgt = GradTensor.from_tensor(ctx, m, False)
        var xm  = self.sub(ctx, mgt)
        var ex  = xm.exp(ctx)
        var s   = ex.sum_axis(ctx, axis, keepdims)
        var ls  = s.log(ctx)
        return mgt.add(ctx, ls)

    # ---------- Stage 10: Stable log_softmax ----------
    fn log_softmax(self, mut ctx: GradContext, axis: Int) -> GradTensor:
        var ls = self.logsumexp(ctx, axis, True)
        return self.sub(ctx, ls)

    # clip: clamp values into [lo, hi] using existing scalar ops
    fn clip(self, mut ctx: GradContext, lo: Float32, hi: Float32) -> GradTensor:
        var l = lo
        var h = hi
        if l > h:
            var tmp = l
            l = h
            h = tmp
        # clip(x, lo, hi) = min(max(x, lo), hi)
        var g1 = self.maximum_scalar(ctx, l)
        var g2 = g1.minimum_scalar(ctx, h)
        return g2

    # clamp: alias to clip
    fn clamp(self, mut ctx: GradContext, lo: Float32, hi: Float32) -> GradTensor:
        return self.clip(ctx, lo, hi)


    # ---------- Comparison ops (Stage 1) ----------
    # These produce masks (0.0 or 1.0) as Float32 tensors and do NOT track gradients.
    fn eq(self, mut ctx: GradContext, other: GradTensor) -> GradTensor:
        var a = ctx.tape.values[self.id]
        var b = ctx.tape.values[other.id]
        var yi = a.eq(b)
        var y  = tensor.to_Float32(yi)
        var nid = ctx.tape.add_leaf(y)
        return GradTensor(nid, False)

    fn ne(self, mut ctx: GradContext, other: GradTensor) -> GradTensor:
        var a = ctx.tape.values[self.id]
        var b = ctx.tape.values[other.id]
        var yi = a.ne(b)
        var y  = tensor.to_Float32(yi)
        var nid = ctx.tape.add_leaf(y)
        return GradTensor(nid, False)

    fn lt(self, mut ctx: GradContext, other: GradTensor) -> GradTensor:
        var a = ctx.tape.values[self.id]
        var b = ctx.tape.values[other.id]
        var yi = a.lt(b)
        var y  = tensor.to_Float32(yi)
        var nid = ctx.tape.add_leaf(y)
        return GradTensor(nid, False)

    fn le(self, mut ctx: GradContext, other: GradTensor) -> GradTensor:
        var a = ctx.tape.values[self.id]
        var b = ctx.tape.values[other.id]
        var yi = a.le(b)
        var y  = tensor.to_Float32(yi)
        var nid = ctx.tape.add_leaf(y)
        return GradTensor(nid, False)

    fn gt(self, mut ctx: GradContext, other: GradTensor) -> GradTensor:
        var a = ctx.tape.values[self.id]
        var b = ctx.tape.values[other.id]
        var yi = a.gt(b)
        var y  = tensor.to_Float32(yi)
        var nid = ctx.tape.add_leaf(y)
        return GradTensor(nid, False)

    fn ge(self, mut ctx: GradContext, other: GradTensor) -> GradTensor:
        var a = ctx.tape.values[self.id]
        var b = ctx.tape.values[other.id]
        var yi = a.ge(b)
        var y  = tensor.to_Float32(yi)
        var nid = ctx.tape.add_leaf(y)
        return GradTensor(nid, False)

    # Scalar variants
    fn eq_scalar(self, mut ctx: GradContext, s: Float32) -> GradTensor:
        var a = ctx.tape.values[self.id]
        var yi = a.eq_scalar(s)
        var y  = tensor.to_Float32(yi)
        var nid = ctx.tape.add_leaf(y)
        return GradTensor(nid, False)

    fn ne_scalar(self, mut ctx: GradContext, s: Float32) -> GradTensor:
        var a = ctx.tape.values[self.id]
        var yi = a.ne_scalar(s)
        var y  = tensor.to_Float32(yi)
        var nid = ctx.tape.add_leaf(y)
        return GradTensor(nid, False)

    fn lt_scalar(self, mut ctx: GradContext, s: Float32) -> GradTensor:
        var a = ctx.tape.values[self.id]
        var yi = a.lt_scalar(s)
        var y  = tensor.to_Float32(yi)
        var nid = ctx.tape.add_leaf(y)
        return GradTensor(nid, False)

    fn le_scalar(self, mut ctx: GradContext, s: Float32) -> GradTensor:
        var a = ctx.tape.values[self.id]
        var yi = a.le_scalar(s)
        var y  = tensor.to_Float32(yi)
        var nid = ctx.tape.add_leaf(y)
        return GradTensor(nid, False)

    fn gt_scalar(self, mut ctx: GradContext, s: Float32) -> GradTensor:
        var a = ctx.tape.values[self.id]
        var yi = a.gt_scalar(s)
        var y  = tensor.to_Float32(yi)
        var nid = ctx.tape.add_leaf(y)
        return GradTensor(nid, False)

    fn ge_scalar(self, mut ctx: GradContext, s: Float32) -> GradTensor:
        var a = ctx.tape.values[self.id]
        var yi = a.ge_scalar(s)
        var y  = tensor.to_Float32(yi)
        var nid = ctx.tape.add_leaf(y)
        return GradTensor(nid, False)
    # clip: clamp values into [lo, hi] using existing scalar ops
    # Gradients:
    #   inside (lo < x < hi):    dL/dx = g_out
    #   at/beyond bounds:
    fn clip(self, mut ctx: GradContext, lo: Float32, hi: Float32) -> GradTensor:
        # Optional safety: if lo > hi,
        var l = lo
        var h = hi
        if l > h:
            var tmp = l
            l = h
            h = tmp

        # clip(x, lo, hi) = min( max(x, lo), hi )
        var g1 = self.maximum_scalar(ctx, l)
        var g2 = g1.minimum_scalar(ctx, h)
        return g2


    # ------------------------ Backward pass ------------------------
    fn backward(self, mut ctx: GradContext) -> None:
        # Seed grad of root as ones with same shape
        ctx.tape.grads[self.id] = tensor.zeros(ctx.tape.values[self.id]._shape.copy()).add_scalar(1.0)

        var stack = List[Int](); stack.append(self.id)

        while len(stack) > 0:
            var top_id = stack[len(stack) - 1]
            _ = stack.pop()

            var op = ctx.tape.op_tags[top_id]
            var g_out = ctx.tape.grads[top_id].copy()
            var p0 = ctx.tape.parents[top_id][0]
            var p1 = ctx.tape.parents[top_id][1]
            var p2 = ctx.tape.parents[top_id][2]

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
                    var gx = tensor.zeros(ctx.tape.values[p0]._shape.copy()).add_scalar(s)
                    ctx.tape.grads[p0] = ctx.tape.grads[p0].add(gx)
                    stack.append(p0)

            elif op == TAG_MEAN_ALL():
                if p0 >= 0:
                    var n = tensor.numel(ctx.tape.values[p0]._shape)
                    var s = tensor.item(g_out) / Float32(n)
                    var gx = tensor.zeros(ctx.tape.values[p0]._shape.copy()).add_scalar(s)
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
                    var inv = _invert_perm(ord)
                    var gin = _apply_permute(g_out, inv)
                    ctx.tape.grads[p0] = ctx.tape.grads[p0].add(gin)
                    stack.append(p0)

            elif op == TAG_SUM_AX():
                if p0 >= 0:
                    var shp = ctx.tape.shapes[top_id].copy()
                    var ax = ctx.tape.axes[top_id].copy()
                    var kd = ctx.tape.keep[top_id].copy()
                    var gout_adj = g_out.copy()
                    if not kd:
                        var out_shape = List[Int]()
                        var i = 0
                        while i < len(shp):
                            if i == ax: out_shape.append(1) else: out_shape.append(shp[i])
                            i = i + 1
                        gout_adj = tensor.reshape(g_out, out_shape)
                    var g_in = tensor.zeros(ctx.tape.values[p0]._shape.copy()).add(gout_adj)
                    ctx.tape.grads[p0] = ctx.tape.grads[p0].add(g_in)
                    stack.append(p0)

            elif op == TAG_MEAN_AX():
                if p0 >= 0:
                    var shp = ctx.tape.shapes[top_id].copy()
                    var ax = ctx.tape.axes[top_id].copy()
                    var kd = ctx.tape.keep[top_id].copy()
                    var dim = 1
                    if ax >= 0 and ax < len(shp): dim = shp[ax]
                    var gout_scaled = g_out.mul_scalar(1.0 / Float32(dim))
                    var gout_adj2 = gout_scaled.copy()
                    if not kd:
                        var out_shape2 = List[Int]()
                        var i2 = 0
                        while i2 < len(shp):
                            if i2 == ax: out_shape2.append(1) else: out_shape2.append(shp[i2])
                            i2 = i2 + 1
                        gout_adj2 = tensor.reshape(gout_scaled, out_shape2)
                    var g_in2 = tensor.zeros(ctx.tape.values[p0]._shape.copy()).add(gout_adj2)
                    ctx.tape.grads[p0] = ctx.tape.grads[p0].add(g_in2)
                    stack.append(p0)

            elif op == TAG_NEG():
                if p0 >= 0:
                    ctx.tape.grads[p0] = ctx.tape.grads[p0].add(g_out.mul_scalar(-1.0))
                    stack.append(p0)

            elif op == TAG_SUB():

                if p0 >= 0:
                    ctx.tape.grads[p0] = ctx.tape.grads[p0].add(g_out)
                    stack.append(p0)
                if p1 >= 0:
                    # ∂(a-b)/∂b = -1
                    var gb = g_out.mul_scalar(-1.0)
                    ctx.tape.grads[p1] = ctx.tape.grads[p1].add(gb)
                    stack.append(p1)

            elif op == TAG_DIVS():
                if p0 >= 0:
                    var s = ctx.tape.scalars[top_id]
                    var inv = 1.0 / s
                    ctx.tape.grads[p0] = ctx.tape.grads[p0].add(g_out.mul_scalar(inv))
                    stack.append(p0)

            elif op == TAG_SQUARE():
                if p0 >= 0:
                    var two_x = ctx.tape.values[p0].mul_scalar(2.0)
                    var gin = g_out.mul(two_x)
                    ctx.tape.grads[p0] = ctx.tape.grads[p0].add(gin)
                    stack.append(p0)

            elif op == TAG_POWS():
                if p0 >= 0:
                    var p = ctx.tape.scalars[top_id]
                    if p == 2.0:
                        var two_x = ctx.tape.values[p0].mul_scalar(2.0)
                        var gin = g_out.mul(two_x)
                        ctx.tape.grads[p0] = ctx.tape.grads[p0].add(gin)
                    elif p == 1.0:
                        ctx.tape.grads[p0] = ctx.tape.grads[p0].add(g_out)
                    elif p == 0.0:
                        # constant -> zero grad
                        pass
                    else:
                        var p_minus_1 = p - 1.0
                        var ip = Int(p_minus_1)
                        if Float32(ip) == p_minus_1 and ip >= 1:
                            var acc = tensor.zeros(ctx.tape.values[p0]._shape.copy()).add_scalar(1.0)
                            var i = 0
                            while i < ip:
                                acc = acc.mul(ctx.tape.values[p0])
                                i = i + 1
                            var coeff = acc.mul_scalar(p)
                            var gin2 = g_out.mul(coeff)
                            ctx.tape.grads[p0] = ctx.tape.grads[p0].add(gin2)
                        else:
                            var two_x2 = ctx.tape.values[p0].mul_scalar(2.0)
                            var gin3 = g_out.mul(two_x2)
                            ctx.tape.grads[p0] = ctx.tape.grads[p0].add(gin3)
                    stack.append(p0)

            elif op == TAG_TRANSPOSE2D():
                if p0 >= 0:
                    var ord = ctx.tape.orders[top_id].copy()   # [1,0]
                    var inv = _invert_perm(ord)
                    var gin = _apply_permute(g_out, inv)
                    ctx.tape.grads[p0] = ctx.tape.grads[p0].add(gin)
                    stack.append(p0)

            elif op == TAG_FLATTEN():
                if p0 >= 0:
                    var orig = ctx.tape.shapes[top_id].copy()
                    var gin = tensor.reshape(g_out, orig)
                    ctx.tape.grads[p0] = ctx.tape.grads[p0].add(gin)
                    stack.append(p0)

            elif op == TAG_MATMUL2D():
                # y = a @ b  (2D × 2D)
                if p0 >= 0:
                    # dL/da = g_out @ b^T
                    var bT = ctx.tape.values[p1].transpose2d()
                    var ga = tensor.matmul2d(g_out, bT)
                    ctx.tape.grads[p0] = ctx.tape.grads[p0].add(ga)
                    stack.append(p0)
                if p1 >= 0:
                    # dL/db = a^T @ g_out
                    var aT = ctx.tape.values[p0].transpose2d()
                    var gb = tensor.matmul2d(aT, g_out)
                    ctx.tape.grads[p1] = ctx.tape.grads[p1].add(gb)
                    stack.append(p1)

            elif op == TAG_TENSORDOT():
                var k = ctx.tape.axes[top_id].copy()
                if p0 >= 0 or p1 >= 0:
                    var a = ctx.tape.values[p0].copy()
                    var b = ctx.tape.values[p1].copy()
                    var sa = a._shape.copy()
                    var sb = b._shape.copy()
                    var na = len(sa)
                    var nb = len(sb)
                    var a_outer = 1
                    var i = 0
                    while i < na - k:
                        a_outer = a_outer * sa[i]; i = i + 1
                    var a_inner = 1
                    i = na - k
                    while i < na:
                        a_inner = a_inner * sa[i]; i = i + 1
                    var b_inner = 1
                    i = 0
                    while i < k:
                        b_inner = b_inner * sb[i]; i = i + 1
                    var b_outer = 1
                    i = k
                    while i < nb:
                        b_outer = b_outer * sb[i]; i = i + 1
                    var a2 = tensor.reshape(a, ([a_outer, a_inner]))
                    var b2 = tensor.reshape(b, ([b_inner, b_outer]))
                    var y2 = tensor.reshape(g_out, ([a_outer, b_outer]))
                    fn _t2(t: tensor.Tensor[Float32]) -> tensor.Tensor[Float32]:
                        return tensor.permute(t, ([1,0]))
                    if p0 >= 0:
                        var g0_2d = tensor.matmul(y2, _t2(b2))
                        var g0 = tensor.reshape(g0_2d, sa)
                        ctx.tape.grads[p0] = ctx.tape.grads[p0].add(g0)
                        stack.append(p0)
                    if p1 >= 0:
                        var g1_2d = tensor.matmul(_t2(a2), y2)
                        var g1 = tensor.reshape(g1_2d, sb)
                        ctx.tape.grads[p1] = ctx.tape.grads[p1].add(g1)
                        stack.append(p1)


            elif op == TAG_SCATTER():
                 # parents: p0 = base(self), p1 = src
                var axis = ctx.tape.axes[top_id].copy()
                # index leaf id was stashed in orders[top_id][0] as Float32; convert back to Int
                var list_ids = ctx.tape.orders[top_id].copy()
                var mid = list_ids[0].copy()
                var idx_f = ctx.tape.values[mid].copy()
                var index = to_int(idx_f)

                # dL/dsrc = gather(g_out, axis, index)
                if p1 >= 0:
                    var g_src = tensor.gather(g_out, axis, (index))
                    ctx.tape.grads[p1] = ctx.tape.grads[p1].add(g_src)
                    stack.append(p1)

                # dL/dself = g_out masked by zeros at overwritten positions
                if p0 >= 0:
                    var base = ctx.tape.values[p0].copy()
                    var ones = tensor.zeros(base._shape.copy()).add_scalar(1.0)  # mask of ones
                    # zeros like src to scatter zeros into mask
                    var zsrc = tensor.zeros(ctx.tape.values[p1]._shape.copy())
                    var keep_mask = ones.scatter(axis, index, zsrc)              # overwritten → 0
                    var g_self = g_out.mul(keep_mask)
                    ctx.tape.grads[p0] = ctx.tape.grads[p0].add(g_self)
                    stack.append(p0)

            elif op == TAG_WHERE():
                var list_ids = ctx.tape.orders[top_id].copy()
                var mid = list_ids[0].copy()
                var m01 = ctx.tape.values[mid].copy()
                if p0 >= 0:
                    var ma = broadcast_to_shape(m01, ctx.tape.values[p0]._shape.copy())
                    var ga = g_out.mul(ma)
                    ctx.tape.grads[p0] = ctx.tape.grads[p0].add(ga)
                    stack.append(p0)
                if p1 >= 0:
                    var one = tensor.zeros(m01._shape.copy()).add_scalar(1.0)
                    var mb = broadcast_to_shape( one.sub(m01), ctx.tape.values[p1]._shape.copy() )
                    var gb = g_out.mul(mb)
                    ctx.tape.grads[p1] = ctx.tape.grads[p1].add(gb)
                    stack.append(p1)


            elif op == TAG_MASKED_SELECT_ELEM():
                # y = x * (mask>0); mask stored as Float32 leaf in p1
                if p0 >= 0:
                    var mf = ctx.tape.values[p1].copy()
                    var mi = to_int(mf)
                    var m01 = tensor.to_Float32(mi.ne_scalar(0))
                    var gin = g_out.mul(m01)
                    ctx.tape.grads[p0] = ctx.tape.grads[p0].add(gin)
                    stack.append(p0)

                var axis = ctx.tape.axes[top_id].copy()
                # Recover index leaf id from orders slot
                var list_ids = ctx.tape.orders[top_id].copy()
                var idx_id = list_ids[0]
                var idx_f = ctx.tape.values[idx_id].copy()
                var idx = to_int(idx_f)
                # Grad wrt base (p0): mask out overwritten positions
                if p0 >= 0:
                    var yshape = ctx.tape.values[top_id]._shape.copy()
                    var ones = tensor.zeros(yshape).add_scalar(1.0)
                    var zeros_like_src = tensor.zeros(ctx.tape.values[p1]._shape.copy())
                    var mask_over = ones.scatter(axis, idx, zeros_like_src)
                    var gbase = g_out.mul(mask_over)
                    ctx.tape.grads[p0] = ctx.tape.grads[p0].add(gbase)
                    stack.append(p0)
                # Grad wrt src (p1): gather grad_out at indices
                if p1 >= 0:
                    var gsrc = tensor.gather(g_out, axis, idx)
                    ctx.tape.grads[p1] = ctx.tape.grads[p1].add(gsrc)
                    stack.append(p1)


            elif op == TAG_TAKE():
                if p0 >= 0:
                    var shp = ctx.tape.shapes[top_id].copy()         # original x shape
                    var axis = ctx.tape.axes[top_id].copy()
                    var js   = ctx.tape.orders[top_id].copy()        # stored indices list
                    var gin  = tensor.zeros(shp)
                    var k = len(js)
                    var i = 0
                    while i < k:
                        var j = js[i]
                        # extract slice i of grad_out along axis, then scatter_add into position j
                        var slice_i = g_out.index_axis( axis, i)
                        var slice_i_exp = tensor.unsqueeze(slice_i, axis)
                        # build full index tensor filled with j
                        var idx_full = tensor.zeros(shp).add_scalar(Float32(j))
                        var idx_int  = to_int(idx_full)
                        gin = gin.scatter_add( axis, idx_int, slice_i_exp)
                        i = i + 1
                    ctx.tape.grads[p0] = ctx.tape.grads[p0].add(gin)
                    stack.append(p0)


            elif op == TAG_GATHER():
                 # parent: p0 = x ; p1 holds index leaf (Float32)
                var axis = ctx.tape.axes[top_id].copy()
                var idx_leaf = p1

                var idx_f = ctx.tape.values[idx_leaf].copy()
                var index = to_int(idx_f)

                if p0 >= 0:
                    # dL/dx = scatter_add(axis, index, g_out)
                    var x = ctx.tape.values[p0].copy()
                    var gx = tensor.zeros(x._shape.copy())
                    gx = gx.scatter_add(axis, index, g_out)   #kernel scatter_add
                    ctx.tape.grads[p0] = ctx.tape.grads[p0].add(gx)
                    stack.append(p0)


            elif op == TAG_INDEX_AXIS():
                # y = index_axis(x, axis, idx); dx is g_out placed at idx along axis, zeros elsewhere.
                if p0 >= 0:
                    var shp = ctx.tape.shapes[top_id].copy()   # original x shape
                    var axis = ctx.tape.axes[top_id].copy()
                    var idx_scalar = ctx.tape.scalars[top_id].copy()
                    # expand grad back to x-shape
                    var gout = g_out.copy().unsqueeze(axis)
                    # build index tensor filled with idx of x-shape
                    var idx_full = tensor.zeros(shp).add_scalar(idx_scalar)   # Float32
                    var idx_int = to_int(idx_full)                             # Tensor[Int]
                    var zeros = tensor.zeros(shp)
                    var gin = zeros.scatter_add( axis, idx_int, gout)
                    ctx.tape.grads[p0] = ctx.tape.grads[p0].add(gin)
                    stack.append(p0)

                # y = gather(x, axis, index); grads: dx = scatter_add(grad_y, axis, index)
                if p0 >= 0:
                    var axis = ctx.tape.axes[top_id].copy()
                    var idx_f = ctx.tape.values[p1].copy()              # stored as Float32
                    var idx = to_int(idx_f)                      # cast back to Int indices
                    var x = ctx.tape.values[p0].copy()
                    var zeros = tensor.zeros(x._shape.copy())
                    var gin = zeros.scatter_add( axis, idx, g_out)
                    ctx.tape.grads[p0] = ctx.tape.grads[p0].add(gin)
                    stack.append(p0)


            elif op == TAG_DIV():
                # y = a / b
                if p0 >= 0:
                    # dL/da = g_out * (1 / b)
                    var invb = tensor.reciprocal(ctx.tape.values[p1])
                    var ga = g_out.mul(invb)
                    ctx.tape.grads[p0] = ctx.tape.grads[p0].add(ga)
                    stack.append(p0)
                if p1 >= 0:
                    # dL/db = -g_out * a / (b*b)
                    var a_val = ctx.tape.values[p0].copy()
                    var invb = tensor.reciprocal(ctx.tape.values[p1])
                    var invb2 = invb.mul(invb)
                    var gb = g_out.mul(a_val).mul(invb2).mul_scalar(-1.0)
                    ctx.tape.grads[p1] = ctx.tape.grads[p1].add(gb)
                    stack.append(p1)


            elif op == TAG_REPEAT():
                if p0 >= 0:
                    var shp = ctx.tape.shapes[top_id].copy()    # original input shape
                    var reps = ctx.tape.orders[top_id].copy()    # reps stored in orders
                    var inter = List[Int]()
                    var i = 0
                    while i < len(shp):
                        inter.append(shp[i])
                        inter.append(reps[i])
                        i = i + 1
                    var gout2 = tensor.reshape(g_out, inter)
                    i = 0
                    while i < len(shp):
                        if reps[i] > 1:
                            gout2 = gout2.sum( axis=(2*i + 1), keepdims=True)
                        i = i + 1
                    var gin = gout2.reshape( shp)
                    ctx.tape.grads[p0] = ctx.tape.grads[p0].add(gin)
                    stack.append(p0)

                if p0 >= 0:
                    var b = ctx.tape.values[p1].copy()
                    var bt = _apply_permute(b, ([1, 0]))
                    var dA = tensor.matmul(g_out, bt)
                    ctx.tape.grads[p0] = ctx.tape.grads[p0].add(dA)
                    stack.append(p0)
                if p1 >= 0:
                    var a = ctx.tape.values[p0].copy()
                    var at = _apply_permute(a, ([1, 0]))
                    var dB = tensor.matmul(at, g_out)
                    ctx.tape.grads[p1] = ctx.tape.grads[p1].add(dB)
                    stack.append(p1)

            elif op == TAG_MAXS():
                if p0 >= 0:
                    var c = ctx.tape.scalars[top_id].copy()
                    var x = ctx.tape.values[p0].copy()
                    var y = ctx.tape.values[top_id].copy()      # y = max(x, c)
                    # mask = (y == x)
                    var mask = y.eq( x)
                    var zeros = tensor.zeros(x._shape.copy())
                    var gin = tensor.where(mask, g_out, zeros)
                    ctx.tape.grads[p0] = ctx.tape.grads[p0].add(gin)
                    stack.append(p0)

            elif op == TAG_MINS():
                # scalar s was saved in scalars[top_id]
                if p0 >= 0:
                    var s = ctx.tape.scalars[top_id].copy()
                    var x = ctx.tape.values[p0].copy()

                    var m =  (x.lt_scalar(s))
                    var mf = tensor.to_Float32(m)
                    var gin = g_out.mul(mf)
                    ctx.tape.grads[p0] = ctx.tape.grads[p0].add(gin)
                    stack.append(p0)

            elif op == TAG_SQRT():
                if p0 >= 0:
                    var y = ctx.tape.values[top_id].copy()         # y = sqrt(x)
                    var gin = g_out.mul( tensor.reciprocal(y).mul_scalar(  0.5) )
                    ctx.tape.grads[p0] = ctx.tape.grads[p0].add(gin)
                    stack.append(p0)

            elif op == TAG_EXP():
                if p0 >= 0:
                    var y = ctx.tape.values[top_id].copy()         # y = exp(x)
                    var gin = g_out.mul(y)                  # d/dx exp = exp
                    ctx.tape.grads[p0] = ctx.tape.grads[p0].add(gin)
                    stack.append(p0)

            elif op == TAG_LOG():
                if p0 >= 0:
                    var x = ctx.tape.values[p0].copy()
                    var gin = tensor.div_t(g_out,x)                  # d/dx log = 1/x
                    ctx.tape.grads[p0] = ctx.tape.grads[p0].add(gin)
                    stack.append(p0)

            elif op == TAG_TANH():
                if p0 >= 0:
                    var y = ctx.tape.values[top_id].copy()         # y = tanh(x)
                    var one = tensor.zeros(y._shape.copy()).add_scalar(1.0)
                    var gin = g_out.mul( one.sub( y.mul(y) ) )  # 1 - y^2
                    ctx.tape.grads[p0] = ctx.tape.grads[p0].add(gin)
                    stack.append(p0)

            elif op == TAG_SIGMOID():
                if p0 >= 0:
                    var y = ctx.tape.values[top_id].copy()       # y = sigmoid(x)
                    var one = tensor.zeros(y._shape.copy()).add_scalar(1.0)
                    var gin = g_out.mul( y.mul( one.sub(y) ) )   # y*(1-y)
                    ctx.tape.grads[p0] = ctx.tape.grads[p0].add(gin)
                    stack.append(p0)

            elif op == TAG_ABS():
                if p0 >= 0:
                    var x = ctx.tape.values[p0].copy()
                    var signx = x.sign()
                    var gin = g_out.mul(signx)              # subgradient; at 0 => 0
                    ctx.tape.grads[p0] = ctx.tape.grads[p0].add(gin)
                    stack.append(p0)

            elif op == TAG_LOG1P():
                if p0 >= 0:
                    var x = ctx.tape.values[p0].copy()
                    var one = tensor.zeros(x._shape.copy()).add_scalar(1.0)
                    var gin = tensor.div_t(g_out, one.add(x) )       # 1/(1+x)
                    ctx.tape.grads[p0] = ctx.tape.grads[p0].add(gin)
                    stack.append(p0)

            elif op == TAG_EXPM1():
                if p0 >= 0:
                    var x = ctx.tape.values[p0].copy()
                    var ex = x.exp()         # ∂/∂x expm1(x) = exp(x)
                    var gin = g_out.mul(ex)
                    ctx.tape.grads[p0] = ctx.tape.grads[p0].add(gin)
                    stack.append(p0)

            elif op == TAG_STD():
                if p0 >= 0:
                    var eps = ctx.tape.scalars[top_id]
                    var axis = ctx.tape.axes[top_id].copy()
                    var keep = ctx.tape.keep[top_id].copy()
                    var x = ctx.tape.values[p0].copy()
                    if axis < 0:
                        var m = x.mean()               # scalar
                        var xm = x.sub(m)
                        var v  = xm.mul(xm.mean())
                        var sd = v.add_scalar(eps).sqrt()
                        var n = 1
                        var i = 0
                        while i < len(x._shape):
                            n = n * x._shape[i]
                            i = i + 1
                        var nf = Float32(n)
                        var coeff = tensor.zeros(sd._shape.copy()).add_scalar(1.0 / nf)
                        var gscale = tensor.div_t(coeff,sd)           # 1/(n*sd)
                        var gout = g_out.mul(gscale)         # scalar scale
                        var gin = xm.mul(gout)               # (x-m)/(n*sd) * g_out
                        ctx.tape.grads[p0] = ctx.tape.grads[p0].add(gin)
                        stack.append(p0)
                    else:
                        var mk = x.mean(axis)  # keepdims for xm
                        var xm = x.sub(mk)
                        var sq = xm.mul(xm)
                        var v  = sq.mean(axis) # keepdims True for sd broadcast
                        var sd = v.add_scalar(eps).sqrt()
                        var n = x._shape[axis]
                        var nf = Float32(n)
                        var coeff = tensor.zeros(sd._shape.copy()).add_scalar(1.0 / nf)
                        var gscale = tensor.div_t(coeff,sd)           # [.. keepdims ..]
                        var gout = g_out.copy()
                        if not keep:
                            gout = tensor.unsqueeze(gout, axis)
                        var scaled = gout.mul(gscale)
                        var gin = xm.mul(scaled)             # (x-m)/(n*sd) * g_out
                        ctx.tape.grads[p0] = ctx.tape.grads[p0].add(gin)
                        stack.append(p0)

                if p0 >= 0:
                    var x = ctx.tape.values[p0].copy()
                    var gin = g_out.mul( x.exp() )    # d/dx expm1 = exp
                    ctx.tape.grads[p0] = ctx.tape.grads[p0].add(gin)
                    stack.append(p0)

            elif op == TAG_CLAMP_DISABLED():
                if p0 >= 0:
                    var x  = ctx.tape.values[p0].copy()
                    var lo = ctx.tape.scalars[top_id].copy()       # min
                    var hi = ctx.tape.orders[top_id][0]     # store hi in orders[0] as Float32 bitcast? not safe

                if p0 >= 0:
                    var c = ctx.tape.scalars[top_id]
                    var x = ctx.tape.values[p0].copy()
                    var y = ctx.tape.values[top_id].copy()      # y = min(x, c)
                    # mask = (y == x)
                    var mask = y.eq( x)
                    var zeros = tensor.zeros(x._shape.copy())
                    var gin = tensor.where(mask, g_out, zeros)
                    ctx.tape.grads[p0] = ctx.tape.grads[p0].add(gin)
                    stack.append(p0)

            elif op == TAG_RELU():
                if p0 >= 0:
                    var x = ctx.tape.values[p0].copy()
                    var y = ctx.tape.values[top_id].copy()      # y = max(x, 0)
                    var mask = y.eq( x)           # true where x>0 (ties at 0 => subgrad=0)
                    var zeros = tensor.zeros(x._shape.copy())
                    var gin = tensor.where(mask, g_out, zeros)
                    ctx.tape.grads[p0] = ctx.tape.grads[p0].add(gin)
                    stack.append(p0)

            elif op == TAG_LRELU():
                if p0 >= 0:
                    var a = ctx.tape.scalars[top_id].copy()     # slope
                    var x = ctx.tape.values[p0].copy()
                    var y_relu = x.maximum_scalar( 0.0)
                    var mask_pos = y_relu.eq( x)  # x > 0
                    var zeros = tensor.zeros(x._shape.copy())
                    var part_pos = tensor.where(mask_pos, g_out, zeros)

                    # negative part contributes alpha where x < 0:
                    var y_min0 = x.minimum_scalar(0.0)  # (<=0)
                    var mask_neg = y_min0.eq( x)         # x < 0
                    var part_neg = tensor.where(mask_neg, g_out.mul_scalar(a), zeros)

                    var gin = part_pos.add(part_neg)
                    ctx.tape.grads[p0] = ctx.tape.grads[p0].add(gin)
                    stack.append(p0)

            elif op == TAG_CLIP() or op == TAG_CLAMP() or op == TAG_CLAMP_DISA():
                # y = clamp(x, lo, hi)
                # والدها طبق نوار: p0=x, p1=lo, p2=hi
                var x  = ctx.tape.values[p0].copy()
                var lo = ctx.tape.values[p1].copy()
                var hi = ctx.tape.values[p2].copy()

                # ماسک‌ها: داخل بازه، زیر lo، بالای hi
                # فرض: API تنسورها has: gt, lt, and, cast_to_float, where, mul, add, etc.
                var m_in  = tensor.and_t(x.gt(lo), x.lt( hi))      # (x>lo) & (x<hi)
                var m_lo  = x.lt( lo)                                    # x<lo
                var m_hi  = x.gt( hi)                                    # x>hi

                # dL/dx: g_out در ناحیه درون‌بازه عبور می‌کند، بیرون بازه صفر
                if p0 >= 0:
                    var pass_through = tensor.where(m_in, g_out, tensor.zeros(g_out.shape()))
                    ctx.tape.grads[p0] = ctx.tape.grads[p0].add(pass_through)
                    stack.append(p0)

                # dL/dlo: اگر x<lo آوت‌پوت دقیقاً lo است ⇒ dy/dlo = 1
                if p1 >= 0:
                    var g_lo = tensor.where(m_lo, g_out, tensor.zeros(g_out.shape()))
                    ctx.tape.grads[p1] = ctx.tape.grads[p1].add(g_lo)
                    stack.append(p1)

                # dL/dhi: اگر x>hi آوت‌پوت دقیقاً hi است ⇒ dy/dhi = 1
                if p2 >= 0:
                    var g_hi = tensor.where(m_hi, g_out, tensor.zeros(g_out.shape()))
                    ctx.tape.grads[p2] = ctx.tape.grads[p2].add(g_hi)
                    stack.append(p2)

            elif op == TAG_CLIP() or op == TAG_CLAMP_DISABLED():
                # y = clamp(x, lo, hi)
                var x  = ctx.tape.values[p0].copy()
                var lo = ctx.tape.values[p1].copy()
                var hi = ctx.tape.values[p2].copy()

                var m_in = tensor.and_t(x.gt( lo), x.lt( hi))
                if p0 >= 0:
                    var pass_grad = tensor.where(m_in, g_out, tensor.zeros(g_out.shape()))
                    ctx.tape.grads[p0] = ctx.tape.grads[p0].add(pass_grad)
                    stack.append(p0)

        # Final consistency: coerce grad shapes to value shapes
        var idx = 0
        while idx < len(ctx.tape.values):
            var vsh = ctx.tape.values[idx]._shape.copy()
            var gsh = ctx.tape.grads[idx]._shape.copy()
            var same = True
            if len(vsh) != len(gsh): same = False
            else:
                var t2 = 0
                while t2 < len(vsh):
                    if vsh[t2] != gsh[t2]: same = False; break
                    t2 = t2 + 1
            if not same:
                ctx.tape.grads[idx] = tensor.zeros(vsh).add(ctx.tape.grads[idx])
            idx = idx + 1


# ---------- Stage 9: Losses

# ---------- Reduction helper for losses ----------
# reduction: "mean" | "sum" | "none"
fn _reduce_loss(mut ctx: GradContext, loss: GradTensor, reduction: String) -> GradTensor:
    if reduction == String("sum"):
        return loss.sum_all(ctx)
    elif reduction == String("none"):
        return loss
    else:
        return loss.mean_all(ctx)



# ---------- Stage 9: Losses (NLLLoss & CrossEntropy) ----------

# Reduction helper for losses: "mean" | "sum" | "none"
fn _reduce_loss(mut ctx: GradContext, loss: GradTensor, reduction: String) -> GradTensor:
    if reduction == String("sum"):
        return loss.sum_all(ctx)
    elif reduction == String("none"):
        return loss
    else:
        return loss.mean_all(ctx)

# nll_loss: log_probs = log-softmax(values) روی محور 'axis'


# reduction: "mean" | "sum" | "none"
fn nll_loss(
    mut ctx: GradContext,
    log_probs: GradTensor,
    target: tensor.Tensor[Int],
    axis: Int = 1,
    class_weight: tensor.Tensor[Float32] = tensor.zeros(List[Int]([0])).add_scalar(0.0),
    reduction: String = String("mean")
) -> GradTensor:
    # -log p_y
    var picked = log_probs.gather(ctx, axis, target).neg(ctx)

    # class weighting
    var use_w = len(class_weight._shape) > 0
    if use_w:
        var wgt  = GradTensor.from_tensor(ctx, class_weight, False)  # [C]
        var wsel = wgt.gather(ctx, 0, target)
        picked = picked.mul(ctx, wsel)

    return _reduce_loss(ctx, picked, reduction)

# cross_entropy: logits → log_softmax → nll_loss
fn cross_entropy(
    mut ctx: GradContext,
    logits: GradTensor,
    target: tensor.Tensor[Int],
    axis: Int = 1,
    class_weight: tensor.Tensor[Float32] = tensor.zeros(List[Int]([0])).add_scalar(0.0),
    reduction: String = String("mean")
) -> GradTensor:
    var lsm = logits.log_softmax(ctx, axis)
    return nll_loss(ctx, lsm, target, axis, class_weight, reduction)


# ---------- Stage 9B: Binary losses (stable) ----------

# Binary Cross-Entropy with logits (numerically stable) + optional pos_weight + reduction
# logits: GradTensor (هر شکلی), target: Tensor[Float32]
# pos_weight: Tensor[Float32] (broadcastable به شکل logits)
fn bce_with_logits(
    mut ctx: GradContext,
    logits: GradTensor,
    target: tensor.Tensor[Float32],
    pos_weight: tensor.Tensor[Float32] = tensor.zeros(List[Int]([0])).add_scalar(0.0),
    reduction: String = String("mean")
) -> GradTensor:
    var x = logits
    var t = GradTensor.from_tensor(ctx, target, False)


    var absx  = x.abs(ctx)
    var lsp   = absx.neg(ctx).exp(ctx).log1p(ctx)     # log(1 + exp(-|x|))
    var x_pos = x.maximum_scalar(ctx, 0.0)


    var x_shape = ctx.tape.values[x.id]._shape.copy()
    var one_t   = tensor.zeros(x_shape).add_scalar(1.0)
    var one     = GradTensor.from_tensor(ctx, one_t, False)

    var use_pw = len(pos_weight._shape) > 0
    var pw     = (GradTensor.from_tensor(ctx, pos_weight, False) if use_pw else one)

    # loss = (1 - t) * (x_pos + lsp) + t * (pos_weight * (x_pos - x) + pos_weight * lsp)
    var neg_term = one.sub(ctx, t).mul(ctx, x_pos.add(ctx, lsp))
    var pos_term = t.mul(ctx, pw.mul(ctx, x_pos.sub(ctx, x).add(ctx, lsp)))
    var loss     = neg_term.add(ctx, pos_term)

    return _reduce_loss(ctx, loss, reduction)

# Binary Cross-Entropy روی احتمال‌ها با weight و reduction
# probs: GradTensor در (0,1)، target: Tensor[Float32]
# weight: Tensor[Float32] (broadcastable)
fn binary_cross_entropy(
    mut ctx: GradContext,
    probs: GradTensor,
    target: tensor.Tensor[Float32],
    eps: Float32 = 1e-12,
    weight: tensor.Tensor[Float32] = tensor.zeros(List[Int]([0])).add_scalar(0.0),
    reduction: String = String("mean")
) -> GradTensor:
    var p = probs.clip(ctx, eps, 1.0 - eps)
    var t = GradTensor.from_tensor(ctx, target, False)

    var shape = ctx.tape.values[p.id]._shape.copy()
    var one_t = tensor.zeros(shape).add_scalar(1.0)
    var one   = GradTensor.from_tensor(ctx, one_t, False)

    # -( t*log(p) + (1-t)*log(1-p) )
    var term_pos = t.mul(ctx, p.log(ctx)).neg(ctx)
    var term_neg = one.sub(ctx, t).mul(ctx, one.sub(ctx, p).log(ctx)).neg(ctx)
    var loss     = term_pos.add(ctx, term_neg)


    var use_w = len(weight._shape) > 0
    if use_w:
        var w = GradTensor.from_tensor(ctx, weight, False)
        loss = loss.mul(ctx, w)

    return _reduce_loss(ctx, loss, reduction)



fn where(mut ctx: GradContext, mask: tensor.Tensor[Int], a: GradTensor, b: GradTensor) -> GradTensor:
    var av = ctx.tape.values[a.id]
    var bv = ctx.tape.values[b.id]
    var m01 = tensor.to_Float32(mask.ne_scalar(0))
    var ma = broadcast_to_shape(m01, av._shape.copy())
    var mb = broadcast_to_shape(m01, bv._shape.copy())
    var oneb = tensor.zeros(mb._shape.copy()).add_scalar(1.0)
    var y = av.mul(ma).add( bv.mul( oneb.sub(mb) ) )
    var track = (a.requires_grad or b.requires_grad) and ctx.grad_enabled()
    if not track:
        var lid = ctx.tape.add_leaf(y)
        return GradTensor(lid, False)
    var mid = ctx.tape.add_leaf(m01)
    var nid = ctx.tape.add_binary(y, TAG_WHERE(), a.id, b.id)
    var ids = List[Int]()
    ids.append(mid)
    ctx.tape.orders[nid] = ids
    return GradTensor(nid, True)


fn mask_not(mask: tensor.Tensor[Int]) -> tensor.Tensor[Int]:
    var f = tensor.to_Float32(mask)
    var one = tensor.zeros(f._shape.copy()).add_scalar(1.0)
    var inv = one.sub(f)
    return to_int(inv)

fn mask_and(a: tensor.Tensor[Int], b: tensor.Tensor[Int]) -> tensor.Tensor[Int]:
    var fa = tensor.to_Float32(a)
    var fb = tensor.to_Float32(b)
    var prod = fa.mul(fb)
    return to_int(prod.ne_scalar(0))

fn mask_or(a: tensor.Tensor[Int], b: tensor.Tensor[Int]) -> tensor.Tensor[Int]:
    var fa = tensor.to_Float32(a)
    var fb = tensor.to_Float32(b)
    var s = fa.add(fb)
    return to_int(s.ne_scalar(0))

@always_inline
fn to_int(x: tensor.Tensor[Float32]) -> tensor.Tensor[Int]:
    # Convert Float32 tensor to Int tensor with the same shape.
    var shp = x.shape().copy()
    var n = len(x._data)

    # Allocate by (shape, fill) to avoid ctor ambiguity.
    var out = tensor.Tensor[Int](shp, Int(0))

    var i = 0
    while i < n:
        # Optional strict check if you expect true integers:
        # assert(x._data[i] == Float32(Int(x._data[i])))
        out._data[i] = Int(x._data[i])
        i = i + 1

    return out.copy()
