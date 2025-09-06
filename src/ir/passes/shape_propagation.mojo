# ============================================================================
# MIT License
# Copyright (c) 2025 Morteza Talebou and Mitra Daneshmand
# Project: momijo  |  Source: https://github.com/taleblou/momijo
# This file is part of the Momijo project. See the LICENSE file at the repository root.
# Momijo
# SPDX-License-Identifier: MIT
# Website: https://taleblou.ir/
# Repository: https://github.com/taleblou/momijo
#
# Project: momijo.ir
# File: src/momijo/ir/shape_propagation.mojo
# Description: Shape propagation / inference for Momijo IR (advanced).
#              Adds: negative axis handling, concat/pad/reduce, batched matmul,
#              conv2d, gather, scatter, gather_nd, reduce_window (N-D) with
#              dilation/stride/padding, and pool2d (NHWC).
# Notes:
#   - Self-contained fallback IR; no globals, no 'export'; only 'var' (no 'let').
#   - Unknown dims are -1; ints may be negative; '?' means unknown in attrs.
#   - Emits attrs: 'shape=[...]' and 'shape.error=*'/'shape.warn=*'.
#   - Includes tiny printer & self-test (prints OK).
# ============================================================================

# -----------------------------
# Minimal IR model
# -----------------------------

struct Location:
    var filename: String
    var line: Int
    var col: Int
    fn __init__(out self, filename: String = String("<unknown>"), line: Int = 0, col: Int = 0):
        self.filename = filename
        self.line = line
        self.col = col

struct TypeDesc:
    var name: String
    fn __init__(out self, name: String): self.name = name

@register_passable
struct Value:
    var id: Int
    var typ: TypeDesc
    fn __init__(out self, id: Int, typ: TypeDesc):
        self.id = id
        self.typ = typ

struct Result:
    var value: Value
    fn __init__(out self, value: Value): self.value = value

struct Operand:
    var value: Value
    fn __init__(out self, value: Value): self.value = value

struct Op:
    var name: String
    var results: List[Result]
    var operands: List[Operand]
    var attrs: List[String]
    var loc: Location

    fn __init__(out self, name: String, loc: Location):
        self.name = name
        self.results = List[Result]()
        self.operands = List[Operand]()
        self.attrs = List[String]()
        self.loc = loc

struct Block:
    var name: String
    var args: List[Value]
    var ops: List[Op]

    fn __init__(out self, name: String):
        self.name = name
        self.args = List[Value]()
        self.ops = List[Op]()

struct Function:
    var name: String
    var arg_types: List[TypeDesc]
    var ret_types: List[TypeDesc]
    var blocks: List[Block]

    fn __init__(out self, name: String):
        self.name = name
        self.arg_types = List[TypeDesc]()
        self.ret_types = List[TypeDesc]()
        self.blocks = List[Block]()

struct Module:
    var name: String
    var functions: List[Function]
    fn __init__(out self, name: String = String("module")):
        self.name = name
        self.functions = List[Function]()

# -----------------------------
# Shape helpers
# -----------------------------

fn is_tensor(t: TypeDesc) -> Bool: return t.name.starts_with(String("tensor<"))
fn is_buffer(t: TypeDesc) -> Bool: return t.name.starts_with(String("buffer<"))

struct ShapeInfo:
    var kind: String    # "tensor" or "buffer" or ""
    var dtype: String
    var dims: List[Int]   # -1 for unknown
    fn __init__(out self):
        self.kind = String("")
        self.dtype = String("")
        self.dims = List[Int]()

fn _parse_int_token(tok: String, out ok: Bool) -> Int:
    ok = False
    if tok.size() == 0: return 0
    if tok == String("?"):
        ok = True; return -1
    var sign = 1
    var i = 0
    if tok.bytes()[0] == 45:
        sign = -1; i = 1
    var num = 0
    while i < tok.size():
        var ch = tok.bytes()[i]
        if ch < 48 or ch > 57: return 0
        num = num * 10 + Int(ch - 48)
        i = i + 1
    ok = True
    return sign * num

fn _parse_int_list(br: String, out vals: List[Int], out ok: Bool) -> None:
    ok = False
    vals = List[Int]()
    if br.size() < 2:
        ok = True; return
    var s = br
    var start = 0; var end = s.size()
    if s.bytes()[0] == 91 and s.bytes()[s.size()-1] == 93:  # '[' and ']'
        start = 1; end = s.size() - 1
    var tok = String("")
    var i = start
    while i < end:
        var ch = s.bytes()[i]
        if ch == 44:
            var ok1 = False; var v = _parse_int_token(tok, ok1)
            if not ok1: return
            vals.push_back(v); tok = String("")
        else:
            tok = tok + String.from_utf8([ch])
        i = i + 1
    if tok.size() > 0:
        var ok2 = False; var v2 = _parse_int_token(tok, ok2)
        if not ok2: return
        vals.push_back(v2)
    ok = True

fn _parse_shape_from_type(t: TypeDesc, out info: ShapeInfo, out ok: Bool) -> None:
    ok = False
    info = ShapeInfo()
    var s = t.name
    if not (s.starts_with(String("tensor<")) or s.starts_with(String("buffer<"))): return
    var lt = s.find(String("<")); if lt < 0: return
    var comma = s.find(String(","), lt+1); if comma < 0: return
    var lbr = s.find(String("["), comma+1); if lbr < 0: return
    var rbr = s.find(String("]"), lbr+1); if rbr < 0: return

    var kind = s.starts_with(String("tensor<")) ? String("tensor") : String("buffer")
    var dt = String("")
    var i = lt + 1
    while i < comma:
        dt = dt + String.from_utf8([s.bytes()[i]])
        i = i + 1
    # dims
    var dimtxt = String("")
    i = lbr + 1
    while i < rbr:
        dimtxt = dimtxt + String.from_utf8([s.bytes()[i]])
        i = i + 1
    var dims = List[Int](); var oklist = False
    _parse_int_list(String("[") + dimtxt + String("]"), dims, oklist)
    info.kind = kind; info.dtype = dt; info.dims = dims
    ok = True

fn _dims_to_brackets(dims: List[Int]) -> String:
    var s = String("[")
    for i in range(dims.size()):
        var d = dims[i]
        if d < 0: s = s + String("?")
        else: s = s + String(d)
        if i + 1 < dims.size(): s = s + String(",")
    s = s + String("]")
    return s

fn _replace_dims_in_type(t: TypeDesc, new_dims: List[Int]) -> TypeDesc:
    var info = ShapeInfo(); var ok = False
    _parse_shape_from_type(t, info, ok)
    if not ok: return t
    var head = info.kind == String("tensor") ? String("tensor<") : String("buffer<")
    var out = head + info.dtype + String(",") + _dims_to_brackets(new_dims) + String(">")
    return TypeDesc(out)

# Broadcasting
fn _broadcast_2(a: List[Int], b: List[Int], out dims: List[Int], out ok: Bool) -> None:
    ok = True
    var ra = a.size(); var rb = b.size()
    var r = (ra > rb) ? ra : rb
    var tmp = List[Int]()
    var i = 0
    while i < r:
        var ia = ra - 1 - i
        var ib = rb - 1 - i
        var da = (ia >= 0) ? a[ia] : 1
        var db = (ib >= 0) ? b[ib] : 1
        var d = -2
        if da == db or da == -1: d = db
        elif db == -1: d = da
        elif da == 1: d = db
        elif db == 1: d = da
        elif da < 0 and db < 0: d = -1
        else: ok = False; d = -1
        tmp.push_back(d)
        i = i + 1
    dims = List[Int]()
    i = tmp.size() - 1
    while i >= 0:
        dims.push_back(tmp[i]); i = i - 1

fn _broadcast_many(arrs: List[List[Int]], out dims: List[Int], out ok: Bool) -> None:
    ok = False; dims = List[Int]()
    if arrs.size() == 0: return
    var cur = arrs[0]
    var i = 1
    while i < arrs.size():
        var outd = List[Int](); var ok2 = False
        _broadcast_2(cur, arrs[i], outd, ok2)
        if not ok2: ok = False; dims = cur; return
        cur = outd
        i = i + 1
    ok = True; dims = cur

# Reshape helpers
fn _total_elems(dims: List[Int], out known: Bool) -> Int:
    known = True
    var p = 1
    for i in range(dims.size()):
        var d = dims[i]
        if d < 0: known = False
        else:
            if p >= 0: p = p * d
    return p

fn _infer_reshape(old_dims: List[Int], new_dims: List[Int], out outdims: List[Int], out ok: Bool) -> None:
    ok = True
    outdims = List[Int]()
    var neg = 0; var neg_idx = -1
    for i in range(new_dims.size()):
        if new_dims[i] == -1: neg = neg + 1; neg_idx = i
    if neg == 0:
        outdims = new_dims; return
    if neg > 1:
        ok = False; outdims = new_dims; return
    var known = False; var tot_old = _total_elems(old_dims, known)
    if not known:
        outdims = new_dims; return
    var prod_known = 1
    for i in range(new_dims.size()):
        if i == neg_idx: continue
        var d = new_dims[i]
        if d >= 0: prod_known = prod_known * d
        else: ok = False
    if prod_known == 0: ok = False
    var inferred = (prod_known > 0) ? (tot_old / prod_known) : -1
    var nd = List[Int]()
    for i in range(new_dims.size()):
        if i == neg_idx: nd.push_back(inferred)
        else: nd.push_back(new_dims[i])
    outdims = nd

# Transpose
fn _apply_perm(dims: List[Int], perm: List[Int], out outdims: List[Int], out ok: Bool) -> None:
    ok = False
    outdims = List[Int]()
    if perm.size() != dims.size(): return
    for i in range(perm.size()):
        var p = perm[i]
        if p < 0 or p >= dims.size(): return
    for i in range(perm.size()):
        outdims.push_back(dims[perm[i]])
    ok = True

# Slice: begin[], size[] ; size[i]==-1 means take rest
fn _apply_slice(dims: List[Int], begin: List[Int], size: List[Int], out outdims: List[Int], out ok: Bool) -> None:
    ok = False
    outdims = List[Int]()
    if begin.size() != dims.size() or size.size() != dims.size(): return
    for i in range(dims.size()):
        var d = dims[i]; var b = begin[i]; var s = size[i]
        if d >= 0 and s == -1:
            var bb = (b >= 0) ? b : 0
            var rest = d - bb
            outdims.push_back(rest >= 0 ? rest : -1)
        elif s >= 0:
            outdims.push_back(s)
        else:
            outdims.push_back(-1)
    ok = True

# Axis helpers
fn _norm_axis(axis: Int, rank: Int, out ok: Bool) -> Int:
    ok = True
    var ax = axis
    if ax < 0: ax = ax + rank
    if ax < 0 or ax >= rank: ok = False
    return ax

fn _norm_axes_list(axes: List[Int], rank: Int, out out_axes: List[Int], out ok: Bool) -> None:
    ok = True; out_axes = List[Int]()
    for i in range(axes.size()):
        var o = False; var ax = _norm_axis(axes[i], rank, o)
        if not o: ok = False
        out_axes.push_back(ax)
    # (No dedup removal; reduce semantics may allow repeats or ignore)

# Concat along axis
fn _concat_dims(inputs: List[List[Int]], axis: Int, out outdims: List[Int], out ok: Bool) -> None:
    ok = False; outdims = List[Int]()
    if inputs.size() == 0: return
    var rank = inputs[0].size()
    for i in range(inputs.size()):
        if inputs[i].size() != rank: return
    var o = False; var ax = _norm_axis(axis, rank, o)
    if not o: return
    var dims = List[Int]()
    for d in range(rank):
        if d == ax:
            var s = 0; var known = True
            for i in range(inputs.size()):
                var v = inputs[i][d]
                if v < 0: known = False
                else: s = s + v
            dims.push_back(s if known else -1)
        else:
            var v0 = inputs[0][d]; var okall = True
            for i in range(1, inputs.size()):
                var vi = inputs[i][d]
                if v0 >= 0 and vi >= 0 and v0 != vi: okall = False
            dims.push_back(v0 if okall else -1)
    ok = True; outdims = dims

# Pad: before[], after[]
fn _pad_dims(dims: List[Int], before: List[Int], after: List[Int], out outdims: List[Int], out ok: Bool) -> None:
    ok = False; outdims = List[Int]()
    if before.size() != dims.size() or after.size() != dims.size(): return
    for i in range(dims.size()):
        var d = dims[i]; var b = before[i]; var a = after[i]
        if d >= 0 and b >= 0 and a >= 0: outdims.push_back(d + b + a)
        else: outdims.push_back(-1)
    ok = True

# Reduce over axes; keepdims bool
fn _reduce_dims(dims: List[Int], axes: List[Int], keep: Bool, out outdims: List[Int], out ok: Bool) -> None:
    ok = True; outdims = List[Int]()
    var rank = dims.size()
    var norm = List[Int](); var o = False
    _norm_axes_list(axes, rank, norm, o)
    if not o: ok = False
    var mark = List[Int]()
    for i in range(rank): mark.push_back(0)
    for i in range(norm.size()): mark[norm[i]] = 1
    if keep:
        for i in range(rank):
            if mark[i] == 1: outdims.push_back(1)
            else: outdims.push_back(dims[i])
    else:
        for i in range(rank):
            if mark[i] == 0: outdims.push_back(dims[i])

# Batched matmul: (..., M, K) x (..., K, N) -> (..., M, N) with batch broadcast
fn _batched_matmul(a: List[Int], b: List[Int], out dims: List[Int], out ok: Bool) -> None:
    ok = False; dims = List[Int]()
    if a.size() < 2 or b.size() < 2: return
    var Ma = a[a.size()-2]; var Ka = a[a.size()-1]
    var Kb = b[b.size()-2]; var Nb = b[b.size()-1]
    if Ka >= 0 and Kb >= 0 and Ka != Kb: return
    var batchA = List[Int](); var batchB = List[Int]()
    for i in range(a.size()-2): batchA.push_back(a[i])
    for i in range(b.size()-2): batchB.push_back(b[i])
    var batch = List[Int](); var okb = False
    _broadcast_2(batchA, batchB, batch, okb)
    if not okb: return
    dims = List[Int]()
    for i in range(batch.size()): dims.push_back(batch[i])
    dims.push_back(Ma if Ma >= 0 else -1)
    dims.push_back(Nb if Nb >= 0 else -1)
    ok = True

# Conv2d NHWC with kernel [KH,KW,C,OC], strides and padding
fn _ceil_div(a: Int, b: Int) -> Int:
    if b <= 0: return -1
    if a < 0: return -1
    return (a + b - 1) / b

fn _conv2d_nhwc(in_dims: List[Int], k_dims: List[Int], strides: List[Int], padding: String, out outdims: List[Int], out ok: Bool) -> None:
    ok = False
    outdims = List[Int]()
    if in_dims.size() != 4 or k_dims.size() != 4: return
    var N = in_dims[0]; var H = in_dims[1]; var W = in_dims[2]; var C = in_dims[3]
    var KH = k_dims[0]; var KW = k_dims[1]; var KC = k_dims[2]; var OC = k_dims[3]
    if C >= 0 and KC >= 0 and C != KC: return
    var sH = 1; var sW = 1
    if strides.size() >= 2:
        sH = (strides[0] > 0) ? strides[0] : 1
        sW = (strides[1] > 0) ? strides[1] : 1
    var OH = -1; var OW = -1
    if padding == String("same"):
        OH = _ceil_div(H, sH)
        OW = _ceil_div(W, sW)
    else:
        if H >= 0 and KH >= 0:
            var t = H - KH
            OH = (t >= 0) ? (t / sH + 1) : -1
        if W >= 0 and KW >= 0:
            var t2 = W - KW
            OW = (t2 >= 0) ? (t2 / sW + 1) : -1
    outdims.push_back(N); outdims.push_back(OH); outdims.push_back(OW); outdims.push_back(OC)
    ok = True

# Reduce window / Pool math
fn _eff_kernel(k: Int, dil: Int) -> Int:
    var d = (dil > 0) ? dil : 1
    var kk = (k > 0) ? k : -1
    if kk < 0: return -1
    return (kk - 1) * d + 1

fn _window_out_dim(D: Int, K: Int, S: Int, Dil: Int, pad_b: Int, pad_a: Int, padding: String) -> Int:
    var effK = _eff_kernel(K, Dil)
    var s = (S > 0) ? S : 1
    if padding == String("same"):
        return _ceil_div(D, s)
    # valid or explicit pads
    var Db = (D >= 0) ? D : -1
    if Db < 0 or effK < 0: return -1
    var total = Db + ((pad_b >= 0) ? pad_b : 0) + ((pad_a >= 0) ? pad_a : 0)
    var t = total - effK
    if t < 0: return -1
    return t / s + 1

fn _reduce_window_dims(in_dims: List[Int], window: List[Int], strides: List[Int], dilation: List[Int], pad_before: List[Int], pad_after: List[Int], padding: String, out outdims: List[Int], out ok: Bool) -> None:
    ok = False; outdims = List[Int]()
    var rank = in_dims.size()
    if window.size() != rank or strides.size() != rank or dilation.size() != rank: return
    var pb = pad_before; var pa = pad_after
    if pb.size() == 0:
        for i in range(rank): pb.push_back(0)
    if pa.size() == 0:
        for i in range(rank): pa.push_back(0)
    var dims = List[Int]()
    for i in range(rank):
        dims.push_back(_window_out_dim(in_dims[i], window[i], strides[i], dilation[i], pb[i], pa[i], padding))
    ok = True; outdims = dims

fn _pool2d_nhwc(in_dims: List[Int], window: List[Int], strides: List[Int], dilation: List[Int], pad_before: List[Int], pad_after: List[Int], padding: String, out outdims: List[Int], out ok: Bool) -> None:
    ok = False; outdims = List[Int]()
    if in_dims.size() != 4: return
    var N = in_dims[0]; var H = in_dims[1]; var W = in_dims[2]; var C = in_dims[3]
    var pb = pad_before; var pa = pad_after
    if pb.size() < 2: pb = [0,0]
    if pa.size() < 2: pa = [0,0]
    var winH = (window.size() >= 2) ? window[0] : -1
    var winW = (window.size() >= 2) ? window[1] : -1
    var sH = (strides.size() >= 2) ? strides[0] : 1
    var sW = (strides.size() >= 2) ? strides[1] : 1
    var dH = (dilation.size() >= 2) ? dilation[0] : 1
    var dW = (dilation.size() >= 2) ? dilation[1] : 1
    var OH = _window_out_dim(H, winH, sH, dH, pb[0], pa[0], padding)
    var OW = _window_out_dim(W, winW, sW, dW, pb[1], pa[1], padding)
    outdims.push_back(N); outdims.push_back(OH); outdims.push_back(OW); outdims.push_back(C)
    ok = True

# -----------------------------
# Attr helpers
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

fn _get_int_attr(op: Op, key: String, out val: Int, out has: Bool) -> None:
    var s = _attr_value(op, key, has)
    if not has: val = 0; return
    var ok = False; var v = _parse_int_token(s, ok)
    has = ok; val = v

fn _get_int_list_attr(op: Op, key: String, out vals: List[Int], out has: Bool) -> None:
    var s = _attr_value(op, key, has)
    if not has:
        vals = List[Int](); return
    var ok = False; var xs = List[Int]()
    _parse_int_list(s, xs, ok)
    has = ok
    vals = xs

# -----------------------------
# Printer (for debugging)
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
                            s = s + self._value(op.operands[pi].value)
                            if pi + 1 < op.operands.size(): s = s + String(", ")
                        s = s + String(")")
                    if op.results.size() > 0:
                        s = s + String(" -> ")
                        for ri2 in range(op.results.size()):
                            s = s + self._value(op.results[ri2].value)
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

# -----------------------------
# Shape propagation pass
# -----------------------------

struct ShapePropagation:
    fn _annotate_shape(self, op: Op, dims: List[Int]) -> Op:
        op.attrs.push_back(String("shape=") + _dims_to_brackets(dims))
        return op

    fn _set_result_shape(self, op: Op, dims: List[Int]) -> Op:
        if op.results.size() == 0: return op
        var r = op.results[0].value
        var info = ShapeInfo(); var ok = False
        _parse_shape_from_type(r.typ, info, ok)
        if ok:
            r.typ = _replace_dims_in_type(r.typ, dims)
            op.results[0] = Result(r)
        return op

    fn _handle_elemwise(self, b: Block, idx: Int) -> None:
        var op = b.ops[idx]
        if op.operands.size() < 1 or op.results.size() == 0: return
        var arrs = List[List[Int]]()
        for i in range(op.operands.size()):
            var info = ShapeInfo(); var ok = False
            _parse_shape_from_type(op.operands[i].value.typ, info, ok)
            if not ok: return
            arrs.push_back(info.dims)
        var dims = List[Int](); var okb = False
        _broadcast_many(arrs, dims, okb)
        if okb:
            op = self._set_result_shape(op, dims)
            op = self._annotate_shape(op, dims)
            b.ops[idx] = op
        else:
            op.attrs.push_back(String("shape.error=broadcast")); b.ops[idx] = op

    fn _handle_reshape(self, b: Block, idx: Int) -> None:
        var op = b.ops[idx]
        if op.operands.size() < 1 or op.results.size() == 0: return
        var inp = op.operands[0].value.typ
        var iinfo = ShapeInfo(); var ok0 = False
        _parse_shape_from_type(inp, iinfo, ok0); if not ok0: return
        var has = False; var s = _attr_value(op, String("shape"), has)
        if not has: s = _attr_value(op, String("newshape"), has)
        if not has: return
        var lst = List[Int](); var ok = False
        _parse_int_list(s, lst, ok); if not ok: return
        var outd = List[Int](); var ok2 = False
        _infer_reshape(iinfo.dims, lst, outd, ok2)
        op = self._set_result_shape(op, outd)
        op = self._annotate_shape(op, outd)
        if not ok2: op.attrs.push_back(String("shape.warn=reshape_incomplete"))
        b.ops[idx] = op

    fn _handle_transpose(self, b: Block, idx: Int) -> None:
        var op = b.ops[idx]
        if op.operands.size() < 1 or op.results.size() == 0: return
        var inp = op.operands[0].value.typ
        var iinfo = ShapeInfo(); var ok0 = False
        _parse_shape_from_type(inp, iinfo, ok0); if not ok0: return
        var perm = List[Int](); var has = False
        _get_int_list_attr(op, String("perm"), perm, has)
        if not has: return
        var outd = List[Int](); var ok = False
        _apply_perm(iinfo.dims, perm, outd, ok)
        if not ok: op.attrs.push_back(String("shape.error=bad_perm"))
        else:
            op = self._set_result_shape(op, outd)
            op = self._annotate_shape(op, outd)
        b.ops[idx] = op

    fn _handle_slice(self, b: Block, idx: Int) -> None:
        var op = b.ops[idx]
        if op.operands.size() < 1 or op.results.size() == 0: return
        var inp = op.operands[0].value.typ
        var iinfo = ShapeInfo(); var ok0 = False
        _parse_shape_from_type(inp, iinfo, ok0); if not ok0: return
        var begin = List[Int](); var size = List[Int](); var hb = False; var hs = False
        _get_int_list_attr(op, String("begin"), begin, hb)
        _get_int_list_attr(op, String("size"), size, hs)
        if not hb or not hs: return
        var outd = List[Int](); var ok = False
        _apply_slice(iinfo.dims, begin, size, outd, ok)
        if ok:
            op = self._set_result_shape(op, outd)
            op = self._annotate_shape(op, outd)
        else:
            op.attrs.push_back(String("shape.error=bad_slice"))
        b.ops[idx] = op

    fn _handle_broadcast_to(self, b: Block, idx: Int) -> None:
        var op = b.ops[idx]
        if op.operands.size() < 1 or op.results.size() == 0: return
        var target = List[Int](); var has = False
        _get_int_list_attr(op, String("shape"), target, has)
        if not has: return
        op = self._set_result_shape(op, target)
        op = self._annotate_shape(op, target)
        b.ops[idx] = op

    fn _handle_concat(self, b: Block, idx: Int) -> None:
        var op = b.ops[idx]
        if op.operands.size() < 1 or op.results.size() == 0: return
        var axis = 0; var has = False
        _get_int_attr(op, String("axis"), axis, has)
        if not has: axis = 0
        var arrs = List[List[Int]]()
        for i in range(op.operands.size()):
            var info = ShapeInfo(); var ok = False
            _parse_shape_from_type(op.operands[i].value.typ, info, ok); if not ok: return
            arrs.push_back(info.dims)
        var outd = List[Int](); var ok2 = False
        _concat_dims(arrs, axis, outd, ok2)
        if ok2:
            op = self._set_result_shape(op, outd)
            op = self._annotate_shape(op, outd)
        else:
            op.attrs.push_back(String("shape.error=concat"))
        b.ops[idx] = op

    fn _handle_pad(self, b: Block, idx: Int) -> None:
        var op = b.ops[idx]
        if op.operands.size() < 1 or op.results.size() == 0: return
        var info = ShapeInfo(); var ok0 = False
        _parse_shape_from_type(op.operands[0].value.typ, info, ok0); if not ok0: return
        var before = List[Int](); var after = List[Int](); var hb = False; var ha = False
        _get_int_list_attr(op, String("pad_before"), before, hb)
        _get_int_list_attr(op, String("pad_after"), after, ha)
        if not hb or not ha: return
        var outd = List[Int](); var ok = False
        _pad_dims(info.dims, before, after, outd, ok)
        if ok:
            op = self._set_result_shape(op, outd)
            op = self._annotate_shape(op, outd)
        else:
            op.attrs.push_back(String("shape.error=pad"))
        b.ops[idx] = op

    fn _handle_reduce(self, b: Block, idx: Int) -> None:
        var op = b.ops[idx]
        if op.operands.size() < 1 or op.results.size() == 0: return
        var info = ShapeInfo(); var ok0 = False
        _parse_shape_from_type(op.operands[0].value.typ, info, ok0); if not ok0: return
        var axes = List[Int](); var has = False
        _get_int_list_attr(op, String("axes"), axes, has)
        var keep = False; var hask = False
        var keep_s = _attr_value(op, String("keepdims"), hask)
        if hask and (keep_s == String("1") or keep_s == String("true")): keep = True
        var norm = List[Int](); var okN = False
        _norm_axes_list(axes, info.dims.size(), norm, okN)
        if not okN: op.attrs.push_back(String("shape.error=axes")); b.ops[idx] = op; return
        var outd = List[Int](); var ok = False
        _reduce_dims(info.dims, norm, keep, outd, ok)
        if ok:
            op = self._set_result_shape(op, outd)
            op = self._annotate_shape(op, outd)
        else:
            op.attrs.push_back(String("shape.error=reduce"))
        b.ops[idx] = op

    fn _handle_matmul(self, b: Block, idx: Int) -> None:
        var op = b.ops[idx]
        if op.operands.size() < 2 or op.results.size() == 0: return
        var a = op.operands[0].value.typ; var b2 = op.operands[1].value.typ
        var ia = ShapeInfo(); var ib = ShapeInfo(); var ok1 = False; var ok2 = False
        _parse_shape_from_type(a, ia, ok1); _parse_shape_from_type(b2, ib, ok2)
        if not ok1 or not ok2: return
        var outd = List[Int](); var ok = False
        _batched_matmul(ia.dims, ib.dims, outd, ok)
        if ok:
            op = self._set_result_shape(op, outd)
            op = self._annotate_shape(op, outd)
        else:
            op.attrs.push_back(String("shape.error=matmul"))
        b.ops[idx] = op

    fn _handle_conv2d(self, b: Block, idx: Int) -> None:
        var op = b.ops[idx]
        if op.operands.size() < 2 or op.results.size() == 0: return
        var inp = op.operands[0].value.typ; var ker = op.operands[1].value.typ
        var ii = ShapeInfo(); var ki = ShapeInfo(); var ok1 = False; var ok2 = False
        _parse_shape_from_type(inp, ii, ok1); _parse_shape_from_type(ker, ki, ok2)
        if not ok1 or not ok2: return
        var strides = List[Int](); var has = False
        _get_int_list_attr(op, String("strides"), strides, has)
        var pad_s = _attr_value(op, String("padding"), has)
        if not has: pad_s = String("valid")
        var outd = List[Int](); var ok = False
        _conv2d_nhwc(ii.dims, ki.dims, strides, pad_s, outd, ok)
        if ok:
            op = self._set_result_shape(op, outd)
            op = self._annotate_shape(op, outd)
        else:
            op.attrs.push_back(String("shape.error=conv2d"))
        b.ops[idx] = op

    fn _handle_gather(self, b: Block, idx: Int) -> None:
        var op = b.ops[idx]
        if op.operands.size() < 2 or op.results.size() == 0: return
        var params = op.operands[0].value.typ; var indices = op.operands[1].value.typ
        var ip = ShapeInfo(); var ii = ShapeInfo(); var ok1 = False; var ok2 = False
        _parse_shape_from_type(params, ip, ok1); _parse_shape_from_type(indices, ii, ok2)
        if not ok1 or not ok2: return
        var axis = 0; var has = False
        _get_int_attr(op, String("axis"), axis, has)
        var o = False; var ax = _norm_axis(axis, ip.dims.size(), o)
        if not o: op.attrs.push_back(String("shape.error=gather_axis")); b.ops[idx] = op; return
        var outd = List[Int]()
        for i in range(ax): outd.push_back(ip.dims[i])
        for i in range(ii.dims.size()): outd.push_back(ii.dims[i])
        for i in range(ax+1, ip.dims.size()): outd.push_back(ip.dims[i])
        op = self._set_result_shape(op, outd)
        op = self._annotate_shape(op, outd)
        b.ops[idx] = op

    fn _handle_scatter(self, b: Block, idx: Int) -> None:
        var op = b.ops[idx]
        # scatter(params, indices, updates, axis) → same shape as params
        if op.operands.size() < 3 or op.results.size() == 0: return
        var params = op.operands[0].value.typ
        var ip = ShapeInfo(); var ok1 = False
        _parse_shape_from_type(params, ip, ok1); if not ok1: return
        op = self._set_result_shape(op, ip.dims)
        op = self._annotate_shape(op, ip.dims)
        b.ops[idx] = op

    fn _handle_gather_nd(self, b: Block, idx: Int) -> None:
        var op = b.ops[idx]
        # gather_nd(params, indices[... , K]) → indices[... ] + params[K:]
        if op.operands.size() < 2 or op.results.size() == 0: return
        var params = op.operands[0].value.typ; var indices = op.operands[1].value.typ
        var ip = ShapeInfo(); var ii = ShapeInfo(); var ok1 = False; var ok2 = False
        _parse_shape_from_type(params, ip, ok1); _parse_shape_from_type(indices, ii, ok2)
        if not ok1 or not ok2 or ii.dims.size() == 0: return
        var K = ii.dims[ii.dims.size()-1]
        if K < 0: K = ip.dims.size()  # unknown → assume full indexing
        if K > ip.dims.size():
            op.attrs.push_back(String("shape.error=gather_nd_K")); b.ops[idx] = op; return
        var outd = List[Int]()
        for i in range(ii.dims.size()-1): outd.push_back(ii.dims[i])
        for i in range(K, ip.dims.size()): outd.push_back(ip.dims[i])
        op = self._set_result_shape(op, outd)
        op = self._annotate_shape(op, outd)
        b.ops[idx] = op

    fn _handle_reduce_window(self, b: Block, idx: Int) -> None:
        var op = b.ops[idx]
        if op.operands.size() < 1 or op.results.size() == 0: return
        var inp = op.operands[0].value.typ
        var ii = ShapeInfo(); var ok0 = False
        _parse_shape_from_type(inp, ii, ok0); if not ok0: return
        var win = List[Int](); var s = List[Int](); var dil = List[Int]()
        var pb = List[Int](); var pa = List[Int]()
        var hw = False; var hs = False; var hd = False; var hpb = False; var hpa = False
        _get_int_list_attr(op, String("window"), win, hw)
        _get_int_list_attr(op, String("strides"), s, hs)
        _get_int_list_attr(op, String("dilation"), dil, hd)
        _get_int_list_attr(op, String("pad_before"), pb, hpb)
        _get_int_list_attr(op, String("pad_after"), pa, hpa)
        var pad_s = _attr_value(op, String("padding"), hw)  # reuse flag
        if not hs: s = [1 for _ in range(ii.dims.size())]
        if not hd: dil = [1 for _ in range(ii.dims.size())]
        if not hw: op.attrs.push_back(String("shape.warn=no_window"))
        if pad_s.size() == 0: pad_s = String("valid")
        var outd = List[Int](); var ok = False
        _reduce_window_dims(ii.dims, win, s, dil, pb, pa, pad_s, outd, ok)
        if ok:
            op = self._set_result_shape(op, outd)
            op = self._annotate_shape(op, outd)
        else:
            op.attrs.push_back(String("shape.error=reduce_window"))
        b.ops[idx] = op

    fn _handle_pool2d(self, b: Block, idx: Int) -> None:
        var op = b.ops[idx]
        if op.operands.size() < 1 or op.results.size() == 0: return
        var inp = op.operands[0].value.typ
        var ii = ShapeInfo(); var ok0 = False
        _parse_shape_from_type(inp, ii, ok0); if not ok0: return
        var win = List[Int](); var s = List[Int](); var dil = List[Int]()
        var pb = List[Int](); var pa = List[Int]()
        var hw = False; var hs = False; var hd = False; var hpb = False; var hpa = False
        _get_int_list_attr(op, String("window"), win, hw)
        _get_int_list_attr(op, String("strides"), s, hs)
        _get_int_list_attr(op, String("dilation"), dil, hd)
        _get_int_list_attr(op, String("pad_before"), pb, hpb)
        _get_int_list_attr(op, String("pad_after"), pa, hpa)
        var pad_s = _attr_value(op, String("padding"), hw)  # reuse flag
        if pad_s.size() == 0: pad_s = String("same")  # typical pooling default
        var outd = List[Int](); var ok = False
        _pool2d_nhwc(ii.dims, win, s, dil, pb, pa, pad_s, outd, ok)
        if ok:
            op = self._set_result_shape(op, outd)
            op = self._annotate_shape(op, outd)
        else:
            op.attrs.push_back(String("shape.error=pool2d"))
        b.ops[idx] = op

    fn run_on_block(self, b: Block) -> Block:
        var i = 0
        while i < b.ops.size():
            var op = b.ops[i]
            var n = op.name

            # Elementwise families
            if n == String("ll.add") or n == String("ll.sub") or n == String("ll.mul") or n == String("ll.div") or
               n == String("mid.add") or n == String("mid.sub") or n == String("mid.mul") or n == String("mid.div") or
               n.find(String("hl.add")) >= 0 or n.find(String("hl.mul")) >= 0:
                self._handle_elemwise(b, i)

            elif n.find(String("reshape")) >= 0:
                self._handle_reshape(b, i)

            elif n.find(String("transpose")) >= 0:
                self._handle_transpose(b, i)

            elif n.find(String("slice")) >= 0:
                self._handle_slice(b, i)

            elif n.find(String("broadcast_to")) >= 0:
                self._handle_broadcast_to(b, i)

            elif n.find(String("concat")) >= 0:
                self._handle_concat(b, i)

            elif n.find(String("pad")) >= 0 and n.find(String("pad")) == n.rfind(String("pad")):
                self._handle_pad(b, i)

            elif n.find(String("reduce_window")) >= 0:
                self._handle_reduce_window(b, i)

            elif n.find(String("pool2d")) >= 0:
                self._handle_pool2d(b, i)

            elif n.find(String("reduce")) >= 0:
                self._handle_reduce(b, i)

            elif n.find(String("matmul")) >= 0:
                self._handle_matmul(b, i)

            elif n.find(String("conv2d")) >= 0:
                self._handle_conv2d(b, i)

            elif n.find(String("gather_nd")) >= 0:
                self._handle_gather_nd(b, i)

            elif n.find(String("gather")) >= 0:
                self._handle_gather(b, i)

            elif n.find(String("scatter")) >= 0:
                self._handle_scatter(b, i)

            i = i + 1
        return b

    fn run_on_function(self, f: Function) -> Function:
        for bi in range(f.blocks.size()):
            f.blocks[bi] = self.run_on_block(f.blocks[bi])
        return f

    fn run_on_module(self, m: Module) -> Module:
        var out = Module(m.name + String("_shapes"))
        for fi in range(m.functions.size()):
            var f = m.functions[fi]
            var nf = Function(f.name)
            for ai in range(f.arg_types.size()): nf.arg_types.push_back(f.arg_types[ai])
            for ri in range(f.ret_types.size()): nf.ret_types.push_back(f.ret_types[ri])
            for bi in range(f.blocks.size()): nf.blocks.push_back(f.blocks[bi])
            out.functions.push_back(self.run_on_function(nf))
        return out

# -----------------------------
# Self-test
# -----------------------------

struct IdGen:
    var next_id: Int
    fn __init__(out self): self.next_id = 0
    fn fresh(self) -> Int:
        var r = self.next_id
        self.next_id = self.next_id + 1
        return r

fn _tensor_i64(dims: List[Int]) -> TypeDesc:
    var s = String("tensor<i64,") + _dims_to_brackets(dims) + String(">")
    return TypeDesc(s)

fn _demo_module() -> Module:
    var m = Module(String("demo_shapes_adv2"))
    var f = Function(String("main"))
    var b = Block(String("entry"))
    var idg = IdGen()

    # Elementwise N-ary: [2,3] + [1,3] + [2,1] -> [2,3]
    var A = Value(idg.fresh(), _tensor_i64([2,3]))
    var B = Value(idg.fresh(), _tensor_i64([1,3]))
    var C = Value(idg.fresh(), _tensor_i64([2,1]))
    var add3 = Op(String("ll.add"), Location(String("d"), 1, 1))
    var r0 = Value(idg.fresh(), _tensor_i64([-1,-1])); add3.results.push_back(Result(r0))
    add3.operands.push_back(Operand(A)); add3.operands.push_back(Operand(B)); add3.operands.push_back(Operand(C))

    # Concat axis=-1: [2,3] ++ [2,5] -> [2,8]
    var X = Value(idg.fresh(), _tensor_i64([2,3]))
    var Y = Value(idg.fresh(), _tensor_i64([2,5]))
    var cat = Op(String("ll.concat"), Location(String("d"), 1, 2))
    var rcat = Value(idg.fresh(), _tensor_i64([-1,-1])); cat.results.push_back(Result(rcat))
    cat.operands.push_back(Operand(X)); cat.operands.push_back(Operand(Y)); cat.attrs.push_back(String("axis=-1"))

    # gather_nd: params [2,3,4], indices [5,2] -> [5,4]
    var GP = Value(idg.fresh(), _tensor_i64([2,3,4]))
    var GI = Value(idg.fresh(), _tensor_i64([5,2]))
    var gnd = Op(String("ll.gather_nd"), Location(String("d"), 1, 3))
    var rgnd = Value(idg.fresh(), _tensor_i64([-1,-1])); gnd.results.push_back(Result(rgnd))
    gnd.operands.push_back(Operand(GP)); gnd.operands.push_back(Operand(GI))

    # scatter(axis=-2): params [2,3,4] -> out [2,3,4]
    var SP = Value(idg.fresh(), _tensor_i64([2,3,4]))
    var SI = Value(idg.fresh(), _tensor_i64([2,2]))
    var SU = Value(idg.fresh(), _tensor_i64([2,2,4]))
    var sc = Op(String("ll.scatter"), Location(String("d"), 1, 4))
    var rsc = Value(idg.fresh(), _tensor_i64([-1,-1,-1])); sc.results.push_back(Result(rsc))
    sc.operands.push_back(Operand(SP)); sc.operands.push_back(Operand(SI)); sc.operands.push_back(Operand(SU))
    sc.attrs.push_back(String("axis=-2"))

    # reduce_window 2D: in [32,32], window=[3,3], strides=[2,2], dilation=[1,1], valid -> [15,15]
    var RW = Value(idg.fresh(), _tensor_i64([32,32]))
    var rw = Op(String("ll.reduce_window"), Location(String("d"), 1, 5))
    var rrw = Value(idg.fresh(), _tensor_i64([-1,-1])); rw.results.push_back(Result(rrw))
    rw.operands.push_back(Operand(RW))
    rw.attrs.push_back(String("window=[3,3]")); rw.attrs.push_back(String("strides=[2,2]"))
    rw.attrs.push_back(String("dilation=[1,1]")); rw.attrs.push_back(String("padding=valid"))

    # pool2d NHWC same: in [1,32,32,3], window=[3,3], strides=[2,2] -> [1,16,16,3]
    var PI = Value(idg.fresh(), _tensor_i64([1,32,32,3]))
    var pool = Op(String("ll.pool2d"), Location(String("d"), 1, 6))
    var rpool = Value(idg.fresh(), _tensor_i64([-1,-1,-1,-1])); pool.results.push_back(Result(rpool))
    pool.operands.push_back(Operand(PI))
    pool.attrs.push_back(String("window=[3,3]")); pool.attrs.push_back(String("strides=[2,2]"))
    pool.attrs.push_back(String("dilation=[1,1]")); pool.attrs.push_back(String("padding=same"))

    var ret = Op(String("ll.ret"), Location(String("d"), 1, 7))
    ret.operands.push_back(Operand(rpool))

    b.ops.push_back(add3); b.ops.push_back(cat); b.ops.push_back(gnd); b.ops.push_back(sc); b.ops.push_back(rw); b.ops.push_back(pool); b.ops.push_back(ret)

    f.blocks.push_back(b); m.functions.push_back(f)
    return m

fn _self_test_shape_propagation() -> Bool:
    var m = _demo_module()
    var sp = ShapePropagation()
    var out = sp.run_on_module(m)

    var pr = Printer()
    var txt = pr.print(out)

    var ok = True
    if txt.find(String("ll.add {shape=[2,3]}")) < 0: ok = False
    if txt.find(String("ll.concat")) < 0 or txt.find(String("shape=[2,8]")) < 0: ok = False
    if txt.find(String("ll.gather_nd")) < 0 or txt.find(String("shape=[5,4]")) < 0: ok = False
    if txt.find(String("ll.scatter")) < 0 or txt.find(String("shape=[2,3,4]")) < 0: ok = False
    if txt.find(String("ll.reduce_window")) < 0 or txt.find(String("shape=[15,15]")) < 0: ok = False
    if txt.find(String("ll.pool2d")) < 0 or txt.find(String("shape=[1,16,16,3]")) < 0: ok = False

    if ok: print(String("OK"))
    else: print(String("FAIL"))
    return ok
 
