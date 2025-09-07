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
# Project: momijo.ir.dialects
# File: src/momijo/ir/dialects/shape_inference.mojo

struct TypeDesc:
    var name: String
fn __init__(out self, name: String) -> None: self.name = name
fn __copyinit__(out self, other: Self) -> None:
        self.name = other.name
fn __moveinit__(out self, deinit other: Self) -> None:
        self.name = other.name
fn t_i64() -> TypeDesc: return TypeDesc(String("i64"))
fn t_f64() -> TypeDesc: return TypeDesc(String("f64"))
fn t_bool() -> TypeDesc: return TypeDesc(String("bool"))
fn t_invalid() -> TypeDesc: return TypeDesc(String("invalid"))
fn is_i64(t: TypeDesc) -> Bool: return t.name == String("i64")
fn is_f64(t: TypeDesc) -> Bool: return t.name == String("f64")
fn is_bool(t: TypeDesc) -> Bool: return t.name == String("bool")
fn is_numeric(t: TypeDesc) -> Bool: return is_i64(t) or is_f64(t)

enum DType(Int):
    Invalid = 0
    I64 = 1
    F64 = 2
    Bool = 3
fn dtype_name(dt: DType) -> String:
    if dt == DType.I64: return String("i64")
    if dt == DType.F64: return String("f64")
    if dt == DType.Bool: return String("bool")
    return String("invalid")
fn dtype_to_scalar(dt: DType) -> TypeDesc:
    if dt == DType.I64: return t_i64()
    if dt == DType.F64: return t_f64()
    if dt == DType.Bool: return t_bool()
    return t_invalid()
fn dtype_of_scalar(t: TypeDesc) -> DType:
    if is_i64(t): return DType.I64
    if is_f64(t): return DType.F64
    if is_bool(t): return DType.Bool
    return DType.Invalid

struct Shape:
    var dims: List[Int]
fn __init__(out self) -> None: self.dims = List[Int]()
fn rank(self) -> Int: return self.dims.size()
fn push(self, d: Int) -> None: self.dims.push_back(d)
fn to_string(self) -> String:
        var s = String("[")
        for i in range(self.dims.size()):
            s = s + String(self.dims[i])
            if i + 1 < self.dims.size(): s = s + String(", ")
        s = s + String("]")
        return s
fn __copyinit__(out self, other: Self) -> None:
        self.dims = other.dims
fn __moveinit__(out self, deinit other: Self) -> None:
        self.dims = other.dims
fn tensor_type(dtype: DType, shape: Shape) -> TypeDesc:
    var s = String("tensor<") + dtype_name(dtype) + String(",") + shape.to_string() + String(">")
    return TypeDesc(s)
fn is_tensor(t: TypeDesc) -> Bool:
    return t.name.starts_with(String("tensor<"))
fn _parse_between(s: String, start_ch: UInt8, end_ch: UInt8, out ok: Bool) -> String:
    var i = 0
    var depth = 0
    var start_idx = -1
    var end_idx = -1
    ok = False
    while i < s.size():
        var ch = s.bytes()[i]
        if ch == start_ch:
            if depth == 0: start_idx = i + 1
            depth = depth + 1
        elif ch == end_ch:
            depth = depth - 1
            if depth == 0:
                end_idx = i
                ok = True
                break
        i = i + 1
    if not ok or start_idx < 0 or end_idx < 0 or end_idx <= start_idx:
        return String("")
    var out_s = String("")
    var j = start_idx
    while j < end_idx:
        out_s = out_s + String.from_utf8([s.bytes()[j]])
        j = j + 1
    return out_s
fn tensor_unpack(t: TypeDesc, out ok: Bool, out out_dtype: DType, out out_shape: Shape) -> None:
    ok = False
    out_dtype = DType.Invalid
    out_shape = Shape()
    if not is_tensor(t): return
    var inner = _parse_between(t.name, 60, 62, ok)  # '<','>'
    if not ok: return
    var comma = inner.find(String(","))
    if comma < 0: return

    var dts = String("")
    var i = 0
    while i < comma:
        dts = dts + String.from_utf8([inner.bytes()[i]])
        i = i + 1
    dts = dts.strip()

    var rest = String("")
    i = comma + 1
    while i < inner.size():
        rest = rest + String.from_utf8([inner.bytes()[i]])
        i = i + 1
    rest = rest.strip()

    if dts == String("i64"): out_dtype = DType.I64
    elif dts == String("f64"): out_dtype = DType.F64
    elif dts == String("bool"): out_dtype = DType.Bool
    else: out_dtype = DType.Invalid

    var ok2 = False
    var ss = _parse_between(rest, 91, 93, ok2)  # '[',']'
    if not ok2: return

    var cur = String("")
    for k in range(ss.size() + 1):
        var is_end = k == ss.size()
        var ch = UInt8(0)
        if not is_end: ch = ss.bytes()[k]
        if is_end or ch == 44:  # ','
            var tok = cur.strip()
            if tok.size() > 0:
                var sign = 1
                var idx = 0
                if tok.size() > 0 and tok.bytes()[0] == 45:  # '-'
                    sign = -1
                    idx = 1
                var num = 0
                while idx < tok.size():
                    var d = tok.bytes()[idx] - UInt8(48)
                    num = num * 10 + Int(d)
                    idx = idx + 1
                out_shape.push(sign * num)
            cur = String("")
        else:
            cur = cur + String.from_utf8([ch])
    ok = True

# -----------------------------
# Utilities: promotion & broadcast
# -----------------------------
fn promote_numeric(a: TypeDesc, b: TypeDesc) -> TypeDesc:
    if is_f64(a) or is_f64(b): return t_f64()
    if is_i64(a) and is_i64(b): return t_i64()
    return t_invalid()

struct InferShapeResult:
    var ok: Bool
    var shape: Shape
    var message: String
fn __init__(out self, ok: Bool = True, shape: Shape = Shape(), message: String = String("")):
        self.ok = ok
        self.shape = shape
        self.message = message
fn __copyinit__(out self, other: Self) -> None:
        self.ok = other.ok
        self.shape = other.shape
        self.message = other.message
fn __moveinit__(out self, deinit other: Self) -> None:
        self.ok = other.ok
        self.shape = other.shape
        self.message = other.message
fn _broadcast_dim(a: Int, b: Int, out ok: Bool) -> Int:
    ok = True
    if a == -1 and b == -1: return -1
    if a == -1: return b
    if b == -1: return a
    if a == 1: return b
    if b == 1: return a
    if a == b: return a
    ok = False
    return -2
fn broadcast_shapes(a: Shape, b: Shape) -> InferShapeResult:
    var ra = a.rank(); var rb = b.rank()
    var r = ra; if rb > r: r = rb
    var rev = Shape()
    var i = 0; var all_ok = True
    while i < r:
        var da = 1; var db = 1
        if ra - 1 - i >= 0: da = a.dims[ra - 1 - i]
        if rb - 1 - i >= 0: db = b.dims[rb - 1 - i]
        var okd = False
        var d = _broadcast_dim(da, db, okd)
        if not okd: all_ok = False
        rev.dims.push_back(d)
        i = i + 1
    # reverse
    var out = Shape()
    var j = rev.rank() - 1
    while j >= 0:
        out.dims.push_back(rev.dims[j])
        j = j - 1
    if not all_ok: return InferShapeResult(False, out, String("Shape broadcast failed"))
    return InferShapeResult(True, out, String(""))

# -----------------------------
# Inference results
# -----------------------------

struct InferTypeResult:
    var ok: Bool
    var typ: TypeDesc
    var message: String
fn __init__(out self, ok: Bool = True, typ: TypeDesc = TypeDesc(String("")), message: String = String("")):
        self.ok = ok
        self.typ = typ
        self.message = message
fn __copyinit__(out self, other: Self) -> None:
        self.ok = other.ok
        self.typ = other.typ
        self.message = other.message
fn __moveinit__(out self, deinit other: Self) -> None:
        self.ok = other.ok
        self.typ = other.typ
        self.message = other.message
# -----------------------------
# Unary ops (pass-through shape/type)
# -----------------------------
fn infer_unary_passthrough(x: TypeDesc) -> InferTypeResult:
    if not is_tensor(x):
        if is_numeric(x) or is_bool(x):
            return InferTypeResult(True, x)
        return InferTypeResult(False, t_invalid(), String("Unsupported unary scalar"))
    var okT = False; var dt = DType.Invalid; var sh = Shape()
    tensor_unpack(x, okT, dt, sh)
    if not okT: return InferTypeResult(False, t_invalid(), String("Bad tensor encoding"))
    return InferTypeResult(True, tensor_type(dt, sh))

# -----------------------------
# Binary numeric ops (elementwise)
# -----------------------------
fn infer_binary_numeric(lhs: TypeDesc, rhs: TypeDesc) -> InferTypeResult:
    var ls = is_tensor(lhs)
    var rs = is_tensor(rhs)

    if not ls and not rs:
        if is_numeric(lhs) and is_numeric(rhs):
            return InferTypeResult(True, promote_numeric(lhs, rhs))
        return InferTypeResult(False, t_invalid(), String("Non-numeric scalars"))

    var okL = False; var okR = False
    var dtL = DType.Invalid; var dtR = DType.Invalid
    var shL = Shape(); var shR = Shape()
    if ls: tensor_unpack(lhs, okL, dtL, shL)
    if rs: tensor_unpack(rhs, okR, dtR, shR)
    if ls and (not okL): return InferTypeResult(False, t_invalid(), String("Bad lhs tensor"))
    if rs and (not okR): return InferTypeResult(False, t_invalid(), String("Bad rhs tensor"))

    if ls and rs:
        var p = promote_numeric(dtype_to_scalar(dtL), dtype_to_scalar(dtR))
        var out_dt = dtype_of_scalar(p)
        var br = broadcast_shapes(shL, shR)
        if not br.ok: return InferTypeResult(False, t_invalid(), br.message)
        return InferTypeResult(True, tensor_type(out_dt, br.shape))

    if ls and (not rs):
        if not is_numeric(rhs): return InferTypeResult(False, t_invalid(), String("RHS not numeric scalar"))
        var p2 = promote_numeric(dtype_to_scalar(dtL), rhs)
        return InferTypeResult(True, tensor_type(dtype_of_scalar(p2), shL))

    if (not ls) and rs:
        if not is_numeric(lhs): return InferTypeResult(False, t_invalid(), String("LHS not numeric scalar"))
        var p3 = promote_numeric(lhs, dtype_to_scalar(dtR))
        return InferTypeResult(True, tensor_type(dtype_of_scalar(p3), shR))

    return InferTypeResult(False, t_invalid(), String("Unsupported combination"))

# -----------------------------
# Comparisons -> bool (elementwise)
# -----------------------------
fn infer_compare(lhs: TypeDesc, rhs: TypeDesc) -> InferTypeResult:
    var r = infer_binary_numeric(lhs, rhs)
    if not r.ok: return r
    if is_tensor(r.typ):
        var okT = False; var dt = DType.Invalid; var sh = Shape()
        tensor_unpack(r.typ, okT, dt, sh)
        if not okT: return InferTypeResult(False, t_invalid(), String("Bad tensor after compare"))
        return InferTypeResult(True, tensor_type(DType.Bool, sh))
    # scalar
    return InferTypeResult(True, t_bool())

# -----------------------------
# select(cond, t, f): cond bool (scalar or broadcastable to out shape)
# t,f numeric with same/broadcastable shape+promoted dtype
# -----------------------------
fn infer_select(cond: TypeDesc, tval: TypeDesc, fval: TypeDesc) -> InferTypeResult:
    # infer t/f result first
    var tf = infer_binary_numeric(tval, fval)
    if not tf.ok: return InferTypeResult(False, t_invalid(), String("t/f not broadcastable: ") + tf.message)

    # determine output shape & dtype
    var out_type = tf.typ
    var out_shape = Shape(); var out_dt = DType.Invalid; var okT = False
    if is_tensor(out_type):
        tensor_unpack(out_type, okT, out_dt, out_shape)
        if not okT: return InferTypeResult(False, t_invalid(), String("Bad t/f inferred tensor"))
    else:
        out_dt = dtype_of_scalar(out_type)

    # cond must be bool (scalar) or tensor<bool,shape> broadcastable to out_shape
    if not is_tensor(cond):
        if not is_bool(cond):
            return InferTypeResult(False, t_invalid(), String("cond must be bool"))
        return InferTypeResult(True, out_type)

    var okC = False; var dtC = DType.Invalid; var shC = Shape()
    tensor_unpack(cond, okC, dtC, shC)
    if not okC or dtC != DType.Bool:
        return InferTypeResult(False, t_invalid(), String("cond tensor must be bool"))
    if is_tensor(out_type):
        var br = broadcast_shapes(shC, out_shape)
        if not br.ok:
            return InferTypeResult(False, t_invalid(), String("cond not broadcastable to result shape"))
        return InferTypeResult(True, tensor_type(out_dt, br.shape))
    # out scalar
    if shC.rank() == 0 or (shC.rank() == 1 and shC.dims.size() == 0):
        return InferTypeResult(True, out_type)
    return InferTypeResult(False, t_invalid(), String("cond cannot broadcast to scalar out"))

# -----------------------------
# transpose(x, perm)
# -----------------------------

struct Perm:
    var idx: List[Int]
fn __init__(out self) -> None: self.idx = List[Int]()
fn __copyinit__(out self, other: Self) -> None:
        self.idx = other.idx
fn __moveinit__(out self, deinit other: Self) -> None:
        self.idx = other.idx
fn infer_transpose(x: TypeDesc, perm: Perm) -> InferTypeResult:
    if not is_tensor(x): return InferTypeResult(False, t_invalid(), String("transpose requires tensor"))
    var okT = False; var dt = DType.Invalid; var sh = Shape()
    tensor_unpack(x, okT, dt, sh)
    if not okT: return InferTypeResult(False, t_invalid(), String("Bad tensor"))
    if perm.idx.size() != sh.rank():
        return InferTypeResult(False, t_invalid(), String("perm size != rank"))
    # Validate permutation indices
    var seen = Dict[Int, Int]()
    for i in range(perm.idx.size()):
        var p = perm.idx[i]
        if p < 0 or p >= sh.rank(): return InferTypeResult(False, t_invalid(), String("perm index OOB"))
        if seen.contains_key(p): return InferTypeResult(False, t_invalid(), String("perm has duplicates"))
        seen[p] = 1
    var out_sh = Shape()
    for i in range(perm.idx.size()):
        out_sh.push(sh.dims[perm.idx[i]])
    return InferTypeResult(True, tensor_type(dt, out_sh))

# -----------------------------
# reshape(x, newshape) with at most one -1
# -----------------------------
fn _product_known(sh: Shape, out known: Bool) -> Int:
    var prod = 1
    known = True
    for i in range(sh.rank()):
        var d = sh.dims[i]
        if d < 0: known = False
        else: prod = prod * d
    return prod
fn infer_reshape(x: TypeDesc, newshape: Shape) -> InferTypeResult:
    if not is_tensor(x): return InferTypeResult(False, t_invalid(), String("reshape requires tensor"))
    var okT = False; var dt = DType.Invalid; var sh = Shape()
    tensor_unpack(x, okT, dt, sh)
    if not okT: return InferTypeResult(False, t_invalid(), String("Bad tensor"))
    # count -1
    var negs = 0
    for i in range(newshape.rank()):
        if newshape.dims[i] == -1: negs = negs + 1
    if negs > 1:
        return InferTypeResult(False, t_invalid(), String("at most one -1 in reshape"))
    # if zero or positive, accept; if one -1, attempt infer if source size known
    if negs == 1:
        var src_known = False
        var src_sz = _product_known(sh, src_known)
        var known = False
        var fixed = 1
        for i in range(newshape.rank()):
            var d = newshape.dims[i]
            if d > 0: fixed = fixed * d
            elif d == 0: known = False  # treat 0 as unknown size here
        if src_known and fixed > 0 and src_sz % fixed == 0:
            # infer -1 = src_sz / fixed
            var out_sh = Shape()
            var placed = False
            for i in range(newshape.rank()):
                var d = newshape.dims[i]
                if d == -1 and (not placed):
                    out_sh.push(src_sz // fixed)
                    placed = True
                else:
                    out_sh.push(d)
            return InferTypeResult(True, tensor_type(dt, out_sh))
        # otherwise accept symbolic result (keep -1)
        return InferTypeResult(True, tensor_type(dt, newshape))
    # no -1, pass-through
    return InferTypeResult(True, tensor_type(dt, newshape))

# -----------------------------
# reduce(x, axes, keepdims)
# -----------------------------

struct Axes:
    var idx: List[Int]
fn __init__(out self) -> None: self.idx = List[Int]()
fn __copyinit__(out self, other: Self) -> None:
        self.idx = other.idx
fn __moveinit__(out self, deinit other: Self) -> None:
        self.idx = other.idx
fn infer_reduce(x: TypeDesc, axes: Axes, keepdims: Bool) -> InferTypeResult:
    if not is_tensor(x): return InferTypeResult(False, t_invalid(), String("reduce requires tensor"))
    var okT = False; var dt = DType.Invalid; var sh = Shape()
    tensor_unpack(x, okT, dt, sh)
    if not okT: return InferTypeResult(False, t_invalid(), String("Bad tensor"))
    # normalize axes (assume unique and in range)
    var seen = Dict[Int, Int]()
    for i in range(axes.idx.size()):
        var a = axes.idx[i]
        if a < 0 or a >= sh.rank(): return InferTypeResult(False, t_invalid(), String("axis OOB"))
        if seen.contains_key(a): return InferTypeResult(False, t_invalid(), String("duplicate axis"))
        seen[a] = 1
    var out_sh = Shape()
    if keepdims:
        for i in range(sh.rank()):
            if seen.contains_key(i): out_sh.push(1)
            else: out_sh.push(sh.dims[i])
    else:
        for i in range(sh.rank()):
            if not seen.contains_key(i): out_sh.push(sh.dims[i])
    return InferTypeResult(True, tensor_type(dt, out_sh))

# -----------------------------

# -----------------------------

struct IntVec:
    var data: List[Int]
fn __init__(out self) -> None: self.data = List[Int]()
fn __copyinit__(out self, other: Self) -> None:
        self.data = other.data
fn __moveinit__(out self, deinit other: Self) -> None:
        self.data = other.data
fn infer_slice(x: TypeDesc, starts: IntVec, sizes: IntVec) -> InferTypeResult:
    if not is_tensor(x): return InferTypeResult(False, t_invalid(), String("slice requires tensor"))
    var okT = False; var dt = DType.Invalid; var sh = Shape()
    tensor_unpack(x, okT, dt, sh)
    if not okT: return InferTypeResult(False, t_invalid(), String("Bad tensor"))
    if starts.data.size() != sh.rank() or sizes.data.size() != sh.rank():
        return InferTypeResult(False, t_invalid(), String("slice vectors must match rank"))
    var out_sh = Shape()
    for i in range(sizes.data.size()):
        var sz = sizes.data[i]
        if sz < 0: out_sh.push(-1)  # unknown
        else: out_sh.push(sz)
    return InferTypeResult(True, tensor_type(dt, out_sh))

# -----------------------------
# concat([x1,x2,...], axis)
# -----------------------------
fn infer_concat(xs: List[TypeDesc], axis: Int) -> InferTypeResult:
    if xs.size() == 0: return InferTypeResult(False, t_invalid(), String("concat needs inputs"))
    # unpack first
    if not is_tensor(xs[0]): return InferTypeResult(False, t_invalid(), String("concat requires tensors"))
    var ok0 = False; var dt0 = DType.Invalid; var sh0 = Shape()
    tensor_unpack(xs[0], ok0, dt0, sh0)
    if not ok0: return InferTypeResult(False, t_invalid(), String("Bad first tensor"))
    if axis < 0 or axis >= sh0.rank():
        return InferTypeResult(False, t_invalid(), String("axis OOB"))

    var out_sh = Shape()
    for i in range(sh0.rank()): out_sh.push(sh0.dims[i])
    var out_dt = dt0

    var sum_axis = 0
    if sh0.dims[axis] >= 0: sum_axis = sh0.dims[axis]
    else: sum_axis = -1

    for j in range(1, xs.size()):
        if not is_tensor(xs[j]): return InferTypeResult(False, t_invalid(), String("concat inputs must be tensors"))
        var okj = False; var dtj = DType.Invalid; var shj = Shape()
        tensor_unpack(xs[j], okj, dtj, shj)
        if not okj: return InferTypeResult(False, t_invalid(), String("Bad tensor at index ") + String(j))
        # require same rank
        if shj.rank() != sh0.rank(): return InferTypeResult(False, t_invalid(), String("rank mismatch"))
        # all dims except axis must match or be -1 compatible
        for d in range(sh0.rank()):
            if d == axis: continue
            var a = out_sh.dims[d]; var b = shj.dims[d]
            if a == -1 or b == -1: out_sh.dims[d] = -1
            elif a != b: return InferTypeResult(False, t_invalid(), String("dim mismatch at ") + String(d))
        # accumulate axis
        var bj = shj.dims[axis]
        if sum_axis == -1 or bj == -1: sum_axis = -1
        else: sum_axis = sum_axis + bj
        # dtype promotion (i64 + f64 -> f64)
        var p = promote_numeric(dtype_to_scalar(out_dt), dtype_to_scalar(dtj))
        out_dt = dtype_of_scalar(p)

    out_sh.dims[axis] = sum_axis
    return InferTypeResult(True, tensor_type(out_dt, out_sh))

# -----------------------------
# matmul (2D only for simplicity): [m,k] x [k,n] -> [m,n]
# -----------------------------
fn infer_matmul(a: TypeDesc, b: TypeDesc) -> InferTypeResult:
    if not is_tensor(a) or not is_tensor(b):
        return InferTypeResult(False, t_invalid(), String("matmul requires tensors"))
    var oka = False; var dta = DType.Invalid; var sha = Shape()
    tensor_unpack(a, oka, dta, sha)
    var okb = False; var dtb = DType.Invalid; var shb = Shape()
    tensor_unpack(b, okb, dtb, shb)
    if not oka or not okb: return InferTypeResult(False, t_invalid(), String("Bad tensor encoding"))
    if sha.rank() != 2 or shb.rank() != 2:
        return InferTypeResult(False, t_invalid(), String("Only 2D matmul implemented"))
    var k1 = sha.dims[1]; var k2 = shb.dims[0]
    if k1 >= 0 and k2 >= 0 and k1 != k2:
        return InferTypeResult(False, t_invalid(), String("Inner dims mismatch"))
    var out = Shape()
    out.push(sha.dims[0])
    out.push(shb.dims[1])
    # dtype promotion
    var p = promote_numeric(dtype_to_scalar(dta), dtype_to_scalar(dtb))
    return InferTypeResult(True, tensor_type(dtype_of_scalar(p), out))

# -----------------------------
# Self-tests
# -----------------------------
fn _self_test_shape_inference() -> Bool:
    var ok = True

    # binary broadcast
    var A = tensor_type(DType.I64, (lambda: 
        { var s = Shape(); s.push(1); s.push(3); s.push(4); return s; })())
    var B = tensor_type(DType.F64, (lambda:
        { var s = Shape(); s.push(2); s.push(1); s.push(4); return s; })())
    var r = infer_binary_numeric(A, B)
    if not r.ok: ok = False
    if not is_tensor(r.typ): ok = False

    # compare -> bool tensor
    var cmp = infer_compare(A, B)
    if not cmp.ok: ok = False
    if not is_tensor(cmp.typ): ok = False
    var okc = False; var dtc = DType.Invalid; var shc = Shape()
    tensor_unpack(cmp.typ, okc, dtc, shc)
    if not okc or dtc != DType.Bool: ok = False

    # select: cond broadcast
    var Cond = tensor_type(DType.Bool, (lambda:
        { var s = Shape(); s.push(2); s.push(3); s.push(4); return s; })())
    var sel = infer_select(Cond, A, B)
    if not sel.ok: ok = False

    # transpose
    var perm = Perm(); perm.idx.push_back(1); perm.idx.push_back(0)
    var X = tensor_type(DType.I64, (lambda: { var s = Shape(); s.push(2); s.push(5); return s; })())
    var tr = infer_transpose(X, perm)
    if not tr.ok: ok = False

    # reshape
    var nsh = Shape(); nsh.push(-1); nsh.push(10)
    var rs = infer_reshape(X, nsh)
    if not rs.ok: ok = False

    # reduce
    var ax = Axes(); ax.idx.push_back(1)
    var rd = infer_reduce(X, ax, True)
    if not rd.ok: ok = False

    # slice
    var st = IntVec(); st.data.push_back(0); st.data.push_back(0)
    var sz = IntVec(); sz.data.push_back(2); sz.data.push_back(3)
    var sl = infer_slice(X, st, sz)
    if not sl.ok: ok = False

    # concat
    var X2 = tensor_type(DType.I64, (lambda: { var s = Shape(); s.push(2); s.push(7); return s; })())
    var arr = List[TypeDesc](); arr.push_back(X); arr.push_back(X2)
    var cc = infer_concat(arr, 1)
    if not cc.ok: ok = False

    # matmul 2D
    var M = tensor_type(DType.I64, (lambda: { var s = Shape(); s.push(3); s.push(4); return s; })())
    var N = tensor_type(DType.F64, (lambda: { var s = Shape(); s.push(4); s.push(5); return s; })())
    var mm = infer_matmul(M, N)
    if not mm.ok: ok = False

    if ok: print(String("OK"))
    else: print(String("FAIL"))
    return ok