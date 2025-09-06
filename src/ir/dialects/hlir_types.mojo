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
# File: src/momijo/ir/hlir_types.mojo
# Description: High-Level IR (HLIR) type utilities: dtype, shape, tensor types,
#              numeric promotion, broadcasting, and binary type inference.
# Notes:
#   - Self-contained and compatible with the minimal TypeDesc(name: String) model used elsewhere.
#   - No globals, no 'export'; only 'var' (no 'let').
#   - Constructors use 'fn __init__(out self, ...)'.
# ============================================================================

# -----------------------------
# Minimal TypeDesc
# -----------------------------

struct TypeDesc:
    var name: String
    fn __init__(out self, name: String):
        self.name = name

# Scalar shorthands (consistent with other modules)
fn t_i64() -> TypeDesc: return TypeDesc(String("i64"))
fn t_f64() -> TypeDesc: return TypeDesc(String("f64"))
fn t_bool() -> TypeDesc: return TypeDesc(String("bool"))
fn t_invalid() -> TypeDesc: return TypeDesc(String("invalid"))

fn is_i64(t: TypeDesc) -> Bool: return t.name == String("i64")
fn is_f64(t: TypeDesc) -> Bool: return t.name == String("f64")
fn is_bool(t: TypeDesc) -> Bool: return t.name == String("bool")
fn is_numeric(t: TypeDesc) -> Bool: return is_i64(t) or is_f64(t)

# -----------------------------
# DType (logical element type) and helpers
# -----------------------------

enum DType(Int):
    Invalid = 0
    I64 = 1
    F64 = 2
    Bool = 3

fn dtype_of_scalar(t: TypeDesc) -> DType:
    if is_i64(t): return DType.I64
    if is_f64(t): return DType.F64
    if is_bool(t): return DType.Bool
    return DType.Invalid

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

# -----------------------------
# Shape
# -----------------------------
# Convention: -1 means unknown dimension (symbolic). 1 means broadcastable.

struct Shape:
    var dims: List[Int]

    fn __init__(out self):
        self.dims = List[Int]()

    fn rank(self) -> Int:
        return self.dims.size()

    fn push(self, d: Int) -> None:
        self.dims.push_back(d)

    fn to_string(self) -> String:
        var s = String("[")
        for i in range(self.dims.size()):
            s = s + String(self.dims[i])
            if i + 1 < self.dims.size():
                s = s + String(", ")
        s = s + String("]")
        return s

# -----------------------------
# Tensor types encoded inside TypeDesc.name
# Format: tensor<DTYPE,[d0,d1,...]>
# -----------------------------

fn tensor_type(dtype: DType, shape: Shape) -> TypeDesc:
    var s = String("tensor<") + dtype_name(dtype) + String(",") + shape.to_string() + String(">")
    return TypeDesc(s)

fn is_tensor(t: TypeDesc) -> Bool:
    # cheap check
    return t.name.starts_with(String("tensor<"))

fn _parse_between(s: String, start_ch: UInt8, end_ch: UInt8, out ok: Bool) -> String:
    # returns substring between first matching start/end chars
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
    # slice
    var out_s = String("")
    var j = start_idx
    while j < end_idx:
        # manual append one char at a time
        out_s = out_s + String.from_utf8([s.bytes()[j]])
        j = j + 1
    return out_s

fn tensor_unpack(t: TypeDesc, out ok: Bool, out out_dtype: DType, out out_shape: Shape) -> None:
    ok = False
    out_dtype = DType.Invalid
    out_shape = Shape()
    if not is_tensor(t): return
    # format: tensor<DTYPE,[...]>
    var inner = _parse_between(t.name, 60, 62, ok)   # '<' = 60, '>' = 62
    if not ok: return

    # split at first comma
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

    # dtype
    if dts == String("i64"): out_dtype = DType.I64
    elif dts == String("f64"): out_dtype = DType.F64
    elif dts == String("bool"): out_dtype = DType.Bool
    else: out_dtype = DType.Invalid

    # parse shape inside [...] 
    var ok2 = False
    var ss = _parse_between(rest, 91, 93, ok2)  # '['=91, ']'=93
    if not ok2: return

    # split by commas (tolerate spaces)
    var cur = String("")
    for k in range(ss.size() + 1):
        var is_end = k == ss.size()
        var ch = UInt8(0)
        if not is_end:
            ch = ss.bytes()[k]
        if is_end or ch == 44:  # ','
            var tok = cur.strip()
            if tok.size() > 0:
                # parse int (allow -1)
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
# Type promotion (numeric)
# -----------------------------

fn promote_numeric(a: TypeDesc, b: TypeDesc) -> TypeDesc:
    # Scalars only; caller handles tensors by unpacking dtype
    if is_f64(a) or is_f64(b): return t_f64()
    if is_i64(a) and is_i64(b): return t_i64()
    return t_invalid()

# -----------------------------
# Broadcasting
# -----------------------------

struct InferShapeResult:
    var ok: Bool
    var shape: Shape
    var message: String

    fn __init__(out self, ok: Bool = True, shape: Shape = Shape(), message: String = String("")):
        self.ok = ok
        self.shape = shape
        self.message = message

fn _broadcast_dim(a: Int, b: Int, out ok: Bool) -> Int:
    # -1 unknown; 1 is broadcastable; otherwise must match
    ok = True
    if a == -1 and b == -1:
        return -1
    if a == -1:
        return b
    if b == -1:
        return a
    if a == 1:
        return b
    if b == 1:
        return a
    if a == b:
        return a
    ok = False
    return -2  # invalid

fn broadcast_shapes(a: Shape, b: Shape) -> InferShapeResult:
    var ra = a.rank()
    var rb = b.rank()
    var r = ra
    if rb > r: r = rb

    var out_shape = Shape()
    # build from right to left
    var i = 0
    var all_ok = True
    while i < r:
        var da = 1
        var db = 1
        if ra - 1 - i >= 0: da = a.dims[ra - 1 - i]
        if rb - 1 - i >= 0: db = b.dims[rb - 1 - i]
        var okd = False
        var d = _broadcast_dim(da, db, okd)
        if not okd:
            all_ok = False
            d = -2
        # prepend by building reversed then reversing
        out_shape.dims.push_back(d)
        i = i + 1

    # reverse dims
    var rev = Shape()
    var j = out_shape.rank() - 1
    while j >= 0:
        rev.dims.push_back(out_shape.dims[j])
        j = j - 1

    if not all_ok:
        return InferShapeResult(False, rev, String("Shape broadcast failed"))
    return InferShapeResult(True, rev, String(""))

# -----------------------------
# Binary inference (numeric/tensor)
# -----------------------------

struct InferTypeResult:
    var ok: Bool
    var typ: TypeDesc
    var message: String

    fn __init__(out self, ok: Bool = True, typ: TypeDesc = TypeDesc(String("")), message: String = String("")):
        self.ok = ok
        self.typ = typ
        self.message = message

fn infer_binary_numeric(lhs: TypeDesc, rhs: TypeDesc) -> InferTypeResult:
    # Cases:
    # 1) both scalars numeric -> scalar(promote)
    # 2) tensor vs tensor -> tensor(promote dtype, broadcast shapes)
    # 3) scalar vs tensor -> tensor(dtype promote, shape of tensor)
    # 4) otherwise invalid
    var ls = is_tensor(lhs)
    var rs = is_tensor(rhs)

    if not ls and not rs:
        if is_numeric(lhs) and is_numeric(rhs):
            return InferTypeResult(True, promote_numeric(lhs, rhs))
        return InferTypeResult(False, t_invalid(), String("Non-numeric scalars"))

    # unpack tensors
    var okL = False
    var okR = False
    var dtL = DType.Invalid
    var dtR = DType.Invalid
    var shL = Shape()
    var shR = Shape()
    if ls: tensor_unpack(lhs, okL, dtL, shL)
    if rs: tensor_unpack(rhs, okR, dtR, shR)
    if ls and (not okL): return InferTypeResult(False, t_invalid(), String("Bad lhs tensor encoding"))
    if rs and (not okR): return InferTypeResult(False, t_invalid(), String("Bad rhs tensor encoding"))

    # dtype
    var out_dt = DType.Invalid
    if ls and rs:
        var a = dtype_to_scalar(dtL)
        var b = dtype_to_scalar(dtR)
        var p = promote_numeric(a, b)
        out_dt = dtype_of_scalar(p)
        # shape broadcast
        var br = broadcast_shapes(shL, shR)
        if not br.ok:
            return InferTypeResult(False, t_invalid(), br.message)
        return InferTypeResult(True, tensor_type(out_dt, br.shape))
    elif ls and (not rs):
        # scalar against tensor -> broadcast scalar
        if not is_numeric(rhs):
            return InferTypeResult(False, t_invalid(), String("RHS not numeric scalar"))
        var p2 = promote_numeric(dtype_to_scalar(dtL), rhs)
        out_dt = dtype_of_scalar(p2)
        return InferTypeResult(True, tensor_type(out_dt, shL))
    elif (not ls) and rs:
        if not is_numeric(lhs):
            return InferTypeResult(False, t_invalid(), String("LHS not numeric scalar"))
        var p3 = promote_numeric(lhs, dtype_to_scalar(dtR))
        out_dt = dtype_of_scalar(p3)
        return InferTypeResult(True, tensor_type(out_dt, shR))

    return InferTypeResult(False, t_invalid(), String("Unsupported combination"))

# -----------------------------
# Self-tests
# -----------------------------

fn _self_test_hlir_types() -> Bool:
    var ok = True

    # promotion
    if promote_numeric(t_i64(), t_f64()).name != String("f64"): ok = False
    if promote_numeric(t_i64(), t_i64()).name != String("i64"): ok = False

    # tensor encode/decode
    var s = Shape(); s.push(2); s.push(3)
    var T = tensor_type(DType.I64, s)
    var good = False
    var dt = DType.Invalid
    var sh = Shape()
    tensor_unpack(T, good, dt, sh)
    if not good: ok = False
    if dt != DType.I64: ok = False
    if sh.rank() != 2 or sh.dims[0] != 2 or sh.dims[1] != 3: ok = False

    # broadcast
    var a = Shape(); a.push(1); a.push(3); a.push(4)
    var b = Shape(); b.push(2); b.push(1); b.push(4)
    var br = broadcast_shapes(a, b)
    if not br.ok: ok = False
    if br.shape.rank() != 3: ok = False
    if br.shape.dims[0] != 2 or br.shape.dims[1] != 3 or br.shape.dims[2] != 4: ok = False

    # binary inference: tensor x tensor
    var Ta = tensor_type(DType.I64, a)
    var Tb = tensor_type(DType.F64, b)
    var ir = infer_binary_numeric(Ta, Tb)
    if not ir.ok: ok = False
    if not is_tensor(ir.typ): ok = False
    var okp = False
    var pdt = DType.Invalid
    var psh = Shape()
    tensor_unpack(ir.typ, okp, pdt, psh)
    if not okp: ok = False
    if pdt != DType.F64: ok = False
    if psh.rank() != 3: ok = False

    if ok: print(String("OK"))
    else: print(String("FAIL"))
    return ok
 
