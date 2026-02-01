# MIT License
# SPDX-License-Identifier: MIT
# Project:      Momijo
# Module:       learn.api.sequential
# File:         src/momijo/learn/api/sequential.mojo
#
# Description:
#   Rich Sequential container over List[Module] + minimal PTQ helpers
#   (QConfig, MinMaxObserver, Prepared/Converted, prepare/convert, NNSequential).
#
# Notes:
#   - English-only comments. No wildcards in imports. var-only; no globals.
#   - Keeps your public API stable and fixes common pitfalls (copy semantics,
#     robust min/max accumulation, forward/run symmetry, eval-mode copy).

from collections.list import List
from momijo.tensor import tensor

from momijo.learn.nn.module import Module
from momijo.learn.nn.layers import Linear
from momijo.learn.nn.layers import ReLU
from momijo.learn.nn.layers import QuantLinear

from momijo.learn.utils.summary import Summarizer

from momijo.learn.api.functional import (
    quantize_symmetric_int8,
    matmul_i8_i8_to_i32,
    dequantize_i32_to_f32_add_bias,
)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
fn _clamp_index(i: Int, lo: Int, hi: Int) -> Int:
    var v = i
    if v < lo: v = lo
    if v > hi: v = hi
    return v

# -----------------------------------------------------------------------------
# Sequential (rich API, copy-safe)
# -----------------------------------------------------------------------------
struct Sequential(Copyable, Movable):
    var modules: List[Module]

    fn __init__(out self):
        self.modules = List[Module]()

    fn __copyinit__(out self, other: Self):
        self.modules = other.modules.copy()  # value semantics

    # -------------------------- mutation --------------------------
    fn add(mut self, m: Module) -> None:
        self.modules.append(m.copy())

    fn insert(mut self, index: Int, m: Module) -> None:
        var n = len(self.modules)
        var i = _clamp_index(index, 0, n)
        var out_list = List[Module]()
        var k = 0
        while k < i:
            out_list.append(self.modules[k])
            k = k + 1
        out_list.append(m)
        while k < n:
            out_list.append(self.modules[k])
            k = k + 1
        self.modules = out_list

    fn remove_at(mut self, index: Int) -> None:
        var n = len(self.modules)
        if index < 0 or index >= n:
            return
        var out_list = List[Module]()
        var i = 0
        while i < n:
            if i != index:
                out_list.append(self.modules[i])
            i = i + 1
        self.modules = out_list

    fn clear(mut self) -> None:
        self.modules = List[Module]()

    fn set(mut self, index: Int, m: Module) -> None:
        var n = len(self.modules)
        if index < 0 or index >= n:
            return
        self.modules[index] = m

    # -------------------------- accessors --------------------------
    fn len(self) -> Int:
        return len(self.modules)

    fn get(self, index: Int) -> Module:
        var n = len(self.modules)
        assert(n > 0 and "Sequential.get called on empty module list")
        var i = _clamp_index(index, 0, n - 1)
        return self.modules[i]

    fn to_list(self) -> List[Module]:
        var out_list = List[Module]()
        var i = 0
        var n = len(self.modules)
        while i < n:
            out_list.append(self.modules[i])
            i = i + 1
        return out_list

    # -------------------------- execution --------------------------
    fn forward(mut self, x: tensor.Tensor[Float32]) -> tensor.Tensor[Float32]:
        var n = len(self.modules)
        if n == 0:
            return x.copy()
        var out = x.copy()
        var i = 0
        while i < n:
            var m = self.modules[i].copy()        # take a mutable lvalue copy
            out = m.forward(out)           # call mutating forward
            self.modules[i] = m.copy()            # write back in case state changed
            i = i + 1
        return out.copy()

    fn forward(mut self, mut ctx: GradContext, x: GradTensor) -> GradTensor:
        var n = len(self.modules)
        if n == 0:
            return x.copy()
        var out = x.copy()
        var i = 0
        while i < n:
            var m = self.modules[i].copy()             # mutable lvalue
            out = m.forward(ctx, out)           # mutating forward with ctx
            self.modules[i] = m.copy()                 # persist potential state changes
            i = i + 1
        return out.copy()


    fn run(self, x: tensor.Tensor[Float32]) -> tensor.Tensor[Float32]:
        return self.forward(x)

    # -------------------------- introspection --------------------------
    fn summary(self) -> String:
        var n = len(self.modules)
        var s = String("Sequential(") + String(n) + String(" modules)")
        return s

    fn summarize(self, s: Pointer[Summarizer]) -> None:
        var i = 0
        var n = len(self.modules)
        while i < n:
            self.modules[i].summarize(s)
            i = i + 1

    # -------------------------- (de)serialization --------------------------
    fn state_dict(self) -> String:
        var n = len(self.modules)
        var s = String("{\"type\":\"Sequential\",\"num_modules\":") + String(n) + String(",\"children\":[")
        var i = 0
        while i < n:
            s = s + self.modules[i].state_dict()
            if i + 1 < n:
                s = s + String(",")
            i = i + 1
        s = s + String("]}")
        return s


    fn load_state_dict(mut self, state: String) -> None:
        # Expecting a shape like:
        # {"type":"Sequential","num_modules":N,"children":[{...},{...}, ... ]}
        var children_str: String
        var ok: Bool
        (children_str, ok) = _extract_array_key(state, String("children"))
        if not ok:
            # No children found: clear modules to be safe
            self.modules = List[Module]()
            return

        var objs = _split_top_level_objects(children_str)
        var new_list = List[Module]()
        var k = 0
        var m = len(objs)
        while k < m:
            var child_json = objs[k]
            var mod = Module()             # assumes default-constructible
            mod.load_state_dict(child_json)  # delegate to Module’s own parser
            new_list.append(mod.copy())
            k = k + 1
        self.modules = new_list.copy()

    # -------------------------- stringification --------------------------
    fn __str__(self) -> String:
        var n = len(self.modules)
        var s = String("Sequential[")
        var i = 0
        while i < n:
            s = s + self.modules[i].__str__()
            if i + 1 < n:
                s = s + String(", ")
            i = i + 1
        s = s + String("]")
        return s
# -----------------------------------------------------------------------------
# QConfig + MinMaxObserver (for simple PTQ flows)
# -----------------------------------------------------------------------------
struct QConfig(Copyable, Movable):
    var backend: String
    fn __init__(out self, backend: String = String("fbgemm")):
        self.backend = backend
    fn __copyinit__(out self, other: Self):
        self.backend = other.backend

fn get_default_qconfig(backend: String = String("fbgemm")) -> QConfig:
    return QConfig(backend)

struct MinMaxObserver(Copyable, Movable):
    var has_data: Bool
    var min_v: Float32
    var max_v: Float32

    fn __init__(out self):
        self.has_data = False
        self.min_v = 0.0
        self.max_v = 0.0

    fn __copyinit__(out self, other: Self):
        self.has_data = other.has_data
        self.min_v = other.min_v
        self.max_v = other.max_v

    fn observe(mut self, x: tensor.Tensor[Float32]) -> None:
        var n = x.numel()
        if n == 0:
            return
        var i = 0
        var mn = x._data[0]
        var mx = x._data[0]
        while i < n:
            var v = x._data[i]
            if v < mn: mn = v
            if v > mx: mx = v
            i = i + 1
        if not self.has_data:
            self.min_v = mn
            self.max_v = mx
            self.has_data = True
        else:
            if mn < self.min_v: self.min_v = mn
            if mx > self.max_v: self.max_v = mx

    fn scale_symmetric(self, max_q: Float32 = 127.0) -> Float32:
        if not self.has_data or max_q <= 0.0:
            return 1.0
        var a = self.min_v
        if a < 0.0: a = -a
        var b = self.max_v
        if b < 0.0: b = -b
        var m = a
        if b > m: m = b
        if m <= 0.0:
            return 1.0
        return m / max_q


# ---------------------------------------------------------------------------
# NNSequential: light facade over Sequential with copy-safe semantics
# ---------------------------------------------------------------------------
struct NNSequential(Copyable, Movable):
    var seq: Sequential
    var qconfig: QConfig
    var is_training: Bool

    # -------------------------- construction --------------------------
    fn __init__(out self, modules: List[Module], qcfg: QConfig = QConfig(String("fbgemm"))):
        var s = Sequential()
        var n = len(modules)
        var i = 0
        while i < n:
            s.add(modules[i])
            i = i + 1
        self.seq = s.copy()
        self.qconfig = qcfg.copy()
        self.is_training = True

    fn __copyinit__(out self, other: Self):
        self.seq = other.seq.copy()
        self.qconfig = other.qconfig.copy()
        self.is_training = other.is_training

    # -------------------------- mutation (delegates) -------------------
    fn add(mut self, m: Module) -> None:
        self.seq.add(m)

    fn insert(mut self, index: Int, m: Module) -> None:
        self.seq.insert(index, m)

    fn remove_at(mut self, index: Int) -> None:
        self.seq.remove_at(index)

    fn clear(mut self) -> None:
        self.seq.clear()

    fn set(mut self, index: Int, m: Module) -> None:
        self.seq.set(index, m)

    # -------------------------- accessors ------------------------------
    fn len(self) -> Int:
        return self.seq.len()

    fn get(self, index: Int) -> Module:
        return self.seq.get(index)

    fn to_list(self) -> List[Module]:
        return self.seq.to_list()

    fn to_sequential(self) -> Sequential:
        return self.seq

    # -------------------------- training mode --------------------------
    fn train(mut self) -> None:
        self.is_training = True
        # If Module.train() exists in your codebase, call it in-place here.

    fn eval(mut self) -> None:
        self.is_training = False
        # If Module.eval() exists in your codebase, call it in-place here.

    # -------------------------- qconfig -------------------------------
    fn set_qconfig(mut self, qcfg: Qconfig) -> None:
        self.qconfig = qcfg

    fn get_qconfig(self) -> QConfig:
        return self.qconfig

    # -------------------------- execution ------------------------------
    fn forward(mut self, x: tensor.Tensor[Float32]) -> tensor.Tensor[Float32]:
        # Sequential.forward requires `mut self`, so NNSequential.forward must also be `mut self`
        return self.seq.forward(x)

    fn forward(mut self, mut ctx: GradContext, x: GradTensor) -> GradTensor:
        # Call the GradTensor variant directly on the mutable field `self.seq`
        return self.seq.forward(ctx, x)


    fn run(self, x: tensor.Tensor[Float32]) -> tensor.Tensor[Float32]:
        return self.forward(x)

    # -------------------------- (de)serialization ----------------------
    fn state_dict(self) -> String:
        var s = String("{\"type\":\"NNSequential\",\"training\":") + String(self.is_training) + String(",\"qconfig\":\"")
        s = s + self.qconfig.__str__() + String("\",\"seq\":") + self.seq.state_dict() + String("}")
        return s
    # -----------------------------------------------------------------------------
    # NNSequential.load_state_dict (complete; no nested functions)
    # -----------------------------------------------------------------------------
    fn load_state_dict(mut self, state: String) -> None:
        # training
        var t1 = _parse_bool_key(state, String("training"))
        if t1[1]:
            self.is_training = t1[0]

        # qconfig
        var t2 = _parse_string_key(state, String("qconfig"))
        if t2[1]:
            self.qconfig = QConfig(t2[0])

        # seq (delegate JSON of the inner Sequential to Sequential.load_state_dict)
        var t3 = _extract_object_key(state, String("seq"))
        if t3[1]:
            self.seq.load_state_dict(t3[0])

    # -------------------------- introspection --------------------------
    fn summary(self) -> String:
        var n = self.seq.len()
        return String("NNSequential(") + String(n) + String(" modules, training=") + String(self.is_training) + String(")")

    fn __str__(self) -> String:
        return String("NNSequential{ training=") + String(self.is_training) + String(", ") + self.seq.__str__() + String(" }")
    # -----------------------------------------------------------------------------
# Prepared/Converted (Linear → ReLU → Linear) + prepare/convert
# -----------------------------------------------------------------------------
struct Prepared(Copyable, Movable):
    var l1: Linear
    var relu: ReLU
    var l2: Linear
    var obs_in: MinMaxObserver
    var obs_relu: MinMaxObserver
    var obs_w1: MinMaxObserver
    var obs_w2: MinMaxObserver

    fn __init__(out self, l1: Linear, relu: ReLU, l2: Linear):
        self.l1 = l1.copy()
        self.relu = relu.copy()
        self.l2 = l2.copy()
        self.obs_in = MinMaxObserver()
        self.obs_relu = MinMaxObserver()
        self.obs_w1 = MinMaxObserver()
        self.obs_w2 = MinMaxObserver()

    fn forward(mut self, x: tensor.Tensor[Float32]) -> tensor.Tensor[Float32]:
        # collect activation stats
        self.obs_in.observe(x)
        var y1 = self.l1.forward(x)
        var y2 = self.relu.forward(y1)
        self.obs_relu.observe(y2)
        # collect weight stats
        self.obs_w1.observe(self.l1.weight)
        self.obs_w2.observe(self.l2.weight)
        var y3 = self.l2.forward(y2)
        return y3.copy()

struct Converted(Copyable, Movable):
    var q1: QuantLinear
    var q2: QuantLinear

    fn __init__(out self, q1: QuantLinear, q2: QuantLinear):
        self.q1 = q1.copy()
        self.q2 = q2.copy()

    fn forward(self, x: tensor.Tensor[Float32]) -> tensor.Tensor[Float32]:
        var y1 = self.q1.forward(x)
        # ReLU in float
        var y2 = y1.maximum_scalar(0.0)
        return self.q2.forward(y2)

# prepare: expects a 3-layer pattern [Linear, ReLU, Linear]
fn prepare(m: NNSequential, inplace: Bool = False) -> Prepared:
    _ = inplace
    var l1 = m.seq.get(0).linear.copy()
    var relu = m.seq.get(1).relu.copy()
    var l2 = m.seq.get(2).linear.copy()
    return Prepared(l1, relu, l2)

# convert: quantize weights sym-int8 using observed scales and make QuantLinear blocks
fn convert(p: Prepared, inplace: Bool = False) -> Converted:
    _ = inplace
    var s_in   = p.obs_in.scale_symmetric(127.0)
    var s_relu = p.obs_relu.scale_symmetric(127.0)
    var s_w1   = p.obs_w1.scale_symmetric(127.0)
    var s_w2   = p.obs_w2.scale_symmetric(127.0)

    # quantize weights
    var w1_q = quantize_symmetric_int8(p.l1.weight, s_w1)
    var w2_q = quantize_symmetric_int8(p.l2.weight, s_w2)

    # build quantized linear modules (assumes QuantLinear(in, out, Wq, bias, s_w, s_in))
    var q1 = QuantLinear(p.l1.in_features, p.l1.out_features, w1_q, p.l1.bias_t, s_w1, s_in)
    var q2 = QuantLinear(p.l2.in_features, p.l2.out_features, w2_q, p.l2.bias_t, s_w2, s_relu)
    return Converted(q1, q2)

# -----------------------------------------------------------------------------
# Optional overloads to be compatible with alt call-sites: prepare(m, qat) etc.
# -----------------------------------------------------------------------------
fn prepare(m: NNSequential, qat: Bool) -> Prepared:
    return prepare(m, inplace=not qat)

fn convert(p: Prepared, qat: Bool) -> Converted:
    return convert(p, inplace=not qat)



@always_inline
fn _sfind(s: String, pat: String, start: Int = 0) -> Int:
    return s.find(pat, start)

fn _sslice(s: String, lo: Int, hi: Int) -> String:
    # half-open [lo, hi)
    var n = len(s)
    var a = (lo if lo >= 0 else 0)
    var b = (hi if hi <= n else n)
    var L = b - a
    if L <= 0:
        return String("")
    # No substring API available; return s only if the slice spans the whole string.
    if a == 0 and b == n:
        return s
    return String("")

@always_inline
fn _slen(s: String) -> Int:
    return len(s)



# Parse quoted string after `"qconfig":"..."`
fn _parse_string_key(s: String, key: String) -> (String, Bool):
    var anchor = String("\"") + key + String("\":\"")
    var p = _sfind(s, anchor, 0)
    if p < 0:
        return (String(""), False)
    var i = p + _slen(anchor)
    var n = _slen(s)
    var j = i
    while j < n:
        if _sslice(s, j, j + 1) == String("\""):
            return (_sslice(s, i, j), True)
        j = j + 1
    return (String(""), False)

# Extract balanced `{...}` object after `"seq":`
fn _extract_object_key(s: String, key: String) -> (String, Bool):
    var anchor = String("\"") + key + String("\":")
    var p = _sfind(s, anchor, 0)
    if p < 0:
        return (String(""), False)
    var i = p + _slen(anchor)
    var n = _slen(s)
    while i < n:
        var ch0 = _sslice(s, i, i + 1)
        if not (ch0 == String(" ") or ch0 == String("\n") or ch0 == String("\t")):
            break
        i = i + 1
    if i >= n or _sslice(s, i, i + 1) != String("{"):
        return (String(""), False)
    var depth = 0
    var j = i
    while j < n:
        var ch = _sslice(s, j, j + 1)
        if ch == String("{"):
            depth = depth + 1
        elif ch == String("}"):
            depth = depth - 1
            if depth == 0:
                return (_sslice(s, i, j + 1), True)
        j = j + 1
    return (String(""), False)

 # Parse boolean after `"training":`
fn _parse_bool_key(s: String, key: String) -> (Bool, Bool):
    var anchor = String("\"") + key + String("\":")
    var p = _sfind(s, anchor, 0)
    if p < 0:
        return (False, False)
    var i = p + _slen(anchor)
    while i < _slen(s):
        var ch = _sslice(s, i, i + 1)
        if not (ch == String(" ") or ch == String("\n") or ch == String("\t")):
            break
        i = i + 1
    if i + 4 <= _slen(s) and _sslice(s, i, i + 4) == String("true"):
        return (True, True)
    if i + 5 <= _slen(s) and _sslice(s, i, i + 5) == String("false"):
        return (False, True)
    return (False, False)

# Extract the content inside an array value: "<key>":[ ... ]
fn _extract_array_key(s: String, key: String) -> (String, Bool):
    var anchor = String("\"") + key + String("\":[")
    var p = _sfind(s, anchor, 0)
    if p < 0:
        return (String(""), False)
    var i = p + _slen(anchor)
    var n = _slen(s)
    var depth = 1            # we are after '[' so depth=1
    var j = i
    while j < n:
        var ch = _sslice(s, j, j + 1)
        if ch == String("["):
            depth = depth + 1
        elif ch == String("]"):
            depth = depth - 1
            if depth == 0:
                # return inside content (without the outer brackets)
                return (_sslice(s, i, j), True)
        j = j + 1
    return (String(""), False)


# ---------------- String helpers (file-scope) ----------------


# Split a comma-separated list of top-level JSON objects '{...},{...},...'
# respecting brace depth.
fn _split_top_level_objects(arr_content: String) -> List[String]:
    var out = List[String]()
    var n = _slen(arr_content)
    var i = 0
    var depth = 0
    var start = -1
    while i < n:
        var ch = _sslice(arr_content, i, i + 1)
        if ch == String("{"):
            if depth == 0:
                start = i
            depth = depth + 1
        elif ch == String("}"):
            depth = depth - 1
            if depth == 0 and start >= 0:
                out.append(_sslice(arr_content, start, i + 1))
                start = -1
        i = i + 1
    return out.copy()
