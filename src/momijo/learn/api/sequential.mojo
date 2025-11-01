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
#   - Fixes prior errors: duplicate methods, bad len() usage, wrong field types
#     like `var seq: Sequential()`, missing QConfig/MinMaxObserver, and use of
#     `QuantLinear()` as a type instead of `QuantLinear`.

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
@always_inline
fn _clamp_index(i: Int, lo: Int, hi: Int) -> Int:
    var v = i
    if v < lo: v = lo
    if v > hi: v = hi
    return v

# -----------------------------------------------------------------------------
# Sequential (rich API)
# -----------------------------------------------------------------------------
struct Sequential(Copyable, Movable):
    # Ordered list of modules applied one after another.
    var modules: List[Module]

    fn __init__(out self):
        self.modules = List[Module]()

    fn __copyinit__(out self, other: Self):
        self.modules = other.modules.copy()

    # -------------------------- mutation --------------------------
    fn add(mut self, m: Module):
        self.modules.append(m.copy())

    fn insert(mut self, index: Int, m: Module):
        var n = len(self.modules)
        var i = _clamp_index(index, 0, n)
        var out_list = List[Module]()
        var k = 0
        while k < i:
            out_list.append(self.modules[k])
            k = k + 1
        out_list.append(m.copy())
        while k < n:
            out_list.append(self.modules[k])
            k = k + 1
        self.modules = out_list

    fn remove_at(mut self, index: Int):
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

    fn clear(mut self):
        self.modules = List[Module]()

    fn set(mut self, index: Int, m: Module):
        var n = len(self.modules)
        if index < 0 or index >= n:
            return
        self.modules[index] = m

    # -------------------------- accessors --------------------------
    fn len(self) -> Int:
        return len(self.modules)

    fn get(self, index: Int) -> Module:
        var n = len(self.modules)
        if n == 0:
            # Fallback: assume Module() default-constructible in your codebase
            var dummy = Module()
            return dummy
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
    fn forward(self, x: tensor.Tensor[Float64]) -> tensor.Tensor[Float64]:
        var out = x.copy()
        var i = 0
        var n = len(self.modules)
        while i < n:
            var m = self.modules[i]
            out = m.forward(out)
            i = i + 1
        return out

    fn run(self, x: tensor.Tensor[Float64]) -> tensor.Tensor[Float64]:
        return self.forward(x)

    # -------------------------- introspection --------------------------
    fn summary(self) -> String:
        var n = len(self.modules)
        var s = String("Sequential(") + String(n) + String(" modules)")
        return s

    fn summarize(self, s: Pointer[Summarizer]):
        var i = 0
        var n = len(self.modules)
        while i < n:
            self.modules[i].summarize(s)
            i = i + 1

    # -------------------------- (de)serialization --------------------------
    fn state_dict(self) -> String:
        var n = len(self.modules)
        var s = String("{\"type\":\"Sequential\",\"num_modules\":") + String(n) + String(",\"children\":")
        s = s + String("[")
        var i = 0
        while i < n:
            s = s + self.modules[i].state_dict()
            if i + 1 < n:
                s = s + String(",")
            i = i + 1
        s = s + String("]}")
        return s

    fn load_state_dict(mut self, state: String):
        # TODO: parse JSON and dispatch into child modules
        _ = state
        pass

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

@always_inline
fn get_default_qconfig(backend: String = String("fbgemm")) -> QConfig:
    return QConfig(backend)


struct MinMaxObserver(Copyable, Movable):
    var min_v: Float64
    var max_v: Float64
    fn __init__(out self):
        self.min_v = 0.0
        self.max_v = 0.0
    fn __copyinit__(out self, other: Self):
        self.min_v = other.min_v
        self.max_v = other.max_v

    fn observe(mut self, x: tensor.Tensor[Float64]):
        var shp = x.shape()
        var n = 1
        var i = 0
        while i < len(shp):
            n = n * shp[i]
            i = i + 1
        if n == 0: return
        var d = x._data.copy()
        var mn = d[0]
        var mx = d[0]
        var j = 1
        while j < n:
            var v = d[j]
            if v < mn: mn = v
            if v > mx: mx = v
            j = j + 1
        if self.min_v == 0.0 or mn < self.min_v: self.min_v = mn
        if self.max_v == 0.0 or mx > self.max_v: self.max_v = mx

    fn scale_symmetric(self, max_q: Float64 = 127.0) -> Float64:
        var a = self.min_v
        if a < 0.0: a = -a
        var b = self.max_v
        if b < 0.0: b = -b
        var m = a if a > b else b
        if m <= 0.0: return 1.0
        if max_q <= 0.0: return 1.0
        return m / max_q

# -----------------------------------------------------------------------------
# PyTorch-like facade (very light)
# -----------------------------------------------------------------------------
struct NNSequential(Copyable, Movable):
    var seq: Sequential
    var qconfig: QConfig

    fn __init__(out self, modules: List[Module], qcfg: QConfig = QConfig(String("fbgemm"))):
        var s = Sequential()
        var n = len(modules)
        var i = 0
        while i < n:
            s.add(modules[i])
            i = i + 1
        self.seq = s.copy()
        self.qconfig = qcfg.copy()

    fn eval(mut self):
        # If Module.eval() exists in your codebase, call it; otherwise no-op.
        var n = len(self.seq.modules)
        var mods = List[Module]()
        var i = 0
        while i < n:
            var m = self.seq.modules[i].copy()
            # m = m.eval()    # Uncomment if defined
            mods.append(m.copy())
            i = i + 1
        self.seq.modules = mods.copy()

    fn forward(self, x: tensor.Tensor[Float64]) -> tensor.Tensor[Float64]:
        return self.seq.forward(x)

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

    fn forward(mut self, x: tensor.Tensor[Float64]) -> tensor.Tensor[Float64]:
        self.obs_in.observe(x)
        var y1 = self.l1.forward(x)
        var y2 = self.relu.forward(y1)
        self.obs_relu.observe(y2)
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

    fn forward(self, x: tensor.Tensor[Float64]) -> tensor.Tensor[Float64]:
        var y1 = self.q1.forward(x)
        # ReLU in float (could be fused/reqt later)
        var y2 = y1.maximum_scalar(0.0)
        return self.q2.forward(y2)

fn prepare(m: NNSequential, inplace: Bool = False) -> Prepared:
    _ = inplace
    # Expect Linear, ReLU, Linear
    var l1 = m.seq.modules[0].linear.copy()
    var relu = m.seq.modules[1].relu.copy()
    var l2 = m.seq.modules[2].linear.copy()
    return Prepared(l1, relu, l2)

fn convert(p: Prepared, inplace: Bool = False) -> Converted:
    _ = inplace
    var s_in = p.obs_in.scale_symmetric(127.0)
    var s_relu = p.obs_relu.scale_symmetric(127.0)
    var s_w1 = p.obs_w1.scale_symmetric(127.0)
    var s_w2 = p.obs_w2.scale_symmetric(127.0)
    var w1_q = quantize_symmetric_int8(p.l1.weight, s_w1)
    var w2_q = quantize_symmetric_int8(p.l2.weight, s_w2)
    var q1 = QuantLinear(p.l1.in_features, p.l1.out_features, w1_q, p.l1.bias_t, s_w1, s_in)
    var q2 = QuantLinear(p.l2.in_features, p.l2.out_features, w2_q, p.l2.bias_t, s_w2, s_relu)
    return Converted(q1, q2)
