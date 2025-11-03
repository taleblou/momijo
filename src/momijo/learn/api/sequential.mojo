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
# Sequential (rich API)
# -----------------------------------------------------------------------------
struct Sequential(Copyable, Movable):
    # Ordered list of modules applied one after another.
    var modules: List[Module]

    fn __init__(out self):
        self.modules = List[Module]()

    fn __copyinit__(out self, other: Self):
        # value semantics for the module list
        self.modules = other.modules.copy()

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
        out_list.append(m.copy())
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
        self.modules[index] = m.copy()

    # -------------------------- accessors --------------------------
    fn len(self) -> Int:
        return len(self.modules)

    fn get(self, index: Int) -> Module:
        var n = len(self.modules)
        if n == 0:
            # Fallback: assume Module() default-constructible in your codebase
            var dummy = Module()
            return dummy.copy()
        var i = _clamp_index(index, 0, n - 1)
        return self.modules[i].copy()

    fn to_list(self) -> List[Module]:
        var out_list = List[Module]()
        var i = 0
        var n = len(self.modules)
        while i < n:
            out_list.append(self.modules[i])
            i = i + 1
        return out_list.copy()

    # -------------------------- execution --------------------------
    fn forward(self, x: tensor.Tensor[Float64]) -> tensor.Tensor[Float64]:
        var out = x.copy()
        var i = 0
        var n = len(self.modules)
        while i < n:
            var m = self.modules[i].copy()
            out = m.forward(out)
            i = i + 1
        return out.copy()

    fn forward(mut self, mut ctx: GradContext, x: GradTensor) -> GradTensor:
        var out = x.copy()  
        var i = 0
        var n = len(self.modules)
        while i < n:
            # Call the GradTensor variant: forward(ctx, GradTensor)
            out = self.modules[i].forward(ctx, out)
            i = i + 1
        return out.copy()  

    fn run(self, x: tensor.Tensor[Float64]) -> tensor.Tensor[Float64]:
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

    fn load_state_dict(mut self, state: String) -> None:
        # NOTE: Implement JSON parse+dispatch if/when your JSON utilities are available.
        # Keeping a no-op to avoid brittle ad-hoc parsers here.
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

fn get_default_qconfig(backend: String = String("fbgemm")) -> QConfig:
    return QConfig(backend)

struct MinMaxObserver(Copyable, Movable):
    var has_data: Bool
    var min_v: Float64
    var max_v: Float64

    fn __init__(out self):
        self.has_data = False
        self.min_v = 0.0
        self.max_v = 0.0

    fn __copyinit__(out self, other: Self):
        self.has_data = other.has_data
        self.min_v = other.min_v
        self.max_v = other.max_v

    fn observe(mut self, x: tensor.Tensor[Float64]) -> None:
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

    fn scale_symmetric(self, max_q: Float64 = 127.0) -> Float64:
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

    fn eval(mut self) -> None:
        # If Module.eval() exists in your codebase, call it; otherwise no-op.
        var n = self.seq.len()
        var mods = List[Module]()
        var i = 0
        while i < n:
            var m = self.seq.get(i)
            # m = m.eval()    # Uncomment if your Module supports eval()
            mods.append(m.copy())
            i = i + 1
        self.seq.modules = mods.copy()

    fn forward(self, x: tensor.Tensor[Float64]) -> tensor.Tensor[Float64]:
        return self.seq.forward(x)
 
    fn forward(self, mut ctx: GradContext, x: GradTensor) -> GradTensor: 
        var s = self.copy()           
        var y = s.forward(ctx, x)        
        return y.copy()     
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

    fn forward(self, x: tensor.Tensor[Float64]) -> tensor.Tensor[Float64]:
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
