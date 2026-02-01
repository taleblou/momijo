# MIT License
# SPDX-License-Identifier: MIT
# Project:      Momijo
# Module:       learn.nn.init
# File:         src/momijo/learn/nn/init.mojo
#
# Description:
#   Initialization utilities (He/Kaiming, Xavier/Glorot) for Momijo Learn.
#   - Backend-agnostic math and fan_in/fan_out helpers
#   - Gain calculation for common nonlinearities
#   - Uniform/Normal bounds/std and fillers
#   - Tiny LCG RNG (bootstrap) + CLT-based normal approximation
#
# Author(s):    Morteza Taleblou & Mitra Daneshmand
# Website:      https://taleblou.ir/
# Repository:   https://github.com/taleblou/momijo

from collections.list import List
from momijo.tensor import tensor   # optional Tensor overloads

# ------------------------------------------------------------
# Small math helpers (backend-agnostic)
# ------------------------------------------------------------

fn _sqrt64(x: Float32) -> Float32:
    if x <= 0.0:
        return 0.0
    var g: Float32 = x
    var i: Int = 0
    while i < 12:
        g = 0.5 * (g + x / g)
        i = i + 1
    return g

fn _prod_int(xs: List[Int], start: Int, end_exclusive: Int) -> Int:
    var out: Int = 1
    var i: Int = start
    while i < end_exclusive:
        out = out * xs[i]
        i = i + 1
    return out

# ------------------------------------------------------------
# Fan-in / Fan-out
#   - Linear:    [out_features, in_features]
#   - ConvNd:    [out_channels, in_channels, k1, k2, ...]
# ------------------------------------------------------------
fn compute_fan_in_out(shape: List[Int]) -> (Int, Int):
    var n: Int = len(shape)
    if n == 0:
        return (1, 1)
    if n == 1:
        var fan: Int = shape[0]
        return (fan, fan)
    var fan_out: Int = shape[0]
    var fan_in: Int = shape[1]
    if n > 2:
        var receptive: Int = _prod_int(shape, 2, n)
        fan_in = fan_in * receptive
        fan_out = fan_out * receptive
    return (fan_in, fan_out)

# ------------------------------------------------------------
# Nonlinearity gain (similar to torch.nn.init.calculate_gain)
# ------------------------------------------------------------
fn calculate_gain(nonlinearity: String, param: Float32 = 0.0) -> Float32:
    if nonlinearity == String("linear"):
        return 1.0
    if nonlinearity == String("sigmoid"):
        return 1.0
    if nonlinearity == String("tanh"):
        return 5.0 / 3.0
    if nonlinearity == String("relu"):
        return _sqrt64(2.0)
    if nonlinearity == String("leaky_relu"):
        return _sqrt64(2.0 / (1.0 + (param * param)))
    if nonlinearity == String("selu"):
        return 0.75
    if nonlinearity == String("gelu"):
        return 1.0
    if nonlinearity == String("silu"):
        return 1.0
    return 1.0

# ------------------------------------------------------------
# Kaiming (He) — Uniform bounds and Normal std
# ------------------------------------------------------------
fn kaiming_uniform_bounds(
    shape: List[Int],
    a: Float32 = 0.0,
    mode: String = String("fan_in"),
    nonlinearity: String = String("leaky_relu")
) -> (Float32, Float32):
    var (fan_in, fan_out) = compute_fan_in_out(shape)
    var fan: Int = fan_in
    if mode == String("fan_out"):
        fan = fan_out
    var gain: Float32 = calculate_gain(nonlinearity, a)
    var denom: Float32 = Float32(fan)
    if denom <= 0.0:
        denom = 1.0
    var std: Float32 = gain / _sqrt64(denom)
    var bound: Float32 = _sqrt64(3.0) * std
    return (-bound, +bound)

fn kaiming_normal_std(
    shape: List[Int],
    a: Float32 = 0.0,
    mode: String = String("fan_in"),
    nonlinearity: String = String("leaky_relu")
) -> Float32:
    var (fan_in, fan_out) = compute_fan_in_out(shape)
    var fan: Int = fan_in
    if mode == String("fan_out"):
        fan = fan_out
    var gain: Float32 = calculate_gain(nonlinearity, a)
    var denom: Float32 = Float32(fan)
    if denom <= 0.0:
        denom = 1.0
    return gain / _sqrt64(denom)

# ------------------------------------------------------------
# Xavier (Glorot) — Uniform bounds and Normal std
# ------------------------------------------------------------
fn xavier_uniform_bounds(shape: List[Int], gain: Float32 = 1.0) -> (Float32, Float32):
    var (fan_in, fan_out) = compute_fan_in_out(shape)
    var denom: Float32 = Float32(fan_in + fan_out)
    if denom <= 0.0:
        denom = 1.0
    var bound: Float32 = gain * _sqrt64(6.0 / denom)
    return (-bound, +bound)

fn xavier_normal_std(shape: List[Int], gain: Float32 = 1.0) -> Float32:
    var (fan_in, fan_out) = compute_fan_in_out(shape)
    var denom: Float32 = Float32(fan_in + fan_out)
    if denom <= 0.0:
        denom = 1.0
    # std = gain * sqrt(2 / (fan_in + fan_out))
    return gain * _sqrt64(2.0 / denom)

# ------------------------------------------------------------
# Tiny RNG (bootstrap only) + distributions
# ------------------------------------------------------------
struct SimpleLCG:
    var state: UInt64

    fn __init__(out self, seed: UInt64 = UInt64(88172645463393265)):
        self.state = seed

    fn _next_u64(mut self) -> UInt64:
        self.state = UInt64(6364136223846793005) * self.state + UInt64(1)
        return self.state

    fn rand01(mut self) -> Float32:
        var x: UInt64 = self._next_u64()
        var top53: UInt64 = (x >> UInt64(11))
        var denom: Float32 = 9007199254740992.0  # 2**53
        return Float32(top53) / denom

    fn uniform(mut self, low: Float32, high: Float32) -> Float32:
        var u: Float32 = self.rand01()
        return low + (high - low) * u

    # Standard normal N(0,1) via CLT approximation: sum(U[0,1]) over 12 draws - 6
    fn normal01(mut self) -> Float32:
        var s: Float32 = 0.0
        var i: Int = 0
        while i < 12:
            s = s + self.rand01()
            i = i + 1
        return s - 6.0

    fn normal(mut self, mean: Float32, std: Float32) -> Float32:
        return mean + std * self.normal01()

# ------------------------------------------------------------
# Fillers (Float32 list bootstrap)
# ------------------------------------------------------------
fn _fill_uniform64(mut arr: List[Float32], low: Float32, high: Float32, mut rng: SimpleLCG):
    var n: Int = len(arr)
    var i: Int = 0
    while i < n:
        arr[i] = rng.uniform(low, high)
        i = i + 1

fn _fill_normal64(mut arr: List[Float32], mean: Float32, std: Float32, mut rng: SimpleLCG):
    var n: Int = len(arr)
    var i: Int = 0
    while i < n:
        arr[i] = rng.normal(mean, std)
        i = i + 1

# Public: Kaiming/Xavier — Uniform (List)
fn kaiming_uniform_fill(
    mut arr: List[Float32],
    shape: List[Int],
    a: Float32 = 0.0,
    mode: String = String("fan_in"),
    nonlinearity: String = String("leaky_relu"),
    mut rng: SimpleLCG = SimpleLCG()
):
    var (low, high) = kaiming_uniform_bounds(shape, a, mode, nonlinearity)
    _fill_uniform64(arr, low, high, rng)

fn xavier_uniform_fill(
    mut arr: List[Float32],
    shape: List[Int],
    gain: Float32 = 1.0,
    mut rng: SimpleLCG = SimpleLCG()
):
    var (low, high) = xavier_uniform_bounds(shape, gain)
    _fill_uniform64(arr, low, high, rng)

# Public: Kaiming/Xavier — Normal (List)
fn kaiming_normal_fill(
    mut arr: List[Float32],
    shape: List[Int],
    a: Float32 = 0.0,
    mode: String = String("fan_in"),
    nonlinearity: String = String("leaky_relu"),
    mean: Float32 = 0.0,
    mut rng: SimpleLCG = SimpleLCG()
):
    var std: Float32 = kaiming_normal_std(shape, a, mode, nonlinearity)
    _fill_normal64(arr, mean, std, rng)

fn xavier_normal_fill(
    mut arr: List[Float32],
    shape: List[Int],
    gain: Float32 = 1.0,
    mean: Float32 = 0.0,
    mut rng: SimpleLCG = SimpleLCG()
):
    var std: Float32 = xavier_normal_std(shape, gain)
    _fill_normal64(arr, mean, std, rng)

# ------------------------------------------------------------
# Optional Tensor overloads (Float32) — use when tensors are desired
# ------------------------------------------------------------
fn _numel_from_shape(shape: List[Int]) -> Int:
    var n: Int = 1
    var i: Int = 0
    while i < len(shape):
        n = n * shape[i]
        i = i + 1
    if n <= 0:
        n = 1
    return n

fn kaiming_uniform_fill_tensor(
    mut t: tensor.Tensor[Float32],
    a: Float32 = 0.0,
    mode: String = String("fan_in"),
    nonlinearity: String = String("leaky_relu"),
    mut rng: SimpleLCG = SimpleLCG()
):
    var shp = t.shape()
    var (low, high) = kaiming_uniform_bounds(shp, a, mode, nonlinearity)
    var n = _numel_from_shape(shp)
    var k = 0
    while k < n:
        t._data[k] = rng.uniform(low, high)   # relies on public _data in momijo.tensor
        k = k + 1

fn xavier_uniform_fill_tensor(
    mut t: tensor.Tensor[Float32],
    gain: Float32 = 1.0,
    mut rng: SimpleLCG = SimpleLCG()
):
    var shp = t.shape()
    var (low, high) = xavier_uniform_bounds(shp, gain)
    var n = _numel_from_shape(shp)
    var k = 0
    while k < n:
        t._data[k] = rng.uniform(low, high)
        k = k + 1

fn kaiming_normal_fill_tensor(
    mut t: tensor.Tensor[Float32],
    a: Float32 = 0.0,
    mode: String = String("fan_in"),
    nonlinearity: String = String("leaky_relu"),
    mean: Float32 = 0.0,
    mut rng: SimpleLCG = SimpleLCG()
):
    var shp = t.shape()
    var std = kaiming_normal_std(shp, a, mode, nonlinearity)
    var n = _numel_from_shape(shp)
    var k = 0
    while k < n:
        t._data[k] = rng.normal(mean, std)
        k = k + 1

fn xavier_normal_fill_tensor(
    mut t: tensor.Tensor[Float32],
    gain: Float32 = 1.0,
    mean: Float32 = 0.0,
    mut rng: SimpleLCG = SimpleLCG()
):
    var shp = t.shape()
    var std = xavier_normal_std(shp, gain)
    var n = _numel_from_shape(shp)
    var k = 0
    while k < n:
        t._data[k] = rng.normal(mean, std)
        k = k + 1
