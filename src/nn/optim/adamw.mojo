# MIT License
# Copyright (c) 2025
# SPDX-License-Identifier: MIT
#
# Module: momijo.nn.adamw
# Path:   src/momijo/nn/adamw.mojo
#
# Minimal AdamW optimizer (decoupled weight decay) for pedagogy/smoke tests.
# Dependency-light, list-based Float64 implementation.
#
# Features:
#   - Supports 1D and 2D parameter tensors (Param1D / Param2D)
#   - Bias correction for moments (timestep-based)
#   - Decoupled weight decay (AdamW): param -= lr * (mhat/(sqrt(vhat)+eps) + wd*param)
#   - Utilities: zero_grad(), set_lr(), set_weight_decay(), reset_state()
#
# Notes:
#   - This is a toy implementation to keep examples runnable without external deps.
#   - No mixed precision, no gradient scaling, no parameter groups.

# --- Helpers ---
fn zeros1d(n: Int) -> List[Float64]:
    var y = List[Float64]()
    for i in range(n): y.push(0.0)
    return y

fn zeros2d(r: Int, c: Int) -> List[List[Float64]]:
    var y = List[List[Float64]]()
    for i in range(r):
        var row = List[Float64]()
        for j in range(c): row.push(0.0)
        y.push(row)
    return y

fn copy1d(x: List[Float64]) -> List[Float64]:
    var n = len(x)
    var y = List[Float64]()
    for i in range(n): y.push(x[i])
    return y

fn copy2d(x: List[List[Float64]]) -> List[List[Float64]]:
    var r = len(x)
    var y = List[List[Float64]]()
    for i in range(r):
        var row = List[Float64]()
        var c = 0
        if len(x[i]) > 0: c = len(x[i])
        for j in range(c): row.push(x[i][j])
        y.push(row)
    return y

fn _sqrt_pos(x: Float64) -> Float64:
    if x <= 0.0: return 0.0
    var s = x
    s = 0.5 * (s + x / s)
    s = 0.5 * (s + x / s)
    return s

fn _pow(a: Float64, k: Int) -> Float64:
    var p = 1.0
    for i in range(k): p *= a
    return p

fn _abs(x: Float64) -> Float64:
    if x < 0.0: return -x
    return x

# --- Parameter containers ---
struct Param1D:
    var data: List[Float64]
    var grad: List[Float64]

    fn __init__(out self, data: List[Float64]):
        self.data = data
        self.grad = zeros1d(len(data))

struct Param2D:
    var data: List[List[Float64]]
    var grad: List[List[Float64]]

    fn __init__(out self, data: List[List[Float64]]):
        var R = len(data)
        var C = 0
        if R > 0: C = len(data[0])
        self.data = data
        self.grad = zeros2d(R, C)

fn zero_grad1d(mut p: Param1D):
    var n = len(p.grad)
    for i in range(n): p.grad[i] = 0.0

fn zero_grad2d(mut p: Param2D):
    var R = len(p.grad)
    if R == 0: return
    var C = len(p.grad[0])
    for i in range(R):
        for j in range(C):
            p.grad[i][j] = 0.0

# --- AdamW ---
struct AdamW:
    var lr: Float64
    var beta1: Float64
    var beta2: Float64
    var eps: Float64
    var weight_decay: Float64
    var t: Int  # time step (starts at 0)

    # parameter lists and optimizer state
    var p1: List[Param1D]
    var m1: List[List[Float64]]  # same shapes as data
    var v1: List[List[Float64]]

    var p2: List[Param2D]
    var m2: List[List[List[Float64]]]
    var v2: List[List[List[Float64]]]

    fn __init__(out self, lr: Float64 = 1e-3, beta1: Float64 = 0.9, beta2: Float64 = 0.999, eps: Float64 = 1e-8, weight_decay: Float64 = 0.0):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        self.p1 = List[Param1D]()
        self.m1 = List[List[Float64]]()
        self.v1 = List[List[Float64]]()
        self.p2 = List[Param2D]()
        self.m2 = List[List[List[Float64]]]()
        self.v2 = List[List[List[Float64]]]()

    fn set_lr(mut self, lr: Float64):
        self.lr = lr

    fn set_weight_decay(mut self, wd: Float64):
        self.weight_decay = wd

    fn reset_state(mut self):
        self.t = 0
        self.m1 = List[List[Float64]]()
        self.v1 = List[List[Float64]]()
        for i in range(len(self.p1)):
            self.m1.push(zeros1d(len(self.p1[i].data)))
            self.v1.push(zeros1d(len(self.p1[i].data)))
        self.m2 = List[List[List[Float64]]]()
        self.v2 = List[List[List[Float64]]]()
        for i in range(len(self.p2)):
            var R = len(self.p2[i].data)
            var C = 0
            if R > 0: C = len(self.p2[i].data[0])
            self.m2.push(zeros2d(R, C))
            self.v2.push(zeros2d(R, C))

    fn add_param1d(mut self, p: Param1D):
        self.p1.push(p)
        self.m1.push(zeros1d(len(p.data)))
        self.v1.push(zeros1d(len(p.data)))

    fn add_param2d(mut self, p: Param2D):
        self.p2.push(p)
        var R = len(p.data)
        var C = 0
        if R > 0: C = len(p.data[0])
        self.m2.push(zeros2d(R, C))
        self.v2.push(zeros2d(R, C))

    fn zero_grad(mut self):
        for i in range(len(self.p1)):
            zero_grad1d(self.p1[i])
        for i in range(len(self.p2)):
            zero_grad2d(self.p2[i])

    fn _step_param1d(mut self, idx: Int):
        var p = self.p1[idx]
        var m = self.m1[idx]
        var v = self.v1[idx]
        var n = len(p.data)
        var b1 = self.beta1; var b2 = self.beta2; var lr = self.lr; var wd = self.weight_decay; var eps = self.eps
        for i in range(n):
            var g = p.grad[i]
            # moments
            m[i] = b1 * m[i] + (1.0 - b1) * g
            v[i] = b2 * v[i] + (1.0 - b2) * (g * g)
            # bias correction
            var mhat = m[i] / (1.0 - _pow(b1, self.t))
            var vhat = v[i] / (1.0 - _pow(b2, self.t))
            var denom = _sqrt_pos(vhat) + eps
            var update = mhat / denom + wd * p.data[i]
            p.data[i] = p.data[i] - lr * update
        # write back
        self.p1[idx] = p
        self.m1[idx] = m
        self.v1[idx] = v

    fn _step_param2d(mut self, idx: Int):
        var p = self.p2[idx]
        var m = self.m2[idx]
        var v = self.v2[idx]
        var R = len(p.data)
        if R == 0: return
        var C = len(p.data[0])
        var b1 = self.beta1; var b2 = self.beta2; var lr = self.lr; var wd = self.weight_decay; var eps = self.eps
        for i in range(R):
            for j in range(C):
                var g = p.grad[i][j]
                m[i][j] = b1 * m[i][j] + (1.0 - b1) * g
                v[i][j] = b2 * v[i][j] + (1.0 - b2) * (g * g)
                var mhat = m[i][j] / (1.0 - _pow(b1, self.t))
                var vhat = v[i][j] / (1.0 - _pow(b2, self.t))
                var denom = _sqrt_pos(vhat) + eps
                var update = mhat / denom + wd * p.data[i][j]
                p.data[i][j] = p.data[i][j] - lr * update
        self.p2[idx] = p
        self.m2[idx] = m
        self.v2[idx] = v

    fn step(mut self):
        # increase time step first (as in many frameworks)
        self.t += 1
        if self.t < 1: self.t = 1
        for i in range(len(self.p1)):
            self._step_param1d(i)
        for i in range(len(self.p2)):
            self._step_param2d(i)

# --- Smoke tests ---
fn _self_test() -> Bool:
    var ok = True

    # Quadratic: f(w)=0.5*||w||^2 -> grad = w ; should decay towards zero
    var w = List[Float64]([1.0, -2.0, 3.0, -4.0])
    var p = Param1D(w)
    var opt = AdamW(1e-2, 0.9, 0.999, 1e-8, 0.01)
    opt.add_param1d(p)
    for t in range(100):
        # grad = w (current data)
        for i in range(len(opt.p1[0].data)):
            opt.p1[0].grad[i] = opt.p1[0].data[i]
        opt.step()
    var s = 0.0
    for i in range(len(opt.p1[0].data)):
        s += _abs(opt.p1[0].data[i])
    ok = ok and (s < 5.0)  # initial sum abs = 10.0

    # 2D test on a tiny matrix
    var M = List[List[Float64]]()
    var r0 = List[Float64]([0.5, -0.5]); var r1 = List[Float64]([1.0, -1.0])
    M.push(r0); M.push(r1)
    var p2 = Param2D(M)
    var opt2 = AdamW(5e-3, 0.9, 0.999, 1e-8, 0.0)
    opt2.add_param2d(p2)
    for t in range(50):
        for i in range(len(opt2.p2[0].data)):
            for j in range(len(opt2.p2[0].data[0])):
                opt2.p2[0].grad[i][j] = opt2.p2[0].data[i][j]
        opt2.step()
    var s2 = 0.0
    for i in range(len(opt2.p2[0].data)):
        for j in range(len(opt2.p2[0].data[0])):
            s2 += _abs(opt2.p2[0].data[i][j])
    ok = ok and (s2 < 3.0)  # initial sum abs = 3.0

    return ok

fn main():
    if _self_test():
        print(String("momijo.nn.adamw: OK"))
    else:
        print(String("momijo.nn.adamw: FAILED"))
