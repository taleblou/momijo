# Project:      Momijo
# Module:       src.momijo.nn.optim.rmsprop
# File:         rmsprop.mojo
# Path:         src/momijo/nn/optim/rmsprop.mojo
#
# Description:  Neural-network utilities for Momijo integrating with tensors,
#               optimizers, and training/evaluation loops.
#
# Author(s):    Morteza Taleblou & Mitra Daneshmand
# Website:      https://taleblou.ir/
# Repository:   https://github.com/taleblou/momijo
#
# License:      MIT License
# SPDX-License-Identifier: MIT
# Copyright:    (c) 2025 Morteza Taleblou & Mitra Daneshmand
#
# Notes:
#   - Structs: Param1D, Param2D, RMSprop
#   - Key functions: zeros1d, zeros2d, _sqrt_pos, _abs, __init__, __copyinit__, __moveinit__, __init__ ...
#   - Uses generic functions/types with explicit trait bounds.


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
fn _sqrt_pos(x: Float64) -> Float64:
    if x <= 0.0: return 0.0
    var s = x
    s = 0.5 * (s + x / s)
    s = 0.5 * (s + x / s)
    return s
fn _abs(x: Float64) -> Float64:
    if x < 0.0: return -x
    return x

# --- Params ---
struct Param1D:
    var data: List[Float64]
    var grad: List[Float64]
fn __init__(out self, data: List[Float64]) -> None:
        self.data = data
        self.grad = zeros1d(len(data))
fn __copyinit__(out self, other: Self) -> None:
        self.data = other.data
        self.grad = other.grad
fn __moveinit__(out self, deinit other: Self) -> None:
        self.data = other.data
        self.grad = other.grad
struct Param2D:
    var data: List[List[Float64]]
    var grad: List[List[Float64]]
fn __init__(out self, data: List[List[Float64]]) -> None:
        var R = len(data); var C = 0
        if R > 0: C = len(data[0])
        self.data = data
        self.grad = zeros2d(R, C)
fn __copyinit__(out self, other: Self) -> None:
        self.data = other.data
        self.grad = other.grad
fn __moveinit__(out self, deinit other: Self) -> None:
        self.data = other.data
        self.grad = other.grad
fn zero_grad1d(mut p: Param1D) -> None:
    for i in range(len(p.grad)): p.grad[i] = 0.0
fn zero_grad2d(mut p: Param2D) -> None:
    var R = len(p.grad)
    if R == 0: return
    var C = len(p.grad[0])
    for i in range(R):
        for j in range(C): p.grad[i][j] = 0.0

# --- RMSprop ---
struct RMSprop:
    var lr: Float64         # learning rate
    var alpha: Float64      # decay for square avg
    var eps: Float64
    var weight_decay: Float64  # L2 (coupled) decay
    var momentum: Float64
    var centered: Bool

    var p1: List[Param1D]
    var square1: List[List[Float64]]
    var buf1: List[List[Float64]]      # momentum
    var mean1: List[List[Float64]]     # for centered

    var p2: List[Param2D]
    var square2: List[List[List[Float64]]]
    var buf2: List[List[List[Float64]]]
    var mean2: List[List[List[Float64]]]
fn __init__(out self, lr: Float64 = 1e-3, alpha: Float64 = 0.99, eps: Float64 = 1e-8, weight_decay: Float64 = 0.0, momentum: Float64 = 0.0, centered: Bool = False) -> None:
        self.lr = lr
        self.alpha = alpha
        self.eps = eps
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.centered = centered
        self.p1 = List[Param1D]()
        self.square1 = List[List[Float64]]()
        self.buf1 = List[List[Float64]]()
        self.mean1 = List[List[Float64]]()
        self.p2 = List[Param2D]()
        self.square2 = List[List[List[Float64]]]]()
        self.buf2 = List[List[List[Float64]]]]()
        self.mean2 = List[List[List[Float64]]]]()
fn set_lr(mut self, lr: Float64) -> None: self.lr = lr
fn set_weight_decay(mut self, wd: Float64) -> None: self.weight_decay = wd
fn add_param1d(mut self, p: Param1D) -> None:
        self.p1.push(p)
        self.square1.push(zeros1d(len(p.data)))
        self.buf1.push(zeros1d(len(p.data)))
        self.mean1.push(zeros1d(len(p.data)))
fn add_param2d(mut self, p: Param2D) -> None:
        self.p2.push(p)
        var R = len(p.data); var C = 0
        if R > 0: C = len(p.data[0])
        self.square2.push(zeros2d(R, C))
        self.buf2.push(zeros2d(R, C))
        self.mean2.push(zeros2d(R, C))
fn zero_grad(mut self) -> None:
        for i in range(len(self.p1)): zero_grad1d(self.p1[i])
        for i in range(len(self.p2)): zero_grad2d(self.p2[i])
fn reset_state(mut self) -> None:
        # reinitialize accumulators to zeros with current param shapes
        self.square1 = List[List[Float64]]()
        self.buf1 = List[List[Float64]]()
        self.mean1 = List[List[Float64]]()
        for i in range(len(self.p1)):
            self.square1.push(zeros1d(len(self.p1[i].data)))
            self.buf1.push(zeros1d(len(self.p1[i].data)))
            self.mean1.push(zeros1d(len(self.p1[i].data)))
        self.square2 = List[List[List[Float64]]]]()
        self.buf2 = List[List[List[Float64]]]]()
        self.mean2 = List[List[List[Float64]]]]()
        for i in range(len(self.p2)):
            var R = len(self.p2[i].data); var C = 0
            if R > 0: C = len(self.p2[i].data[0])
            self.square2.push(zeros2d(R, C))
            self.buf2.push(zeros2d(R, C))
            self.mean2.push(zeros2d(R, C))
fn _step1d(mut self, idx: Int) -> None:
        var p = self.p1[idx]
        var sq = self.square1[idx]
        var buf = self.buf1[idx]
        var mn = self.mean1[idx]
        var n = len(p.data)
        var a = self.alpha; var lr = self.lr; var wd = self.weight_decay; var mom = self.momentum; var eps = self.eps
        for i in range(n):
            var g = p.grad[i]
            if wd != 0.0: g += wd * p.data[i]   # L2 weight decay (coupled)
            sq[i] = a * sq[i] + (1.0 - a) * (g * g)
            var denom = 0.0
            if self.centered:
                mn[i] = a * mn[i] + (1.0 - a) * g
                var var_t = sq[i] - mn[i] * mn[i]
                if var_t < 0.0: var_t = 0.0
                denom = _sqrt_pos(var_t) + eps
            else:
                denom = _sqrt_pos(sq[i]) + eps
            var grad_scaled = g / denom
            if mom > 0.0:
                buf[i] = mom * buf[i] + grad_scaled
                p.data[i] = p.data[i] - lr * buf[i]
            else:
                p.data[i] = p.data[i] - lr * grad_scaled
        self.p1[idx] = p; self.square1[idx] = sq; self.buf1[idx] = buf; self.mean1[idx] = mn
fn _step2d(mut self, idx: Int) -> None:
        var p = self.p2[idx]
        var sq = self.square2[idx]
        var buf = self.buf2[idx]
        var mn = self.mean2[idx]
        var R = len(p.data)
        if R == 0: return
        var C = len(p.data[0])
        var a = self.alpha; var lr = self.lr; var wd = self.weight_decay; var mom = self.momentum; var eps = self.eps
        for i in range(R):
            for j in range(C):
                var g = p.grad[i][j]
                if wd != 0.0: g += wd * p.data[i][j]
                sq[i][j] = a * sq[i][j] + (1.0 - a) * (g * g)
                var denom = 0.0
                if self.centered:
                    mn[i][j] = a * mn[i][j] + (1.0 - a) * g
                    var var_t = sq[i][j] - mn[i][j] * mn[i][j]
                    if var_t < 0.0: var_t = 0.0
                    denom = _sqrt_pos(var_t) + eps
                else:
                    denom = _sqrt_pos(sq[i][j]) + eps
                var grad_scaled = g / denom
                if mom > 0.0:
                    buf[i][j] = mom * buf[i][j] + grad_scaled
                    p.data[i][j] = p.data[i][j] - lr * buf[i][j]
                else:
                    p.data[i][j] = p.data[i][j] - lr * grad_scaled
        self.p2[idx] = p; self.square2[idx] = sq; self.buf2[idx] = buf; self.mean2[idx] = mn
fn step(mut self) -> None:
        for i in range(len(self.p1)): self._step1d(i)
        for i in range(len(self.p2)): self._step2d(i)
fn __copyinit__(out self, other: Self) -> None:
        self.lr = other.lr
        self.alpha = other.alpha
        self.eps = other.eps
        self.weight_decay = other.weight_decay
        self.momentum = other.momentum
        self.centered = other.centered
        self.p1 = other.p1
        self.square1 = other.square1
        self.buf1 = other.buf1
        self.mean1 = other.mean1
        self.p2 = other.p2
        self.square2 = other.square2
        self.buf2 = other.buf2
        self.mean2 = other.mean2
fn __moveinit__(out self, deinit other: Self) -> None:
        self.lr = other.lr
        self.alpha = other.alpha
        self.eps = other.eps
        self.weight_decay = other.weight_decay
        self.momentum = other.momentum
        self.centered = other.centered
        self.p1 = other.p1
        self.square1 = other.square1
        self.buf1 = other.buf1
        self.mean1 = other.mean1
        self.p2 = other.p2
        self.square2 = other.square2
        self.buf2 = other.buf2
        self.mean2 = other.mean2
# --- Smoke tests ---
fn _self_test() -> Bool:
    var ok = True

    # 1D quadratic f(w)=0.5*||w||^2 -> grad = w
    var w = List[Float64]([1.0, -2.0, 3.0, -4.0])
    var p = Param1D(w)
    var opt = RMSprop(1e-2, 0.99, 1e-8, 0.0, 0.9, False)  # momentum
    opt.add_param1d(p)
    for t in range(200):
        for i in range(len(opt.p1[0].data)):
            opt.p1[0].grad[i] = opt.p1[0].data[i]
        opt.step()
    var s = 0.0
    for i in range(len(opt.p1[0].data)): s += _abs(opt.p1[0].data[i])
    ok = ok and (s < 8.0)  # initial sum abs = 10

    # 2D centered variant
    var M = List[List[Float64]]()
    var r0 = List[Float64]([0.5, -0.5]); var r1 = List[Float64]([1.0, -1.0])
    M.push(r0); M.push(r1)
    var p2 = Param2D(M)
    var opt2 = RMSprop(5e-3, 0.99, 1e-8, 0.01, 0.0, True)  # centered + wd
    opt2.add_param2d(p2)
    for t in range(200):
        for i in range(len(opt2.p2[0].data)):
            for j in range(len(opt2.p2[0].data[0])):
                opt2.p2[0].grad[i][j] = opt2.p2[0].data[i][j]
        opt2.step()
    var s2 = 0.0
    for i in range(len(opt2.p2[0].data)):
        for j in range(len(opt2.p2[0].data[0])):
            s2 += _abs(opt2.p2[0].data[i][j])
    ok = ok and (s2 < 2.8)  # initial sum abs = 3.0

    return ok