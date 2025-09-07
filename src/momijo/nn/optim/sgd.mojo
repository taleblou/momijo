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
# Project: momijo.nn.optim
# File: src/momijo/nn/optim/sgd.mojo

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

# --- SGD ---
struct SGD:
    var lr: Float64
    var momentum: Float64
    var dampening: Float64
    var weight_decay: Float64  # scalar
    var nesterov: Bool
    var decoupled_wd: Bool     # if true, use decoupled weight decay

    var p1: List[Param1D]
    var buf1: List[List[Float64]]  # momentum buffers

    var p2: List[Param2D]
    var buf2: List[List[List[Float64]]]
fn __init__(out self, lr: Float64 = 1e-2, momentum: Float64 = 0.0, dampening: Float64 = 0.0, weight_decay: Float64 = 0.0, nesterov: Bool = False, decoupled_wd: Bool = False) -> None:
        self.lr = lr
        self.momentum = momentum
        self.dampening = dampening
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        self.decoupled_wd = decoupled_wd
        self.p1 = List[Param1D]()
        self.buf1 = List[List[Float64]]()
        self.p2 = List[Param2D]()
        self.buf2 = List[List[List[Float64]]]]()
fn set_lr(mut self, lr: Float64) -> None: self.lr = lr
fn set_weight_decay(mut self, wd: Float64) -> None: self.weight_decay = wd
fn set_momentum(mut self, m: Float64) -> None: self.momentum = m
fn add_param1d(mut self, p: Param1D) -> None:
        self.p1.push(p)
        self.buf1.push(zeros1d(len(p.data)))
fn add_param2d(mut self, p: Param2D) -> None:
        self.p2.push(p)
        var R = len(p.data); var C = 0
        if R > 0: C = len(p.data[0])
        self.buf2.push(zeros2d(R, C))
fn zero_grad(mut self) -> None:
        for i in range(len(self.p1)): zero_grad1d(self.p1[i])
        for i in range(len(self.p2)): zero_grad2d(self.p2[i])
fn reset_state(mut self) -> None:
        self.buf1 = List[List[Float64]]()
        for i in range(len(self.p1)): self.buf1.push(zeros1d(len(self.p1[i].data)))
        self.buf2 = List[List[List[Float64]]]]()
        for i in range(len(self.p2)):
            var R = len(self.p2[i].data); var C = 0
            if R > 0: C = len(self.p2[i].data[0])
            self.buf2.push(zeros2d(R, C))
fn _update1d(mut self, idx: Int) -> None:
        var p = self.p1[idx]
        var buf = self.buf1[idx]
        var n = len(p.data)
        var lr = self.lr; var m = self.momentum; var d = self.dampening; var wd = self.weight_decay
        var use_m = (m > 0.0)
        for i in range(n):
            var g = p.grad[i]
            if not self.decoupled_wd and wd != 0.0:
                g += wd * p.data[i]
            if use_m:
                buf[i] = m * buf[i] + (1.0 - d) * g
                var step_dir = buf[i]
                if self.nesterov:
                    step_dir = g + m * buf[i]
                p.data[i] = p.data[i] - lr * step_dir
            else:
                p.data[i] = p.data[i] - lr * g
            if self.decoupled_wd and wd != 0.0:
                p.data[i] = p.data[i] - lr * wd * p.data[i]
        self.p1[idx] = p
        self.buf1[idx] = buf
fn _update2d(mut self, idx: Int) -> None:
        var p = self.p2[idx]
        var buf = self.buf2[idx]
        var R = len(p.data)
        if R == 0: return
        var C = len(p.data[0])
        var lr = self.lr; var m = self.momentum; var d = self.dampening; var wd = self.weight_decay
        var use_m = (m > 0.0)
        for i in range(R):
            for j in range(C):
                var g = p.grad[i][j]
                if not self.decoupled_wd and wd != 0.0:
                    g += wd * p.data[i][j]
                if use_m:
                    buf[i][j] = m * buf[i][j] + (1.0 - d) * g
                    var step_dir = buf[i][j]
                    if self.nesterov:
                        step_dir = g + m * buf[i][j]
                    p.data[i][j] = p.data[i][j] - lr * step_dir
                else:
                    p.data[i][j] = p.data[i][j] - lr * g
                if self.decoupled_wd and wd != 0.0:
                    p.data[i][j] = p.data[i][j] - lr * wd * p.data[i][j]
        self.p2[idx] = p
        self.buf2[idx] = buf
fn step(mut self) -> None:
        for i in range(len(self.p1)): self._update1d(i)
        for i in range(len(self.p2)): self._update2d(i)
fn __copyinit__(out self, other: Self) -> None:
        self.lr = other.lr
        self.momentum = other.momentum
        self.dampening = other.dampening
        self.weight_decay = other.weight_decay
        self.nesterov = other.nesterov
        self.decoupled_wd = other.decoupled_wd
        self.p1 = other.p1
        self.buf1 = other.buf1
        self.p2 = other.p2
        self.buf2 = other.buf2
fn __moveinit__(out self, deinit other: Self) -> None:
        self.lr = other.lr
        self.momentum = other.momentum
        self.dampening = other.dampening
        self.weight_decay = other.weight_decay
        self.nesterov = other.nesterov
        self.decoupled_wd = other.decoupled_wd
        self.p1 = other.p1
        self.buf1 = other.buf1
        self.p2 = other.p2
        self.buf2 = other.buf2
# --- Smoke tests ---
fn _self_test() -> Bool:
    var ok = True

    # 1D quadratic f(w)=0.5*||w||^2 -> grad = w (no momentum)
    var w = List[Float64]([1.0, -2.0, 3.0, -4.0])
    var p = Param1D(w)
    var opt = SGD(1e-2, 0.0, 0.0, 0.0, False, False)
    opt.add_param1d(p)
    for t in range(200):
        for i in range(len(opt.p1[0].data)):
            opt.p1[0].grad[i] = opt.p1[0].data[i]
        opt.step()
    var s = 0.0
    for i in range(len(opt.p1[0].data)): s += _abs(opt.p1[0].data[i])
    ok = ok and (s < 9.0)  # initial sum abs = 10

    # 1D with momentum + Nesterov
    var w2 = List[Float64]([1.0, 1.0, 1.0, 1.0])
    var p2 = Param1D(w2)
    var opt2 = SGD(1e-2, 0.9, 0.0, 0.0, True, False)
    opt2.add_param1d(p2)
    for t in range(200):
        for i in range(len(opt2.p1[0].data)):
            opt2.p1[0].grad[i] = opt2.p1[0].data[i]
        opt2.step()
    var s2 = 0.0
    for i in range(len(opt2.p1[0].data)): s2 += _abs(opt2.p1[0].data[i])
    ok = ok and (s2 < 3.5)  # initial = 4.0

    # 2D + decoupled weight decay
    var M = List[List[Float64]]()
    var r0 = List[Float64]([0.5, -0.5]); var r1 = List[Float64]([1.0, -1.0])
    M.push(r0); M.push(r1)
    var pM = Param2D(M)
    var opt3 = SGD(5e-3, 0.0, 0.0, 0.1, False, True)  # decoupled wd
    opt3.add_param2d(pM)
    for t in range(300):
        for i in range(len(opt3.p2[0].data)):
            for j in range(len(opt3.p2[0].data[0])):
                opt3.p2[0].grad[i][j] = opt3.p2[0].data[i][j]
        opt3.step()
    var s3 = 0.0
    for i in range(len(opt3.p2[0].data)):
        for j in range(len(opt3.p2[0].data[0])):
            s3 += _abs(opt3.p2[0].data[i][j])
    ok = ok and (s3 < 2.9)  # initial sum abs = 3.0

    return ok