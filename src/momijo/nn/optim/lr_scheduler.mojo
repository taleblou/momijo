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
# File: src/momijo/nn/optim/lr_scheduler.mojo

fn _abs(x: Float64) -> Float64:
    if x < 0.0: return -x
    return x
fn _powf(a: Float64, k: Int) -> Float64:
    var p = 1.0
    var n = k
    if n < 0:
        n = -n
        for i in range(n): p *= a
        if p == 0.0: return 0.0
        return 1.0 / p
    for i in range(n): p *= a
    return p
fn _pi() -> Float64: return 3.141592653589793
fn _two_pi() -> Float64: return 6.283185307179586
fn _wrap_pi(x: Float64) -> Float64:
    var y = x
    var two_pi = _two_pi()
    if y > 1e6 or y < -1e6:  # avoid huge loops; crude clamp
        y = 0.0
    while y > _pi(): y -= two_pi
    while y < -_pi(): y += two_pi
    return y

# cos via 8th-order taylor around 0 with basic range reduction
fn _cos(x: Float64) -> Float64:
    var t = _wrap_pi(x)
    var t2 = t * t
    var t4 = t2 * t2
    var t6 = t4 * t2
    var t8 = t4 * t4
    return 1.0 - t2 / 2.0 + t4 / 24.0 - t6 / 720.0 + t8 / 40320.0
fn _max(a: Float64, b: Float64) -> Float64:
    if a >= b: return a
    return b
fn _min(a: Float64, b: Float64) -> Float64:
    if a <= b: return a
    return b
fn _floor_div(a: Int, b: Int) -> Int:
    if b == 0: return 0
    var q = a / b
    return q

# --- Base concept (just a pattern; not inherited) ---
# Each scheduler keeps: base_lr, current_lr, last_epoch (Int)

# --- StepLR ---
struct StepLR:
    var base_lr: Float64
    var step_size: Int
    var gamma: Float64
    var last_epoch: Int
    var current_lr: Float64
fn __init__(out self, base_lr: Float64, step_size: Int, gamma: Float64 = 0.1) -> None:
        if step_size <= 0: step_size = 1
        self.base_lr = base_lr
        self.step_size = step_size
        self.gamma = gamma
        self.last_epoch = 0
        self.current_lr = base_lr
fn _recompute(mut self) -> None:
        var k = _floor_div(self.last_epoch, self.step_size)
        self.current_lr = self.base_lr * _powf(self.gamma, k)
fn set_epoch(mut self, epoch: Int) -> None:
        self.last_epoch = (epoch if epoch >= 0 else 0)
        self._recompute()
fn step(mut self) -> None:
        self.last_epoch += 1
        self._recompute()
fn get_lr(self) -> Float64:
        return self.current_lr
fn __copyinit__(out self, other: Self) -> None:
        self.base_lr = other.base_lr
        self.step_size = other.step_size
        self.gamma = other.gamma
        self.last_epoch = other.last_epoch
        self.current_lr = other.current_lr
fn __moveinit__(out self, deinit other: Self) -> None:
        self.base_lr = other.base_lr
        self.step_size = other.step_size
        self.gamma = other.gamma
        self.last_epoch = other.last_epoch
        self.current_lr = other.current_lr
# --- MultiStepLR ---
struct MultiStepLR:
    var base_lr: Float64
    var milestones: List[Int]  # sorted not required; we count <= last_epoch
    var gamma: Float64
    var last_epoch: Int
    var current_lr: Float64
fn __init__(out self, base_lr: Float64, milestones: List[Int], gamma: Float64 = 0.1) -> None:
        self.base_lr = base_lr
        self.milestones = milestones
        self.gamma = gamma
        self.last_epoch = 0
        self.current_lr = base_lr
fn _count_leq(self, t: Int) -> Int:
        var c = 0
        for m in self.milestones:
            if t >= m: c += 1
        return c
fn _recompute(mut self) -> None:
        var n = self._count_leq(self.last_epoch)
        self.current_lr = self.base_lr * _powf(self.gamma, n)
fn set_epoch(mut self, epoch: Int) -> None:
        self.last_epoch = (epoch if epoch >= 0 else 0)
        self._recompute()
fn step(mut self) -> None:
        self.last_epoch += 1
        self._recompute()
fn get_lr(self) -> Float64:
        return self.current_lr
fn __copyinit__(out self, other: Self) -> None:
        self.base_lr = other.base_lr
        self.milestones = other.milestones
        self.gamma = other.gamma
        self.last_epoch = other.last_epoch
        self.current_lr = other.current_lr
fn __moveinit__(out self, deinit other: Self) -> None:
        self.base_lr = other.base_lr
        self.milestones = other.milestones
        self.gamma = other.gamma
        self.last_epoch = other.last_epoch
        self.current_lr = other.current_lr
# --- ExponentialLR ---
struct ExponentialLR:
    var base_lr: Float64
    var gamma: Float64
    var last_epoch: Int
    var current_lr: Float64
fn __init__(out self, base_lr: Float64, gamma: Float64 = 0.99) -> None:
        self.base_lr = base_lr
        self.gamma = gamma
        self.last_epoch = 0
        self.current_lr = base_lr
fn _recompute(mut self) -> None:
        self.current_lr = self.base_lr * _powf(self.gamma, self.last_epoch)
fn set_epoch(mut self, epoch: Int) -> None:
        self.last_epoch = (epoch if epoch >= 0 else 0)
        self._recompute()
fn step(mut self) -> None:
        self.last_epoch += 1
        self._recompute()
fn get_lr(self) -> Float64:
        return self.current_lr
fn __copyinit__(out self, other: Self) -> None:
        self.base_lr = other.base_lr
        self.gamma = other.gamma
        self.last_epoch = other.last_epoch
        self.current_lr = other.current_lr
fn __moveinit__(out self, deinit other: Self) -> None:
        self.base_lr = other.base_lr
        self.gamma = other.gamma
        self.last_epoch = other.last_epoch
        self.current_lr = other.current_lr
# --- CosineAnnealingLR ---
struct CosineAnnealingLR:
    var base_lr: Float64
    var T_max: Int
    var eta_min: Float64
    var last_epoch: Int
    var current_lr: Float64
fn __init__(out self, base_lr: Float64, T_max: Int, eta_min: Float64 = 0.0) -> None:
        if T_max <= 0: T_max = 1
        self.base_lr = base_lr
        self.T_max = T_max
        self.eta_min = eta_min
        self.last_epoch = 0
        self.current_lr = base_lr
fn _recompute(mut self):
        var t = self.last_epoch
        if t >= self.T_max:
            self.current_lr = self.eta_min
            return
        var cosv = _cos(_pi() * Float64(t) / Float64(self.T_max))
        self.current_lr = self.eta_min + (self.base_lr - self.eta_min) * (1.0 + cosv) * 0.5
fn set_epoch(mut self, epoch: Int) -> None:
        self.last_epoch = (epoch if epoch >= 0 else 0)
        self._recompute()
fn step(mut self) -> None:
        self.last_epoch += 1
        self._recompute()
fn get_lr(self) -> Float64:
        return self.current_lr
fn __copyinit__(out self, other: Self) -> None:
        self.base_lr = other.base_lr
        self.T_max = other.T_max
        self.eta_min = other.eta_min
        self.last_epoch = other.last_epoch
        self.current_lr = other.current_lr
fn __moveinit__(out self, deinit other: Self) -> None:
        self.base_lr = other.base_lr
        self.T_max = other.T_max
        self.eta_min = other.eta_min
        self.last_epoch = other.last_epoch
        self.current_lr = other.current_lr
# --- CosineAnnealingWarmRestarts ---
struct CosineAnnealingWarmRestarts:
    var base_lr: Float64
    var T_0: Int
    var T_mult: Int
    var eta_min: Float64
    var last_epoch: Int
    var current_lr: Float64
fn __init__(out self, base_lr: Float64, T_0: Int, T_mult: Int = 2, eta_min: Float64 = 0.0) -> None:
        if T_0 <= 0: T_0 = 1
        if T_mult < 1: T_mult = 1
        self.base_lr = base_lr
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.last_epoch = 0
        self.current_lr = base_lr
fn _cycle_params(self) -> (Int, Int, Int):
        # returns (Ti, Ti_accum, t_cur) for current epoch
        var Ti = self.T_0
        var accum = 0
        var t = self.last_epoch
        while t >= Ti:
            t -= Ti
            accum += Ti
            Ti = Ti * self.T_mult
            if Ti <= 0: Ti = 1
        return (Ti, accum, t)
fn _recompute(mut self) -> None:
        var (Ti, _, tcur) = self._cycle_params()
        var cosv = _cos(_pi() * Float64(tcur) / Float64(Ti))
        self.current_lr = self.eta_min + (self.base_lr - self.eta_min) * (1.0 + cosv) * 0.5
fn set_epoch(mut self, epoch: Int) -> None:
        self.last_epoch = (epoch if epoch >= 0 else 0)
        self._recompute()
fn step(mut self) -> None:
        self.last_epoch += 1
        self._recompute()
fn get_lr(self) -> Float64:
        return self.current_lr
fn __copyinit__(out self, other: Self) -> None:
        self.base_lr = other.base_lr
        self.T_0 = other.T_0
        self.T_mult = other.T_mult
        self.eta_min = other.eta_min
        self.last_epoch = other.last_epoch
        self.current_lr = other.current_lr
fn __moveinit__(out self, deinit other: Self) -> None:
        self.base_lr = other.base_lr
        self.T_0 = other.T_0
        self.T_mult = other.T_mult
        self.eta_min = other.eta_min
        self.last_epoch = other.last_epoch
        self.current_lr = other.current_lr
# --- LinearWarmup ---
struct LinearWarmup:
    var base_lr: Float64
    var warmup_steps: Int
    var start_lr: Float64
    var last_epoch: Int
    var current_lr: Float64
fn __init__(out self, base_lr: Float64, warmup_steps: Int, start_lr: Float64 = 0.0) -> None:
        if warmup_steps < 1: warmup_steps = 1
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        self.start_lr = start_lr
        self.last_epoch = 0
        self.current_lr = start_lr
fn _recompute(mut self):
        var t = self.last_epoch
        if t >= self.warmup_steps:
            self.current_lr = self.base_lr
            return
        var alpha = Float64(t) / Float64(self.warmup_steps)
        self.current_lr = self.start_lr + (self.base_lr - self.start_lr) * alpha
fn set_epoch(mut self, epoch: Int) -> None:
        self.last_epoch = (epoch if epoch >= 0 else 0)
        self._recompute()
fn step(mut self) -> None:
        self.last_epoch += 1
        self._recompute()
fn get_lr(self) -> Float64:
        return self.current_lr
fn __copyinit__(out self, other: Self) -> None:
        self.base_lr = other.base_lr
        self.warmup_steps = other.warmup_steps
        self.start_lr = other.start_lr
        self.last_epoch = other.last_epoch
        self.current_lr = other.current_lr
fn __moveinit__(out self, deinit other: Self) -> None:
        self.base_lr = other.base_lr
        self.warmup_steps = other.warmup_steps
        self.start_lr = other.start_lr
        self.last_epoch = other.last_epoch
        self.current_lr = other.current_lr
# --- WarmupCosine (linear warmup -> cosine decay) ---
struct WarmupCosine:
    var base_lr: Float64
    var total_steps: Int
    var warmup_steps: Int
    var eta_min: Float64
    var start_lr: Float64
    var last_epoch: Int
    var current_lr: Float64
fn __init__(out self, base_lr: Float64, total_steps: Int, warmup_steps: Int = 0, eta_min: Float64 = 0.0, start_lr: Float64 = 0.0) -> None:
        if total_steps < 1: total_steps = 1
        if warmup_steps < 0: warmup_steps = 0
        if warmup_steps > total_steps: warmup_steps = total_steps
        self.base_lr = base_lr
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.eta_min = eta_min
        self.start_lr = start_lr
        self.last_epoch = 0
        self.current_lr = (start_lr if warmup_steps > 0 else base_lr)
fn _recompute(mut self):
        var t = self.last_epoch
        if t < self.warmup_steps and self.warmup_steps > 0:
            var a = Float64(t) / Float64(self.warmup_steps)
            self.current_lr = self.start_lr + (self.base_lr - self.start_lr) * a
            return
        var t2 = t - self.warmup_steps
        var T = self.total_steps - self.warmup_steps
        if T <= 0:
            self.current_lr = self.base_lr
            return
        if t2 >= T:
            self.current_lr = self.eta_min
            return
        var cosv = _cos(_pi() * Float64(t2) / Float64(T))
        self.current_lr = self.eta_min + (self.base_lr - self.eta_min) * (1.0 + cosv) * 0.5
fn set_epoch(mut self, epoch: Int) -> None:
        self.last_epoch = (epoch if epoch >= 0 else 0)
        self._recompute()
fn step(mut self) -> None:
        self.last_epoch += 1
        self._recompute()
fn get_lr(self) -> Float64:
        return self.current_lr
fn __copyinit__(out self, other: Self) -> None:
        self.base_lr = other.base_lr
        self.total_steps = other.total_steps
        self.warmup_steps = other.warmup_steps
        self.eta_min = other.eta_min
        self.start_lr = other.start_lr
        self.last_epoch = other.last_epoch
        self.current_lr = other.current_lr
fn __moveinit__(out self, deinit other: Self) -> None:
        self.base_lr = other.base_lr
        self.total_steps = other.total_steps
        self.warmup_steps = other.warmup_steps
        self.eta_min = other.eta_min
        self.start_lr = other.start_lr
        self.last_epoch = other.last_epoch
        self.current_lr = other.current_lr
# --- Mock optimizer (for tests) ---
struct _MockOpt:
    var lr: Float64
fn __init__(out self, lr: Float64) -> None: self.lr = lr
fn set_lr(mut self, lr: Float64) -> None: self.lr = lr
fn __copyinit__(out self, other: Self) -> None:
        self.lr = other.lr
fn __moveinit__(out self, deinit other: Self) -> None:
        self.lr = other.lr
# --- Smoke tests ---
fn _self_test() -> Bool:
    var ok = True

    # StepLR
    var s = StepLR(0.1, 3, 0.5)
    ok = ok and (s.get_lr() == 0.1)
    s.step(); s.step(); s.step()  # at epoch=3 -> lr halves
    ok = ok and (s.get_lr() <= 0.051) and (s.get_lr() >= 0.049)

    # MultiStepLR
    var ms = MultiStepLR(0.2, List[Int]([2, 5]), 0.1)
    ms.set_epoch(0); ok = ok and (ms.get_lr() == 0.2)
    ms.set_epoch(2); ok = ok and (ms.get_lr() <= 0.0200001)
    ms.set_epoch(6); ok = ok and (ms.get_lr() <= 0.0020001)

    # ExponentialLR
    var e = ExponentialLR(1.0, 0.9)
    e.set_epoch(2)
    ok = ok and (e.get_lr() <= 0.82) and (e.get_lr() >= 0.80)

    # CosineAnnealingLR
    var c = CosineAnnealingLR(0.1, 4, 0.0)
    c.set_epoch(0); ok = ok and (c.get_lr() <= 0.1000001)
    c.set_epoch(4); ok = ok and (c.get_lr() <= 0.0000001)

    # CosineAnnealingWarmRestarts
    var cr = CosineAnnealingWarmRestarts(0.1, 2, 2, 0.0)
    cr.set_epoch(0); var lr0 = cr.get_lr()
    cr.set_epoch(1); var lr1 = cr.get_lr()
    cr.set_epoch(2); var lr2 = cr.get_lr()  # restart
    ok = ok and (lr0 > lr1) and (lr2 >= lr1)

    # LinearWarmup
    var w = LinearWarmup(0.1, 5, 0.0)
    w.set_epoch(2)
    ok = ok and (w.get_lr() >= 0.03) and (w.get_lr() <= 0.05)
    w.set_epoch(10); ok = ok and (w.get_lr() == 0.1)

    # WarmupCosine
    var wc = WarmupCosine(0.1, 10, 3, 0.0, 0.0)
    wc.set_epoch(0); var a0 = wc.get_lr()
    wc.set_epoch(3); var a3 = wc.get_lr()
    wc.set_epoch(9); var a9 = wc.get_lr()
    ok = ok and (a0 <= 0.000001) and (a3 <= 0.100001) and (a9 <= 0.001)

    # Integration with mock optimizer (manual apply)
    var opt = _MockOpt(0.1)
    var steplr = StepLR(opt.lr, 2, 0.5)
    for t in range(5):
        # apply current lr to opt
        opt.set_lr(steplr.get_lr())
        steplr.step()
    ok = ok and (opt.lr <= 0.1)

    return ok
fn main() -> None:
    if _self_test():
        print(String("momijo.nn.lr_scheduler: OK"))
    else:
        print(String("momijo.nn.lr_scheduler: FAILED"))