# MIT License
# SPDX-License-Identifier: MIT
# Project:      Momijo
# Module:       learn.optim.schedulers
# File:         src/momijo/learn/optim/schedulers.mojo
#
# Description:
#   Learning rate schedulers for Momijo Learn (backend-agnostic).
#   Included:
#     - StepLR
#     - ExponentialLR
#     - MultiStepLR
#     - PolynomialLR
#     - CosineAnnealingLR
#     - CosineAnnealingWarmRestarts
#     - LinearWarmup
#     - WarmupThen (warmup + main)
#     - ChainedLR
#     - ReduceLROnPlateau
#
# Author(s):    Morteza Taleblou & Mitra Daneshmand
# Website:      https://taleblou.ir/
# Repository:   https://github.com/taleblou/momijo

from collections.list import List

# -----------------------------------------------------------------------------
# Tiny numeric helpers (no stdlib.math dependency)
# -----------------------------------------------------------------------------

@always_inline
fn _max(a: Float64, b: Float64) -> Float64:
    var r = a
    if b > a:
        r = b
    return r

@always_inline
fn _min(a: Float64, b: Float64) -> Float64:
    var r = a
    if b < a:
        r = b
    return r

@always_inline
fn _abs(x: Float64) -> Float64:
    if x < 0.0:
        return -x
    return x

@always_inline
fn _fmod(x: Float64, y: Float64) -> Float64:
    # x - y*floor(x/y) with floor emulation for negatives
    var q = x / y
    var qi = Int(q)
    if Float64(qi) > q:
        qi = qi - 1
    return x - (Float64(qi) * y)

@always_inline
fn _wrap_pi(x: Float64) -> Float64:
    # map x to [-PI, PI]
    var two_pi = 6.283185307179586476925286766559
    var y = _fmod(x, two_pi)
    var pi = 3.1415926535897932384626433832795
    if y > pi:
        y = y - two_pi
    if y < -pi:
        y = y + two_pi
    return y

@always_inline
fn _cos_approx(x_in: Float64) -> Float64:
    # cos x ≈ 1 - x^2/2! + x^4/4! - x^6/6! after wrapping to [-pi, pi]
    var x = _wrap_pi(x_in)
    var x2 = x * x
    var x4 = x2 * x2
    var x6 = x4 * x2
    return 1.0 - (x2 * 0.5) + (x4 * (1.0 / 24.0)) - (x6 * (1.0 / 720.0))

@always_inline
fn _pow_scalar(x: Float64, n: Int) -> Float64:
    var r = 1.0
    var i = 0
    while i < n:
        r = r * x
        i = i + 1
    return r

# Safe clamp-to-nonnegative integer
@always_inline
fn _clamp_epoch_nonneg(t: Int) -> Int:
    if t < 0:
        return 0
    return t

# -----------------------------------------------------------------------------
# StepLR: decay base_lr by gamma every step_size steps
# -----------------------------------------------------------------------------

struct StepLR:
    var base_lr: Float64
    var gamma: Float64
    var step_size: Int
    var last_epoch: Int
    var current_lr: Float64

    fn __init__(out self, base_lr: Float64, step_size: Int, gamma: Float64 = 0.1, last_epoch: Int = -1):
        assert(step_size > 0)
        assert(gamma > 0.0)
        assert(base_lr >= 0.0)
        self.base_lr = base_lr
        self.gamma = gamma
        self.step_size = step_size
        self.last_epoch = last_epoch
        self.current_lr = base_lr

    fn get_lr(self) -> Float64:
        return self.current_lr

    fn set_base_lr(mut self, lr: Float64):
        assert(lr >= 0.0)
        self.base_lr = lr

    fn _compute_lr(self, epoch: Int) -> Float64:
        var k = epoch // self.step_size
        var scale = 1.0
        var i = 0
        while i < k:
            scale = scale * self.gamma
            i = i + 1
        return self.base_lr * scale

    fn step(mut self) -> Float64:
        self.last_epoch = self.last_epoch + 1
        if self.last_epoch < 0:
            self.current_lr = self.base_lr
            return self.current_lr
        self.current_lr = self._compute_lr(self.last_epoch)
        return self.current_lr

    fn step_to(mut self, epoch: Int) -> Float64:
        self.last_epoch = epoch
        if epoch < 0:
            self.current_lr = self.base_lr
            return self.current_lr
        self.current_lr = self._compute_lr(epoch)
        return self.current_lr

    fn state_dict(self) -> String:
        var s = String("{'type':'StepLR'")
        s = s + ",'base_lr':" + String(self.base_lr)
        s = s + ",'gamma':" + String(self.gamma)
        s = s + ",'step_size':" + String(self.step_size)
        s = s + ",'last_epoch':" + String(self.last_epoch)
        s = s + ",'current_lr':" + String(self.current_lr) + "}"
        return s

    fn load_state_dict(mut self, state: String):
        self.current_lr = self.base_lr
        self.last_epoch = -1

# -----------------------------------------------------------------------------
# ExponentialLR: lr = base_lr * gamma^t
# -----------------------------------------------------------------------------

struct ExponentialLR:
    var base_lr: Float64
    var gamma: Float64
    var last_epoch: Int
    var current_lr: Float64

    fn __init__(out self, base_lr: Float64, gamma: Float64 = 0.99, last_epoch: Int = -1):
        assert(base_lr >= 0.0)
        assert(gamma > 0.0)
        self.base_lr = base_lr
        self.gamma = gamma
        self.last_epoch = last_epoch
        self.current_lr = base_lr

    fn get_lr(self) -> Float64:
        return self.current_lr

    fn _compute_lr(self, t: Int) -> Float64:
        var tt = _clamp_epoch_nonneg(t)
        if tt == 0:
            return self.base_lr
        var scale = 1.0
        var i = 0
        while i < tt:
            scale = scale * self.gamma
            i = i + 1
        return self.base_lr * scale

    fn step(mut self) -> Float64:
        self.last_epoch = self.last_epoch + 1
        self.current_lr = self._compute_lr(self.last_epoch)
        return self.current_lr

    fn step_to(mut self, epoch: Int) -> Float64:
        self.last_epoch = epoch
        self.current_lr = self._compute_lr(epoch)
        return self.current_lr

    fn state_dict(self) -> String:
        var s = String("{'type':'ExponentialLR'")
        s = s + ",'base_lr':" + String(self.base_lr)
        s = s + ",'gamma':" + String(self.gamma)
        s = s + ",'last_epoch':" + String(self.last_epoch)
        s = s + ",'current_lr':" + String(self.current_lr) + "}"
        return s

    fn load_state_dict(mut self, state: String):
        self.current_lr = self.base_lr
        self.last_epoch = -1

# -----------------------------------------------------------------------------
# MultiStepLR: decay at specific epochs in milestones
# -----------------------------------------------------------------------------

struct MultiStepLR:
    var base_lr: Float64
    var gamma: Float64
    var milestones: List[Int]
    var last_epoch: Int
    var current_lr: Float64

    fn __init__(out self, base_lr: Float64, milestones: List[Int], gamma: Float64 = 0.1, last_epoch: Int = -1):
        assert(base_lr >= 0.0)
        assert(gamma > 0.0)
        # milestones strictly increasing, non-negative
        var i = 0
        var prev = -1
        while i < Int(milestones.size()):
            var m = milestones[i]
            assert(m >= 0)
            assert(m > prev)
            prev = m
            i = i + 1
        self.base_lr = base_lr
        self.gamma = gamma
        self.milestones = milestones
        self.last_epoch = last_epoch
        self.current_lr = base_lr

    fn get_lr(self) -> Float64:
        return self.current_lr

    fn _compute_lr(self, epoch: Int) -> Float64:
        var count = 0
        var i = 0
        var n = Int(self.milestones.size())
        while i < n:
            if epoch >= self.milestones[i]:
                count = count + 1
            i = i + 1
        var scale = 1.0
        var j = 0
        while j < count:
            scale = scale * self.gamma
            j = j + 1
        return self.base_lr * scale

    fn step(mut self) -> Float64:
        self.last_epoch = self.last_epoch + 1
        if self.last_epoch < 0:
            self.current_lr = self.base_lr
            return self.current_lr
        self.current_lr = self._compute_lr(self.last_epoch)
        return self.current_lr

    fn step_to(mut self, epoch: Int) -> Float64:
        self.last_epoch = epoch
        if epoch < 0:
            self.current_lr = self.base_lr
            return self.current_lr
        self.current_lr = self._compute_lr(epoch)
        return self.current_lr

    fn state_dict(self) -> String:
        var s = String("{'type':'MultiStepLR'")
        s = s + ",'base_lr':" + String(self.base_lr)
        s = s + ",'gamma':" + String(self.gamma)
        s = s + ",'milestones':" + String(Int(self.milestones.size()))
        s = s + ",'last_epoch':" + String(self.last_epoch)
        s = s + ",'current_lr':" + String(self.current_lr) + "}"
        return s

    fn load_state_dict(mut self, state: String):
        self.current_lr = self.base_lr
        self.last_epoch = -1

# -----------------------------------------------------------------------------
# PolynomialLR: lr = min_lr + (base_lr - min_lr) * (1 - t/T_max)^power
# -----------------------------------------------------------------------------

struct PolynomialLR:
    var base_lr: Float64
    var min_lr: Float64
    var power: Float64
    var T_max: Int
    var last_epoch: Int
    var current_lr: Float64

    fn __init__(out self, base_lr: Float64, T_max: Int, power: Float64 = 1.0, min_lr: Float64 = 0.0, last_epoch: Int = -1):
        assert(T_max > 0)
        assert(base_lr >= 0.0)
        assert(min_lr >= 0.0)
        assert(base_lr >= min_lr)
        assert(power >= 0.0)
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.power = power
        self.T_max = T_max
        self.last_epoch = last_epoch
        self.current_lr = base_lr

    fn get_lr(self) -> Float64:
        return self.current_lr

    fn _compute_lr(self, t: Int) -> Float64:
        var tt = t
        if tt < 0:
            tt = 0
        if tt > self.T_max:
            tt = self.T_max
        var frac = 1.0 - (Float64(tt) / Float64(self.T_max))
        # integer part via repeated multiplication
        var ipart = Int(self.power)
        var decay = 1.0
        var k = 0
        while k < ipart:
            decay = decay * frac
            k = k + 1
        # crude fractional refinement
        var rem = self.power - Float64(ipart)
        if rem > 0.0:
            # ln(frac) ~ taylor around 1 for stability
            var f = frac
            if f <= 0.0:
                f = 1e-12
            var d = f - 1.0
            var d2 = d * d
            var d3 = d2 * d
            var d4 = d3 * d
            var lnx = d - 0.5 * d2 + (1.0/3.0) * d3 - 0.25 * d4
            decay = decay * (1.0 + rem * lnx)
        return self.min_lr + (self.base_lr - self.min_lr) * decay

    fn step(mut self) -> Float64:
        self.last_epoch = self.last_epoch + 1
        self.current_lr = self._compute_lr(self.last_epoch)
        return self.current_lr

    fn step_to(mut self, epoch: Int) -> Float64:
        self.last_epoch = epoch
        self.current_lr = self._compute_lr(epoch)
        return self.current_lr

    fn state_dict(self) -> String:
        var s = String("{'type':'PolynomialLR'")
        s = s + ",'base_lr':" + String(self.base_lr)
        s = s + ",'min_lr':" + String(self.min_lr)
        s = s + ",'power':" + String(self.power)
        s = s + ",'T_max':" + String(self.T_max)
        s = s + ",'last_epoch':" + String(self.last_epoch)
        s = s + ",'current_lr':" + String(self.current_lr) + "}"
        return s

    fn load_state_dict(mut self, state: String):
        self.current_lr = self.base_lr
        self.last_epoch = -1

# -----------------------------------------------------------------------------
# CosineAnnealingLR
# lr(t) = min_lr + 0.5*(base_lr - min_lr)*(1 + cos(pi*t/T_max)), t in [0, T_max]
# -----------------------------------------------------------------------------

struct CosineAnnealingLR:
    var base_lr: Float64
    var min_lr: Float64
    var T_max: Int
    var last_epoch: Int
    var current_lr: Float64

    fn __init__(out self, base_lr: Float64, T_max: Int, min_lr: Float64 = 0.0, last_epoch: Int = -1):
        assert(T_max > 0)
        assert(base_lr >= 0.0)
        assert(min_lr >= 0.0)
        assert(base_lr >= min_lr)
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.T_max = T_max
        self.last_epoch = last_epoch
        self.current_lr = base_lr

    fn get_lr(self) -> Float64:
        return self.current_lr

    fn set_base_lr(mut self, lr: Float64):
        assert(lr >= 0.0)
        assert(lr >= self.min_lr)
        self.base_lr = lr

    fn set_min_lr(mut self, lr: Float64):
        assert(lr >= 0.0)
        assert(self.base_lr >= lr)
        self.min_lr = lr

    fn _compute_lr(self, t: Int) -> Float64:
        var tt = t
        if tt < 0:
            tt = 0
        if tt > self.T_max:
            tt = self.T_max
        var pi = 3.1415926535897932384626433832795
        var cos_term = _cos_approx(pi * (Float64(tt) / Float64(self.T_max)))
        var frac = 0.5 * (1.0 + cos_term)
        return self.min_lr + (self.base_lr - self.min_lr) * frac

    fn step(mut self) -> Float64:
        self.last_epoch = self.last_epoch + 1
        if self.last_epoch < 0:
            self.current_lr = self.base_lr
            return self.current_lr
        self.current_lr = self._compute_lr(self.last_epoch)
        return self.current_lr

    fn step_to(mut self, epoch: Int) -> Float64:
        self.last_epoch = epoch
        if epoch < 0:
            self.current_lr = self.base_lr
            return self.current_lr
        self.current_lr = self._compute_lr(epoch)
        return self.current_lr

    fn state_dict(self) -> String:
        var s = String("{'type':'CosineAnnealingLR'")
        s = s + ",'base_lr':" + String(self.base_lr)
        s = s + ",'min_lr':" + String(self.min_lr)
        s = s + ",'T_max':" + String(self.T_max)
        s = s + ",'last_epoch':" + String(self.last_epoch)
        s = s + ",'current_lr':" + String(self.current_lr) + "}"
        return s

    fn load_state_dict(mut self, state: String):
        self.current_lr = self.base_lr
        self.last_epoch = -1

# -----------------------------------------------------------------------------
# CosineAnnealingWarmRestarts
# Warm restarts with periods: T_0, T_0*T_mult, T_0*T_mult^2, ...
# -----------------------------------------------------------------------------

struct CosineAnnealingWarmRestarts:
    var base_lr: Float64
    var min_lr: Float64
    var T_0: Int
    var T_mult: Int
    var last_epoch: Int
    var current_lr: Float64

    # internal counters
    var _Ti: Int     # current cycle length
    var _t_cur: Int  # position in current cycle

    fn __init__(out self, base_lr: Float64, T_0: Int, T_mult: Int = 1, min_lr: Float64 = 0.0, last_epoch: Int = -1):
        assert(base_lr >= 0.0)
        assert(min_lr >= 0.0)
        assert(base_lr >= min_lr)
        assert(T_0 > 0)
        assert(T_mult >= 1)
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.T_0 = T_0
        self.T_mult = T_mult
        self.last_epoch = last_epoch
        self.current_lr = base_lr
        self._Ti = T_0
        self._t_cur = -1

    fn get_lr(self) -> Float64:
        return self.current_lr

    fn _compute_lr(self) -> Float64:
        var pi = 3.1415926535897932384626433832795
        var cos_term = _cos_approx(pi * (Float64(self._t_cur) / Float64(self._Ti)))
        var frac = 0.5 * (1.0 + cos_term)
        return self.min_lr + (self.base_lr - self.min_lr) * frac

    fn step(mut self) -> Float64:
        self.last_epoch = self.last_epoch + 1
        self._t_cur = self._t_cur + 1
        if self._t_cur >= self._Ti:
            self._t_cur = 0
            self._Ti = self._Ti * self.T_mult
        self.current_lr = self._compute_lr()
        return self.current_lr

    fn step_to(mut self, epoch: Int) -> Float64:
        # recompute cycle position from scratch
        self.last_epoch = epoch
        var remaining = epoch
        if remaining < 0:
            remaining = 0
        self._Ti = self.T_0
        self._t_cur = 0
        while remaining >= self._Ti:
            remaining = remaining - self._Ti
            self._Ti = self._Ti * self.T_mult
        self._t_cur = remaining
        self.current_lr = self._compute_lr()
        return self.current_lr

    fn state_dict(self) -> String:
        var s = String("{'type':'CosineAnnealingWarmRestarts'")
        s = s + ",'base_lr':" + String(self.base_lr)
        s = s + ",'min_lr':" + String(self.min_lr)
        s = s + ",'T_0':" + String(self.T_0)
        s = s + ",'T_mult':" + String(self.T_mult)
        s = s + ",'last_epoch':" + String(self.last_epoch)
        s = s + ",'_Ti':" + String(self._Ti)
        s = s + ",'_t_cur':" + String(self._t_cur)
        s = s + ",'current_lr':" + String(self.current_lr) + "}"
        return s

    fn load_state_dict(mut self, state: String):
        self.current_lr = self.base_lr
        self.last_epoch = -1
        self._Ti = self.T_0
        self._t_cur = -1

# -----------------------------------------------------------------------------
# LinearWarmup: linearly increase LR from start_lr to base_lr over warmup_steps
# -----------------------------------------------------------------------------

struct LinearWarmup:
    var start_lr: Float64
    var base_lr: Float64
    var warmup_steps: Int
    var last_epoch: Int
    var current_lr: Float64

    fn __init__(out self, start_lr: Float64, base_lr: Float64, warmup_steps: Int, last_epoch: Int = -1):
        assert(warmup_steps > 0)
        assert(start_lr >= 0.0)
        assert(base_lr >= 0.0)
        self.start_lr = start_lr
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        self.last_epoch = last_epoch
        self.current_lr = start_lr

    fn get_lr(self) -> Float64:
        return self.current_lr

    fn _compute_lr(self, t: Int) -> Float64:
        var tt = t
        if tt < 0:
            tt = 0
        if tt >= self.warmup_steps:
            return self.base_lr
        var frac = Float64(tt) / Float64(self.warmup_steps)
        return self.start_lr + (self.base_lr - self.start_lr) * frac

    fn step(mut self) -> Float64:
        self.last_epoch = self.last_epoch + 1
        self.current_lr = self._compute_lr(self.last_epoch)
        return self.current_lr

    fn step_to(mut self, epoch: Int) -> Float64:
        self.last_epoch = epoch
        self.current_lr = self._compute_lr(epoch)
        return self.current_lr

    fn state_dict(self) -> String:
        var s = String("{'type':'LinearWarmup'")
        s = s + ",'start_lr':" + String(self.start_lr)
        s = s + ",'base_lr':" + String(self.base_lr)
        s = s + ",'warmup_steps':" + String(self.warmup_steps)
        s = s + ",'last_epoch':" + String(self.last_epoch)
        s = s + ",'current_lr':" + String(self.current_lr) + "}"
        return s

    fn load_state_dict(mut self, state: String):
        self.current_lr = self.start_lr
        self.last_epoch = -1

# -----------------------------------------------------------------------------
# WarmupThen: compose a warmup scheduler with a main scheduler
#   For t < warmup_steps: warmup; else main with shifted epoch
# -----------------------------------------------------------------------------

struct WarmupThen:
    var warmup: LinearWarmup
    var main: CosineAnnealingLR    # could be any scheduler type in practice
    var warmup_steps: Int
    var last_epoch: Int
    var current_lr: Float64

    fn __init__(out self, warmup: LinearWarmup, main: CosineAnnealingLR):
        self.warmup = warmup
        self.main = main
        self.warmup_steps = warmup.warmup_steps
        self.last_epoch = -1
        self.current_lr = warmup.start_lr

    fn get_lr(self) -> Float64:
        return self.current_lr

    fn step(mut self) -> Float64:
        self.last_epoch = self.last_epoch + 1
        if self.last_epoch < self.warmup_steps:
            self.current_lr = self.warmup.step()
            return self.current_lr
        var shifted = self.last_epoch - self.warmup_steps
        self.current_lr = self.main.step_to(shifted)
        return self.current_lr

    fn step_to(mut self, epoch: Int) -> Float64:
        self.last_epoch = epoch
        if epoch < self.warmup_steps:
            self.current_lr = self.warmup.step_to(epoch)
            return self.current_lr
        var shifted = epoch - self.warmup_steps
        self.current_lr = self.main.step_to(shifted)
        return self.current_lr

    fn state_dict(self) -> String:
        var s = String("{'type':'WarmupThen'")
        s = s + ",'warmup_last':" + String(self.warmup.last_epoch)
        s = s + ",'main_last':" + String(self.main.last_epoch)
        s = s + ",'warmup_steps':" + String(self.warmup_steps)
        s = s + ",'last_epoch':" + String(self.last_epoch)
        s = s + ",'current_lr':" + String(self.current_lr) + "}"
        return s

    fn load_state_dict(mut self, state: String):
        self.last_epoch = -1
        self.current_lr = self.warmup.start_lr
        self.warmup.load_state_dict(String(""))
        self.main.load_state_dict(String(""))

# -----------------------------------------------------------------------------
# ChainedLR: run schedulers in sequence across spans [len_0, len_1, ...]
# -----------------------------------------------------------------------------

struct ChainedLR:
    var schedulers: List[CosineAnnealingLR]  # could be heterogeneous; kept simple
    var spans: List[Int]                      # steps per scheduler (last can be -1 for "rest")
    var last_epoch: Int
    var current_lr: Float64

    fn __init__(out self, schedulers: List[CosineAnnealingLR], spans: List[Int]):
        assert(Int(schedulers.size()) == Int(spans.size()))
        self.schedulers = schedulers
        self.spans = spans
        self.last_epoch = -1
        self.current_lr = schedulers[0].get_lr()

    fn get_lr(self) -> Float64:
        return self.current_lr

    fn _locate(self, t: Int) -> (Int, Int):
        # returns (which, local_t)
        var acc = 0
        var i = 0
        var n = Int(self.spans.size())
        while i < n:
            var sp = self.spans[i]
            if sp < 0:
                return (i, t - acc)
            if t < acc + sp:
                return (i, t - acc)
            acc = acc + sp
            i = i + 1
        # beyond all spans -> clamp to last
        var last = n - 1
        return (last, self.spans[last] - 1)

    fn step(mut self) -> Float64:
        self.last_epoch = self.last_epoch + 1
        var which_local = self._locate(self.last_epoch)
        var idx = which_local[0]
        var loc = which_local[1]
        self.current_lr = self.schedulers[idx].step_to(loc)
        return self.current_lr

    fn step_to(mut self, epoch: Int) -> Float64:
        self.last_epoch = epoch
        var which_local = self._locate(epoch)
        var idx = which_local[0]
        var loc = which_local[1]
        self.current_lr = self.schedulers[idx].step_to(loc)
        return self.current_lr

    fn state_dict(self) -> String:
        var s = String("{'type':'ChainedLR'")
        s = s + ",'last_epoch':" + String(self.last_epoch)
        s = s + ",'current_lr':" + String(self.current_lr) + "}"
        return s

    fn load_state_dict(mut self, state: String):
        self.last_epoch = -1
        if Int(self.schedulers.size()) > 0:
            self.current_lr = self.schedulers[0].get_lr()

# -----------------------------------------------------------------------------
# ReduceLROnPlateau: reduce LR by factor if metric does not improve for 'patience' steps
# Mode:
#   - 'min' : improvement if metric < best - threshold
#   - 'max' : improvement if metric > best + threshold
# -----------------------------------------------------------------------------

struct ReduceLROnPlateau:
    var lr: Float64
    var factor: Float64
    var patience: Int
    var threshold: Float64
    var cooldown: Int
    var min_lr: Float64
    var mode_max: Bool    # True for 'max', False for 'min'

    var _best: Float64
    var _num_bad: Int
    var _cooldown: Int

    fn __init__(
        out self,
        lr: Float64,
        factor: Float64 = 0.1,
        patience: Int = 10,
        threshold: Float64 = 1e-4,
        cooldown: Int = 0,
        min_lr: Float64 = 0.0,
        mode: String = "min"
    ):
        assert(lr >= 0.0)
        assert(factor > 0.0)
        assert(patience >= 0)
        assert(threshold >= 0.0)
        assert(cooldown >= 0)
        assert(min_lr >= 0.0)

        self.lr = lr
        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self.cooldown = cooldown
        self.min_lr = min_lr
        self.mode_max = False
        if mode == String("max"):
            self.mode_max = True

        if self.mode_max:
            self._best = -1.7976931348623157e308
        else:
            self._best = 1.7976931348623157e308
        self._num_bad = 0
        self._cooldown = 0

    fn get_lr(self) -> Float64:
        return self.lr

    fn _is_improved(self, metric: Float64) -> Bool:
        if self.mode_max:
            return metric > self._best + self.threshold
        return metric < self._best - self.threshold

    fn step(mut self, metric: Float64) -> Float64:
        # cooldown handling
        if self._cooldown > 0:
            self._cooldown = self._cooldown - 1

        if self._is_improved(metric):
            self._best = metric
            self._num_bad = 0
            return self.lr

        self._num_bad = self._num_bad + 1
        if self._num_bad > self.patience and self._cooldown == 0:
            self.lr = self.lr * self.factor
            if self.lr < self.min_lr:
                self.lr = self.min_lr
            self._cooldown = self.cooldown
            self._num_bad = 0
        return self.lr

    fn state_dict(self) -> String:
        var s = String("{'type':'ReduceLROnPlateau'")
        s = s + ",'lr':" + String(self.lr)
        s = s + ",'factor':" + String(self.factor)
        s = s + ",'patience':" + String(self.patience)
        s = s + ",'threshold':" + String(self.threshold)
        s = s + ",'cooldown':" + String(self.cooldown)
        s = s + ",'min_lr':" + String(self.min_lr)
        s = s + ",'mode_max':" + String(self.mode_max)
        s = s + ",'_best':" + String(self._best)
        s = s + ",'_num_bad':" + String(self._num_bad)
        s = s + ",'_cooldown':" + String(self._cooldown) + "}"
        return s

    fn load_state_dict(mut self, state: String):
        if self.mode_max:
            self._best = -1.7976931348623157e308
        else:
            self._best = 1.7976931348623157e308
        self._num_bad = 0
        self._cooldown = 0
