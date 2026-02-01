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
#     - WarmupThen (wrap a warmup scheduler with a main scheduler)
#     - ChainedLR (chain multiple schedulers over predefined spans)
#     - ReduceLROnPlateau (simple/patient-based LR decay)
#
#   API per scheduler:
#     fn step(mut self) -> Float32
#     fn step_to(mut self, epoch: Int) -> Float32
#     fn get_lr(self) -> Float32
#     fn state_dict(self) -> String
#     fn load_state_dict(mut self, state: String)
#
# Author(s):    Morteza Taleblou & Mitra Daneshmand
# Website:      https://taleblou.ir/
# Repository:   https://github.com/taleblou/momijo

# -----------------------------------------------------------------------------
# Tiny numeric helpers (no stdlib.math dependency)
# -----------------------------------------------------------------------------

@always_inline
fn _max(a: Float32, b: Float32) -> Float32:
    return a if a > b else b

@always_inline
fn _min(a: Float32, b: Float32) -> Float32:
    return a if a < b else b

@always_inline
fn _abs(x: Float32) -> Float32:
    if x < 0.0:
        return -x
    return x

@always_inline
fn _fmod(x: Float32, y: Float32) -> Float32:
    # x - y*floor(x/y) with floor emulation for negatives
    var q = x / y
    var qi = Int(q)
    if Float32(qi) > q:
        qi = qi - 1
    return x - (Float32(qi) * y)

@always_inline
fn _wrap_pi(x: Float32) -> Float32:
    # map x to [-PI, PI]
    var two_pi = 6.283185307179586476925286766559
    var y = _fmod(x, two_pi)
    if y > 3.1415926535897932384626433832795:
        y = y - two_pi
    if y < -3.1415926535897932384626433832795:
        y = y + two_pi
    return y

@always_inline
fn _cos_approx(x_in: Float32) -> Float32:
    # cos x ≈ 1 - x^2/2! + x^4/4! - x^6/6! after wrapping to [-pi, pi]
    var x = _wrap_pi(x_in)
    var x2 = x * x
    var x4 = x2 * x2
    var x6 = x4 * x2
    return 1.0 - (x2 * 0.5) + (x4 * (1.0 / 24.0)) - (x6 * (1.0 / 720.0))

@always_inline
fn _pow_scalar(x: Float32, n: Int) -> Float32:
    var r:Float32 = 1.0
    var i = 0
    while i < n:
        r = r * x
        i = i + 1
    return r

# -----------------------------------------------------------------------------
# StepLR: decay base_lr by gamma every step_size steps
# -----------------------------------------------------------------------------
struct StepLR(Copyable, Movable):
    var base_lr: Float32
    var gamma: Float32
    var step_size: Int
    var last_epoch: Int
    var current_lr: Float32

    fn __init__(
        out self,
        base_lr: Float32,
        step_size: Int,
        gamma: Float32 = 0.1,
        last_epoch: Int = -1
    ):
        # ---- sanitize inputs (no assert) ----
        var ss = step_size
        if ss <= 0: ss = 1

        var g = gamma
        if g <= 0.0: g = 0.1

        var lr0 = base_lr
        if lr0 < 0.0: lr0 = 0.0

        var le = last_epoch  # can be -1 for "just created"

        self.base_lr   = lr0
        self.gamma     = g
        self.step_size = ss
        self.last_epoch = le
        self.current_lr = lr0

    fn get_lr(self) -> Float32:
        return self.current_lr

    fn set_base_lr(mut self, lr: Float32):
        # keep non-negative
        var v = lr
        if v < 0.0: v = 0.0
        self.base_lr = v
        # do not auto-recompute here; user can call step()/step_to() to refresh

    fn _compute_lr(self, epoch: Int) -> Float32:
        # k = floor(epoch / step_size)
        var k = (epoch // self.step_size)
        if k <= 0:
            return self.base_lr

        # scale = gamma^k  (iterative to avoid pow)
        var scale = 1.0
        var i = 0
        while i < k:
            scale = scale * self.gamma
            i = i + 1
        return self.base_lr * scale

    fn step(mut self) -> Float32:
        # Typical usage: call once per epoch AFTER optimizer step
        self.last_epoch = self.last_epoch + 1
        if self.last_epoch < 0:
            self.current_lr = self.base_lr
            return self.current_lr
        self.current_lr = self._compute_lr(self.last_epoch)
        return self.current_lr

    fn step_to(mut self, epoch: Int) -> Float32:
        # Directly jump to epoch index
        var e = epoch
        # allow any integer; negative means "pre-start"
        self.last_epoch = e
        if self.last_epoch < 0:
            self.current_lr = self.base_lr
            return self.current_lr
        self.current_lr = self._compute_lr(self.last_epoch)
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
        # Minimal safe restore:
        # Since 'state' is not strict JSON (single quotes), and we avoid heavy parsing here,
        # we reset to a consistent start; callers may set fields explicitly if they need.
        self.current_lr = self.base_lr
        self.last_epoch = -1

# -----------------------------------------------------------------------------
# ExponentialLR: lr = base_lr * gamma^t
# -----------------------------------------------------------------------------
struct ExponentialLR(Copyable, Movable):
    var base_lr: Float32
    var gamma: Float32
    var last_epoch: Int
    var current_lr: Float32

    fn __init__(
        out self,
        base_lr: Float32,
        gamma: Float32 = 0.99,
        last_epoch: Int = -1
    ):
        # ---- sanitize inputs ----
        var lr0 = base_lr
        if lr0 < 0.0: lr0 = 0.0

        var g = gamma
        if g <= 0.0: g = 0.99

        self.base_lr   = lr0
        self.gamma     = g
        self.last_epoch = last_epoch     # -1 => not started
        self.current_lr = lr0

    fn get_lr(self) -> Float32:
        return self.current_lr

    fn set_base_lr(mut self, lr: Float32):
        var v = lr
        if v < 0.0: v = 0.0
        self.base_lr = v
        # do not auto-recompute current_lr; call step()/step_to() as needed

    fn set_gamma(mut self, g: Float32):
        var v = g
        if v <= 0.0: v = 0.99
        self.gamma = v

    fn _compute_lr(self, t: Int) -> Float32:
        # t clamped to >= 0
        var steps = t
        if steps <= 0:
            return self.base_lr

        # scale = gamma^steps via iterative multiply (avoid pow)
        var scale = 1.0
        var i = 0
        while i < steps:
            scale = scale * self.gamma
            i = i + 1
        return self.base_lr * scale

    fn step(mut self) -> Float32:
        # typical usage: call once per epoch after optimizer step
        self.last_epoch = self.last_epoch + 1
        var t = self.last_epoch
        if t < 0: t = 0
        self.current_lr = self._compute_lr(t)
        return self.current_lr

    fn step_to(mut self, epoch: Int) -> Float32:
        # jump directly to a given epoch
        self.last_epoch = epoch
        var t = epoch
        if t < 0: t = 0
        self.current_lr = self._compute_lr(t)
        return self.current_lr

    fn state_dict(self) -> String:
        var s = String("{'type':'ExponentialLR'")
        s = s + ",'base_lr':" + String(self.base_lr)
        s = s + ",'gamma':" + String(self.gamma)
        s = s + ",'last_epoch':" + String(self.last_epoch)
        s = s + ",'current_lr':" + String(self.current_lr) + "}"
        return s

    fn load_state_dict(mut self, state: String):
        # Minimal safe restore (no parser for non-JSON string provided)
        self.current_lr = self.base_lr
        self.last_epoch = -1
# -----------------------------------------------------------------------------
# MultiStepLR: decay at specific epochs in milestones
# -----------------------------------------------------------------------------

struct MultiStepLR:
    var base_lr: Float32
    var gamma: Float32
    var milestones: List[Int]
    var last_epoch: Int
    var current_lr: Float32

    fn __init__(out self, base_lr: Float32, milestones: List[Int], gamma: Float32 = 0.1, last_epoch: Int = -1):
        assert(base_lr >= 0.0)
        assert(gamma > 0.0)
        # milestones should be strictly increasing non-negative
        var i = 0
        var prev = -1
        while i < len(milestones):
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

    fn get_lr(self) -> Float32:
        return self.current_lr

    fn _compute_lr(self, epoch: Int) -> Float32:
        var count = 0
        var i = 0
        while i < len(self.milestones):
            if epoch >= self.milestones[i]:
                count = count + 1
            i = i + 1
        var scale = 1.0
        var j = 0
        while j < count:
            scale = scale * self.gamma
            j = j + 1
        return self.base_lr * scale

    fn step(mut self) -> Float32:
        self.last_epoch = self.last_epoch + 1
        if self.last_epoch < 0:
            self.current_lr = self.base_lr
            return self.current_lr
        self.current_lr = self._compute_lr(self.last_epoch)
        return self.current_lr

    fn step_to(mut self, epoch: Int) -> Float32:
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
        s = s + ",'milestones':" + String(len(self.milestones))
        s = s + ",'last_epoch':" + String(self.last_epoch)
        s = s + ",'current_lr':" + String(self.current_lr) + "}"
        return s

    fn load_state_dict(mut self, state: String):
        self.current_lr = self.base_lr
        self.last_epoch = -1

# -----------------------------------------------------------------------------
# PolynomialLR: lr = min_lr + (base_lr - min_lr) * (1 - t/T_max)^power
# -----------------------------------------------------------------------------

struct PolynomialLR(Copyable, Movable):
    var base_lr: Float32
    var min_lr: Float32
    var power: Float32
    var T_max: Int
    var last_epoch: Int
    var current_lr: Float32

    fn __init__(
        out self,
        base_lr: Float32,
        T_max: Int,
        power: Float32 = 1.0,
        min_lr: Float32 = 0.0,
        last_epoch: Int = -1
    ):
        # ---- sanitize inputs (no asserts) ----
        var Tm = T_max
        if Tm <= 0: Tm = 1

        var lr0 = base_lr
        if lr0 < 0.0: lr0 = 0.0

        var lmin = min_lr
        if lmin < 0.0: lmin = 0.0

        # enforce invariant base_lr >= min_lr
        if lr0 < lmin:
            lr0 = lmin

        var p = power
        if p < 0.0: p = 0.0

        self.base_lr = lr0
        self.min_lr  = lmin
        self.power   = p
        self.T_max   = Tm
        self.last_epoch = last_epoch       # -1 => not started
        self.current_lr = lr0

    fn get_lr(self) -> Float32:
        return self.current_lr

    fn set_base_lr(mut self, lr: Float32):
        var v = lr
        if v < 0.0: v = 0.0
        if v < self.min_lr: v = self.min_lr
        self.base_lr = v

    fn set_min_lr(mut self, lr: Float32):
        var v = lr
        if v < 0.0: v = 0.0
        if self.base_lr < v:
            self.base_lr = v
        self.min_lr = v

    fn set_power(mut self, p: Float32):
        var v = p
        if v < 0.0: v = 0.0
        self.power = v

    @always_inline
    fn _ln_approx_around1(f: Float32) -> Float32:
        # تقریب تیلور ln(f) دور و بر 1: ln f ≈ (f-1) - (f-1)^2/2 + (f-1)^3/3 - (f-1)^4/4
        var x = f
        if x <= 0.0: x = 1e-12
        var d  = x - 1.0
        var d2 = d * d
        var d3 = d2 * d
        var d4 = d3 * d
        return d - 0.5 * d2 + (1.0/3.0) * d3 - 0.25 * d4

    fn _pow_frac(self, frac: Float32, power: Float32) -> Float32:
        # محاسبهٔ frac^power بدون math pow/exp
        # اگر power عدد صحیح باشد، ضرب تکراری؛ در غیر اینصورت از exp(p*ln(frac)) با تقریب ln استفاده می‌کنیم.
        var f = frac
        if f < 0.0: f = 0.0
        if f > 1.0: f = 1.0

        # quick outs
        if power == 0.0: return 1.0
        if f == 0.0:     return 0.0
        if f == 1.0:     return 1.0

        var ip = Int(power)
        var rem = power - Float32(ip)

        # integer part
        var decay = 1.0
        if ip > 0:
            var k = 0
            while k < ip:
                decay = decay * f
                k = k + 1
        # fractional part using ln approx around 1
        if rem > 0.0:
            var lnx = self._ln_approx_around1(f)    # ~ ln(f)
            var exp_arg = rem * lnx
            # exp(z) ≈ 1 + z + z^2/2 + z^3/6 (سری کوتاه)
            var z  = exp_arg
            var z2 = z * z
            var z3 = z2 * z
            var exp_approx = 1.0 + z + 0.5 * z2 + (1.0/6.0) * z3
            decay = decay * exp_approx

        # clamp for numeric safety
        if decay < 0.0: decay = 0.0
        if decay > 1.0: decay = 1.0
        return decay

    fn _compute_lr(self, t: Int) -> Float32:
        # clamp t to [0, T_max]
        var tt = t
        if tt < 0: tt = 0
        if tt > self.T_max: tt = self.T_max

        # decay = (1 - t/T_max)^power
        var frac = 1.0 - (Float32(tt) / Float32(self.T_max))
        var decay = self._pow_frac(frac, self.power)

        return self.min_lr + (self.base_lr - self.min_lr) * decay

    fn step(mut self) -> Float32:
        self.last_epoch = self.last_epoch + 1
        if self.last_epoch < 0:
            self.current_lr = self.base_lr
            return self.current_lr
        self.current_lr = self._compute_lr(self.last_epoch)
        return self.current_lr

    fn step_to(mut self, epoch: Int) -> Float32:
        self.last_epoch = epoch
        if epoch < 0:
            self.current_lr = self.base_lr
            return self.current_lr
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
        # بازگردانی حداقلی امن (پارسر برای استرینگ non-JSON اضافه نشده)
        self.current_lr = self.base_lr
        self.last_epoch = -1
# -----------------------------------------------------------------------------
# CosineAnnealingLR
# lr(t) = min_lr + 0.5*(base_lr - min_lr)*(1 + cos(pi*t/T_max)), t in [0, T_max]
# -----------------------------------------------------------------------------


struct CosineAnnealingLR(Copyable, Movable):
    var base_lr: Float32
    var min_lr: Float32
    var T_max: Int
    var last_epoch: Int
    var current_lr: Float32

    fn __init__(out self, base_lr: Float32, T_max: Int, min_lr: Float32 = 0.0, last_epoch: Int = -1):
        # ---- sanitize inputs (no assert) ----
        var Tm = T_max
        if Tm <= 0: Tm = 1

        var lr0 = base_lr
        if lr0 < 0.0: lr0 = 0.0

        var lmin = min_lr
        if lmin < 0.0: lmin = 0.0

        # enforce invariant base_lr >= min_lr
        if lr0 < lmin:
            # bring base up to min
            lr0 = lmin

        self.base_lr = lr0
        self.min_lr = lmin
        self.T_max = Tm
        self.last_epoch = last_epoch   # -1 means "not started"
        self.current_lr = lr0

    fn get_lr(self) -> Float32:
        return self.current_lr

    fn set_base_lr(mut self, lr: Float32):
        var v = lr
        if v < 0.0: v = 0.0
        # keep invariant base_lr >= min_lr
        if v < self.min_lr:
            v = self.min_lr
        self.base_lr = v
        # not auto-recomputing current_lr; call step()/step_to() after changing

    fn set_min_lr(mut self, lr: Float32):
        var v = lr
        if v < 0.0: v = 0.0
        # keep invariant base_lr >= min_lr
        if self.base_lr < v:
            self.base_lr = v
        self.min_lr = v

    fn _compute_lr(self, t: Int) -> Float32:
        # clamp t into [0, T_max]
        var tt = t
        if tt < 0: tt = 0
        if tt > self.T_max: tt = self.T_max

        # r in [0,1]
        var r = Float32(tt) / Float32(self.T_max)
        # cos(pi * r) approximation
        var c = _cos_pi_approx(r)
        # standard cosine annealing formula:
        # lr = min_lr + 0.5*(base_lr - min_lr)*(1 + cos(pi * t/T_max))
        var frac = 0.5 * (1.0 + c)
        return self.min_lr + (self.base_lr - self.min_lr) * frac

    fn step(mut self) -> Float32:
        self.last_epoch = self.last_epoch + 1
        if self.last_epoch < 0:
            self.current_lr = self.base_lr
            return self.current_lr
        self.current_lr = self._compute_lr(self.last_epoch)
        return self.current_lr

    fn step_to(mut self, epoch: Int) -> Float32:
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
        # Minimal safe restore (state parsing intentionally omitted):
        self.current_lr = self.base_lr
        self.last_epoch = -1
# -----------------------------------------------------------------------------
# CosineAnnealingWarmRestarts
# Warm restarts with periods: T_0, T_0*T_mult, T_0*T_mult^2, ...
# -----------------------------------------------------------------------------
@always_inline
fn _cos_pi_approx(r: Float32) -> Float32:
    # cos(π r) ≈ 1 - 4 r^2 + (4/3) r^4,   r∈[0,1]
    var x = r
    if x < 0.0: x = 0.0
    if x > 1.0: x = 1.0
    var x2 = x * x
    return 1.0 - 4.0 * x2 + (4.0 / 3.0) * x2 * x2

struct CosineAnnealingWarmRestarts(Copyable, Movable):
    var base_lr: Float32
    var min_lr: Float32
    var T_0: Int
    var T_mult: Int
    var last_epoch: Int
    var current_lr: Float32

    # internal counters
    var _Ti: Int     # current cycle length
    var _t_cur: Int  # position in current cycle (0.._Ti-1)

    fn __init__(
        out self,
        base_lr: Float32,
        T_0: Int,
        T_mult: Int = 1,
        min_lr: Float32 = 0.0,
        last_epoch: Int = -1
    ):
        # ---- sanitize inputs ----
        var lr0 = base_lr
        if lr0 < 0.0: lr0 = 0.0

        var lmin = min_lr
        if lmin < 0.0: lmin = 0.0

        if lr0 < lmin:
            lr0 = lmin

        var t0 = T_0
        if t0 <= 0: t0 = 1

        var tm = T_mult
        if tm < 1: tm = 1

        self.base_lr = lr0
        self.min_lr  = lmin
        self.T_0     = t0
        self.T_mult  = tm
        self.last_epoch = last_epoch   # -1 => not started
        self.current_lr = lr0
        self._Ti = t0
        self._t_cur = -1               # so that first step() sets it to 0

    fn get_lr(self) -> Float32:
        return self.current_lr

    fn set_base_lr(mut self, lr: Float32):
        var v = lr
        if v < 0.0: v = 0.0
        if v < self.min_lr: v = self.min_lr
        self.base_lr = v
        # current_lr به صورت تنبل با step()/step_to() به‌روز می‌شود

    fn set_min_lr(mut self, lr: Float32):
        var v = lr
        if v < 0.0: v = 0.0
        if self.base_lr < v:
            self.base_lr = v
        self.min_lr = v

    fn _compute_lr(self) -> Float32:
        # r = t_cur / Ti  ϵ [0,1]
        var tt = self._t_cur
        if tt < 0: tt = 0
        if tt > self._Ti: tt = self._Ti
        var r = Float32(tt) / Float32(self._Ti)
        var c = _cos_pi_approx(r)
        # lr = min_lr + 0.5*(base_lr - min_lr)*(1 + cos(pi * r))
        var frac = 0.5 * (1.0 + c)
        return self.min_lr + (self.base_lr - self.min_lr) * frac

    fn _advance_cycle(mut self):
        # Called when current cycle is finished
        self._t_cur = 0
        # grow cycle length if T_mult > 1
        var next_T = self._Ti * self.T_mult
        if next_T <= 0: next_T = self._Ti  # safety
        self._Ti = next_T

    fn step(mut self) -> Float32:
        # Typical: call once per epoch AFTER optimizer step
        self.last_epoch = self.last_epoch + 1

        # advance position in current cycle
        self._t_cur = self._t_cur + 1
        if self._t_cur >= self._Ti:
            self._advance_cycle()

        self.current_lr = self._compute_lr()
        return self.current_lr

    fn step_to(mut self, epoch: Int) -> Float32:
        # Recompute cycle position deterministically from epoch
        self.last_epoch = epoch
        var e = epoch
        if e < 0: e = 0

        # start from the first cycle
        self._Ti = self.T_0
        var remaining = e

        # subtract full cycles
        while remaining >= self._Ti:
            remaining = remaining - self._Ti
            var nxt = self._Ti * self.T_mult
            if nxt <= 0: nxt = self._Ti    # safety
            self._Ti = nxt

        # remaining is position in current cycle
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
        # Minimal safe restore (no parser for non-JSON string)
        self.current_lr = self.base_lr
        self.last_epoch = -1
        self._Ti = self.T_0
        self._t_cur = -1

# -----------------------------------------------------------------------------
# LinearWarmup: linearly increase LR from start_lr to base_lr over warmup_steps
# -----------------------------------------------------------------------------

struct LinearWarmup(Copyable, Movable):
    var start_lr: Float32
    var base_lr: Float32
    var warmup_steps: Int
    var last_epoch: Int
    var current_lr: Float32

    fn __init__(
        out self,
        start_lr: Float32,
        base_lr: Float32,
        warmup_steps: Int,
        last_epoch: Int = -1
    ):
        # ---- sanitize inputs ----
        var s = start_lr
        if s < 0.0: s = 0.0

        var b = base_lr
        if b < 0.0: b = 0.0

        var ws = warmup_steps
        if ws <= 0: ws = 1

        self.start_lr = s
        self.base_lr = b
        self.warmup_steps = ws
        self.last_epoch = last_epoch      # -1 => not started
        # مقدار جاری را با start_lr شروع می‌کنیم
        self.current_lr = s

    fn get_lr(self) -> Float32:
        return self.current_lr

    fn set_start_lr(mut self, lr: Float32):
        var v = lr
        if v < 0.0: v = 0.0
        self.start_lr = v
        # بروزرسانی current_lr به‌صورت تنبل با step()/step_to()

    fn set_base_lr(mut self, lr: Float32):
        var v = lr
        if v < 0.0: v = 0.0
        self.base_lr = v

    fn set_warmup_steps(mut self, steps: Int):
        var k = steps
        if k <= 0: k = 1
        self.warmup_steps = k

    fn _compute_lr(self, t: Int) -> Float32:
        # t منفی را صفر می‌گیریم
        var tt = t
        if tt < 0: tt = 0

        if tt >= self.warmup_steps:
            return self.base_lr

        # lr = start + (base-start) * tt / warmup_steps
        var frac = Float32(tt) / Float32(self.warmup_steps)
        return self.start_lr + (self.base_lr - self.start_lr) * frac

    fn step(mut self) -> Float32:
        # معمولاً هر epoch یک‌بار، بعد از optimizer.step()
        self.last_epoch = self.last_epoch + 1
        self.current_lr = self._compute_lr(self.last_epoch)
        return self.current_lr

    fn step_to(mut self, epoch: Int) -> Float32:
        # پرش مستقیم به اندیس epoch دلخواه
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
        # بازگردانی حداقلی امن (پارسر برای استرینگ غیر-JSON نداریم)
        self.current_lr = self.start_lr
        self.last_epoch = -1
# -----------------------------------------------------------------------------
# WarmupThen: compose a warmup scheduler with a main scheduler
#   For t < warmup_steps: delegate to warmup; else to main (with shifted epoch)
# -----------------------------------------------------------------------------

struct WarmupThen:
    var warmup: LinearWarmup
    var main: CosineAnnealingLR  # can be any scheduler; using Cosine as a typical one
    var warmup_steps: Int
    var last_epoch: Int
    var current_lr: Float32

    fn __init__(out self, warmup: LinearWarmup, main: CosineAnnealingLR):
        self.warmup = warmup
        self.main = main
        self.warmup_steps = warmup.warmup_steps
        self.last_epoch = -1
        self.current_lr = warmup.start_lr

    fn get_lr(self) -> Float32:
        return self.current_lr

    fn step(mut self) -> Float32:
        self.last_epoch = self.last_epoch + 1
        if self.last_epoch < self.warmup_steps:
            self.current_lr = self.warmup.step()
            return self.current_lr
        var shifted = self.last_epoch - self.warmup_steps
        self.current_lr = self.main.step_to(shifted)
        return self.current_lr

    fn step_to(mut self, epoch: Int) -> Float32:
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

struct ChainedLR(Copyable, Movable):
    var schedulers: List[CosineAnnealingLR]
    var spans: List[Int]          # spans[i] >= 0, except a negative span to mean "rest"
    var last_epoch: Int
    var current_lr: Float32

    fn __init__(out self, schedulers: List[CosineAnnealingLR], spans: List[Int]):
        # ---- sanitize inputs ----
        var n_s = len(schedulers)
        var n_p = len(spans)

        var sch = List[CosineAnnealingLR]()
        var spn = List[Int]()

        if n_s <= 0:
            # empty chain -> stable zero LR
            self.schedulers = sch
            self.spans = spn
            self.last_epoch = -1
            self.current_lr = 0.0
            return

        # align lengths
        var n = n_s
        if n_p < n_s:
            # copy given and pad the rest with a single "rest" (-1) only at the last
            var i = 0
            while i < n_p:
                spn.append(spans[i])
                i = i + 1
            while i < n_s - 1:
                spn.append(0)   # zero-length span for middle placeholders
                i = i + 1
            spn.append(-1)      # last takes the rest
        else:
            # truncate extra spans if any
            var i = 0
            while i < n_s:
                spn.append(spans[i])
                i = i + 1

        # normalize spans: non-last negatives become 0; keep last negative as "rest"
        var last_idx = n - 1
        var i = 0
        while i < n:
            var v = spn[i]
            if i == last_idx:
                # allow negative meaning "rest of time"
                # but if zero/negatives are not desired, keep as -1
                if v == 0:
                    spn[i] = -1
                # if positive, keep it as is (chain will clamp beyond)
            else:
                # non-last negative spans become 0-length
                if v < 0:
                    spn[i] = 0
            i = i + 1

        # copy schedulers
        var j = 0
        while j < n_s:
            sch.append(schedulers[j])
            j = j + 1

        self.schedulers = sch
        self.spans = spn
        self.last_epoch = -1
        # initial LR = first scheduler's current lr
        self.current_lr = self.schedulers[0].get_lr()

    fn get_lr(self) -> Float32:
        return self.current_lr

    fn _locate(self, t: Int) -> (Int, Int):
        # Returns (which_scheduler_idx, local_t) for the chain.
        var n = len(self.schedulers)
        if n == 0:
            return (0, 0)

        var acc = 0
        var i = 0
        while i < n:
            var sp = self.spans[i]
            if sp < 0:
                # "rest of time" → stick with this scheduler
                return (i, t - acc)
            if t < acc + sp:
                return (i, t - acc)
            acc = acc + sp
            i = i + 1

        # beyond all spans: clamp to last scheduler
        var last = n - 1
        var last_span = self.spans[last]
        if last_span < 0:
            # last is "rest"
            return (last, t - (acc - last_span))  # local t doesn't really matter; use step_to below
        # positive finite span: clamp to its last valid local step
        var loc = last_span - 1
        if loc < 0: loc = 0
        return (last, loc)

    fn step(mut self) -> Float32:
        self.last_epoch = self.last_epoch + 1
        var n = len(self.schedulers)
        if n == 0:
            # empty chain: keep stable
            return self.current_lr

        var which_local = self._locate(self.last_epoch)
        var idx = which_local[0]
        var loc = which_local[1]

        # step underlying scheduler "to" the local epoch
        self.current_lr = self.schedulers[idx].step_to(loc)
        return self.current_lr

    fn step_to(mut self, epoch: Int) -> Float32:
        self.last_epoch = epoch
        var n = len(self.schedulers)
        if n == 0:
            return self.current_lr

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
        # Minimal safe restore
        self.last_epoch = -1
        if len(self.schedulers) > 0:
            self.current_lr = self.schedulers[0].get_lr()
        else:
            self.current_lr = 0.0

# -----------------------------------------------------------------------------
# ReduceLROnPlateau: reduce LR by factor if metric does not improve for 'patience' steps
# Mode:
#   - 'min' : improvement if metric < best - threshold
#   - 'max' : improvement if metric > best + threshold
# -----------------------------------------------------------------------------


struct ReduceLROnPlateau(Copyable, Movable):
    var lr: Float32
    var factor: Float32
    var patience: Int
    var threshold: Float32
    var cooldown: Int
    var min_lr: Float32
    var mode_max: Bool    # True for 'max', False for 'min'

    var _best: Float32
    var _num_bad: Int
    var _cooldown: Int

    fn __init__(
        out self,
        lr: Float32,
        factor: Float32 = 0.1,
        patience: Int = 10,
        threshold: Float32 = 1e-4,
        cooldown: Int = 0,
        min_lr: Float32 = 0.0,
        mode: String = "min"
    ):
        # ---- sanitize inputs (no asserts) ----
        var v_lr = lr
        if v_lr < 0.0: v_lr = 0.0

        var v_factor = factor
        if v_factor <= 0.0: v_factor = 0.1

        var v_pat = patience
        if v_pat < 0: v_pat = 0

        var v_thr = threshold
        if v_thr < 0.0: v_thr = 0.0

        var v_cd = cooldown
        if v_cd < 0: v_cd = 0

        var v_min = min_lr
        if v_min < 0.0: v_min = 0.0
        if v_lr < v_min: v_lr = v_min

        var is_max = False
        if mode == String("max"):
            is_max = True

        self.lr = v_lr
        self.factor = v_factor
        self.patience = v_pat
        self.threshold = v_thr
        self.cooldown = v_cd
        self.min_lr = v_min
        self.mode_max = is_max

        # initialize best
        if self.mode_max:
            self._best = -1.7976931348623157e308   # ~ -DBL_MAX
        else:
            self._best = 1.7976931348623157e308    # ~  DBL_MAX
        self._num_bad = 0
        self._cooldown = 0

    fn get_lr(self) -> Float32:
        return self.lr

    fn set_mode(mut self, mode: String):
        var is_max = False
        if mode == String("max"):
            is_max = True
        self.mode_max = is_max
        # reset best accordingly (optional; keep existing best if you prefer)
        if self.mode_max:
            self._best = -1.7976931348623157e308
        else:
            self._best = 1.7976931348623157e308
        self._num_bad = 0
        self._cooldown = 0

    fn set_min_lr(mut self, v: Float32):
        var x = v
        if x < 0.0: x = 0.0
        self.min_lr = x
        if self.lr < self.min_lr:
            self.lr = self.min_lr

    fn set_factor(mut self, v: Float32):
        var x = v
        if x <= 0.0: x = 0.1
        self.factor = x

    fn set_patience(mut self, k: Int):
        var x = k
        if x < 0: x = 0
        self.patience = x

    fn set_threshold(mut self, v: Float32):
        var x = v
        if x < 0.0: x = 0.0
        self.threshold = x

    fn set_cooldown(mut self, v: Int):
        var x = v
        if x < 0: x = 0
        self.cooldown = x

    fn _is_improved(self, metric: Float32) -> Bool:
        if self.mode_max:
            # improvement if metric strictly larger than best + threshold
            return metric > self._best + self.threshold
        # mode 'min'
        return metric < self._best - self.threshold

    fn step(mut self, metric: Float32) -> Float32:
        # cooldown handling (counts down to zero)
        if self._cooldown > 0:
            self._cooldown = self._cooldown - 1

        if self._is_improved(metric):
            self._best = metric
            self._num_bad = 0
            return self.lr

        # No improvement
        self._num_bad = self._num_bad + 1

        # Reduce if patience exceeded and not cooling down
        if self._num_bad > self.patience and self._cooldown == 0:
            var new_lr = self.lr * self.factor
            if new_lr < self.min_lr:
                new_lr = self.min_lr
            # apply
            self.lr = new_lr
            # start cooldown and reset counters
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
        # Minimal safe restore (پارسر برای استرینگ غیر-JSON اضافه نشده)
        if self.mode_max:
            self._best = -1.7976931348623157e308
        else:
            self._best = 1.7976931348623157e308
        self._num_bad = 0
        self._cooldown = 0
        if self.lr < self.min_lr:
            self.lr = self.min_lr
