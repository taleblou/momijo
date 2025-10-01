# Project:      Momijo
# Module:       learn.optim.schedulers
# File:         optim/schedulers.mojo
# Path:         src/momijo/learn/optim/schedulers.mojo
#
# Description:  Learning rate schedulers for Momijo Learn (backend-agnostic).
#               - StepLR: decay LR by gamma every 'step_size' epochs.
#               - CosineAnnealingLR: cosine schedule from base_lr to min_lr over T_max steps.
#               Each `step()` returns the new LR so the caller can apply it to the optimizer.
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
#   - Types: StepLR, CosineAnnealingLR
#   - Key fns: step() -> Float64, get_lr() -> Float64, state_dict()/load_state_dict()
#   - Caller pattern:
#       var lr = scheduler.step()
#       optimizer.lr = lr

# -----------------------------------------------------------------------------
# Tiny numeric helpers (since stdlib.math isn't available in this project)
# -----------------------------------------------------------------------------

fn _abs(x: Float64) -> Float64:
    if x < 0.0:
        return -x
    return x

fn _fmod(x: Float64, y: Float64) -> Float64:
    # simple floating remainder x - y*floor(x/y)
    var q = x / y
    # truncate toward -inf for safety; emulate floor for positive/negative
    var qi = Int(q)
    if Float64(qi) > q:
        qi = qi - 1
    return x - (Float64(qi) * y)

fn _wrap_pi(x: Float64) -> Float64:
    # map x to [-PI, PI]
    var two_pi = 6.283185307179586476925286766559
    var y = _fmod(x, two_pi)
    if y > 3.1415926535897932384626433832795:
        y = y - two_pi
    if y < -3.1415926535897932384626433832795:
        y = y + two_pi
    return y

fn _cos_approx(x_in: Float64) -> Float64:
    # Range reduction then 6th-order even polynomial: cos x ≈ 1 - x^2/2! + x^4/4! - x^6/6!
    var x = _wrap_pi(x_in)
    var x2 = x * x
    var x4 = x2 * x2
    var x6 = x4 * x2
    return 1.0 - (x2 * 0.5) + (x4 * (1.0 / 24.0)) - (x6 * (1.0 / 720.0))

# -----------------------------------------------------------------------------
# StepLR
# -----------------------------------------------------------------------------

struct StepLR:
    var base_lr: Float64
    var gamma: Float64
    var step_size: Int
    var last_epoch: Int
    var current_lr: Float64

    fn __init__(out self, base_lr: Float64, step_size: Int, gamma: Float64 = 0.1, last_epoch: Int = -1):
        # Validate inputs
        assert(step_size > 0)
        assert(gamma > 0.0)
        assert(base_lr >= 0.0)

        self.base_lr = base_lr
        self.gamma = gamma
        self.step_size = step_size
        self.last_epoch = last_epoch
        # Initialize current_lr from base (before first step)
        self.current_lr = base_lr

    fn get_lr(self) -> Float64:
        return self.current_lr

    fn _compute_lr(self, epoch: Int) -> Float64:
        # epoch >= 0
        # number of decays = floor(epoch / step_size)
        var k = epoch // self.step_size
        # pow(gamma, k) without stdlib pow: repeated multiplication
        var scale = 1.0
        var i = 0
        while i < k:
            scale = scale * self.gamma
            i = i + 1
        return self.base_lr * scale

    # Advance one epoch/step and return new LR.
    fn step(mut self) -> Float64:
        self.last_epoch = self.last_epoch + 1
        if self.last_epoch < 0:
            self.current_lr = self.base_lr
            return self.current_lr
        self.current_lr = self._compute_lr(self.last_epoch)
        return self.current_lr

    # Serialize minimal state (for checkpoints)
    fn state_dict(self) -> String:
        # lightweight JSON-like string (actual JSON writer can replace this)
        return String("{'type':'StepLR','base_lr':") + String(self.base_lr) +
               String(",'gamma':") + String(self.gamma) +
               String(",'step_size':") + String(self.step_size) +
               String(",'last_epoch':") + String(self.last_epoch) +
               String(",'current_lr':") + String(self.current_lr) + String("}")

    # Best-effort loader (expects values as written by state_dict)
    fn load_state_dict(mut self, state: String):
        # Placeholder: in real impl parse JSON. Here we just reset to base.
        # Keep API for compatibility; no-op to avoid partial/buggy parsing.
        self.current_lr = self.base_lr
        self.last_epoch = -1


# -----------------------------------------------------------------------------
# CosineAnnealingLR
# -----------------------------------------------------------------------------
# lr(t) = min_lr + 0.5 * (base_lr - min_lr) * (1 + cos(pi * t / T_max)), t in [0, T_max]
# After T_max, it keeps cycling at the end value unless caller resets.

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

    fn _compute_lr(self, t: Int) -> Float64:
        # clamp t in [0, T_max]
        var tt = t
        if tt < 0:
            tt = 0
        if tt > self.T_max:
            tt = self.T_max
        var pi = 3.1415926535897932384626433832795
        var cos_term = _cos_approx(pi * (Float64(tt) / Float64(self.T_max)))
        # 0.5*(1+cos) goes from 1 -> 0 over [0, T_max]
        var frac = 0.5 * (1.0 + cos_term)
        return self.min_lr + (self.base_lr - self.min_lr) * frac

    fn step(mut self) -> Float64:
        self.last_epoch = self.last_epoch + 1
        if self.last_epoch < 0:
            self.current_lr = self.base_lr
            return self.current_lr
        self.current_lr = self._compute_lr(self.last_epoch)
        return self.current_lr

    fn state_dict(self) -> String:
        return String("{'type':'CosineAnnealingLR','base_lr':") + String(self.base_lr) +
               String(",'min_lr':") + String(self.min_lr) +
               String(",'T_max':") + String(self.T_max) +
               String(",'last_epoch':") + String(self.last_epoch) +
               String(",'current_lr':") + String(self.current_lr) + String("}")

    fn load_state_dict(mut self, state: String):
        # Placeholder: reset; real impl should parse JSON.
        self.current_lr = self.base_lr
        self.last_epoch = -1
