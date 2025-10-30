# MIT License
# SPDX-License-Identifier: MIT
# Project:      Momijo
# Module:       learn.amp.autocast
# File:         src/momijo/learn/amp/autocast.mojo
#
# Description:
#   Automatic Mixed Precision (AMP) utilities for Momijo Learn.
#   - AMPPrecision: compact precision policy (fp32/fp16/bf16).
#   - autocast: lightweight controller to toggle AMP and carry precision.
#   - GradScaler: scale/unscale values, manage growth/backoff, and coordinate
#                 optimizer steps under potential FP16/BF16 overflow.
#
# Author(s):    Morteza Taleblou & Mitra Daneshmand
# Website:      https://taleblou.ir/
# Repository:   https://github.com/taleblou/momijo
#
# Notes:
#   - This module is backend-agnostic: it does not import tensor backends.
#   - Optimizer duck-typing: expects optimizer to expose step() and zero_grad().
#   - Use .scale(loss) before backward, and .unscale(grads) before clipping/checks.
#   - To wire found_inf detection, your training loop should check grads (e.g.,
#     isfinite) in your tensor backend and pass found_inf=True to .step/.update.

# -----------------------------------------------------------------------------
# Minimal tensor-facing import (no backend coupling)
# -----------------------------------------------------------------------------
from momijo.tensor.tensor import Tensor

# -----------------------------------------------------------------------------
# AMP precision policy
# -----------------------------------------------------------------------------

struct AMPPrecision:
    var value: Int

    fn __init__(out self, value: Int):
        self.value = value

    @staticmethod
    fn fp32() -> AMPPrecision:
        return AMPPrecision(32)

    @staticmethod
    fn fp16() -> AMPPrecision:
        return AMPPrecision(16)

    @staticmethod
    fn bf16() -> AMPPrecision:
        # Effective precision bits ~15; represented here with sentinel 15.
        return AMPPrecision(15)

    fn __str__(self) -> String:
        if self.value == 32:
            return "AMPPrecision(fp32)"
        if self.value == 16:
            return "AMPPrecision(fp16)"
        if self.value == 15:
            return "AMPPrecision(bf16)"
        return "AMPPrecision(" + String(self.value) + ")"


# -----------------------------------------------------------------------------
# autocast controller
# -----------------------------------------------------------------------------
# Carries the AMP policy; kernels/layers can query it to decide casting.

struct autocast:
    var enabled: Bool
    var precision: AMPPrecision

    fn __init__(out self, enabled: Bool = True, precision: AMPPrecision = AMPPrecision.fp16()):
        self.enabled = enabled
        self.precision = precision

    fn enable(mut self) -> autocast:
        self.enabled = True
        return self

    fn disable(mut self) -> autocast:
        self.enabled = False
        return self

    fn set_precision(mut self, p: AMPPrecision) -> autocast:
        self.precision = p
        return self

    fn is_enabled(self) -> Bool:
        return self.enabled

    fn get_precision(self) -> AMPPrecision:
        return self.precision

    fn __str__(self) -> String:
        var s = "autocast(enabled="
        s = s + (String("true") if self.enabled else String("false"))
        s = s + ", precision=" + String(self.precision) + ")"
        return s


# -----------------------------------------------------------------------------
# AutocastGuard (optional RAII scope)
# -----------------------------------------------------------------------------
# Temporarily flips autocast enabled state within a scope and restores on drop.

struct AutocastGuard:
    var ref: Pointer[autocast]
    var prev_enabled: Bool
    var prev_precision: AMPPrecision

    fn __init__(out self, ref: Pointer[autocast], enable: Bool? = None, precision: AMPPrecision? = None):
        self.ref = ref
        self.prev_enabled = ref.value.enabled
        self.prev_precision = ref.value.precision
        if enable is not None:
            ref.value.enabled = enable.value()
        if precision is not None:
            ref.value.precision = precision.value()

    fn __del__(deinit self):
        # Restore previous state
        if self.ref is not None:
            self.ref.value.enabled = self.prev_enabled
            self.ref.value.precision = self.prev_precision


# -----------------------------------------------------------------------------
# GradScaler
# -----------------------------------------------------------------------------
# Scales losses to mitigate FP16 underflow; adapts scale with growth/backoff.

struct GradScaler:
    var enabled: Bool
    var scale_factor: Float64
    var growth_factor: Float64
    var backoff_factor: Float64
    var growth_interval: Int
    var growth_tracker: Int
    var min_scale: Float64
    var max_scale: Float64

    fn __init__(
        out self,
        enabled: Bool = True,
        init_scale: Float64 = 65536.0,    # 2^16 default-like
        growth_factor: Float64 = 2.0,     # multiplicative growth
        backoff_factor: Float64 = 0.5,    # multiplicative decay on overflow
        growth_interval: Int = 2000,      # steps between successful growths
        min_scale: Float64 = 1.0,
        max_scale: Float64 = 1.8446744e19 # ~2^64 / 10 as a safety cap
    ):
        self.enabled = enabled
        self.scale_factor = init_scale
        self.growth_factor = growth_factor
        self.backoff_factor = backoff_factor
        self.growth_interval = growth_interval
        self.growth_tracker = 0
        self.min_scale = min_scale
        self.max_scale = max_scale

    # -----------------------------
    # Scale / Unscale — Scalars
    # -----------------------------
    fn scale(self, loss: Float32) -> Float32:
        return Float32(loss * Float32(self.scale_factor)) if self.enabled else loss

    fn scale(self, loss: Float64) -> Float64:
        return loss * self.scale_factor if self.enabled else loss

    fn unscale(self, value: Float32) -> Float32:
        if not self.enabled: return value
        if self.scale_factor == 0.0: return value
        return Float32(value / Float32(self.scale_factor))

    fn unscale(self, value: Float64) -> Float64:
        if not self.enabled: return value
        if self.scale_factor == 0.0: return value
        return value / self.scale_factor

    # -----------------------------
    # Scale / Unscale — Tensors
    # -----------------------------
    # Assumption: Tensor * scalar and Tensor / scalar are defined in momijo.tensor
    fn scale[T: Copyable & Movable](self, t: Tensor[T]) -> Tensor[T]:
        if self.enabled:
            return t * self.scale_factor
        return t

    fn unscale[T: Copyable & Movable](self, t: Tensor[T]) -> Tensor[T]:
        if not self.enabled: return t
        if self.scale_factor == 0.0: return t
        return t / self.scale_factor

    # -----------------------------
    # Utilities
    # -----------------------------
    fn is_enabled(self) -> Bool:
        return self.enabled

    fn get_scale(self) -> Float64:
        return self.scale_factor

    fn clamp_scale(mut self, lo: Float64, hi: Float64) -> GradScaler:
        # Keep scale within [lo, hi]
        var s = self.scale_factor
        if s < lo: s = lo
        if s > hi: s = hi
        self.scale_factor = s
        return self

    fn reset(mut self, new_scale: Float64 = 65536.0) -> GradScaler:
        self.scale_factor = new_scale
        self.growth_tracker = 0
        return self

    # -----------------------------
    # Optimizer Integration
    # -----------------------------
    # Expects optimizer.step() and optimizer.zero_grad() exist (duck-typed).
    fn step(mut self, optimizer, found_inf: Bool = False):
        if self.enabled and found_inf:
            return
        optimizer.step()

    fn update(mut self, found_inf: Bool = False):
        if not self.enabled:
            return

        if found_inf:
            self.scale_factor = self.scale_factor * self.backoff_factor
            if self.scale_factor < self.min_scale:
                self.scale_factor = self.min_scale
            self.growth_tracker = 0
            return

        self.growth_tracker = self.growth_tracker + 1
        if self.growth_tracker >= self.growth_interval:
            var s = self.scale_factor * self.growth_factor
            if s > self.max_scale:
                s = self.max_scale
            self.scale_factor = s
            self.growth_tracker = 0

    fn step_and_update(mut self, optimizer, found_inf: Bool = False, zero_grad_after: Bool = True):
        self.step(optimizer, found_inf)
        self.update(found_inf)
        if zero_grad_after:
            optimizer.zero_grad()

    fn __str__(self) -> String:
        var s = "GradScaler(enabled="
        s = s + (String("true") if self.enabled else String("false"))
        s = s + ", scale=" + String(self.scale_factor)
        s = s + ", growth_factor=" + String(self.growth_factor)
        s = s + ", backoff_factor=" + String(self.backoff_factor)
        s = s + ", growth_interval=" + String(self.growth_interval)
        s = s + ", tracker=" + String(self.growth_tracker)
        s = s + ", min_scale=" + String(self.min_scale)
        s = s + ", max_scale=" + String(self.max_scale) + ")"
        return s
