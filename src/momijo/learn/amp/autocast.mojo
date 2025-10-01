# Project:      Momijo
# Module:       learn.amp.autocast
# File:         amp/autocast.mojo
# Path:         src/momijo/learn/amp/autocast.mojo
#
# Description:  Automatic Mixed Precision (AMP) utilities for Momijo Learn.
#               Provides an autocast controller (enable/disable + precision policy)
#               and a GradScaler that scales losses to reduce FP16 underflow and
#               adapts the scale factor with simple growth/backoff rules.
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
#   - Types: AMPPrecision, autocast, GradScaler
#   - Key fns: autocast.enable(), autocast.disable(), GradScaler.scale(loss),
#              GradScaler.step(optimizer, found_inf), GradScaler.update(found_inf)
#   - Backend-agnostic: does not assume a specific tensor type; you can later
#     wire casting logic and inf/NaN checks to momijo.tensor ops.

# -----------------------------------------------------------------------------
# AMP precision policy
# -----------------------------------------------------------------------------

struct AMPPrecision:
    var value: Int

    fn __init__(out self, value: Int):
        self.value = value

    # Common presets
    @staticmethod
    fn fp32() -> AMPPrecision:
        return AMPPrecision(32)

    @staticmethod
    fn fp16() -> AMPPrecision:
        return AMPPrecision(16)

    @staticmethod
    fn bf16() -> AMPPrecision:
        return AMPPrecision(15)  # BF16 has 8-bit exponent, 7-bit mantissa (≈15 bits precision)


# -----------------------------------------------------------------------------
# autocast controller
# -----------------------------------------------------------------------------
# This is a lightweight, backend-agnostic controller. It does not perform the
# casts itself; instead it carries the "policy" (enabled + target precision).
# Your kernels / layers can query this policy to decide whether to cast.

struct autocast:
    var enabled: Bool
    var precision: AMPPrecision

    fn __init__(out self, enabled: Bool = True, precision: AMPPrecision = AMPPrecision.fp16()):
        self.enabled = enabled
        self.precision = precision

    # Toggle on/off during a scope
    fn enable(mut self) -> autocast:
        self.enabled = True
        return self

    fn disable(mut self) -> autocast:
        self.enabled = False
        return self

    fn set_precision(mut self, p: AMPPrecision) -> autocast:
        self.precision = p
        return self

    # Query helpers
    fn is_enabled(self) -> Bool:
        return self.enabled

    fn get_precision(self) -> AMPPrecision:
        return self.precision


# -----------------------------------------------------------------------------
# GradScaler
# -----------------------------------------------------------------------------
# Scales the loss to improve FP16/BF16 numerical stability. The "found_inf"
# flag is provided by the caller (e.g., after checking grads for inf/NaN).
# If found_inf is true, the optimizer step is skipped and the scale factor
# is backed off; otherwise the scale factor can grow over time.

struct GradScaler:
    var enabled: Bool
    var scale_factor: Float64
    var growth_factor: Float64
    var backoff_factor: Float64
    var growth_interval: Int
    var growth_tracker: Int

    fn __init__(
        out self,
        enabled: Bool = True,
        init_scale: Float64 = 65536.0,        # 2^16 like PyTorch default
        growth_factor: Float64 = 2.0,
        backoff_factor: Float64 = 0.5,
        growth_interval: Int = 2000
    ):
        self.enabled = enabled
        self.scale_factor = init_scale
        self.growth_factor = growth_factor
        self.backoff_factor = backoff_factor
        self.growth_interval = growth_interval
        self.growth_tracker = 0

    # Returns "loss * scale" if AMP is enabled; otherwise returns loss unchanged.
    # Type of `loss` is intentionally generic; it must support multiplication by Float64.
    fn scale(self, loss):
        if self.enabled:
            return loss * self.scale_factor
        return loss

    # Unscale helper – returns (value / scale) if enabled; else returns value.
    # Useful if you want to unscale grads before clipping or inf/NaN checks.
    fn unscale(self, value):
        if self.enabled:
            if self.scale_factor == 0.0:
                # Avoid division by zero; return value as-is.
                return value
            return value / self.scale_factor
        return value

    # Perform an optimizer step unless an overflow/inf was detected.
    # The caller should pass found_inf=true if gradients contain inf/NaN.
    # Optimizer must implement: step(mut self) and zero_grad(mut self) (duck-typed).
    fn step(mut self, optimizer, found_inf: Bool = False):
        if self.enabled:
            if found_inf:
                # Skip the step on overflow.
                return
        # Proceed with the actual parameter update.
        optimizer.step()

    # Update the scale factor based on overflow signal.
    # If found_inf: scale *= backoff_factor and reset tracker.
    # Else: increase tracker; when it reaches growth_interval, scale *= growth_factor.
    fn update(mut self, found_inf: Bool = False):
        if not self.enabled:
            return

        if found_inf:
            self.scale_factor = self.scale_factor * self.backoff_factor
            if self.scale_factor < 1.0:
                self.scale_factor = 1.0  # clamp to a reasonable minimum
            self.growth_tracker = 0
            return

        # No overflow: consider growth
        self.growth_tracker = self.growth_tracker + 1
        if self.growth_tracker >= self.growth_interval:
            self.scale_factor = self.scale_factor * self.growth_factor
            self.growth_tracker = 0

    # Convenience: typical training step sequence in one call (optional).
    # 1) opt.step() unless overflow, 2) update scale, 3) opt.zero_grad()
    fn step_and_update(mut self, optimizer, found_inf: Bool = False, zero_grad_after: Bool = True):
        self.step(optimizer, found_inf)
        self.update(found_inf)
        if zero_grad_after:
            optimizer.zero_grad()
