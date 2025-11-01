# MIT License
# SPDX-License-Identifier: MIT
# Project: momijo
# File: src/momijo/learn/optim/scheduler_cosine.mojo
# Description: CosineAnnealingLR scheduler (no warm restarts).

fn _cos_taylor(x: Float64) -> Float64:
    var x2 = x * x
    var x4 = x2 * x2
    return 1.0 - (x2 / 2.0) + (x4 / 24.0)

struct CosineAnnealingLR:
    var base_lr: Float64
    var min_lr: Float64
    var T_max: Int
    var epoch: Int

    fn __init__(out self, base_lr: Float64, T_max: Int, min_lr: Float64 = 0.0):
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.T_max = T_max
        self.epoch = 0

    fn __copyinit__(out self, other: Self):
        self.base_lr = other.base_lr
        self.min_lr = other.min_lr
        self.T_max = other.T_max
        self.epoch = other.epoch

    fn step(mut self) -> Float64:
        self.epoch = self.epoch + 1
        var t = Float64(self.epoch % self.T_max)
        var c = _cos_taylor(3.141592653589793 * (t / Float64(self.T_max)))
        return self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1.0 + c)
