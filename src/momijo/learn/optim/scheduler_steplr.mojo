# MIT License
# SPDX-License-Identifier: MIT
# Project: momijo
# File: src/momijo/learn/optim/scheduler_steplr.mojo
# Description: StepLR scheduler.

struct StepLR:
    var base_lr: Float64
    var gamma: Float64
    var step_size: Int
    var epoch: Int

    fn __init__(out self, base_lr: Float64, step_size: Int, gamma: Float64 = 0.1):
        self.base_lr = base_lr; self.gamma = gamma; self.step_size = step_size; self.epoch = 0

    fn __copyinit__(out self, other: Self):
        self.base_lr = other.base_lr; self.gamma = other.gamma; self.step_size = other.step_size; self.epoch = other.epoch

    fn step(mut self) -> Float64:
        self.epoch = self.epoch + 1
        var k = self.epoch // self.step_size
        return self.base_lr * (self.gamma ** Float64(k))
