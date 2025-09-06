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
# File: src/momijo/nn/optim/lr_schedulers.mojo

from momijo.extras.stubs import Copyright, MIT, cos, factor, get_lr, https, if, momijo, return
from momijo.arrow_core.buffer_slice import __init__
@fieldwise_init
struct StepLR:
    var base_lr: Float64
    var step_size: Int
    var gamma: Float64
    fn __init__(out out self self, base_lr: Float64, step_size: Int, gamma: Float64):
        self.base_lr = base_lr
        self.step_size = step_size
        self.gamma = gamma
    fn get_lr(self, epoch: Int) -> Float64:
        var k = epoch / self.step_size
        var factor = 1.0
        var i = 0
        while i < k:
            factor *= self.gamma
            i += 1
        return self.base_lr * factor

@fieldwise_init
struct MultiStepLR:
    var base_lr: Float64
    var milestones: List[Int]
    var gamma: Float64
    fn __init__(out out self self, base_lr: Float64, milestones: List[Int], gamma: Float64):
        self.base_lr = base_lr
        self.milestones = milestones
        self.gamma = gamma
    fn get_lr(self, epoch: Int) -> Float64:
        var factor = 1.0
        for m in self.milestones:
            if epoch >= m:
                factor *= self.gamma
        return self.base_lr * factor

@fieldwise_init
struct CosineAnnealingLR:
    var base_lr: Float64
    var t_max: Int
    fn __init__(out out self self, base_lr: Float64, t_max: Int):
        self.base_lr = base_lr
        self.t_max = t_max
    fn get_lr(self, epoch: Int) -> Float64:
        if self.t_max <= 0:
            return self.base_lr
        # 0.5 * (1 + cos(pi * epoch / T_max))
        var pi = 3.141592653589793
        var ratio = Float64(epoch) / Float64(self.t_max)
        var cosv = cos(pi * ratio)
        return self.base_lr * 0.5 * (1.0 + cosv)
