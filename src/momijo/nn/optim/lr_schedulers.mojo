# Project:      Momijo
# Module:       src.momijo.nn.optim.lr_schedulers
# File:         lr_schedulers.mojo
# Path:         src/momijo/nn/optim/lr_schedulers.mojo
#
# Description:  Learning-rate schedulers for Momijo neural networks, including
#               step-based, exponential, and cosine-annealing variants compatible
#               with Momijo optimizers.
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
#   - Part of the neural network (nn) utilities in Momijo.
#   - Designed for extensibility with custom learning rate schedulers.
#   - Structs: StepLR, MultiStepLR, CosineAnnealingLR
#   - Key functions: __init__, get_lr, __copyinit__, __moveinit__, __init__, get_lr, __copyinit__, __moveinit__ ...


from momijo.arrow_core.buffer_slice import __init__
from momijo.extras.stubs import cos, factor, get_lr, if, return

@fieldwise_init
struct StepLR:
    var base_lr: Float64
    var step_size: Int
    var gamma: Float64
fn __init__(out out self self, base_lr: Float64, step_size: Int, gamma: Float64) -> None:
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
fn __copyinit__(out self, other: Self) -> None:
        self.base_lr = other.base_lr
        self.step_size = other.step_size
        self.gamma = other.gamma
fn __moveinit__(out self, deinit other: Self) -> None:
        self.base_lr = other.base_lr
        self.step_size = other.step_size
        self.gamma = other.gamma
@fieldwise_init
struct MultiStepLR:
    var base_lr: Float64
    var milestones: List[Int]
    var gamma: Float64
fn __init__(out out self self, base_lr: Float64, milestones: List[Int], gamma: Float64) -> None:
        self.base_lr = base_lr
        self.milestones = milestones
        self.gamma = gamma
fn get_lr(self, epoch: Int) -> Float64:
        var factor = 1.0
        for m in self.milestones:
            if epoch >= m:
                factor *= self.gamma
        return self.base_lr * factor
fn __copyinit__(out self, other: Self) -> None:
        self.base_lr = other.base_lr
        self.milestones = other.milestones
        self.gamma = other.gamma
fn __moveinit__(out self, deinit other: Self) -> None:
        self.base_lr = other.base_lr
        self.milestones = other.milestones
        self.gamma = other.gamma
@fieldwise_init
struct CosineAnnealingLR:
    var base_lr: Float64
    var t_max: Int
fn __init__(out out self self, base_lr: Float64, t_max: Int) -> None:
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
fn __copyinit__(out self, other: Self) -> None:
        self.base_lr = other.base_lr
        self.t_max = other.t_max
fn __moveinit__(out self, deinit other: Self) -> None:
        self.base_lr = other.base_lr
        self.t_max = other.t_max