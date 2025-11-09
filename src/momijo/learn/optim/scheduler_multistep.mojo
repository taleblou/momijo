# MIT License
# SPDX-License-Identifier: MIT
# Project: momijo
# File: src/momijo/learn/optim/scheduler_multistep.mojo
# Description: MultiStepLR scheduler.

struct MultiStepLR:
    var base_lr: Float32
    var gamma: Float32
    var milestones: List[Int]
    var epoch: Int

    fn __init__(out self, base_lr: Float32, milestones: List[Int], gamma: Float32 = 0.1):
        self.base_lr = base_lr
        self.gamma = gamma
        self.milestones = milestones
        self.epoch = 0

    fn __copyinit__(out self, other: Self):
        self.base_lr = other.base_lr
        self.gamma = other.gamma
        self.milestones = other.milestones
        self.epoch = other.epoch

    fn step(mut self) -> Float32:
        self.epoch = self.epoch + 1
        var drops = 0
        var i = 0
        while i < len(self.milestones):
            if self.epoch >= self.milestones[i]:
                drops += 1
            i += 1
        return self.base_lr * (self.gamma ** Float32(drops))
