# MIT License
# SPDX-License-Identifier: MIT
# Project: momijo
# File: src/momijo/examples/learn/sequential_full.mojo
# Description: Example: Linear → BN → ReLU → Dropout → Linear

from momijo.tensor import tensor
from momijo.learn.nn.layers import Linear, ReLU, BatchNorm1d, Dropout
from momijo.learn.nn.module import Module
from momijo.learn.api.sequential import Sequential

fn main():
    var net = Sequential()
    net.add(Module.from_linear( Linear(10, 32) ))
    net.add(Module.from_batchnorm1d( BatchNorm1d(32) ))
    net.add(Module.from_relu( ReLU() ))
    net.add(Module.from_dropout( Dropout(0.2) ))
    net.add(Module.from_linear( Linear(32, 2) ))
    var x = tensor.randn([4,10])
    var y = net.forward(x)
    print("y shape:", y.shape())
