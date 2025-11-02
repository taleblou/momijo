# MIT License
# SPDX-License-Identifier: MIT
# Project: momijo
# File: src/momijo/examples/learn/sequential_min.mojo
# Description: Minimal Sequential demo: Linear(10,32) -> ReLU -> Linear(32,2)

from momijo.tensor import tensor
from momijo.learn.nn.layers import Linear, ReLU
from momijo.learn.nn.module import Module
from momijo.learn.api.sequential import Sequential

fn main():
    var net = Sequential()
    net.add(Module.from_linear( Linear(10, 32)  ))
    net.add(Module.from_relu(   ReLU()          ))
    net.add(Module.from_linear( Linear(32, 2)   ))
    var x = tensor.randn([4,10])
    var y = net.forward(x)
    print("y shape:", y.shape())
