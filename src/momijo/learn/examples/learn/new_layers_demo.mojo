# MIT License
# SPDX-License-Identifier: MIT
# Project: momijo
# File: src/momijo/examples/learn/new_layers_demo.mojo
# Description: Demo for AvgPool2d, ConvTranspose2d, and LayerNorm1d.

from momijo.tensor import tensor
from momijo.learn.nn.avgpool import AvgPool2d
from momijo.learn.nn.conv_transpose import ConvTranspose2d
from momijo.learn.nn.layernorm import LayerNorm1d

fn main():
    # AvgPool2d
    var x = tensor.randn([2, 3, 6, 6])
    var avg = AvgPool2d((2,2))
    var y = avg.forward(x)
    print("AvgPool out:", y.shape())

    # ConvTranspose2d (upsample-ish)
    var deconv = ConvTranspose2d(3, 2, (3,3), (2,2), (1,1))
    var z = deconv.forward(x)
    print("Deconv out:", z.shape())

    # LayerNorm1d over last dim
    var ln = LayerNorm1d(10)
    var t = tensor.randn([4,10])
    var out = ln.forward(t)
    print("LayerNorm1d out:", out.shape())
