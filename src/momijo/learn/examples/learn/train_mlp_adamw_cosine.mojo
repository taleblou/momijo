# MIT License
# SPDX-License-Identifier: MIT
# Project: momijo
# File: src/momijo/examples/learn/train_mlp_adamw_cosine.mojo
# Description: MLP training with AdamW + CosineAnnealingLR.

from momijo.tensor import tensor
from momijo.learn.nn.layers import Linear, ReLU
from momijo.learn.losses.losses import softmax_cross_entropy, softmax
from momijo.learn.optim.adamw import AdamW
from momijo.learn.optim.scheduler_cosine import CosineAnnealingLR

fn main():
    var N = 64; var D = 10; var H = 64; var C = 3; var epochs = 6
    var l1 = Linear(D, H); var relu = ReLU(); var l2 = Linear(H, C)
    var opt = AdamW(3e-3, weight_decay=1e-2)
    var sch = CosineAnnealingLR(opt.lr, 3, min_lr=3e-4)

    var x = tensor.randn([N, D])
    var y = softmax(tensor.randn([N, C]))  # soft one-hot

    var ep = 0
    while ep < epochs:
        var a1 = relu.forward(l1.forward(x))
        var logits = l2.forward(a1)
        var pair = softmax_cross_entropy(logits, y); var loss = pair[0]; var dlogits = pair[1]
        var dW2 = tensor.matmul(dlogits.transpose(), a1); var db2 = tensor.matmul(dlogits.transpose(), tensor.ones([N,1])).transpose()[0]
        var da1 = tensor.matmul(dlogits, l2.weight)
        var mask = tensor.maximum_scalar(a1, 0.0) / (tensor.maximum_scalar(a1, 0.0) + 1e-12)
        var dh1 = da1 * mask
        var dW1 = tensor.matmul(dh1.transpose(), x); var db1 = tensor.matmul(dh1.transpose(), tensor.ones([N,1])).transpose()[0]
        opt.step_linear(l2, dW2, db2); opt.step_linear(l1, dW1, db1)
        # scheduler step
        opt.lr = sch.step()
        print("epoch=", ep, " loss=", loss, " lr=", opt.lr)
        ep += 1
