# MIT License
# SPDX-License-Identifier: MIT
# Project: momijo
# File: src/momijo/examples/learn/train_full.mojo
# Description: Training loop (MLP) with SGD and softmax CE.

from momijo.tensor import tensor
from momijo.learn.nn.layers import Linear, ReLU
from momijo.learn.losses.losses import softmax_cross_entropy, softmax
from momijo.learn.optim.sgd import SGD

fn main():
    var N = 64; var D = 10; var H = 32; var C = 2; var epochs = 3
    var l1 = Linear(D, H); var relu = ReLU(); var l2 = Linear(H, C); var opt = SGD(0.1)
    var x = tensor.randn([N, D])
    var r = tensor.randn([N, 1]); var p = tensor.exp(r) / (tensor.exp(r) + tensor.exp(-r))
    var half = tensor.zeros([N,1]) + 0.5; var q = p - half
    var ind1 = tensor.maximum_scalar(q, 0.0) / (tensor.maximum_scalar(q, 0.0) + 1e-12)
    var ind0 = 1.0 - ind1
    var y = tensor.concat([ind0, ind1], axis=1)

    var ep = 0
    while ep < epochs:
        var h1 = l1.forward(x); var a1 = relu.forward(h1); var logits = l2.forward(a1)
        var pair = softmax_cross_entropy(logits, y); var loss = pair[0]; var dlogits = pair[1]
        var dW2 = tensor.matmul(dlogits.transpose(), a1); var onesN = tensor.ones([N,1])
        var db2 = tensor.matmul(dlogits.transpose(), onesN).transpose()[0]
        var da1 = tensor.matmul(dlogits, l2.weight)
        var mask = tensor.maximum_scalar(a1, 0.0) / (tensor.maximum_scalar(a1, 0.0) + 1e-12)
        var dh1 = da1 * mask
        var dW1 = tensor.matmul(dh1.transpose(), x); var db1 = tensor.matmul(dh1.transpose(), onesN).transpose()[0]
        opt.step_linear(l2, dW2, db2); opt.step_linear(l1, dW1, db1)
        print("epoch=", ep, " loss=", loss); ep += 1

    var probs = softmax(l2.forward(relu.forward(l1.forward(x))))
    print("probs shape:", probs.shape())
