# MIT License
# SPDX-License-Identifier: MIT
# Project: momijo
# File: src/momijo/examples/learn/train_cnn_backprop.mojo
# Description: CNN training loop with manual backprop for Conv2d + Linear + ReLU (random data demo).

from momijo.tensor import tensor
from momijo.learn.nn.conv import Conv2d, MaxPool2d
from momijo.learn.nn.layers import ReLU, Linear
from momijo.learn.losses.losses import softmax_cross_entropy, softmax
from momijo.learn.optim.sgd import SGD

fn relu_backward(x: tensor.Tensor[Float64], grad_y: tensor.Tensor[Float64]) -> tensor.Tensor[Float64]:
    var pos = tensor.maximum_scalar(x, 0.0)
    var mask = pos / (pos + 1e-12)
    return grad_y * mask

fn main():
    var N = 16; var C = 1; var H = 16; var W = 16; var epochs = 3
    var conv = Conv2d(C, 4, (3,3), (1,1), (1,1))
    var relu = ReLU()
    var pool = MaxPool2d((2,2))
    var fc = Linear(4 * (H//2) * (W//2), 5)
    var opt = SGD(0.05)

    var x = tensor.randn([N, C, H, W])
    # Build random one-hot labels for 5 classes
    var raw = tensor.randn([N, 5])
    var y = softmax(raw)  # soft one-hot-ish
    # sharpen to pseudo one-hot
    var k = 64.0
    var w = tensor.exp(y * k)
    var p = w / (tensor.matmul(w, tensor.ones([5,1])))
    y = p  # target_onehot-ish

    var ep = 0
    while ep < epochs:
        var h = conv.forward(x)           # [N,4,H,W]
        var a = relu.forward(h)           # [N,4,H,W]
        var p2 = pool.forward(a)          # [N,4,H/2,W/2]
        var OH = H // 2; var OW = W // 2
        var flat = tensor.zeros([N, 4*OH*OW])
        var n = 0
        while n < N:
            var oc = 0
            while oc < 4:
                var y2 = 0
                while y2 < OH:
                    var x2 = 0
                    while x2 < OW:
                        var src = n*4*OH*OW + oc*OH*OW + y2*OW + x2
                        var dst = n*(4*OH*OW) + oc*OH*OW + y2*OW + x2
                        flat._data[dst] = p2._data[src]
                        x2 += 1
                    y2 += 1
                oc += 1
            n += 1
        var logits = fc.forward(flat)      # [N,5]

        var pair = softmax_cross_entropy(logits, y)
        var loss = pair[0]; var dlogits = pair[1]

        # Backprop to fc
        var dW2 = tensor.matmul(dlogits.transpose(), flat)
        var onesN = tensor.ones([N,1])
        var db2_col = tensor.matmul(dlogits.transpose(), onesN)
        var db2 = db2_col.transpose()[0]
        var dflat = tensor.matmul(dlogits, fc.weight)

        # Unflatten to pool grad
        var dp = tensor.zeros([N, 4, OH, OW])
        var n2 = 0
        while n2 < N:
            var oc2 = 0
            while oc2 < 4:
                var y3 = 0
                while y3 < OH:
                    var x3 = 0
                    while x3 < OW:
                        var dst = n2*4*OH*OW + oc2*OH*OW + y3*OW + x3
                        var src = n2*(4*OH*OW) + oc2*OH*OW + y3*OW + x3
                        dp._data[dst] = dflat._data[src]
                        x3 += 1
                    y3 += 1
                oc2 += 1
            n2 += 1

        # MaxPool backward
        var da = pool.backward(a, dp)
        # ReLU backward
        var dh = relu_backward(h, da)
        # Conv backward
        var grads = conv.backward(x, dh)
        var dW = grads[0]; var db = grads[1]

        # Update
        opt.step_linear(fc, dW2, db2)
        conv.weight = conv.weight - dW * opt.lr
        conv.bias   = conv.bias   - db * opt.lr

        print("epoch=", ep, " loss=", loss)
        ep += 1

    var probs = softmax(fc.forward(flat))
    print("probs shape:", probs.shape())
