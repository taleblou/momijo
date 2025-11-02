# MIT License
# SPDX-License-Identifier: MIT
# Project: momijo
# File: src/momijo/learn/nn/conv.mojo
# Description: Naive Conv2d and MaxPool2d (NCHW) with forward+backward.

from momijo.tensor import tensor

struct Conv2d(Copyable, Movable):
    var in_channels: Int
    var out_channels: Int
    var kernel_h: Int
    var kernel_w: Int
    var stride_h: Int
    var stride_w: Int
    var pad_h: Int
    var pad_w: Int
    var weight: tensor.Tensor[Float64]  # [out, in, kh, kw]
    var bias: tensor.Tensor[Float64]    # [out]

    fn __init__(out self, in_channels: Int, out_channels: Int, kernel_size: Tuple[Int, Int], stride: Tuple[Int, Int] = (1,1), padding: Tuple[Int, Int] = (0,0)):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_h = kernel_size[0]; self.kernel_w = kernel_size[1]
        self.stride_h = stride[0]; self.stride_w = stride[1]
        self.pad_h = padding[0]; self.pad_w = padding[1]
        var w = tensor.randn([out_channels, in_channels, self.kernel_h, self.kernel_w])
        var scale = (2.0 / Float64(in_channels * self.kernel_h * self.kernel_w)) ** 0.5
        self.weight = w * scale
        self.bias = tensor.zeros([out_channels])

    fn __copyinit__(out self, other: Self):
        self.in_channels = other.in_channels
        self.out_channels = other.out_channels
        self.kernel_h = other.kernel_h; self.kernel_w = other.kernel_w
        self.stride_h = other.stride_h; self.stride_w = other.stride_w
        self.pad_h = other.pad_h; self.pad_w = other.pad_w
        self.weight = other.weight.copy(); self.bias = other.bias.copy()

    fn forward(self, x: tensor.Tensor[Float64]) -> tensor.Tensor[Float64]:
        var N = x.shape()[0]; var C = x.shape()[1]; var H = x.shape()[2]; var W = x.shape()[3]
        var OH = (H + 2*self.pad_h - self.kernel_h) // self.stride_h + 1
        var OW = (W + 2*self.pad_w - self.kernel_w) // self.stride_w + 1
        var y = tensor.zeros([N, self.out_channels, OH, OW])
        var n = 0
        while n < N:
            var oc = 0
            while oc < self.out_channels:
                var oy = 0
                while oy < OH:
                    var ox = 0
                    while ox < OW:
                        var sum = 0.0
                        var ic = 0
                        while ic < C:
                            var ky = 0
                            while ky < self.kernel_h:
                                var kx = 0
                                while kx < self.kernel_w:
                                    var iy = oy * self.stride_h - self.pad_h + ky
                                    var ix = ox * self.stride_w - self.pad_w + kx
                                    if (iy >= 0 and iy < H and ix >= 0 and ix < W):
                                        var xi = n*C*H*W + ic*H*W + iy*W + ix
                                        var wi = oc*C*self.kernel_h*self.kernel_w + ic*self.kernel_h*self.kernel_w + ky*self.kernel_w + kx
                                        sum = sum + x._data[xi] * self.weight._data[wi]
                                    kx += 1
                                ky += 1
                            ic += 1
                        y._data[n*self.out_channels*OH*OW + oc*OH*OW + oy*OW + ox] = sum + self.bias._data[oc]
                        ox += 1
                    oy += 1
                oc += 1
            n += 1
        return y.copy()


    fn forward(self, x:tensor.GradTensor) -> tensor.GradTensor:
        var N = x.shape()[0]; var C = x.shape()[1]; var H = x.shape()[2]; var W = x.shape()[3]
        var OH = (H + 2*self.pad_h - self.kernel_h) // self.stride_h + 1
        var OW = (W + 2*self.pad_w - self.kernel_w) // self.stride_w + 1
        var y = tensor.zeros([N, self.out_channels, OH, OW])
        var n = 0
        while n < N:
            var oc = 0
            while oc < self.out_channels:
                var oy = 0
                while oy < OH:
                    var ox = 0
                    while ox < OW:
                        var sum = 0.0
                        var ic = 0
                        while ic < C:
                            var ky = 0
                            while ky < self.kernel_h:
                                var kx = 0
                                while kx < self.kernel_w:
                                    var iy = oy * self.stride_h - self.pad_h + ky
                                    var ix = ox * self.stride_w - self.pad_w + kx
                                    if (iy >= 0 and iy < H and ix >= 0 and ix < W):
                                        var xi = n*C*H*W + ic*H*W + iy*W + ix
                                        var wi = oc*C*self.kernel_h*self.kernel_w + ic*self.kernel_h*self.kernel_w + ky*self.kernel_w + kx
                                        sum = sum + x._data[xi] * self.weight._data[wi]
                                    kx += 1
                                ky += 1
                            ic += 1
                        y._data[n*self.out_channels*OH*OW + oc*OH*OW + oy*OW + ox] = sum + self.bias._data[oc]
                        ox += 1
                    oy += 1
                oc += 1
            n += 1
        return y.copy()

    fn backward(self, x: tensor.Tensor[Float64], grad_y: tensor.Tensor[Float64]) -> (tensor.Tensor[Float64], tensor.Tensor[Float64], tensor.Tensor[Float64]):
        var N = x.shape()[0]; var C = x.shape()[1]; var H = x.shape()[2]; var W = x.shape()[3]
        var OH = grad_y.shape()[2]; var OW = grad_y.shape()[3]
        var dW = tensor.zeros_like(self.weight)
        var db = tensor.zeros_like(self.bias)
        var dx = tensor.zeros_like(x)
        var n = 0
        while n < N:
            var oc = 0
            while oc < self.out_channels:
                var oy = 0
                while oy < OH:
                    var ox = 0
                    while ox < OW:
                        var gy = grad_y._data[n*self.out_channels*OH*OW + oc*OH*OW + oy*OW + ox]
                        db._data[oc] = db._data[oc] + gy
                        var ic = 0
                        while ic < C:
                            var ky = 0
                            while ky < self.kernel_h:
                                var kx = 0
                                while kx < self.kernel_w:
                                    var iy = oy * self.stride_h - self.pad_h + ky
                                    var ix = ox * self.stride_w - self.pad_w + kx
                                    if (iy >= 0 and iy < H and ix >= 0 and ix < W):
                                        var xi = n*C*H*W + ic*H*W + iy*W + ix
                                        var wi = oc*C*self.kernel_h*self.kernel_w + ic*self.kernel_h*self.kernel_w + ky*self.kernel_w + kx
                                        dW._data[wi] = dW._data[wi] + x._data[xi] * gy
                                        dx._data[xi] = dx._data[xi] + self.weight._data[wi] * gy
                                    kx += 1
                                ky += 1
                            ic += 1
                        ox += 1
                    oy += 1
                oc += 1
            n += 1
        return (dW, db, dx)
        
    fn backward(self, x: tensor.GradTensor, grad_y: tensor.GradTensor) -> (tensor.GradTensor, tensor.GradTensor, tensor.GradTensor):
        var N = x.shape()[0]; var C = x.shape()[1]; var H = x.shape()[2]; var W = x.shape()[3]
        var OH = grad_y.shape()[2]; var OW = grad_y.shape()[3]
        var dW = tensor.zeros_like(self.weight)
        var db = tensor.zeros_like(self.bias)
        var dx = tensor.zeros_like(x)
        var n = 0
        while n < N:
            var oc = 0
            while oc < self.out_channels:
                var oy = 0
                while oy < OH:
                    var ox = 0
                    while ox < OW:
                        var gy = grad_y._data[n*self.out_channels*OH*OW + oc*OH*OW + oy*OW + ox]
                        db._data[oc] = db._data[oc] + gy
                        var ic = 0
                        while ic < C:
                            var ky = 0
                            while ky < self.kernel_h:
                                var kx = 0
                                while kx < self.kernel_w:
                                    var iy = oy * self.stride_h - self.pad_h + ky
                                    var ix = ox * self.stride_w - self.pad_w + kx
                                    if (iy >= 0 and iy < H and ix >= 0 and ix < W):
                                        var xi = n*C*H*W + ic*H*W + iy*W + ix
                                        var wi = oc*C*self.kernel_h*self.kernel_w + ic*self.kernel_h*self.kernel_w + ky*self.kernel_w + kx
                                        dW._data[wi] = dW._data[wi] + x._data[xi] * gy
                                        dx._data[xi] = dx._data[xi] + self.weight._data[wi] * gy
                                    kx += 1
                                ky += 1
                            ic += 1
                        ox += 1
                    oy += 1
                oc += 1
            n += 1
        return (dW, db, dx)

struct MaxPool2d(Copyable, Movable):
    var kernel_h: Int
    var kernel_w: Int
    var stride_h: Int
    var stride_w: Int
    var pad_h: Int
    var pad_w: Int

    fn __init__(out self, kernel_size: Tuple[Int, Int], stride: Tuple[Int, Int] = (0,0), padding: Tuple[Int, Int] = (0,0)):
        self.kernel_h = kernel_size[0]; self.kernel_w = kernel_size[1]
        self.stride_h = stride[0] if stride[0] > 0 else self.kernel_h
        self.stride_w = stride[1] if stride[1] > 0 else self.kernel_w
        self.pad_h = padding[0]; self.pad_w = padding[1]

    fn __copyinit__(out self, other: Self):
        self.kernel_h = other.kernel_h; self.kernel_w = other.kernel_w
        self.stride_h = other.stride_h; self.stride_w = other.stride_w
        self.pad_h = other.pad_h; self.pad_w = other.pad_w

    fn forward(self, x: tensor.Tensor[Float64]) -> tensor.Tensor[Float64]:
        var N = x.shape()[0]; var C = x.shape()[1]; var H = x.shape()[2]; var W = x.shape()[3]
        var OH = (H + 2*self.pad_h - self.kernel_h) // self.stride_h + 1
        var OW = (W + 2*self.pad_w - self.kernel_w) // self.stride_w + 1
        var y = tensor.zeros([N, C, OH, OW])
        var n = 0
        while n < N:
            var c = 0
            while c < C:
                var oy = 0
                while oy < OH:
                    var ox = 0
                    while ox < OW:
                        var m = -1e300
                        var ky = 0
                        while ky < self.kernel_h:
                            var kx = 0
                            while kx < self.kernel_w:
                                var iy = oy * self.stride_h - self.pad_h + ky
                                var ix = ox * self.stride_w - self.pad_w + kx
                                if (iy >= 0 and iy < H and ix >= 0 and ix < W):
                                    var idx = n*C*H*W + c*H*W + iy*W + ix
                                    var v = x._data[idx]
                                    if v > m: m = v
                                kx += 1
                            ky += 1
                        y._data[n*C*OH*OW + c*OH*OW + oy*OW + ox] = m
                        ox += 1
                    oy += 1
                c += 1
            n += 1
        return y.copy()


    fn forward(self, x: tensor.GradTensor) -> tensor.GradTensor:
        var N = x.shape()[0]; var C = x.shape()[1]; var H = x.shape()[2]; var W = x.shape()[3]
        var OH = (H + 2*self.pad_h - self.kernel_h) // self.stride_h + 1
        var OW = (W + 2*self.pad_w - self.kernel_w) // self.stride_w + 1
        var y = tensor.zeros([N, C, OH, OW])
        var n = 0
        while n < N:
            var c = 0
            while c < C:
                var oy = 0
                while oy < OH:
                    var ox = 0
                    while ox < OW:
                        var m = -1e300
                        var ky = 0
                        while ky < self.kernel_h:
                            var kx = 0
                            while kx < self.kernel_w:
                                var iy = oy * self.stride_h - self.pad_h + ky
                                var ix = ox * self.stride_w - self.pad_w + kx
                                if (iy >= 0 and iy < H and ix >= 0 and ix < W):
                                    var idx = n*C*H*W + c*H*W + iy*W + ix
                                    var v = x._data[idx]
                                    if v > m: m = v
                                kx += 1
                            ky += 1
                        y._data[n*C*OH*OW + c*OH*OW + oy*OW + ox] = m
                        ox += 1
                    oy += 1
                c += 1
            n += 1
        return y.copy()
    fn backward(self, x: tensor.Tensor[Float64], grad_y: tensor.Tensor[Float64]) -> tensor.Tensor[Float64]:
        var N = x.shape()[0]; var C = x.shape()[1]; var H = x.shape()[2]; var W = x.shape()[3]
        var OH = grad_y.shape()[2]; var OW = grad_y.shape()[3]
        var dx = tensor.zeros_like(x)
        var n = 0
        while n < N:
            var c = 0
            while c < C:
                var oy = 0
                while oy < OH:
                    var ox = 0
                    while ox < OW:
                        var m = -1e300; var my = -1; var mx = -1
                        var ky = 0
                        while ky < self.kernel_h:
                            var kx = 0
                            while kx < self.kernel_w:
                                var iy = oy * self.stride_h - self.pad_h + ky
                                var ix = ox * self.stride_w - self.pad_w + kx
                                if (iy >= 0 and iy < H and ix >= 0 and ix < W):
                                    var idx = n*C*H*W + c*H*W + iy*W + ix
                                    var v = x._data[idx]
                                    if v > m: m = v; my = iy; mx = ix
                                kx += 1
                            ky += 1
                        var gy = grad_y._data[n*C*OH*OW + c*OH*OW + oy*OW + ox]
                        if my >= 0 and mx >= 0:
                            var di = n*C*H*W + c*H*W + my*W + mx
                            dx._data[di] = dx._data[di] + gy
                        ox += 1
                    oy += 1
                c += 1
            n += 1
        return dx.copy()
   fn backward(self, x: tensor.GradTensor, grad_y: tensor.GradTensor) ->tensor.GradTensor:
        var N = x.shape()[0]; var C = x.shape()[1]; var H = x.shape()[2]; var W = x.shape()[3]
        var OH = grad_y.shape()[2]; var OW = grad_y.shape()[3]
        var dx = tensor.zeros_like(x)
        var n = 0
        while n < N:
            var c = 0
            while c < C:
                var oy = 0
                while oy < OH:
                    var ox = 0
                    while ox < OW:
                        var m = -1e300; var my = -1; var mx = -1
                        var ky = 0
                        while ky < self.kernel_h:
                            var kx = 0
                            while kx < self.kernel_w:
                                var iy = oy * self.stride_h - self.pad_h + ky
                                var ix = ox * self.stride_w - self.pad_w + kx
                                if (iy >= 0 and iy < H and ix >= 0 and ix < W):
                                    var idx = n*C*H*W + c*H*W + iy*W + ix
                                    var v = x._data[idx]
                                    if v > m: m = v; my = iy; mx = ix
                                kx += 1
                            ky += 1
                        var gy = grad_y._data[n*C*OH*OW + c*OH*OW + oy*OW + ox]
                        if my >= 0 and mx >= 0:
                            var di = n*C*H*W + c*H*W + my*W + mx
                            dx._data[di] = dx._data[di] + gy
                        ox += 1
                    oy += 1
                c += 1
            n += 1
        return dx.copy()
