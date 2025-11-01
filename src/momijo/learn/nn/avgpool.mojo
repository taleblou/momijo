# MIT License
# SPDX-License-Identifier: MIT
# Project: momijo
# File: src/momijo/learn/nn/avgpool.mojo
# Description: AvgPool2d (NCHW) forward + backward.

from momijo.tensor import tensor

struct AvgPool2d:
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
                        var acc = 0.0; var cnt = 0.0
                        var ky = 0
                        while ky < self.kernel_h:
                            var kx = 0
                            while kx < self.kernel_w:
                                var iy = oy * self.stride_h - self.pad_h + ky
                                var ix = ox * self.stride_w - self.pad_w + kx
                                if (iy >= 0 and iy < H and ix >= 0 and ix < W):
                                    var idx = n*C*H*W + c*H*W + iy*W + ix
                                    acc = acc + x._data[idx]; cnt = cnt + 1.0
                                kx += 1
                            ky += 1
                        y._data[n*C*OH*OW + c*OH*OW + oy*OW + ox] = acc / cnt
                        ox += 1
                    oy += 1
                c += 1
            n += 1
        return y

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
                        var cnt = 0.0
                        var ky = 0
                        while ky < self.kernel_h:
                            var kx = 0
                            while kx < self.kernel_w:
                                var iy = oy * self.stride_h - self.pad_h + ky
                                var ix = ox * self.stride_w - self.pad_w + kx
                                if (iy >= 0 and iy < H and ix >= 0 and ix < W):
                                    cnt = cnt + 1.0
                                kx += 1
                            ky += 1
                        var gy = grad_y._data[n*C*OH*OW + c*OH*OW + oy*OW + ox] / cnt
                        ky = 0
                        while ky < self.kernel_h:
                            var kx2 = 0
                            while kx2 < self.kernel_w:
                                var iy2 = oy * self.stride_h - self.pad_h + ky
                                var ix2 = ox * self.stride_w - self.pad_w + kx2
                                if (iy2 >= 0 and iy2 < H and ix2 >= 0 and ix2 < W):
                                    var di = n*C*H*W + c*H*W + iy2*W + ix2
                                    dx._data[di] = dx._data[di] + gy
                                kx2 += 1
                            ky += 1
                        ox += 1
                    oy += 1
                c += 1
            n += 1
        return dx
