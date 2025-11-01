# MIT License
# SPDX-License-Identifier: MIT
# Project: momijo
# File: src/momijo/learn/nn/conv_transpose.mojo
# Description: ConvTranspose2d (deconvolution) forward + backward, naive loops (NCHW).

from momijo.tensor import tensor

struct ConvTranspose2d:
    var in_channels: Int
    var out_channels: Int
    var kernel_h: Int
    var kernel_w: Int
    var stride_h: Int
    var stride_w: Int
    var pad_h: Int
    var pad_w: Int
    var weight: tensor.Tensor[Float64]  # [in, out, kh, kw] (note reversed order vs Conv2d)
    var bias: tensor.Tensor[Float64]    # [out]

    fn __init__(out self, in_channels: Int, out_channels: Int, kernel_size: Tuple[Int, Int], stride: Tuple[Int, Int] = (1,1), padding: Tuple[Int, Int] = (0,0)):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_h = kernel_size[0]; self.kernel_w = kernel_size[1]
        self.stride_h = stride[0]; self.stride_w = stride[1]
        self.pad_h = padding[0]; self.pad_w = padding[1]
        var w = tensor.randn([in_channels, out_channels, self.kernel_h, self.kernel_w])
        var scale = (2.0 / Float64(out_channels * self.kernel_h * self.kernel_w)) ** 0.5
        self.weight = w * scale
        self.bias = tensor.zeros([out_channels])

    fn __copyinit__(out self, other: Self):
        self.in_channels = other.in_channels
        self.out_channels = other.out_channels
        self.kernel_h = other.kernel_h; self.kernel_w = other.kernel_w
        self.stride_h = other.stride_h; self.stride_w = other.stride_w
        self.pad_h = other.pad_h; self.pad_w = other.pad_w
        self.weight = other.weight; self.bias = other.bias

    fn out_shape(self, H: Int, W: Int) -> (Int, Int):
        var OH = (H - 1) * self.stride_h - 2*self.pad_h + self.kernel_h
        var OW = (W - 1) * self.stride_w - 2*self.pad_w + self.kernel_w
        return (OH, OW)

    fn forward(self, x: tensor.Tensor[Float64]) -> tensor.Tensor[Float64]:
        var N = x.shape()[0]; var C = x.shape()[1]; var H = x.shape()[2]; var W = x.shape()[3]
        var s = self.out_shape(H, W)
        var OH = s[0]; var OW = s[1]
        var y = tensor.zeros([N, self.out_channels, OH, OW])
        var n = 0
        while n < N:
            var ic = 0
            while ic < C:
                var oy = 0
                while oy < OH:
                    var ox = 0
                    while ox < OW:
                        # find contributions from x at positions that map to (oy,ox)
                        var sum = 0.0
                        var ky = 0
                        while ky < self.kernel_h:
                            var kx = 0
                            while kx < self.kernel_w:
                                # reverse mapping
                                var iy_num = oy + self.pad_h - ky
                                var ix_num = ox + self.pad_w - kx
                                if iy_num % self.stride_h == 0 and ix_num % self.stride_w == 0:
                                    var iy = iy_num // self.stride_h
                                    var ix = ix_num // self.stride_w
                                    if (iy >= 0 and iy < H and ix >= 0 and ix < W):
                                        var oc = 0
                                        while oc < self.out_channels:
                                            var xi = n*C*H*W + ic*H*W + iy*W + ix
                                            var wi = ic*self.out_channels*self.kernel_h*self.kernel_w + oc*self.kernel_h*self.kernel_w + ky*self.kernel_w + kx
                                            sum = sum + x._data[xi] * self.weight._data[wi]
                                            oc += 1
                                kx += 1
                            ky += 1
                        # bias add distributed by out_channel loop separately
                        var oc2 = 0
                        while oc2 < self.out_channels:
                            var yi = n*self.out_channels*OH*OW + oc2*OH*OW + oy*OW + ox
                            y._data[yi] = y._data[yi] + sum + self.bias._data[oc2]
                            oc2 += 1
                        ox += 1
                    oy += 1
                ic += 1
            n += 1
        return y

    fn backward(self, x: tensor.Tensor[Float64], grad_y: tensor.Tensor[Float64]) -> (tensor.Tensor[Float64], tensor.Tensor[Float64], tensor.Tensor[Float64]):
        var N = x.shape()[0]; var C = x.shape()[1]; var H = x.shape()[2]; var W = x.shape()[3]
        var OH = grad_y.shape()[2]; var OW = grad_y.shape()[3]
        var dW = tensor.zeros_like(self.weight)
        var db = tensor.zeros_like(self.bias)
        var dx = tensor.zeros_like(x)

        var n = 0
        while n < N:
            var ic = 0
            while ic < C:
                var oy = 0
                while oy < OH:
                    var ox = 0
                    while ox < OW:
                        var oc = 0
                        while oc < self.out_channels:
                            var gy = grad_y._data[n*self.out_channels*OH*OW + oc*OH*OW + oy*OW + ox]
                            db._data[oc] = db._data[oc] + gy
                            # Map back to x
                            var ky = 0
                            while ky < self.kernel_h:
                                var kx = 0
                                while kx < self.kernel_w:
                                    var iy_num = oy + self.pad_h - ky
                                    var ix_num = ox + self.pad_w - kx
                                    if iy_num % self.stride_h == 0 and ix_num % self.stride_w == 0:
                                        var iy = iy_num // self.stride_h
                                        var ix = ix_num // self.stride_w
                                        if (iy >= 0 and iy < H and ix >= 0 and ix < W):
                                            var xi = n*C*H*W + ic*H*W + iy*W + ix
                                            var wi = ic*self.out_channels*self.kernel_h*self.kernel_w + oc*self.kernel_h*self.kernel_w + ky*self.kernel_w + kx
                                            dW._data[wi] = dW._data[wi] + x._data[xi] * gy
                                            dx._data[xi] = dx._data[xi] + self.weight._data[wi] * gy
                                    kx += 1
                                ky += 1
                            oc += 1
                        ox += 1
                    oy += 1
                ic += 1
            n += 1
        return (dW, db, dx)
