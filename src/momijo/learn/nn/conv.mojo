# MIT License
# SPDX-License-Identifier: MIT
# Project: momijo
# File: src/momijo/learn/nn/conv.mojo
# Description: Naive Conv2d and MaxPool2d (NCHW) with forward+backward.

from momijo.tensor import tensor
from collections.list import List
from momijo.tensor.gpu.runtime import (
    gpu_available,
    memset_f32,
    atomic_add_f32,
    # 1D launchers:
    launch_1d_maxpool_fw,
    launch_1d_maxpool_bw,
    launch_1d_conv2d_fw,
    # kernel types for signature checking
    Kernel1D_MaxPoolFW,
    Kernel1D_MaxPoolBW,
    Kernel1D_Conv2DFW,
)

from math import sqrt


# =====================
# MaxPool2d
# =====================
struct MaxPool2d(Copyable, Movable):
    fn __init__(
        out self,
        kernel: (Int, Int),
        stride: (Int, Int) = (1, 1),
        pad: (Int, Int) = (0, 0)
    ):
        self.kernel_h = kernel[0]; self.kernel_w = kernel[1]
        self.stride_h = stride[0]; self.stride_w = stride[1]
        self.pad_h    = pad[0];    self.pad_w    = pad[1]

    var kernel_h: Int
    var kernel_w: Int
    var stride_h: Int
    var stride_w: Int
    var pad_h:    Int
    var pad_w:    Int

    @always_inline
    fn _has_gpu(self) -> Bool:
        return gpu_available()

    @staticmethod
    fn _f32_min() -> Float32:
        return -3.4028235e38

    @staticmethod
    fn _unflatten_ncyx_fw(tid: Int, C: Int, OH: Int, OW: Int) -> (Int, Int, Int, Int):
        var c_oh_ow = C * OH * OW
        var n = tid // c_oh_ow
        var r = tid - n * c_oh_ow
        var c = r // (OH * OW)
        r = r - c * (OH * OW)
        var oy = r // OW
        var ox = r - oy * OW
        return (n, c, oy, ox)

    @staticmethod
    fn _unflatten_ncyx_bw(tid: Int, C: Int, OH: Int, OW: Int) -> (Int, Int, Int, Int):
        var c_oh_ow = C * OH * OW
        var n = tid // c_oh_ow
        var r = tid - n * c_oh_ow
        var c = r // (OH * OW)
        r = r - c * (OH * OW)
        var oy = r // OW
        var ox = r - oy * OW
        return (n, c, oy, ox)

    # -------- MaxPool Forward (CPU) --------
    fn forward_cpu_parallel(self, x: tensor.Tensor[Float32]) -> tensor.Tensor[Float32]:
        var N = x.shape()[0]; var C = x.shape()[1]; var H = x.shape()[2]; var W = x.shape()[3]
        var OH = (H + 2*self.pad_h - self.kernel_h) // self.stride_h + 1
        var OW = (W + 2*self.pad_w - self.kernel_w) // self.stride_w + 1
        var y = tensor.zeros([N, C, OH, OW])

        var HW    = H * W
        var CHW   = C * HW
        var OH_OW = OH * OW

        var n = 0
        while n < N:
            var c = 0
            while c < C:
                var x_base  = n * CHW + c * HW
                var y_base  = n * (C * OH_OW) + c * OH_OW

                var oy = 0
                while oy < OH:
                    var iy0 = oy * self.stride_h - self.pad_h
                    var ky_start = 0
                    if iy0 < 0:
                        ky_start = -iy0
                    var ky_end = self.kernel_h
                    var last_y = iy0 + self.kernel_h
                    if last_y > H:
                        ky_end = ky_end - (last_y - H)

                    var ox = 0
                    while ox < OW:
                        var ix0 = ox * self.stride_w - self.pad_w
                        var kx_start = 0
                        if ix0 < 0:
                            kx_start = -ix0
                        var kx_end = self.kernel_w
                        var last_x = ix0 + self.kernel_w
                        if last_x > W:
                            kx_end = kx_end - (last_x - W)

                        var best: Float32 = Self._f32_min()

                        var ky = ky_start
                        while ky < ky_end:
                            var iy = iy0 + ky
                            var row = x_base + iy * W
                            var kx = kx_start

                            # unroll 8
                            while kx + 7 < kx_end:
                                var ix = ix0 + kx
                                var v0: Float32 = x._data[row + ix + 0]
                                var v1: Float32 = x._data[row + ix + 1]
                                var v2: Float32 = x._data[row + ix + 2]
                                var v3: Float32 = x._data[row + ix + 3]
                                var v4: Float32 = x._data[row + ix + 4]
                                var v5: Float32 = x._data[row + ix + 5]
                                var v6: Float32 = x._data[row + ix + 6]
                                var v7: Float32 = x._data[row + ix + 7]
                                if v0 > best: best = v0
                                if v1 > best: best = v1
                                if v2 > best: best = v2
                                if v3 > best: best = v3
                                if v4 > best: best = v4
                                if v5 > best: best = v5
                                if v6 > best: best = v6
                                if v7 > best: best = v7
                                kx = kx + 8

                            # unroll 4
                            while kx + 3 < kx_end:
                                var ix4 = ix0 + kx
                                var u0: Float32 = x._data[row + ix4 + 0]
                                var u1: Float32 = x._data[row + ix4 + 1]
                                var u2: Float32 = x._data[row + ix4 + 2]
                                var u3: Float32 = x._data[row + ix4 + 3]
                                if u0 > best: best = u0
                                if u1 > best: best = u1
                                if u2 > best: best = u2
                                if u3 > best: best = u3
                                kx = kx + 4

                            while kx < kx_end:
                                var ix1 = ix0 + kx
                                var vv: Float32 = x._data[row + ix1]
                                if vv > best: best = vv
                                kx = kx + 1

                            ky = ky + 1

                        var yi = y_base + oy * OW + ox
                        y._data[yi] = best

                        ox = ox + 1
                    oy = oy + 1
                c = c + 1
            n = n + 1

        return y.copy()

    # GPU kernel uses List buffers
    @staticmethod
    fn _kernel_maxpool2d_fwd(
        tid: Int,
        x: List[Float32],
        mut y: List[Float32],
        N: Int, C: Int, H: Int, W: Int,
        OH: Int, OW: Int,
        kH: Int, kW: Int,
        sH: Int, sW: Int,
        pH: Int, pW: Int
    )  -> None:
        var total = N * C * OH * OW
        if tid >= total:
            return
        var ncyx = Self._unflatten_ncyx_fw(tid, C, OH, OW)
        var n = ncyx[0]; var c = ncyx[1]; var oy = ncyx[2]; var ox = ncyx[3]

        var HW  = H * W
        var CHW = C * HW
        var x_base = n * CHW + c * HW

        var iy0 = oy * sH - pH
        var ky_start = 0
        if iy0 < 0:
            ky_start = -iy0
        var ky_end = kH
        var last_y = iy0 + kH
        if last_y > H:
            ky_end = ky_end - (last_y - H)

        var ix0 = ox * sW - pW
        var kx_start = 0
        if ix0 < 0:
            kx_start = -ix0
        var kx_end = kW
        var last_x = ix0 + kW
        if last_x > W:
            kx_end = kx_end - (last_x - W)

        var best: Float32 = Self._f32_min()

        var ky = ky_start
        while ky < ky_end:
            var iy = iy0 + ky
            var row = x_base + iy * W
            var kx = kx_start
            while kx < kx_end:
                var ix = ix0 + kx
                var v: Float32 = x[row + ix]
                if v > best:
                    best = v
                kx = kx + 1
            ky = ky + 1

        var yi = (((n * C) + c) * OH + oy) * OW + ox
        y[yi] = best

    fn forward_gpu(self, x: tensor.Tensor[Float32]) -> tensor.Tensor[Float32]:
        var N = x.shape()[0]; var C = x.shape()[1]; var H = x.shape()[2]; var W = x.shape()[3]
        var OH = (H + 2*self.pad_h - self.kernel_h) // self.stride_h + 1
        var OW = (W + 2*self.pad_w - self.kernel_w) // self.stride_w + 1

        var y = tensor.zeros([N, C, OH, OW])

        # بافر محلیِ قابل‌تغییر برای خروجی
        var ybuf = y._data.copy()

        var total_threads = N * C * OH * OW
        var block = 256
        if block > total_threads:
            block = total_threads

        launch_1d_maxpool_fw(
            total_threads, block, Self._kernel_maxpool2d_fwd,
            x._data, ybuf,
            N, C, H, W, OH, OW,
            self.kernel_h, self.kernel_w,
            self.stride_h, self.stride_w,
            self.pad_h, self.pad_w
        )

        # برگرداندن نتیجه به تنسور
        y._data = ybuf.copy()
        return y.copy()

    fn forward_auto(self, x: tensor.Tensor[Float32]) -> tensor.Tensor[Float32]:
        if self._has_gpu():
            return self.forward_gpu(x)
        return self.forward_cpu_parallel(x)

    fn forward(self, x: tensor.Tensor[Float32]) -> tensor.Tensor[Float32]:
        return self.forward_auto(x)

    # -------- MaxPool Backward --------
    fn backward_cpu_parallel(self, x: tensor.Tensor[Float32], grad_y: tensor.Tensor[Float32]) -> tensor.Tensor[Float32]:
        var N = x.shape()[0]; var C = x.shape()[1]; var H = x.shape()[2]; var W = x.shape()[3]
        var OH = grad_y.shape()[2]; var OW = grad_y.shape()[3]

        var dx = tensor.zeros_like(x)

        var HW    = H * W
        var CHW   = C * HW
        var OH_OW = OH * OW

        var n = 0
        while n < N:
            var c = 0
            while c < C:
                var x_base  = n * CHW + c * HW
                var dx_base = x_base
                var gy_base = n * (C * OH_OW) + c * OH_OW

                var oy = 0
                while oy < OH:
                    var base_iy = oy * self.stride_h - self.pad_h
                    var ky_start = 0
                    if base_iy < 0:
                        ky_start = -base_iy
                    var ky_end = self.kernel_h
                    var last_y = base_iy + self.kernel_h
                    if last_y > H:
                        ky_end = ky_end - (last_y - H)

                    var ox = 0
                    while ox < OW:
                        var base_ix = ox * self.stride_w - self.pad_w
                        var kx_start = 0
                        if base_ix < 0:
                            kx_start = -base_ix
                        var kx_end = self.kernel_w
                        var last_x = base_ix + self.kernel_w
                        if last_x > W:
                            kx_end = kx_end - (last_x - W)

                        var m: Float32 = Self._f32_min()
                        var my: Int = -1
                        var mx: Int = -1

                        var ky = ky_start
                        while ky < ky_end:
                            var iy = base_iy + ky
                            var row = x_base + iy * W
                            var kx = kx_start
                            while kx < kx_end:
                                var ix = base_ix + kx
                                var v: Float32 = x._data[row + ix]
                                if v > m:
                                    m = v; my = iy; mx = ix
                                kx = kx + 1
                            ky = ky + 1

                        var gy_val: Float32 = grad_y._data[gy_base + oy * OW + ox]
                        if my >= 0 and mx >= 0:
                            var di = dx_base + my * W + mx
                            dx._data[di] = dx._data[di] + gy_val

                        ox = ox + 1
                    oy = oy + 1
                c = c + 1
            n = n + 1

        return dx.copy()

    @staticmethod
    fn _kernel_maxpool2d_bw(
        tid: Int,
        x: List[Float32],
        gy: List[Float32],
        mut dx: List[Float32],
        N: Int, C: Int, H: Int, W: Int,
        OH: Int, OW: Int,
        kH: Int, kW: Int,
        sH: Int, sW: Int,
        pH: Int, pW: Int
    )  -> None:
        var total = N * C * OH * OW
        if tid >= total:
            return
        var ncyx = Self._unflatten_ncyx_bw(tid, C, OH, OW)
        var n = ncyx[0]; var c = ncyx[1]; var oy = ncyx[2]; var ox = ncyx[3]

        var HW    = H * W
        var CHW   = C * HW
        var OH_OW = OH * OW

        var x_base  = n * CHW + c * HW
        var dx_base = x_base
        var gy_base = n * (C * OH_OW) + c * OH_OW

        var base_iy = oy * sH - pH
        var ky_start = 0
        if base_iy < 0:
            ky_start = -base_iy
        var ky_end = kH
        var last_y = base_iy + kH
        if last_y > H:
            ky_end = ky_end - (last_y - H)

        var base_ix = ox * sW - pW
        var kx_start = 0
        if base_ix < 0:
            kx_start = -base_ix
        var kx_end = kW
        var last_x = base_ix + kW
        if last_x > W:
            kx_end = kx_end - (last_x - W)

        var m: Float32 = Self._f32_min()
        var my: Int = -1
        var mx: Int = -1

        var ky = ky_start
        while ky < ky_end:
            var iy = base_iy + ky
            var row = x_base + iy * W
            var kx = kx_start
            while kx < kx_end:
                var ix = base_ix + kx
                var v: Float32 = x[row + ix]
                if v > m:
                    m = v; my = iy; mx = ix
                kx = kx + 1
            ky = ky + 1

        var gy_val: Float32 = gy[gy_base + oy * OW + ox]
        if my >= 0 and mx >= 0:
            var di = dx_base + my * W + mx
            dx[di] = dx[di] + gy_val

    fn backward_gpu(self, x: tensor.Tensor[Float32], grad_y: tensor.Tensor[Float32]) -> tensor.Tensor[Float32]:
        var N = x.shape()[0]; var C = x.shape()[1]; var H = x.shape()[2]; var W = x.shape()[3]
        var OH = grad_y.shape()[2]; var OW = grad_y.shape()[3]
        var dx = tensor.zeros_like(x)

        var dxbuf = dx._data.copy()

        var total_threads = N * C * OH * OW
        var block = 256
        if block > total_threads:
            block = total_threads

        launch_1d_maxpool_bw(
            total_threads, block, Self._kernel_maxpool2d_bw,
            x._data, grad_y._data, dxbuf,
            N, C, H, W, OH, OW,
            self.kernel_h, self.kernel_w,
            self.stride_h, self.stride_w,
            self.pad_h, self.pad_w
        )

        dx._data = dxbuf.copy()
        return dx.copy()

    fn backward_auto(self, x: tensor.Tensor[Float32], grad_y: tensor.Tensor[Float32]) -> tensor.Tensor[Float32]:
        if self._has_gpu():
            return self.backward_gpu(x, grad_y)
        return self.backward_cpu_parallel(x, grad_y)

    fn backward(self, x: tensor.Tensor[Float32], grad_y: tensor.Tensor[Float32]) -> tensor.Tensor[Float32]:
        return self.backward_auto(x, grad_y)


# =====================
# Conv2d
# =====================
struct Conv2d(Copyable, Movable):
    fn __init__(
        out self,
        in_channels: Int,
        out_channels: Int,
        kernel: (Int, Int),
        stride: (Int, Int) = (1, 1),
        pad: (Int, Int) = (0, 0),
        bias: Bool = True
    ):
        self.kernel_h = kernel[0]; self.kernel_w = kernel[1]
        self.stride_h = stride[0]; self.stride_w = stride[1]
        self.pad_h    = pad[0];    self.pad_w    = pad[1]
        self.out_channels = out_channels
        self.bias = bias
        var kH = self.kernel_h; var kW = self.kernel_w
        var fan_in = in_channels * kH * kW
        self.weight = tensor.randn([out_channels, in_channels, kH, kW]) * Float32(sqrt(2.0 / Float32(fan_in)))
        if bias:
            self.bias_t = tensor.zeros([out_channels])
        else:
            self.bias_t = tensor.zeros([0])

    var kernel_h: Int
    var kernel_w: Int
    var stride_h: Int
    var stride_w: Int
    var pad_h:    Int
    var pad_w:    Int
    var out_channels: Int

    var weight: tensor.Tensor[Float32]   # [OC, C, kH, kW]
    var bias:   Bool
    var bias_t: tensor.Tensor[Float32]   # [OC] if bias==True else []

    @always_inline
    fn _has_gpu(self) -> Bool:
        return gpu_available()

    @staticmethod
    fn _unflatten_ncoxy(tid: Int, OC: Int, OH: Int, OW: Int) -> (Int, Int, Int, Int):
        var oc_oh_ow = OC * OH * OW
        var n = tid // oc_oh_ow
        var r = tid - n * oc_oh_ow
        var oc = r // (OH * OW)
        r = r - oc * (OH * OW)
        var oy = r // OW
        var ox = r - oy * OW
        return (n, oc, oy, ox)

    # -------- Conv Forward (CPU) --------
    fn forward_cpu_parallel(self, x: tensor.Tensor[Float32]) -> tensor.Tensor[Float32]:
        var N = x.shape()[0]; var C = x.shape()[1]; var H = x.shape()[2]; var W = x.shape()[3]
        var OH = (H + 2*self.pad_h - self.kernel_h) // self.stride_h + 1
        var OW = (W + 2*self.pad_w - self.kernel_w) // self.stride_w + 1

        var y = tensor.zeros([N, self.out_channels, OH, OW])

        var HW    = H * W
        var CHW   = C * HW
        var kH    = self.kernel_h
        var kW    = self.kernel_w
        var kHW   = kH * kW
        var CkHW  = C * kHW
        var OH_OW = OH * OW

        var n = 0
        while n < N:
            var oc = 0
            while oc < self.out_channels:
                var y_base = n * (self.out_channels * OH_OW) + oc * OH_OW
                var w_base = oc * CkHW

                var oy = 0
                while oy < OH:
                    var iy0 = oy * self.stride_h - self.pad_h
                    var ky_start = 0
                    if iy0 < 0:
                        ky_start = -iy0
                    var ky_end = kH
                    var last_y = iy0 + kH
                    if last_y > H:
                        ky_end = ky_end - (last_y - H)

                    var ox = 0
                    while ox < OW:
                        var ix0 = ox * self.stride_w - self.pad_w
                        var kx_start = 0
                        if ix0 < 0:
                            kx_start = -ix0
                        var kx_end = kW
                        var last_x = ix0 + kW
                        if last_x > W:
                            kx_end = kx_end - (last_x - W)

                        var sum: Float32 = 0.0

                        var ic = 0
                        while ic < C:
                            var x_c_base = n * CHW + ic * HW
                            var w_c_base = w_base + ic * kHW

                            var ky = ky_start
                            while ky < ky_end:
                                var iy = iy0 + ky
                                var row = iy * W
                                var w_row = w_c_base + ky * kW

                                var kx = kx_start
                                # unroll 8
                                while kx + 7 < kx_end:
                                    var ix = ix0 + kx
                                    var xi = x_c_base + row + ix
                                    var wi = w_row + kx
                                    sum = sum + x._data[xi + 0] * self.weight._data[wi + 0]
                                    sum = sum + x._data[xi + 1] * self.weight._data[wi + 1]
                                    sum = sum + x._data[xi + 2] * self.weight._data[wi + 2]
                                    sum = sum + x._data[xi + 3] * self.weight._data[wi + 3]
                                    sum = sum + x._data[xi + 4] * self.weight._data[wi + 4]
                                    sum = sum + x._data[xi + 5] * self.weight._data[wi + 5]
                                    sum = sum + x._data[xi + 6] * self.weight._data[wi + 6]
                                    sum = sum + x._data[xi + 7] * self.weight._data[wi + 7]
                                    kx = kx + 8
                                # unroll 4
                                while kx + 3 < kx_end:
                                    var ix4 = ix0 + kx
                                    var xi4 = x_c_base + row + ix4
                                    var wi4 = w_row + kx
                                    sum = sum + x._data[xi4 + 0] * self.weight._data[wi4 + 0]
                                    sum = sum + x._data[xi4 + 1] * self.weight._data[wi4 + 1]
                                    sum = sum + x._data[xi4 + 2] * self.weight._data[wi4 + 2]
                                    sum = sum + x._data[xi4 + 3] * self.weight._data[wi4 + 3]
                                    kx = kx + 4
                                # tail
                                while kx < kx_end:
                                    var ix1 = ix0 + kx
                                    var xi1 = x_c_base + row + ix1
                                    var wi1 = w_row + kx
                                    sum = sum + x._data[xi1] * self.weight._data[wi1]
                                    kx = kx + 1
                                ky = ky + 1
                            ic = ic + 1

                        if self.bias:
                            sum = sum + self.bias_t._data[oc]

                        var yi = y_base + oy * OW + ox
                        y._data[yi] = sum

                        ox = ox + 1
                    oy = oy + 1
                oc = oc + 1
            n = n + 1

        return y.copy()

    @staticmethod
    fn  _kernel_conv2d_fwd(
        tid: Int,
        x: List[Float32],
        w: List[Float32],
        b: List[Float32],
        mut y: List[Float32],
        N: Int, C: Int, H: Int, W: Int,
        OH: Int, OW: Int,
        OC: Int,
        kH: Int, kW: Int,
        sH: Int, sW: Int,
        pH: Int, pW: Int,
        has_bias: Bool
    ) -> None:
        var total = N * OC * OH * OW
        if tid >= total:
            return

        var ncoxy = Self._unflatten_ncoxy(tid, OC, OH, OW)
        var n = ncoxy[0]; var oc = ncoxy[1]; var oy = ncoxy[2]; var ox = ncoxy[3]

        var HW   = H * W
        var CHW  = C * HW
        var kHW  = kH * kW
        var CkHW = C * kHW

        var iy0 = oy * sH - pH
        var ky_start = 0
        if iy0 < 0:
            ky_start = -iy0
        var ky_end = kH
        var last_y = iy0 + kH
        if last_y > H:
            ky_end = ky_end - (last_y - H)

        var ix0 = ox * sW - pW
        var kx_start = 0
        if ix0 < 0:
            kx_start = -ix0
        var kx_end = kW
        var last_x = ix0 + kW
        if last_x > W:
            kx_end = kx_end - (last_x - W)

        var sum: Float32 = 0.0
        var w_base = oc * CkHW

        var ic = 0
        while ic < C:
            var x_c_base = n * CHW + ic * HW
            var w_c_base = w_base + ic * kHW

            var ky = ky_start
            while ky < ky_end:
                var iy = iy0 + ky
                var row = iy * W
                var w_row = w_c_base + ky * kW

                var kx = kx_start
                while kx < kx_end:
                    var ix = ix0 + kx
                    var xi = x_c_base + row + ix
                    var wi = w_row + kx
                    sum = sum + x[xi] * w[wi]
                    kx = kx + 1
                ky = ky + 1
            ic = ic + 1

        if has_bias:
            sum = sum + b[oc]

        var yi = (((n * OC) + oc) * OH + oy) * OW + ox
        y[yi] = sum

    fn forward_gpu(self, x: tensor.Tensor[Float32]) -> tensor.Tensor[Float32]:
        var N  = x.shape()[0]
        var C  = x.shape()[1]
        var H  = x.shape()[2]
        var W  = x.shape()[3]

        var OH = (H + 2 * self.pad_h - self.kernel_h) // self.stride_h + 1
        var OW = (W + 2 * self.pad_w - self.kernel_w) // self.stride_w + 1
        var OC = self.out_channels

        var y = tensor.zeros([N, OC, OH, OW])

        # Create a single work buffer for output (no double copy)
        var total = N * OC * OH * OW
        var ybuf = List[Float32]()
        ybuf.reserve(total)
        var i = 0
        while i < total:
            ybuf.append(0.0)
            i += 1

        var block = 256
        if block > total:
            block = total

        launch_1d_conv2d_fw(
            total, block, Self._kernel_conv2d_fwd,
            x._data, self.weight._data, self.bias_t._data, ybuf,
            N, C, H, W, OH, OW, OC,
            self.kernel_h, self.kernel_w,
            self.stride_h, self.stride_w,
            self.pad_h, self.pad_w,
            self.bias
        )

        # Single assignment; no extra copy
        y._data = ybuf.copy()
        return y.copy()

    fn forward_auto(self, x: tensor.Tensor[Float32]) -> tensor.Tensor[Float32]:
        if self._has_gpu():
            return self.forward_gpu(x)
        return self.forward_cpu_parallel(x)

    fn forward(self, x: tensor.Tensor[Float32]) -> tensor.Tensor[Float32]:
        return self.forward_auto(x)

    # -------- Conv Backward (CPU) --------
    fn backward_cpu_parallel(
        self,
        x: tensor.Tensor[Float32],
        grad_y: tensor.Tensor[Float32]
    ) -> (tensor.Tensor[Float32], tensor.Tensor[Float32], tensor.Tensor[Float32]):
        var N = x.shape()[0]; var C = x.shape()[1]; var H = x.shape()[2]; var W = x.shape()[3]
        var OC = self.out_channels
        var kH = self.kernel_h; var kW = self.kernel_w
        var sH = self.stride_h; var sW = self.stride_w
        var pH = self.pad_h;    var pW = self.pad_w
        var OH = grad_y.shape()[2]; var OW = grad_y.shape()[3]

        var dW = tensor.zeros([OC, C, kH, kW])
        var db = tensor.zeros([OC])
        var dx = tensor.zeros_like(x)

        var HW = H*W; var CHW = C*HW
        var kHW = kH*kW; var CkHW = C*kHW
        var OHOW = OH*OW

        var n = 0
        while n < N:
            var oc = 0
            while oc < OC:
                var gy_base = n*(OC*OHOW) + oc*OHOW

                var oyb = 0
                while oyb < OH:
                    var oxb = 0
                    while oxb < OW:
                        db._data[oc] = db._data[oc] + grad_y._data[gy_base + oyb*OW + oxb]
                        oxb = oxb + 1
                    oyb = oyb + 1

                var oy = 0
                while oy < OH:
                    var iy0 = oy * sH - pH
                    var ky_start = 0
                    if iy0 < 0:
                        ky_start = -iy0
                    var ky_end = kH
                    var last_y = iy0 + kH
                    if last_y > H:
                        ky_end = ky_end - (last_y - H)

                    var ox = 0
                    while ox < OW:
                        var ix0 = ox * sW - pW
                        var kx_start = 0
                        if ix0 < 0:
                            kx_start = -ix0
                        var kx_end = kW
                        var last_x = ix0 + kW
                        if last_x > W:
                            kx_end = kx_end - (last_x - W)

                        var gy_val: Float32 = grad_y._data[gy_base + oy*OW + ox]

                        var ic = 0
                        while ic < C:
                            var x_c_base = n*CHW + ic*HW
                            var w_c_base = oc*CkHW + ic*kHW

                            var ky = ky_start
                            while ky < ky_end:
                                var iy = iy0 + ky
                                var row = iy * W
                                var w_row = w_c_base + ky * kW

                                var kx = kx_start
                                # unroll 8
                                while kx + 7 < kx_end:
                                    var ix = ix0 + kx
                                    var xi = x_c_base + row + ix
                                    var wi = w_row + kx
                                    dW._data[wi + 0] = dW._data[wi + 0] + x._data[xi + 0] * gy_val
                                    dW._data[wi + 1] = dW._data[wi + 1] + x._data[xi + 1] * gy_val
                                    dW._data[wi + 2] = dW._data[wi + 2] + x._data[xi + 2] * gy_val
                                    dW._data[wi + 3] = dW._data[wi + 3] + x._data[xi + 3] * gy_val
                                    dW._data[wi + 4] = dW._data[wi + 4] + x._data[xi + 4] * gy_val
                                    dW._data[wi + 5] = dW._data[wi + 5] + x._data[xi + 5] * gy_val
                                    dW._data[wi + 6] = dW._data[wi + 6] + x._data[xi + 6] * gy_val
                                    dW._data[wi + 7] = dW._data[wi + 7] + x._data[xi + 7] * gy_val
                                    dx._data[xi + 0] = dx._data[xi + 0] + self.weight._data[wi + 0] * gy_val
                                    dx._data[xi + 1] = dx._data[xi + 1] + self.weight._data[wi + 1] * gy_val
                                    dx._data[xi + 2] = dx._data[xi + 2] + self.weight._data[wi + 2] * gy_val
                                    dx._data[xi + 3] = dx._data[xi + 3] + self.weight._data[wi + 3] * gy_val
                                    dx._data[xi + 4] = dx._data[xi + 4] + self.weight._data[wi + 4] * gy_val
                                    dx._data[xi + 5] = dx._data[xi + 5] + self.weight._data[wi + 5] * gy_val
                                    dx._data[xi + 6] = dx._data[xi + 6] + self.weight._data[wi + 6] * gy_val
                                    dx._data[xi + 7] = dx._data[xi + 7] + self.weight._data[wi + 7] * gy_val
                                    kx = kx + 8
                                # unroll 4
                                while kx + 3 < kx_end:
                                    var ix4 = ix0 + kx
                                    var xi4 = x_c_base + row + ix4
                                    var wi4 = w_row + kx
                                    dW._data[wi4 + 0] = dW._data[wi4 + 0] + x._data[xi4 + 0] * gy_val
                                    dW._data[wi4 + 1] = dW._data[wi4 + 1] + x._data[xi4 + 1] * gy_val
                                    dW._data[wi4 + 2] = dW._data[wi4 + 2] + x._data[xi4 + 2] * gy_val
                                    dW._data[wi4 + 3] = dW._data[wi4 + 3] + x._data[xi4 + 3] * gy_val
                                    dx._data[xi4 + 0] = dx._data[xi4 + 0] + self.weight._data[wi4 + 0] * gy_val
                                    dx._data[xi4 + 1] = dx._data[xi4 + 1] + self.weight._data[wi4 + 1] * gy_val
                                    dx._data[xi4 + 2] = dx._data[xi4 + 2] + self.weight._data[wi4 + 2] * gy_val
                                    dx._data[xi4 + 3] = dx._data[xi4 + 3] + self.weight._data[wi4 + 3] * gy_val
                                    kx = kx + 4
                                # tail
                                while kx < kx_end:
                                    var ix1 = ix0 + kx
                                    var xi1 = x_c_base + row + ix1
                                    var wi1 = w_row + kx
                                    dW._data[wi1] = dW._data[wi1] + x._data[xi1] * gy_val
                                    dx._data[xi1] = dx._data[xi1] + self.weight._data[wi1] * gy_val
                                    kx = kx + 1
                                ky = ky + 1
                            ic = ic + 1
                        ox = ox + 1
                    oy = oy + 1
                oc = oc + 1
            n = n + 1

        return (dW.copy(), db.copy(), dx.copy())

    fn backward_gpu(
        self,
        x: tensor.Tensor[Float32],
        grad_y: tensor.Tensor[Float32]
    ) -> (tensor.Tensor[Float32], tensor.Tensor[Float32], tensor.Tensor[Float32]):
        # Fallback to CPU until dedicated GPU kernels for backprop are added
        return self.backward_cpu_parallel(x, grad_y)

    fn backward_auto(
        self,
        x: tensor.Tensor[Float32],
        grad_y: tensor.Tensor[Float32]
    ) -> (tensor.Tensor[Float32], tensor.Tensor[Float32], tensor.Tensor[Float32]):
        if self._has_gpu():
            return self.backward_gpu(x, grad_y)
        return self.backward_cpu_parallel(x, grad_y)

    fn backward(
        self,
        x: tensor.Tensor[Float32],
        grad_y: tensor.Tensor[Float32]
    ) -> (tensor.Tensor[Float32], tensor.Tensor[Float32], tensor.Tensor[Float32]):
        return self.backward_auto(x, grad_y)
