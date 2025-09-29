# MIT License
# Copyright (c) 2025 Morteza Talebou and Mitra Daneshmand
# Project: momijo  |  Source: https://github.com/taleblou/momijo
# This file is part of the Momijo project. See the LICENSE file at the repository root.
# Momijo 
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Morteza Talebou and Mitra Daneshmand
# Website: https://taleblou.ir/
# Repository: https://github.com/taleblou/momijo
#
# Project: momijo.vision.backend.cpu
# File: momijo/vision/backend/cpu/convert_color_cpu.mojo
 
from vision.tensor import Tensor, packed_hwc_strides
from vision.dtypes import DType

fn rgb_to_gray_u8_hwc(src: Tensor) -> Tensor:
    var h = src.height()
    var w = src.width()
    var c = src.channels()
    if c < 3:
        # nothing to do; return copy
        return src.copy_to_packed_hwc()
    var out_c = 1
    var (s0, s1, s2) = packed_hwc_strides(h, w, out_c)
    var out_len = h * w * out_c
    var out_buf = UnsafePointer[UInt8].alloc(out_len)
    var dst = Tensor(out_buf, out_len, h, w, out_c, s0, s1, s2, DType.UInt8)

    var s_stride0 = src.stride0()
    var s_stride1 = src.stride1()
    var s_stride2 = src.stride2()
    var sp = src.data()
    var dp = dst.data()

    var y:Int = 0
    while y < h:
        var x:Int = 0
        while x < w:
            var base = y*s_stride0 + x*s_stride1
            var r = sp[base + 0*s_stride2]
            var g = sp[base + 1*s_stride2]
            var b = sp[base + 2*s_stride2]
            # integer approx
            var yv = (Int(r)*299 + Int(g)*587 + Int(b)*114) / 1000
            dp[y*s0 + x*s1] = UInt8(yv)
            x += 1
        y += 1
    return dst