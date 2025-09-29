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
# File: momijo/vision/backend/cpu/resize_cpu.mojo
 
from vision.tensor import Tensor, packed_hwc_strides
from vision.dtypes import DType

fn clamp_i32(x: Int, lo: Int, hi: Int) -> Int:
    if x < lo: return lo
    if x > hi: return hi
    return x

fn resize_u8_hwc_nearest(src: Tensor, out_h: Int, out_w: Int) -> Tensor:
    var in_h = src.height()
    var in_w = src.width()
    var in_c = src.channels()
    var out_c = in_c

    var (d_stride0, d_stride1, d_stride2) = packed_hwc_strides(out_h, out_w, out_c)
    var out_len = out_h * out_w * out_c
    var d_ptr = UnsafePointer[UInt8].alloc(out_len)
    var dst = Tensor(d_ptr, out_len, out_h, out_w, out_c, d_stride0, d_stride1, d_stride2, DType.UInt8)

    var s_stride0 = src.stride0()
    var s_stride1 = src.stride1()
    var s_stride2 = src.stride2()
    var s_ptr = src.data()

    var scale_y = (in_h * 1.0) / (out_h * 1.0)
    var scale_x = (in_w * 1.0) / (out_w * 1.0)

    var y: Int = 0
    while y < out_h:
        var x: Int = 0
        var sy = clamp_i32(Int(y * scale_y), 0, in_h - 1)
        while x < out_w:
            var sx = clamp_i32(Int(x * scale_x), 0, in_w - 1)
            var s_base = sy * s_stride0 + sx * s_stride1
            var d_base = y * d_stride0 + x * d_stride1
            var c: Int = 0
            while c < in_c:
                d_ptr[d_base + c] = s_ptr[s_base + c * s_stride2]
                c += 1
            x += 1
        y += 1
    return dst