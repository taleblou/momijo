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
# File: momijo/vision/backend/cpu/resize_tiled_cpu.mojo
 
from vision.tensor import Tensor, packed_hwc_strides
from vision.dtypes import DType

fn clamp_i32(x: Int, lo: Int, hi: Int) -> Int:
    if x < lo: return lo
    if x > hi: return hi
    return x

fn resize_u8_hwc_nearest_tiled_into(src: Tensor, dst: Tensor, tile_h: Int, tile_w: Int):
    var in_h = src.height()
    var in_w = src.width()
    var in_c = src.channels()

    var out_h = dst.height()
    var out_w = dst.width()
    var out_c = dst.channels()

    var s_stride0 = src.stride0()
    var s_stride1 = src.stride1()
    var s_stride2 = src.stride2()
    var s_ptr = src.data()

    var d_stride0 = dst.stride0()
    var d_stride1 = dst.stride1()
    var d_stride2 = dst.stride2()
    var d_ptr = dst.data()

    var scale_y = (in_h * 1.0) / (out_h * 1.0)
    var scale_x = (in_w * 1.0) / (out_w * 1.0)

    var y0:Int = 0
    while y0 < out_h:
        var x0:Int = 0
        var y1 = (y0 + tile_h < out_h) ? (y0 + tile_h) : out_h
        while x0 < out_w:
            var x1 = (x0 + tile_w < out_w) ? (x0 + tile_w) : out_w
            var y:Int = y0
            while y < y1:
                var sy = clamp_i32(Int(y * scale_y), 0, in_h - 1)
                var x:Int = x0
                while x < x1:
                    var sx = clamp_i32(Int(x * scale_x), 0, in_w - 1)
                    var s_base = sy * s_stride0 + sx * s_stride1
                    var d_base = y * d_stride0 + x * d_stride1
                    var c:Int = 0
                    while c < out_c:
                        d_ptr[d_base + c] = s_ptr[s_base + c * s_stride2]
                        c += 1
                    x += 1
                y += 1
            x0 += tile_w
        y0 += tile_h

fn resize_u8_hwc_nearest_tiled(src: Tensor, out_h: Int, out_w: Int, tile_h: Int, tile_w: Int) -> Tensor:
    var out_c = src.channels()
    var (d0,d1,d2) = packed_hwc_strides(out_h, out_w, out_c)
    var n = out_h * out_w * out_c
    var buf = UnsafePointer[UInt8].alloc(n)
    var dst = Tensor(buf, n, out_h, out_w, out_c, d0,d1,d2, DType.UInt8)
    resize_u8_hwc_nearest_tiled_into(src, dst, tile_h, tile_w)
    return dst