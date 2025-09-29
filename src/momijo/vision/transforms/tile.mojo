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
# Project: momijo.vision.transforms
# File: momijo/vision/transforms/tile.mojo
 
 

from momijo.vision.tensor import Tensor, packed_hwc_strides
from momijo.vision.dtypes import DType
from momijo.vision.schedule.schedule import Schedule, default_schedule_64x16


# -----------------------------------------------------------------------------
# Core tiled kernel (HWC/u8)
# -----------------------------------------------------------------------------
fn resize_nearest_u8_hwc_tiled_scheduled(src: Tensor, out_h: Int, out_w: Int, sched: Schedule) -> Tensor:
    # Preconditions
    assert(src.dtype() == DType.UInt8, "resize_nearest_u8_hwc_tiled: only UInt8 supported")
    var h = src.height()
    var w = src.width()
    var c = src.channels()
    assert(h > 0 and w > 0 and c > 0, "resize_nearest_u8_hwc_tiled: bad src shape")
    assert(out_h > 0 and out_w > 0, "resize_nearest_u8_hwc_tiled: bad output shape")

    # Prepare output (packed HWC)
    var (s0_out, s1_out, s2_out) = packed_hwc_strides(out_h, out_w, c)
    var out_len = out_h * out_w * c
    var out_buf = UnsafePointer[UInt8].alloc(out_len)
    var out = Tensor(out_buf, out_len, out_h, out_w, c, s0_out, s1_out, s2_out, DType.UInt8)

    # Source strides/pointer
    var s0 = src.stride0()
    var s1 = src.stride1()
    var s2 = src.stride2()
    var sp = src.data()

    # Scale factors (same as monolithic nearest)
    var y_scale = Float64(h) / Float64(out_h)
    var x_scale = Float64(w) / Float64(out_w)

    # Tile parameters
    var TH = sched.tile_h
    var TW = sched.tile_w
    if TH <= 0: TH = 64
    if TW <= 0: TW = 64

    var ty:Int = 0
    while ty < out_h:
        var tile_h = TH
        if ty + tile_h > out_h:
            tile_h = out_h - ty

        var tx:Int = 0
        while tx < out_w:
            var tile_w = TW
            if tx + tile_w > out_w:
                tile_w = out_w - tx

            # Process one output tile [ty:ty+tile_h), [tx:tx+tile_w)
            var y:Int = 0
            while y < tile_h:
                var oy = ty + y
                var src_y = _clamp_i(Int(Float64(oy) * y_scale), 0, h - 1)
                var x:Int = 0
                while x < tile_w:
                    var ox = tx + x
                    var src_x = _clamp_i(Int(Float64(ox) * x_scale), 0, w - 1)

                    var base_src = src_y * s0 + src_x * s1
                    var base_out = oy * s0_out + ox * s1_out

                    var ch:Int = 0
                    while ch < c:
                        out_buf[base_out + ch * s2_out] = sp[base_src + ch * s2]
                        ch += 1
                    x += 1
                y += 1

            tx += tile_w
        ty += tile_h

    return out

# Public API: pick a reasonable default schedule and run the tiled kernel.
fn resize_nearest_u8_hwc_tiled(src: Tensor, out_h: Int, out_w: Int) -> Tensor:
    var sched = default_schedule_64x16()
    return resize_nearest_u8_hwc_tiled_scheduled(src, out_h, out_w, sched)
