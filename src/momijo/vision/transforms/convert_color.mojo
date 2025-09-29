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
# File: momijo/vision/transforms/convert_color.mojo
 
from momijo.vision.tensor import Tensor
from momijo.vision.backend.registry import default_registry
 
from momijo.vision.tensor import Tensor
from momijo.vision.dtypes import DType

fn rgb_to_gray(src: Tensor) -> Tensor:
    # Preconditions & shape
    assert(src.dtype() == DType.UInt8, "rgb_to_gray: only UInt8 supported")
    var h = src.height()
    var w = src.width()
    var c = src.channels()
    assert(h > 0 and w > 0 and c > 0, "rgb_to_gray: bad source shape")

    # If already single channel, just return a packed copy to normalize layout.
    if c == 1:
        return src.copy_to_packed_hwc()

    # Output: packed HWC with C=1
    var out_c = 1
    var s0_out = w * out_c
    var s1_out = out_c
    var s2_out = 1
    var out_len = h * w * out_c
    var out_buf = UnsafePointer[UInt8].alloc(out_len)
    var out = Tensor(out_buf, out_len, h, w, out_c, s0_out, s1_out, s2_out, DType.UInt8)

    # Read using source strides (works for any strided HWC)
    var s0 = src.stride0(); var s1 = src.stride1(); var s2 = src.stride2()
    var sp = src.data()

    var y: Int = 0
    while y < h:
        var x: Int = 0
        while x < w:
            var base = y * s0 + x * s1
            # If c >= 3 assume RGB in channels 0,1,2. If fewer/more, do best-effort.
            var r: UInt8 = 0
            var g: UInt8 = 0
            var b: UInt8 = 0

            if c >= 3:
                r = sp[base + 0 * s2]
                g = sp[base + 1 * s2]
                b = sp[base + 2 * s2]
            elif c == 2:
                # Treat the second channel as G; use R from first and B=G for a reasonable fallback.
                r = sp[base + 0 * s2]
                g = sp[base + 1 * s2]
                b = g
            else:
                # c==1 case already handled; this is defensive.
                var v = sp[base]
                r = v; g = v; b = v

            # Integer BT.601 luma with rounding
            var y_int = (77 * Int(r)) + (150 * Int(g)) + (29 * Int(b)) + 128
            var y_u8 = UInt8( (y_int >> 8) & 0xFF )

            out_buf[y * s0_out + x * s1_out + 0 * s2_out] = y_u8
            x += 1
        y += 1

    return out
