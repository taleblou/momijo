# MIT License
# SPDX-License-Identifier: MIT
# Project: momijo
# File: momijo/vision/io/jpeg_dct.mojo
# Description: Forward DCT (8x8) used by the JPEG encoder (float version → int coeffs).

from math import cos, pi

fn dct_8x8(src: UnsafePointer[UInt8], stride: Int, dst: UnsafePointer[Int]):
    var DCT_SHIFT = 3  # optional downscale

    var u = 0
    while u < 8:
        var v = 0
        while v < 8:
            var sum: Float64 = 0.0

            var y = 0
            while y < 8:
                var x = 0
                while x < 8:
                    var s = Float64(Int(src[y * stride + x]) - 128)
                    var cu = cos((Float64(2 * x + 1) * Float64(u) * pi) / 16.0)
                    var cv = cos((Float64(2 * y + 1) * Float64(v) * pi) / 16.0)
                    sum = sum + (s * cu * cv)
                    x = x + 1
                y = y + 1

            var au = 1.0
            if u == 0:
                au = 0.7071067811865476
            var av = 1.0
            if v == 0:
                av = 0.7071067811865476

            var scale = 0.25 * au * av
            var val = sum * scale

            var rounded = val
            if val >= 0.0:
                rounded = val + 0.5
            else:
                rounded = val - 0.5

            var coeff = Int(rounded)

            if DCT_SHIFT > 0:
                coeff = coeff >> DCT_SHIFT  # arithmetic on Int

            dst[v * 8 + u] = coeff
            v = v + 1
        u = u + 1
