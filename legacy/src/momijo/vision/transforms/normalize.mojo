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
# File: momijo/vision/transforms/normalize.mojo
 

@fieldwise_init
struct Normalize:
    var mean0: Float32
    var mean1: Float32
    var mean2: Float32
    var std0:  Float32
    var std1:  Float32
    var std2:  Float32

    fn __init__(out self, mean: (Float32, Float32, Float32), std: (Float32, Float32, Float32)):
        self.mean0 = mean[0]
        self.mean1 = mean[1]
        self.mean2 = mean[2]
        self.std0  = std[0]
        self.std1  = std[1]
        self.std2  = std[2]

    fn __call__(self, src: UnsafePointer[Float32], c: Int, h: Int, w: Int) -> UnsafePointer[Float32]:
        assert(c > 0 and h > 0 and w > 0, "Normalize: invalid input shape")

        # Destination allocation
        var oh = h
        var ow = w
        var dst_elems = c * oh * ow
        var dst = UnsafePointer[Float32].alloc(dst_elems)

        # Contiguous CHW strides
        var sC = h * w
        var sH = w
        var sW = 1

        # Prepare per-channel params with safe std
        fn _safe_std(x: Float32) -> Float32:
            if x <= 0.0:
                return 1e-6
            return x

        var m0 = self.mean0
        var m1 = self.mean1
        var m2 = self.mean2
        var v0 = _safe_std(self.std0)
        var v1 = _safe_std(self.std1)
        var v2 = _safe_std(self.std2)

        var ch: Int = 0
        while ch < c:
            var mean_c: Float32 = 0.0
            var std_c:  Float32 = 1.0
            if ch == 0:
                mean_c = m0; std_c = v0
            elif ch == 1:
                mean_c = m1; std_c = v1
            else:
                mean_c = m2; std_c = v2

            var y: Int = 0
            while y < h:
                var x: Int = 0
                while x < w:
                    var src_idx = ch * sC + y * sH + x * sW
                    var dst_idx = src_idx
                    dst[dst_idx] = (src[src_idx] - mean_c) / std_c
                    x += 1
                y += 1
            ch += 1

        return dst

# Convenience functional wrapper
fn normalize(ptr: UnsafePointer[Float32], c: Int, h: Int, w: Int,
             mean: (Float32, Float32, Float32),
             std:  (Float32, Float32, Float32)) -> UnsafePointer[Float32]:
    var op = Normalize(mean, std)
    return op(ptr, c, h, w)
