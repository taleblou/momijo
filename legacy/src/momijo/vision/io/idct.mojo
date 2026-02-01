# MIT License
# Copyright (c) 2025 Morteza Talebou and Mitra Daneshmand
# Project: momijo
# SPDX-License-Identifier: MIT
# File: momijo/vision/io/idct.mojo

# Integer-based Inverse Discrete Cosine Transform (IDCT) for an 8x8 block.
# This is a compact AAN-style integer approximation with two 1-D passes
# (rows then columns), working in-place on an Int buffer of length 64.

# Safe arithmetic downscaling with rounding
fn descale(x: Int, n: Int) -> Int:
    return (x + (1 << (n - 1))) >> n

# Perform in-place IDCT on a single 8x8 block.
# 'data' must point to 64 Ints laid out in row-major order.
fn idct_8x8(data: UnsafePointer[Int]):
    # All "constants" are local vars (project rule: no file-scope const/let).
    var CONST_BITS = 13
    var PASS1_BITS = 2

    # Fixed-point constants (scaled by 2^CONST_BITS)
    var FIX_0_298 = 2446    # FIX(0.298631336)
    var FIX_0_390 = 3196    # FIX(0.390180644)
    var FIX_0_541 = 4433    # FIX(0.541196100)
    var FIX_0_765 = 6270    # FIX(0.765366865)
    var FIX_0_899 = 7373    # FIX(0.899976223)
    var FIX_1_175 = 9633    # FIX(1.175875602)
    var FIX_1_501 = 12299   # FIX(1.501321110)
    var FIX_1_847 = 15137   # FIX(1.847759065)
    var FIX_1_961 = 16069   # FIX(1.961570560)
    var FIX_2_053 = 16819   # FIX(2.053119869)
    var FIX_2_562 = 20995   # FIX(2.562915447)
    var FIX_3_072 = 25172   # FIX(3.072711026)

    # Temporary buffer for the row pass
    var tmp = UnsafePointer[Int].alloc(64)

    # Pass 1: process rows
    var i = 0
    while i < 8:
        var off = i * 8
        var d0 = data[off + 0]
        var d1 = data[off + 1]
        var d2 = data[off + 2]
        var d3 = data[off + 3]
        var d4 = data[off + 4]
        var d5 = data[off + 5]
        var d6 = data[off + 6]
        var d7 = data[off + 7]

        # Even part
        var t0 = (d0 + d4) << CONST_BITS
        var t1 = (d0 - d4) << CONST_BITS
        var t2 = d2 * FIX_1_847 + d6 * FIX_0_765
        var t3 = d2 * FIX_0_765 - d6 * FIX_1_847

        var even0 = t0 + t2
        var even1 = t1 + t3
        var even2 = t1 - t3
        var even3 = t0 - t2

        # Odd part
        var z1 = d7 + d1
        var z2 = d5 + d3
        var z3 = d7 + d3
        var z4 = d5 + d1
        var z5 = (z3 + z4) * FIX_1_175

        var p0 = d7 * FIX_0_298
        var p1 = d5 * FIX_2_053
        var p2 = d3 * FIX_3_072
        var p3 = d1 * FIX_1_501

        var odd0 = p0 + p1 + z5 - z3 * FIX_0_899 - z4 * FIX_2_562
        var odd1 = p2 + p3 + z5 - z3 * FIX_2_562 - z4 * FIX_0_899
        var odd2 = p2 - p3 + z5 - z3 * FIX_0_899 - z4 * FIX_2_562
        var odd3 = p0 - p1 + z5 - z3 * FIX_2_562 - z4 * FIX_0_899

        # Store row results (scaled down by CONST_BITS - PASS1_BITS)
        tmp[off + 0] = descale(even0 + odd0, CONST_BITS - PASS1_BITS)
        tmp[off + 1] = descale(even1 + odd1, CONST_BITS - PASS1_BITS)
        tmp[off + 2] = descale(even2 + odd2, CONST_BITS - PASS1_BITS)
        tmp[off + 3] = descale(even3 + odd3, CONST_BITS - PASS1_BITS)
        tmp[off + 4] = descale(even3 - odd3, CONST_BITS - PASS1_BITS)
        tmp[off + 5] = descale(even2 - odd2, CONST_BITS - PASS1_BITS)
        tmp[off + 6] = descale(even1 - odd1, CONST_BITS - PASS1_BITS)
        tmp[off + 7] = descale(even0 - odd0, CONST_BITS - PASS1_BITS)

        i = i + 1

    # Pass 2: process columns
    i = 0
    while i < 8:
        var d0c = tmp[i + 0 * 8]
        var d1c = tmp[i + 1 * 8]
        var d2c = tmp[i + 2 * 8]
        var d3c = tmp[i + 3 * 8]
        var d4c = tmp[i + 4 * 8]
        var d5c = tmp[i + 5 * 8]
        var d6c = tmp[i + 6 * 8]
        var d7c = tmp[i + 7 * 8]

        var t0c = (d0c + d4c) << CONST_BITS
        var t1c = (d0c - d4c) << CONST_BITS
        var t2c = d2c * FIX_1_847 + d6c * FIX_0_765
        var t3c = d2c * FIX_0_765 - d6c * FIX_1_847

        var even0c = t0c + t2c
        var even1c = t1c + t3c
        var even2c = t1c - t3c
        var even3c = t0c - t2c

        var z1c = d7c + d1c
        var z2c = d5c + d3c
        var z3c = d7c + d3c
        var z4c = d5c + d1c
        var z5c = (z3c + z4c) * FIX_1_175

        var p0c = d7c * FIX_0_298
        var p1c = d5c * FIX_2_053
        var p2c = d3c * FIX_3_072
        var p3c = d1c * FIX_1_501

        var odd0c = p0c + p1c + z5c - z3c * FIX_0_899 - z4c * FIX_2_562
        var odd1c = p2c + p3c + z5c - z3c * FIX_2_562 - z4c * FIX_0_899
        var odd2c = p2c - p3c + z5c - z3c * FIX_0_899 - z4c * FIX_2_562
        var odd3c = p0c - p1c + z5c - z3c * FIX_2_562 - z4c * FIX_0_899

        # Final scale includes PASS1_BITS and additional +3 to approximate 8-bit range
        var final_shift = CONST_BITS + PASS1_BITS + 3
        data[i + 0 * 8] = descale(even0c + odd0c, final_shift)
        data[i + 1 * 8] = descale(even1c + odd1c, final_shift)
        data[i + 2 * 8] = descale(even2c + odd2c, final_shift)
        data[i + 3 * 8] = descale(even3c + odd3c, final_shift)
        data[i + 4 * 8] = descale(even3c - odd3c, final_shift)
        data[i + 5 * 8] = descale(even2c - odd2c, final_shift)
        data[i + 6 * 8] = descale(even1c - odd1c, final_shift)
        data[i + 7 * 8] = descale(even0c - odd0c, final_shift)

        i = i + 1
