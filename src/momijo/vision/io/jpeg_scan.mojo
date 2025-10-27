# MIT License
# Copyright (c) 2025
# Project: momijo
# SPDX-License-Identifier: MIT
# File: src/momijo/vision/io/jpeg_scan.mojo
# Description: Decode one 8x8 block from a JPEG scan: entropy (DC+AC) -> dequant -> IDCT.

from momijo.vision.io.bitreader import BitReader
from momijo.vision.io.huffman import HuffmanTable, receive_extend
from momijo.vision.io.idct import idct_8x8

# Build standard zigzag index table (zigzag position -> natural 8x8 index).
# Returned buffer is heap-allocated to respect "no globals"; caller frees.
@always_inline
fn _zigzag_table() -> UnsafePointer[Int]:
    var p = UnsafePointer[Int].alloc(64)
    # Row 0
    p[ 0]=0;  p[ 1]=1;  p[ 2]=5;  p[ 3]=6;  p[ 4]=14; p[ 5]=15; p[ 6]=27; p[ 7]=28
    # Row 1
    p[ 8]=2;  p[ 9]=4;  p[10]=7;  p[11]=13; p[12]=16; p[13]=26; p[14]=29; p[15]=42
    # Row 2
    p[16]=3;  p[17]=8;  p[18]=12; p[19]=17; p[20]=25; p[21]=30; p[22]=41; p[23]=43
    # Row 3
    p[24]=9;  p[25]=11; p[26]=18; p[27]=24; p[28]=31; p[29]=40; p[30]=44; p[31]=53
    # Row 4
    p[32]=10; p[33]=19; p[34]=23; p[35]=32; p[36]=39; p[37]=45; p[38]=52; p[39]=54
    # Row 5
    p[40]=20; p[41]=22; p[42]=33; p[43]=38; p[44]=46; p[45]=51; p[46]=55; p[47]=60
    # Row 6
    p[48]=21; p[49]=34; p[50]=37; p[51]=47; p[52]=50; p[53]=56; p[54]=59; p[55]=61
    # Row 7
    p[56]=35; p[57]=36; p[58]=48; p[59]=49; p[60]=57; p[61]=58; p[62]=62; p[63]=63
    return p

# Decode one 8x8 block: DC + AC in zigzag order, de-quantize to natural order, then IDCT in-place.
# Returns the new DC predictor (actual DC value) for chaining, or a negative error code.
fn decode_block(
    mut br: BitReader,
    dc_table: HuffmanTable,
    ac_table: HuffmanTable,
    quant: UnsafePointer[UInt8],   # quant table in natural order (64 bytes)
    prev_dc: Int,
    dst: UnsafePointer[Int]        # output: spatial domain after IDCT (8x8), Int per sample
) -> Int:
    # Clear block coefficients (frequency domain)
    var i = 0
    while i < 64:
        dst[i] = 0
        i += 1

    var zz = _zigzag_table()

    # ---- DC ----
    var t = dc_table.decode(br)        # number of bits (category)
    if t < 0:
        UnsafePointer[Int].free(zz)
        return -1
    var diff = receive_extend(br, t)   # sign-extended t-bit value
    var dc = prev_dc + diff
    # DC is position zigzag 0 -> natural index 0
    dst[0] = dc * Int(quant[0])

    # ---- AC ----
    var k = 1
    while k < 64:
        var rs = ac_table.decode(br)
        if rs < 0:
            UnsafePointer[Int].free(zz)
            return -1

        if rs == 0:
            # End-Of-Block
            break

        if rs == 0xF0:
            # ZRL: run of 16 zeros
            k = k + 16
            if k >= 64:
                UnsafePointer[Int].free(zz)
                return -2
            continue

        var run = (rs >> 4) & 0x0F
        var size = rs & 0x0F

        k = k + run
        if k >= 64:
            UnsafePointer[Int].free(zz)
            return -2

        var ac_val = receive_extend(br, size)
        var nat = zz[k]  # map zigzag position -> natural coefficient index
        dst[nat] = ac_val * Int(quant[nat])
        k = k + 1

    # Done with coefficients; free zigzag table
    UnsafePointer[Int].free(zz)

    # Inverse DCT in place: dst transforms to spatial (still centered around 0)
    idct_8x8(dst)
    return dc
