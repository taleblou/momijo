# MIT License
# Copyright (c) 2025 Morteza
# Project: momijo
# SPDX-License-Identifier: MIT
# File: momijo/vision/io/jpeg_quantize.mojo
# Description: Quantize and zigzag utilities for JPEG encoder (no globals, var-only).

# Quantize one 8x8 DCT block using an 8-bit quantization table.
# dct  : 64 Int coefficients (input)
# quant: 64 UInt8 quantizers (input)
# dst  : 64 Int quantized coefficients (output)
fn quantize_block(dct: UnsafePointer[Int],
                  quant: UnsafePointer[UInt8],
                  dst: UnsafePointer[Int]):
    var i = 0
    while i < 64:
        # Integer division; cast quant[i] to Int explicitly.
        dst[i] = dct[i] // Int(quant[i])
        i = i + 1

# Build and return the standard JPEG zigzag index table (64 entries).
# Returned buffer is heap-allocated to satisfy "no globals" policy.
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

# Copy 8x8 block coefficients from natural order into zigzag order.
# src : 64 Int (natural order)
# dst : 64 Int (zigzag order)
fn zigzag_copy(src: UnsafePointer[Int], dst: UnsafePointer[Int]):
    var zz = _zigzag_table()
    var i = 0
    while i < 64:
        dst[i] = src[zz[i]]
        i = i + 1
