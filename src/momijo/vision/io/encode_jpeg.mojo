# MIT License
# Copyright (c) 2025 Morteza Talebou and Mitra Daneshmand
# Project: momijo
# SPDX-License-Identifier: MIT
# File: momijo/vision/io/encode_jpeg.mojo
# Description: Minimal JPEG encoder (grayscale + RGB), pure Mojo.

from momijo.vision.io.bitwriter import BitWriter
from momijo.vision.io.jpeg_dct import dct_8x8
from momijo.vision.io.jpeg_quantize import quantize_block, zigzag_copy
from momijo.vision.io.jpeg_huffman_encode import build_huffman_encoder, emit_block
from momijo.vision.io.jpeg_writer import (
    write_soi, write_app0_jfif, write_dqt, write_dqt_chroma,
    write_sof0, write_sof0_color, write_dht, write_sos,
    write_sos_color, write_eoi
)
from momijo.vision.io.color_ycbcr import rgb_to_ycbcr
from momijo.vision.io.tables import (
    std_dc_lengths, std_dc_symbols,
    std_ac_lengths, std_ac_symbols,
    std_luma_qt, std_chroma_qt
)

fn encode_jpeg(
    ptr: UnsafePointer[UInt8],
    width: Int,
    height: Int,
    out_buf: UnsafePointer[UInt8],
    out_max: Int,
    channels: Int = 1
) -> (Bool, Int):
    var bw = BitWriter(out_buf)

    # Headers
    write_soi(bw)
    write_app0_jfif(bw)

    # Quantization tables
    write_dqt(bw)
    if channels != 1:
        write_dqt_chroma(bw)

    # SOF0
    if channels == 1:
        write_sof0(bw, width, height)
    else:
        write_sof0_color(bw, width, height)

    # Huffman tables
    var dc_len_u = std_dc_lengths()
    var dc_sym_u = std_dc_symbols()
    var ac_len_u = std_ac_lengths()
    var ac_sym_u = std_ac_symbols()

    write_dht(bw, dc_len_u, dc_sym_u, True)
    write_dht(bw, ac_len_u, ac_sym_u, False)

    var dc_table = build_huffman_encoder(dc_len_u, dc_sym_u)
    var ac_table = build_huffman_encoder(ac_len_u, ac_sym_u)

    if channels == 1:
        # Grayscale scan
        write_sos(bw)

        var block_dct = UnsafePointer[Int].alloc(64)
        var block_q   = UnsafePointer[Int].alloc(64)
        var block_zig = UnsafePointer[Int].alloc(64)
        var qtab      = std_luma_qt()

        var prev_dc = 0
        var y = 0
        while y < height:
            var x = 0
            while x < width:
                dct_8x8(ptr + y * width + x, width, block_dct)
                quantize_block(block_dct, qtab, block_q)
                zigzag_copy(block_q, block_zig)
                prev_dc = emit_block(bw, dc_table, ac_table, block_zig, prev_dc)
                x = x + 8
            y = y + 8

    else:
        # RGB -> YCbCr (4:4:4)
        write_sos_color(bw, 3)

        var npix  = width * height
        var y_buf  = UnsafePointer[UInt8].alloc(npix)
        var cb_buf = UnsafePointer[UInt8].alloc(npix)
        var cr_buf = UnsafePointer[UInt8].alloc(npix)
        rgb_to_ycbcr(ptr, width, height, y_buf, cb_buf, cr_buf)

        var block_dct = UnsafePointer[Int].alloc(64)
        var block_q   = UnsafePointer[Int].alloc(64)
        var block_zig = UnsafePointer[Int].alloc(64)

        var comp = 0
        while comp < 3:
            var src: UnsafePointer[UInt8]
            var qtab: UnsafePointer[UInt8]
            if comp == 0:
                src = y_buf
                qtab = std_luma_qt()
            else:
                if comp == 1:
                    src = cb_buf
                else:
                    src = cr_buf
                qtab = std_chroma_qt()

            var prev_dc = 0
            var y = 0
            while y < height:
                var x = 0
                while x < width:
                    dct_8x8(src + y * width + x, width, block_dct)
                    quantize_block(block_dct, qtab, block_q)
                    zigzag_copy(block_q, block_zig)
                    prev_dc = emit_block(bw, dc_table, ac_table, block_zig, prev_dc)
                    x = x + 8
                y = y + 8
            comp = comp + 1

    bw.flush_final()
    write_eoi(bw)
    return (True, bw.bytes_written())
