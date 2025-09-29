# MIT License
# Copyright (c) 2025 Morteza Talebou and Mitra Daneshmand
# Project: momijo
# SPDX-License-Identifier: MIT
# File: momijo/vision/io/jpeg_huffman_encode.mojo

from momijo.vision.io.bitwriter import BitWriter

# -----------------------------------------------------------------------------
# Minimal Huffman table encoder (prebuilt table for JPEG baseline)
# -----------------------------------------------------------------------------

# A single Huffman code entry: bit pattern and bit-length.
struct HuffCode(ExplicitlyCopyable, Movable):
    var code: UInt16
    var size: Int

    fn __init__(out self, code: UInt16, size: Int):
        self.code = code
        self.size = size

    fn __copyinit__(out self, other: Self):
        self.code = other.code
        self.size = other.size

# A table mapping 0..255 symbols to HuffCode.
struct HuffmanEncTable(ExplicitlyCopyable, Movable):
    var values: List[HuffCode]  # index = symbol

    fn __init__(out self):
        self.values = List[HuffCode]()

    fn __copyinit__(out self, other: Self):
        self.values = other.values

    # Emit the Huffman code for 'symbol' to the bitstream.
    fn encode(self, mut bw: BitWriter, symbol: Int):
        var h = self.values[symbol]
        bw.write_bits(UInt32(h.code), h.size)

# Build an encoder table from JPEG 'lengths' (16 counts) and 'symbols' array.
# - lengths[0..15] = number of codes of length 1..16
# - symbols are listed in order of increasing code length
fn build_huffman_encoder(lengths: UnsafePointer[UInt8],
                         symbols: UnsafePointer[UInt8]) -> HuffmanEncTable:
    var table = HuffmanEncTable()

    # Pre-size list to 256 default entries so we can assign by index.
    var i = 0
    while i < 256:
        table.values.append(HuffCode(UInt16(0), 0))
        i = i + 1

    var code: UInt32 = 0
    var k = 0        # index into 'symbols'
    var bitlen = 1
    while bitlen <= 16:
        var count = Int(lengths[bitlen - 1])
        var j = 0
        while j < count:
            var sym = Int(symbols[k])
            table.values[sym] = HuffCode(UInt16(code), bitlen)
            code = code + 1
            k = k + 1
            j = j + 1
        # Next code length: left shift the running code by 1
        code = code << 1
        bitlen = bitlen + 1

    return table

# Return number of bits needed to represent |v| (JPEG category).
fn bits_required(v: Int) -> Int:
    var absval = v
    if absval < 0:
        absval = -absval
    var bits = 0
    if absval == 0:
        return 0
    while absval > 0:
        bits = bits + 1
        absval = absval >> 1
    return bits

# Emit one 8x8 block (already quantized and zigzagged) using given Huffman tables.
# Returns the current DC as new predictor for the next block.
fn emit_block(mut bw: BitWriter,
              dc_table: HuffmanEncTable,
              ac_table: HuffmanEncTable,
              block: UnsafePointer[Int],
              prev_dc: Int) -> Int:
    # ---- DC ----
    var dc = block[0]
    var diff = dc - prev_dc
    var s = bits_required(diff)
    dc_table.encode(bw, s)
    if s > 0:
        var val = diff
        if val < 0:
            # JPEG negative representation: (value - 1) masked to s bits
            val = val - 1
        bw.write_bits(UInt32(val & ((1 << s) - 1)), s)

    # ---- AC ----
    var zero_run = 0
    var k = 1
    while k < 64:
        var ac = block[k]
        if ac == 0:
            zero_run = zero_run + 1
        else:
            # Emit ZRL (0xF0) for each group of 16 zeros
            while zero_run >= 16:
                ac_table.encode(bw, 0xF0)
                zero_run = zero_run - 16
            var s2 = bits_required(ac)
            var symbol = (zero_run << 4) | s2
            ac_table.encode(bw, symbol)
            var val2 = ac
            if val2 < 0:
                val2 = val2 - 1
            bw.write_bits(UInt32(val2 & ((1 << s2) - 1)), s2)
            zero_run = 0
        k = k + 1

    # End-of-block if we ended with zeros
    if zero_run > 0:
        ac_table.encode(bw, 0x00)

    return dc
