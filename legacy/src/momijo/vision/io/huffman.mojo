# MIT License
# Copyright (c) 2025 Morteza Talebou and Mitra Daneshmand
# Project: momijo
# SPDX-License-Identifier: MIT
# File: momijo/vision/io/huffman.mojo

from momijo.vision.io.bitreader import BitReader

# -----------------------------------------------------------------------------
# Huffman Table for JPEG decoding
# -----------------------------------------------------------------------------
struct HuffmanTable:
    var codes: List[UInt16]   # Huffman codes, ordered by length then value
    var sizes: List[Int]      # Number of bits for each code
    var symbols: List[UInt8]  # Output symbol for each code

    fn __init__(out self):
        self.codes = List[UInt16]()
        self.sizes = List[Int]()
        self.symbols = List[UInt8]()

    # Build from JPEG count table (16 bytes) and symbol list.
    # counts[l-1] is the number of codes of length 'l' for l = 1..16.
    # 'symbols' supplies the symbols in order of increasing code length.
    fn build(mut self, counts: UnsafePointer[UInt8], symbols: UnsafePointer[UInt8]):
        var code: UInt16 = 0
        var si = 1
        var sym_idx = 0
        while si <= 16:
            var n = Int(counts[si - 1])
            var i = 0
            while i < n:
                self.codes.append(code)
                self.sizes.append(si)
                self.symbols.append(symbols[sym_idx])
                code = code + 1
                i = i + 1
                sym_idx = sym_idx + 1
            # Next code length: left-shift the running code
            code = code << 1
            si = si + 1

    # Decode a single Huffman-coded symbol from the bitstream.
    # Linear search is used here for simplicity (fine for a minimal decoder).
    fn decode(self, mut br: BitReader) -> Int:
        var code_acc: UInt32 = 0
        var len = 0
        while len < 16:
            # Read one bit (accumulate MSB-first)
            code_acc = (code_acc << 1) | br.read_bits(1)
            len = len + 1

            var i = 0
            var n = self.codes.__len__()
            while i < n:
                if self.sizes[i] == len and UInt32(self.codes[i]) == code_acc:
                    return Int(self.symbols[i])
                i = i + 1
        # Invalid or not found
        return -1

# -----------------------------------------------------------------------------
# Decode additional bits for a value with JPEG sign extension
# -----------------------------------------------------------------------------
fn receive_extend(mut br: BitReader, s: Int) -> Int:
    if s == 0:
        return 0
    var val = Int(br.read_bits(s))
    var vt = 1 << (s - 1)
    if val < vt:
        val = val - ((1 << s) - 1)
    return val
