# MIT License
# SPDX-License-Identifier: MIT
# Project: momijo.vision.io.jpeg
# File: src/momijo/vision/io/internal/jpeg_huffman_encode.mojo
# Description: JPEG Baseline Huffman (DC/AC) encoder helpers.
# Notes:
# - English-only comments per project rules.
# - Provides a full 'build_default_codebooks(out ...)' that fills caller-owned Lists.
# - Keeps a no-arg shim 'build_default_codebooks() -> Int' for legacy callers.
# - HuffCode is ImplicitlyCopyable to simplify list element usage.

 
from collections.list import List
from momijo.vision.io.bitwriter import BitWriter


# ------------------------------ Types ------------------------------ #
struct HuffCode(ImplicitlyCopyable, Copyable, Movable):
    var code: Int  # MSB-first code bits
    var size: Int  # code length in bits
    fn __init__(out self, code: Int = 0, size: Int = 0):
        self.code = code
        self.size = size

# ----------------------- Canonical builder ----------------------- #
# Build canonical codes from 'bits' (length counts) and 'vals' order.
fn build_canonical_codes(bits: List[UInt8], vals: List[UInt8]) -> List[HuffCode]:
    var sizes = List[Int]()
    var i = 0
    while i < 16:
        var cnt = Int(bits[i])
        var j = 0
        while j < cnt:
            sizes.append(i + 1)
            j += 1
        i += 1

    var codes = List[HuffCode]()
    codes.reserve(len(vals))

    var code = 0
    var k = 0
    while k < len(sizes):
        var si = sizes[k]
        if k > 0:
            var prev = sizes[k-1]
            var s = si - prev
            var t = 0
            while t < s:
                code = code << 1
                t += 1
        codes.append(HuffCode(code, si))
        code += 1
        k += 1
    return codes.copy()

# Expand AC code list (in 'vals' order) into a 256-entry table indexed by symbol 0x00..0xFF.
fn expand_ac(vals: List[UInt8], codes: List[HuffCode]) -> List[HuffCode]:
    var table = List[HuffCode]()
    var i = 0
    while i < 256:
        table.append(HuffCode(0, 0))
        i += 1
    var j = 0
    while j < len(vals):
        var sym = Int(vals[j])
        table[sym] = codes[j]
        j += 1
    return table.copy()

# Expand DC code list for categories 0..11 into a 12-entry table.
fn expand_dc(codes: List[HuffCode]) -> List[HuffCode]:
    var t = List[HuffCode]()
    var i = 0
    while i < 12:
        t.append(HuffCode(0, 0))
        i += 1
    var k = 0
    while k < 12:
        t[k] = codes[k]
        k += 1
    return t.copy()

# -------------------- Default Annex K descriptors ------------------- #
fn bits_dc_luma() -> List[UInt8]:
    return List[UInt8]([
        0x00,0x01,0x05,0x01,0x01,0x01,0x01,0x01,
        0x01,0x01,0x00,0x00,0x00,0x00,0x00,0x00
    ])

fn bits_dc_chroma() -> List[UInt8]:
    return List[UInt8]([
        0x00,0x03,0x01,0x01,0x01,0x01,0x01,0x01,
        0x01,0x01,0x01,0x00,0x00,0x00,0x00,0x00
    ])

fn vals_dc_common() -> List[UInt8]:
    return List[UInt8]([0,1,2,3,4,5,6,7,8,9,10,11])

fn bits_ac_luma() -> List[UInt8]:
    return List[UInt8]([
        0x00,0x02,0x01,0x03,0x03,0x02,0x04,0x03,
        0x05,0x05,0x04,0x04,0x00,0x00,0x01,0x7D
    ])

fn vals_ac_luma() -> List[UInt8]:
    return List[UInt8]([
        0x01,0x02,0x03,0x00,0x04,0x11,0x05,0x12,0x21,0x31,0x41,0x06,0x13,0x51,0x61,0x07,
        0x22,0x71,0x14,0x32,0x81,0x91,0xA1,0x08,0x23,0x42,0xB1,0xC1,0x15,0x52,0xD1,0xF0,
        0x24,0x33,0x62,0x72,0x82,0x09,0x0A,0x16,0x17,0x18,0x19,0x1A,0x25,0x26,0x27,0x28,
        0x29,0x2A,0x34,0x35,0x36,0x37,0x38,0x39,0x3A,0x43,0x44,0x45,0x46,0x47,0x48,0x49,
        0x4A,0x53,0x54,0x55,0x56,0x57,0x58,0x59,0x5A,0x63,0x64,0x65,0x66,0x67,0x68,0x69,
        0x6A,0x73,0x74,0x75,0x76,0x77,0x78,0x79,0x7A,0x83,0x84,0x85,0x86,0x87,0x88,0x89,
        0x8A,0x92,0x93,0x94,0x95,0x96,0x97,0x98,0x99,0x9A,0xA2,0xA3,0xA4,0xA5,0xA6,0xA7,
        0xA8,0xA9,0xAA,0xB2,0xB3,0xB4,0xB5,0xB6,0xB7,0xB8,0xB9,0xBA,0xC2,0xC3,0xC4,0xC5,
        0xC6,0xC7,0xC8,0xC9,0xCA,0xD2,0xD3,0xD4,0xD5,0xD6,0xD7,0xD8,0xD9,0xDA,0xE1,0xE2,
        0xE3,0xE4,0xE5,0xE6,0xE7,0xE8,0xE9,0xEA,0xF1,0xF2,0xF3,0xF4,0xF5,0xF6,0xF7,0xF8,
        0xF9,0xFA
    ])

fn bits_ac_chroma() -> List[UInt8]:
    return List[UInt8]([
        0x00,0x02,0x01,0x02,0x04,0x04,0x03,0x04,
        0x07,0x05,0x04,0x04,0x00,0x01,0x02,0x77
    ])

fn vals_ac_chroma() -> List[UInt8]:
    return List[UInt8]([
        0x00,0x01,0x02,0x03,0x11,0x04,0x05,0x21,0x31,0x06,0x12,0x41,0x51,0x07,0x61,0x71,
        0x13,0x22,0x32,0x81,0x08,0x14,0x42,0x91,0xA1,0xB1,0xC1,0x09,0x23,0x33,0x52,0xF0,
        0x15,0x62,0x72,0xD1,0x0A,0x16,0x24,0x34,0xE1,0x25,0xF1,0x17,0x18,0x19,0x1A,0x26,
        0x27,0x28,0x29,0x2A,0x35,0x36,0x37,0x38,0x39,0x3A,0x43,0x44,0x45,0x46,0x47,0x48,
        0x49,0x4A,0x53,0x54,0x55,0x56,0x57,0x58,0x59,0x5A,0x63,0x64,0x65,0x66,0x67,0x68,
        0x69,0x6A,0x73,0x74,0x75,0x76,0x77,0x78,0x79,0x7A,0x82,0x83,0x84,0x85,0x86,0x87,
        0x88,0x89,0x8A,0x92,0x93,0x94,0x95,0x96,0x97,0x98,0x99,0x9A,0xA2,0xA3,0xA4,0xA5,
        0xA6,0xA7,0xA8,0xA9,0xAA,0xB2,0xB3,0xB4,0xB5,0xB6,0xB7,0xB8,0xB9,0xBA,0xC2,0xC3,
        0xC4,0xC5,0xC6,0xC7,0xC8,0xC9,0xCA,0xD2,0xD3,0xD4,0xD5,0xD6,0xD7,0xD8,0xD9,0xDA,
        0xE2,0xE3,0xE4,0xE5,0xE6,0xE7,0xE8,0xE9,0xEA,0xF2,0xF3,0xF4,0xF5,0xF6,0xF7,0xF8,
        0xF9,0xFA
    ])

# ---------------------- Full default codebooks ---------------------- #
fn build_default_codebooks() -> (List[HuffCode], List[HuffCode], List[HuffCode], List[HuffCode]):
    var dcL_can = build_canonical_codes(bits_dc_luma(),   vals_dc_common())
    var dcC_can = build_canonical_codes(bits_dc_chroma(), vals_dc_common())
    var acL_can = build_canonical_codes(bits_ac_luma(),   vals_ac_luma())
    var acC_can = build_canonical_codes(bits_ac_chroma(), vals_ac_chroma())

    var dcL = expand_dc(dcL_can)
    var dcC = expand_dc(dcC_can)
    var acL = expand_ac(vals_ac_luma(),  acL_can)
    var acC = expand_ac(vals_ac_chroma(), acC_can)

    return (dcL.copy(), dcC.copy(), acL.copy(), acC.copy())

# ------------------------- DC magnitude helper ------------------------- #
@always_inline
fn dc_mag_bits(v: Int) -> (Int, Int):
    if v == 0:
        return (0, 0)
    var a = v
    if a < 0:
        a = -a
    var n = 0
    var t = a
    while t > 0:
        n += 1
        t = t >> 1
    var bits = 0
    if v >= 0:
        bits = a
    else:
        var mask = (1 << n) - 1
        bits = (~a) & mask
    return (n, bits)

# --------------------------- Block encoding ---------------------------- #
fn encode_block(mut bw: BitWriter,
                zz: UnsafePointer[Int],
                is_luma: Bool,
                prev_dc: Int,
                dcL: List[HuffCode], dcC: List[HuffCode],
                acL: List[HuffCode], acC: List[HuffCode]) -> Int:
    # ---- DC ----
    var dc_curr = zz[0]
    var diff = dc_curr - prev_dc
    var (nbits, addbits) = dc_mag_bits(diff)
    var dc_sym = nbits
    if dc_sym < 0: dc_sym = 0
    if dc_sym > 11: dc_sym = 11
    var h = (dcL[dc_sym] if is_luma else dcC[dc_sym])
    _ = bw.write_bits(h.code, h.size)
    if nbits > 0:
        _ = bw.write_bits(addbits, nbits)
    var new_prev = dc_curr

    # ---- AC ----
    var run = 0
    var k = 1
    while k < 64:
        var v = zz[k]
        if v == 0:
            run += 1
        else:
            while run >= 16:
                # ZRL
                var zrl = (acL[0xF0] if is_luma else acC[0xF0])
                _ = bw.write_bits(zrl.code, zrl.size)
                run -= 16
            var a = v
            var sign = 0
            if a < 0:
                sign = 1
                a = -a
            var n = 0
            var t = a
            while t > 0:
                n += 1
                t = t >> 1
            var sym = (run << 4) | n
            var hc = (acL[sym] if is_luma else acC[sym])
            _ = bw.write_bits(hc.code, hc.size)
            var add = 0
            if sign == 0:
                add = a
            else:
                var mask = (1 << n) - 1
                add = (~a) & mask
            _ = bw.write_bits(add, n)
            run = 0
        k += 1

    if run > 0:
        var eob = (acL[0x00] if is_luma else acC[0x00])
        _ = bw.write_bits(eob.code, eob.size)

    return new_prev
