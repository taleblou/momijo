# MIT License
# SPDX-License-Identifier: MIT
# Project: momijo.vision.io
# File: src/momijo/vision/io/inflate.mojo
# Description: Minimal zlib/deflate inflater supporting only "stored" (BTYPE=0) blocks.

# -----------------------------------------------------------------------------
# Helpers: bit reader over a byte buffer
# -----------------------------------------------------------------------------

struct BitReader:
    var base: UnsafePointer[UInt8]
    var size: Int       # total input bytes
    var byte_pos: Int   # current byte index
    var bit_buf: Int    # bit accumulator (LSB-first)
    var bit_cnt: Int    # number of valid bits in bit_buf

    # Default initializer to satisfy struct construction (project: var-only, no globals)
    fn __init__(out self):
        self.base = UnsafePointer[UInt8].alloc(1)
        self.size = 0
        self.byte_pos = 0
        self.bit_buf = 0
        self.bit_cnt = 0

fn br_init(mut br: BitReader, p: UnsafePointer[UInt8], n: Int):
    br.base = p
    br.size = n
    br.byte_pos = 0
    br.bit_buf = 0
    br.bit_cnt = 0

fn br_need_bits(mut br: BitReader, nbits: Int) -> Bool:
    # Ensure we have at least nbits in bit_buf
    while br.bit_cnt < nbits:
        if br.byte_pos >= br.size:
            return False
        # Append next byte (LSB-first)
        var b = Int(br.base[br.byte_pos])
        br.bit_buf = br.bit_buf | (b << br.bit_cnt)
        br.bit_cnt = br.bit_cnt + 8
        br.byte_pos = br.byte_pos + 1
    return True

fn br_get_bits(mut br: BitReader, nbits: Int) -> Int:
    # Assumes br_need_bits succeeded
    var val = br.bit_buf & ((1 << nbits) - 1)
    br.bit_buf = br.bit_buf >> nbits
    br.bit_cnt = br.bit_cnt - nbits
    return val

fn br_align_to_byte(mut br: BitReader) -> Bool:
    var drop = br.bit_cnt % 8
    if drop != 0:
        if not br_need_bits(br, drop):
            return False
        _ = br_get_bits(br, drop)
    return True

# -----------------------------------------------------------------------------
# zlib header helpers
# -----------------------------------------------------------------------------

fn _zlib_header_ok(cmf: Int, flg: Int) -> Bool:
    # CMF: bits 0..3 = compression method (must be 8 for deflate)
    # FLG: check (CMF*256 + FLG) % 31 == 0
    if (cmf & 0x0F) != 8:
        return False
    var chk = (cmf << 8) + flg
    return (chk % 31) == 0

# -----------------------------------------------------------------------------
# Public API: inflate only "stored" (BTYPE=0) DEFLATE blocks inside zlib wrapper
# Returns: number of bytes written to dst, or <= 0 on error.
# -----------------------------------------------------------------------------

fn inflate(src: UnsafePointer[UInt8], src_len: Int,
           dst: UnsafePointer[UInt8], dst_cap: Int) -> Int:
    # Basic guards
    if src_len < 2:
        return -1

    # Read zlib header
    var cmf = Int(src[0])
    var flg = Int(src[1])
    if not _zlib_header_ok(cmf, flg):
        return -2

    var pos = 2
    # Check preset dictionary flag (FLG bit 5)
    if (flg & 0x20) != 0:
        # Must have 4-byte DICTID
        if src_len - pos < 4:
            return -3
        pos = pos + 4  # skip DICTID

    if pos >= src_len:
        return -4

    # Set up bit reader starting at 'pos'
    var br = BitReader()
    br_init(br, src + pos, src_len - pos)

    var total_out = 0
    var last_block = False

    # Process DEFLATE blocks
    while True:
        # Read BFINAL (1 bit) + BTYPE (2 bits)
        if not br_need_bits(br, 3):
            return -5
        var bfinal = br_get_bits(br, 1)
        var btype = br_get_bits(br, 2)
        last_block = (bfinal != 0)

        if btype == 0:
            # "stored" block: align to byte, then LEN/NLEN (little-endian), then raw bytes
            if not br_align_to_byte(br):
                return -6

            # Now the stream is byte-aligned; need 4 bytes for LEN/NLEN
            if (br.byte_pos + 4) > br.size:
                return -7

            var p = br.base + br.byte_pos
            var len_le = Int(p[0]) | (Int(p[1]) << 8)
            var nlen_le = Int(p[2]) | (Int(p[3]) << 8)

            # Check one's complement
            if ((len_le ^ 0xFFFF) & 0xFFFF) != (nlen_le & 0xFFFF):
                return -8
            br.byte_pos = br.byte_pos + 4

            # Copy len_le bytes as-is
            if (br.byte_pos + len_le) > br.size:
                return -9
            if (total_out + len_le) > dst_cap:
                return -10

            var i = 0
            var src_ptr = br.base + br.byte_pos
            while i < len_le:
                dst[total_out + i] = src_ptr[i]
                i = i + 1

            br.byte_pos = br.byte_pos + len_le
            total_out = total_out + len_le

        elif btype == 1 or btype == 2:
            # Fixed/Dynamic Huffman: not implemented in this minimal inflater
            return -11
        else:
            # Reserved (error)
            return -12

        if last_block:
            break

    # Adler-32 is not validated in this minimal implementation.
    return total_out
