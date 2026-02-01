# MIT License
# SPDX-License-Identifier: MIT
# Project: momijo.vision.io
# File: src/momijo/vision/io/png_decoder.mojo
# Description: Minimal PNG chunk/header parsing (no inflate)

from math.bit_ops import lshift, rshift, bitand
 
# -----------------------------------------------------------------------------
# Helper: read big-endian u32 from byte pointer
# -----------------------------------------------------------------------------
fn read_be_u32(ptr: UnsafePointer[UInt8]) -> UInt32:
    return (UInt32(ptr[0]) << 24) | (UInt32(ptr[1]) << 16) | (UInt32(ptr[2]) << 8) | UInt32(ptr[3])

# -----------------------------------------------------------------------------
# Helper: PNG magic header check (no global const needed)
# -----------------------------------------------------------------------------
fn is_png_magic(ptr: UnsafePointer[UInt8]) -> Bool:
    # 8-byte PNG signature: 137 80 78 71 13 10 26 10
    if ptr[0] != UInt8(137): return False
    if ptr[1] != UInt8(80):  return False
    if ptr[2] != UInt8(78):  return False
    if ptr[3] != UInt8(71):  return False
    if ptr[4] != UInt8(13):  return False
    if ptr[5] != UInt8(10):  return False
    if ptr[6] != UInt8(26):  return False
    if ptr[7] != UInt8(10):  return False
    return True

# -----------------------------------------------------------------------------
# PNG Chunk header
# -----------------------------------------------------------------------------
struct PNGChunk:
    var length: UInt32
    var chunk_type: (UInt8, UInt8, UInt8, UInt8)
    var data_ptr: UnsafePointer[UInt8]
    var crc: UInt32

# -----------------------------------------------------------------------------
# Parse PNGChunk (no CRC verification here)
# -----------------------------------------------------------------------------
fn parse_chunk(ptr: UnsafePointer[UInt8]) -> PNGChunk:
    var length = read_be_u32(ptr)
    var chunk_type = (ptr[4], ptr[5], ptr[6], ptr[7])
    var data_ptr = ptr + 8
    var crc = read_be_u32(ptr + 8 + Int(length))
    return PNGChunk(length, chunk_type, data_ptr, crc)

# -----------------------------------------------------------------------------
# PNG IHDR Header
# -----------------------------------------------------------------------------
struct PNGHeader:
    var width: Int
    var height: Int
    var bit_depth: UInt8
    var color_type: UInt8
    var compression: UInt8
    var filter: UInt8
    var interlace: UInt8

    # Explicit initializer (fieldwise), per project "var-only / no-globals" rules
    fn __init__(out self,
                width: Int,
                height: Int,
                bit_depth: UInt8,
                color_type: UInt8,
                compression: UInt8,
                filter: UInt8,
                interlace: UInt8):
        self.width = width
        self.height = height
        self.bit_depth = bit_depth
        self.color_type = color_type
        self.compression = compression
        self.filter = filter
        self.interlace = interlace

fn parse_ihdr(data: UnsafePointer[UInt8]) -> PNGHeader:
    return PNGHeader(
        Int(read_be_u32(data)),
        Int(read_be_u32(data + 4)),
        data[8],
        data[9],
        data[10],
        data[11],
        data[12]
    )
 

# -----------------------------------------------------------------------------
# Header-only reader (no zlib/inflate)
# -----------------------------------------------------------------------------
fn read_png_header_only(file_data: UnsafePointer[UInt8], length: Int) -> (Bool, PNGHeader):
    if length < 8:
        return (False, PNGHeader(0, 0, UInt8(0), UInt8(0), UInt8(0), UInt8(0), UInt8(0)))
    if not is_png_magic(file_data):
        return (False, PNGHeader(0, 0, UInt8(0), UInt8(0), UInt8(0), UInt8(0), UInt8(0)))

    var end_ptr = file_data + length
    var cursor = file_data + 8

    while (cursor + 12) <= end_ptr:
        var chunk = parse_chunk(cursor)
        var type_str = String.utf8_from_array([chunk.chunk_type[0], chunk.chunk_type[1], chunk.chunk_type[2], chunk.chunk_type[3]])

        if type_str == "IHDR":
            var header = parse_ihdr(chunk.data_ptr)
            return (True, header)
        elif type_str == "IEND":
            break

        var advance = 12 + Int(chunk.length)
        cursor = cursor + advance

    return (False, PNGHeader(0, 0, UInt8(0), UInt8(0), UInt8(0), UInt8(0), UInt8(0)))

