# MIT License
# SPDX-License-Identifier: MIT
# Project: momijo.vision.io
# File: src/momijo/vision/io/encode_png.mojo

from momijo.vision.tensor import Tensor
from momijo.vision.dtypes import DType

# --------------------------- PNG helpers (pure Mojo) ---------------------------

fn _u32_be(x: Int) -> (UInt8, UInt8, UInt8, UInt8):
    var b0 = UInt8((x >> 24) & 0xFF)
    var b1 = UInt8((x >> 16) & 0xFF)
    var b2 = UInt8((x >> 8)  & 0xFF)
    var b3 = UInt8(x & 0xFF)
    return (b0, b1, b2, b3)

fn _append_u32_be(mut out: List[UInt8], x: Int):
    var (a, b, c, d) = _u32_be(x)
    out.append(a); out.append(b); out.append(c); out.append(d)

# CRC-32 (bitwise, poly 0xEDB88320)
fn _crc32_update(crc_in: UInt32, byte: UInt8) -> UInt32:
    var crc = crc_in ^ UInt32(byte)
    var k = 0
    while k < 8:
        var mask = UInt32(0) - UInt32(crc & UInt32(1))
        crc = (crc >> 1) ^ (UInt32(0xEDB88320) & mask)
        k += 1
    return crc

fn _crc32(mut bytes: List[UInt8]) -> UInt32:
    var crc = UInt32(0xFFFFFFFF)
    var i = 0
    while i < bytes.__len__():
        crc = _crc32_update(crc, bytes[i])
        i += 1
    return crc ^ UInt32(0xFFFFFFFF)

# Adler-32 for zlib footer
fn _adler32(mut bytes: List[UInt8]) -> UInt32:
    var s1 = UInt32(1)
    var s2 = UInt32(0)
    var i = 0
    while i < bytes.__len__():
        s1 = (s1 + UInt32(bytes[i])) % UInt32(65521)
        s2 = (s2 + s1) % UInt32(65521)
        i += 1
    return (s2 << 16) | s1

# zlib stream with stored (uncompressed) DEFLATE blocks
fn _zlib_wrap_stored(mut raw: List[UInt8]) -> List[UInt8]:
    var out = List[UInt8]()
    out.append(UInt8(0x78))   # CMF
    out.append(UInt8(0x01))   # FLG for 32K window, stored blocks

    var pos = 0
    var n = raw.__len__()
    while pos < n:
        var remaining = n - pos
        var block_len: Int
        if remaining > 65535:
            block_len = 65535
        else:
            block_len = remaining

        var is_last = (pos + block_len) == n

        var flag: UInt8
        if is_last:
            flag = UInt8(0x01)
        else:
            flag = UInt8(0x00)

        out.append(flag)  # BFINAL + BTYPE=00 

        var len_lo = UInt8(block_len & 0xFF)
        var len_hi = UInt8((block_len >> 8) & 0xFF)
        var nlen = 65535 - block_len
        var nlen_lo = UInt8(nlen & 0xFF)
        var nlen_hi = UInt8((nlen >> 8) & 0xFF)

        out.append(len_lo); out.append(len_hi)
        out.append(nlen_lo); out.append(nlen_hi)

        var i = 0
        while i < block_len:
            out.append(raw[pos + i])
            i += 1

        pos += block_len

    var adler = _adler32(raw)
    out.append(UInt8((adler >> 24) & 0xFF))
    out.append(UInt8((adler >> 16) & 0xFF))
    out.append(UInt8((adler >> 8)  & 0xFF))
    out.append(UInt8(adler & 0xFF))
    return out

# Append a PNG chunk
fn _append_chunk(mut png: List[UInt8], typ0: UInt8, typ1: UInt8, typ2: UInt8, typ3: UInt8, mut data: List[UInt8]):
    _append_u32_be(png, data.__len__())

    var header_and_data = List[UInt8]()
    header_and_data.append(typ0)
    header_and_data.append(typ1)
    header_and_data.append(typ2)
    header_and_data.append(typ3)

    var i = 0
    while i < data.__len__():
        header_and_data.append(data[i])
        i += 1

    png.append(typ0); png.append(typ1); png.append(typ2); png.append(typ3)

    i = 0
    while i < data.__len__():
        png.append(data[i])
        i += 1

    var crc = _crc32(header_and_data)
    _append_u32_be(png, Int(crc))

# ---------------------------- Public encoder API ----------------------------
# Encode PNG from a Tensor with packed HWC UInt8 and channels=1 or 3.
fn encode_png_tensor(t_in: Tensor) -> (Bool, List[UInt8]):
    var c = t_in.channels()
    if not (c == 1 or c == 3):
        return (False, List[UInt8]())
 
    if t_in.dtype() != DType.UInt8():
        return (False, List[UInt8]())

    var w = t_in.width()
    var h = t_in.height()

    var s0 = t_in.stride0()
    var s1 = t_in.stride1()
    var s2 = t_in.stride2()
    var ptr = t_in.data()  # UnsafePointer[UInt8]

    var row_bytes = w * c
    var scan = List[UInt8]()
    var y = 0
    while y < h:
        scan.append(UInt8(0))  # filter 0
        var x = 0
        while x < w:
            var ch = 0
            while ch < c:
                scan.append(ptr[y * s0 + x * s1 + ch * s2])
                ch += 1
            x += 1
        y += 1

    var z = _zlib_wrap_stored(scan)

    var png = List[UInt8]()
    # Signature
    png.append(UInt8(137)); png.append(UInt8(80));  png.append(UInt8(78));  png.append(UInt8(71))
    png.append(UInt8(13));  png.append(UInt8(10));  png.append(UInt8(26));  png.append(UInt8(10))

    # IHDR
    var ihdr = List[UInt8]()
    _append_u32_be(ihdr, w)
    _append_u32_be(ihdr, h)
    ihdr.append(UInt8(8))  # bit depth

    var ct: UInt8 = UInt8(0)
    if c == 1:
        ct = UInt8(0)   # grayscale
    else:
        ct = UInt8(2)   # truecolor (RGB)
    ihdr.append(ct)

    ihdr.append(UInt8(0))  # compression
    ihdr.append(UInt8(0))  # filter
    ihdr.append(UInt8(0))  # interlace
    _append_chunk(png, UInt8(73), UInt8(72), UInt8(68), UInt8(82), ihdr) # "IHDR"

    # IDAT
    _append_chunk(png, UInt8(73), UInt8(68), UInt8(65), UInt8(84), z)     # "IDAT"

    # IEND
    var empty = List[UInt8]()
    _append_chunk(png, UInt8(73), UInt8(69), UInt8(78), UInt8(68), empty) # "IEND"

    return (True, png)
