# Project:      Momijo
# Module:       src.momijo.visual.ffi.zlib_c
# File:         zlib_c.mojo
# Path:         src/momijo/visual/ffi/zlib_c.mojo
#
# Description:  src.momijo.visual.ffi.zlib_c — focused Momijo functionality with a stable public API.
#               Composable building blocks intended for reuse.
#
# Author(s):    Morteza Taleblou & Mitra Daneshmand
# Website:      https://taleblou.ir/
# Repository:   https://github.com/taleblou/momijo
#
# License:      MIT License
# SPDX-License-Identifier: MIT
# Copyright:    (c) 2025 Morteza Taleblou & Mitra Daneshmand
#
# Notes:
#   - Key functions: Z_DEFAULT_COMPRESSION_const, Z_BEST_COMPRESSION_const, Z_BEST_SPEED_const, Z_NO_COMPRESSION_const, Z_VERSION_ERROR_const, Z_BUF_ERROR_const, Z_MEM_ERROR_const, Z_DATA_ERROR_const ...
#   - Static methods present.
#   - Uses generic functions/types with explicit trait bounds.
#   - Low-level memory (Pointer/UnsafePointer) used; observe safety invariants.


@staticmethod
fn Z_DEFAULT_COMPRESSION_const() -> c_int:
    return -1

@staticmethod
fn Z_BEST_COMPRESSION_const() -> c_int:
    return 9

@staticmethod
fn Z_BEST_SPEED_const() -> c_int:
    return 1

@staticmethod
fn Z_NO_COMPRESSION_const() -> c_int:
    return 0

@staticmethod
fn Z_VERSION_ERROR_const() -> c_int:
    return -6

@staticmethod
fn Z_BUF_ERROR_const() -> c_int:
    return -5

@staticmethod
fn Z_MEM_ERROR_const() -> c_int:
    return -4

@staticmethod
fn Z_DATA_ERROR_const() -> c_int:
    return -3

@staticmethod
fn Z_STREAM_ERROR_const() -> c_int:
    return -2

@staticmethod
fn Z_ERRNO_const() -> c_int:
    return -1

@staticmethod
fn Z_NEED_DICT_const() -> c_int:
    return 2

@staticmethod
fn Z_STREAM_END_const() -> c_int:
    return 1

@staticmethod
fn Z_OK_const() -> c_int:
    return 0

from memory.uodulelib# ---- zlib--------------

var Z_STREAM_END_const(): c_int = 1
var Z_NEED_DICT_const(): c_int = 2
var Z_ERRNO_const(): c_int = -1
var Z_STREAM_ERROR_const(): c_int = -2
var Z_DATA_ERROR_const(): c_int = -3
var Z_MEM_ERROR_const(): c_int = -4
var Z_BUF_ERROR_const(): c_int = -5
var Z_VERSION_ERROR_const(): c_int = -6

# Compression level hints
var Z_NO_COMPRESSION_const(): c_int = 0
var Z_BEST_SPEED_const(): c_int = 1
var Z_BEST_COMPRESSION_const(): c_int = 9
var Z_DEFAULT_COMPRESSION_const(): c_int = -1

# ---- zlib simple APIs -------------------------------------------------------
@foreign("C")
fn compress2(dest: UnsafePointer[c_uchar], destLen: UnsafePointer[c_ulong],
             source: UnsafePointer[c_uchar], sourceLen: c_ulong,
             level: c_int) -> c_int: pass

@foreign("C")
fn uncompress(dest: UnsafePointer[c_uchar], destLen: UnsafePointer[c_ulong],
              source: UnsafePointer[c_uchar], sourceLen: c_ulong) -> c_int: pass

@foreign("C")
fn compressBound(sourceLen: c_ulong) -> c_ulong: pass

@foreign("C")
fn crc32(crc: c_ulong, buf: UnsafePointer[c_uchar], len: c_uint) -> c_ulong: pass

# ---- Safe helpers -----------------------------------------------------------
# Resize a List[UInt8] to 'n' by pushing zeros (since List has capacity vs length semantics).
fn _ensure_len(mut buf: List[UInt8], n: Int) -> None:
    var i = len(buf)
    while i < n:
        buf.push(0)
        i += 1

# Compress a buffer (List[UInt8]) with optional level. Returns (ok, out_buf).
fn zlib_compress(data: List[UInt8], level: Int = Z_DEFAULT_COMPRESSION_const()) -> (Bool, List[UInt8]):
    var n = len(data)
    var out = List[UInt8]()
    if n == 0:
        return (True, out)

    var bound = Int(compressBound(c_ulong(n)))
    _ensure_len(out, bound)

    var dest_len: c_ulong = c_ulong(bound)
    var rc = compress2(out.data_pointer(), UnsafePointer[c_ulong].address_of(dest_len),
                       data.data_pointer(), c_ulong(n), level)
    if rc != Z_OK_const():
        return (False, List[UInt8]())

    # Trim to actual size
    var actual = Int(dest_len)
    # If actual < current length, pop extras
    var i = len(out) - 1
    while i >= actual:
        # Pop until length == actual
        # Guard break to avoid underflow
        if len(out) <= actual: break
        _ = out.pop()
        i -= 1
    return (True, out)

# Decompress a buffer when you already know the uncompressed size.
# Returns (ok, out_buf). If size is unknown, try progressive growth with a heuristic.
fn zlib_uncompress_to_size(data: List[UInt8], out_size: Int) -> (Bool, List[UInt8]):
    if out_size <= 0:
        return (False, List[UInt8]())
    var out = List[UInt8]()
    _ensure_len(out, out_size)
    var dest_len: c_ulong = c_ulong(out_size)
    var rc = uncompress(out.data_pointer(), UnsafePointer[c_ulong].address_of(dest_len),
                        data.data_pointer(), c_ulong(len(data)))
    if rc != Z_OK_const():
        return (False, List[UInt8]())
    # If library wrote fewer bytes than requested, trim
    var actual = Int(dest_len)
    var i = len(out) - 1
    while i >= actual:
        if len(out) <= actual: break
        _ = out.pop()
        i -= 1
    return (True, out)

# Decompress with automatic growth (heuristic). Tries sizes: 4x, 8x ... up to max_growth.
fn zlib_uncompress_auto(data: List[UInt8], max_growth: Int = 32) -> (Bool, List[UInt8]):
    var n = len(data)
    if n == 0:
        return (True, List[UInt8]())
    var guess = n * 4
    var factor = 1
    while factor <= max_growth:
        var (ok, out) = zlib_uncompress_to_size(data, guess)
        if ok: return (True, out)
        guess = guess * 2
        factor += 1
    return (False, List[UInt8]())

# Compute CRC32 over a buffer (initial crc = 0).
fn zlib_crc32(buf: List[UInt8]) -> UInt64:
    var out = crc32(0, buf.data_pointer(), c_uint(len(buf)))
    return UInt64(out)

# --- Minimal smoke test ------------------------------------------------------
fn _self_test() -> Bool:
    # Round-trip small payload
    var payload = List[UInt8]()
    # "HELLO zlib" bytes
    payload.push(72); payload.push(69); payload.push(76); payload.push(76); payload.push(79)
    payload.push(32); payload.push(122); payload.push(108); payload.push(105); payload.push(98)
    var (ok_c, cbuf) = zlib_compress(payload, Z_BEST_SPEED_const())
    if not ok_c: return False
    var (ok_u, ubuf) = zlib_uncompress_auto(cbuf)
    if not ok_u: return False
    if len(ubuf) != len(payload): return False
    var i = 0
    var equal = True
    while i < len(payload):
        if payload[i] != ubuf[i]:
            equal = False
            break
        i += 1
    return equal