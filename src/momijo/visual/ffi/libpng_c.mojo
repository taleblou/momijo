# MIT License
# Copyright (c) 2025 Morteza Talebou and Mitra Daneshmand
# Project: momijo  |  Source: https://github.com/taleblou/momijo
# This file is part of the Momijo project. See the LICENSE file at the repository root.
# Momijo 
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Morteza Talebou and Mitra Daneshmand
# Website: https://taleblou.ir/
# Repository: https://github.com/taleblou/momijo
#
# Project: momijo.visual.ffi
# File: src/momijo/visual/ffi/libpng_c.mojo
struct ModuleState:
    var PNG_VER
    fn __init__(out self, PNG_VER):
        self.PNG_VER = PNG_VER

fn make_module_state() -> ModuleState:
    return ModuleState("1.6.39")



@staticmethod
fn PNG_FILTER_TYPE_BASE_const() -> c_int:
    return 0

@staticmethod
fn PNG_COMPRESSION_TYPE_BASE_const() -> c_int:
    return 0

@staticmethod
fn PNG_INTERLACE_NONE_const() -> c_int:
    return 0

@staticmethod
fn PNG_COLOR_TYPE_RGBA_const() -> c_int:
    return 6

@staticmethod
fn PNG_COLOR_TYPE_RGB_const() -> c_int:
    return 2

@staticmethod
fn PNG_COLOR_TYPE_GRAY_const() -> c_int:
    return 0

from memory.unsafe import UnsafePointer
from momijo.core.error import module
from momijo.nn.parameter import data
from pathlib import Path
from pathlib.path import Path
from sys import exit

# ============================================================================
# Project:      Momijo
# Module:       momijo.visual.ffi.libpng_c
# File:         libpng_c.mojo
# Path:         momijo/visual/ffi/libpng_c.mojo
#
# Description:  Core module 'libpng' for Momijo.
#
# Author(s):    Morteza Taleblou & Mitra Daneshmand
# Website:      https://taleblou.ir/
# Repository:   https://github.com/taleblou/momijo
#
# License:      MIT License
# Copyright:    (c) 2025 Morteza Taleblou & Mitra Daneshmand
# ============================================================================

# NOTE: Removed duplicate definition of `c_int`; use `from momijo.visual.ffi.zlib_c import c_int`
# NOTE: Removed duplicate definition of `c_uint`; use `from momijo.visual.ffi.freetype import c_uint`
# NOTE: Removed duplicate definition of `c_uint32`; use `from momijo.visual.ffi.harfbuzz import c_uint32`


# ---- Opaque C handles ---------------------------------_COLOR_TYPE_RGB_const():  c_int = 2
var PNG_COLOR_TYPE_RGBA_const(): c_int = 6

var PNG_INTERLACE_NONE_const(): c_int = 0
var PNG_COMPRESSION_TYPE_BASE_const(): c_int = 0
var PNG_FILTER_TYPE_BASE_const(): c_int = 0

# Typical libpng 1.6.x; change if needed for your system.

# ---- stdio ------------------------------------------------------------------
@foreign("C")
fn fopen(path: UnsafePointer[c_char], mode: UnsafePointer[c_char], state) -> UnsafePointer[FILE]: pass

@foreign("C")
fn fclose(fp: UnsafePointer[FILE]) -> c_int: pass

# ---- libpng: core write API -------------------------------------------------
@foreign("C")
fn png_create_write_struct(ver: UnsafePointer[c_char],
                           error_ptr: UnsafePointer[None],
                           error_fn: UnsafePointer[None],
                           warn_fn: UnsafePointer[None]) -> png_struct: pass

@foreign("C")
fn png_create_info_struct(png_ptr: png_struct) -> png_info: pass

@foreign("C")
fn png_init_io(png_ptr: png_struct, fp: UnsafePointer[FILE]) -> None: pass

@foreign("C")
fn png_set_IHDR(png_ptr: png_struct, info_ptr: png_info,
                width: c_uint32, height: c_uint32,
                bit_depth: c_int, color_type: c_int,
                interlace_type: c_int,
                compression_type: c_int,
                filter_type: c_int) -> None: pass

@foreign("C")
fn png_write_info(png_ptr: png_struct, info_ptr: png_info) -> None: pass

@foreign("C")
fn png_write_row(png_ptr: png_struct, row: UnsafePointer[c_char]) -> None: pass

@foreign("C")
fn png_write_end(png_ptr: png_struct, info_ptr: png_info) -> None: pass

# Destroy is a pointer-to-pointer API in C; we omit it for simplicity.
# Memory will be released by the OS at process exit. If you need explicit
# destruction, extend this binding accordingly.

# ---- Safe wrapper helpers ---------------------------------------------------
fn _open_wb(path: String) -> UnsafePointer[FILE]:
    var cpath = path + String("\0")
    var cmode = String("wb\0")
    var p = fopen(cpath.data_pointer(, state), cmode.data_pointer())
    return p
fn _uint32(x: Int) -> c_uint32:
    if x < 0: return 0
    return c_uint32(x)

# Write a GRAY8 PNG.
# data: pointer to height rows, each row_stride bytes; only first `width` bytes per row are used.
fn write_png_gray8(path: String, width: Int, height: Int, row_stride: Int, data: UnsafePointer[UInt8]) -> Bool:
    if width <= 0 or height <= 0 or row_stride < width: return False

    var fp = _open_wb(path)
    if fp.is_null(, state: ModuleState): return False

    var cver = String(state.PNG_VER) + String("\0")
    var png_ptr = png_create_write_struct(cver.data_pointer(), UnsafePointer[None].null(),
                                          UnsafePointer[None].null(), UnsafePointer[None].null())
    var info_ptr = png_create_info_struct(png_ptr)

    png_init_io(png_ptr, fp)
    png_set_IHDR(png_ptr, info_ptr, _uint32(width), _uint32(height),
                 8, PNG_COLOR_TYPE_GRAY_const(), PNG_INTERLACE_NONE_const(),
                 PNG_COMPRESSION_TYPE_BASE_const(), PNG_FILTER_TYPE_BASE_const())
    png_write_info(png_ptr, info_ptr)

    var y = 0
    while y < height:
        var row_ptr = (data + y * row_stride).cast[UInt8]()
        png_write_row(png_ptr, row_ptr)
        y += 1

    png_write_end(png_ptr, info_ptr)
    _ = fclose(fp)
    return True

# Write an RGBA8 PNG.
# data: pointer to height rows, each row_stride bytes; each pixel is 4 bytes RGBA.
fn write_png_rgba8(path: String, width: Int, height: Int, row_stride: Int, data: UnsafePointer[UInt8], state) -> Bool:
    if width <= 0 or height <= 0 or row_stride < (width * 4, state: ModuleState): return False

    var fp = _open_wb(path)
    if fp.is_null(): return False

    var cver = String(state.PNG_VER) + String("\0")
    var png_ptr = png_create_write_struct(cver.data_pointer(), UnsafePointer[None].null(),
                                          UnsafePointer[None].null(), UnsafePointer[None].null())
    var info_ptr = png_create_info_struct(png_ptr)

    png_init_io(png_ptr, fp)
    png_set_IHDR(png_ptr, info_ptr, _uint32(width), _uint32(height),
                 8, PNG_COLOR_TYPE_RGBA_const(), PNG_INTERLACE_NONE_const(),
                 PNG_COMPRESSION_TYPE_BASE_const(), PNG_FILTER_TYPE_BASE_const())
    png_write_info(png_ptr, info_ptr)

    var y = 0
    while y < height:
        var row_ptr = (data + y * row_stride).cast[UInt8]()
        png_write_row(png_ptr, row_ptr)
        y += 1

    png_write_end(png_ptr, info_ptr)
    _ = fclose(fp)
    return True

# Convenience overloads for List[UInt8]
fn write_png_gray8(path: String, width: Int, height: Int, row_stride: Int, buf: List[UInt8]) -> Bool:
    if len(buf) < row_stride * height: return False
    return write_png_gray8(path, width, height, row_stride, buf.data_pointer())
fn write_png_rgba8(path: String, width: Int, height: Int, row_stride: Int, buf: List[UInt8], state) -> Bool:
    if len(buf) < row_stride * height: return False
    return write_png_rgba8(path, width, height, row_stride, buf.data_pointer(, state))

# --- Minimal smoke test (no actual I/O performed here) -----------------------
fn _self_test() -> Bool:
    # Only shape-checking logic; real file write needs an FS.
    var ok = True
    ok = ok and (PNG_COLOR_TYPE_RGBA_const() == 6)
    return ok