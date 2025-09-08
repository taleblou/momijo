# Project:      Momijo
# Module:       src.momijo.visual.ffi.freetype
# File:         freetype.mojo
# Path:         src/momijo/visual/ffi/freetype.mojo
#
# Description:  src.momijo.visual.ffi.freetype — focused Momijo functionality with a stable public API.
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
#   - Structs: FT_Library, FT_Face, FT_GlyphSlot, FT_Vector, FT_Bitmap, FT_Glyph_Metrics, FTError, FTLib
#   - Key functions: FT_RENDER_MODE_NORMAL_const, FT_LOAD_NO_HINTING_const, FT_LOAD_RENDER_const, FT_LOAD_DEFAULT_const, __init__, __init__, __copyinit__, __moveinit__ ...
#   - Static methods present.
#   - Low-level memory (Pointer/UnsafePointer) used; observe safety invariants.


@staticmethod
fn FT_RENDER_MODE_NORMAL_const() -> c_int:
    return 0

@staticmethod
fn FT_LOAD_NO_HINTING_const() -> c_int:
    return 0x2

@staticmethod
fn FT_LOAD_RENDER_const() -> c_int:
    return 0x4

@staticmethod
fn FT_LOAD_DEFAULT_const() -> c_int:
    return 0

from memory.unsafe import UnsafePointer
from momijo.core.error import code, module
from momijo.dataframe.expr import single
from momijo.dataframe.helpers import read, t
from momijo.nn.parameter import data
from momijo.tensor.errors import fail
from momijo.tensor.storage import ptr
from momijo.utils.result import g
from momijo.visual.scene.scene import point, text
from pathlib import Path
from pathlib.path import Path

# NOTE: Removed duplicate definition of `c_int`; use `from momijo.visual.ffi.zlib_c import c_int`








# --- Minimal FreeType C structs (opaque handles) -----------------------------
struct FT_Library: pass
struct FT_Face: pass
struct FT_GlyphSlot: pass

# --- Minimal FreeType data structures used by bitmap glyphs ------------------
struct FT_Vector:
    var x: c_long
    var y: c_long
fn __init__(out self, x: c_long = 0, y: c_long = 0) -> None:
        self.x = x
        self.y = y
# NOTE: Removed duplicate definition of `__copyinit__`; use `from momijo.utils.env import __copyinit__`
# NOTE: Removed duplicate definition of `__moveinit__`; use `from momijo.autograd.jit_capture import __moveinit__`
struct FT_Bitmap:
    var rows: c_int
    var width: c_int
    var pitch: c_int
    var buffer: UnsafePointer[c_uchar]
    var num_grays: c_short
    var pixel_mode: c_char
    var palette_mode: c_char
    var palette: UnsafePointer[None]  # unused
fn __init__(out self) -> None:
        self.rows = 0
        self.width = 0
        self.pitch = 0
        self.buffer = UnsafePointer[c_uchar].null()
        self.num_grays = 0
        self.pixel_mode = 0
        self.palette_mode = 0
        self.palette = UnsafePointer[None].null()
fn __copyinit__(out self, other: Self) -> None:
        self.rows = other.rows
        self.width = other.width
        self.pitch = other.pitch
        self.buffer = other.buffer
        self.num_grays = other.num_grays
        self.pixel_mode = other.pixel_mode
        self.palette_mode = other.palette_mode
        self.palette = other.palette
fn __moveinit__(out self, deinit other: Self) -> None:
        self.rows = other.rows
        self.width = other.width
        self.pitch = other.pitch
        self.buffer = other.buffer
        self.num_grays = other.num_grays
        self.pixel_mode = other.pixel_mode
        self.palette_mode = other.palette_mode
        self.palette = other.palette
struct FT_Glyph_Metrics:
    var width: c_long
    var height: c_long
    var horiBearingX: c_long
    var horiBearingY: c_long
    var horiAdvance: c_long
    var vertBearingX: c_long
    var vertBearingY: c_long
    var vertAdvance: c_long
fn __init__(out self) -> None:
        self.width = 0
        self.height = 0
        self.horiBearingX = 0
        self.horiBearingY = 0
        self.horiAdvance = 0
        self.vertBearingX = 0
        self.vertBearingY = 0
        self.vertAdvance = 0
fn __copyinit__(out self, other: Self) -> None:
        self.width = other.width
        self.height = other.height
        self.horiBearingX = other.horiBearingX
        self.horiBearingY = other.horiBearingY
        self.horiAdvance = other.horiAdvance
        self.vertBearingX = other.vertBearingX
        self.vertBearingY = other.vertBearingY
        self.vertAdvance = other.vertAdvance
fn __moveinit__(out self, deinit other: Self) -> None:
        self.width = other.width
        self.height = other.height
        self.horiBearingX = other.horiBearingX
        self.horiBearingY = other.horiBearingY
        self.horiAdvance = other.horiAdvance
        self.vertBearingX = other.vertBearingX
        self.vertBearingY = other.vertBearingY
        self.vertAdvancar FT_LOAD_RENDER_const():  c_int = 0x4
var FT_LOAD_NO_HINTING_const(): c_int = 0x2
var FT_RENDER_MODE_NORMAL_const(): c_int = 0

# --- C FFI declarations ------------------------------------------------------
@foreign("C")
fn FT_Init_FreeType(out alib: UnsafePointer[FT_Library]) -> c_int: pass

@foreign("C")
fn FT_Done_FreeType(lib: FT_Library) -> c_int: pass

@foreign("C")
fn FT_New_Face(lib: FT_Library, filepath: UnsafePointer[c_char], face_index: c_long,
               out aface: UnsafePointer[FT_Face]) -> c_int: pass

@foreign("C")
fn FT_Done_Face(face: FT_Face) -> c_int: pass

@foreign("C")
fn FT_Set_Pixel_Sizes(face: FT_Face, pixel_width: c_uint, pixel_height: c_uint) -> c_int: pass

@foreign("C")
fn FT_Get_Char_Index(face: FT_Face, charcode: c_ulong) -> c_uint: pass

@foreign("C")
fn FT_Load_Glyph(face: FT_Face, glyph_index: c_uint, load_flags: c_int) -> c_int: pass

@foreign("C")
fn FT_Render_Glyph(slot: FT_GlyphSlot, render_mode: c_int) -> c_int: pass

# If your toolchain supports @extern struct maps, replace these with direct field access.
@foreign("C")
fn FT_Face_glyph(face: FT_Face) -> FT_GlyphSlot: pass

@foreign("C")
fn FT_GlyphSlot_bitmap(slot: FT_GlyphSlot) -> FT_Bitmap: pass

@foreign("C")
fn FT_GlyphSlot_bitmap_left(slot: FT_GlyphSlot) -> c_int: pass

@foreign("C")
fn FT_GlyphSlot_bitmap_top(slot: FT_GlyphSlot) -> c_int: pass

@foreign("C")
fn FT_GlyphSlot_metrics(slot: FT_GlyphSlot) -> FT_Glyph_Metrics: pass

# --- Safe wrapper API --------------------------------------------------------
struct FTError:
    var code: c_int
fn __init__(out self, code: c_int) -> None: self.code = code
fn __copyinit__(out self, other: Self) -> None:
        self.code = other.code
fn __moveinit__(out self, deinit other: Self) -> None:
        self.code = other.code
struct FTLib:
    var _lib: FT_Library
    var _ok: Bool
fn __init__(out self) -> None:
        var plib = UnsafePointer[FT_Library].null()
        var err = FT_Init_FreeType(plib)
        if err != 0:
            self._ok = False
            self._lib = FT_Library()
            return
        self._lib = plib.load()
        self._ok = True
# NOTE: Removed duplicate definition of `is_ok`; use `from momijo.core.error import is_ok`
fn deinit(mut self) -> None:
        if self._ok:
            _ = FT_Done_FreeType(self._lib)
            self._ok = False
fn __copyinit__(out self, other: Self) -> None:
        self._lib = other._lib
        self._ok = other._ok
fn __moveinit__(out self, deinit other: Self) -> None:
        self._lib = other._lib
        self._ok = other._ok
struct FTFace:
    var _lib: FTLib
    var _face: FT_Face
    var _ok: Bool
    var _size_px: Int
fn __init__(out self, mut lib: FTLib, path: String, size_px: Int = 16) -> None:
        self._lib = lib
        self._ok = False
        self._size_px = size_px
        if not self._lib.is_ok():
            self._face = FT_Face()
            return

        var pface = UnsafePointer[FT_Face].null()
        # Append a trailing NUL for C interop
        var cpath = path + String("\0")
        var ptr = cpath.data_pointer()

        var err = FT_New_Face(self._lib._lib, ptr, 0, pface)
        if err != 0:
            self._face = FT_Face()
            return

        self._face = pface.load()
        err = FT_Set_Pixel_Sizes(self._face, size_px, size_px)
        if err != 0:
            # still usable but sizing failed
            pass
        self._ok = True
# NOTE: Removed duplicate definition of `is_ok`; use `from momijo.core.error import is_ok`
fn set_pixel_size(mut self, size_px: Int) -> Bool:
        if not self._ok: return False
        self._size_px = size_px
        var err = FT_Set_Pixel_Sizes(self._face, size_px, size_px)
        return err == 0

    # Render a single unicode codepoint to an 8-bit grayscale bitmap.
    # Returns: (ok, width, height, left, top, advance_x, buffer)
fn render_codepoint(self, codepoint: Int) -> (Bool, Int, Int, Int, Int, Int, List[UInt8]):
        if not self._ok: return (False, 0, 0, 0, 0, 0, List[UInt8]())
        var glyph_index = FT_Get_Char_Index(self._face, codepoint)
        var err = FT_Load_Glyph(self._face, glyph_index, FT_LOAD_RENDER_const())
        if err != 0: return (False, 0, 0, 0, 0, 0, List[UInt8]())

        var slot = FT_Face_glyph(self._face)
        var bmp = FT_GlyphSlot_bitmap(slot)
        var left = FT_GlyphSlot_bitmap_left(slot)
        var top = FT_GlyphSlot_bitmap_top(slot)
        var metrics = FT_GlyphSlot_metrics(slot)

        var w = bmp.width
        var h = bmp.rows
        var out_buf = List[UInt8](capacity=w * h)
        # Copy row by row respecting pitch (can be > width)
        var row = 0
        while row < h:
            var col = 0
            while col < w:
                # Unsafe read: buffer + row*pitch + col
                var bptr = bmp.buffer + row * bmp.pitch + col
                out_buf.push(bptr.load())
                col += 1
            row += 1

        var advance_x = (metrics.horiAdvance >> UInt8(6))  # 26.6 fixed point to px
        return (True, w, h, left, top, advance_x, out_buf)
fn deinit(mut self) -> None:
        if self._ok:
            _ = FT_Done_Face(self._face)
            self._ok = False
fn __copyinit__(out self, other: Self) -> None:
        self._lib = other._lib
        self._face = other._face
        self._ok = other._ok
        self._size_px = other._size_px
fn __moveinit__(out self, deinit other: Self) -> None:
        self._lib = other._lib
        self._face = other._face
        self._ok = other._ok
        self._size_px = other._size_px
# --- High-level convenience for plotting text lines --------------------------
struct GlyphBitmap:
    var ok: Bool
    var width: Int
    var height: Int
    var left: Int
    var top: Int
    var advance_x: Int
    var pixels: List[UInt8]
fn __init__(out self) -> None:
        self.ok = False
        self.width = 0
        self.height = 0
        self.left = 0
        self.top = 0
        self.advance_x = 0
        self.pixels = List[UInt8]()
fn __copyinit__(out self, other: Self) -> None:
        self.ok = other.ok
        self.width = other.width
        self.height = other.height
        self.left = other.left
        self.top = other.top
        self.advance_x = other.advance_x
        self.pixels = other.pixels
fn __moveinit__(out self, deinit other: Self) -> None:
        self.ok = other.ok
        self.width = other.width
        self.height = other.height
        self.left = other.left
        self.top = other.top
        self.advance_x = other.advance_x
        self.pixels = other.pixels
fn ft_render_text_line(face: FTFace, text: String) -> List[GlyphBitmap]:
    var out = List[GlyphBitmap]()
    var i = 0
    while i < len(text):
        var cp = Int(text[i])
        var (ok, w, h, l, t, adv, buf) = face.render_codepoint(cp)
        var g = GlyphBitmap()
        g.ok = ok
        g.width = w
        g.height = h
        g.left = l
        g.top = t
        g.advance_x = adv
        g.pixels = buf
        out.push(g)
        i += 1
    return out

# --- Minimal smoke self-test (does not print complex data) -------------------
fn _self_test() -> Bool:
    # This will not fail even if FT is unavailable; we just instantiate FTLib.
    var lib = FTLib()
    # Do not try to open a real font here (environment-dependent).
    return True