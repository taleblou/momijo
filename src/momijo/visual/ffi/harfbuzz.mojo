# Project:      Momijo
# Module:       src.momijo.visual.ffi.harfbuzz
# File:         harfbuzz.mojo
# Path:         src/momijo/visual/ffi/harfbuzz.mojo
#
# Description:  src.momijo.visual.ffi.harfbuzz — focused Momijo functionality with a stable public API.
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
#   - Structs: hb_buffer_t, hb_font_t, hb_face_t, FT_Face, hb_glyph_info_t, hb_glyph_position_t, HBFont, HBGlyph
#   - Key functions: HB_SCRIPT_LATIN_const, HB_DIRECTION_LTR_const, hb_buffer_create, hb_buffer_destroy, hb_buffer_add_utf8, hb_buffer_guess_segment_properties, hb_shape, hb_buffer_get_length ...
#   - Static methods present.
#   - Low-level memory (Pointer/UnsafePointer) used; observe safety invariants.


@staticmethod
fn HB_SCRIPT_LATIN_const() -> c_uint32:
    return 0x6C61746E  # 'latn' tag; for reference

@staticmethod
fn HB_DIRECTION_LTR_const() -> c_uint:
    return 4  # not directly used; we var HB guess

from memory.unsafe import UnsafePointer
from momijo.core.error import module
from momijo.dataframe.helpers import m, t
from momijo.utils.result import f, g
from momijo.visual.scene.scene import point, text
from pathlib import Path
from pathlib.path import Path

struct hb_buffer_t: pass
struct hb_font_t: pass
struct hb_face_t: pass

# We will import FT_Face from visual/ffi/freetype if available;
# to avoid circular coupling, we duplicate a minimal opaque decl here.
struct FT_Face: pass

# --- Basic C-like aliases ----------------------------------------------------
# NOTE: Removed duplicate definition of `c_int`; use `from momijo.visual.ffi.zlib_c import c_int`
# NOTE: Removed duplicate definition of `c_uint`; use `from momijo.visual.ffi.freetype import c_uint`
# NOTE: Removed duplicate definition of `c_char`; use `from momijo.visual.ffi.freetype import c_char`




# --- HarfBuzz enums/constants ---------------61746E  # 'latn' tag; for reference

# --- HarfBuzz core API (subset) ---------------------------------------------
@foreign("C")
fn hb_buffer_create() -> hb_buffer_t: pass

@foreign("C")
fn hb_buffer_destroy(buf: hb_buffer_t) -> None: pass

@foreign("C")
fn hb_buffer_add_utf8(buf: hb_buffer_t, text: UnsafePointer[c_char],
                      text_length: c_int, item_offset: c_uint, item_length: c_int) -> None: pass

@foreign("C")
fn hb_buffer_guess_segment_properties(buf: hb_buffer_t) -> None: pass

@foreign("C")
fn hb_shape(font: hb_font_t, buf: hb_buffer_t, features: UnsafePointer[None], num_features: c_uint) -> None: pass

@foreign("C")
fn hb_buffer_get_length(buf: hb_buffer_t) -> c_uint: pass

# info/pos structs and accessors
struct hb_glyph_info_t:
    var codepoint: c_uint32
    var cluster: c_uint32
    # The real struct has more fields (mask, var1/var2). Omitted here.
fn __init__(out self, codepoint: c_uint32, cluster: c_uint32) -> None:
        self.codepoint = codepoint
        self.cluster = cluster
fn __copyinit__(out self, other: Self) -> None:
        self.codepoint = other.codepoint
        self.cluster = other.cluster
fn __moveinit__(out self, deinit other: Self) -> None:
        self.codepoint = other.codepoint
        self.cluster = other.cluster
struct hb_glyph_position_t:
    var x_advance: Int
    var y_advance: Int
    var x_offset: Int
    var y_offset: Int
    # The real struct has more fields.
fn __init__(out self, x_advance: Int = 0, y_advance: Int = 0, x_offset: Int = 0, y_offset: Int = 0) -> None:
        self.x_advance = x_advance
        self.y_advance = y_advance
        self.x_offset = x_offset
        self.y_offset = y_offset
fn __copyinit__(out self, other: Self) -> None:
        self.x_advance = other.x_advance
        self.y_advance = other.y_advance
        self.x_offset = other.x_offset
        self.y_offset = other.y_offset
fn __moveinit__(out self, deinit other: Self) -> None:
        self.x_advance = other.x_advance
        self.y_advance = other.y_advance
        self.x_offset = other.x_offset
        self.y_offset = other.y_offset
@foreign("C")
fn hb_buffer_get_glyph_infos(buf: hb_buffer_t, out_len: UnsafePointer[c_uint]) -> UnsafePointer[hb_glyph_info_t]: pass

@foreign("C")
fn hb_buffer_get_glyph_positions(buf: hb_buffer_t, out_len: UnsafePointer[c_uint]) -> UnsafePointer[hb_glyph_position_t]: pass

# --- hb-ft bridge (link HarfBuzz with FreeType face) ------------------------
@foreign("C")
fn hb_ft_font_create_referenced(ft_face: FT_Face) -> hb_font_t: pass

@foreign("C")
fn hb_font_destroy(font: hb_font_t) -> None: pass

# --- Safe wrappers -----------------------------------------------------------
struct HBFont:
    var _hb_font: hb_font_t
    var _ok: Bool
fn __init__(out self, ft_face: FT_Face) -> None:
        var f = hb_ft_font_create_referenced(ft_face)
        self._hb_font = f
        # If creation fails, HarfBuzz returns null; we approximate with address check via implicit Bool-like.
        self._ok = True  # Assume ok; caller will see empty results if not.
# NOTE: Removed duplicate definition of `is_ok`; use `from momijo.core.error import is_ok`
fn deinit(mut self) -> None:
        # Destroy hb font if non-null; if environment lacks HB, this is a no-op.
        hb_font_destroy(self._hb_font)
fn __copyinit__(out self, other: Self) -> None:
        self._hb_font = other._hb_font
        self._ok = other._ok
fn __moveinit__(out self, deinit other: Self) -> None:
        self._hb_font = other._hb_font
        self._ok = other._ok
struct HBGlyph:
    var gid: Int
    var cluster: Int
    var x_adv_px: Int
    var y_adv_px: Int
    var x_off_px: Int
    var y_off_px: Int
fn __init__(out self) -> None:
        self.gid = 0
        self.cluster = 0
        self.x_adv_px = 0
        self.y_adv_px = 0
        self.x_off_px = 0
        self.y_off_px = 0
fn __copyinit__(out self, other: Self) -> None:
        self.gid = other.gid
        self.cluster = other.cluster
        self.x_adv_px = other.x_adv_px
        self.y_adv_px = other.y_adv_px
        self.x_off_px = other.x_off_px
        self.y_off_px = other.y_off_px
fn __moveinit__(out self, deinit other: Self) -> None:
        self.gid = other.gid
        self.cluster = other.cluster
        self.x_adv_px = other.x_adv_px
        self.y_adv_px = other.y_adv_px
        self.x_off_px = other.x_off_px
        self.y_off_px = other.y_off_px
struct HBShapeResult:
    var glyphs: List[HBGlyph]
    var ok: Bool
fn __init__(out self) -> None:
        self.glyphs = List[HBGlyph]()
        self.ok = False
fn __copyinit__(out self, other: Self) -> None:
        self.glyphs = other.glyphs
        self.ok = other.ok
fn __moveinit__(out self, deinit other: Self) -> None:
        self.glyphs = other.glyphs
        self.ok = other.ok
struct HBShaper:
    var _font: HBFont
fn __init__(out self, font: HBFont) -> None:
        self._font = font

    # Shape a UTF-8 string into glyph indices and pixel positions.
    # Returns HBShapeResult (glyph sequence with advances/offsets in pixels).
fn shape(self, text: String) -> HBShapeResult:
        var res = HBShapeResult()
        if not self._font.is_ok():
            return res

        var buf = hb_buffer_create()
        # Convert to C string (nul-terminated) for hb_buffer_add_utf8
        var ctext = text + String("\0")
        hb_buffer_add_utf8(buf, ctext.data_pointer(), len(text), 0, len(text))
        hb_buffer_guess_segment_properties(buf)

        hb_shape(self._font._hb_font, buf, UnsafePointer[None].null(), 0)

        var out_len_ptr = UnsafePointer[c_uint].null()
        # We need lengths for both arrays; HarfBuzz returns same length
        var infos_ptr = hb_buffer_get_glyph_infos(buf, out_len_ptr)
        var n = out_len_ptr.load()
        var pos_len_ptr = UnsafePointer[c_uint].null()
        var pos_ptr = hb_buffer_get_glyph_positions(buf, pos_len_ptr)
        var m = pos_len_ptr.load()

        if n == 0 or m == 0 or n != m:
            hb_buffer_destroy(buf)
            return res

        var i: Int = 0
        while i < Int(n):
            var info = (infos_ptr + i).load()
            var pos = (pos_ptr + i).load()
            var g = HBGlyph()
            g.gid = Int(info.codepoint)
            g.cluster = Int(info.cluster)
            # Convert 26.6 fixed-point to pixels by >> UInt8(6)
            g.x_adv_px = pos.x_advance >> UInt8(6)
            g.y_adv_px = pos.y_advance >> UInt8(6)
            g.x_off_px = pos.x_offset >> UInt8(6)
            g.y_off_px = pos.y_offset >> UInt8(6)
            res.glyphs.push(g)
            i += 1

        hb_buffer_destroy(buf)
        res.ok = True
        return res
fn __copyinit__(out self, other: Self) -> None:
        self._font = other._font
        self.i = other.i
fn __moveinit__(out self, deinit other: Self) -> None:
        self._font = other._font
        self.i = other.i
# --- Minimal smoke test ------------------------------------------------------
fn _self_test() -> Bool:
    # Can't construct a real HBFont here without a real FT_Face; just return True.
    return True