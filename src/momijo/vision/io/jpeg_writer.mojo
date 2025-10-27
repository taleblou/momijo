# MIT License
# SPDX-License-Identifier: MIT
# Project: momijo.vision.io.jpeg
# File: src/momijo/vision/io/internal/jpeg_writer.mojo
# Description: JPEG marker/segment writers (SOI/APP0/DQT/SOF0/DHT/SOS/EOI)
#              for Baseline DCT, 8-bit samples, 4:4:4 (no subsampling).
# Notes:
# - English-only comments per project rules.
# - Works with BitWriter from 'bitwriter.mojo'.
# - Width/Height ordering in SOF0 is correct: [Height][Width] big-endian.
# - APP0 JFIF is minimal (version 1.01, 72x72 dpi, no thumbnail).
# - DQT uses 8-bit precision (Pq=0).
# - DHT uses ITU T.81 Annex K Baseline default tables (Std Luma/Chroma DC/AC).
# - SOS for gray (1 comp) or color (3 comps, Y:(0,0), Cb/Cr:(1,1)).


from collections.list import List
from momijo.vision.io.bitwriter import BitWriter

# ------------------------------- Markers -------------------------------- #
@always_inline
fn write_soi(mut bw: BitWriter) -> Bool:
    # SOI: 0xFF D8
    return bw.write_marker(UInt8(0xD8))

@always_inline
fn write_eoi(mut bw: BitWriter) -> Bool:
    # EOI: 0xFF D9
    return bw.write_marker(UInt8(0xD9))

# ------------------------------- APP0 JFIF ------------------------------ #
fn write_app0_jfif(mut bw: BitWriter, xdpi: Int = 72, ydpi: Int = 72) -> Bool:
    # APP0 (JFIF v1.01, density in DPI, no thumbnail)
    if not bw.write_marker(UInt8(0xE0)): return False  # APP0
    # Length = 16 (including these two bytes)
    if not bw.write_be16(16): return False
    # Identifier "JFIF\0"
    if not bw.write_raw_byte(UInt8(ord("J"))): return False
    if not bw.write_raw_byte(UInt8(ord("F"))): return False
    if not bw.write_raw_byte(UInt8(ord("I"))): return False
    if not bw.write_raw_byte(UInt8(ord("F"))): return False
    if not bw.write_raw_byte(UInt8(0x00)): return False
    # Version 1.01
    if not bw.write_raw_byte(UInt8(0x01)): return False
    if not bw.write_raw_byte(UInt8(0x01)): return False
    # Units: 1 = dots per inch
    if not bw.write_raw_byte(UInt8(0x01)): return False
    # Xdensity, Ydensity (big-endian 16-bit)
    if not bw.write_be16(xdpi): return False
    if not bw.write_be16(ydpi): return False
    # No thumbnail
    if not bw.write_raw_byte(UInt8(0x00)): return False  # Xthumb
    if not bw.write_raw_byte(UInt8(0x00)): return False  # Ythumb
    return True

# --------------------------------- DQT ---------------------------------- #
# Write an 8-bit quantization table (64 entries, natural order).
fn write_dqt(mut bw: BitWriter, table_id: Int, table: UnsafePointer[UInt8]) -> Bool:
    # DQT marker
    if not bw.write_marker(UInt8(0xDB)): return False
    # Length = 2 + 1 + 64 = 67
    if not bw.write_be16(67): return False
    # Pq/Tq (Pq=0 for 8-bit tables)
    var pq_tq = UInt8((0 << 4) | (table_id & 0x0F))
    if not bw.write_raw_byte(pq_tq): return False
    # Table bytes in natural order
    var i = 0
    while i < 64:
        if not bw.write_raw_byte(table[i]): return False
        i += 1
    return True

# -------------------------------- SOF0 ---------------------------------- #
# Baseline DCT (8-bit), 4:4:4 sampling for color case.
fn write_sof0(mut bw: BitWriter, width: Int, height: Int) -> Bool:
    # Grayscale: Nf = 1 -> length = 8 + 3*1 = 11
    if not bw.write_marker(UInt8(0xC0)): return False
    if not bw.write_be16(11): return False
    # Sample precision
    if not bw.write_raw_byte(UInt8(8)): return False
    # Height then Width (big-endian) – IMPORTANT!
    if not bw.write_be16(height): return False
    if not bw.write_be16(width):  return False
    # Number of components
    if not bw.write_raw_byte(UInt8(1)): return False
    # Component 1 (Y): id=1, sampling 1x1 (0x11), quant table 0
    if not bw.write_raw_byte(UInt8(1)):    return False  # C1 id
    if not bw.write_raw_byte(UInt8(0x11)): return False  # H=1, V=1
    if not bw.write_raw_byte(UInt8(0)):    return False  # Tq=0
    return True

fn write_sof0_color(mut bw: BitWriter, width: Int, height: Int) -> Bool:
    # Color: Nf = 3 -> length = 8 + 3*3 = 17
    if not bw.write_marker(UInt8(0xC0)): return False
    if not bw.write_be16(17): return False
    # Sample precision
    if not bw.write_raw_byte(UInt8(8)): return False
    # Height then Width (big-endian) – IMPORTANT!
    if not bw.write_be16(height): return False
    if not bw.write_be16(width):  return False
    # Number of components
    if not bw.write_raw_byte(UInt8(3)): return False
    # Y component: id=1, 1x1, Tq=0
    if not bw.write_raw_byte(UInt8(1)):    return False  # Y id
    if not bw.write_raw_byte(UInt8(0x11)): return False  # H=1, V=1 (4:4:4)
    if not bw.write_raw_byte(UInt8(0)):    return False  # Tq=0
    # Cb component: id=2, 1x1, Tq=1
    if not bw.write_raw_byte(UInt8(2)):    return False
    if not bw.write_raw_byte(UInt8(0x11)): return False
    if not bw.write_raw_byte(UInt8(1)):    return False
    # Cr component: id=3, 1x1, Tq=1
    if not bw.write_raw_byte(UInt8(3)):    return False
    if not bw.write_raw_byte(UInt8(0x11)): return False
    if not bw.write_raw_byte(UInt8(1)):    return False
    return True

# --------------------------------- DHT ---------------------------------- #
# Write baseline default Huffman tables (Annex K) for DC/AC, Luma/Chroma.
# This emits four separate DHT segments (one per table).
fn write_dht_std(mut bw: BitWriter) -> Bool:
    # Each table: [bits(16), huffval(n)] as per JPEG spec. Class/Id in TcTh byte.
    # -- DC Luma (Tc=0, Th=0) --
    var bits_dc_l = List[UInt8]([
        0x00,0x01,0x05,0x01,0x01,0x01,0x01,0x01,
        0x01,0x01,0x00,0x00,0x00,0x00,0x00,0x00
    ])
    var val_dc_l = List[UInt8]([
        0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,0x08,0x09,0x0A,0x0B
    ])

    if not _write_single_dht(bw, 0, 0, bits_dc_l, val_dc_l): return False

    # -- AC Luma (Tc=1, Th=0) --
    var bits_ac_l = List[UInt8]([
        0x00,0x02,0x01,0x03,0x03,0x02,0x04,0x03,
        0x05,0x05,0x04,0x04,0x00,0x00,0x01,0x7D
    ])
    var val_ac_l = List[UInt8]([
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
    if not _write_single_dht(bw, 1, 0, bits_ac_l, val_ac_l): return False

    # -- DC Chroma (Tc=0, Th=1) --
    var bits_dc_c = List[UInt8]([
        0x00,0x03,0x01,0x01,0x01,0x01,0x01,0x01,
        0x01,0x01,0x01,0x00,0x00,0x00,0x00,0x00
    ])
    var val_dc_c = List[UInt8]([
        0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,0x08,0x09,0x0A,0x0B
    ])
    if not _write_single_dht(bw, 0, 1, bits_dc_c, val_dc_c): return False

    # -- AC Chroma (Tc=1, Th=1) --
    var bits_ac_c = List[UInt8]([
        0x00,0x02,0x01,0x02,0x04,0x04,0x03,0x04,
        0x07,0x05,0x04,0x04,0x00,0x01,0x02,0x77
    ])
    var val_ac_c = List[UInt8]([
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
    if not _write_single_dht(bw, 1, 1, bits_ac_c, val_ac_c): return False

    return True

# Helper to write one DHT table (class: 0=DC,1=AC; id: 0=luma,1=chroma)
fn _write_single_dht(mut bw: BitWriter, tc: Int, th: Int, bits: List[UInt8], vals: List[UInt8]) -> Bool:
    if not bw.write_marker(UInt8(0xC4)): return False  # DHT
    var nvals = 0
    var i = 0
    while i < 16:
        nvals += Int(bits[i])
        i += 1
    # Length = 2 + 1 + 16 + nvals
    var length = 2 + 1 + 16 + nvals
    if not bw.write_be16(length): return False
    var tc_th = UInt8(((tc & 1) << 4) | (th & 0x0F))
    if not bw.write_raw_byte(tc_th): return False
    # 16 bytes of 'bits'
    i = 0
    while i < 16:
        if not bw.write_raw_byte(bits[i]): return False
        i += 1
    # 'nvals' symbols
    i = 0
    while i < nvals:
        if not bw.write_raw_byte(vals[i]): return False
        i += 1
    return True

# --------------------------------- SOS ---------------------------------- #
@always_inline
fn write_sos_gray(mut bw: BitWriter) -> Bool:
    # SOS length = 2 + 1 + 2*Nc + 3 ; Nc=1 => 8
    if not bw.write_marker(UInt8(0xDA)): return False
    if not bw.write_be16(8): return False
    if not bw.write_raw_byte(UInt8(1)): return False  # Nc=1
    # Component 1 selector and Td/Ta (Y uses DC table 0, AC table 0)
    if not bw.write_raw_byte(UInt8(1)):  return False # Cs1 = 1
    if not bw.write_raw_byte(UInt8(0x00)): return False # Td=0, Ta=0
    # Spectral selection Ss/Se and Ah/Al
    if not bw.write_raw_byte(UInt8(0)):   return False # Ss
    if not bw.write_raw_byte(UInt8(63)):  return False # Se
    if not bw.write_raw_byte(UInt8(0)):   return False # AhAl
    return True

@always_inline
fn write_sos_color(mut bw: BitWriter) -> Bool:
    # SOS length = 2 + 1 + 2*Nc + 3 ; Nc=3 => 12
    if not bw.write_marker(UInt8(0xDA)): return False
    if not bw.write_be16(12): return False
    if not bw.write_raw_byte(UInt8(3)): return False  # Nc=3
    # Y: Cs=1, Td/Ta = (0,0)
    if not bw.write_raw_byte(UInt8(1)):  return False
    if not bw.write_raw_byte(UInt8(0x00)): return False
    # Cb: Cs=2, Td/Ta = (1,1)
    if not bw.write_raw_byte(UInt8(2)):  return False
    if not bw.write_raw_byte(UInt8(0x11)): return False
    # Cr: Cs=3, Td/Ta = (1,1)
    if not bw.write_raw_byte(UInt8(3)):  return False
    if not bw.write_raw_byte(UInt8(0x11)): return False
    # Ss/Se, Ah/Al
    if not bw.write_raw_byte(UInt8(0)):   return False
    if not bw.write_raw_byte(UInt8(63)):  return False
    if not bw.write_raw_byte(UInt8(0)):   return False
    return True
