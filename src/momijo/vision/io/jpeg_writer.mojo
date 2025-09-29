# MIT License
# SPDX-License-Identifier: MIT
# Project: momijo
# File: momijo/vision/io/jpeg_writer.mojo
# Description: Minimal JPEG segment writers (grayscale + color), no file-scope consts.

from momijo.vision.io.bitwriter import BitWriter

# ---------------- Utilities ----------------

# Write generic segment header: marker (low byte; writer is expected to add 0xFF) + 2-byte length
fn write_segment(mut bw: BitWriter, marker: UInt16, length: Int):
    bw.write_marker(marker)
    bw.write_byte(UInt8((length >> 8) & 0xFF))
    bw.write_byte(UInt8(length & 0xFF))

# ---------------- SOI / EOI ----------------

fn write_soi(mut bw: BitWriter):
    # SOI = 0xFFD8 -> pass 0xD8 (writer adds 0xFF)
    bw.write_marker(UInt16(0xD8))

fn write_eoi(mut bw: BitWriter):
    # EOI = 0xFFD9
    bw.write_marker(UInt16(0xD9))

# ---------------- APP0 (JFIF) ----------------
# Minimal JFIF header; many decoders expect this but it's optional per baseline JPEG.
fn write_app0_jfif(mut bw: BitWriter):
    # Length = 16: 2(len) + "JFIF\0"(5) + ver(2) + units(1) + Xdensity(2) + Ydensity(2) + Xthumb(1) + Ythumb(1)
    write_segment(bw, UInt16(0xE0), 16)
    # "JFIF\0"
    bw.write_byte(UInt8(0x4A))  # J
    bw.write_byte(UInt8(0x46))  # F
    bw.write_byte(UInt8(0x49))  # I
    bw.write_byte(UInt8(0x46))  # F
    bw.write_byte(UInt8(0x00))  # \0
    # Version 1.01
    bw.write_byte(UInt8(0x01))
    bw.write_byte(UInt8(0x01))
    # Units: 0 = no units, aspect ratio only
    bw.write_byte(UInt8(0x00))
    # Xdensity, Ydensity
    bw.write_byte(UInt8(0x00)); bw.write_byte(UInt8(0x48))  # 72
    bw.write_byte(UInt8(0x00)); bw.write_byte(UInt8(0x48))  # 72
    # No thumbnail
    bw.write_byte(UInt8(0x00))  # Xthumb
    bw.write_byte(UInt8(0x00))  # Ythumb

# ---------------- DQT (Quantization Tables) ----------------
# Overload A: write a provided 8-bit (precision=0) 64-entry table with given table_id (0..3)
fn write_dqt_table(mut bw: BitWriter, table_id: UInt8, table_ptr: UnsafePointer[UInt8]):
    # Length = 2 (len field itself) + 1 (Pq/Tq) + 64 (table) = 67
    write_segment(bw, UInt16(0xDB), 67)
    # Pq/Tq: high nibble = precision (0 => 8-bit), low nibble = table_id
    bw.write_byte(UInt8((0 << 4) | (table_id & 0x0F)))
    var i = 0
    while i < 64:
        bw.write_byte(table_ptr[i])
        i = i + 1

# Overload B: standard luminance Q-table (quality ~50) as table #0
fn write_dqt(mut bw: BitWriter):
    var qt = UnsafePointer[UInt8].alloc(64)
    var vals = List[UInt8]()
    vals.extend([
        16,11,10,16,24,40,51,61,
        12,12,14,19,26,58,60,55,
        14,13,16,24,40,57,69,56,
        14,17,22,29,51,87,80,62,
        18,22,37,56,68,109,103,77,
        24,35,55,64,81,104,113,92,
        49,64,78,87,103,121,120,101,
        72,92,95,98,112,100,103,99
    ])
    var i = 0
    while i < 64:
        qt[i] = UInt8(vals[i])
        i = i + 1
    write_dqt_table(bw, UInt8(0), qt)

# Optional: standard chroma table as table #1
fn write_dqt_chroma(mut bw: BitWriter):
    var qt = UnsafePointer[UInt8].alloc(64)
    var vals = List[UInt8]()
    # A common chroma table used by many minimal encoders (roughly IJG standard chroma)
    vals.extend([
        17,18,18,24,21,24,47,26,
        26,47,99,66,56,66,99,99,
        99,99,99,99,99,99,99,99,
        99,99,99,99,99,99,99,99,
        99,99,99,99,99,99,99,99,
        99,99,99,99,99,99,99,99,
        99,99,99,99,99,99,99,99,
        99,99,99,99,99,99,99,99
    ])
    var i = 0
    while i < 64:
        qt[i] = UInt8(vals[i])
        i = i + 1
    write_dqt_table(bw, UInt8(1), qt)

# ---------------- SOF0 (Baseline DCT) ----------------

# Grayscale SOF0 (1 component, quant table #0)
fn write_sof0(mut bw: BitWriter, width: Int, height: Int):
    # Length = 8 + 3*Nf = 11 when Nf=1
    write_segment(bw, UInt16(0xC0), 11)
    bw.write_byte(UInt8(8))  # sample precision
    bw.write_byte(UInt8((height >> 8) & 0xFF))
    bw.write_byte(UInt8(height & 0xFF))
    bw.write_byte(UInt8((width >> 8) & 0xFF))
    bw.write_byte(UInt8(width & 0xFF))
    bw.write_byte(UInt8(1))  # Nf = 1
    # Component #1 (Y)
    bw.write_byte(UInt8(1))      # C1 id
    bw.write_byte(UInt8(0x11))   # H=1, V=1
    bw.write_byte(UInt8(0))      # Tq=0 (luma)

# Color SOF0 (3 components, Y Cb Cr)
fn write_sof0_color(mut bw: BitWriter, width: Int, height: Int):
    # Length = 8 + 3*Nf = 17 when Nf=3
    write_segment(bw, UInt16(0xC0), 17)
    bw.write_byte(UInt8(8))  # precision
    bw.write_byte(UInt8((height >> 8) & 0xFF))
    bw.write_byte(UInt8(height & 0xFF))
    bw.write_byte(UInt8((width >> 8) & 0xFF))
    bw.write_byte(UInt8(width & 0xFF))
    bw.write_byte(UInt8(3))  # Nf = 3

    # Y
    bw.write_byte(UInt8(1))      # C1 id
    bw.write_byte(UInt8(0x11))   # H=1,V=1 (no subsampling)
    bw.write_byte(UInt8(0))      # Tq=0 (luma)

    # Cb
    bw.write_byte(UInt8(2))      # C2 id
    bw.write_byte(UInt8(0x11))   # H=1,V=1
    bw.write_byte(UInt8(1))      # Tq=1 (chroma)

    # Cr
    bw.write_byte(UInt8(3))      # C3 id
    bw.write_byte(UInt8(0x11))   # H=1,V=1
    bw.write_byte(UInt8(1))      # Tq=1 (chroma)

# ---------------- DHT (Huffman Tables) ----------------
# lengths: 16 bytes (counts for code lengths 1..16)
# symbols: total bytes = sum(lengths[i])
fn write_dht(
    mut bw: BitWriter,
    lengths: UnsafePointer[UInt8],
    symbols: UnsafePointer[UInt8],
    is_dc: Bool
):
    # Compute total number of symbols
    var total = 0
    var i = 0
    while i < 16:
        total = total + Int(lengths[i])
        i = i + 1

    # Length field = 2(len) + 1(HT info) + 16(counts) + total(symbols)
    write_segment(bw, UInt16(0xC4), 3 + 16 + total)

    # HT info: high nibble = class (0=DC,1=AC), low nibble = table id (0)
    var ht_info: UInt8 = UInt8(0)
    if is_dc:
        ht_info = UInt8(0x00)  # DC, table 0
    else:
        ht_info = UInt8(0x10)  # AC, table 0
    bw.write_byte(ht_info)

    # 16 length counts
    i = 0
    while i < 16:
        bw.write_byte(lengths[i])
        i = i + 1

    # Symbols
    i = 0
    while i < total:
        bw.write_byte(symbols[i])
        i = i + 1


# ---------------- SOS (Start of Scan) ----------------

# Grayscale SOS
fn write_sos(mut bw: BitWriter):
    # Length = 6 + 2*Nf = 8 for Nf=1
    write_segment(bw, UInt16(0xDA), 8)
    bw.write_byte(UInt8(1))      # Ns
    bw.write_byte(UInt8(1))      # C1
    bw.write_byte(UInt8(0x00))   # (Td=0,Ta=0) both DC/AC table 0
    bw.write_byte(UInt8(0))      # Ss
    bw.write_byte(UInt8(63))     # Se
    bw.write_byte(UInt8(0))      # AhAl

# Color SOS (3 components Y Cb Cr)
fn write_sos_color(mut bw: BitWriter, ncomp: Int = 3):
    # Length = 6 + 2*Nf = 12 for Nf=3
    write_segment(bw, UInt16(0xDA), 12)
    bw.write_byte(UInt8(ncomp))  # Ns

    # Y
    bw.write_byte(UInt8(1))      # C1 id
    bw.write_byte(UInt8(0x00))   # Td=0, Ta=0

    # Cb
    bw.write_byte(UInt8(2))      # C2 id
    bw.write_byte(UInt8(0x00))   # Td=0, Ta=0 (shared tables in minimal encoder)

    # Cr
    bw.write_byte(UInt8(3))      # C3 id
    bw.write_byte(UInt8(0x00))   # Td=0, Ta=0

    bw.write_byte(UInt8(0))      # Ss
    bw.write_byte(UInt8(63))     # Se
    bw.write_byte(UInt8(0))      # AhAl
