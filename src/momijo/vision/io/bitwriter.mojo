# MIT License
# SPDX-License-Identifier: MIT
# Project: momijo.vision.io
# File: momijo/vision/io/bitwriter.mojo
# Description: MSB-first JPEG bit writer with 0xFF byte stuffing.

struct BitWriter:
    var ptr: UnsafePointer[UInt8]  # non-owning output buffer
    var pos: Int                   # bytes written (including stuffed 0x00 after 0xFF)
    var bit_buf: UInt32            # MSB-first shift register
    var bit_count: Int             # number of valid bits currently in bit_buf

    fn __init__(out self, ptr: UnsafePointer[UInt8]):
        self.ptr = ptr
        self.pos = 0
        self.bit_buf = 0
        self.bit_count = 0

    # Write one already-aligned byte with JPEG 0xFF stuffing.
    fn write_byte(mut self, b: UInt8):
        self.ptr[self.pos] = b
        self.pos = self.pos + 1
        if b == UInt8(0xFF):
            # Stuff 0x00 after any 0xFF data byte
            self.ptr[self.pos] = UInt8(0x00)
            self.pos = self.pos + 1

    # Write a 16-bit JPEG marker (0xFF, then low byte of marker).
    # E.g., SOI=0xFFD8 -> writes 0xFF, 0xD8.
    fn write_marker(mut self, marker: UInt16):
        self.write_byte(UInt8(0xFF))
        self.write_byte(UInt8(marker & UInt16(0xFF)))

    # Ensure at least 8 bits are emitted to bytes.
    fn _flush_bytes(mut self):
        while self.bit_count >= 8:
            var byte = UInt8((self.bit_buf >> UInt32(self.bit_count - 8)) & UInt32(0xFF))
            self.write_byte(byte)
            self.bit_count = self.bit_count - 8

    # Write 'count' bits from 'bits' (bits must already be masked, MSB-first).
    fn write_bits(mut self, bits: UInt32, count: Int):
        if count <= 0:
            return
        self.bit_buf = (self.bit_buf << UInt32(count)) | (bits & ((UInt32(1) << UInt32(count)) - UInt32(1)))
        self.bit_count = self.bit_count + count
        self._flush_bytes()

    # Byte-align by emitting remaining bits (left-justified into a byte).
    fn flush_final(mut self):
        if self.bit_count > 0:
            var rem = self.bit_count
            var byte = UInt8((self.bit_buf << UInt32(8 - rem)) & UInt32(0xFF))
            self.write_byte(byte)
            self.bit_buf = 0
            self.bit_count = 0

    fn bytes_written(self) -> Int:
        return self.pos
