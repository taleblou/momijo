# MIT License
# SPDX-License-Identifier: MIT
# Project: momijo.vision.io
# File: src/momijo/vision/io/bit_reader.mojo
# Description: MSB-first JPEG bitstream reader with 0xFF/0x00 stuffing handling and safe EOF/marker detection.

# BitReader consumes entropy-coded JPEG scan data where 0xFF bytes are stuffed
# as 0xFF 0x00 within the scan. When a non-zero byte follows 0xFF, that indicates
# a marker boundary and we stop (EOF for the scan).
struct BitReader(Copyable, Movable):
    var ptr: UnsafePointer[UInt8]   # non-owning input buffer (scan payload start)
    var length: Int                 # total available bytes
    var pos: Int                    # current byte offset into ptr
    var bit_buf: UInt32             # MSB-first shift register
    var bit_count: Int              # number of valid bits in bit_buf
    var eof_flag: Bool              # set when marker encountered or end reached

    fn __init__(out self, ptr: UnsafePointer[UInt8], length: Int):
        self.ptr = ptr
        self.length = length
        self.pos = 0
        self.bit_buf = UInt32(0)
        self.bit_count = 0
        self.eof_flag = False

    fn __copyinit__(out self, other: Self):
        self.ptr = other.ptr
        self.length = other.length
        self.pos = other.pos
        self.bit_buf = other.bit_buf
        self.bit_count = other.bit_count
        self.eof_flag = other.eof_flag

    # Reuse the reader with a new buffer.
    fn reset(mut self, ptr: UnsafePointer[UInt8], length: Int):
        self.ptr = ptr
        self.length = length
        self.pos = 0
        self.bit_buf = UInt32(0)
        self.bit_count = 0
        self.eof_flag = False

    fn is_eof(self) -> Bool:
        return self.eof_flag

    fn bytes_consumed(self) -> Int:
        return self.pos

    fn remaining_bytes(self) -> Int:
        var r = self.length - self.pos
        if r < 0:
            r = 0
        return r

    # Pull whole bytes until at least `need` bits are available (or EOF/marker).
    # Handles JPEG byte stuffing: 0xFF 0x00 -> literal 0xFF data byte.
    # If we see 0xFF followed by a non-zero byte, we stop (marker encountered).
    fn _fill(mut self, need: Int):
        var req = need
        if req < 0:
            req = 0
        while self.bit_count < req and self.pos < self.length and not self.eof_flag:
            var b = self.ptr[self.pos]
            self.pos = self.pos + 1

            if b == UInt8(0xFF):
                if self.pos < self.length:
                    var nextb = self.ptr[self.pos]
                    if nextb == UInt8(0x00):
                        # stuffed 0xFF -> consume the 0x00 and treat 0xFF as data
                        self.pos = self.pos + 1
                    else:
                        # marker boundary detected; stop feeding scan data
                        self.eof_flag = True
                        break
                else:
                    # 0xFF at end of buffer -> treat as marker/EOF boundary
                    self.eof_flag = True
                    break

            # push byte into MSB-first buffer
            self.bit_buf = (self.bit_buf << UInt32(8)) | UInt32(b)
            self.bit_count = self.bit_count + 8

    # Read n bits (1..24) MSB-first. Returns 0 on underflow/EOF.
    fn read_bits(mut self, n: Int) -> UInt32:
        var k = n
        if k <= 0:
            return UInt32(0)
        if k > 24:
            k = 24

        self._fill(k)

        if self.bit_count < k:
            self.eof_flag = True
            return UInt32(0)

        var shift = self.bit_count - k
        var mask = (UInt32(1) << UInt32(k)) - UInt32(1)
        var outv = (self.bit_buf >> UInt32(shift)) & mask

        # consume k bits
        if shift == 0:
            self.bit_buf = UInt32(0)
        else:
            self.bit_buf = self.bit_buf & ((UInt32(1) << UInt32(shift)) - UInt32(1))
        self.bit_count = self.bit_count - k
        return outv

    # Read a single bit. Returns 0 on underflow/EOF.
    fn read_bit(mut self) -> UInt32:
        return self.read_bits(1)

    # Peek n bits without consuming. Returns 0 on underflow/EOF.
    fn peek_bits(mut self, n: Int) -> UInt32:
        if n <= 0:
            return UInt32(0)
        var want = n
        if want > 24:
            want = 24
        self._fill(want)
        if self.bit_count < n:
            # Do not forcibly set eof if we simply lack bits due to boundary;
            # but if boundary was hit during _fill, eof_flag is already set.
            return UInt32(0)
        var shift = UInt32(self.bit_count - n)
        var mask = (UInt32(1) << UInt32(n)) - UInt32(1)
        return (self.bit_buf >> shift) & mask

    # Skip n bits (best-effort). Sets eof if not enough data available.
    fn skip_bits(mut self, n: Int):
        if n <= 0:
            return
        self._fill(n)
        if self.bit_count < n:
            self.bit_buf = UInt32(0)
            self.bit_count = 0
            self.eof_flag = True
            return
        var shift = UInt32(self.bit_count - n)
        self.bit_buf = self.bit_buf & ((UInt32(1) << shift) - UInt32(1))
        self.bit_count = self.bit_count - n

    # Discard residual bits to align to the next byte boundary.
    fn align_byte(mut self):
        var rem = self.bit_count % 8
        if rem != 0:
            _ = self.read_bits(rem)

    # Read a single "data byte" from the entropy stream, honoring stuffing.
    # Returns (ok, byte). If a marker boundary is reached, ok=False.
    fn read_data_byte(mut self) -> (Bool, UInt8):
        # First flush to a byte boundary if needed
        if (self.bit_count % 8) != 0:
            _ = self.align_byte()

        if self.eof_flag or self.pos >= self.length:
            self.eof_flag = True
            return (False, UInt8(0))

        var b = self.ptr[self.pos]
        self.pos = self.pos + 1

        if b == UInt8(0xFF):
            if self.pos < self.length:
                var nextb = self.ptr[self.pos]
                if nextb == UInt8(0x00):
                    # stuffed 0xFF; consume the 0x00 and return 0xFF
                    self.pos = self.pos + 1
                    return (True, UInt8(0xFF))
                else:
                    # marker boundary -> no byte returned
                    self.eof_flag = True
                    return (False, UInt8(0))
            else:
                # 0xFF at end -> boundary
                self.eof_flag = True
                return (False, UInt8(0))

        return (True, b)
