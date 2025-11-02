# MIT License
# SPDX-License-Identifier: MIT
# Project: momijo.vision
# File: src/momijo/vision/memory.mojo
# Description: Memory utilities, byte buffer (owning), and light-weight views.

# --------------------------------
# UInt8 allocation helpers
# --------------------------------

fn alloc_u8(nbytes: Int) -> UnsafePointer[UInt8]:
    # Allocate nbytes of uninitialized memory (bytes).
    if nbytes <= 0:
        return UnsafePointer[UInt8]()
    return UnsafePointer[UInt8].alloc(nbytes)

fn calloc_u8(nbytes: Int) -> UnsafePointer[UInt8]:
    # Allocate and zero-initialize nbytes.
    if nbytes <= 0:
        return UnsafePointer[UInt8]()
    var ptr = UnsafePointer[UInt8].alloc(nbytes)
    var i = 0
    while i < nbytes:
        ptr[i] = 0
        i += 1
    return ptr

fn free_u8(ptr: UnsafePointer[UInt8]):
    # Explicit deallocation (safe to call on default/null pointer).
    UnsafePointer[UInt8].free(ptr)

# --------------------------------
# Int32 / Float32 / Float64 / UInt16 / Int64 helpers
# (Allocate by element count, not bytes)
# --------------------------------

fn alloc_i32(nelems: Int) -> UnsafePointer[Int32]:
    if nelems <= 0: return UnsafePointer[Int32]()
    return UnsafePointer[Int32].alloc(nelems)

fn calloc_i32(nelems: Int) -> UnsafePointer[Int32]:
    if nelems <= 0: return UnsafePointer[Int32]()
    var p = UnsafePointer[Int32].alloc(nelems)
    var i = 0
    while i < nelems:
        p[i] = 0
        i += 1
    return p

fn free_i32(ptr: UnsafePointer[Int32]):
    UnsafePointer[Int32].free(ptr)

fn alloc_f32(nelems: Int) -> UnsafePointer[Float32]:
    if nelems <= 0: return UnsafePointer[Float32]()
    return UnsafePointer[Float32].alloc(nelems)

fn calloc_f32(nelems: Int) -> UnsafePointer[Float32]:
    if nelems <= 0: return UnsafePointer[Float32]()
    var p = UnsafePointer[Float32].alloc(nelems)
    var i = 0
    while i < nelems:
        p[i] = 0.0
        i += 1
    return p

fn free_f32(ptr: UnsafePointer[Float32]):
    UnsafePointer[Float32].free(ptr)

fn alloc_f64(nelems: Int) -> UnsafePointer[Float64]:
    if nelems <= 0: return UnsafePointer[Float64]()
    return UnsafePointer[Float64].alloc(nelems)

fn calloc_f64(nelems: Int) -> UnsafePointer[Float64]:
    if nelems <= 0: return UnsafePointer[Float64]()
    var p = UnsafePointer[Float64].alloc(nelems)
    var i = 0
    while i < nelems:
        p[i] = 0.0
        i += 1
    return p

fn free_f64(ptr: UnsafePointer[Float64]):
    UnsafePointer[Float64].free(ptr)

fn alloc_u16(nelems: Int) -> UnsafePointer[UInt16]:
    if nelems <= 0: return UnsafePointer[UInt16]()
    return UnsafePointer[UInt16].alloc(nelems)

fn calloc_u16(nelems: Int) -> UnsafePointer[UInt16]:
    if nelems <= 0: return UnsafePointer[UInt16]()
    var p = UnsafePointer[UInt16].alloc(nelems)
    var i = 0
    while i < nelems:
        p[i] = UInt16(0)
        i += 1
    return p

fn free_u16(ptr: UnsafePointer[UInt16]):
    UnsafePointer[UInt16].free(ptr)

fn alloc_i64(nelems: Int) -> UnsafePointer[Int64]:
    if nelems <= 0: return UnsafePointer[Int64]()
    return UnsafePointer[Int64].alloc(nelems)

fn calloc_i64(nelems: Int) -> UnsafePointer[Int64]:
    if nelems <= 0: return UnsafePointer[Int64]()
    var p = UnsafePointer[Int64].alloc(nelems)
    var i = 0
    while i < nelems:
        p[i] = Int64(0)
        i += 1
    return p

fn free_i64(ptr: UnsafePointer[Int64]):
    UnsafePointer[Int64].free(ptr)

# --------------------------------
# Byte-level utilities
# --------------------------------

fn memcpy_u8(dst: UnsafePointer[UInt8], src: UnsafePointer[UInt8], nbytes: Int):
    # Simple forward copy; the caller must if non-overlapping regions.
    var i = 0
    while i < nbytes:
        dst[i] = src[i]
        i += 1

fn memmove_u8(dst: UnsafePointer[UInt8], src: UnsafePointer[UInt8], nbytes: Int):
    # Overlap-safe copy using a temporary buffer. Slower but safe and portable.
    if nbytes <= 0:
        return
    var tmp = alloc_u8(nbytes)
    memcpy_u8(tmp, src, nbytes)
    memcpy_u8(dst, tmp, nbytes)
    free_u8(tmp)

fn memset_u8(dst: UnsafePointer[UInt8], value: UInt8, nbytes: Int):
    var i = 0
    while i < nbytes:
        dst[i] = value
        i += 1

fn memcmp_u8(a: UnsafePointer[UInt8], b: UnsafePointer[UInt8], nbytes: Int) -> Int:
    # Returns 0 if equal, <0 if a<b at first diff, >0 if a>b.
    var i = 0
    while i < nbytes:
        var av = a[i]
        var bv = b[i]
        if av != bv:
            return Int(av) - Int(bv)
        i += 1
    return 0

fn pointer_add_u8(p: UnsafePointer[UInt8], offset_bytes: Int) -> UnsafePointer[UInt8]:
    # Return a pointer advanced by offset_bytes. Caller must if bounds.
    return p + offset_bytes

# --------------------------------
# Hex dump helpers (human-readable)
# --------------------------------

fn hexdigit(n: UInt8) -> UInt8:
    var v = n
    if v < 10: return UInt8(ord("0")) + v
    return UInt8(ord("A")) + (v - UInt8(10))

fn byte_to_hex(b: UInt8) -> String:
    var hi = (b >> 4) & UInt8(0xF)
    var lo = b & UInt8(0xF)
    var s = String("")
    s += String(Char(hexdigit(hi)))
    s += String(Char(hexdigit(lo)))
    return s

# Dump first 'maxn' bytes as "00 1A FF ..."
fn hexdump_head(p: UnsafePointer[UInt8], nbytes: Int, maxn: Int) -> String:
    var n = nbytes
    var k = maxn
    if k < 0: k = 0
    if n < k: k = n
    var out = String("")
    var i = 0
    while i < k:
        if i > 0:
            out += String(" ")
        out += byte_to_hex(p[i])
        i += 1
    return out

# Full dump (bounded lines of 16 bytes), useful for debugging
fn hexdump_full(p: UnsafePointer[UInt8], nbytes: Int) -> String:
    var out = String("")
    var i = 0
    while i < nbytes:
        var line = String("")
        var j = 0
        while j < 16 and (i + j) < nbytes:
            if j > 0:
                line += String(" ")
            line += byte_to_hex(p[i + j])
            j += 1
        out += line
        i += 16
        if i < nbytes:
            out += String("\n")
    return out

# --------------------------------
# Non-owning byte span (view)
# --------------------------------

struct ViewU8(Copyable, Movable):
    var _data: UnsafePointer[UInt8]
    var _length: Int

    fn __init__(out self, data: UnsafePointer[UInt8], length_bytes: Int):
        self._data = data
        var n = length_bytes
        if n < 0:
            n = 0
        self._length = n


    fn __copyinit__(out self, other: ViewU8):
        self._data = other._data
        self._length = other._length

    fn data(self) -> UnsafePointer[UInt8]:
        return self._data

    fn length(self) -> Int:
        return self._length

    fn is_empty(self) -> Bool:
        return self._length == 0

    fn slice(self, offset: Int, size: Int) -> ViewU8:
        var off = offset
        var sz = size
        if off < 0: off = 0
        if sz < 0: sz = 0
        if off >= self._length:
            return ViewU8(UnsafePointer[UInt8](), 0)
        var maxn = self._length - off
        if sz > maxn: sz = maxn
        return ViewU8(self._data + off, sz)

# --------------------------------
# RAII-like byte buffer (UInt8, owning)
# --------------------------------

struct BufferU8(Copyable, Movable):
    var _data: UnsafePointer[UInt8]
    var _length: Int   # bytes

    fn __init__(out self, length_bytes: Int, zero_init: Bool = False):
        var n = length_bytes
        if n <= 0:
            self._data = UnsafePointer[UInt8]()
            self._length = 0
            return
        self._length = n 
        if zero_init:
            self._data = calloc_u8(n)
        else:
            self._data = alloc_u8(n)

    fn __copyinit__(out self, other: BufferU8):
        # Deep copy to avoid double free.
        var n = other._length
        if n <= 0:
            self._data = UnsafePointer[UInt8]()
            self._length = 0
            return
        self._data = alloc_u8(n)
        self._length = n
        memcpy_u8(self._data, other._data, n)

    fn __del__(deinit self):
        free_u8(self._data)
        self._data = UnsafePointer[UInt8]()
        self._length = 0

    fn copy(self) -> BufferU8:
        # Deep copy, consistent with __copyinit__.
        var out = BufferU8(self._length, zero_init=False)
        if self._length > 0:
            memcpy_u8(out._data, self._data, self._length)
        return out

    # --- basic API ---
    fn data(self) -> UnsafePointer[UInt8]:
        return self._data

    fn length(self) -> Int:
        return self._length

    fn view(self) -> ViewU8:
        return ViewU8(self._data, self._length)

    fn zero(self):
        if self._length > 0:
            memset_u8(self._data, 0, self._length)

    fn fill(self, value: UInt8):
        if self._length > 0:
            memset_u8(self._data, value, self._length)

    fn clone(self) -> BufferU8:
        return self.copy()

    # --- slicing / range ops (clamped, no throws) ---
    fn slice(self, offset: Int, size: Int) -> ViewU8:
        var off = offset
        var sz = size
        if off < 0: off = 0
        if sz < 0: sz = 0
        if off >= self._length:
            return ViewU8(UnsafePointer[UInt8](), 0)
        var maxn = self._length - off
        if sz > maxn: sz = maxn
        return ViewU8(self._data + off, sz)

    # Copy from src bytes into this buffer at dst_offset (clamped).
    fn write_from(self, src: UnsafePointer[UInt8], nbytes: Int, dst_offset: Int):
        if nbytes <= 0 or dst_offset >= self._length:
            return
        var off = dst_offset
        if off < 0: off = 0
        var maxn = self._length - off
        var k = nbytes
        if k > maxn: k = maxn
        if k <= 0:
            return
        memcpy_u8(self._data + off, src, k)

    # Copy out of this buffer into dst bytes from src_offset (clamped).
    fn read_into(self, dst: UnsafePointer[UInt8], nbytes: Int, src_offset: Int):
        if nbytes <= 0 or src_offset >= self._length:
            return
        var off = src_offset
        if off < 0: off = 0
        var maxn = self._length - off
        var k = nbytes
        if k > maxn: k = maxn
        if k <= 0:
            return
        memcpy_u8(dst, self._data + off, k)

    # Copy a range within this buffer (memmove semantics; handles overlap).
    fn copy_range(self, src_offset: Int, dst_offset: Int, nbytes: Int):
        if nbytes <= 0 or src_offset >= self._length or dst_offset >= self._length:
            return
        var so = src_offset
        var doff = dst_offset
        if so < 0: so = 0
        if doff < 0: doff = 0
        var max_src = self._length - so
        var max_dst = self._length - doff
        var k = nbytes
        if k > max_src: k = max_src
        if k > max_dst: k = max_dst
        if k <= 0:
            return
        memmove_u8(self._data + doff, self._data + so, k)

# --------------------------------
# Simple typed views over BufferU8
# (Bounds-checked by byte size; returns null pointer on insufficient size)
# --------------------------------

fn as_u16_view(buf: BufferU8, nelems: Int) -> UnsafePointer[UInt16]:
    if nelems <= 0: return UnsafePointer[UInt16]()
    var need = nelems * 2
    if need > buf._length: return UnsafePointer[UInt16]()
    return UnsafePointer[UInt16](buf._data)

fn as_i32_view(buf: BufferU8, nelems: Int) -> UnsafePointer[Int32]:
    if nelems <= 0: return UnsafePointer[Int32]()
    var need = nelems * 4
    if need > buf._length: return UnsafePointer[Int32]()
    return UnsafePointer[Int32](buf._data)

fn as_i64_view(buf: BufferU8, nelems: Int) -> UnsafePointer[Int64]:
    if nelems <= 0: return UnsafePointer[Int64]()
    var need = nelems * 8
    if need > buf._length: return UnsafePointer[Int64]()
    return UnsafePointer[Int64](buf._data)

fn as_f32_view(buf: BufferU8, nelems: Int) -> UnsafePointer[Float32]:
    if nelems <= 0: return UnsafePointer[Float32]()
    var need = nelems * 4
    if need > buf._length: return UnsafePointer[Float32]()
    return UnsafePointer[Float32](buf._data)

fn as_f64_view(buf: BufferU8, nelems: Int) -> UnsafePointer[Float64]:
    if nelems <= 0: return UnsafePointer[Float64]()
    var need = nelems * 8
    if need > buf._length: return UnsafePointer[Float64]()
    return UnsafePointer[Float64](buf._data)

# --------------------------------
# Convenience: copy constructors between buffers
# --------------------------------

fn copy_bytes_to_new(src: UnsafePointer[UInt8], nbytes: Int) -> BufferU8:
    var out = BufferU8(nbytes, zero_init=False)
    if nbytes > 0:
        memcpy_u8(out.data(), src, nbytes)
    return out

fn copy_buffer(src: BufferU8) -> BufferU8:
    return src.copy()
