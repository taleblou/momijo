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
# Project: momijo.vision
# File: momijo/vision/memory.mojo
 
# -------------------------------
# UInt8 allocation helpers
# -------------------------------

fn alloc_u8(nbytes: Int) -> UnsafePointer[UInt8]:
    # Allocate nbytes of uninitialized memory (bytes)
    return UnsafePointer[UInt8].alloc(nbytes)

fn calloc_u8(nbytes: Int) -> UnsafePointer[UInt8]:
    # Allocate and zero-initialize nbytes
    var ptr = UnsafePointer[UInt8].alloc(nbytes)
    var i = 0
    while i < nbytes:
        ptr[i] = 0
        i += 1
    return ptr

fn free_u8(ptr: UnsafePointer[UInt8]):
    # Placeholder for explicit deallocation when needed by the runtime.
    # Keep for API symmetry with alloc_u8/calloc_u8.
    return

# -------------------------------
# Int32 / Float32 / Float64 helpers
# (Allocate by element count, not bytes)
# -------------------------------

fn alloc_i32(nelems: Int) -> UnsafePointer[Int32]:
    return UnsafePointer[Int32].alloc(nelems)

fn calloc_i32(nelems: Int) -> Pointer[Int32]:
    var p = UnsafePointer[Int32].alloc(nelems)
    var i = 0
    while i < nelems:
        p[i] = 0
        i += 1
    return p

fn free_i32(ptr: UnsafePointer[Int32]):
    return

fn alloc_f32(nelems: Int) -> Pointer[Float32]:
    return UnsafePointer[Float32].alloc(nelems)

fn calloc_f32(nelems: Int) -> UnsafePointer[Float32]:
    var p = UnsafePointer[Float32].alloc(nelems)
    var i = 0
    while i < nelems:
        p[i] = 0.0
        i += 1
    return p

fn free_f32(ptr: UnsafePointer[Float32]):
    return

fn alloc_f64(nelems: Int) -> UnsafePointer[Float64]:
    return UnsafePointer[Float64].alloc(nelems)

fn calloc_f64(nelems: Int) -> UnsafePointer[Float64]:
    var p = UnsafePointer[Float64].alloc(nelems)
    var i = 0
    while i < nelems:
        p[i] = 0.0
        i += 1
    return p

fn free_f64(ptr: UnsafePointer[Float64]):
    return

# -------------------------------
# Byte-level utilities
# -------------------------------

fn memcpy_u8(dst: UnsafePointer[UInt8], src: UnsafePointer[UInt8], nbytes: Int):
    # Safe, simple forward copy (works for non-overlapping regions).
    # For overlapping ranges, prefer memmove semantics (not provided here).
    var i = 0
    while i < nbytes:
        dst[i] = src[i]
        i += 1

fn memset_u8(dst: UnsafePointer[UInt8], value: UInt8, nbytes: Int):
    var i = 0
    while i < nbytes:
        dst[i] = value
        i += 1

fn pointer_add_u8(p: UnsafePointer[UInt8], offset_bytes: Int) -> UnsafePointer[UInt8]:
    # Return a pointer advanced by offset_bytes. Caller must ensure bounds.
    return p + offset_bytes

# -------------------------------
# RAII-like byte buffer (uint8)
# - Owns a UInt8 buffer.
# - Allocates in __init__, frees in __del__ (if runtime supports).
# -------------------------------

 
struct BufferU8:
    var _data: UnsafePointer[UInt8]
    var _length: Int         # length in bytes

    fn __init__(out self, length_bytes: Int, zero_init: Bool = False):
        self._length = length_bytes
        if zero_init:
            self._data = calloc_u8(length_bytes)
        else:
            self._data = alloc_u8(length_bytes)

    fn __del__(deinit self):
        # Keep symmetric with alloc/free. If explicit free is required by the
        # runtime/FFI, hook it here.
        free_u8(self._data)

    fn data(self) -> UnsafePointer[UInt8]:
        return self._data

    fn length(self) -> Int:
        return self._length

    fn zero(self):
        memset_u8(self._data, 0, self._length)

    fn fill(self, value: UInt8):
        memset_u8(self._data, value, self._length)

    fn clone(self) -> BufferU8:
        var out = BufferU8(self._length, zero_init=False)
        memcpy_u8(out._data, self._data, self._length)
        return out

# -------------------------------
# Simple typed views over BufferU8
# (No bounds metadata; caller ensures n fits within _length)
# -------------------------------

fn as_i32_view(buf: BufferU8, nelems: Int) -> UnsafePointer[Int32]:
    # Reinterpret the underlying bytes as Int32 array
    return UnsafePointer[Int32](buf._data)

fn as_f32_view(buf: BufferU8, nelems: Int) -> UnsafePointer[Float32]:
    return UnsafePointer[Float32](buf._data)

fn as_f64_view(buf: BufferU8, nelems: Int) -> UnsafePointer[Float64]:
    return UnsafePointer[Float64](buf._data)

# -------------------------------
# Convenience: copy constructors between buffers
# -------------------------------

fn copy_bytes_to_new(src: UnsafePointer[UInt8], nbytes: Int) -> BufferU8:
    var out = BufferU8(nbytes, zero_init=False)
    memcpy_u8(out.data(), src, nbytes)
    return out

fn copy_buffer(src: BufferU8) -> BufferU8:
    return src.clone()
