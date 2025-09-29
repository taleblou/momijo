# MIT License
# Copyright (c) 2025 Morteza...
# Project: momijo  |  Source: https://github.com/taleblou/momijo
# SPDX-License-Identifier: MIT
# File: src/momijo/vision/tensor.mojo
# Description: Minimal Tensor using UnsafePointer with ownership flag; HWC/CHW helpers.

from momijo.vision.dtypes import DType

# -------------------------------------------------------------------
# Size in bytes (delegate to DType)
# -------------------------------------------------------------------
fn dtype_bytes(dt: DType) -> Int:
    return dt.nbytes()

# -------------------------------------------------------------------
# Stride helpers (logical strides are in ELEMENTS, not bytes)
# -------------------------------------------------------------------
fn packed_hwc_strides(h: Int, w: Int, c: Int, item_size: Int = 1) -> (Int, Int, Int):
    var s2 = item_size         # channel step
    var s1 = c * s2            # step per column (W)
    var s0 = w * s1            # step per row (H)
    return (s0, s1, s2)

fn packed_chw_strides(c: Int, h: Int, w: Int) -> (Int, Int, Int):
    var s2 = 1                 # W contiguous
    var s1 = w                 # step per row (H)
    var s0 = h * w             # step per channel (C)
    return (s0, s1, s2)

# -------------------------------------------------------------------
# Tensor (ExplicitlyCopyable & Movable)
# Owns a raw UnsafePointer[UInt8] buffer optionally.
# -------------------------------------------------------------------
struct Tensor(ExplicitlyCopyable, Movable):
    var _data: UnsafePointer[UInt8]
    var _len: Int
    var _dtype: DType
    var _ndim: Int
    var _shape0: Int
    var _shape1: Int
    var _shape2: Int
    var _stride0: Int
    var _stride1: Int
    var _stride2: Int
    var _owns: Bool   # whether this Tensor is responsible to free _data

    # Empty tensor
    fn __init__(out self):
        self._data = UnsafePointer[UInt8]()  # default null
        self._len = 0
        self._dtype = DType.UInt8()
        self._ndim = 0
        self._shape0 = 0; self._shape1 = 0; self._shape2 = 0
        self._stride0 = 0; self._stride1 = 0; self._stride2 = 0
        self._owns = False

    # Raw-buffer constructor; by default we "own" the buffer and will free it.
    fn __init__(out self,
                ptr: UnsafePointer[UInt8],
                nbytes: Int,
                h: Int, w: Int, c: Int,
                s0: Int, s1: Int, s2: Int,
                dtype: DType,
                owns: Bool = True):
        self._data = ptr
        self._len = nbytes
        self._dtype = dtype
        self._ndim = 3
        self._shape0 = h; self._shape1 = w; self._shape2 = c
        self._stride0 = s0; self._stride1 = s1; self._stride2 = s2
        self._owns = owns

    # Shallow copy; does NOT duplicate bytes; preserves ownership flag.
    fn __copyinit__(out self, other: Tensor):
        self._data = other._data
        self._len = other._len
        self._dtype = other._dtype
        self._ndim = other._ndim
        self._shape0 = other._shape0; self._shape1 = other._shape1; self._shape2 = other._shape2
        self._stride0 = other._stride0; self._stride1 = other._stride1; self._stride2 = other._stride2
        self._owns    = False

    fn __del__(deinit self):
        # UnsafePointer[...] has no is_null(); use length guard instead.
        if self._owns and self._len > 0:
            UnsafePointer[UInt8].free(self._data)
            # defensive: clear fields to avoid accidental reuse
            self._data = UnsafePointer[UInt8]()
            self._len = 0

    # ---- basic API ----
    fn dtype(self) -> DType: return self._dtype
    fn ndim(self) -> Int: return self._ndim
    fn height(self) -> Int: return self._shape0
    fn width(self) -> Int: return self._shape1
    fn channels(self) -> Int: return self._shape2
    fn shape0(self) -> Int: return self._shape0
    fn shape1(self) -> Int: return self._shape1
    fn shape2(self) -> Int: return self._shape2
    fn stride0(self) -> Int: return self._stride0
    fn stride1(self) -> Int: return self._stride1
    fn stride2(self) -> Int: return self._stride2

    fn nbytes(self) -> Int: return self._len

    # For sites calling `.data()` (kept for compatibility)
    fn data(self) -> UnsafePointer[UInt8]:
        return self._data

    # Typical name used elsewhere
    fn data_ptr(self) -> UnsafePointer[UInt8]:
        return self._data

    fn is_empty(self) -> Bool:
        # Do not call is_null() on UnsafePointer; length is authoritative here.
        return self._len == 0

    # ---- layout checks ----
    fn is_contiguous_hwc_u8(self) -> Bool:
        var cond_stride2 = (self._stride2 == 1)
        var cond_stride1 = (self._stride1 == self._shape2)
        var cond_stride0 = (self._stride0 == self._shape1 * self._shape2)
        var cond_dtype   = (self._dtype == DType.UInt8())
        return cond_stride2 and cond_stride1 and cond_stride0 and cond_dtype

    fn clone(self) -> Tensor:
        return self.clone_bytes() 
    # ---- cloning / copies ----
    fn clone_bytes(self) -> Tensor:
        var n = self._len
        var buf = UnsafePointer[UInt8].alloc(n)
        var i = 0
        while i < n:
            buf[i] = self._data[i]
            i = i + 1
        # cloned buffer is owned by the new Tensor
        return Tensor(buf, n, self._shape0, self._shape1, self._shape2,
                      self._stride0, self._stride1, self._stride2, self._dtype, True)

    fn copy_to_packed_hwc(self) -> Tensor:
        var h = self._shape0
        var w = self._shape1
        var c = self._shape2
        var (s0, s1, s2) = packed_hwc_strides(h, w, c)
        var elem_size = dtype_bytes(self._dtype)
        var out_len = h * w * c * elem_size
        var out_buf = UnsafePointer[UInt8].alloc(out_len)
        var out = Tensor(out_buf, out_len, h, w, c, s0, s1, s2, self._dtype, True)

        if self._dtype == DType.UInt8():
            var y = 0
            while y < h:
                var x = 0
                while x < w:
                    var ch = 0
                    while ch < c:
                        out._data[y*s0 + x*s1 + ch*s2] =
                            self._data[y*self._stride0 + x*self._stride1 + ch*self._stride2]
                        ch = ch + 1
                    x = x + 1
                y = y + 1
        else:
            var i = 0
            while i < out_len and i < self._len:
                out._data[i] = self._data[i]
                i = i + 1

        return out

    # ---- convenience pointer accessors for u8 ----
    fn ptr_u8(self) -> UnsafePointer[UInt8]:
        return self._data

    fn store_u8_at(mut self, idx: Int, v: UInt8):
        self._data[idx] = v

    fn load_u8_at(self, idx: Int) -> UInt8:
        return self._data[idx]

    # ---- ROI copy (safe) ----
    fn copy_roi(self, y_in: Int, x_in: Int, h_in: Int, w_in: Int) -> Tensor:
        # Guard invalid sizes
        var h = h_in
        var w = w_in
        if h <= 0 or w <= 0:
            return Tensor()

        # Clamp ROI to source bounds
        var y = y_in
        var x = x_in
        if y < 0: y = 0
        if x < 0: x = 0
        var max_h = self._shape0 - y
        var max_w = self._shape1 - x
        if max_h <= 0 or max_w <= 0:
            return Tensor()
        if h > max_h: h = max_h
        if w > max_w: w = max_w

        var c = self._shape2
        var (s0, s1, s2) = packed_hwc_strides(h, w, c)
        var elem_size = dtype_bytes(self._dtype)
        var out_len = h * w * c * elem_size
        var out_buf = UnsafePointer[UInt8].alloc(out_len)
        var out = Tensor(out_buf, out_len, h, w, c, s0, s1, s2, self._dtype, True)

        if self._dtype == DType.UInt8():
            var yy = 0
            while yy < h:
                var xx = 0
                while xx < w:
                    var ch = 0
                    while ch < c:
                        out._data[yy*s0 + xx*s1 + ch*s2] =
                            self._data[(y+yy)*self._stride0 + (x+xx)*self._stride1 + ch*self._stride2]
                        ch = ch + 1
                    xx = xx + 1
                yy = yy + 1
        else:
            var i = 0
            while i < out_len and i < self._len:
                out._data[i] = self._data[i]
                i = i + 1

        return out

    # ---- static constructors (owning) ----
    @staticmethod
    fn empty_1d(length_in: Int, dtype: DType = DType.UInt8()) -> Tensor:
        var length = length_in
        if length <= 0:
            return Tensor()
        var item = dtype_bytes(dtype)
        var nbytes = length * item
        var buf = UnsafePointer[UInt8].alloc(nbytes)
        # zero-init
        var i = 0
        while i < nbytes:
            buf[i] = 0
            i = i + 1
        # packed 1D view: (length,1,1) with stride0=item
        return Tensor(buf, nbytes, length, 1, 1, item, 0, 0, dtype, True)

    @staticmethod
    fn empty_hwc(h_in: Int, w_in: Int, c_in: Int, dtype: DType = DType.UInt8()) -> Tensor:
        var h = h_in; var w = w_in; var c = c_in
        if h <= 0 or w <= 0 or c <= 0:
            return Tensor()
        var item = dtype_bytes(dtype)
        var nbytes = h * w * c * item
        var buf = UnsafePointer[UInt8].alloc(nbytes)
        # zero-init
        var i = 0
        while i < nbytes:
            buf[i] = 0
            i = i + 1
        var (s0, s1, s2) = packed_hwc_strides(h, w, c, 1)
        return Tensor(buf, nbytes, h, w, c, s0, s1, s2, dtype, True)

# -------------------------------------------------------------------
# ToTensor: HWC(u8) -> Float32 array (raw pointer)
# -------------------------------------------------------------------
struct ToTensor:
    var _scale: Float32
    var _to_chw: Bool

    fn __init__(out self, scale: Float32 = 1.0/255.0, to_chw: Bool = True):
        self._scale = scale
        self._to_chw = to_chw

    fn __call__(self, src_ptr: UnsafePointer[UInt8], h_in: Int, w_in: Int, c_in: Int) -> UnsafePointer[Float32]:
        var h = h_in; var w = w_in; var c = c_in
        if h <= 0 or w <= 0 or c <= 0:
            # Return empty pointer on invalid shape
            return UnsafePointer[Float32]()
        if self._to_chw:
            return _to_chw(self._scale, src_ptr, h, w, c)
        else:
            return _to_hwc(self._scale, src_ptr, h, w, c)

# Internal helpers (HWC u8 -> CHW/HWC f32)
fn _to_chw(scale: Float32, src: UnsafePointer[UInt8], h: Int, w: Int, c: Int) -> UnsafePointer[Float32]:
    var s0 = w * c
    var s1 = c
    var s2 = 1

    var elems = c * h * w
    var dst = UnsafePointer[Float32].alloc(elems)

    var dC = h * w
    var dH = w
    var dW = 1

    var ch = 0
    while ch < c:
        var y = 0
        while y < h:
            var x = 0
            while x < w:
                var src_idx = y * s0 + x * s1 + ch * s2
                var dst_idx = ch * dC + y * dH + x * dW
                dst[dst_idx] = Float32(src[src_idx]) * scale
                x = x + 1
            y = y + 1
        ch = ch + 1
    return dst

fn _to_hwc(scale: Float32, src: UnsafePointer[UInt8], h: Int, w: Int, c: Int) -> UnsafePointer[Float32]:
    var s0 = w * c
    var s1 = c
    var s2 = 1

    var elems = c * h * w
    var dst = UnsafePointer[Float32].alloc(elems)

    var d0 = w * c
    var d1 = c
    var d2 = 1

    var y = 0
    while y < h:
        var x = 0
        while x < w:
            var ch = 0
            while ch < c:
                var src_idx = y * s0 + x * s1 + ch * s2
                var dst_idx = y * d0 + x * d1 + ch * d2
                dst[dst_idx] = Float32(src[src_idx]) * scale
                ch = ch + 1
            x = x + 1
        y = y + 1
    return dst

# -------------------------------------------------------------------
# Convenience wrappers for ToTensor
# -------------------------------------------------------------------
fn to_tensor_chw(src_ptr: UnsafePointer[UInt8], h: Int, w: Int, c: Int, scale: Float32 = 1.0/255.0) -> UnsafePointer[Float32]:
    var op = ToTensor(scale, True)
    return op(src_ptr, h, w, c)

fn to_tensor_hwc(src_ptr: UnsafePointer[UInt8], h: Int, w: Int, c: Int, scale: Float32 = 1.0/255.0) -> UnsafePointer[Float32]:
    var op = ToTensor(scale, False)
    return op(src_ptr, h, w, c)

# -------------------------------------------------------------------
# Raw buffer wrappers (no copy). Caller controls lifetime.
# -------------------------------------------------------------------
fn unsafe_view_from_raw_u8(ptr: UnsafePointer[UInt8], nbytes_in: Int,
                           h_in: Int, w_in: Int, c_in: Int,
                           s0: Int, s1: Int, s2: Int, dt: DType) -> Tensor:
    var nbytes = nbytes_in
    var h = h_in; var w = w_in; var c = c_in
    if nbytes <= 0 or h <= 0 or w <= 0 or c <= 0 or s0 <= 0 or s1 <= 0 or s2 <= 0:
        return Tensor()
    # owns = False → caller must free 'ptr'
    return Tensor(ptr, nbytes, h, w, c, s0, s1, s2, dt, False)

fn tensor_from_u8_array_hwc(data: UnsafePointer[UInt8], nbytes_in: Int, h_in: Int, w_in: Int, c_in: Int) -> Tensor:
    var h = h_in; var w = w_in; var c = c_in
    var nbytes = nbytes_in
    if h <= 0 or w <= 0 or c <= 0:
        return Tensor()
    var (s0, s1, s2) = packed_hwc_strides(h, w, c)
    var min_bytes = h * w * c
    if nbytes < min_bytes:
        return Tensor()
    # owns = False → caller must free 'data'
    return Tensor(data, nbytes, h, w, c, s0, s1, s2, DType.UInt8(), False)

fn array_from_tensor_u8_hwc(t: Tensor) -> UnsafePointer[UInt8]:
    return t.data()

fn tensor_from_raw_chw(ptr: UnsafePointer[UInt8], nbytes_in: Int,
                       c_in: Int, h_in: Int, w_in: Int, dt: DType) -> Tensor:
    var c = c_in; var h = h_in; var w = w_in
    var nbytes = nbytes_in
    if c <= 0 or h <= 0 or w <= 0:
        return Tensor()
    var (s0, s1, s2) = packed_chw_strides(c, h, w)
    var elems = c * h * w
    var need = elems * dtype_bytes(dt)
    if nbytes < need:
        return Tensor()
    # owns = False
    return Tensor(ptr, nbytes, c, h, w, s0, s1, s2, dt, False)

fn tensor_from_raw_strided(ptr: UnsafePointer[UInt8],
                           nbytes_in: Int,
                           d0_in: Int, d1_in: Int, d2_in: Int,
                           s0: Int, s1: Int, s2: Int,
                           dt: DType) -> Tensor:
    var nbytes = nbytes_in
    var d0 = d0_in; var d1 = d1_in; var d2 = d2_in
    if d0 <= 0 or d1 <= 0 or d2 <= 0 or s0 <= 0 or s1 <= 0 or s2 <= 0 or nbytes <= 0:
        return Tensor()
    # owns = False
    return Tensor(ptr, nbytes, d0, d1, d2, s0, s1, s2, dt, False)
