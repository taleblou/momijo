# MIT License
# SPDX-License-Identifier: MIT
# Project: momijo.tensor
# File: src/momijo/tensor/device.mojo
# Description: Single-file CPU/GPU device handling with runtime switching.

from momijo.tensor.tensor import Tensor
from momijo.tensor.helpers import compute_row_major_strides
from sys.info import has_accelerator

# =============================================================================
# Device + runtime context
# =============================================================================
struct Device(ImplicitlyCopyable, Copyable, Movable):
    var tag: Int8  # 0=cpu, 1=gpu

    fn __init__(out self, t: Int8):
        self.tag = t

    fn __copyinit__(out self, other: Device):
        self.tag = other.tag

    @staticmethod
    fn cpu() -> Device:
        return Device(0)

    @staticmethod
    fn gpu() -> Device:
        return Device(1)

    fn is_cpu(self) -> Bool:
        return self.tag == 0

    fn is_gpu(self) -> Bool:
        return self.tag == 1

    fn __eq__(self, other: Device) -> Bool:
        return self.tag == other.tag

    fn __ne__(self, other: Device) -> Bool:
        return self.tag != other.tag

    fn __str__(self) -> String:
        return "cpu" if self.tag == 0 else "gpu"


struct RuntimeDeviceContext(ImplicitlyCopyable, Copyable, Movable):
    var current: Device

    fn __init__(out self, initial: Device):
        self.current = initial

    fn __copyinit__(out self, other: RuntimeDeviceContext):
        self.current = other.current

    @staticmethod
    fn cpu() -> RuntimeDeviceContext:
        return RuntimeDeviceContext(Device.cpu())

    @staticmethod
    fn gpu() -> RuntimeDeviceContext:
        return RuntimeDeviceContext(Device.gpu())

    fn set_cpu(mut self) -> None:
        self.current = Device.cpu()

    fn set_gpu(mut self) -> None:
        self.current = Device.gpu()

    fn get(self) -> Device:
        return self.current

    fn __str__(self) -> String:
        return self.current.__str__()


struct ScopedDevice:
    var ctx_ptr: Pointer[RuntimeDeviceContext]
    var saved: Device

    fn __init__(out self, ctx_ptr: Pointer[RuntimeDeviceContext], new_dev: Device):
        assert(ctx_ptr != Pointer, "ScopedDevice: ctx_ptr is NULL")
        self.ctx_ptr = ctx_ptr
        var v = self.ctx_ptr.load()
        self.saved = v.get()
        if new_dev.is_gpu():
            v.set_gpu()
        else:
            v.set_cpu()
        self.ctx_ptr.store(v)

    fn __del__(deinit self):
        var v = self.ctx_ptr.load()
        if self.saved.is_gpu():
            v.set_gpu()
        else:
            v.set_cpu()
        self.ctx_ptr.store(v)


# =============================================================================
# Resolution helpers
# =============================================================================
fn resolve_device(ctx: Optional[RuntimeDeviceContext]) -> Device:
    if ctx is None:
        return Device.cpu()
    return ctx.value().get()

fn resolve_effective_device(ctx: Optional[RuntimeDeviceContext]) -> Device:
    var req = resolve_device(ctx)
    if req.is_gpu() and not has_accelerator():
        # Requested GPU but system has no accelerator -> fallback to CPU
        return Device.cpu()
    return req

fn get_device_name(ctx: Optional[RuntimeDeviceContext] = None) -> String:
    return resolve_effective_device(ctx).__str__()

fn set_device_cpu(mut ctx: RuntimeDeviceContext) -> RuntimeDeviceContext:
    ctx.set_cpu()
    return ctx

fn set_device_gpu(mut ctx: RuntimeDeviceContext) -> RuntimeDeviceContext:
    ctx.set_gpu()
    return ctx


# =============================================================================
# CPU alloc helpers
# =============================================================================
@always_inline
fn _numel(shape: List[Int]) -> Int:
    var n = 1
    var i = 0
    while i < len(shape):
        n = n * shape[i]
        i += 1
    return n

fn _alloc_fill_f64_cpu(shape: List[Int], val: Float64) -> Tensor[Float64]:
    var n = _numel(shape)
    var data = List[Float64]()
    data.reserve(n)
    var j = 0
    var lim = (n // 16) * 16
    while j < lim:
        data.append(val); data.append(val); data.append(val); data.append(val)
        data.append(val); data.append(val); data.append(val); data.append(val)
        data.append(val); data.append(val); data.append(val); data.append(val)
        data.append(val); data.append(val); data.append(val); data.append(val)
        j += 16
    while j < n:
        data.append(val)
        j += 1
    var strides = compute_row_major_strides(shape)
    return Tensor[Float64](data, shape, strides,0)

fn _alloc_fill_f32_cpu(shape: List[Int], val: Float32) -> Tensor[Float32]:
    var n = _numel(shape)
    var data = List[Float32]()
    data.reserve(n)
    var j = 0
    var lim = (n // 16) * 16
    while j < lim:
        data.append(val); data.append(val); data.append(val); data.append(val)
        data.append(val); data.append(val); data.append(val); data.append(val)
        data.append(val); data.append(val); data.append(val); data.append(val)
        data.append(val); data.append(val); data.append(val); data.append(val)
        j += 16
    while j < n:
        data.append(val)
        j += 1
    var strides = compute_row_major_strides(shape)
    return Tensor[Float32](data, shape, strides,0)

fn _alloc_fill_i_cpu(shape: List[Int], val: Int) -> Tensor[Int]:
    var n = _numel(shape)
    var data = List[Int]()
    data.reserve(n)
    var j = 0
    var lim = (n // 16) * 16
    while j < lim:
        data.append(val); data.append(val); data.append(val); data.append(val)
        data.append(val); data.append(val); data.append(val); data.append(val)
        data.append(val); data.append(val); data.append(val); data.append(val)
        data.append(val); data.append(val); data.append(val); data.append(val)
        j += 16
    while j < n:
        data.append(val)
        j += 1
    var strides = compute_row_major_strides(shape)
    return Tensor[Int](data, shape, strides,0)


# =============================================================================
# GPU helpers (imports and kernels *inside* functions)
# =============================================================================
fn _alloc_fill_f32_gpu(shape: List[Int], val: Float32) raises -> Tensor[Float32]:
    from gpu.host import DeviceContext
    from gpu.id import block_dim, block_idx, thread_idx

    @always_inline
    fn k_fill(vec: UnsafePointer[Float32], size: Int, v: Float32):
        var idx = block_idx.x * block_dim.x + thread_idx.x
        if idx < UInt(size):
            vec[idx] = v

    var n = _numel(shape)
    var dc = DeviceContext()
    var hb = dc.enqueue_create_host_buffer[DType.float32](n)
    var db = dc.enqueue_create_buffer[DType.float32](n)

    var k = dc.compile_function[k_fill]()
    var tpb = 256
    var blocks = (n + tpb - 1) // tpb
    dc.enqueue_function(k, db, n, val, grid_dim=blocks, block_dim=tpb)

    dc.enqueue_copy(src_buf=db, dst_buf=hb)
    dc.synchronize()

    var out = List[Float32](); out.reserve(n)
    var i = 0
    while i < n:
        out.append(hb[i]); i += 1
    var strides = compute_row_major_strides(shape)
    return Tensor[Float32](out, shape, strides,0)

fn _alloc_fill_f64_gpu(shape: List[Int], val: Float64) raises -> Tensor[Float64]:
    from gpu.host import DeviceContext
    from gpu.id import block_dim, block_idx, thread_idx

    @always_inline
    fn k_fill(vec: UnsafePointer[Float64], size: Int, v: Float64):
        var idx = block_idx.x * block_dim.x + thread_idx.x
        if idx < UInt(size):
            vec[idx] = v

    var n = _numel(shape)
    var dc = DeviceContext()
    var hb = dc.enqueue_create_host_buffer[DType.float64](n)
    var db = dc.enqueue_create_buffer[DType.float64](n)

    var k = dc.compile_function[k_fill]()
    var tpb = 256
    var blocks = (n + tpb - 1) // tpb
    dc.enqueue_function(k, db, n, val, grid_dim=blocks, block_dim=tpb)

    dc.enqueue_copy(src_buf=db, dst_buf=hb)
    dc.synchronize()

    var out = List[Float64](); out.reserve(n)
    var i = 0
    while i < n:
        out.append(hb[i]); i += 1
    var strides = compute_row_major_strides(shape)
    return Tensor[Float64](out, shape, strides,0)

fn _to_device_f32_gpu(x: Tensor[Float32]) raises -> Tensor[Float32]:
    from gpu.host import DeviceContext

    var n = len(x._data)
    var dc = DeviceContext()
    var hb = dc.enqueue_create_host_buffer[DType.float32](n)

    var i = 0
    while i < n:
        hb[i] = x._data[i]
        i += 1

    var db = dc.enqueue_create_buffer[DType.float32](n)
    dc.enqueue_copy(src_buf=hb, dst_buf=db)
    dc.enqueue_copy(src_buf=db, dst_buf=hb)
    dc.synchronize()

    var out = List[Float32](); out.reserve(n)
    var j = 0
    while j < n:
        out.append(hb[j]); j += 1
    return Tensor[Float32](out, x._shape.copy(), x._strides.copy(),x._offset)

fn _to_device_f64_gpu(x: Tensor[Float64]) raises -> Tensor[Float64]:
    from gpu.host import DeviceContext

    var n = len(x._data)
    var dc = DeviceContext()
    var hb = dc.enqueue_create_host_buffer[DType.float64](n)

    var i = 0
    while i < n:
        hb[i] = x._data[i]
        i += 1

    var db = dc.enqueue_create_buffer[DType.float64](n)
    dc.enqueue_copy(src_buf=hb, dst_buf=db)
    dc.enqueue_copy(src_buf=db, dst_buf=hb)
    dc.synchronize()

    var out = List[Float64](); out.reserve(n)
    var j = 0
    while j < n:
        out.append(hb[j]); j += 1
    return Tensor[Float64](out, x._shape.copy(), x._strides.copy(),x._offset)


# =============================================================================
# Public allocators with runtime switching
# =============================================================================
fn _want_gpu(ctx: Optional[RuntimeDeviceContext]) -> Bool:
    return resolve_device(ctx).is_gpu() and has_accelerator()

fn zeros_f64(shape: List[Int], ctx: Optional[RuntimeDeviceContext] = None) -> Tensor[Float64]:
    if _want_gpu(ctx):
        try:
            return _alloc_fill_f64_gpu(shape, 0.0)
        except e:
            pass
    return _alloc_fill_f64_cpu(shape, 0.0)

fn ones_f64(shape: List[Int], ctx: Optional[RuntimeDeviceContext] = None) -> Tensor[Float64]:
    if _want_gpu(ctx):
        try:
            return _alloc_fill_f64_gpu(shape, 1.0)
        except e:
            pass
    return _alloc_fill_f64_cpu(shape, 1.0)

fn zeros_f32(shape: List[Int], ctx: Optional[RuntimeDeviceContext] = None) -> Tensor[Float32]:
    if _want_gpu(ctx):
        try:
            return _alloc_fill_f32_gpu(shape, Float32(0.0))
        except e:
            pass
    return _alloc_fill_f32_cpu(shape, Float32(0.0))

fn ones_f32(shape: List[Int], ctx: Optional[RuntimeDeviceContext] = None) -> Tensor[Float32]:
    if _want_gpu(ctx):
        try:
            return _alloc_fill_f32_gpu(shape, Float32(1.0))
        except e:
            pass
    return _alloc_fill_f32_cpu(shape, Float32(1.0))

fn zeros_i(shape: List[Int], ctx: Optional[RuntimeDeviceContext] = None) -> Tensor[Int]:
    var _ = ctx
    return _alloc_fill_i_cpu(shape, 0)

fn ones_i(shape: List[Int], ctx: Optional[RuntimeDeviceContext] = None) -> Tensor[Int]:
    var _ = ctx
    return _alloc_fill_i_cpu(shape, 1)


# =============================================================================
# Device transfer and query
# =============================================================================
fn device_of[T: ImplicitlyCopyable & Copyable & Movable](x: Tensor[T]) -> Device:
    # Placeholder until Tensor carries device info.
    return Device.cpu()

fn to_device(x: Tensor[Float32], target: Device) -> Tensor[Float32]:
    if target.is_gpu() and has_accelerator():
        try:
            return _to_device_f32_gpu(x)
        except e:
            pass
    return Tensor[Float32](x._data.copy(), x._shape.copy(), x._strides.copy(),x._offset)

fn to_device(x: Tensor[Float64], target: Device) -> Tensor[Float64]:
    if target.is_gpu() and has_accelerator():
        try:
            return _to_device_f64_gpu(x)
        except e:
            pass
    return Tensor[Float64](x._data.copy(), x._shape.copy(), x._strides.copy(),x._offset)

fn to_device[T: ImplicitlyCopyable & Copyable & Movable](x: Tensor[T], target: Device) -> Tensor[T]:
    var _ = target
    return Tensor[T](x._data.copy(), x._shape.copy(), x._strides.copy(),x._offset)


# =============================================================================
# Generic dispatch helpers (effective device at runtime)
# =============================================================================
fn dispatch_unary_f64(
    x: Tensor[Float64],
    ctx: Optional[RuntimeDeviceContext],
    cpu_impl: fn(Tensor[Float64]) -> Tensor[Float64],
    gpu_impl: fn(Tensor[Float64]) -> Tensor[Float64]
) -> Tensor[Float64]:
    if _want_gpu(ctx):
        return gpu_impl(x)
    return cpu_impl(x)

fn dispatch_binary_f64(
    a: Tensor[Float64],
    b: Tensor[Float64],
    ctx: Optional[RuntimeDeviceContext],
    cpu_impl: fn(Tensor[Float64], Tensor[Float64]) -> Tensor[Float64],
    gpu_impl: fn(Tensor[Float64], Tensor[Float64]) -> Tensor[Float64]
) -> Tensor[Float64]:
    if _want_gpu(ctx):
        return gpu_impl(a, b)
    return cpu_impl(a, b)

fn dispatch_unary_f32(
    x: Tensor[Float32],
    ctx: Optional[RuntimeDeviceContext],
    cpu_impl: fn(Tensor[Float32]) -> Tensor[Float32],
    gpu_impl: fn(Tensor[Float32]) -> Tensor[Float32]
) -> Tensor[Float32]:
    if _want_gpu(ctx):
        return gpu_impl(x)
    return cpu_impl(x)

fn dispatch_binary_f32(
    a: Tensor[Float32],
    b: Tensor[Float32],
    ctx: Optional[RuntimeDeviceContext],
    cpu_impl: fn(Tensor[Float32], Tensor[Float32]) -> Tensor[Float32],
    gpu_impl: fn(Tensor[Float32], Tensor[Float32]) -> Tensor[Float32]
) -> Tensor[Float32]:
    if _want_gpu(ctx):
        return gpu_impl(a, b)
    return cpu_impl(a, b)

fn dispatch_unary_i(
    x: Tensor[Int],
    ctx: Optional[RuntimeDeviceContext],
    cpu_impl: fn(Tensor[Int]) -> Tensor[Int],
    gpu_impl: fn(Tensor[Int]) -> Tensor[Int]
) -> Tensor[Int]:
    var _ = ctx
    return cpu_impl(x)

fn dispatch_binary_i(
    a: Tensor[Int],
    b: Tensor[Int],
    ctx: Optional[RuntimeDeviceContext],
    cpu_impl: fn(Tensor[Int], Tensor[Int]) -> Tensor[Int],
    gpu_impl: fn(Tensor[Int], Tensor[Int]) -> Tensor[Int]
) -> Tensor[Int]:
    var _ = ctx
    return cpu_impl(a, b)
