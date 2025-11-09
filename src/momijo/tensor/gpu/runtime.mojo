# MIT License
# SPDX-License-Identifier: MIT
# Project:      momijo
# File:         src/momijo/gpu/runtime.mojo
# Authors:      Morteza Taleblou & Mitra Daneshmand
# Website:      https://taleblou.ir/
# Repository:   https://github.com/taleblou/momijo


from momijo.tensor import tensor
from sys.info import has_accelerator, has_nvidia_gpu_accelerator, has_amd_gpu_accelerator, has_apple_gpu_accelerator
from collections.list import List

# If your project already wraps gpu stdlib, import from there.
# Otherwise, use Modular GPU stdlib directly (no wildcard).
from gpu import global_idx, block_dim, grid_dim
from gpu.host import DeviceContext
from gpu.globals import WARP_SIZE
from gpu import profiler
from layout import Layout, LayoutTensor
from math import sqrt
from math import sin, cos, sqrt
from gpu import block_dim, block_idx, thread_idx

# ------------------------------
# GPU availability helpers
# ------------------------------
@always_inline
fn gpu_available() -> Bool:
    if has_nvidia_gpu_accelerator():
        return True
    if has_amd_gpu_accelerator():
        return True
    if has_apple_gpu_accelerator():
        return True
    return has_accelerator()

@always_inline
fn _off2(H: Int, W: Int, h: Int, w: Int) -> Int:
    # row-major [H,W]
    return h * W + w

@always_inline
fn _off3(C: Int, H: Int, W: Int, c: Int, h: Int, w: Int) -> Int:
    # CHW contiguous: c-blocks of (H*W)
    return c * (H * W) + h * W + w


# === GPU kernels: transpose2d, linear fw/bw, maxpool fw, conv2d fw ===

# Linear forward: y[n, out] = dot(x[n, :], wT[out, :]) + b[out]
fn _gpu_linear_fw_kernel(x: UnsafePointer[Float32],
                         wT: UnsafePointer[Float32],
                         b: UnsafePointer[Float32],
                         y: UnsafePointer[Float32],
                         N: Int, InF: Int, OutF: Int, has_bias_i: Int):

                        var stride = block_dim.x * grid_dim.x
                        var tid = block_idx.x * block_dim.x + thread_idx.x
                        var total = N * OutF
                        var i = tid
                        while i < total:
                            var n = i // OutF
                            var o = i - n * OutF
                            var acc: Float32 = 0.0
                            var k = 0
                            while k < InF:
                                acc = acc + x[n*InF + k] * wT[o*InF + k]
                                k = k + 1
                            if has_bias_i != 0:
                                acc = acc + b[o]
                            y[n*OutF + o] = acc
                            i = i + stride

fn _gpu_linear_bw_dx_kernel(dy: UnsafePointer[Float32],
                            w:  UnsafePointer[Float32],
                            dx: UnsafePointer[Float32],
                            N: Int, InF: Int, OutF: Int):
    var tid = block_idx.x * block_dim.x + thread_idx.x
    var total = N * InF
    if tid < total:
        var n = tid // InF
        var k = tid %  InF
        var acc: Float32 = 0.0
        var o = 0
        while o < OutF:
            acc = acc + dy[n*OutF + o] * w[k*OutF + o]
            o = o + 1
        dx[n*InF + k] = acc

# Linear backward dW: dW[k,o] = sum_n x[n,k] * dy[n,o]
fn _gpu_linear_bw_dw_kernel(dy: UnsafePointer[Float32],
                            x:  UnsafePointer[Float32],
                            dW: UnsafePointer[Float32],
                            N: Int, InF: Int, OutF: Int):
    var tid = block_idx.x * block_dim.x + thread_idx.x
    var total = InF * OutF
    if tid < total:
        var k = tid // OutF
        var o = tid %  OutF
        var acc: Float32 = 0.0
        var n = 0
        while n < N:
            acc = acc + x[n*InF + k] * dy[n*OutF + o]
            n = n + 1
        dW[k*OutF + o] = acc

# Linear backward db: db[o] = sum_n dy[n,o]
fn _gpu_linear_bw_db_kernel(dy: UnsafePointer[Float32],
                            db: UnsafePointer[Float32],
                            N: Int, OutF: Int):
    var o = block_idx.x * block_dim.x + thread_idx.x
    if o < OutF:
        var acc: Float32 = 0.0
        var n = 0
        while n < N:
            acc = acc + dy[n*OutF + o]
            n = n + 1
        db[o] = acc

# MaxPool forward (NCHW). Each thread computes one output (n,c,oh,ow).
fn _gpu_maxpool_fw_kernel(x: UnsafePointer[Float32],
                          y: UnsafePointer[Float32],
                          N: Int, C: Int, H: Int, W: Int,
                          OH: Int, OW: Int,
                          kH: Int, kW: Int,
                          sH: Int, sW: Int,
                          pH: Int, pW: Int):
    var tid = block_idx.x * block_dim.x + thread_idx.x
    var total = N * C * OH * OW
    if tid < total:
        var ow = tid % OW
        var tmp = tid // OW
        var oh = tmp % OH
        tmp = tmp // OH
        var c = tmp % C
        var n = tmp // C

        var h_start = oh * sH - pH
        var w_start = ow * sW - pW
        var h_end = h_start + kH
        var w_end = w_start + kW

        var maxv: Float32 = -3.4028235e+38  # -FLT_MAX
        var h = h_start
        while h < h_end:
            var w2 = w_start
            while w2 < w_end:
                if (h >= 0) and (h < H) and (w2 >= 0) and (w2 < W):
                    var idx = (((n*C + c) * H) + h) * W + w2
                    var v = x[idx]
                    if v > maxv:
                        maxv = v
                w2 = w2 + 1
            h = h + 1
        y[tid] = maxv

# Conv2D forward (NCHW) naive: y[n,oc,oh,ow] = sum_{c,kh,kw} x * w + b
fn _gpu_conv2d_fw_kernel(x: UnsafePointer[Float32],
                         w: UnsafePointer[Float32],
                         b: UnsafePointer[Float32],
                         y: UnsafePointer[Float32],
                         N: Int, C: Int, H: Int, W: Int,
                         OH: Int, OW: Int,
                         OC: Int,
                         kH: Int, kW: Int,
                         sH: Int, sW: Int,
                         pH: Int, pW: Int, has_bias_i: Int):
    var stride = block_dim.x * grid_dim.x
    var tid = block_idx.x * block_dim.x + thread_idx.x
    var total = N * OC * OH * OW
    var i = tid
    while i < total:
        var ow = i % OW
        var tmp0 = i // OW
        var oh = tmp0 % OH
        var tmp1 = tmp0 // OH
        var oc = tmp1 % OC
        var n  = tmp1 // OC

        var acc: Float32 = 0.0
        var ic = 0
        while ic < C:
            var kh = 0
            while kh < kH:
                var ih = oh * sH - pH + kh
                var kw = 0
                while kw < kW:
                    var iw = ow * sW - pW + kw
                    if ih >= 0 and ih < H and iw >= 0 and iw < W:
                        var xv = x[((n*C + ic) * H + ih) * W + iw]
                        var wv = w[(((oc*C + ic) * kH + kh) * kW + kw)]
                        acc = acc + xv * wv
                    kw = kw + 1
                kh = kh + 1
            ic = ic + 1
        if has_bias_i != 0:
            acc = acc + b[oc]
        y[(((n*OC) + oc) * OH + oh) * OW + ow] = acc
        i = i + stride

fn memset_f32(mut buf: List[Float32], value: Float32) -> None:
    var i = 0
    var n = len(buf)
    while i < n:
        buf[i] = value
        i = i + 1

fn atomic_add_f32(mut buf: List[Float32], idx: Int, val: Float32) -> None:
    # Non-atomic CPU fallback
    buf[idx] = buf[idx] + val

# ------------------------------
# Kernel type aliases (use List buffers)
# Note: outputs are 'mut' so kernels can write into them.
# ------------------------------
alias Kernel1D_MaxPoolFW = fn(
    tid: Int,
    x:  List[Float32],
    mut y:  List[Float32],
    N: Int, C: Int, H: Int, W: Int,
    OH: Int, OW: Int,
    kH: Int, kW: Int,
    sH: Int, sW: Int,
    pH: Int, pW: Int
) -> None


# ------------------------------
# 1D launchers (CPU fallback loops)
# ------------------------------
fn launch_1d_maxpool_fw(
    total_threads: Int,
    block_size: Int,
    kernel: Kernel1D_MaxPoolFW,
    x: List[Float32],
    mut y: List[Float32],
    N: Int, C: Int, H: Int, W: Int,
    OH: Int, OW: Int,
    kH: Int, kW: Int,
    sH: Int, sW: Int,
    pH: Int, pW: Int
) -> None:

    # --- GPU path (guarded in one try block) ---
    if gpu_available():
        try:
            var ctx = DeviceContext()
            var total = N * C * OH * OW
            if total <= 0:
                return

            # Allocate host and device buffers
            var hx = ctx.enqueue_create_host_buffer[DType.float32](N * C * H * W)
            var hy = ctx.enqueue_create_host_buffer[DType.float32](total)
            var dx = ctx.enqueue_create_buffer[DType.float32](N * C * H * W)
            var dy = ctx.enqueue_create_buffer[DType.float32](total)

            # Copy input data
            var i = 0
            while i < N * C * H * W:
                hx[i] = x[i]
                i += 1

            ctx.enqueue_copy(src_buf=hx, dst_buf=dx)

            # Launch GPU kernel
            var grid = (total + 255) // 256
            var k = ctx.compile_function_checked[_gpu_maxpool_fw_kernel, _gpu_maxpool_fw_kernel]()

            ctx.enqueue_function_checked(
                k, dx, dy,
                N, C, H, W, OH, OW,
                kH, kW, sH, sW, pH, pW,
                grid_dim=grid, block_dim=256
            )

            ctx.synchronize()
            ctx.enqueue_copy(src_buf=dy, dst_buf=hy)
            ctx.synchronize()

            # Copy result back to CPU
            i = 0
            while i < total:
                y[i] = hy[i]
                i += 1

            return
        except e:
            # Any GPU error -> fall back to CPU
            pass

    # --- CPU fallback path ---
    if total_threads <= 0:
        return

    var bs = block_size
    if bs <= 0:
        bs = 256

    var blocks = (total_threads + bs - 1) // bs
    var blk = 0

    while blk < blocks:
        var base = blk * bs
        var t = 0
        while t < bs:
            var tid = base + t
            if tid < total_threads:
                kernel(tid, x, y, N, C, H, W, OH, OW, kH, kW, sH, sW, pH, pW)
            t += 1
        blk += 1


alias Kernel1D_MaxPoolBW = fn(
    tid: Int,
    x:  List[Float32],
    gy: List[Float32],
    mut dx: List[Float32],
    N: Int, C: Int, H: Int, W: Int,
    OH: Int, OW: Int,
    kH: Int, kW: Int,
    sH: Int, sW: Int,
    pH: Int, pW: Int
) -> None


fn launch_1d_maxpool_bw(
    total_threads: Int,
    block_size: Int,
    kernel: Kernel1D_MaxPoolBW,
    x:  List[Float32],
    gy: List[Float32],
    mut dx: List[Float32],
    N: Int, C: Int, H: Int, W: Int,
    OH: Int, OW: Int,
    kH: Int, kW: Int,
    sH: Int, sW: Int,
    pH: Int, pW: Int
) -> None:
    if total_threads <= 0:
        return
    var bs = block_size
    if bs <= 0:
        bs = 256
    var blocks = (total_threads + bs - 1) // bs
    var blk = 0
    while blk < blocks:
        var base = blk * bs
        var t = 0
        while t < bs:
            var tid = base + t
            if tid < total_threads:
                kernel(tid, x, gy, dx, N, C, H, W, OH, OW, kH, kW, sH, sW, pH, pW)
            t = t + 1
        blk = blk + 1
# ------------------------------
# 1D launchers (Conv2D Forward)
# ------------------------------
alias Kernel1D_Conv2DFW = fn(
    tid: Int,
    x:  List[Float32],
    w:  List[Float32],
    b:  List[Float32],
    mut y:  List[Float32],
    N: Int, C: Int, H: Int, W: Int,
    OH: Int, OW: Int,
    OC: Int,
    kH: Int, kW: Int,
    sH: Int, sW: Int,
    pH: Int, pW: Int,
    has_bias: Bool
) -> None


fn launch_1d_conv2d_fw(
    total_threads: Int,
    block_size: Int,
    kernel: Kernel1D_Conv2DFW,
    x: List[Float32],
    w: List[Float32],
    b: List[Float32],
    mut y: List[Float32],
    N: Int, C: Int, H: Int, W: Int,
    OH: Int, OW: Int,
    OC: Int,
    kH: Int, kW: Int,
    sH: Int, sW: Int,
    pH: Int, pW: Int,
    has_bias: Bool
) -> None:

    # --- GPU path (guarded) ---
    if gpu_available():
        try:
            var ctx = DeviceContext()

            var total = N * OC * OH * OW
            if total <= 0:
                return

            # Host buffers
            var hx = ctx.enqueue_create_host_buffer[DType.float32](N * C * H * W)
            var hw = ctx.enqueue_create_host_buffer[DType.float32](OC * C * kH * kW)
            var hy = ctx.enqueue_create_host_buffer[DType.float32](total)

            # Device buffers
            var dx = ctx.enqueue_create_buffer[DType.float32](N * C * H * W)
            var dw = ctx.enqueue_create_buffer[DType.float32](OC * C * kH * kW)
            var dy = ctx.enqueue_create_buffer[DType.float32](total)

            # ---- Bias handling (SAFE for has_bias = False) ----
            # Always allocate OC-sized bias buffers; fill with zeros when has_bias=False.
            var hb = ctx.enqueue_create_host_buffer[DType.float32](OC)
            var db = ctx.enqueue_create_buffer[DType.float32](OC)
            var has_bias_i: Int = 1 if has_bias else 0

            # Copy inputs to host buffers
            var i = 0
            var x_size = N * C * H * W
            while i < x_size:
                hx[i] = x[i]
                i += 1

            i = 0
            var w_size = OC * C * kH * kW
            while i < w_size:
                hw[i] = w[i]
                i += 1

            i = 0
            if has_bias:
                while i < OC:
                    hb[i] = b[i]      # read only when bias present
                    i += 1
            else:
                while i < OC:
                    hb[i] = 0.0       # zero-fill for safe device path
                    i += 1

            # H2D copies
            ctx.enqueue_copy(src_buf=hx, dst_buf=dx)
            ctx.enqueue_copy(src_buf=hw, dst_buf=dw)
            ctx.enqueue_copy(src_buf=hb, dst_buf=db)

            # Launch GPU kernel (must exist and match the ABI)
            var grid = (total + 255) // 256
            var k = ctx.compile_function_checked[_gpu_conv2d_fw_kernel, _gpu_conv2d_fw_kernel]()

            ctx.enqueue_function_checked(
                k, dx, dw, db, dy,
                N, C, H, W,
                OH, OW, OC,
                kH, kW, sH, sW,
                pH, pW, has_bias_i,
                grid_dim=grid, block_dim=256
            )

            ctx.synchronize()

            # D2H copy
            ctx.enqueue_copy(src_buf=dy, dst_buf=hy)
            ctx.synchronize()

            # Write back to y
            i = 0
            while i < total:
                y[i] = hy[i]
                i += 1

            return
        except e:
            # Any GPU error => fallback to CPU path below
            pass

    # --- CPU fallback path ---
    if total_threads <= 0:
        return

    var bs = block_size
    if bs <= 0:
        bs = 256

    var blocks = (total_threads + bs - 1) // bs
    var blk = 0
    while blk < blocks:
        var base = blk * bs
        var t = 0
        while t < bs:
            var tid = base + t
            if tid < total_threads:
                kernel(
                    tid, x, w, b, y,
                    N, C, H, W,
                    OH, OW, OC,
                    kH, kW, sH, sW,
                    pH, pW, has_bias
                )
            t += 1
        blk += 1



alias Kernel1D_LinearFW = fn(
    tid: Int,
    x:  List[Float32],
    wT: List[Float32],
    b:  List[Float32],
    mut y:  List[Float32],
    N: Int, InF: Int, OutF: Int,
    has_bias: Bool
) -> None

alias Kernel1D_LinearBW_DX = fn(
    tid: Int,
    dy: List[Float32],
    w:  List[Float32],
    mut dx: List[Float32],
    N: Int, InF: Int, OutF: Int
) -> None

alias Kernel1D_LinearBW_DW = fn(
    tid: Int,
    dy: List[Float32],
    x:  List[Float32],
    mut dW: List[Float32],
    N: Int, InF: Int, OutF: Int
) -> None

alias Kernel1D_LinearBW_DB = fn(
    tid: Int,
    dy: List[Float32],
    mut db: List[Float32],
    N: Int, OutF: Int
) -> None
# ------------------------------
# 1D launchers (Linear Forward)
# ------------------------------
fn launch_1d_linear_fw(
    total_threads: Int,
    block_size: Int,
    kernel: Kernel1D_LinearFW,
    x: List[Float32],
    wT: List[Float32],
    b: List[Float32],
    mut y: List[Float32],
    N: Int, InF: Int, OutF: Int,
    has_bias: Bool
) -> None:

    # --- GPU path (guarded in one try block) ---
    if gpu_available():
        try:
            var ctx = DeviceContext()
            var total = N * OutF
            if total <= 0:
                return

            # Allocate host and device buffers
            var hx = ctx.enqueue_create_host_buffer[DType.float32](N * InF)
            var hw = ctx.enqueue_create_host_buffer[DType.float32](OutF * InF)
            var hb = ctx.enqueue_create_host_buffer[DType.float32](OutF)
            var hy = ctx.enqueue_create_host_buffer[DType.float32](N * OutF)
            var dx = ctx.enqueue_create_buffer[DType.float32](N * InF)
            var dw = ctx.enqueue_create_buffer[DType.float32](OutF * InF)
            var db = ctx.enqueue_create_buffer[DType.float32](OutF)
            var dy = ctx.enqueue_create_buffer[DType.float32](N * OutF)

            # Copy data to host buffers
            var i = 0
            while i < N * InF:
                hx[i] = x[i]
                i += 1

            i = 0
            while i < OutF * InF:
                hw[i] = wT[i]
                i += 1

            i = 0
            while i < OutF:
                hb[i] = b[i]
                i += 1

            # Copy host → device
            ctx.enqueue_copy(src_buf=hx, dst_buf=dx)
            ctx.enqueue_copy(src_buf=hw, dst_buf=dw)
            ctx.enqueue_copy(src_buf=hb, dst_buf=db)

            # Launch GPU kernel
            var grid = (total + 255) // 256
            var k = ctx.compile_function_checked[_gpu_linear_fw_kernel, _gpu_linear_fw_kernel]()
            var has_bias_i: Int = 1 if has_bias else 0
            ctx.enqueue_function_checked(k, dx, dw, db, dy, N, InF, OutF, has_bias_i, grid_dim=grid, block_dim=256)

            ctx.synchronize()
            ctx.enqueue_copy(src_buf=dy, dst_buf=hy)
            ctx.synchronize()

            # Copy result back to CPU
            i = 0
            while i < N * OutF:
                y[i] = hy[i]
                i += 1

            return

        except e:
            # Any GPU error → fall back to CPU
            pass

    # --- CPU fallback path ---
    if total_threads <= 0:
        return

    var bs = block_size
    if bs <= 0:
        bs = 256

    var blocks = (total_threads + bs - 1) // bs
    var blk = 0

    while blk < blocks:
        var base = blk * bs
        var t = 0
        while t < bs:
            var tid = base + t
            if tid < total_threads:
                kernel(tid, x, wT, b, y, N, InF, OutF, has_bias)
            t += 1
        blk += 1

# ------------------------------
# 1D launchers (Linear Backward DX)
# ------------------------------
fn launch_1d_linear_bw_dx(
    total_threads: Int,
    block_size: Int,
    kernel: Kernel1D_LinearBW_DX,
    dy: List[Float32],
    w:  List[Float32],
    mut dx: List[Float32],
    N: Int, InF: Int, OutF: Int
) -> None:

    # --- GPU path (guarded in try) ---
    if gpu_available():
        try:
            var ctx = DeviceContext()
            var total = N * InF
            if total <= 0:
                return

            # --- Allocate host & device buffers ---
            var hdy = ctx.enqueue_create_host_buffer[DType.float32](N * OutF)
            var hw  = ctx.enqueue_create_host_buffer[DType.float32](InF * OutF)
            var hdx = ctx.enqueue_create_host_buffer[DType.float32](N * InF)
            var ddy = ctx.enqueue_create_buffer[DType.float32](N * OutF)
            var dw  = ctx.enqueue_create_buffer[DType.float32](InF * OutF)
            var ddx = ctx.enqueue_create_buffer[DType.float32](N * InF)

            # --- Copy input data ---
            var i = 0
            while i < N * OutF:
                hdy[i] = dy[i]
                i = i + 1

            i = 0
            while i < InF * OutF:
                hw[i] = w[i]
                i = i + 1

            ctx.enqueue_copy(src_buf=hdy, dst_buf=ddy)
            ctx.enqueue_copy(src_buf=hw, dst_buf=dw)

            # --- Launch GPU kernel ---
            var grid = (total + 255) // 256
            var k = ctx.compile_function_checked[_gpu_linear_bw_dx_kernel, _gpu_linear_bw_dx_kernel]()
            ctx.enqueue_function_checked(k, ddy, dw, ddx, N, InF, OutF, grid_dim=grid, block_dim=256)
            ctx.synchronize()

            # --- Copy results back ---
            ctx.enqueue_copy(src_buf=ddx, dst_buf=hdx)
            ctx.synchronize()

            i = 0
            while i < N * InF:
                dx[i] = hdx[i]
                i = i + 1

            return
        except e:
            pass

    # --- CPU fallback ---
    if total_threads <= 0:
        return
    var bs = block_size
    if bs <= 0:
        bs = 256

    var blocks = (total_threads + bs - 1) // bs
    var blk = 0
    while blk < blocks:
        var base = blk * bs
        var t = 0
        while t < bs:
            var tid = base + t
            if tid < total_threads:
                kernel(tid, dy, w, dx, N, InF, OutF)
            t = t + 1
        blk = blk + 1

fn launch_1d_linear_bw_dw(
    total_threads: Int,
    block_size: Int,
    kernel: Kernel1D_LinearBW_DW,
    dy: List[Float32],
    x:  List[Float32],
    mut dW: List[Float32],
    N: Int, InF: Int, OutF: Int
) -> None:

    # --- GPU path (guarded) ---
    if gpu_available():

        var _ctx: Optional[DeviceContext] = None
        try:
            var tmp = DeviceContext()
            _ctx = tmp
            var ctx = _ctx.value()
                    var total = InF * OutF
                    if total <= 0:
                return
                    var hdy = ctx.enqueue_create_host_buffer[DType.float32](N*OutF)
                    var hx  = ctx.enqueue_create_host_buffer[DType.float32](N*InF)
                    var hdW = ctx.enqueue_create_host_buffer[DType.float32](InF*OutF)
                    var ddy = ctx.enqueue_create_buffer[DType.float32](N*OutF)
                    var dx  = ctx.enqueue_create_buffer[DType.float32](N*InF)
                    var ddW = ctx.enqueue_create_buffer[DType.float32](InF*OutF)
                    var i = 0
                    while i < N*OutF:
                        hdy[i] = dy[i]
                        i = i + 1
                    i = 0
                    while i < N*InF:
                        hx[i]  = x[i]
                        i = i + 1
                    ctx.enqueue_copy(src_buf=hdy, dst_buf=ddy)
                    ctx.enqueue_copy(src_buf=hx,  dst_buf=dx)
                    var grid = (total + 255) // 256
                    var k = ctx.compile_function_checked[_gpu_linear_bw_dw_kernel, _gpu_linear_bw_dw_kernel]()
                    ctx.enqueue_function_checked(k, ddy, dx, ddW, N, InF, OutF, grid_dim=grid, block_dim=256)
                    ctx.synchronize()
                    ctx.enqueue_copy(src_buf=ddW, dst_buf=hdW)
                    ctx.synchronize()
                    i = 0
                    while i < InF*OutF:
                        dW[i] = hdW[i]
                        i = i + 1

            return
        except e:
            pass
    # --- GPU path (linear bw dW) ---
    if gpu_available():
        var total = InF * OutF
        if total > 0:
            var _ctx: Optional[DeviceContext] = None
            try:
                var tmp = DeviceContext()
                _ctx = tmp
            except e:
                pass
            if _ctx is not None:
                var ctx = _ctx.value()
                try:
                    var hdy = ctx.enqueue_create_host_buffer[DType.float32](N*OutF)
                    var hx  = ctx.enqueue_create_host_buffer[DType.float32](N*InF)
                    var hdW = ctx.enqueue_create_host_buffer[DType.float32](InF*OutF)
                    var ddy = ctx.enqueue_create_buffer[DType.float32](N*OutF)
                    var dx  = ctx.enqueue_create_buffer[DType.float32](N*InF)
                    var ddW = ctx.enqueue_create_buffer[DType.float32](InF*OutF)
                except e:
                    pass
                else:
                    var i = 0
                    while i < N*OutF: hdy[i] = dy[i]; i = i + 1
                    i = 0
                    while i < N*InF:  hx[i]  = x[i];  i = i + 1
                    try:
                        ctx.enqueue_copy(src_buf=hdy, dst_buf=ddy)
                        ctx.enqueue_copy(src_buf=hx,  dst_buf=dx)
                    except e:
                        pass
                    else:
                        var grid = (total + 255) // 256
                        try:
                            var k = ctx.compile_function_checked[_gpu_linear_bw_dw_kernel, _gpu_linear_bw_dw_kernel]()
                            ctx.enqueue_function_checked(k, ddy, dx, ddW, N, InF, OutF, grid_dim=grid, block_dim=256)
                            ctx.synchronize()
                            ctx.enqueue_copy(src_buf=ddW, dst_buf=hdW)
                            ctx.synchronize()
                        except e:
                            pass
                        else:
                            i = 0
                            while i < InF*OutF:
                                dW[i] = hdW[i]
                                i = i + 1
                            return

        if total_threads <= 0:
            return
        var bs = block_size
        if bs <= 0:
            bs = 256
        var blocks = (total_threads + bs - 1) // bs
        var blk = 0
        while blk < blocks:
            var base = blk * bs
            var t = 0
            while t < bs:
                var tid = base + t
                if tid < total_threads:
                    kernel(tid, dy, x, dW, N, InF, OutF)
                t = t + 1
            blk = blk + 1


# ------------------------------
# 1D launchers (Linear Backward DB)
# ------------------------------
fn launch_1d_linear_bw_db(
    total_threads: Int,
    block_size: Int,
    kernel: Kernel1D_LinearBW_DB,
    dy: List[Float32],
    mut db: List[Float32],
    N: Int, OutF: Int
) -> None:

    # --- GPU path (all under try) ---
    if gpu_available():
        try:
            var ctx = DeviceContext()
            if OutF <= 0:
                return

            # --- Allocate host & device buffers ---
            var hdy = ctx.enqueue_create_host_buffer[DType.float32](N * OutF)
            var hdb = ctx.enqueue_create_host_buffer[DType.float32](OutF)
            var ddy = ctx.enqueue_create_buffer[DType.float32](N * OutF)
            var ddb = ctx.enqueue_create_buffer[DType.float32](OutF)

            # --- Copy input ---
            var i = 0
            while i < N * OutF:
                hdy[i] = dy[i]
                i = i + 1

            ctx.enqueue_copy(src_buf=hdy, dst_buf=ddy)

            # --- Launch kernel ---
            var grid = (OutF + 255) // 256
            var k = ctx.compile_function_checked[_gpu_linear_bw_db_kernel, _gpu_linear_bw_db_kernel]()
            ctx.enqueue_function_checked(k, ddy, ddb, N, OutF, grid_dim=grid, block_dim=256)
            ctx.synchronize()

            # --- Copy output back ---
            ctx.enqueue_copy(src_buf=ddb, dst_buf=hdb)
            ctx.synchronize()

            i = 0
            while i < OutF:
                db[i] = hdb[i]
                i = i + 1

            return
        except e:
            pass

    # --- CPU fallback ---
    if total_threads <= 0:
        return

    var bs = block_size
    if bs <= 0:
        bs = 256

    var blocks = (total_threads + bs - 1) // bs
    var blk = 0
    while blk < blocks:
        var base = blk * bs
        var t = 0
        while t < bs:
            var tid = base + t
            if tid < total_threads:
                kernel(tid, dy, db, N, OutF)
            t = t + 1
        blk = blk + 1



# ---------- kernel + launcher ----------
alias Kernel1D_WhereSameF32 = fn(
    tid: Int,
    cond: List[Int],
    x:    List[Float32],
    y:    List[Float32],
    mut o: List[Float32],
    n: Int
) -> None

@always_inline
fn _kernel_where_same_f32(
    tid: Int,
    cond: List[Int],
    x:    List[Float32],
    y:    List[Float32],
    mut o: List[Float32],
    n: Int
) -> None:
    if tid >= n: return
    if cond[tid] != 0: o[tid] = x[tid] else: o[tid] = y[tid]

fn _launch_1d_where_same_f32(
    total_threads: Int,
    block_size: Int,
    kernel: Kernel1D_WhereSameF32,
    cond: List[Int],
    x:    List[Float32],
    y:    List[Float32],
    mut o: List[Float32],
    n: Int
) -> None:
    if total_threads <= 0: return
    var bs =256
    if block_size > 0: var bs =block_size
    var blocks = (total_threads + bs - 1) // bs
    var blk = 0
    while blk < blocks:
        var base = blk * bs
        var t = 0
        while t < bs:
            var tid = base + t
            if tid < total_threads:
                kernel(tid, cond, x, y, o, n)
            t = t + 1
        blk = blk + 1




# --- Kernel type alias for Dropout forward ---
alias Kernel1D_DropoutFW = fn(
    tid: Int,
    x:   List[Float32],
    mut y:   List[Float32],
    keep_prob: Float32,
    seed: UInt64,
    n: Int
) -> None

# --- 1D launcher (CPU fallback loop) ---
fn launch_1d_dropout_fw(
    total_threads: Int,
    block_size: Int,
    kernel: Kernel1D_DropoutFW,
    x:  List[Float32],
    mut y:  List[Float32],
    keep_prob: Float32,
    seed: UInt64,
    n: Int
) -> None:
    if total_threads <= 0:
        return
    var bs = block_size
    if bs <= 0:
        bs = 256
    var blocks = (total_threads + bs - 1) // bs
    var blk = 0
    while blk < blocks:
        var base = blk * bs
        var t = 0
        while t < bs:
            var tid = base + t
            if tid < total_threads:
                kernel(tid, x, y, keep_prob, seed, n)
            t = t + 1
        blk = blk + 1


# ------------------------------------------------
# Kernel type for Transpose2D (List buffers)
# ------------------------------------------------
alias Kernel1D_Transpose2D = fn(
    tid: Int,
    x:   List[Float32],
    mut y:   List[Float32],
    H: Int, W: Int
) -> None

# ------------------------------------------------
# Reference CPU kernel (fallback)
# y[w * H + h] = x[h * W + w]
# ------------------------------------------------
fn _k_transpose2d(
    tid: Int,
    x:   List[Float32],
    mut y: List[Float32],
    H: Int, W: Int
) -> None:
    var n_tot = H * W
    if tid >= n_tot:
        return
    var h = tid // W
    var w = tid - h * W
    var src = h * W + w
    var dst = w * H + h
    y[dst] = x[src]

# ------------------------------------------------
# GPU kernel
# Arguments mapped from device buffers appear as UnsafePointer in kernel
# ------------------------------------------------
fn _gpu_transpose2d_kernel(
    x_dev: UnsafePointer[Float32],
    y_dev: UnsafePointer[Float32],
    H: Int, W: Int
) -> None:
    var tid = block_idx.x * block_dim.x + thread_idx.x
    var n_tot = H * W
    if tid < n_tot:
        var h = tid // W
        var w = tid - h * W
        var src = h * W + w
        var dst = w * H + h
        var val = (x_dev + src).load()
        (y_dev + dst).store(val)

# ------------------------------------------------
# 1D launcher (GPU-first with CPU fallback)
# y must be the REAL output buffer (pre-allocated length H*W)
# ------------------------------------------------
fn launch_1d_transpose2d(
    total_threads: Int,
    block_size: Int,
    kernel: Kernel1D_Transpose2D,
    x:  List[Float32],
    mut y: List[Float32],
    H: Int, W: Int
) -> None:
    # Normalize thread config
    var n_tot = H * W
    if n_tot <= 0:
        return

    var tt = total_threads
    if tt <= 0 or tt != n_tot:
        tt = n_tot

    var bs = block_size
    if bs <= 0:
        bs = 256
    var grid = (tt + bs - 1) // bs

    # --- GPU path ---
    if gpu_available():
        try:
            var ctx = DeviceContext()

            # Host/device staging buffers
            var hx = ctx.enqueue_create_host_buffer[DType.float32](n_tot)
            var hy = ctx.enqueue_create_host_buffer[DType.float32](n_tot)
            var dx = ctx.enqueue_create_buffer[DType.float32](n_tot)
            var dy = ctx.enqueue_create_buffer[DType.float32](n_tot)

            # H: copy x -> hx
            var i = 0
            while i < n_tot:
                hx[i] = x[i]
                i += 1

            # H2D
            ctx.enqueue_copy(src_buf=hx, dst_buf=dx)

            # Launch GPU kernel
            var k = ctx.compile_function_checked[_gpu_transpose2d_kernel, _gpu_transpose2d_kernel]()
            ctx.enqueue_function_checked(k, dx, dy, H, W, grid_dim=grid, block_dim=bs)
            ctx.synchronize()

            # D2H
            ctx.enqueue_copy(src_buf=dy, dst_buf=hy)
            ctx.synchronize()

            # Scatter back to y (REAL buffer)
            i = 0
            while i < n_tot:
                y[i] = hy[i]
                i += 1

            return
        except e:
            # Any GPU error -> fallback to CPU
            pass

    # --- CPU fallback ---
    var blocks = (tt + bs - 1) // bs
    var blk = 0
    while blk < blocks:
        var base = blk * bs
        var t = 0
        while t < bs:
            var tid = base + t
            if tid < tt:
                kernel(tid, x, y, H, W)
            t += 1
        blk += 1

    # --- 1) Kernel: grid-stride loop + fused AdamW update ---
    # ---- safe sqrt for Float32 (no 'math.sqrt' dependency) ----
@always_inline
fn sqrt32(x: Float32) -> Float32:
    if x <= 0.0:
        return 0.0
    var g = if x > 1.0 { x / 2.0 } else { 1.0 }   # simple init
    g = 0.5 * (g + x / g)
    g = 0.5 * (g + x / g)
    return g


# ---------------- kernel (grid-stride) ----------------
fn adam_update_kernel(
    w: UnsafePointer[Float32],
    g: UnsafePointer[Float32],
    m: UnsafePointer[Float32],
    v: UnsafePointer[Float32],
    n: Int,
    lr: Float32,
    beta1: Float32,
    beta2: Float32,
    eps: Float32,
    weight_decay: Float32,
    bc1: Float32,
    bc2: Float32
):
    var tid = block_idx.x * block_dim.x + thread_idx.x
    var stride = block_dim.x * grid_dim.x
    var i = tid
    while i < n:
        var wi = w[i]
        var gi = g[i]
        var mi = m[i]
        var vi = v[i]

        mi = beta1 * mi + (1.0 - beta1) * gi
        vi = beta2 * vi + (1.0 - beta2) * (gi * gi)

        var mhat = mi / bc1
        var vhat = vi / bc2
        var step = mhat / (sqrt32(vhat) + eps)

        if weight_decay != 0.0:
            wi = wi - lr * weight_decay * wi
        wi = wi - lr * step

        w[i] = wi
        m[i] = mi
        v[i] = vi

        i = i + stride

fn launch_adam_1d_gpu(
    mut w_host: List[Float32],
    g_host: List[Float32],
    mut m_host: List[Float32],
    mut v_host: List[Float32],
    lr: Float32,
    beta1: Float32,
    beta2: Float32,
    eps: Float32,
    weight_decay: Float32,
    bc1: Float32,
    bc2: Float32,
    block_size_hint: Int = 256
):
    var n = len(w_host)
    if n == 0:
        return
    if len(g_host) != n or len(m_host) != n or len(v_host) != n:
        return
    if not has_accelerator():
        return

    # --- try برای کل بخش GPU ---
    try:
        var ctx = DeviceContext()

        # --- تنظیم block و grid ---
        var warp: Int = Int(WARP_SIZE)
        var blk: Int = block_size_hint
        if blk <= 0:
            blk = 256
        if (blk % warp) != 0:
            blk = (blk // warp) * warp
            if blk <= 0:
                blk = warp

        var grid = (n + blk - 1) // blk

        # --- ساخت بافرها ---
        var db_w = ctx.enqueue_create_buffer_float32(n)
        var db_g = ctx.enqueue_create_buffer_float32(n)
        var db_m = ctx.enqueue_create_buffer_float32(n)
        var db_v = ctx.enqueue_create_buffer_float32(n)

        # --- H2D (کپی از host به device) ---
        var i = 0
        var pw = db_w.map_to_host()
        while i < n:
            pw[i] = w_host[i]
            i += 1
        db_w.unmap_from_host(pw)

        i = 0
        var pg = db_g.map_to_host()
        while i < n:
            pg[i] = g_host[i]
            i += 1
        db_g.unmap_from_host(pg)

        i = 0
        var pm = db_m.map_to_host()
        while i < n:
            pm[i] = m_host[i]
            i += 1
        db_m.unmap_from_host(pm)

        i = 0
        var pv = db_v.map_to_host()
        while i < n:
            pv[i] = v_host[i]
            i += 1
        db_v.unmap_from_host(pv)

        # --- اجرای کرنل GPU ---
        ctx.enqueue_function_checked[
            adam_update_kernel,
            adam_update_kernel
        ](
            db_w.ptr(), db_g.ptr(), db_m.ptr(), db_v.ptr(),
            n, lr, beta1, beta2, eps, weight_decay, bc1, bc2,
            grid_dim=(grid, 1, 1),
            block_dim=(blk, 1, 1)
        )

        # --- D2H (برگرداندن داده‌ها از GPU) ---
        i = 0
        pw = db_w.map_to_host()
        while i < n:
            w_host[i] = pw[i]
            i += 1
        db_w.unmap_from_host(pw)

        i = 0
        pm = db_m.map_to_host()
        while i < n:
            m_host[i] = pm[i]
            i += 1
        db_m.unmap_from_host(pm)

        i = 0
        pv = db_v.map_to_host()
        while i < n:
            v_host[i] = pv[i]
            i += 1
        db_v.unmap_from_host(pv)

        ctx.synchronize()

    except e:
        # هر خطا در GPU را ساکت عبور می‌دهیم
        return


# ---------------- CPU fallback (same signature; single-thread) ----------------
fn launch_adam_1d_cpu(
    mut w: List[Float32],
    g: List[Float32],
    mut m: List[Float32],
    mut v: List[Float32],
    lr: Float32,
    beta1: Float32,
    beta2: Float32,
    eps: Float32,
    weight_decay: Float32,
    bc1: Float32,
    bc2: Float32,
    block_size_hint: Int = 256
):
    var n = len(w)
    if n == 0:
        return
    from algorithm.functional import parallelize, vectorize
    parallelize(fn (i: Int) -> None:
        var wi = w[i]
        var gi = g[i]
        var mi = m[i]
        var vi = v[i]

        mi = beta1 * mi + (1.0 - beta1) * gi
        vi = beta2 * vi + (1.0 - beta2) * (gi * gi)

        var mhat = mi / bc1
        var vhat = vi / bc2
        var step = mhat / (sqrt32(vhat) + eps)

        if weight_decay != 0.0:
            wi = wi - lr * weight_decay * wi
        wi = wi - lr * step

        w[i] = wi
        m[i] = mi
        v[i] = vi
    , n)

fn k_hw_to_chw1(
    tid: Int,
    src: List[Float32],
    mut dst: List[Float32],
    H: Int, W: Int
) -> None:
    var n = H * W
    if tid >= n:
        return
    var h = tid // W
    var w = tid - h * W
    dst[_off3(1, H, W, 0, h, w)] = src[_off2(H, W, h, w)]

fn k_hwc1_to_chw1(
    tid: Int,
    src: List[Float32],
    mut dst: List[Float32],
    H: Int, W: Int
) -> None:
    var n = H * W
    if tid >= n:
        return
    var h = tid // W
    var w = tid - h * W
    # HWC with C=1 collapses to H*W
    var src_off = h * (W * 1) + w * 1 + 0
    dst[_off3(1, H, W, 0, h, w)] = src[src_off]
# ------------------------------------------
# CPU-side kernel signature (HW -> CHW with C=1)
# ------------------------------------------
alias Kernel1D_ToCHW_HW = fn(
    tid: Int,
    src: List[Float32],
    mut dst: List[Float32],
    H: Int, W: Int
) -> None
# GPU kernel: receives DeviceBuffers as UnsafePointer[T]
# Assumes HW -> CHW with C=1, hence a 1:1 copy for n_tot = H*W
# ------------------------------------------
fn _gpu_to_chw_hw_kernel(
    src: UnsafePointer[Float32],
    dst: UnsafePointer[Float32],
    H: Int, W: Int
) -> None:
    var tid = block_idx.x * block_dim.x + thread_idx.x
    var n_tot = H * W
    if tid < n_tot:
        # Copy one-to-one
        (dst + tid).store((src + tid).load())

# ------------------------------------------
# 1D launcher (ToCHW from HW), GPU-first with CPU fallback
# ------------------------------------------
fn launch_1d_to_chw_hw(
    total_threads: Int,
    block_size: Int,
    kernel: Kernel1D_ToCHW_HW,         # CPU kernel to call in fallback
    src: List[Float32],
    mut dst: List[Float32],
    H: Int, W: Int
) -> None:

    # --- GPU path (guarded) ---
    if gpu_available():
        try:
            var ctx = DeviceContext()
            var n_tot = H * W
            if n_tot <= 0:
                return

            # Create host/device buffers
            var hx = ctx.enqueue_create_host_buffer[DType.float32](n_tot)
            var hy = ctx.enqueue_create_host_buffer[DType.float32](n_tot)
            var dx = ctx.enqueue_create_buffer[DType.float32](n_tot)
            var dy = ctx.enqueue_create_buffer[DType.float32](n_tot)

            # Host copy into hx
            var i = 0
            while i < n_tot:
                hx[i] = src[i]
                i += 1

            # H2D copy
            ctx.enqueue_copy(src_buf=hx, dst_buf=dx)

            # Launch GPU kernel
            var bs = block_size
            if bs <= 0:
                bs = 256
            var grid = (n_tot + bs - 1) // bs

            # If you have your own GPU kernel, replace the symbol here:
            var k = ctx.compile_function_checked[_gpu_to_chw_hw_kernel, _gpu_to_chw_hw_kernel]()
            ctx.enqueue_function_checked(k, dx, dy, H, W, grid_dim=grid, block_dim=bs)
            ctx.synchronize()

            # D2H copy
            ctx.enqueue_copy(src_buf=dy, dst_buf=hy)
            ctx.synchronize()

            # Scatter back to dst
            i = 0
            while i < n_tot:
                dst[i] = hy[i]
                i += 1

            return
        except e:
            # Any GPU error falls back to CPU path
            pass

    # --- CPU fallback ---
    if total_threads <= 0:
        return

    var bs2 = block_size
    if bs2 <= 0:
        bs2 = 256

    var blocks = (total_threads + bs2 - 1) // bs2
    var blk = 0
    while blk < blocks:
        var base = blk * bs2
        var t = 0
        while t < bs2:
            var tid = base + t
            if tid < total_threads:
                kernel(tid, src, dst, H, W)
            t += 1
        blk += 1


# ------------------------------------------
# CPU-side kernel signature (HWC1 -> CHW with C=1)
# ------------------------------------------
alias Kernel1D_ToCHW_HWC1 = fn(
    tid: Int,
    src: List[Float32],
    mut dst: List[Float32],
    H: Int, W: Int
) -> None

# ------------------------------------------
# GPU kernel: receives DeviceBuffers as UnsafePointer[T]
# HWC1 (n_tot = H*W) -> CHW (C=1) == 1:1 copy
# ------------------------------------------
fn _gpu_to_chw_hwc1_kernel(
    src: UnsafePointer[Float32],
    dst: UnsafePointer[Float32],
    H: Int, W: Int
) -> None:
    var tid = block_idx.x * block_dim.x + thread_idx.x
    var n_tot = H * W
    if tid < n_tot:
        (dst + tid).store((src + tid).load())

# ------------------------------------------
# 1D launcher (ToCHW from HWC1), GPU-first with CPU fallback
# ------------------------------------------
fn launch_1d_to_chw_hwc1(
    total_threads: Int,
    block_size: Int,
    kernel: Kernel1D_ToCHW_HWC1,   # CPU kernel for fallback
    src: List[Float32],
    mut dst: List[Float32],
    H: Int, W: Int
) -> None:

    # --- GPU path (guarded) ---
    if gpu_available():
        try:
            var ctx = DeviceContext()
            var n_tot = H * W
            if n_tot <= 0:
                return

            # Host/Device buffers
            var hx = ctx.enqueue_create_host_buffer[DType.float32](n_tot)
            var hy = ctx.enqueue_create_host_buffer[DType.float32](n_tot)
            var dx = ctx.enqueue_create_buffer[DType.float32](n_tot)
            var dy = ctx.enqueue_create_buffer[DType.float32](n_tot)

            # Fill host input
            var i = 0
            while i < n_tot:
                hx[i] = src[i]
                i += 1

            # H2D
            ctx.enqueue_copy(src_buf=hx, dst_buf=dx)

            # Launch
            var bs = block_size
            if bs <= 0:
                bs = 256
            var grid = (n_tot + bs - 1) // bs

            var k = ctx.compile_function_checked[_gpu_to_chw_hwc1_kernel, _gpu_to_chw_hwc1_kernel]()
            ctx.enqueue_function_checked(k, dx, dy, H, W, grid_dim=grid, block_dim=bs)
            ctx.synchronize()

            # D2H
            ctx.enqueue_copy(src_buf=dy, dst_buf=hy)
            ctx.synchronize()

            # Scatter back
            i = 0
            while i < n_tot:
                dst[i] = hy[i]
                i += 1

            return
        except e:
            # Any GPU error => CPU fallback
            pass

    # --- CPU fallback ---
    if total_threads <= 0:
        return

    var bs2 = block_size
    if bs2 <= 0:
        bs2 = 256

    var blocks = (total_threads + bs2 - 1) // bs2
    var blk = 0
    while blk < blocks:
        var base = blk * bs2
        var t = 0
        while t < bs2:
            var tid = base + t
            if tid < total_threads:
                kernel(tid, src, dst, H, W)
            t += 1
        blk += 1

# ------------------------------------------
# Kernel type aliases (CPU path signatures)
# ------------------------------------------
alias K_HW_to_NCHW0 = fn(
    tid: Int, src: List[Float32], mut dst: List[Float32],
    H: Int, W: Int, dst_base: Int
) -> None

alias K_HWC1_to_NCHW0 = fn(
    tid: Int, src: List[Float32], mut dst: List[Float32],
    H: Int, W: Int, dst_base: Int
) -> None

alias K_CHW_copy = fn(
    tid: Int, src: List[Float32], mut dst: List[Float32],
    C: Int, H: Int, W: Int, dst_base: Int
) -> None


# ------------------------------------------
# GPU kernels (UnsafePointer form)
# - We pass full-size dst DeviceBuffer so kernel can honor dst_base.
# ------------------------------------------

# HW -> NCHW0 (write into channel 0 plane with dst_base)
fn _gpu_hw_to_nchw0_kernel(
    src: UnsafePointer[Float32],
    dst: UnsafePointer[Float32],
    H: Int, W: Int, dst_base: Int
) -> None:
    var tid = block_idx.x * block_dim.x + thread_idx.x
    var n = H * W
    if tid < n:
        var h = tid // W
        var w = tid - h * W
        var s = h * W + w
        var d = dst_base + h * W + w     # write into X[n, 0, :, :]
        (dst + d).store((src + s).load())

# HWC (C=1) -> NCHW0
fn _gpu_hwc1_to_nchw0_kernel(
    src: UnsafePointer[Float32],
    dst: UnsafePointer[Float32],
    H: Int, W: Int, dst_base: Int
) -> None:
    var tid = block_idx.x * block_dim.x + thread_idx.x
    var n = H * W
    if tid < n:
        var h = tid // W
        var w = tid - h * W
        var s = h * W + w                 # C=1 so linear like HW
        var d = dst_base + h * W + w
        (dst + d).store((src + s).load())

# CHW copy into dst (with base offset)
fn _gpu_chw_copy_kernel(
    src: UnsafePointer[Float32],
    dst: UnsafePointer[Float32],
    C: Int, H: Int, W: Int, dst_base: Int
) -> None:
    var tid = block_idx.x * block_dim.x + thread_idx.x
    var hw = H * W
    var n = C * hw
    if tid < n:
        var c = tid // hw
        var rem = tid - c * hw
        var h = rem // W
        var w = rem - h * W
        var s = c * hw + h * W + w
        var d = dst_base + c * hw + h * W + w
        (dst + d).store((src + s).load())

# ------------------------------------------
# Launchers (GPU-first with CPU fallback)
# ------------------------------------------

# HW -> NCHW0
fn launch_1d_hw(
    total_threads: Int,
    block_size: Int,
    kernel: K_HW_to_NCHW0,         # CPU kernel for fallback
    src: List[Float32],
    mut dst: List[Float32],
    H: Int, W: Int, dst_base: Int
) -> None:

    # --- GPU path ---
    if gpu_available():
        try:
            var ctx = DeviceContext()
            var n_src = H * W
            if n_src <= 0:
                return

            var out_len = len(dst)
            if out_len <= 0:
                return

            # Host/device buffers: src sized to n_src, dst sized to full out_len (to honor dst_base)
            var hx = ctx.enqueue_create_host_buffer[DType.float32](n_src)
            var hy = ctx.enqueue_create_host_buffer[DType.float32](out_len)
            var dx = ctx.enqueue_create_buffer[DType.float32](n_src)
            var dy = ctx.enqueue_create_buffer[DType.float32](out_len)

            # Fill hx from src; hy from dst (so unrelated regions remain intact)
            var i = 0
            while i < n_src:
                hx[i] = src[i]
                i += 1
            i = 0
            while i < out_len:
                hy[i] = dst[i]
                i += 1

            # Copies
            ctx.enqueue_copy(src_buf=hx, dst_buf=dx)
            ctx.enqueue_copy(src_buf=hy, dst_buf=dy)

            # Grid/block
            var bs = block_size
            if bs <= 0:
                bs = 256
            var grid = (n_src + bs - 1) // bs

            # Compile+launch
            var k = ctx.compile_function_checked[_gpu_hw_to_nchw0_kernel, _gpu_hw_to_nchw0_kernel]()
            ctx.enqueue_function_checked(k, dx, dy, H, W, dst_base, grid_dim=grid, block_dim=bs)
            ctx.synchronize()

            # Back to host and then to dst
            ctx.enqueue_copy(src_buf=dy, dst_buf=hy)
            ctx.synchronize()

            i = 0
            while i < out_len:
                dst[i] = hy[i]
                i += 1

            return
        except e:
            pass

    # --- CPU fallback ---
    if total_threads <= 0:
        return

    var bs2 = block_size
    if bs2 <= 0:
        bs2 = 256

    var blocks = (total_threads + bs2 - 1) // bs2
    var b = 0
    while b < blocks:
        var base = b * bs2
        var t = 0
        while t < bs2:
            var tid = base + t
            if tid < total_threads:
                kernel(tid, src, dst, H, W, dst_base)
            t += 1
        b += 1

# HWC1 -> NCHW0
fn launch_1d_hwc1(
    total_threads: Int,
    block_size: Int,
    kernel: K_HWC1_to_NCHW0,       # CPU kernel for fallback
    src: List[Float32],
    mut dst: List[Float32],
    H: Int, W: Int, dst_base: Int
) -> None:

    # --- GPU path ---
    if gpu_available():
        try:
            var ctx = DeviceContext()
            var n_src = H * W            # C=1
            if n_src <= 0:
                return

            var out_len = len(dst)
            if out_len <= 0:
                return

            var hx = ctx.enqueue_create_host_buffer[DType.float32](n_src)
            var hy = ctx.enqueue_create_host_buffer[DType.float32](out_len)
            var dx = ctx.enqueue_create_buffer[DType.float32](n_src)
            var dy = ctx.enqueue_create_buffer[DType.float32](out_len)

            var i = 0
            while i < n_src:
                hx[i] = src[i]
                i += 1
            i = 0
            while i < out_len:
                hy[i] = dst[i]
                i += 1

            ctx.enqueue_copy(src_buf=hx, dst_buf=dx)
            ctx.enqueue_copy(src_buf=hy, dst_buf=dy)

            var bs = block_size
            if bs <= 0:
                bs = 256
            var grid = (n_src + bs - 1) // bs

            var k = ctx.compile_function_checked[_gpu_hwc1_to_nchw0_kernel, _gpu_hwc1_to_nchw0_kernel]()
            ctx.enqueue_function_checked(k, dx, dy, H, W, dst_base, grid_dim=grid, block_dim=bs)
            ctx.synchronize()

            ctx.enqueue_copy(src_buf=dy, dst_buf=hy)
            ctx.synchronize()

            i = 0
            while i < out_len:
                dst[i] = hy[i]
                i += 1

            return
        except e:
            pass

    # --- CPU fallback ---
    if total_threads <= 0:
        return

    var bs2 = block_size
    if bs2 <= 0:
        bs2 = 256

    var blocks = (total_threads + bs2 - 1) // bs2
    var b = 0
    while b < blocks:
        var base = b * bs2
        var t = 0
        while t < bs2:
            var tid = base + t
            if tid < total_threads:
                kernel(tid, src, dst, H, W, dst_base)
            t += 1
        b += 1

# CHW copy
fn launch_1d_chw(
    total_threads: Int,
    block_size: Int,
    kernel: K_CHW_copy,             # CPU kernel for fallback
    src: List[Float32],
    mut dst: List[Float32],
    C: Int, H: Int, W: Int, dst_base: Int
) -> None:

    # --- GPU path ---
    if gpu_available():
        try:
            var ctx = DeviceContext()
            var n_src = C * H * W
            if n_src <= 0:
                return

            var out_len = len(dst)
            if out_len <= 0:
                return

            var hx = ctx.enqueue_create_host_buffer[DType.float32](n_src)
            var hy = ctx.enqueue_create_host_buffer[DType.float32](out_len)
            var dx = ctx.enqueue_create_buffer[DType.float32](n_src)
            var dy = ctx.enqueue_create_buffer[DType.float32](out_len)

            var i = 0
            while i < n_src:
                hx[i] = src[i]
                i += 1
            i = 0
            while i < out_len:
                hy[i] = dst[i]
                i += 1

            ctx.enqueue_copy(src_buf=hx, dst_buf=dx)
            ctx.enqueue_copy(src_buf=hy, dst_buf=dy)

            var bs = block_size
            if bs <= 0:
                bs = 256
            var grid = (n_src + bs - 1) // bs

            var k = ctx.compile_function_checked[_gpu_chw_copy_kernel, _gpu_chw_copy_kernel]()
            ctx.enqueue_function_checked(k, dx, dy, C, H, W, dst_base, grid_dim=grid, block_dim=bs)
            ctx.synchronize()

            ctx.enqueue_copy(src_buf=dy, dst_buf=hy)
            ctx.synchronize()

            i = 0
            while i < out_len:
                dst[i] = hy[i]
                i += 1

            return
        except e:
            pass

    # --- CPU fallback ---
    if total_threads <= 0:
        return

    var bs2 = block_size
    if bs2 <= 0:
        bs2 = 256

    var blocks = (total_threads + bs2 - 1) // bs2
    var b = 0
    while b < blocks:
        var base = b * bs2
        var t = 0
        while t < bs2:
            var tid = base + t
            if tid < total_threads:
                kernel(tid, src, dst, C, H, W, dst_base)
            t += 1
        b += 1

# ------------------------------------------
# (Optional) CPU reference kernels (for fallback/testing)
# ------------------------------------------
fn k_hw_to_nchw0(
    tid: Int, src: List[Float32], mut dst: List[Float32],
    H: Int, W: Int, dst_base: Int
) -> None:
    var n_elem = H * W
    if tid >= n_elem:
        return
    var h = tid // W
    var w = tid - h * W
    var s = h * W + w
    var d = dst_base + h * W + w
    dst[d] = src[s]

fn k_hwc1_to_nchw0(
    tid: Int, src: List[Float32], mut dst: List[Float32],
    H: Int, W: Int, dst_base: Int
) -> None:
    var n_elem = H * W
    if tid >= n_elem:
        return
    var h = tid // W
    var w = tid - h * W
    var s = h * (W * 1) + w * 1 + 0    # C=1
    var d = dst_base + h * W + w
    dst[d] = src[s]

fn k_chw_copy(
    tid: Int, src: List[Float32], mut dst: List[Float32],
    C: Int, H: Int, W: Int, dst_base: Int
) -> None:
    var n_elem = C * H * W
    if tid >= n_elem:
        return
    var hw = H * W
    var c = tid // hw
    var rem = tid - c * hw
    var h = rem // W
    var w = rem - h * W
    var s = c * hw + h * W + w
    var d = dst_base + c * hw + h * W + w
    dst[d] = src[s]

# ------------------------------------------
# CPU-side kernel
# ------------------------------------------
alias Kernel1D_OneHot = fn(
    tid: Int,
    labels: List[Int],
    mut dst: List[Float32],
    N: Int,
    C: Int
) -> None

fn _k_one_hot(
    tid: Int,
    labels: List[Int],
    mut dst: List[Float32],
    N: Int,
    C: Int
) -> None:
    if tid >= N:
        return
    var cls = labels[tid]
    if cls >= 0 and cls < C:
        dst[tid * C + cls] = Float32(1.0)   # نوع صریح

# ------------------------------------------
# GPU kernel
# ------------------------------------------
fn _gpu_one_hot_kernel(
    labels_dev: UnsafePointer[Int32],
    dst_dev:    UnsafePointer[Float32],
    N: Int,
    C: Int
) -> None:
    var tid = block_idx.x * block_dim.x + thread_idx.x
    if tid < N:
        var cls32 = (labels_dev + tid).load()
        var cls = Int(cls32)
        if cls >= 0 and cls < C:
            var offset = tid * C + cls
            (dst_dev + offset).store(Float32(1.0))

# ------------------------------------------
# 1D launcher (GPU-first with CPU fallback)
# dst MUST be pre-zeroed and must be the REAL buffer (not a copy).
# ------------------------------------------
fn launch_1d_one_hot(
    total_threads: Int,
    block_size: Int,
    kernel: Kernel1D_OneHot,
    labels: List[Int],
    mut dst: List[Float32],
    N: Int,
    C: Int
) -> None:
    # --- GPU path ---
    if gpu_available():
        try:
            var ctx = DeviceContext()
            if N <= 0 or C <= 0:
                return

            # host/device buffers
            var hlabels = ctx.enqueue_create_host_buffer[DType.int32](N)
            var dlabels = ctx.enqueue_create_buffer[DType.int32](N)
            var dout    = ctx.enqueue_create_buffer[DType.float32](N * C)
            var hout    = ctx.enqueue_create_host_buffer[DType.float32](N * C)

            # H: fill labels (cast Int -> Int32)
            var i = 0
            while i < N:
                hlabels[i] = Int32(labels[i])
                i += 1

            # H2D copies
            ctx.enqueue_copy(src_buf=hlabels, dst_buf=dlabels)

            # zero device output (stage zeros from host)
            i = 0
            var n_tot = N * C
            while i < n_tot:
                hout[i] = Float32(0.0)
                i += 1
            ctx.enqueue_copy(src_buf=hout, dst_buf=dout)

            # launch
            var bs = block_size
            if bs <= 0: bs = 256
            var grid = (total_threads + bs - 1) // bs

            # NOTE: امضای دقیق compile_function_checked به رانتایم شما بستگی دارد.
            # اگر نیاز بود، به امضایی که برای کرنل‌های قبلی‌تان جواب داده تطبیق بدهید.
            var k = ctx.compile_function_checked[_gpu_one_hot_kernel, _gpu_one_hot_kernel]()
            ctx.enqueue_function_checked(k, dlabels, dout, N, C, grid_dim=grid, block_dim=bs)
            ctx.synchronize()

            # D2H
            ctx.enqueue_copy(src_buf=dout, dst_buf=hout)
            ctx.synchronize()

            # scatter back to dst (REAL buffer)
            i = 0
            while i < n_tot:
                dst[i] = hout[i]
                i += 1

            return
        except e:
            # GPU failed -> fallback to CPU
            pass

    # --- CPU fallback ---
    if total_threads <= 0:
        return

    # ensure dst zeroed (safety)
    var n_tot2 = N * C
    var j = 0
    while j < n_tot2:
        dst[j] = Float32(0.0)
        j += 1

    var bs2 = block_size
    if bs2 <= 0: bs2 = 256

    var blocks = (total_threads + bs2 - 1) // bs2
    var blk = 0
    while blk < blocks:
        var base = blk * bs2
        var t = 0
        while t < bs2:
            var tid = base + t
            if tid < total_threads:
                kernel(tid, labels, dst, N, C)
            t += 1
        blk += 1

# ------------------------------------------
# Public API: build one-hot tensor on GPU/CPU
# ------------------------------------------
fn one_hot_gpu(y: tensor.Tensor[Int], num_classes: Int) -> tensor.Tensor[Float32]:
    var shp = y.shape()
    if len(shp) != 1 or num_classes <= 0:
        return tensor.zeros([0, (num_classes if num_classes > 0 else 0)])

    var N = shp[0]
    if N <= 0:
        return tensor.zeros([0, num_classes])

    # allocate REAL output tensor (zeroed)
    var out = tensor.zeros([N, num_classes])

    # labels can be a copy (read-only for kernels)
    var labels: List[Int] = y._data.copy()

    # IMPORTANT: pass the REAL buffer (no copy) to the launcher
    launch_1d_one_hot(
        N,              # total_threads
        256,            # block_size
        _k_one_hot,     # CPU fallback kernel
        labels,         # List[Int] (copied)
        out._data,      # List[Float32] (REAL buffer, NOT copy)
        N,
        num_classes
    )

    # Return: avoid implicit copy by returning an explicit copy
    return out.copy()



# ------------------------------------------
# CPU-side kernel signature (linear copy)
# ------------------------------------------
alias Kernel1D_Copy = fn(
    tid: Int,
    src: List[Float32],
    mut dst: List[Float32],
    total: Int
) -> None

# Reference CPU kernel (linear copy)
fn _k_copy_linear(
    tid: Int,
    src: List[Float32],
    mut dst: List[Float32],
    total: Int
) -> None:
    if tid < total:
        dst[tid] = src[tid]

# ------------------------------------------
# GPU kernel: DeviceBuffers arrive as UnsafePointer inside kernel
# Performs a linear copy for total elements
# ------------------------------------------
fn _gpu_copy_linear_kernel(
    src: UnsafePointer[Float32],
    dst: UnsafePointer[Float32],
    total: Int
) -> None:
    var tid = block_idx.x * block_dim.x + thread_idx.x
    if tid < total:
        (dst + tid).store((src + tid).load())

# ------------------------------------------
# 1D launcher (Copy Linear), GPU-first with CPU fallback
# ------------------------------------------
fn _launch_1d_copy(
    total_threads: Int,
    block_size: Int,
    kernel: Kernel1D_Copy,          # CPU fallback kernel
    src: List[Float32],
    mut dst: List[Float32]
) -> None:

    # --- GPU path (guarded) ---
    if gpu_available():
        try:
            var ctx = DeviceContext()
            var n_tot = total_threads
            if n_tot <= 0:
                return

            # Allocate host/device buffers
            var hx = ctx.enqueue_create_host_buffer[DType.float32](n_tot)
            var hy = ctx.enqueue_create_host_buffer[DType.float32](n_tot)
            var dx = ctx.enqueue_create_buffer[DType.float32](n_tot)
            var dy = ctx.enqueue_create_buffer[DType.float32](n_tot)

            # Host -> hx
            var i = 0
            while i < n_tot:
                hx[i] = src[i]
                i += 1

            # H2D
            ctx.enqueue_copy(src_buf=hx, dst_buf=dx)

            # GPU launch
            var bs = block_size
            if bs <= 0:
                bs = 256
            var grid = (n_tot + bs - 1) // bs

            var k = ctx.compile_function_checked[_gpu_copy_linear_kernel, _gpu_copy_linear_kernel]()
            ctx.enqueue_function_checked(k, dx, dy, n_tot, grid_dim=grid, block_dim=bs)
            ctx.synchronize()

            # D2H
            ctx.enqueue_copy(src_buf=dy, dst_buf=hy)
            ctx.synchronize()

            # hy -> dst
            i = 0
            while i < n_tot:
                dst[i] = hy[i]
                i += 1

            return
        except e:
            # Any GPU error => fallback to CPU
            pass

    # --- CPU fallback ---
    if total_threads <= 0:
        return

    var bs2 = block_size
    if bs2 <= 0:
        bs2 = 256

    var blocks = (total_threads + bs2 - 1) // bs2
    var blk = 0
    while blk < blocks:
        var base = blk * bs2
        var t = 0
        while t < bs2:
            var tid = base + t
            if tid < total_threads:
                kernel(tid, src, dst, total_threads)
            t += 1
        blk += 1

# ------------------------------------------
# Flatten [N,C,H,W] -> [N, C*H*W]  (linear copy)
# ------------------------------------------
fn flatten_nchw_gpu(x: tensor.Tensor[Float32]) -> tensor.Tensor[Float32]:
    var s = x.shape()
    if len(s) != 4:
        return x.copy()

    var N = s[0]; var C = s[1]; var H = s[2]; var W = s[3]
    var CHW = C * H * W
    var total = N * CHW

    var out = tensor.zeros([N, CHW])

    # Flatten is linear (same physical order)
    var src = x._data.copy()
    var dst = out._data.copy()
    _launch_1d_copy(total, 256, _k_copy_linear, src, dst)

    return out.copy()

# ------------------------------------------
# Unflatten [N, C*H*W] -> [N,C,H,W]  (linear copy)
# ------------------------------------------
fn unflatten_nchw_gpu(
    x: tensor.Tensor[Float32],
    C: Int, H: Int, W: Int
) -> tensor.Tensor[Float32]:
    var s = x.shape()
    if len(s) != 2:
        return x.copy()

    var N = s[0]
    var expect = C * H * W
    if not (s[1]== expect):
        return x.copy()

    var total = N * expect
    var out = tensor.zeros([N, C, H, W])

    # Unflatten is linear (same physical order)
    var src = x._data.copy()
    var dst = out._data.copy()
    _launch_1d_copy(total, 256, _k_copy_linear, src, dst)

    return out.copy()
