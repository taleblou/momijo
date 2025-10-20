# MIT License
# SPDX-License-Identifier: MIT
# Project:      momijo.tensor
# Module:       tests.tensor
# File:         examples/Tensor/demo_dtypes_devices_strides.mojo
#
# Description:
#   Demo for dtypes, runtime device switching (CPU/GPU), casting, contiguity,
#   and strides.
#
# Author(s):    Morteza Taleblou & Mitra Daneshmand
# Website:      https://taleblou.ir/
# Repository:   https://github.com/taleblou/momijo

from momijo.tensor import tensor
from momijo.tensor.device import (
    Device,
    RuntimeDeviceContext,
    get_device_name,
    set_device_cpu,
    set_device_gpu,
    to_device,
)

# -----------------------------------------------------------------------------
# Small banner printer
# -----------------------------------------------------------------------------
fn banner(title: String) -> None:
    print("\n=== " + title + " ===")

# -----------------------------------------------------------------------------
# 2) Dtypes, Devices, Casting, Contiguity, Strides
# -----------------------------------------------------------------------------
fn demo_dtypes_devices_strides() -> None:
    banner("2) DTYPES / DEVICES / CONTIGUITY / STRIDES")

    # Start on CPU context; you can switch to GPU via set_device_gpu(ctx).
    var ctx = RuntimeDeviceContext.cpu()
    print("requested device (start): " + ctx.__str__())
    print("effective device (start): " + get_device_name(ctx))  # may fallback to "cpu" at runtime

    # Base tensor: shape (3, 4), dtype inferred by factory
    var x = tensor.arange(0, 12, 1).reshape([3, 4])

    print(
        "x dtype: " + x.dtype_name()
        + " | device(effective): " + get_device_name(ctx)
        + " | is_contiguous: " + String(x.is_contiguous())
    )
    print("x.shape: " + x.shape().__str__() + " | x.strides(): " + x.strides().__str__())

    # Casting (to Float64 and to Int)
    var x_f64 = x.to_float64()
    var x_i = x.to_int()
    print("cast -> float64: " + x_f64.dtype_name() + " | int: " + x_i.dtype_name())

    # Request GPU; effective device may still be CPU if no accelerator exists.
    ctx = set_device_gpu(ctx)
    print("requested device (after switch): " + ctx.__str__())
    print("effective device (after switch): " + get_device_name(ctx))

    # Move to requested device (runtime-safe: GPU path if available, else CPU copy)
    var x_dev = to_device[Float64](x_f64, Device.gpu())
    print("moved to (requested) device: gpu | effective device now: " + get_device_name(ctx))
    # Optional: use x_dev in follow-up computations as needed.

    # Transpose to create a non-contiguous view, then make it contiguous
    var y = tensor.arange(0, 12, 1).reshape([3, 4]).transpose([1, 0])
    print(
        "y (transposed).is_contiguous(): " + String(y.is_contiguous())
        + " | y.strides(): " + y.strides().__str__()
    )

    var y_c = y.contiguous()
    print(
        "y.contiguous().is_contiguous(): " + String(y_c.is_contiguous())
        + " | y_c.strides(): " + y_c.strides().__str__()
    )

    # Switch back to CPU to show symmetry
    ctx = set_device_cpu(ctx)
    print("requested device (back to cpu): " + ctx.__str__())
    print("effective device (back to cpu): " + get_device_name(ctx))

# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------
fn main() -> None:
    demo_dtypes_devices_strides()
