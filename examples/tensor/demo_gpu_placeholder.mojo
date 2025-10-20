# MIT License
# SPDX-License-Identifier: MIT
# Project:      momijo.tensor
# Module:       tests.tensor
# File:         examples/Tensor/demo_gpu_placeholder.mojo
#
# Description:
#   GPU demo — runs on CPU by default; switches to GPU at runtime if an
#   accelerator exists and the user requests GPU via the runtime context.
#
# Author(s):    Morteza Taleblou & Mitra Daneshmand
# Website:      https://taleblou.ir/
# Repository:   https://github.com/taleblou/momijo

from sys.info import has_accelerator

from momijo.tensor import tensor
from momijo.tensor.device import (
    Device,
    RuntimeDeviceContext,
    get_device_name,
    set_device_cpu,
    set_device_gpu,
    zeros_f32,
    ones_f32,
    zeros_f64,
    ones_f64,
    to_device,
)

# -----------------------------------------------------------------------------
# Small banner printer
# -----------------------------------------------------------------------------
fn banner(title: String) -> None:
    print("\n=== " + title + " ===")

# -----------------------------------------------------------------------------
# Helper to report the current effective device name
# -----------------------------------------------------------------------------
fn get_device() -> String:
    var ctx = RuntimeDeviceContext.cpu()
    return get_device_name(ctx)

# -----------------------------------------------------------------------------
# 19) GPU demo (runtime switch)
# -----------------------------------------------------------------------------
fn demo_gpu() -> None:
    banner("19) GPU DEMO (runtime switch)")

    if has_accelerator():
        print("Accelerator detected")
    else:
        print("No accelerator detected (will run on CPU)")

    # Start on CPU
    var ctx = RuntimeDeviceContext.cpu()
    print("requested device (start): " + ctx.__str__())
    print("effective device (start): " + get_device_name(ctx))

    # CPU allocations (device-aware helpers)
    var a_cpu = zeros_f32([8], ctx)
    var b_cpu = ones_f32([8], ctx)
    print("a_cpu zeros_f32([8]) -> effective=" + get_device_name(ctx) + " | shape=" + a_cpu.shape().__str__())
    print("b_cpu ones_f32([8])  -> effective=" + get_device_name(ctx) + " | shape=" + b_cpu.shape().__str__())

    # Baseline matmul (likely CPU implementation)
    var x = tensor.randn_f64([512, 512])
    var y = tensor.randn_f64([512, 512])
    var z = x.matmul(y)
    print("matmul baseline effective=" + get_device_name(ctx) + " | z.shape: " + z.shape().__str__())

    # Request GPU (effective may remain CPU if no accelerator/runtime backend)
    ctx = set_device_gpu(ctx)
    print("requested device (after switch): " + ctx.__str__())
    print("effective device (after switch): " + get_device_name(ctx))

    # Device-aware allocations after switch
    var a = zeros_f32([1024 * 1024], ctx)
    var b = ones_f32([1024 * 1024], ctx)
    print("a zeros_f32(1M) -> effective=" + get_device_name(ctx) + " | shape: " + a.shape().__str__())
    print("b ones_f32(1M)  -> effective=" + get_device_name(ctx) + " | shape: " + b.shape().__str__())

    # to_device demo (Float64)
    var x64 = ones_f64([16], ctx)
    var x64_gpu = to_device[Float64](x64, Device.gpu())
    print("to_device(requested=gpu) -> effective=" + get_device_name(ctx) + " | shape: " + x64_gpu.shape().__str__())

    # Matmul again (still CPU unless GPU matmul is implemented)
    var z2 = x.matmul(y)
    print("matmul after requested=gpu -> effective=" + get_device_name(ctx) + " | z2.shape: " + z2.shape().__str__())

    # Back to CPU
    ctx = set_device_cpu(ctx)
    var c = zeros_f32([64], ctx)
    print("back to CPU -> effective=" + get_device_name(ctx) + " | shape: " + c.shape().__str__())

# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------
fn main() -> None:
    demo_gpu()
