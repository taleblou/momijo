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
# Project: momijo.kernels.common
# File: src/momijo/kernels/common/launch.mojo

from momijo.core.device import Device
from momijo.core.error import Error
from momijo.tensor.tensor import Tensor

struct LaunchConfig:
    var grid_dim: Int
    var block_dim: Int
    var shared_mem: Int
fn __init__(out self, grid_dim: Int, block_dim: Int, shared_mem: Int = 0) -> None:
        self.grid_dim = grid_dim
        self.block_dim = block_dim
        self.shared_mem = shared_mem
fn __copyinit__(out self, other: Self) -> None:
        self.grid_dim = other.grid_dim
        self.block_dim = other.block_dim
        self.shared_mem = other.shared_mem
fn __moveinit__(out self, deinit other: Self) -> None:
        self.grid_dim = other.grid_dim
        self.block_dim = other.block_dim
        self.shared_mem = other.shared_mem
# Abstract launcher trait
trait KernelLauncher:
fn launch(self, fn_name: String, args: List[Tensor], config: LaunchConfig) raises -> Error

# CPU launcher (sequential or threaded execution)
struct CPULauncher(KernelLauncher):
fn __init__(out self) -> None:
        pass
fn launch(self, fn_name: String, args: List[Tensor], config: LaunchConfig) raises -> Error:
        # For CPU backend, kernel launch is a direct function call (no grid/block decomposition)
        # Placeholder implementation - dispatch logic to be extended
        return Error.ok()

# GPU launcher (CUDA/ROCm/MPS backends)
struct GPULauncher(KernelLauncher):
    var device: Device
fn __init__(out self, device: Device) -> None:
        self.device = device
fn launch(self, fn_name: String, args: List[Tensor], config: LaunchConfig) raises -> Error:
        # Placeholder GPU kernel launch (actual CUDA/ROCm/MPS integration required)
        return Error.ok()

# Utility function for selecting launcher
fn get_launcher(device: Device) -> KernelLauncher:
    if device.is_cpu():
        return CPULauncher()
    else:
        return GPULauncher(device)

# --- Minimal self-test for smoke testing ---
fn _self_test() -> Bool:
    var cfg = LaunchConfig(1, 1)
    var dev = Device("cpu")
    var launcher = get_launcher(dev)
    try:
        var err = launcher.launch("noop", [], cfg)
    except e:
        return False
    return True