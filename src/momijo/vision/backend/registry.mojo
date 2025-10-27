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
# Project: momijo.vision.backend
# File: momijo/vision/backend/registry.mojo
 

from momijo.vision.tensor import Tensor
from momijo.vision.schedule.schedule import Schedule, auto_schedule
from momijo.vision.backend.cpu.resize_cpu import resize_u8_hwc_nearest
from momijo.vision.backend.cpu.simd.resize_simd_u8_hwc import resize_u8_hwc_nearest_simd
from momijo.vision.backend.cpu.convert_color_cpu import rgb_to_gray_u8_hwc
from momijo.vision.backend.cpu.simd.convert_simd_u8_hwc import rgb_to_gray_u8_hwc_simd
from momijo.vision.backend.cpu.resize_tiled_cpu import resize_u8_hwc_nearest_tiled

@value
@fieldwise_init
struct Backend:
    var CPUScalar: Int32 = 0
    var CPUSIMD: Int32 = 1

@fieldwise_init
struct KernelRegistry:
    var _backend: Backend
    var _sched: Schedule
    fn __init__(out self self, backend: Backend, sched: Schedule):
        self._backend = backend
        self._sched = sched

    fn with_auto(out self) -> KernelRegistry:
        var s = auto_schedule(0, 0)
        return KernelRegistry(self._backend, s)

    fn resize_nearest(self, src: Tensor, oh: Int, ow: Int) -> Tensor:
        # Tile-runner if tiles are set
        if self._sched.tile_h() > 0 and self._sched.tile_w() > 0:
            return resize_u8_hwc_nearest_tiled(src, oh, ow, self._sched.tile_h(), self._sched.tile_w())
        # Otherwise pick SIMD if available
        if self._backend == Backend.CPUSIMD and self._sched.vec() > 1:
            return resize_u8_hwc_nearest_simd(src, oh, ow)
        return resize_u8_hwc_nearest(src, oh, ow)

    fn convert_rgb_to_gray(self, src: Tensor) -> Tensor:
        if self._backend == Backend.CPUSIMD and self._sched.vec() > 1:
            return rgb_to_gray_u8_hwc_simd(src)
        return rgb_to_gray_u8_hwc(src)

fn default_registry() -> KernelRegistry:
    return KernelRegistry(Backend.CPUSIMD, auto_schedule(0,0))