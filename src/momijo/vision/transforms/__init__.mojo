# MIT License — Short Header
# Project: momijo | Package: vision.transforms.__init__
# File: vision/transforms/__init__.mojo
# Copyright (c) 2025 Morteza Talebou and Mitra Daneshmand
# SPDX-License-Identifier: MIT

# Core API re-exports for vision.transforms.__init__

from vision.transforms.convert_color import rgb_to_gray
from vision.transforms.resize import  resize_nearest, resize_nearest_u8_hwc
from vision.transforms.tile import  resize_nearest_u8_hwc_tiled
from vision.transforms.tile import resize_nearest_u8_hwc_tiled_scheduled
# ----------------------------
# Utilities
# ----------------------------
fn _clamp_i(x: Int, lo: Int, hi: Int) -> Int:
    if x < lo: return lo
    if x > hi: return hi
    return x
