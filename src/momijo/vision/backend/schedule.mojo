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
# File: momijo/vision/backend/schedule.mojo
 
@fieldwise_init
struct Schedule:
    var _tile_h: Int
    var _tile_w: Int
    var _vec: Int
    fn __init__(out self self, tile_h: Int, tile_w: Int, vec: Int):
        self._tile_h = tile_h
        self._tile_w = tile_w
        self._vec = vec

    fn tile_h(self) -> Int: return self._tile_h
    fn tile_w(self) -> Int: return self._tile_w
    fn vec(self) -> Int: return self._vec

fn auto_schedule(img_h: Int, img_w: Int) -> Schedule:
    # naive heuristic: small tiles for cache-friendliness
    var th = 64
    var tw = 64
    var v = 16
    return Schedule(th, tw, v)