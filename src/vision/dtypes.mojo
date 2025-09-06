# MIT License
# Copyright (c) 2025 Morteza Talebou and Mitra Daneshmand
# Project: momijo  |  Source: https://github.com/taleblou/momijo
# This file is part of the Momijo project. See the LICENSE file at the repository root.
# Momijo
# SPDX-License-Identifier: MIT
# Website: https://taleblou.ir/
# Repository: https://github.com/taleblou/momijo
#
# Project: momijo.vision
# File: src/momijo/vision/dtypes.mojo
#
# Notes:
# - This module defines lightweight, dependency-minimal tags and helpers for vision data types.
# - It intentionally avoids importing other Momijo modules so it stays usable even if other modules are WIP.
# - No 'export', no 'let', no 'inout', constructors use 'fn __init__(out self, ...)'.

# -------------------------
# Channel order (GRAY, RGB, BGR, RGBA, BGRA, ARGB)
# -------------------------
struct ChannelOrder(Copyable, Movable):
    var id: Int

    fn __init__(out self, id: Int):
        self.id = id

    @staticmethod
    fn GRAY() -> ChannelOrder:
        return ChannelOrder(1)

    @staticmethod
    fn RGB() -> ChannelOrder:
        return ChannelOrder(2)

    @staticmethod
    fn BGR() -> ChannelOrder:
        return ChannelOrder(3)

    @staticmethod
    fn RGBA() -> ChannelOrder:
        return ChannelOrder(4)

    @staticmethod
    fn BGRA() -> ChannelOrder:
        return ChannelOrder(5)

    @staticmethod
    fn ARGB() -> ChannelOrder:
        return ChannelOrder(6)

    fn __eq__(self, other: ChannelOrder) -> Bool:
        return self.id == other.id

    fn to_string(self) -> String:
        if self.id == 1:
            return String("GRAY")
        if self.id == 2:
            return String("RGB")
        if self.id == 3:
            return String("BGR")
        if self.id == 4:
            return String("RGBA")
        if self.id == 5:
            return String("BGRA")
        if self.id == 6:
            return String("ARGB")
        return String("UNKNOWN")


@staticmethod
fn channel_order_from_string(s: String) -> ChannelOrder:
    var up = s.upper()
    if up == String("GRAY"):
        return ChannelOrder.GRAY()
    if up == String("RGB"):
        return ChannelOrder.RGB()
    if up == String("BGR"):
        return ChannelOrder.BGR()
    if up == String("RGBA"):
        return ChannelOrder.RGBA()
    if up == String("BGRA"):
        return ChannelOrder.BGRA()
    if up == String("ARGB"):
        return ChannelOrder.ARGB()
    # default fallback
    return ChannelOrder.RGB()


# -------------------------
# Layout (HWC vs CHW)
# -------------------------
struct Layout(Copyable, Movable):
    var id: Int

    fn __init__(out self, id: Int):
        self.id = id

    @staticmethod
    fn HWC() -> Layout:
        return Layout(1)

    @staticmethod
    fn CHW() -> Layout:
        return Layout(2)

    fn __eq__(self, other: Layout) -> Bool:
        return self.id == other.id

    fn to_string(self) -> String:
        if self.id == 1:
            return String("HWC")
        if self.id == 2:
            return String("CHW")
        return String("UNKNOWN")


@staticmethod
fn layout_from_string(s: String) -> Layout:
    var up = s.upper()
    if up == String("HWC"):
        return Layout.HWC()
    if up == String("CHW"):
        return Layout.CHW()
    return Layout.HWC()


# -------------------------
# Alpha mode (none/straight/premultiplied)
# -------------------------
struct AlphaMode(Copyable, Movable):
    var id: Int

    fn __init__(out self, id: Int):
        self.id = id

    @staticmethod
    fn NONE() -> AlphaMode:
        return AlphaMode(0)

    @staticmethod
    fn STRAIGHT() -> AlphaMode:
        return AlphaMode(1)

    @staticmethod
    fn PREMULTIPLIED() -> AlphaMode:
        return AlphaMode(2)

    fn __eq__(self, other: AlphaMode) -> Bool:
        return self.id == other.id

    fn to_string(self) -> String:
        if self.id == 0:
            return String("NONE")
        if self.id == 1:
            return String("STRAIGHT")
        if self.id == 2:
            return String("PREMULTIPLIED")
        return String("UNKNOWN")


# -------------------------
# Helpers
# -------------------------
@staticmethod
fn num_channels(order: ChannelOrder) -> Int:
    if order == ChannelOrder.GRAY():
        return 1
    if order == ChannelOrder.RGB() or order == ChannelOrder.BGR():
        return 3
    # RGBA variants
    return 4


@staticmethod
fn has_alpha(order: ChannelOrder) -> Bool:
    return order == ChannelOrder.RGBA() or order == ChannelOrder.BGRA() or order == ChannelOrder.ARGB()


# Expected shape given H, W, order, layout
@staticmethod
fn expected_shape(height: Int, width: Int, order: ChannelOrder, layout: Layout) -> List[Int]:
    var c = num_channels(order)
    var out: List[Int] = List[Int]()
    if layout == Layout.HWC():
        out.append(height)
        out.append(width)
        out.append(c)
        return out
    # CHW
    out.append(c)
    out.append(height)
    out.append(width)
    return out


# Minimal smoke test (does not rely on any external modules).
@staticmethod
fn __self_test__() -> Bool:
    var ok = True
    if num_channels(ChannelOrder.RGBA()) != 4:
        ok = False
    if has_alpha(ChannelOrder.RGB()):
        ok = False
    var shp = expected_shape(480, 640, ChannelOrder.RGB(), Layout.HWC())
    if len(shp) != 3:
        ok = False
    return ok
