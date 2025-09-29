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
# File: momijo/vision/dtypes.mojo
#
# NOTE:
# We implement enum-like "newtypes" using lightweight structs + static factories.
# This avoids unstable enum/decorator features while keeping a stable public API.

# -----------------------------------------------------------------------------
# DType: enum-like newtype (UInt8, UInt16, Int32, Float16, Float32, Int64, Float64)
# -----------------------------------------------------------------------------

struct DType(ExplicitlyCopyable, Movable):
    var id: Int

    fn __init__(out self, id: Int):
        self.id = id

    fn __copyinit__(out self, other: DType):
        self.id = other.id

    # Factories (enum-like constants)
    @staticmethod
    fn UInt8() -> DType:
        return DType(0)

    @staticmethod
    fn UInt16() -> DType:
        return DType(1)

    @staticmethod
    fn Int32() -> DType:
        return DType(2)

    @staticmethod
    fn Float16() -> DType:
        return DType(3)

    @staticmethod
    fn Float32() -> DType:
        return DType(4)

    @staticmethod
    fn Int64() -> DType:
        return DType(5)

    @staticmethod
    fn Float64() -> DType:
        return DType(6)

    # Equality
    fn __eq__(self, other: DType) -> Bool:
        return self.id == other.id

    fn __ne__(self, other: DType) -> Bool:
        return self.id != other.id

    # Human-readable name
    fn name(self) -> String:
        var i = self.id
        if i == 0: return "UInt8"
        if i == 1: return "UInt16"
        if i == 2: return "Int32"
        if i == 3: return "Float16"
        if i == 4: return "Float32"
        if i == 5: return "Int64"
        if i == 6: return "Float64"
        return "Unknown"

    # Size in bytes
    fn nbytes(self) -> Int:
        var i = self.id
        if i == 0: return 1
        if i == 1: return 2
        if i == 2: return 4
        if i == 3: return 2
        if i == 4: return 4
        if i == 5: return 8
        if i == 6: return 8
        return 1

    fn is_integer(self) -> Bool:
        var i = self.id
        return i == 0 or i == 1 or i == 2 or i == 5

    fn is_float(self) -> Bool:
        var i = self.id
        return i == 3 or i == 4 or i == 6

    @staticmethod
    fn from_name(name: String) -> (Bool, DType):
        if name == "UInt8":   return (True, DType.UInt8())
        if name == "UInt16":  return (True, DType.UInt16())
        if name == "Int32":   return (True, DType.Int32())
        if name == "Float16": return (True, DType.Float16())
        if name == "Float32": return (True, DType.Float32())
        if name == "Int64":   return (True, DType.Int64())
        if name == "Float64": return (True, DType.Float64())
        return (False, DType.UInt8())

# Free helpers (compat with old API)
fn dtype_bytes(dt: DType) -> Int:     return dt.nbytes()
fn dtype_name(dt: DType) -> String:   return dt.name()
fn dtype_from_name(name: String) -> (Bool, DType): return DType.from_name(name)
fn dtype_is_integer(dt: DType) -> Bool: return dt.is_integer()
fn dtype_is_float(dt: DType) -> Bool:   return dt.is_float()

# -----------------------------------------------------------------------------
# Layout: enum-like newtype
# -----------------------------------------------------------------------------

struct Layout(ExplicitlyCopyable, Movable):
    var id: Int

    fn __init__(out self, id: Int):
        self.id = id

    fn __copyinit__(out self, other: Layout):
        self.id = other.id

    @staticmethod
    fn HWC() -> Layout:
        return Layout(0)

    @staticmethod
    fn CHW() -> Layout:
        return Layout(1)

    @staticmethod
    fn PLANAR() -> Layout:
        return Layout(2)

    fn __eq__(self, other: Layout) -> Bool:
        return self.id == other.id

    fn __ne__(self, other: Layout) -> Bool:
        return self.id != other.id

    fn name(self) -> String:
        var i = self.id
        if i == 0: return "HWC"
        if i == 1: return "CHW"
        if i == 2: return "PLANAR"
        return "Unknown"

    @staticmethod
    fn from_name(name: String) -> (Bool, Layout):
        if name == "HWC":    return (True, Layout.HWC())
        if name == "CHW":    return (True, Layout.CHW())
        if name == "PLANAR": return (True, Layout.PLANAR())
        return (False, Layout.HWC())

fn layout_name(l: Layout) -> String: return l.name()
fn layout_from_name(name: String) -> (Bool, Layout): return Layout.from_name(name)

# -----------------------------------------------------------------------------
# ColorSpace: enum-like newtype
# -----------------------------------------------------------------------------

struct ColorSpace(ExplicitlyCopyable, Movable):
    var id: Int

    fn __init__(out self, id: Int):
        self.id = id

    fn __copyinit__(out self, other: ColorSpace):
        self.id = other.id

    @staticmethod
    fn SRGB() -> ColorSpace:
        return ColorSpace(0)

    @staticmethod
    fn Linear() -> ColorSpace:
        return ColorSpace(1)

    @staticmethod
    fn Gray() -> ColorSpace:
        return ColorSpace(2)

    @staticmethod
    fn YCbCr() -> ColorSpace:
        return ColorSpace(3)

    @staticmethod
    fn Unknown() -> ColorSpace:
        return ColorSpace(255)

    fn __eq__(self, other: ColorSpace) -> Bool:
        return self.id == other.id

    fn __ne__(self, other: ColorSpace) -> Bool:
        return self.id != other.id

    fn name(self) -> String:
        var i = self.id
        if i == 0:   return "SRGB"
        if i == 1:   return "Linear"
        if i == 2:   return "Gray"
        if i == 3:   return "YCbCr"
        if i == 255: return "Unknown"
        return "Unknown"

    @staticmethod
    fn from_name(name: String) -> (Bool, ColorSpace):
        if name == "SRGB":    return (True, ColorSpace.SRGB())
        if name == "Linear":  return (True, ColorSpace.Linear())
        if name == "Gray":    return (True, ColorSpace.Gray())
        if name == "YCbCr":   return (True, ColorSpace.YCbCr())
        if name == "Unknown": return (True, ColorSpace.Unknown())
        return (False, ColorSpace.Unknown())

fn colorspace_name(cs: ColorSpace) -> String: return cs.name()
fn colorspace_from_name(name: String) -> (Bool, ColorSpace): return ColorSpace.from_name(name)
