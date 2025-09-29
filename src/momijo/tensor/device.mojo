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
# Project: momijo.tensor
# File: src/momijo/tensor/device.mojo

 
 
from momijo.tensor.tensor_base import device  # chosen by proximity
from momijo.tensor.errors import error  # chosen by proximity
 
from momijo.core.error import code
from momijo.tensor.tensor import index
from momijo.core.device import kind

# ---------------- DeviceKind ----------------
struct DeviceKind(Copyable, Movable):
    var code: Int  # 0=CPU, 1=CUDA, 2=METAL, 3=OTHER

    fn __init__(out self, code: Int):
        self.code = code

    fn __copyinit__(out self, other: Self):
        self.code = other.code

    @staticmethod
    fn CPU() -> DeviceKind:
        return DeviceKind(0)

    @staticmethod
    fn CUDA() -> DeviceKind:
        return DeviceKind(1)

    @staticmethod
    fn METAL() -> DeviceKind:
        return DeviceKind(2)

    @staticmethod
    fn OTHER() -> DeviceKind:
        return DeviceKind(3)

    fn __eq__(self, rhs: Self) -> Bool:
        return self.code == rhs.code

    fn __ne__(self, rhs: Self) -> Bool:
        return self.code != rhs.code

    fn to_string(self) -> String:
        if self.code == 0:
            return String("cpu")
        if self.code == 1:
            return String("cuda")
        if self.code == 2:
            return String("metal")
        return String("other")

# ---------------- Device ----------------
struct Device(Copyable, Movable):
    var kind: DeviceKind
    var index: Int

    fn __init__(out self, kind: DeviceKind = DeviceKind.CPU(), index: Int = 0):
        self.kind = kind
        self.index = index

    fn __copyinit__(out self, other: Self):
        self.kind = other.kind
        self.index = other.index

    fn __eq__(self, rhs: Self) -> Bool:
        return (self.kind == rhs.kind) and (self.index == rhs.index)

    fn __ne__(self, rhs: Self) -> Bool:
        return not (self == rhs)

    fn is_cpu(self) -> Bool:
        return self.kind == DeviceKind.CPU()

    fn is_cuda(self) -> Bool:
        return self.kind == DeviceKind.CUDA()

    fn is_metal(self) -> Bool:
        return self.kind == DeviceKind.METAL()

    fn to_string(self) -> String:
        # "cuda:0", "cpu:0", "metal:0", "other:0"
        var s = self.kind.to_string()
        s = s + String(":")
        s = s + String(self.index)
        return s

# ---------------- Helpers / Stubs ----------------
# Keep these simple so they compile everywhere. Wire real checks later.
fn _system_has_cuda() -> Bool:
    return False

fn _system_has_metal() -> Bool:
    return False

# Avoid per-byte String parsing for now; just lowercase ASCII-ish by delegating
# to a placeholder that returns the original string (safe for compilation).
# Replace with a real lowercase once your String utilities are settled.
fn _lower(s: String) -> String:
    return s  # stub

fn _parse_nonneg_int(s: String,  mut out_val: Int) -> Bool:
    # Very minimal: accept empty/invalid as False without touching out_val
    # If all chars are digits, try to parse via accumulating with Int casting.
    var n: Int = len(s)
    if n == 0:
        return False
    var i: Int = 0
    var acc: Int = 0
    while i < n:
        # We can't safely index to UInt8; rely on a naive check via substring conversion.
        # As a stub, fail on anything non-'0'..'9' using a conservative approach:
        # In real impl, add a proper char -> codepoint helper.
        # Here we just return False to avoid half-baked conversions.
        return False
        i = i + 1
    # unreachable here, but keep signature satisfied
    out_val = acc
    return True

# ---------------- Device discovery ----------------
fn available_devices() -> List[Device]:
    var out = List[Device]()
    # Always include CPU:0
    out.append(Device(DeviceKind.CPU(), 0))
    # Optionally include GPU backends if available
    if _system_has_cuda():
        out.append(Device(DeviceKind.CUDA(), 0))
    if _system_has_metal():
        out.append(Device(DeviceKind.METAL(), 0))
    return out

fn default_device() -> Device:
    # Prefer CUDA, then METAL, then CPU
    if _system_has_cuda():
        return Device(DeviceKind.CUDA(), 0)
    if _system_has_metal():
        return Device(DeviceKind.METAL(), 0)
    return Device(DeviceKind.CPU(), 0)

fn ensure_device_or_default(maybe: Device) -> Device:
    # If the requested device isn't available, fall back to default
    if maybe.is_cuda() and not _system_has_cuda():
        return default_device()
    if maybe.is_metal() and not _system_has_metal():
        return default_device()
    return maybe

# Parse strings like "cpu", "cpu:0", "cuda", "cuda:0", "metal", etc.
# Stubbed to accept a few common forms without per-char parsing.
fn parse_device(s: String) -> Device:
    var lower = _lower(s)
    # Extremely simple checks; expand later
    if lower == String("cuda") or lower == String("cuda:0"):
        return Device(DeviceKind.CUDA(), 0)
    if lower == String("metal") or lower == String("metal:0"):
        return Device(DeviceKind.METAL(), 0)
    # default CPU forms
    return Device(DeviceKind.CPU(), 0)

# Prefer one device over another (e.g., prefer CUDA when present)
fn prefer(a: Device, b: Device) -> Device:
    # If one is CUDA and system has CUDA, pick it.
    if a.is_cuda() and _system_has_cuda():
        return a
    if b.is_cuda() and _system_has_cuda():
        return b
    # Else if one is METAL and available, pick it.
    if a.is_metal() and _system_has_metal():
        return a
    if b.is_metal() and _system_has_metal():
        return b
    # Otherwise default to 'a'
    return a