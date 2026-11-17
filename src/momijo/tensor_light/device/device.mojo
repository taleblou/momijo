# SPDX-FileCopyrightText: 2026 Morteza Taleblou
# SPDX-FileCopyrightText: 2026 Mitra Daneshmand
# SPDX-License-Identifier: MIT
#
# ==============================================================================
# MOMIJO | TENSOR DEVICE / DEVICE IDENTITY
# ==============================================================================
#
# Project:       Momijo
# Component:     Tensor
# Subcomponent:  FP32 Light / Device and Stream Control
# Module:        momijo.tensor_light.device.device
# Source:        src/momijo/tensor_light/device/device.mojo
# Language:      Mojo
#
# ------------------------------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------------------------------
#
# Description:
#   Defines device identity in the device and stream control layer of the FP32
#   Light tensor implementation.
#
#   Public declarations include `DeviceType`, `Device`, `cpu_device`,
#   `gpu_device`; preconditions and supported targets are determined by their
#   implementations.
#
# Responsibilities:
#   - Represent checked device/stream/event state and compatibility.
#   - Preserve documented behavior for explicit device/context identity and
#     completion requirements.
#   - Provide source-declared interfaces such as `DeviceType`, `Device`,
#     `cpu_device`, `gpu_device`, `gpu_device_checked` and related symbols.
#   - Reject unsupported target or unverified completion states.
#
# ------------------------------------------------------------------------------
# DESIGN AND ENGINEERING GUIDELINES
# ------------------------------------------------------------------------------
#
# Correctness:
#   - Match device ordinals, contexts, streams, and events explicitly.
#   - Do not treat an unrecorded event as proof of completion.
#   - Reject unsupported or unauthenticated execution paths explicitly.
#
# Performance Objectives:
#   - Avoid unnecessary synchronization outside required boundaries.
#   - Keep capability queries independent from data movement.
#   - Avoid implicit host/device migration or compilation at execution time.
#
# Maintainability:
#   - Differentiate structural device metadata from active resources.
#   - Test mismatched device/stream/context paths.
#   - Keep device-specific capability statements conservative.
#
# ------------------------------------------------------------------------------
# PROJECT INFORMATION
# ------------------------------------------------------------------------------
#
# Authors:
#   Morteza Taleblou
#   Mitra Daneshmand
#
# Organization:  https://github.com/momijo
# Repository:    https://github.com/taleblou/momijo
#
# Maintainer Website:
#   https://taleblou.ir/
#
# Maintainer LinkedIn:
#   https://www.linkedin.com/in/morteza-taleblou
#
# ------------------------------------------------------------------------------
# LICENSING AND ATTRIBUTION
# ------------------------------------------------------------------------------
#
# License:       MIT
# License File:  LICENSE
#
# Copyright and Licensing:
#   This source file is distributed under the MIT License.
#   The SPDX declarations above identify its applicable license and
#   stated copyright holders.
#
#   Redistributions containing this software or substantial portions
#   of it must retain the applicable copyright and permission notices
#   as required by the MIT License.
#
# Attribution:
#   Acknowledgment of the Momijo project in downstream documentation
#   is appreciated but is not an additional licensing requirement.
#
# ==============================================================================

# Device identity is stable metadata, not a placement scheduler.
from ..internal.status import TensorResult, TensorStatus
@fieldwise_init
struct DeviceType(Copyable, Equatable):
    var code: Int
    comptime CPU = Self(0)
    comptime GPU = Self(1)
    def __eq__(self, other: Self) -> Bool:
        return self.code == other.code

@fieldwise_init
struct Device(Copyable):
    var kind: DeviceType
    var ordinal: Int

    def is_cpu(self) -> Bool:
        return self.kind == DeviceType.CPU

    def is_gpu(self) -> Bool:
        return self.kind == DeviceType.GPU

def cpu_device() -> Device:
    return Device(DeviceType.CPU, 0)

def _gpu_device_unchecked(index: Int) -> Device:
    # Internal only, after an explicit nonnegative ordinal proof.
    return Device(DeviceType.GPU, index)


def gpu_device(index: Int) -> TensorResult[Device]:
    # Public construction must not expose a negative GPU ordinal as a valid
    # target. The device backend separately verifies physical availability.
    if index < 0:
        return TensorResult[Device](TensorStatus.INVALID_AXIS)
    return TensorResult[Device](_gpu_device_unchecked(index))


def gpu_device_checked(index: Int) -> TensorResult[Device]:
    return gpu_device(index)


def validate_device(device: Device) -> TensorStatus:
    if device.kind == DeviceType.CPU:
        if device.ordinal == 0:
            return TensorStatus.OK
        return TensorStatus.DEVICE_MISMATCH
    if device.kind == DeviceType.GPU and device.ordinal >= 0:
        return TensorStatus.OK
    return TensorStatus.DEVICE_MISMATCH


def is_cpu(device: Device) -> Bool:
    return validate_device(device) == TensorStatus.OK and device.is_cpu()


def is_gpu(device: Device) -> Bool:
    return validate_device(device) == TensorStatus.OK and device.is_gpu()

def device_index(device: Device) -> TensorResult[Int]:
    if device.is_cpu():
        if device.ordinal != 0:
            return TensorResult[Int](TensorStatus.DEVICE_MISMATCH)
        return TensorResult[Int](0)
    if device.is_gpu() and device.ordinal >= 0:
        return TensorResult[Int](device.ordinal)
    return TensorResult[Int](TensorStatus.DEVICE_MISMATCH)


def same_device(first: Device, second: Device) -> Bool:
    # Invalid fieldwise-constructed identities do not denote usable devices.
    if validate_device(first) != TensorStatus.OK or validate_device(second) != TensorStatus.OK:
        return False
    return first.kind == second.kind and first.ordinal == second.ordinal
