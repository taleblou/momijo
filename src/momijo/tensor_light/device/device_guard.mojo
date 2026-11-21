# SPDX-FileCopyrightText: 2026 Morteza Taleblou
# SPDX-FileCopyrightText: 2026 Mitra Daneshmand
# SPDX-License-Identifier: MIT
#
# ==============================================================================
# MOMIJO | TENSOR DEVICE / DEVICE GUARD
# ==============================================================================
#
# Project:       Momijo
# Component:     Tensor
# Subcomponent:  FP32 Light / Device and Stream Control
# Module:        momijo.tensor_light.device.device_guard
# Source:        src/momijo/tensor_light/device/device_guard.mojo
# Language:      Mojo
#
# ------------------------------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------------------------------
#
# Description:
#   Defines device guard in the device and stream control layer of the FP32
#   Light tensor implementation.
#
#   Public declarations include `DeviceGuard`; preconditions and supported
#   targets are determined by their implementations.
#
# Responsibilities:
#   - Represent checked device/stream/event state and compatibility.
#   - Preserve documented behavior for explicit device/context identity and
#     completion requirements.
#   - Provide source-declared interfaces such as `DeviceGuard`.
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

# Device selection is explicit. This checked compatibility adapter rejects
# changes of logical device instead of mutating ambient process-wide state.
from ..internal.status import TensorStatus
from .device import Device

struct DeviceGuard(Copyable):
    var selected: Device
    def __init__(out self, selected: Device):
        self.selected = selected.copy()

    def validate_device(self, actual: Device) -> TensorStatus:
        if self.selected.kind == actual.kind and self.selected.ordinal == actual.ordinal:
            return TensorStatus.OK
        return TensorStatus.DEVICE_MISMATCH
