# SPDX-FileCopyrightText: 2026 Morteza Taleblou
# SPDX-FileCopyrightText: 2026 Mitra Daneshmand
# SPDX-License-Identifier: MIT
#
# ==============================================================================
# MOMIJO | TENSOR DEVICE / DEVICE EVENTS
# ==============================================================================
#
# Project:       Momijo
# Component:     Tensor
# Subcomponent:  FP32 Light / Device and Stream Control
# Module:        momijo.tensor_light.device.event
# Source:        src/momijo/tensor_light/device/event.mojo
# Language:      Mojo
#
# ------------------------------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------------------------------
#
# Description:
#   Defines device events in the device and stream control layer of the FP32
#   Light tensor implementation.
#
#   Public declarations include `create_event`, `wait_event`, `record`,
#   `wait`; preconditions and supported targets are determined by their
#   implementations.
#
# Responsibilities:
#   - Represent checked device/stream/event state and compatibility.
#   - Preserve documented behavior for explicit device/context identity and
#     completion requirements.
#   - Provide source-declared interfaces such as `create_event`, `wait_event`,
#     `record`, `wait`.
#   - Reject unsupported target or unverified completion states.
#
# ------------------------------------------------------------------------------
# DESIGN AND ENGINEERING GUIDELINES
# ------------------------------------------------------------------------------
#
# Correctness:
#   - Match device ordinals, contexts, streams, and events explicitly.
#   - Do not treat an unrecorded event as proof of completion.
#   - Keep owners alive until completion is actually established.
#
# Performance Objectives:
#   - Avoid unnecessary synchronization outside required boundaries.
#   - Keep capability queries independent from data movement.
#   - Overlap useful work only when event and ownership dependencies permit.
#
# Maintainability:
#   - Differentiate structural device metadata from active resources.
#   - Test mismatched device/stream/context paths.
#   - Test failed event joins, stale leases, and cleanup paths.
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

# Narrow MAX adapter. All failures from the raises API stay on this boundary.
from max.gpu.host import DeviceContext, DeviceEvent, DeviceStream

def create_event(ctx: DeviceContext) raises -> DeviceEvent:
    return ctx.create_event()

def wait_event(event: DeviceEvent) raises:
    event.synchronize()


def record(event: DeviceEvent, stream: DeviceStream) raises:
    # Enqueue a fence on the exact stream that owns the preceding writes.
    # No device-wide barrier or implied buffer-ownership transfer occurs.
    stream.record_event(event)


def wait(event: DeviceEvent) raises:
    # Host-side completion boundary. Owners must remain alive while waiting.
    wait_event(event)
