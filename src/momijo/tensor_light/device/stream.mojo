# SPDX-FileCopyrightText: 2026 Morteza Taleblou
# SPDX-FileCopyrightText: 2026 Mitra Daneshmand
# SPDX-License-Identifier: MIT
#
# ==============================================================================
# MOMIJO | TENSOR DEVICE / DEVICE STREAMS
# ==============================================================================
#
# Project:       Momijo
# Component:     Tensor
# Subcomponent:  FP32 Light / Device and Stream Control
# Module:        momijo.tensor_light.device.stream
# Source:        src/momijo/tensor_light/device/stream.mojo
# Language:      Mojo
#
# ------------------------------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------------------------------
#
# Description:
#   Defines device streams in the device and stream control layer of the FP32
#   Light tensor implementation.
#
#   Public declarations include `default_stream`, `create_stream`,
#   `record_event`, `enqueue_wait_event`; preconditions and supported targets
#   are determined by their implementations.
#
# Responsibilities:
#   - Represent checked device/stream/event state and compatibility.
#   - Preserve documented behavior for explicit device/context identity and
#     completion requirements.
#   - Provide source-declared interfaces such as `default_stream`,
#     `create_stream`, `record_event`, `enqueue_wait_event`, `wait_event`.
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

# Narrow stream adapter; no device-wide waits in inner execution loops.
from max.gpu.host import DeviceContext, DeviceStream, DeviceEvent
from .synchronization import synchronize_stream

def default_stream(ctx: DeviceContext) raises -> DeviceStream:
    return ctx.stream()

def create_stream(ctx: DeviceContext, priority: Int = 0) raises -> DeviceStream:
    return ctx.create_stream(priority=priority)

def record_event(stream: DeviceStream, event: DeviceEvent) raises:
    stream.record_event(event)

def enqueue_wait_event(stream: DeviceStream, event: DeviceEvent) raises:
    stream.enqueue_wait_for(event)



def wait_event(stream: DeviceStream, event: DeviceEvent) raises:
    # Device-side stream dependency: the caller still owns both buffers.
    enqueue_wait_event(stream, event)
