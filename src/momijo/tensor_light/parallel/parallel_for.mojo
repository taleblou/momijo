# SPDX-FileCopyrightText: 2026 Morteza Taleblou
# SPDX-FileCopyrightText: 2026 Mitra Daneshmand
# SPDX-License-Identifier: MIT
#
# ==============================================================================
# MOMIJO | TENSOR PARALLEL / CPU PARALLEL RANGES
# ==============================================================================
#
# Project:       Momijo
# Component:     Tensor
# Subcomponent:  FP32 Light / CPU Parallel Work
# Module:        momijo.tensor_light.parallel.parallel_for
# Source:        src/momijo/tensor_light/parallel/parallel_for.mojo
# Language:      Mojo
#
# ------------------------------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------------------------------
#
# Description:
#   Defines CPU parallel ranges in the CPU parallel work layer of the FP32
#   Light tensor implementation.
#
#   Public declarations include `ParallelRange`, `CpuWorkPolicy`,
#   `recommend_worker_count`, `shard_range`; preconditions and supported
#   targets are determined by their implementations.
#
# Responsibilities:
#   - Partition CPU work and validate independent ranges.
#   - Preserve documented behavior for shape, dtype, storage, and API
#     preconditions.
#   - Provide source-declared interfaces such as `ParallelRange`,
#     `CpuWorkPolicy`, `recommend_worker_count`, `shard_range`,
#     `execute_parallel_shards`.
#
# ------------------------------------------------------------------------------
# DESIGN AND ENGINEERING GUIDELINES
# ------------------------------------------------------------------------------
#
# Correctness:
#   - Partition work into valid, disjoint ranges.
#   - Preserve deterministic reductions when required by the API.
#
# Performance Objectives:
#   - Limit scheduling overhead for small workloads.
#   - Align partitions with profitable memory and vector boundaries.
#
# Maintainability:
#   - Keep work distribution separate from kernel semantics.
#   - Test irregular tails and empty partitions.
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

# Deterministic, non-overlapping CPU work partition. The kernel runs per-shard
# through AOT target lowering (no runtime closure in inner element loops).
from ..internal.status import TensorStatus, TensorResult
from max.algorithm import parallelize
from std.runtime import initialize_runtime

@fieldwise_init
struct ParallelRange(Copyable):
    var begin: Int
    var end: Int

@fieldwise_init
struct CpuWorkPolicy(Copyable):
    var minimum_items_per_worker: Int
    var maximum_workers: Int

def recommend_worker_count(items: Int, policy: CpuWorkPolicy) -> Int:
    if items <= 0 or policy.maximum_workers <= 1:
        return 1
    var minimum = max(1, policy.minimum_items_per_worker)
    # Ceiling division without adding (minimum - 1) to a potentially MAX Int.
    var chunks = items // minimum
    if items % minimum != 0:
        chunks += 1
    return max(1, min(policy.maximum_workers, chunks))

def shard_range(count: Int, worker: Int, workers: Int) -> TensorResult[ParallelRange]:
    if count < 0 or workers <= 0 or worker < 0 or worker >= workers:
        return TensorResult[ParallelRange](TensorStatus.INVALID_SHAPE)
    # CPU shards in this library address FP32 elements, not arbitrary byte
    # buffers. A forged giant count must not enter offset/byte arithmetic.
    if count > Int.MAX // 4:
        return TensorResult[ParallelRange](TensorStatus.OVERFLOW)
    # Division/remainder formula avoids worker * count overflow.
    var q = count // workers
    var rem = count % workers
    var begin = worker * q + min(worker, rem)
    var length = q
    if worker < rem:
        length += 1
    return TensorResult[ParallelRange](ParallelRange(begin, begin + length))


# Real CPU parallel substrate for AOT caller-provided, noncapturing or
# explicitly borrowed task bodies. `task(shard_id)` owns exactly the range
# returned by shard_range(count, shard_id, worker_count). All task bodies
# must be independently proven write-disjoint BEFORE this function is called.
# This is synchronous; MAX joins all workers before the owners may be freed.
# No nested closure or runtime virtual callback is constructed by Tensor.
def execute_parallel_shards[Task: def(Int) -> None](
    task: Task, count: Int, policy: CpuWorkPolicy,
) -> TensorStatus:
    if count < 0 or policy.maximum_workers <= 0 or policy.minimum_items_per_worker <= 0:
        return TensorStatus.INVALID_SHAPE
    if count > Int.MAX // 4:
        return TensorStatus.OVERFLOW
    if count == 0:
        return TensorStatus.OK
    var worker_count = recommend_worker_count(count, policy)
    if worker_count == 1:
        task(0)
        return TensorStatus.OK
    # The threshold has already rejected tiny tasks. A tensor library can be
    # entered from a non-Mojo host without a Mojo main() startup sequence.
    # On the real parallel path, explicitly initialize the shared worker
    # runtime (the 1.1 API is idempotent); avoid this cost on serial inputs.
    initialize_runtime()
    # MAX joins every shard before the owning storage may be destroyed.
    parallelize(task, worker_count, worker_count)
    return TensorStatus.OK
