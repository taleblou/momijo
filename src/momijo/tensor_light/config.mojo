# SPDX-FileCopyrightText: 2026 Morteza Taleblou
# SPDX-FileCopyrightText: 2026 Mitra Daneshmand
# SPDX-License-Identifier: MIT
#
# ==============================================================================
# MOMIJO | TENSOR / RUNTIME CONFIGURATION
# ==============================================================================
#
# Project:       Momijo
# Component:     Tensor
# Subcomponent:  FP32 Light / Runtime Configuration
# Module:        momijo.tensor_light.config
# Source:        src/momijo/tensor_light/config.mojo
# Language:      Mojo
#
# ------------------------------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------------------------------
#
# Description:
#   Defines runtime configuration in the runtime configuration layer of the
#   FP32 Light tensor implementation.
#
#   Public declarations include `validate_rank_config`,
#   `default_contiguous_order`, `preferred_simd_width`, `preferred_alignment`;
#   preconditions and supported targets are determined by their
#   implementations.
#
# Responsibilities:
#   - Provide reusable checked primitives and status contracts.
#   - Preserve documented behavior for shape, dtype, storage, and API
#     preconditions.
#   - Provide source-declared interfaces such as `validate_rank_config`,
#     `default_contiguous_order`, `preferred_simd_width`,
#     `preferred_alignment`, `default_fp32_math_mode` and related symbols.
#
# ------------------------------------------------------------------------------
# DESIGN AND ENGINEERING GUIDELINES
# ------------------------------------------------------------------------------
#
# Correctness:
#   - Validate target/rank tuning values before consumer selection.
#   - Treat offline calibration as policy metadata, not execution proof.
#
# Performance Objectives:
#   - Keep target policies static or cheap to query at runtime.
#   - Prefer ahead-of-time tuning decisions when the target is known.
#
# Maintainability:
#   - Centralize target, rank, and alignment defaults.
#   - Update configuration consumers and validation together.
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

# UASDEA compile-time FP32 / static-rank policy, never mutable global state.
# Workload-specific tuning is supplied by the offline TargetTuningProfile,
# not hard-coded or measured inside the Tensor library at runtime.
from std.sys.info import simd_width_of
from .internal.status import TensorStatus, TensorResult
from .layout.memory_format import ContiguousOrder
from .dtype.math_mode import Fp32MathMode, ReductionAccuracyMode
from .device.target import TargetKind, TargetSpec, validate_target_spec
from .execution.execution_context import ExecutionPreference
from .execution.launch_policy import HybridTuningProfile, validate_hybrid_tuning_profile

comptime MAX_TENSOR_RANK = 8
comptime TENSOR_DTYPE = DType.float32
comptime TENSOR_ITEMSIZE = 4


def validate_rank_config[rank: Int]() -> TensorStatus:
    comptime assert rank >= 0 and rank <= MAX_TENSOR_RANK, "FP32 Tensor rank must be 0..8"
    return TensorStatus.OK


def default_contiguous_order() -> ContiguousOrder:
    return ContiguousOrder.C


def preferred_simd_width() -> Int:
    # Native target width, specialized offline; no dtype or ISA switch in loops.
    return simd_width_of[DType.float32]()


def preferred_alignment[is_device: Bool]() -> Int:
    # Conservative creation default; larger target-specific alignments remain
    # properties of an offline tuned allocator/launch policy.
    comptime if is_device:
        return 256
    return 64


def default_fp32_math_mode() -> Fp32MathMode:
    # No implicit TF32 or multiply-add contraction permission by default.
    return Fp32MathMode.STRICT_FP32


def default_reduction_accuracy_mode() -> ReductionAccuracyMode:
    # Stable reduction association unless FAST_TREE is explicitly requested.
    return ReductionAccuracyMode.DETERMINISTIC_PAIRWISE

# The core's execution-policy vocabulary is defined once in the execution
# layer. Do not create incompatible local copies of these types.


@fieldwise_init
struct AlignmentPolicy(Copyable):
    # These are conservative allocation minimums, not measured ISA claims.
    var host_bytes: Int
    var device_bytes: Int

    @staticmethod
    def defaults() -> Self:
        return Self(64, 256)

    def validate(self) -> TensorStatus:
        if self.host_bytes < TENSOR_ITEMSIZE or self.device_bytes < TENSOR_ITEMSIZE:
            return TensorStatus.INVALID_ALIGNMENT
        if (self.host_bytes & (self.host_bytes - 1) != 0
            or self.device_bytes & (self.device_bytes - 1) != 0):
            return TensorStatus.INVALID_ALIGNMENT
        return TensorStatus.OK


@fieldwise_init
struct TensorCoreConfig(Copyable):
    # Immutable-by-policy execution inputs; no mutable process-wide registry.
    # Public fields are revalidated at each checked dispatch boundary.
    var preference: ExecutionPreference
    var fp32_mode: Fp32MathMode
    var reduction_mode: ReductionAccuracyMode
    var accelerator_bound: Bool
    var allow_hybrid: Bool
    var offline_hybrid_profile_bound: Bool

    @staticmethod
    def cpu_only() -> Self:
        return Self(
            ExecutionPreference.CPU, Fp32MathMode.STRICT_FP32,
            ReductionAccuracyMode.DETERMINISTIC_PAIRWISE, False, False, False,
        )

    def validate(self) -> TensorStatus:
        if self.preference.code < 0 or self.preference.code > 2:
            return TensorStatus.INVALID_LAYOUT
        if self.fp32_mode.code < 0 or self.fp32_mode.code > 3:
            return TensorStatus.MATH_MODE_UNSUPPORTED
        if self.reduction_mode.code < 0 or self.reduction_mode.code > 1:
            return TensorStatus.MATH_MODE_UNSUPPORTED
        if self.preference.code == ExecutionPreference.GPU.code and not self.accelerator_bound:
            return TensorStatus.BACKEND_UNAVAILABLE
        if self.allow_hybrid and not self.accelerator_bound:
            return TensorStatus.BACKEND_UNAVAILABLE
        if self.offline_hybrid_profile_bound and not self.accelerator_bound:
            return TensorStatus.BACKEND_UNAVAILABLE
        return TensorStatus.OK


@fieldwise_init
struct TargetTuningProfile(Copyable):
    # AOT selection metadata only. A profile does not authenticate a device,
    # its loaded machine code or the calibrated performance of that code.
    var target_kind: TargetKind
    var alignment_bytes: Int
    var fp32_vector_lanes: Int
    var tile_m: Int
    var tile_n: Int
    var tile_k: Int
    var unroll_factor: Int
    var cpu_parallel_grain: Int
    var gpu_threads_per_block: Int
    var cache_block_bytes: Int
    var requested_gpu_shared_bytes: Int
    var prefetch_distance: Int
    var aot_variant_id: Int
    var offline_calibrated: Bool

    def validate(self) -> TensorStatus:
        if self.target_kind.code < TargetKind.CPU.code or self.target_kind.code > TargetKind.APPLE_GPU.code:
            return TensorStatus.DEVICE_MISMATCH
        if self.alignment_bytes < TENSOR_ITEMSIZE:
            return TensorStatus.INVALID_ALIGNMENT
        if self.alignment_bytes & (self.alignment_bytes - 1) != 0:
            return TensorStatus.INVALID_ALIGNMENT
        if self.fp32_vector_lanes < 1:
            return TensorStatus.INVALID_LAYOUT
        if self.fp32_vector_lanes & (self.fp32_vector_lanes - 1) != 0:
            return TensorStatus.INVALID_LAYOUT
        if self.tile_m < 1 or self.tile_n < 1 or self.tile_k < 1:
            return TensorStatus.INVALID_LAYOUT
        if self.tile_m > Int.MAX // self.tile_n:
            return TensorStatus.OVERFLOW
        if self.tile_m * self.tile_n > Int.MAX // self.tile_k:
            return TensorStatus.OVERFLOW
        if self.unroll_factor < 1 or self.cpu_parallel_grain < 0:
            return TensorStatus.INVALID_LAYOUT
        if (self.gpu_threads_per_block < 0 or self.aot_variant_id < -1
            or self.cache_block_bytes < 0 or self.requested_gpu_shared_bytes < 0
            or self.prefetch_distance < 0):
            return TensorStatus.INVALID_LAYOUT
        if self.target_kind.code == TargetKind.CPU.code:
            if self.gpu_threads_per_block != 0 or self.requested_gpu_shared_bytes != 0:
                return TensorStatus.INVALID_LAYOUT
            if self.cpu_parallel_grain < 1:
                return TensorStatus.INVALID_LAYOUT
        else:
            if self.cpu_parallel_grain != 0:
                return TensorStatus.INVALID_LAYOUT
        # Untuned defaults must never advertise a trusted native variant.
        if not self.offline_calibrated:
            if self.aot_variant_id != -1:
                return TensorStatus.UNSUPPORTED
        elif self.aot_variant_id < 0:
            return TensorStatus.INVALID_LAYOUT
        if self.offline_calibrated and self.target_kind.code != TargetKind.CPU.code:
            if self.gpu_threads_per_block < 1:
                return TensorStatus.INVALID_LAYOUT
        return TensorStatus.OK

    def checked_snapshot(self) -> TensorResult[Self]:
        var status = self.validate()
        if status != TensorStatus.OK:
            return TensorResult[Self](status)
        return TensorResult[Self](self.copy())


def default_execution_preference[target_kind: Int]() -> ExecutionPreference:
    comptime assert target_kind >= 0 and target_kind <= 3, "Unsupported target family"
    # Merely selecting a preference never grants a native device lease.
    comptime if target_kind == 0:
        return ExecutionPreference.CPU
    return ExecutionPreference.GPU


def hybrid_execution_enabled(config: TensorCoreConfig) -> Bool:
    # This is an eligibility gate, not proof of the GPU stream, memory leases,
    # artifact binding or actual cost-model profitability.
    if config.validate() != TensorStatus.OK:
        return False
    return (
        config.preference.code == ExecutionPreference.AUTO.code
        and config.accelerator_bound and config.allow_hybrid
        and config.offline_hybrid_profile_bound
    )


def hybrid_execution_enabled[
    allow_hybrid: Bool, accelerator_bound: Bool, offline_profile_bound: Bool,
]() -> Bool:
    # Compile-time counterpart for AOT specialization without runtime state.
    return allow_hybrid and accelerator_bound and offline_profile_bound


def preferred_alignment[target_kind: Int]() -> Int:
    comptime assert target_kind >= 0 and target_kind <= 3, "Unsupported target family"
    comptime if target_kind == 0:
        return 64
    return 256


def preferred_simd_width[target_kind: Int]() -> Int:
    comptime assert target_kind >= 0 and target_kind <= 3, "Unsupported target family"
    comptime if target_kind == 0:
        return simd_width_of[DType.float32]()
    # Warp width and the SIMD width of one GPU thread are not interchangeable.
    return 1


def default_fp32_math_mode[target_kind: Int]() -> Fp32MathMode:
    comptime assert target_kind >= 0 and target_kind <= 3, "Unsupported target family"
    # STRICT_FP32 by default: faster reduced-precision modes require opt-in.
    return Fp32MathMode.STRICT_FP32


def target_tuning_profile[target_kind: Int]() -> TargetTuningProfile:
    comptime assert target_kind >= 0 and target_kind <= 3, "Unsupported target family"
    # Deliberately safe, noncalibrated scalar fallback. Offline optimization
    # may replace this with a checked, externally supplied AOT profile.
    # -1 means NO selected executable variant, not a working GPU kernel.
    comptime if target_kind == 0:
        return TargetTuningProfile(
            TargetKind.CPU, 64, 1, 1, 1, 1, 1, 65536, 0, 0, 0, 0, -1, False,
        )
    comptime if target_kind == 1:
        return TargetTuningProfile(
            TargetKind.NVIDIA_GPU, 256, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, -1, False,
        )
    comptime if target_kind == 2:
        return TargetTuningProfile(
            TargetKind.AMD_GPU, 256, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, -1, False,
        )
    return TargetTuningProfile(
        TargetKind.APPLE_GPU, 256, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, -1, False,
    )


def target_tuning_profile_from_offline[
    target_kind: Int,
](profile: TargetTuningProfile, bound_target: TargetSpec) -> TensorResult[TargetTuningProfile]:
    comptime assert target_kind >= 0 and target_kind <= 3, "Unsupported target family"
    # Metadata equality is not physical hardware or native ISA proof.
    if not validate_target_spec(bound_target):
        return TensorResult[TargetTuningProfile](TensorStatus.DEVICE_MISMATCH)
    if bound_target.kind.code != target_kind or profile.target_kind.code != target_kind:
        return TensorResult[TargetTuningProfile](TensorStatus.DEVICE_MISMATCH)
    var checked = profile.checked_snapshot()
    if not checked.is_ok():
        return TensorResult[TargetTuningProfile](checked.status)
    if not profile.offline_calibrated:
        return TensorResult[TargetTuningProfile](TensorStatus.UNSUPPORTED)
    if target_kind == 0 and profile.fp32_vector_lanes > bound_target.simd_lanes:
        return TensorResult[TargetTuningProfile](TensorStatus.UNSUPPORTED)
    if target_kind != 0:
        if profile.gpu_threads_per_block < 1:
            return TensorResult[TargetTuningProfile](TensorStatus.INVALID_LAYOUT)
        if profile.requested_gpu_shared_bytes > bound_target.shared_memory_bytes:
            return TensorResult[TargetTuningProfile](TensorStatus.INVALID_LAYOUT)
    return TensorResult[TargetTuningProfile](profile.copy())


def hybrid_tuning_profile[cpu_target: Int, gpu_target: Int]() -> TensorResult[HybridTuningProfile]:
    comptime assert cpu_target == 0, "Hybrid CPU target must use the CPU family"
    comptime assert gpu_target >= 1 and gpu_target <= 3, "Hybrid accelerator target invalid"
    # No compilation-time profile can honestly infer measured transfer,
    # launch or throughput coefficients from target family identifiers alone.
    return TensorResult[HybridTuningProfile](TensorStatus.UNSUPPORTED)


def hybrid_tuning_profile_from_offline[
    cpu_target: Int, gpu_target: Int,
](
    measured: HybridTuningProfile,
    bound_cpu: TargetSpec,
    bound_gpu: TargetSpec,
) -> TensorResult[HybridTuningProfile]:
    comptime assert cpu_target == 0, "Hybrid CPU target must use the CPU family"
    comptime assert gpu_target >= 1 and gpu_target <= 3, "Hybrid accelerator target invalid"
    if not validate_target_spec(bound_cpu) or not validate_target_spec(bound_gpu):
        return TensorResult[HybridTuningProfile](TensorStatus.DEVICE_MISMATCH)
    if bound_cpu.kind.code != cpu_target or bound_gpu.kind.code != gpu_target:
        return TensorResult[HybridTuningProfile](TensorStatus.DEVICE_MISMATCH)
    var valid = validate_hybrid_tuning_profile(measured)
    if valid != TensorStatus.OK:
        return TensorResult[HybridTuningProfile](valid)
    return TensorResult[HybridTuningProfile](measured.copy())

# The same source-spec entry point also accepts the actual immutable offline
# calibration. The no-argument overload above refuses to fabricate one.
def target_tuning_profile[target_kind: Int](
    profile: TargetTuningProfile, bound_target: TargetSpec,
) -> TensorResult[TargetTuningProfile]:
    return target_tuning_profile_from_offline[target_kind](profile, bound_target)


def hybrid_tuning_profile[cpu_target: Int, gpu_target: Int](
    measured: HybridTuningProfile,
    bound_cpu: TargetSpec,
    bound_gpu: TargetSpec,
) -> TensorResult[HybridTuningProfile]:
    return hybrid_tuning_profile_from_offline[cpu_target, gpu_target](
        measured, bound_cpu, bound_gpu,
    )
