# SPDX-FileCopyrightText: 2026 Morteza Taleblou
# SPDX-FileCopyrightText: 2026 Mitra Daneshmand
# SPDX-License-Identifier: MIT
#
# ==============================================================================
# MOMIJO | TENSOR DISPATCH / AOT SPECIALIZATION
# ==============================================================================
#
# Project:       Momijo
# Component:     Tensor
# Subcomponent:  FP32 Light / Dispatch and Specialization
# Module:        momijo.tensor_light.dispatch.specialization
# Source:        src/momijo/tensor_light/dispatch/specialization.mojo
# Language:      Mojo
#
# ------------------------------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------------------------------
#
# Description:
#   Defines AOT specialization in the dispatch and specialization layer of the
#   FP32 Light tensor implementation.
#
#   Public declarations include `SpecializationDecision`,
#   `choose_avx2_elementwise`, `choose_sm89_gemm`,
#   `choose_avx2_elementwise_for_capabilities`; preconditions and supported
#   targets are determined by their implementations.
#
# Responsibilities:
#   - Validate dispatch choices against target and artifact contracts.
#   - Preserve documented behavior for immutable plan metadata and trusted
#     dispatch preconditions.
#   - Provide source-declared interfaces such as `SpecializationDecision`,
#     `choose_avx2_elementwise`, `choose_sm89_gemm`,
#     `choose_avx2_elementwise_for_capabilities`,
#     `choose_checked_scalar_elementwise_for_spans` and related symbols.
#
# ------------------------------------------------------------------------------
# DESIGN AND ENGINEERING GUIDELINES
# ------------------------------------------------------------------------------
#
# Correctness:
#   - Validate dispatch keys and trusted artifact identities before selection.
#   - Treat tuning metadata as hints, not proof of executable code.
#
# Performance Objectives:
#   - Prefer precomputed finite strategies over runtime trial-and-error.
#   - Avoid scheduling or kernel compilation in pure selection utilities.
#
# Maintainability:
#   - Keep release schemas and validation checks versioned together.
#   - Test negative cases for unbound or inconsistent specialization data.
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

# AOT finite-variant selection: fallback is explicit, no per-call compilation.
from .dispatch_key import DispatchKey, dispatch_key_matches
from ..internal.status import TensorResult, TensorStatus
from .release_artifact_index import checked_release_gemm_binding
from ..device.capabilities import CpuCapabilities, GpuCapabilities
from ..device.target import TargetKind
from std.collections import Span
from ..alias.overlap import host_output_disjoint_from_input, host_output_exactly_matches_input

@fieldwise_init
struct SpecializationDecision(Copyable):
    var aot_variant: Int
    var vector_width: Int
    var tile_m: Int
    var tile_n: Int
    var tile_k: Int

    @staticmethod
    def checked(
        variant: Int, vector_width: Int,
        tile_m: Int, tile_n: Int, tile_k: Int,
    ) -> TensorResult[Self]:
        # This is a structural proof only, not an executable/ISA proof.
        # Avoid invalid geometry reaching an unchecked AOT launcher.
        if variant < 0 or vector_width < 1:
            return TensorResult[Self](TensorStatus.INVALID_LAYOUT)
        if vector_width & (vector_width - 1) != 0:
            return TensorResult[Self](TensorStatus.INVALID_LAYOUT)
        if tile_m < 1 or tile_n < 1 or tile_k < 1:
            return TensorResult[Self](TensorStatus.INVALID_LAYOUT)
        if tile_m > Int.MAX // tile_n:
            return TensorResult[Self](TensorStatus.OVERFLOW)
        var tile_area = tile_m * tile_n
        if tile_area > Int.MAX // tile_k:
            return TensorResult[Self](TensorStatus.OVERFLOW)
        return TensorResult[Self](Self(variant, vector_width, tile_m, tile_n, tile_k))

    def checked_snapshot(self) -> TensorResult[Self]:
        # Public fieldwise construction and var members are not a proof.
        return Self.checked(
            self.aot_variant, self.vector_width,
            self.tile_m, self.tile_n, self.tile_k,
        )

def choose_avx2_elementwise(key: DispatchKey) -> TensorResult[SpecializationDecision]:
    # This is a finite offline variant selector, not a CPU ISA detector.
    # The caller must bind this function to a separately verified AVX2/FMA
    # target; the integer DispatchKey alone does not grant ISA capabilities.
    var proof = key.checked_snapshot()
    if not proof.is_ok():
        return TensorResult[SpecializationDecision](proof.status)
    var selected = proof.payload.value()
    if selected.target_arch != 0:
        return TensorResult[SpecializationDecision](TensorStatus.DEVICE_MISMATCH)
    if selected.layout_class != 0:
        return TensorResult[SpecializationDecision](TensorStatus.NOT_REPRESENTABLE)
    if selected.shape_bucket <= 1:
        return SpecializationDecision.checked(0, 1, 1, 1, 1)
    # DispatchKey carries untrusted numeric metadata, not a live pointer.
    # Therefore it cannot authorize variant 2's aligned memory operations.
    # choose_avx2_elementwise_for_spans() proves every binary operand.
    return SpecializationDecision.checked(1, 8, 1, 1, 1)


def choose_sm89_gemm(key: DispatchKey) -> TensorResult[SpecializationDecision]:
    # This selector is ONLY legal for an externally verified NVIDIA Ada SM89
    # target. Runtime metadata does not authenticate the device or executable.
    var proof = key.checked_snapshot()
    if not proof.is_ok():
        return TensorResult[SpecializationDecision](proof.status)
    var selected = proof.payload.value()
    if selected.target_arch != 89:
        return TensorResult[SpecializationDecision](TensorStatus.DEVICE_MISMATCH)
    # Only strict-FP32 SIMT candidates exist here. TF32 Tensor-Core code
    # would require its own compiled instruction/ABI and validation.
    if selected.shape_bucket <= 1:
        return SpecializationDecision.checked(11, 1, 16, 32, 8)
    return SpecializationDecision.checked(10, 1, 32, 64, 8)


def choose_avx2_elementwise_for_capabilities(
    key: DispatchKey, capabilities: CpuCapabilities,
) -> TensorResult[SpecializationDecision]:
    # CPU ISA capability does not authenticate a memory address. A caller can
    # freely set DispatchKey.alignment_class=64. Without an owner-backed host
    # pointer, choose only the unaligned SIMD implementation (or scalar).
    var target_status = capabilities.validate()
    if target_status != TensorStatus.OK:
        return TensorResult[SpecializationDecision](target_status)
    if not capabilities.has_avx2 or not capabilities.has_fma:
        return TensorResult[SpecializationDecision](TensorStatus.UNSUPPORTED)
    var verified = key.checked_snapshot()
    if not verified.is_ok():
        return TensorResult[SpecializationDecision](verified.status)
    var selected = verified.payload.value()
    if selected.target_arch != 0:
        return TensorResult[SpecializationDecision](TensorStatus.DEVICE_MISMATCH)
    if selected.layout_class != 0:
        return TensorResult[SpecializationDecision](TensorStatus.NOT_REPRESENTABLE)
    if selected.shape_bucket <= 1:
        return SpecializationDecision.checked(0, 1, 1, 1, 1)
    return SpecializationDecision.checked(1, 8, 1, 1, 1)


# The fallback scalar variant is not a license for overlapping shifted
# operands. Scalar and SIMD kernels share one owner-backed input/output alias
# proof, so selecting a smaller ISA cannot silently change tensor semantics.
# An empty output never dereferences the (possibly dangling) empty Span ptr.
def choose_checked_scalar_elementwise_for_spans[
    out_origin: Origin[mut=True], left_origin: Origin, right_origin: Origin,
](
    key: DispatchKey, capabilities: CpuCapabilities,
    result: Span[True, Float32, out_origin],
    left: Span[False, Float32, left_origin],
    right: Span[False, Float32, right_origin],
) -> TensorResult[SpecializationDecision]:
    var cpu = capabilities.validate()
    if cpu != TensorStatus.OK:
        return TensorResult[SpecializationDecision](cpu)
    var verified = key.checked_snapshot()
    if not verified.is_ok():
        return TensorResult[SpecializationDecision](verified.status)
    var requested = verified.payload.value()
    # A borrowed host Span is a C-order flat region. A numeric GPU arch or
    # noncontiguous layout key cannot authorize its scalar native ABI.
    if requested.target_arch != 0:
        return TensorResult[SpecializationDecision](TensorStatus.DEVICE_MISMATCH)
    if requested.layout_class != 0:
        return TensorResult[SpecializationDecision](TensorStatus.NOT_REPRESENTABLE)
    var count = len(result)
    if count < 0 or count != len(left) or count != len(right):
        return TensorResult[SpecializationDecision](TensorStatus.SIZE_MISMATCH)
    if count > Int.MAX // 4:
        return TensorResult[SpecializationDecision](TensorStatus.OVERFLOW)
    if count == 0:
        return SpecializationDecision.checked(0, 1, 1, 1, 1)
    # Scalar loops are forward traversals just like SIMD loops. Shifted
    # overlaps would destroy a later unread input; demand exact in-place
    # elementwise alias or a genuinely disjoint range for EACH operand.
    var left_safe = host_output_disjoint_from_input(result, left)
    if not left_safe:
        left_safe = host_output_exactly_matches_input(result, left)
    var right_safe = host_output_disjoint_from_input(result, right)
    if not right_safe:
        right_safe = host_output_exactly_matches_input(result, right)
    if not left_safe or not right_safe:
        return TensorResult[SpecializationDecision](TensorStatus.ALIAS_UNKNOWN)
    return SpecializationDecision.checked(0, 1, 1, 1, 1)


# The aligned AVX2 binary variant is legal only after proving ALL concrete
# host addresses involved in its loads and stores. Checking one input is not
# enough: either second operand or the result may have a different alignment.
# This is a checked OWNER-BORROW boundary, not ISA autodetection or a JIT.
def choose_avx2_elementwise_for_spans[
    out_origin: Origin[mut=True], left_origin: Origin, right_origin: Origin,
](
    key: DispatchKey, capabilities: CpuCapabilities,
    result: Span[True, Float32, out_origin],
    left: Span[False, Float32, left_origin],
    right: Span[False, Float32, right_origin],
) -> TensorResult[SpecializationDecision]:
    var scalar = choose_checked_scalar_elementwise_for_spans(
        key, capabilities, result, left, right,
    )
    if not scalar.is_ok():
        return scalar
    var count = len(result)
    var verified = key.checked_snapshot()
    if not verified.is_ok():
        return TensorResult[SpecializationDecision](verified.status)
    # Both scalar and SIMD use the SAME proved alias contract, including
    # empty/small inputs. No fake AVX2 requirement for scalar work.
    if count < 8 or verified.payload.value().shape_bucket <= 1:
        return scalar
    if not capabilities.has_avx2 or not capabilities.has_fma:
        return TensorResult[SpecializationDecision](TensorStatus.UNSUPPORTED)
    # The checked scalar gate above proved count>0 and all input/output
    # ranges safe; now prove byte alignment for the faster native ABI.
    var out_address = Int(result.unsafe_ptr())
    var left_address = Int(left.unsafe_ptr())
    var right_address = Int(right.unsafe_ptr())
    var length_bytes = count * 4
    if out_address <= 0 or left_address <= 0 or right_address <= 0:
        return TensorResult[SpecializationDecision](TensorStatus.INVALID_ALIGNMENT)
    if (out_address > Int.MAX - length_bytes
        or left_address > Int.MAX - length_bytes
        or right_address > Int.MAX - length_bytes):
        return TensorResult[SpecializationDecision](TensorStatus.OVERFLOW)
    if out_address % 32 == 0 and left_address % 32 == 0 and right_address % 32 == 0:
        return SpecializationDecision.checked(2, 8, 1, 1, 1)
    # Never assume alignment from the untrusted dispatch key.
    return SpecializationDecision.checked(1, 8, 1, 1, 1)


def choose_sm89_gemm_for_capabilities(
    key: DispatchKey, capabilities: GpuCapabilities,
) -> TensorResult[SpecializationDecision]:
    # Declared capability metadata must agree with the AOT target key.
    # The actual DeviceContext <-> binary binding still belongs to the
    # trusted native loader, not to this pure selector.
    var target_status = capabilities.validate()
    if target_status != TensorStatus.OK:
        return TensorResult[SpecializationDecision](target_status)
    if capabilities.target_kind.code != TargetKind.NVIDIA_GPU.code:
        return TensorResult[SpecializationDecision](TensorStatus.DEVICE_MISMATCH)
    if capabilities.architecture_id != 89 or capabilities.warp_size != 32:
        return TensorResult[SpecializationDecision](TensorStatus.DEVICE_MISMATCH)
    return choose_sm89_gemm(key)


def choose_sm89_gemm_with_artifact(
    key: DispatchKey, tf32_artifact_bound: Bool,
) -> TensorResult[SpecializationDecision]:
    # A boolean cannot prove that an executable artifact implements TF32.
    # Until a separately validated MMA ABI exists, fall back to strict SIMT.
    return choose_sm89_gemm(key)


# Binding metadata is established by the offline artifact loader, not inferred
# from the caller's chosen operation. This validates a DECLARED contract; it
# cannot authenticate machine code that was loaded from an untrusted file.
@fieldwise_init
struct BinaryAOTArtifactIdentity(Copyable):
    var operation: Int
    var variant: Int
    var target_arch: Int
    var abi_version: Int
    var threads_per_block: Int

    # Arithmetic ops 0..3 plus extrema 4=maximum / 5=minimum share the
    # exact flat out/a/b/count ABI. The operation-to-native-symbol binding
    # must still be supplied by a trusted offline artifact loader.
    comptime FP32_BINARY_ABI_VERSION = 1

    @staticmethod
    def checked(
        operation: Int, variant: Int, target_arch: Int,
        abi_version: Int, threads_per_block: Int,
    ) -> TensorResult[Self]:
        if operation < 0 or operation > 5:
            return TensorResult[Self](TensorStatus.INVALID_LAYOUT)
        # ABI v1 has exactly one native flat binary specialization (0).
        # Accepting arbitrary caller-supplied variants would let an unrelated
        # precompiled GPU function claim the same declared ABI. Additional
        # variants require an explicit reviewed native ABI/catalog extension.
        if variant != 0:
            return TensorResult[Self](TensorStatus.UNSUPPORTED)
        if target_arch <= 0 or abi_version != Self.FP32_BINARY_ABI_VERSION:
            return TensorResult[Self](TensorStatus.UNSUPPORTED)
        # Version 1 is the exact flat (out, left, right, count) kernel ABI
        # used by every checked launch adapter in this release. Accepting
        # arbitrary launch geometry here would produce a misleading proof;
        # future block geometries require a separately versioned ABI.
        if threads_per_block != 256:
            return TensorResult[Self](TensorStatus.UNSUPPORTED)
        return TensorResult[Self](Self(operation, variant, target_arch, abi_version, threads_per_block))

    def validate_for(
        self, operation: Int, variant: Int, target_arch: Int
    ) -> TensorStatus:
        var verified = Self.checked(
            self.operation, self.variant, self.target_arch,
            self.abi_version, self.threads_per_block,
        )
        if not verified.is_ok():
            return verified.status
        if operation < 0 or operation > 5:
            return TensorStatus.UNSUPPORTED
        if self.operation != operation or self.variant != variant:
            return TensorStatus.DEVICE_MISMATCH
        if self.target_arch != target_arch:
            return TensorStatus.DEVICE_MISMATCH
        return TensorStatus.OK


@fieldwise_init
struct GemmAOTArtifactIdentity(Copyable):
    # AOT kernel contract (out_buf, a_buf, b_buf, m, k, n).
    # 0=STRICT_FP32_SIMT ABI v1; 1=TF32 MMA tile ABI v2.
    # TF32 is never enabled without an offline-verified release binding.
    # The build/packaging system, NOT caller-controlled metadata, must bind
    # this identity to a trusted native DeviceExternalFunction artifact.
    var flavor: Int
    var variant: Int
    var target_arch: Int
    var abi_version: Int
    var threads_per_block: Int

    comptime GEMM_FP32_ABI_VERSION = 1
    comptime GEMM_TF32_ABI_VERSION = 2
    comptime STRICT_SIMT = 0
    comptime TF32_MMA = 1

    @staticmethod
    def checked(
        flavor: Int, variant: Int, target_arch: Int,
        abi_version: Int, threads_per_block: Int,
    ) -> TensorResult[Self]:
        if threads_per_block != 256:
            return TensorResult[Self](TensorStatus.UNSUPPORTED)
        if flavor == Self.STRICT_SIMT:
            if abi_version != Self.GEMM_FP32_ABI_VERSION:
                return TensorResult[Self](TensorStatus.UNSUPPORTED)
            # Variant 14 is the external flat ABI (one output/lane). The
            # intrinsic SIMT variants 10/11 use separate AOT kernel symbols.
            if variant != 14:
                return TensorResult[Self](TensorStatus.UNSUPPORTED)
            if target_arch < 1:
                return TensorResult[Self](TensorStatus.DEVICE_MISMATCH)
        elif flavor == Self.TF32_MMA:
            # This is a distinct tile ABI: one 16x16 output tile per CTA,
            # 256 threads, shared-matrix K tile 8, FP32 storage/output.
            # The artifact must be pinned to the exact NVIDIA architecture.
            if (abi_version != Self.GEMM_TF32_ABI_VERSION or variant != 15
                or target_arch < 80 or target_arch >= 200):
                return TensorResult[Self](TensorStatus.UNSUPPORTED)
            # A generated, signed-and-hashed offline release catalog is the
            # only source of enabled TF32 variants. The shipped source-only
            # catalog returns UNSUPPORTED; no JIT or fake Tensor-Core claim.
            var trusted = checked_release_gemm_binding(
                flavor, variant, target_arch, abi_version, threads_per_block,
            )
            if trusted != TensorStatus.OK:
                return TensorResult[Self](trusted)
        else:
            return TensorResult[Self](TensorStatus.UNSUPPORTED)
        return TensorResult[Self](
            Self(flavor, variant, target_arch, abi_version, threads_per_block)
        )

    def validate_for(
        self, selected_variant: Int, target_arch: Int, math_mode: Int
    ) -> TensorStatus:
        var identity = Self.checked(
            self.flavor, self.variant, self.target_arch,
            self.abi_version, self.threads_per_block,
        )
        if not identity.is_ok():
            return identity.status
        if self.variant != selected_variant or self.target_arch != target_arch:
            return TensorStatus.DEVICE_MISMATCH
        # STRICT_FP32=0, FMA_FP32=1, TF32_ALLOWED=2, FASTEST=3.
        if math_mode < 0 or math_mode > 3:
            return TensorStatus.MATH_MODE_UNSUPPORTED
        if self.flavor == Self.TF32_MMA and math_mode < 2:
            return TensorStatus.MATH_MODE_UNSUPPORTED
        return TensorStatus.OK


@fieldwise_init
struct UnaryAOTArtifactIdentity(Copyable):
    # Offline binding for one native GPU unary opcode, flat FP32 storage.
    # Contract: (out_buffer, in_buffer, element_count), 256 threads per block.
    var operation: Int
    var variant: Int
    var target_arch: Int
    var abi_version: Int
    var threads_per_block: Int

    comptime FP32_UNARY_ABI_VERSION = 1

    @staticmethod
    def checked(
        operation: Int, variant: Int, target_arch: Int,
        abi_version: Int, threads_per_block: Int,
    ) -> TensorResult[Self]:
        if operation < 0 or operation > 8 or variant != 20 + operation:
            return TensorResult[Self](TensorStatus.UNSUPPORTED)
        if target_arch <= 0 or abi_version != Self.FP32_UNARY_ABI_VERSION:
            return TensorResult[Self](TensorStatus.UNSUPPORTED)
        if threads_per_block != 256:
            return TensorResult[Self](TensorStatus.UNSUPPORTED)
        return TensorResult[Self](
            Self(operation, variant, target_arch, abi_version, threads_per_block)
        )

    def validate_for(
        self, operation: Int, variant: Int, target_arch: Int
    ) -> TensorStatus:
        var checked = Self.checked(
            self.operation, self.variant, self.target_arch,
            self.abi_version, self.threads_per_block,
        )
        if not checked.is_ok():
            return checked.status
        if self.operation != operation or self.variant != variant:
            return TensorStatus.DEVICE_MISMATCH
        if self.target_arch != target_arch:
            return TensorStatus.DEVICE_MISMATCH
        return TensorStatus.OK


# Exact offline AOT ABI for an FP32 row reduction, NOT a general Tensor dtype.
# The executable must accept (out, input, rows, cols) and reduce input in
# contiguous row-major order. Empty reduction axes are legal only for
# SUM and PROD; MIN, MAX and MEAN require cols > 0 when rows > 0.
@fieldwise_init
struct ReduceRowsAOTArtifactIdentity(Copyable):
    var variant: Int
    var target_arch: Int
    var abi_version: Int
    var threads_per_block: Int

    comptime FP32_REDUCE_ROWS_ABI_VERSION = 1
    comptime SUM_ROWS = 30
    comptime PROD_ROWS = 31
    comptime MIN_ROWS = 32
    comptime MAX_ROWS = 33
    comptime MEAN_ROWS = 34

    @staticmethod
    def checked(
        variant: Int, target_arch: Int, abi_version: Int,
        threads_per_block: Int,
    ) -> TensorResult[Self]:
        # Each operation has a distinct prelinked native symbol. The
        # argument ABI is identical, but a SUM symbol must never be treated
        # as a MIN/MEAN kernel merely because their shapes match.
        if variant < Self.SUM_ROWS or variant > Self.MEAN_ROWS:
            return TensorResult[Self](TensorStatus.UNSUPPORTED)
        if target_arch <= 0 or abi_version != Self.FP32_REDUCE_ROWS_ABI_VERSION:
            return TensorResult[Self](TensorStatus.UNSUPPORTED)
        if threads_per_block != 256:
            return TensorResult[Self](TensorStatus.UNSUPPORTED)
        return TensorResult[Self](
            Self(variant, target_arch, abi_version, threads_per_block)
        )

    def validate_for_op(
        self, operation: Int, selected_variant: Int, target_arch: Int,
    ) -> TensorStatus:
        if operation < 0 or operation > 4:
            return TensorStatus.UNSUPPORTED
        if self.variant != Self.SUM_ROWS + operation:
            return TensorStatus.DEVICE_MISMATCH
        return self.validate_for(selected_variant, target_arch)

    def validate_for(self, selected_variant: Int, target_arch: Int) -> TensorStatus:
        var proof = Self.checked(
            self.variant, self.target_arch, self.abi_version,
            self.threads_per_block,
        )
        if not proof.is_ok():
            return proof.status
        if self.variant != selected_variant or self.target_arch != target_arch:
            return TensorStatus.DEVICE_MISMATCH
        return TensorStatus.OK

# AOT-only fixed FP32 scan ABI. This identity DECLARES the native symbol ABI;
# it does not authenticate the DeviceExternalFunction supplied by a caller.
# Only a trusted offline loader may associate an actual machine-code artifact.
@fieldwise_init
struct ScanRowsAOTArtifactIdentity(Copyable):
    var variant: Int
    var target_arch: Int
    var abi_version: Int
    var threads_per_block: Int

    comptime FP32_SCAN_ROWS_ABI_VERSION = 1
    comptime CUMSUM_ROWS = 40
    comptime CUMPROD_ROWS = 41
    comptime CUMMIN_ROWS = 42
    comptime CUMMAX_ROWS = 43

    @staticmethod
    def checked(
        variant: Int, target_arch: Int, abi_version: Int,
        threads_per_block: Int,
    ) -> TensorResult[Self]:
        if variant < Self.CUMSUM_ROWS or variant > Self.CUMMAX_ROWS:
            return TensorResult[Self](TensorStatus.UNSUPPORTED)
        if target_arch <= 0 or abi_version != Self.FP32_SCAN_ROWS_ABI_VERSION:
            return TensorResult[Self](TensorStatus.UNSUPPORTED)
        if threads_per_block != 256:
            return TensorResult[Self](TensorStatus.UNSUPPORTED)
        return TensorResult[Self](
            Self(variant, target_arch, abi_version, threads_per_block)
        )

    def validate_for_op(
        self, operation: Int, selected_variant: Int, target_arch: Int,
    ) -> TensorStatus:
        if operation < 0 or operation > 3:
            return TensorStatus.UNSUPPORTED
        var validated = Self.checked(
            self.variant, self.target_arch, self.abi_version,
            self.threads_per_block,
        )
        if not validated.is_ok():
            return validated.status
        if self.variant != Self.CUMSUM_ROWS + operation:
            return TensorStatus.DEVICE_MISMATCH
        if self.variant != selected_variant or self.target_arch != target_arch:
            return TensorStatus.DEVICE_MISMATCH
        return TensorStatus.OK


# Offline-produced FINITE source-level catalog. No runtime registration,
# registry lookup, mutable global state, JIT or speculative "fallback" to an
# arbitrary unlisted variant. The catalog is owned by its execution context
# or plan builder, and callers consume only a checked copied snapshot.
# This deliberately does NOT authenticate a native symbol or binary image:
# that separate requirement is the U07/U08 AOT loader/release boundary.
@fieldwise_init
struct FiniteAOTVariant(Copyable):
    var key: DispatchKey
    var decision: SpecializationDecision
    var abi_version: Int

    def checked_snapshot(self) -> TensorResult[Self]:
        var key_proof = self.key.checked_snapshot()
        if not key_proof.is_ok():
            return TensorResult[Self](key_proof.status)
        var decision_proof = self.decision.checked_snapshot()
        if not decision_proof.is_ok():
            return TensorResult[Self](decision_proof.status)
        if self.abi_version <= 0 or self.abi_version > 65535:
            return TensorResult[Self](TensorStatus.UNSUPPORTED)
        # A declared GPU strategy must not be bound to an architecture-zero
        # key; the actual vendor/hardware identity remains a loader proof.
        return TensorResult[Self](Self(
            key_proof.payload.value(), decision_proof.payload.value(),
            self.abi_version,
        ))


struct FiniteAOTStrategyCatalog[capacity: Int](Copyable):
    var _entries: Array[FiniteAOTVariant, Self.capacity]
    var _count: Int

    def __init__(
        out self, entries: Array[FiniteAOTVariant, Self.capacity], count: Int,
    ):
        self._entries = entries.copy()
        self._count = count

    def __init__(out self, *, copy: Self):
        self._entries = copy._entries.copy()
        self._count = copy._count

    @staticmethod
    def checked(
        entries: Array[FiniteAOTVariant, Self.capacity], count: Int,
    ) -> TensorResult[Self]:
        if Self.capacity < 1 or Self.capacity > 64:
            return TensorResult[Self](TensorStatus.UNSUPPORTED)
        if count <= 0 or count > Self.capacity:
            return TensorResult[Self](TensorStatus.INVALID_LAYOUT)
        # Validate every selected variant before accepting any fast-path key;
        # reject duplicate keys rather than trusting iteration order.
        for i in range(count):
            var candidate = entries[i].checked_snapshot()
            if not candidate.is_ok():
                return TensorResult[Self](candidate.status)
            for j in range(i):
                var previous = entries[j].checked_snapshot()
                if not previous.is_ok():
                    return TensorResult[Self](previous.status)
                if dispatch_key_matches(candidate.payload.value().key,
                                        previous.payload.value().key):
                    return TensorResult[Self](TensorStatus.INVALID_LAYOUT)
        return TensorResult[Self](Self(entries, count))

    def checked_snapshot(self) -> TensorResult[Self]:
        return Self.checked(self._entries.copy(), self._count)

    def select_variant_checked(self, request: DispatchKey) -> TensorResult[FiniteAOTVariant]:
        # Return the declared *entry*, not just an unbound variant number.
        # The caller must compare its ABI version against the native binding.
        var key_proof = request.checked_snapshot()
        if not key_proof.is_ok():
            return TensorResult[FiniteAOTVariant](key_proof.status)
        var catalog = self.checked_snapshot()
        if not catalog.is_ok():
            return TensorResult[FiniteAOTVariant](catalog.status)
        var safe = catalog.payload.value()
        for i in range(safe._count):
            var candidate = safe._entries[i]
            if dispatch_key_matches(candidate.key, key_proof.payload.value()):
                return candidate.checked_snapshot()
        # A miss is a hard error. No JIT and no arbitrary runtime variant.
        return TensorResult[FiniteAOTVariant](TensorStatus.UNSUPPORTED)

    def select_checked(self, request: DispatchKey) -> TensorResult[SpecializationDecision]:
        var variant = self.select_variant_checked(request)
        if not variant.is_ok():
            return TensorResult[SpecializationDecision](variant.status)
        return variant.payload.value().decision.checked_snapshot()

    def contains_key(self, request: DispatchKey) -> Bool:
        return self.select_variant_checked(request).is_ok()
