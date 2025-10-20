# MIT License
# SPDX-License-Identifier: MIT
# Project:      momijo.tensor
# Module:       tests.tensor
# File:         examples/Tensor/demo_reshape.mojo
#
# Description:
#   Demo & quick tests for reshape / reshape_infer / reshape_like
#   and resize_like_with_pad (non-reshape helper). 

from momijo.tensor import tensor

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
fn banner(title: String) -> None:
    print("\n=== " + title + " ===")

fn assert_eq_int(a: Int, b: Int, msg: String) -> None:
    if a != b:
        print("ASSERT FAIL: " + msg + " | got=" + String(a) + " expected=" + String(b))

# Lightweight previews (overloads)
fn preview(x: tensor.Tensor[Int]) -> None:
    print(x.__str__())

fn preview(x: tensor.Tensor[Float64]) -> None:
    print(x.__str__())

fn preview(x: tensor.Tensor[Float32]) -> None:
    print(x.__str__())

# -----------------------------------------------------------------------------
# 1) Basic contiguous reshape: view-like header change (no data copy)
# -----------------------------------------------------------------------------
fn test_contiguous_view() -> None:
    banner("1) Contiguous -> reshape view path")
    var x = tensor.arange(0, 12, 1)            # shape [12]
    x = tensor.reshape(x, [3, 4])              # expected to be a view-like header change
    assert_eq_int(x.numel(), 12, "numel must stay 12")
    print("shape after [3,4]: " + x.shape().__str__())
    preview(x)

    # back to [2, 6]
    var y = tensor.reshape(x, [2, 6])
    assert_eq_int(y.numel(), 12, "numel must stay 12")
    print("shape after [2,6]: " + y.shape().__str__())
    preview(y)

# -----------------------------------------------------------------------------
# 2) reshape_infer with a single -1 (exactly one -1 allowed)
# -----------------------------------------------------------------------------
fn test_infer_minus_one() -> None:
    banner("2) reshape_infer: single -1 inference")
    var r = tensor.arange(1, 25, 1)                 # [24]
    var a = tensor.reshape_infer(r, [2, -1, 3])     # => [2, 4, 3]
    print("inferred shape [2,-1,3] -> " + a.shape().__str__())
    assert_eq_int(a.numel(), 24, "numel must stay 24")
    preview(a)

    # invalid: multiple -1 → should be a no-op (copy of input)
    var b = tensor.reshape_infer(r, [-1, -1])
    print("multiple -1 -> expected no-op copy (still [24]) | got shape: " + b.shape().__str__())
    preview(b)

# -----------------------------------------------------------------------------
# 3) reshape_like: match the shape of another tensor with equal numel
# -----------------------------------------------------------------------------
fn test_reshape_like() -> None:
    banner("3) reshape_like")
    var src = tensor.arange(0, 24, 1).reshape([4, 6])
    var r   = tensor.arange(0, 24, 1).reshape([2, 3, 4])

    var v = tensor.reshape_like(src, r)             # -> [2,3,4]
    print("reshape_like to r [2,3,4]: " + v.shape().__str__())
    assert_eq_int(v.numel(), 24, "numel must stay 24")
    preview(v)

# -----------------------------------------------------------------------------
# 4) No-op on mismatch: reshape must not change total element count
# -----------------------------------------------------------------------------
fn test_noop_on_mismatch() -> None:
    banner("4) reshape: mismatch should no-op (return copy)")
    var x = tensor.arange(0, 12, 1)     # [12]
    var bad = tensor.reshape(x, [5, 5]) # invalid (25 != 12): should be no-op
    print("request [5,5] on [12] -> shape: " + bad.shape().__str__())
    assert_eq_int(bad.numel(), 12, "reshape must not change numel")
    preview(bad)

# -----------------------------------------------------------------------------
# 5) resize_like_with_pad (NOT reshape): grow/shrink with pad/truncate
# -----------------------------------------------------------------------------
fn test_resize_like_with_pad() -> None:
    banner("5) resize_like_with_pad (NOT reshape)")
    var x = tensor.arange(1, 6, 1)                            # data: [1,2,3,4,5] shape [5]
    var target_big = tensor.arange(0, 12, 1).reshape([3, 4])  # 12 elems

    # Grow: copy [1..5] and pad to reach 12
    var y = tensor.resize_like_with_pad(x, target_big)        # -> shape [3,4], elems 12
    print("resize_like_with_pad to [3,4]: " + y.shape().__str__())
    preview(y)

    # Shrink: truncate to fit
    var target_small = tensor.arange(0, 6, 1).reshape([2, 3]) # 6 elems
    var z = tensor.resize_like_with_pad(x, target_small)      # -> first up-to-6 elems of x
    print("resize_like_with_pad to [2,3]: " + z.shape().__str__())
    preview(z)

# -----------------------------------------------------------------------------
# 6) Multi-d reshapes: 1D->3D and 3D->2D
# -----------------------------------------------------------------------------
fn test_multi_dim_roundtrips() -> None:
    banner("6) Multi-d reshapes")
    var a  = tensor.arange(0, 24, 1)           # [24]
    var a3 = tensor.reshape(a, [2, 3, 4])      # [2,3,4]
    print("1D->3D: " + a3.shape().__str__())
    preview(a3)

    var a2 = tensor.reshape(a3, [6, 4])        # [6,4]
    print("3D->2D: " + a2.shape().__str__())
    preview(a2)

# -----------------------------------------------------------------------------
# 7) reshape_infer edge cases: zero-sized dims (if supported)
# -----------------------------------------------------------------------------
fn test_zero_dim_cases() -> None:
    banner("7) Zero-sized dims (if supported)")
    var z  = tensor.arange(0, 0, 1)                # []
    var z2 = tensor.reshape_infer(z, [0, -1])      # -> [0, ?] with numel=0
    print("zero-sized reshape_infer [0,-1] -> " + z2.shape().__str__())
    preview(z2)

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
fn main() -> None:
    test_contiguous_view()
    test_infer_minus_one()
    test_reshape_like()
    test_noop_on_mismatch()
    test_resize_like_with_pad()
    test_multi_dim_roundtrips()
    test_zero_dim_cases()
