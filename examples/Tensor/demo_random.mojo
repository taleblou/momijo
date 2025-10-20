# MIT License
# SPDX-License-Identifier: MIT
# Project:      momijo.tensor
# Module:       tests.tensor
# File:         tests/tensor/demo_random.mojo
#
# Description:
#   Demo for random generators:
#   - rand / randn
#   - normal(mean, std, shape)
#   - uniform_ (in-place fill)
#   - bernoulli, multinomial
#   - randperm
#
# Author(s):    Morteza Taleblou & Mitra Daneshmand
# Website:      https://taleblou.ir/
# Repository:   https://github.com/taleblou/momijo

from momijo.tensor import tensor

# -----------------------------------------------------------------------------
# Small banner printer
# -----------------------------------------------------------------------------
fn banner(title: String) -> None:
    print("\n=== " + title + " ===")

# -----------------------------------------------------------------------------
# 10) Random
# -----------------------------------------------------------------------------
fn demo_random() -> None:
    banner("10) RANDOM")

    # Basic random tensors
    var r_uniform = tensor.rand([2, 2])      # Float64 uniform in [0, 1)
    var r_normal_i = tensor.randn_int([2, 2])  # Discrete normal-ish (Int), per your API
    print("rand / randn:\n" + r_uniform.__str__() + "\n" + r_normal_i.__str__())

    # Normal(mean, std, shape)
    var r_normal = tensor.normal(3.0, 0.5, [2])
    print("normal (mean=3, std=0.5): " + r_normal.__str__())

    # In-place uniform fill (API returns the tensor; keep assignment explicit)
    var t = tensor.empty([3])
    t = t.uniform_(-1.0, 1.0)
    # print("uniform demo:\n" + t.__str__())

    # Bernoulli and multinomial
    var p = tensor.from_list_float64([0.1, 0.7, 0.2])
    var bern = tensor.full([5], 0.3).bernoulli()
    print("bernoulli(0.3): " + bern.__str__())
    print("multinomial from p (10 draws): " + p.multinomial(10, True).__str__())

    # Random permutation
    print("randperm(10): " + tensor.randperm_int(10).__str__())

# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------
fn main() -> None:
    demo_random()
