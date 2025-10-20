# MIT License
# SPDX-License-Identifier: MIT
# Project:      momijo.tensor
# Module:       tests.tensor
# File:         examples/Tensor/demo_build_from_literal_lists.mojo
#
# Description:
#   EXACTLY build tensors from the user's literal lists (1D/2D/3D/4D; float & int)
#   without changing formatting or padding. This assumes Tensor constructors for
#   nested lists are implemented for 1D/2D/3D/4D. If compilation fails with
#   “no matching function in initialization”, add those nested-list __init__
#   overloads in your Tensor implementation.
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
# Main demo: build tensors directly from literal lists
# -----------------------------------------------------------------------------
fn main() -> None:
    banner("BUILD FROM LITERAL LISTS (1D/2D/3D/4D; FLOAT & INT)")

    # 1) 1D (float)
    var mixed_list = tensor.Tensor([1.1, 2.5, 0.3, 4.7, 5.6, 6.1, 10.0])

    # 2) 1D (int)
    var int_list = tensor.Tensor([1, 2, 3, 4, 5, 10, 100, 1000])

    # 3) 2D (float)
    var two_dimensional_mixed = tensor.Tensor([
        [1.1, 2.5, 0.3],
        [4.7, 5.6, 6.1],
        [10.0, 1.1, 2.5]
    ])

    # 4) 2D (int)
    var two_dimensional_int = tensor.Tensor([
        [1, 2, 3],
        [4, 5, 10],
        [100, 1000, 1]
    ])

    # 5) 3D (float)
    var three_dimensional_mixed = tensor.Tensor([
        [
            [1.1, 2.5, 0.3],
            [4.7, 5.6, 6.1]
        ],
        [
            [10.0, 1.1, 2.5],
            [4.7, 0.3, 5.6]
        ],
        [
            [6.1, 10.0, 1.1],
            [2.5, 4.7, 0.3]
        ]
    ])

    # 6) 3D (int)
    var three_dimensional_int = tensor.Tensor([
        [
            [1, 2, 3],
            [4, 5, 10]
        ],
        [
            [100, 1000, 1],
            [2, 3, 4]
        ],
        [
            [5, 10, 100],
            [1000, 1, 2]
        ]
    ])

    # 7) 4D (float)
    var four_dimensional_mixed = tensor.Tensor([
        [
            [
                [1.1, 2.5],
                [0.3, 4.7]
            ],
            [
                [5.6, 6.1],
                [10.0, 1.1]
            ]
        ],
        [
            [
                [2.5, 0.3],
                [4.7, 5.6]
            ],
            [
                [6.1, 10.0],
                [1.1, 2.5]
            ]
        ]
    ])

    # 8) 4D (int)
    var four_dimensional_int = tensor.Tensor([
        [
            [
                [1, 2],
                [3, 4]
            ],
            [
                [5, 10],
                [100, 1000]
            ]
        ],
        [
            [
                [1, 2],
                [3, 4]
            ],
            [
                [5, 10],
                [100, 1000]
            ]
        ]
    ])

    # Print shapes (String-only prints)
    print("mixed_list shape: " + mixed_list.shape().__str__())
    print("int_list shape: " + int_list.shape().__str__())
    print("two_dimensional_mixed shape: " + two_dimensional_mixed.shape().__str__())
    print("two_dimensional_int shape: " + two_dimensional_int.shape().__str__())
    print("three_dimensional_mixed shape: " + three_dimensional_mixed.shape().__str__())
    print("three_dimensional_int shape: " + three_dimensional_int.shape().__str__())
    print("four_dimensional_mixed shape: " + four_dimensional_mixed.shape().__str__())
    print("four_dimensional_int shape: " + four_dimensional_int.shape().__str__())

    # Print full tensors (relies on your Tensor.__str__)
    print("mixed_list:\n" + mixed_list.__str__())
    print("int_list:\n" + int_list.__str__())
    print("two_dimensional_mixed:\n" + two_dimensional_mixed.__str__())
    print("two_dimensional_int:\n" + two_dimensional_int.__str__())
    print("three_dimensional_mixed:\n" + three_dimensional_mixed.__str__())
    print("three_dimensional_int:\n" + three_dimensional_int.__str__())
    print("four_dimensional_mixed:\n" + four_dimensional_mixed.__str__())
    print("four_dimensional_int:\n" + four_dimensional_int.__str__())
