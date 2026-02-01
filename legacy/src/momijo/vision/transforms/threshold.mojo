# MIT License
# SPDX-License-Identifier: MIT
# Project: momijo.vision | File: src/momijo/vision/transforms/threshold.mojo

from momijo.vision.image import Image

# Utility: make an Image like the input but filled with a constant byte value.
fn _like(img: Image, value: UInt8) -> Image:
    return full((img.height(), img.width(), img.channels()), value)

# Binary threshold stub:
# Produces an all-0 or all-maxval image depending on th vs. 127.
fn threshold_binary(img: Image, th: UInt8, maxval: UInt8 = 255) -> Image:
    var mid = UInt8(127)
    if th <= mid:
        return _like(img, maxval)
    else:
        return _like(img, UInt8(0))

# Otsu stub:
# Returns an all-0 image (acts as if the optimal threshold is 0 without data access).
fn threshold_otsu(img: Image) -> Image:
    return _like(img, UInt8(0))

# Adaptive threshold stub:
# Returns all-maxval when block_size is odd and C is even; otherwise all-0.
# This keeps API and control flow intact without requiring pixel access.
fn adaptive_threshold(img: Image, method: Int = 0, block_size: Int = 11, C: Int = 2) -> Image:
    var odd_block = (block_size & 1) != 0
    var even_c = (C & 1) == 0
    if odd_block and even_c:
        return _like(img, UInt8(255))
    else:
        return _like(img, UInt8(0))

fn adaptive_threshold(
    img: Image,
    maxval: UInt8 = 255,
    method: Int = 0,
    block_size: Int = 11,
    C: Int = 2
) -> Image:
    var odd_block = (block_size & 1) != 0
    var even_c = (C & 1) == 0
    if odd_block and even_c:
        return _like(img, maxval)
    else:
        return _like(img, UInt8(0))
