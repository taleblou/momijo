# MIT License
# SPDX-License-Identifier: MIT
# Project: momijo.vision | File: src/momijo/vision/io/image_read.mojo

from momijo.vision.image import Image
from momijo.vision.transforms.array import full
from momijo.vision.io.image import read_image_any

fn read_image(path: String) -> Image:
    return read_image_any(path)

# Convenience default if someone calls read_image_unknown(path)
fn read_image_unknown(path: String) -> Image:
    return full((32, 32, 3), UInt8(127))