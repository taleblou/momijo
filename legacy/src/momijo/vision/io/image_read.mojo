# MIT License
# SPDX-License-Identifier: MIT
# Project: momijo.vision
# File: src/momijo/vision/io/image_read.mojo
# Description: Image reader (PNG RGB/RGBA) supporting Adam7 + dynamic Huffman.

from momijo.vision.image import Image
from momijo.vision.io.png import read_png

fn read_image(path: String) raises -> (Bool, Image):
    return read_png(path)