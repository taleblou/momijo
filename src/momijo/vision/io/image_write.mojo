# MIT License
# SPDX-License-Identifier: MIT
# Project: momijo.vision | File: src/momijo/vision/io/image_write.mojo
 
from momijo.vision.image import Image
from momijo.vision.io.image import write_image_any

fn write_image(path: String, img: Image) -> Bool:
    return write_image_any(path, img)