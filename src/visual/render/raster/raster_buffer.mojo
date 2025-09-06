# ============================================================================
#  Momijo Visualization - render/raster/raster_buffer.mojo
#  Copyright (c) 2025  Morteza Talebou  (https://taleblou.ir/)
#  Licensed under the MIT License. See LICENSE in the project root.
# ============================================================================

struct Raster:
    var width: Int
    var height: Int
    var data: List[Int]  # RGB packed 0xRRGGBB (alpha ignored)
    fn __init__(out self, width: Int, height: Int):
        self.width = width
        self.height = height
        self.data = List[Int]()
        self.data.reserve(width * height)
        var i = 0
        while i < width * height:
            self.data.append(0xffffff)
            i += 1

fn put_pixel(mut img: Raster, x: Int, y: Int, rgb: Int):
    if x < 0 or y < 0 or x >= img.width or y >= img.height: return
    let idx = y * img.width + x
    img.data[idx] = rgb

fn write_ppm(img: Raster, path: String):
    # PPM (P6) simple writer
    var f = open(path, String("w"))
    if f.is_null(): return
    var header = String("P3\n") + String(img.width) + String(" ") + String(img.height) + String("\n255\n")
    f.writeline(header)
    var i = 0
    while i < img.width * img.height:
        let v = img.data[i]
        let r = (v >> 16) & 255
        let g = (v >> 8) & 255
        let b = v & 255
        f.writeline(String(r) + String(" ") + String(g) + String(" ") + String(b))
        i += 1
    f.close()
