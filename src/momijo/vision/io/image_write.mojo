# MIT License
# SPDX-License-Identifier: MIT
# File: src/momijo/vision/io/image_write.mojo

from momijo.vision.image import Image
from momijo.vision.io.png import write_png

fn write_image(path: String, img: Image, interlace: Int = 0, compress: Int = 2, filter_mode: Int = -1, palette_mode: Int = 0, max_colors: Int = 256, bit_depth_out: Int = 8) raises -> Bool:
    # var (r,g,b) = img.get_rgb_u8(0,0)
    # print("write_image px=", r, g, b)
    return write_png(path, img.copy(), interlace, compress, filter_mode, palette_mode, max_colors, bit_depth_out)

# # --- image_write.mojo ---

# from momijo.vision.image import Image
# from momijo.vision.io.encode_png import png_from_hwc_u8  # مطمئن شو نام و ماژول درست است
# from momijo.vision.io.file_io import write_all_bytes
# from momijo.vision.io.png import _to_hwc_u8              # همین بالایی که پچ شد

# # API اصلی برای کاربر
# fn write_image(
#     path: String,
#     img: Image,
#     interlace: Int = 0,
#     compress: Int = 2,
#     filter_mode: Int = -1,
#     palette_mode: Int = 0,
#     max_colors: Int = 256,
#     bit_depth_out: Int = 8
# ) raises -> Bool:
#     var dims = _to_hwc_u8(img.copy())
#     var H = dims[0]; var W = dims[1]; var C = dims[2]; var buf = dims[3].copy()
#     if H <= 0 or W <= 0 or C < 1 or C > 4:
#         print("[write_image] FAIL: invalid dims H=", H, " W=", W, " C=", C)
#         return False

#     # 2) انکود PNG
#     var ok_bytes = png_from_hwc_u8(
#         W, H, C,
#         buf.copy(),
#         interlace,
#         compress,
#         filter_mode,
#         palette_mode,
#         max_colors,
#         bit_depth_out
#     )
#     if not ok_bytes[0]:
#         print("[write_image] FAIL: png_from_hwc_u8 returned False")
#         return False

#     var png_bytes = ok_bytes[1].copy()
#     if len(png_bytes) < 8:
#         print("[write_image] FAIL: encoded bytes too small: ", len(png_bytes))
#         return False

#     # 3) نوشتن روی دیسک (باینری)
#     var ok = write_all_bytes(path, png_bytes.copy())
#     if not ok:
#         print("[write_image] FAIL: write_all_bytes returned False for path=", path)
#         return False

#     print("[write_image] OK: wrote ", len(png_bytes), " bytes to ", path)
#     return True
