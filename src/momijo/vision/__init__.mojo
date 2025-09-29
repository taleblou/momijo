# MIT License
# SPDX-License-Identifier: MIT
# Project: momijo.vision | File: src/momijo/vision/__init__.mojo

from momijo.vision.image import Image
from momijo.vision.io.image_read import read_image
from momijo.vision.io.image_write import write_image
from momijo.vision.transforms.array import zeros
import time

from momijo.vision.transforms.bitwise import bitwise_and
from momijo.vision.transforms.bitwise import bitwise_or
from momijo.vision.transforms.bitwise import bitwise_not

from momijo.vision.transforms.features import to_int_triples


from momijo.vision.transforms.segmentation import connected_components
from momijo.vision.io.image import write_image_any,read_image_any
from momijo.vision.transforms.color import (
    bgr_to_gray,
    rgb_to_bgr,
    bgr_to_rgb,
    split3,
    merge3,
    equalize_hist,
    histogram,
    plot_hist_u8,
    clahe_color_bgr,
)

from momijo.vision.transforms.geom import (
    resize,
    rotate,
    rotate90,
    translate, 
    flip, affine, perspective,deg2rad,
    crop, BorderSpec,copy_make_border, 
    BORDER_CONSTANT_BS, BORDER_REPLICATE_BS, BORDER_REFLECT_BS,
    BORDER_CONSTANT, BORDER_REPLICATE, BORDER_REFLECT,
    INTER_LINEAR, INTER_AREA,
    MORPH_OPEN, MORPH_CLOSE, MORPH_GRADIENT,
    ADAPTIVE_GAUSSIAN,
    FONT_PLAIN, FONT_SIMPLEX,
   
)
 
from momijo.vision.transforms.filter import (
    gaussian_blur,
    median_blur,
    bilateral_filter,
    sobel,
    laplacian,
    canny, 
    magnitude_u8,
    ones_u8,
    bilateral_blur,
)

from momijo.vision.transforms.threshold import (
    threshold_binary,
    threshold_otsu,
    adaptive_threshold,
)

from momijo.vision.transforms.morph import (
    erode,
    dilate,
    morphology,
)

from momijo.vision.transforms.draw import (
    line,
    rectangle,
    circle,
    Int32Mat,
    fill_poly,
    arrowed_line,
    draw_circles,
    draw_contours,
    draw_matches,
    put_text,
    label_to_color,
    label_to_color_image,
    draw_lines_p,
    bgr_u8,
)

from momijo.vision.transforms.contour import (
    find_contours,
    len_contours,
    contour_area,
    bounding_rect,
    get_contour,
    arc_length,
)

from momijo.vision.transforms.hough import (
    hough_lines_p,
    hough_circles, 
)
 
from momijo.vision.transforms.array import (
    add_weighted,
    abs_u8,
    fill,fill_scalar,fill3,
    full,
    full_like,
    copy_to,
    top_k_matches,len_keypoints,bf_match_hamming,
    valid_descriptors,Keypoint, Descriptor,orb_detect_and_compute
)
 

# Display an image (stub: prints info instead of opening a real window).
fn imshow(name: String, img: Image) -> None:
    print("imshow -> window:", name)
    # print("  shape: ", img.h, "x", img.w, " channels:", img.c)

# Wait for a key press (stub: sleeps for ms, returns -1).
fn wait_key(ms: Int = 0) -> Int:
    if ms > 0:
        # Convert ms to seconds
        var secs = Float64(ms) / 1000.0
        time.sleep(secs)
    return -1

# Capability check for real GUI windows.
fn supports_windows() -> Bool:
    return False