# MIT License
# SPDX-License-Identifier: MIT
# Project: momijo.examples
# File: examples/vision_filters_threshold_edges.mojo
# Description: Filters, thresholds, and edge detection using a single `import momijo.vision as vision`.
# Notes: Each step is saved to PPM/PNG/JPG for easy verification.
 
 
import momijo.vision as vision
from pathlib import Path
from os import makedirs
from os import mkdir
from collections.list import List


# ----------------------- Utility Helpers -----------------------
fn ensure_outdir(outdir: String) -> None:
    var p = Path(outdir)
    if not p.exists():
        try:
            makedirs(String(p))
        except _:
            pass


fn _save_all(outdir: String, stem: String, img: vision.Image) -> None: 
    try:
        vision.write_image(outdir + "/" + stem + ".png", img) 
    
    except e:
        print(e) 
    print("Saved: " + outdir + "/" + stem + ".{png}")

fn save(outdir: String, name: String, img: vision.Image) -> String:
    var path = outdir + "/" + name
    try:
        vision.write_image(path, img)
    
    except e:
        print(e) 
    print("Saved: " + path)
    return path

fn show_if(show: Bool, pause_ms: Int, title: String, img: vision.Image) -> None:
    if show and vision.supports_windows():
        vision.imshow(title, img)
        vision.wait_key(pause_ms)

# ----------------------- Demo / Safe Load -----------------------

fn make_demo_image() -> vision.Image:
    # Build a synthetic scene that is good for filters/edges
    var w = 800
    var h = 500
    var img = vision.zeros(h, w, 3, vision.UInt8())
    img = vision.fill(img, [UInt8(28), UInt8(28), UInt8(32)])  # BGR background

    # High-contrast shapes
    img = vision.rectangle(img, 40, 40, 300, 200, (UInt8(255), UInt8(255), UInt8(255)), -1)   # white block
    img = vision.circle(img, 520, 160, 70, [UInt8(0), UInt8(0), UInt8(255)], -1)              # red disk (BGR)
    img = vision.line(img, 60, 260, 740, 260, [UInt8(255), UInt8(255), UInt8(255)], 3)        # bright line

    try:
        # Text edges
        img = vision.put_text(
            img, "Filters & Edges",
            180, 360, vision.FONT_SIMPLEX(), 1.2,
            [UInt8(220), UInt8(220), UInt8(220)], 2, True
        )
    except e:
        print(e) 
    return img.copy()

fn load_image(path: String) -> vision.Image:
    # Ensure parent directory exists; then try to read the image.
    # If reading fails or the image is invalid, generate a demo image, save, and return it.

    var p = Path(path)
 
    if not p.exists():
        # Best-effort: try recursive create, then single-level, ignore errors
        try:
            makedirs(String(p))
        except _:
            try:
                mkdir(String(p))
            except _:
                pass  # ignore

    # 2) If file exists, try to read it
    if p.exists():
        try:
            var img = vision.read_image(path)
            # Sanity check: reject zero-sized images
            if img[1].width() > 0 and img[1].height() > 0:
                return img[1].copy()
            # Otherwise fall through to demo generation
        except _:
            # Fall through to demo generation on any error
            pass

    # 3) Fallback: generate demo image and try to save it
    var demo = make_demo_image()
    try:
        vision.write_image(path, demo)
    except _:
        # Ignore write errors; still return a valid image
        pass

    return demo.copy()

# ----------------------- 5) Filters, Thresholds, Edges -----------------------
# Every intermediate result is saved immediately.

fn filters_threshold_edges(img: vision.Image, outdir: String) -> None:
    # 0) Original (BGR)
    _save_all(outdir, "12_00_original_bgr", img)

    # 1) Grayscale
    var gray = vision.bgr_to_gray(img)
    _save_all(outdir, "12_01_gray", gray)

    # 2) Blurs / Denoising
    #    - Gaussian smooths noise with a Gaussian kernel (odd sizes)
    #    - Median removes impulse noise while preserving edges
    #    - Bilateral preserves edges (uses color + spatial sigma)
    var blur_g = vision.gaussian_blur(gray, 7, 7, 1.5)
    _save_all(outdir, "12_02_blur_gaussian", blur_g)

    var blur_m = vision.median_blur(gray, 5)
    _save_all(outdir, "12_03_blur_median", blur_m)

    var blur_bi = vision.bilateral_blur(gray, 9, 75, 75)
    _save_all(outdir, "12_04_blur_bilateral", blur_bi)

    # 3) Thresholds
    #    - Binary with fixed threshold
    #    - Otsu automatic threshold
    #    - Adaptive (Gaussian window)
    var th_bin = vision.threshold_binary(gray, 120, 255)
    _save_all(outdir, "12_05_thresh_binary", th_bin)

    var th_otsu = vision.threshold_otsu(gray)
    _save_all(outdir, "12_06_thresh_otsu", th_otsu)

    var th_adapt = vision.adaptive_threshold(gray, 255, vision.ADAPTIVE_GAUSSIAN(), 11, 2)
    _save_all(outdir, "12_07_thresh_adaptive", th_adapt)

    # 4) Gradients and Edges
    #    - Sobel X/Y and magnitude
    #    - Laplacian (then absolute to map to u8)
    #    - Canny on the Gaussian-blurred image
    var sobelx = vision.sobel(gray, 1, 0, 3)
    _save_all(outdir, "12_08_sobel_x", sobelx)

    var sobely = vision.sobel(gray, 0, 1, 3)
    _save_all(outdir, "12_09_sobel_y", sobely)

    var sobel_mag = vision.magnitude_u8(sobelx, sobely)
    _save_all(outdir, "12_10_sobel_magnitude", sobel_mag)

    var lap = vision.laplacian(gray, 3)
    var lap_u8 = vision.abs_u8(lap)
    _save_all(outdir, "12_11_laplacian_abs", lap_u8)

    var edges = vision.canny(blur_g, 80, 160)
    _save_all(outdir, "12_12_canny", edges)

# ----------------------- Runnable -----------------------

fn main() -> None:
    var outdir = "outputs_filters_threshold_edges"
    ensure_outdir(outdir)

    # Prepare input (use demo if missing)
    var input_path = outdir + "/input_demo.png"
    var img = load_image(input_path)

    # Run Section 5 and save each change
    filters_threshold_edges(img, outdir)

    # Optional on-screen preview
    show_if(False, 1200, "Filters/Thresholds/Edges", img)
