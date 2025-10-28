# MIT License
# SPDX-License-Identifier: MIT
# Project: momijo.examples
# File: examples/vision_hough.mojo
# Description: Hough line and circle detection using `import momijo.vision as vision`.
# Notes: Each detection step is saved to PPM/PNG/JPG.

 
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

# ----------------------- 11) Hough (Lines & Circles) -----------------------

fn hough_detections(outdir: String) -> None:
    # 1) Hough Line Detection
    var canvas = vision.full(300, 400, 3, UInt8(255))
    canvas = vision.line(canvas, 20, 280, 380, 20, [UInt8(0), UInt8(0), UInt8(0)], 2)
    canvas = vision.line(canvas, 20, 20, 380, 280, (UInt8(0), UInt8(0), UInt8(0)), 2)
    _save_all(outdir, "43_00_lines_input", canvas)

    var edges = vision.canny(vision.bgr_to_gray(canvas), 50, 150)
    _save_all(outdir, "43_01_edges", edges)

    var lines = vision.hough_lines_p(edges, 1.0, vision.deg2rad(1.0), 80, 60, 10)
    var canvas_lines = vision.draw_lines_p(canvas, lines, (UInt8(0), UInt8(0), UInt8(255)), 2)
    _save_all(outdir, "43_02_hough_lines", canvas_lines)

    # 2) Hough Circle Detection
    var canvas2 = vision.full(300, 400, 3, UInt8(255))
    canvas2 = vision.circle(canvas2, 120, 150, 40, [UInt8(0), UInt8(0), UInt8(0)], 2)
    canvas2 = vision.circle(canvas2, 260, 150, 60, [UInt8(0), UInt8(0), UInt8(0)], 2)
    _save_all(outdir, "44_00_circles_input", canvas2)

    var gray2 = vision.median_blur(vision.bgr_to_gray(canvas2), 5)
    _save_all(outdir, "44_01_gray_blurred", gray2)

    var circles = vision.hough_circles(gray2, 1.2, 60.0, 100.0, 30.0, 20, 80)
    var canvas_circles = vision.draw_circles(
        canvas2,
        circles,
        (UInt8(0), UInt8(0), UInt8(255)),
        2,
        (UInt8(0), UInt8(255), UInt8(0)),
        2
    )
    _save_all(outdir, "44_02_hough_circles", canvas_circles)

# ----------------------- Runnable -----------------------

fn main() -> None:
    var outdir = "outputs_hough"
    ensure_outdir(outdir)

    hough_detections(outdir)

    show_if(False, 1200, "Hough Detections", vision.full(100, 100, 3, UInt8(128)))  # dummy show
