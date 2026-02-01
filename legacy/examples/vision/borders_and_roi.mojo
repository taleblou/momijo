# Project:      Momijo
# Module:       examples.borders_roi
# File:         borders_roi.mojo
# Path:         src/momijo/examples/borders_roi.mojo
#
# Description:  Demo for adding borders and extracting/displaying a region of interest (ROI).
#
# Author(s):    Morteza Taleblou & Mitra Daneshmand
# Website:      https://taleblou.ir/
# Repository:   https://github.com/taleblou/momijo
#
# License:      MIT License
# SPDX-License-Identifier: MIT
# Copyright:    (c) 2025 Morteza Taleblou & Mitra Daneshmand
#
# Notes:
#   - Shows constant border padding, ROI crop, and ROI visualization with a rectangle.
#   - Uses defensive I/O helpers and generates a demo image if input is missing.

import momijo.vision as vision            # Import vision module (image types, drawing, I/O).
from pathlib import Path                  # Path abstraction for file-system operations.
from os import makedirs                   # Recursive directory creation.
from os import mkdir                      # Single-level directory creation (fallback).
from collections.list import List         # Dynamic list container (used for demo polygon/text args if needed).

# ----------------------- Utility Helpers -----------------------

fn ensure_outdir(outdir: String) -> None: # Ensure the output directory exists; create it if missing.
    var p = Path(outdir)                  # Wrap the directory path in a Path object.
    if not p.exists():                    # Only attempt creation if it does not exist.
        try:                              # Guard against I/O exceptions.
            makedirs(String(p))           # Create the directory tree recursively.
        except _:                         # On any failure (permissions, race, etc.), ignore.
            pass                          # Swallow the error to keep the demo resilient.

fn _save_all(outdir: String, stem: String, img: vision.Image) -> None:  # Save image as PNG using a name stem.
    try:                                # Try to write the image to disk.
        vision.write_image(outdir + "/" + stem + ".png", img)  # Save as PNG with the provided stem.
    except e:                           # If writing fails, capture the exception.
        print(e)                        # Print the error for diagnostics.
    print("Saved: " + outdir + "/" + stem + ".png")  # Log the actual saved path (fixed extension print).

fn save(outdir: String, name: String, img: vision.Image) -> String:  # Save image to an explicit filename; return path.
    var path = outdir + "/" + name      # Compose the full output path.
    try:                                # Attempt to write the image.
        vision.write_image(path, img)   # Write the image to disk (format inferred from extension).
    except e:                           # Capture any I/O/codec errors.
        print(e)                        # Log the error to standard output.
    print("Saved: " + path)             # Confirm where the file was saved.
    return path                         # Return the file path for chaining.

fn show_if(show: Bool, pause_ms: Int, title: String, img: vision.Image) -> None:  # Conditionally show a window.
    if show and vision.supports_windows():   # Only show if requested and windowing is supported.
        vision.imshow(title, img)            # Display the image with a title.
        vision.wait_key(pause_ms)            # Wait for a key press or the given timeout.

# ----------------------- Demo / Safe Load -----------------------

fn make_demo_image() -> vision.Image:   # Create a synthetic image to make borders/ROI effects obvious.
    # Simple demo with blocks to make ROI clear
    var w = 600                         # Canvas width in pixels.
    var h = 400                         # Canvas height in pixels.
    var img = vision.zeros(h, w, 3, vision.UInt8())  # Allocate HxWx3 uint8 image initialized to zeros.
    img = vision.fill(img, [UInt8(50), UInt8(50), UInt8(60)])  # Fill the background with a dark bluish tone.

    img = vision.rectangle(             # Draw a filled cyan rectangle to the left area.
        img, 80, 80, 280, 220, (UInt8(0), UInt8(255), UInt8(255)), -1
    )
    img = vision.circle(                # Draw a filled red circle on the right area.
        img, 400, 160, 60, [UInt8(0), UInt8(0), UInt8(255)], -1
    )
    try:                                # Attempt to overlay explanatory text.
        img = vision.put_text(          # Put a label near the bottom.
            img, "Borders & ROI",       # Text string to draw.
            140, 360,                   # Baseline (x, y) position.
            vision.FONT_SIMPLEX(),      # Font face.
            0.9,                        # Font scale.
            [UInt8(230), UInt8(230), UInt8(255)],  # Light text color (BGR).
            2,                          # Stroke thickness in pixels.
            True                        # Enable anti-aliasing for smoother text.
        )
    except e:                           # In case font/text rendering is not available.
        print(e)                        # Log the error and proceed.
    return img.copy()                   # Return a copy to avoid unintended aliasing.

fn load_image(path: String) -> vision.Image:   # Load an image; if not present/invalid, create and save a demo image.
    # Ensure parent directory exists; then try to read the image.
    # If reading fails or the image is invalid, generate a demo image, save, and return it.

    var p = Path(path)                  # Normalize the file path.

    if not p.exists():                  # If the file itself does not exist, try to ensure the path exists.
        # Best-effort: try recursive create, then single-level, ignore errors
        try:                            # First attempt: recursive directory creation on the full path string.
            makedirs(String(p))         # Note: may fail if 'p' includes a filename; that's acceptable here.
        except _:                       # Ignore the failure and attempt a simpler mkdir.
            try:                        # Second attempt: single-level mkdir (works if 'p' is a directory).
                mkdir(String(p))        # Create one directory level.
            except _:                   # If this also fails, ignore and continue.
                pass                    # Do nothing; we will still produce a valid image later.

    # 2) If file exists, try to read it
    if p.exists():                      # Only proceed to read if the path exists.
        try:                            # Guard the read operation.
            var img = vision.read_image(path)  # Read image; API may return a tuple (status, image).
            # Sanity check: reject zero-sized images
            if img[1].width() > 0 and img[1].height() > 0:  # Validate non-zero dimensions.
                return img[1].copy()     # Return a safe copy of the loaded image.
            # Otherwise fall through to demo generation
        except _:                       # Any read/codec error falls back to demo.
            # Fall through to demo generation on any error
            pass

    # 3) Fallback: generate demo image and try to save it
    var demo = make_demo_image()        # Build a synthetic image as a fallback.
    try:                                # Attempt to persist the demo to the requested path.
        vision.write_image(path, demo)  # Write the demo file (helps users find the generated input).
    except _:                           # Ignore any write failure (e.g., permission issues).
        # Ignore write errors; still return a valid image
        pass

    return demo.copy()                  # Return a valid image even if saving failed.

# ----------------------- 10) Borders & ROI -----------------------

fn borders_and_roi(img: vision.Image, outdir: String) -> None:  # Demonstrate border padding and ROI extraction.
    # 1) Add constant border (top, bottom, left, right)
    var bordered = vision.copy_make_border(   # Pad the image with constant pixels on all sides.
        img, 20, 20, 30, 30,                  # Top=20, Bottom=20, Left=30, Right=30 pixels.
        vision.BORDER_CONSTANT(),             # Border mode: constant color.
        [UInt8(0), UInt8(0), UInt8(255)]      # Border color (BGR): red.
    )
    _save_all(outdir, "40_01_bordered", bordered)  # Save the bordered result.

    # 2) Extract ROI: rectangular region from 1/3 height and 1/3 width
    var h = img.height()                  # Read image height.
    var w = img.width()                   # Read image width.
    var y = h // 3                        # Top-left y of ROI as one-third of height (integer division).
    var x = w // 3                        # Top-left x of ROI as one-third of width  (integer division).

    var roi = vision.crop(img, y, x, 140, 220)     # Crop a 140x220 (h x w) rectangle from (y, x).
    _save_all(outdir, "40_02_roi_cropped", roi)    # Save the cropped ROI.

    # 3) Visualize ROI on original by drawing bounding rectangle
    var roi_vis = vision.rectangle(                # Draw a rectangle on a copy/alias of the original to show ROI.
        img, x, y, x + 220, y + 140,               # Rectangle from (x, y) to (x+220, y+140).
        [UInt8(0), UInt8(255), UInt8(255)], 2      # Cyan outline, thickness=2.
    )
    _save_all(outdir, "40_03_roi_visual", roi_vis) # Save visualization with ROI rectangle.

# ----------------------- Runnable -----------------------

fn main() -> None:                         # Program entry point.
    var outdir = "outputs_borders_roi"     # Output directory for this demo.
    ensure_outdir(outdir)                  # Make sure the output directory exists.

    var input_path = outdir + "/input_demo.png"  # Expected input image path.
    var img = load_image(input_path)       # Load existing input or create a demo image.

    borders_and_roi(img, outdir)           # Run the borders & ROI pipeline and save outputs.

    show_if(False, 1200, "Borders & ROI", img)  # Optional window display (disabled by default).
