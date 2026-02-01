# Project:      Momijo                          # Repository/project name.
# Module:       examples.blending_bitwise       # Logical module path for this example.
# File:         blending_bitwise.mojo           # Source filename for this demo.
# Path:         src/momijo/examples/blending_bitwise.mojo  # Repo-relative path per packaging rules.
#
# Description:  Demonstrates image blending and bitwise operations using momijo.vision.
#
# Author(s):    Morteza Taleblou & Mitra Daneshmand  # File authors/maintainers.
# Website:      https://taleblou.ir/                 # Project/author website.
# Repository:   https://github.com/taleblou/momijo   # Canonical repository URL.
#
# License:      MIT License                     # Short license label (do not paste full text).
# SPDX-License-Identifier: MIT                  # SPDX identifier required by project policies.
# Copyright:    (c) 2025 Morteza Taleblou & Mitra Daneshmand  # Copyright notice.

import momijo.vision as vision                 # Import the vision API for image operations.

from pathlib import Path                       # Path utility for filesystem paths.
from os import makedirs                        # Recursive directory creation function.
from os import mkdir                           # Single-level directory creation function.
from collections.list import List              # Typed dynamic list container.

# ----------------------- Utility Helpers -----------------------

fn ensure_outdir(outdir: String) -> None:      # Ensure the output directory exists; create it if missing.
    var p = Path(outdir)                       # Convert the string to a Path object.
    if not p.exists():                         # If the directory does not exist yet:
        try:                                   # Try to create it gracefully.
            makedirs(String(p))                # Recursively create all needed directories.
        except _:                              # Swallow any exception (permissions, race, etc.).
            pass                               # No-op to keep demo robust.

fn _save_all(outdir: String, stem: String, img: vision.Image) -> None:  # Save 'img' as outdir/stem.png (helper).
    try:                                       # Attempt to write the image file.
        vision.write_image(outdir + "/" + stem + ".png", img)  # Save as PNG using the provided stem.
    except e:                                   # If an error occurs during write:
        print(e)                                # Print the error without aborting the program.
    print("Saved: " + outdir + "/" + stem + ".png")  # Log the saved path (fixed from '.{png}' to '.png').

fn save(outdir: String, name: String, img: vision.Image) -> String:   # Save 'img' as outdir/name; return full path.
    var path = outdir + "/" + name             # Build the output path string.
    try:                                       # Attempt to write the image file.
        vision.write_image(path, img)          # Save image; format is inferred by the extension.
    except e:                                   # If writing fails:
        print(e)                                # Print the error for debugging.
    print("Saved: " + path)                    # Confirm save operation to the console.
    return path                                 # Return the path to the caller.

# ----------------------- Demo / Safe Load -----------------------

fn make_demo_image() -> vision.Image:          # Create a synthetic demo image for bitwise operations.
    # Simple colored blocks for bitwise operations (English-only comment).
    var w = 600                                # Canvas width in pixels.
    var h = 400                                # Canvas height in pixels.
    var img = vision.zeros(h, w, 3, vision.UInt8())  # Allocate HxWx3 uint8 image initialized to zeros.
    img = vision.fill(img, [UInt8(40), UInt8(80), UInt8(160)])  # Fill with a blue-ish background color (BGR).

    img = vision.rectangle(                    # Draw a filled cyan rectangle to create solid geometry.
        img, 60, 60, 260, 200, (UInt8(0), UInt8(255), UInt8(255)), -1
    )
    img = vision.circle(                       # Draw a filled red circle to overlap shapes.
        img, 420, 150, 70, [UInt8(0), UInt8(0), UInt8(255)], -1
    )
    try:                                       # Text rendering may fail if font subsystem is unavailable.
        img = vision.put_text(                 # Render a label for the demo image.
            img, "Bitwise Demo",               # Target image and text string.
            140, 360,                          # Baseline origin (x, y).
            vision.FONT_SIMPLEX(),             # Font face.
            0.9,                               # Font scaling.
            [UInt8(255), UInt8(255), UInt8(255)],  # Text color in BGR (white).
            2,                                 # Stroke thickness.
            True                               # Enable anti-aliased rendering.
        )
    except e:                                   # If put_text fails:
        print(e)                                # Log the error and continue.
    return img.copy()                           # Return a defensive copy to avoid shared ownership issues.

fn load_image(path: String) -> vision.Image:   # Load an image from disk; if not available, create/save a demo.
    # Ensure parent directory exists; then try to read the image.
    # If reading fails or image is invalid, generate and return a demo image (and try to save it).

    var p = Path(path)                          # Wrap the input path string.

    if not p.exists():                          # If the target file does not exist:
        # Best-effort: try recursive create, then single-level, ignore errors (English-only comment).
        try:                                    # First attempt: create recursively (may be a file path).
            makedirs(String(p))                 # Create nested directories if applicable.
        except _:                               # If that fails (likely because 'p' is a file path):
            try:                                # Second attempt: single directory creation.
                mkdir(String(p))                # Create the directory if possible.
            except _:                           # Ignore any failure (permissions, race, etc.).
                pass                            # No-op.

    # 2) If file exists, try to read it
    if p.exists():                              # Only attempt to read when something exists at 'path'.
        try:                                    # Guard against read/codec errors.
            var img = vision.read_image(path)   # Read image; API may return (status, image) tuple.
            # Sanity check: reject zero-sized images
            if img[1].width() > 0 and img[1].height() > 0:  # Validate the loaded image dimensions.
                return img[1].copy()            # Return a safe copy of the loaded image.
            # Otherwise fall through to demo generation
        except _:                               # On any read error:
            # Fall through to demo generation on any error (English-only comment).
            pass

    # 3) Fallback: generate demo image and try to save it
    var demo = make_demo_image()                # Build a synthetic image as a fallback.
    try:                                        # Attempt to persist the fallback for user visibility.
        vision.write_image(path, demo)          # Save the demo image to the desired path.
    except _:                                   # Ignore write errors (e.g., invalid path).
        # Ignore write errors; still return a valid image (English-only comment).
        pass

    return demo.copy()                          # Always return a valid image object.

# ----------------------- 8) Blending & Bitwise Ops -----------------------

fn blending_and_bitwise(img: vision.Image, outdir: String) -> None:  # Run blending + bitwise examples and save outputs.
    var h = img.height()                        # Read image height once (used multiple times).
    var w = img.width()                         # Read image width once (used multiple times).

    # Second image: white circle mask
    var img2 = vision.zeros(h, w, 3, vision.UInt8())  # Allocate a blank image for mask geometry.
    img2 = vision.circle(                       # Draw a filled white circle centered in the frame.
        img2, w // 2, h // 2, (h if h < w else w) // 4, [UInt8(255), UInt8(255), UInt8(255)], -1
    )
    _save_all(outdir, "33_00_img2_circle", img2)  # Save the geometric mask helper image.

    # Weighted blending
    var blend = vision.add_weighted(            # Create a linear blend of img and img2.
        img, 0.7,                               # First image with weight 0.7.
        img2, 0.3,                              # Second image with weight 0.3.
        0.0                                     # Scalar added to the result (bias), set to 0.
    )
    _save_all(outdir, "33_01_blend", blend)     # Save the blended result.

    # Binary mask derived from img2
    var mask = vision.bgr_to_gray(img2)         # Convert mask image to single-channel grayscale.
    mask = vision.threshold_binary(mask, 1, 255)  # Threshold to produce a binary mask (0 or 255).
    _save_all(outdir, "33_02_mask", mask)       # Save the binary mask.

    # Bitwise AND (apply mask)
    var and_ = vision.bitwise_and(img, img, mask)  # Keep pixels of 'img' where mask is 255.
    _save_all(outdir, "33_03_bitwise_and", and_)   # Save the masked image.

    # Bitwise OR
    var or_ = vision.bitwise_or(img, img2)      # Combine 'img' with the geometric helper via OR.
    _save_all(outdir, "33_04_bitwise_or", or_)  # Save the OR result.

    # Bitwise NOT
    var not_ = vision.bitwise_not(img)          # Invert all bits of the source image (per channel).
    _save_all(outdir, "33_05_bitwise_not", not_)  # Save the inverted image.

# ----------------------- Runnable -----------------------

fn main() -> None:                              # Program entry point.
    var outdir = "outputs_blending_bitwise"     # Choose output directory for artifacts.
    ensure_outdir(outdir)                       # Create the directory if necessary.

    var input_path = outdir + "/input_demo2.png"  # Define the input image path.
    var img = load_image(input_path)            # Load existing image or generate a demo fallback.

    blending_and_bitwise(img, outdir)           # Run the demo and write all result images.
