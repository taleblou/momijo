# Project:      Momijo
# Module:       examples.io_colors
# File:         io_colors.mojo
# Path:         src/momijo/examples/io_colors.mojo
#
# Description:  Basic image I/O and color conversions demo using momijo.vision.
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
#   - Demonstrates: ensure output directory, save helper, demo image synthesis,
#     robust image loading (with fallback), grayscale/RGB conversion, split/merge.

import momijo.vision as vision          # Import the vision module under a short alias for image ops.

from pathlib import Path                # Import Path for filesystem path handling.
from os import makedirs                 # Import recursive directory creation utility.
from os import mkdir                    # Import single-level directory creation utility.
from collections.list import List       # Import List for typed dynamic arrays.

# ----------------------- Utility Helpers -----------------------

fn ensure_outdir(outdir: String) -> None:   # Ensure the output directory exists; create if missing.
    var p = Path(outdir)                    # Construct a filesystem Path from the provided string.
    if not p.exists():                      # If the directory does not exist, attempt to create it.
        try:                                # Enter a protected block to avoid throwing on failure.
            makedirs(String(p))             # Recursively create the directory tree (best effort).
        except _:                           # Swallow any exception (permissions, races, etc.).
            pass                            # Intentionally ignore errors to keep the demo resilient.

fn save(outdir: String, name: String, img: vision.Image) -> String:  # Save an image to outdir/name and return the path.
    var path = outdir + "/" + name          # Build the output file path as a simple join.
    try:                                     # Attempt to write the image using the vision I/O API.
        vision.write_image(path, img)        # Write image to disk in a format inferred from extension.
    except e:                                # Catch any write error (I/O, unsupported codec, etc.).
        print(e)                             # Print the error to aid debugging without aborting.
    
    print("Saved: " + path)                  # Log the saved path for user feedback.
    return path                              # Return the final file path to the caller.
 
# ----------------------- Demo Image -----------------------

fn make_demo_image() -> vision.Image:        # Build a synthetic demo image with shapes and text.
    var w = 800                              # Set canvas width in pixels.
    var h = 500                              # Set canvas height in pixels.
    var img = vision.zeros(h, w, 3, vision.UInt8())   # Allocate HxWx3 uint8 image initialized to zeros (BGR).
    img = vision.fill(img, [UInt8(32), UInt8(32), UInt8(36)])  # Fill background with a dark BGR color.

    # Rectangles
    img = vision.rectangle(img, 24, 24, 240, 140, (UInt8(0), UInt8(255), UInt8(255)), -1)  # Solid cyan rectangle.
    img = vision.rectangle(img, 280, 24, 520, 140, (UInt8(255), UInt8(0), UInt8(255)), 3)  # Magenta rectangle outline.

    # Circles
    img = vision.circle(img, 120, 310, 70, [UInt8(0), UInt8(128), UInt8(255)], -1)         # Solid orange-ish circle.
    img = vision.circle(img, 360, 300, 80, [UInt8(255), UInt8(255), UInt8(0)], 3)          # Yellow circle outline.

    # Arrow & line
    img = vision.arrowed_line(img, 24, 180, 260, 250, [UInt8(0), UInt8(255), UInt8(0)], 2, 0.15)  # Green arrow.
    img = vision.line(img, 560, 70, 760, 70, [UInt8(255), UInt8(255), UInt8(255)], 2)             # White line.

    # Polygon (flattened x,y coords): [[600,40],[760,40],[720,140],[580,140]]
    var pts:List[Int] = [600, 40, 760, 40, 720, 140, 580, 140]  # Define polygon vertices as interleaved ints.
    img = vision.fill_poly(img, pts, [UInt8(0), UInt8(200), UInt8(80)])   # Fill polygon with a greenish color.

    # Text
    try:                                   # Draw text; this may fail if fonts are unavailable.
        img = vision.put_text(             # Render a string onto the image.
            img,                           # Target image.
            "TEST ONE",                    # Text content.
            250, 200,                      # Baseline origin (x, y).
            vision.FONT_SIMPLEX(),         # Font face enumeration or factory.
            1.2,                           # Font scale factor.
            [UInt8(200), UInt8(200), UInt8(255)],  # Text color in BGR.
            2,                             # Thickness in pixels.
            True                           # Anti-aliased rendering flag.
        )
    except e:                              # If text rendering fails, continue gracefully.
        print(e)                           # Log the error for visibility.
     
    return img.copy()                      # Return a defensive copy to avoid shared ownership issues.
 
fn load_image(path: String) -> vision.Image:   # Load an image; if missing/invalid, generate and return a demo image.
    # Ensure parent directory exists; then try to read the image.
    # If reading fails or the image is invalid, generate a demo image, save, and return it.

    var p = Path(path)                      # Wrap the given path string in a Path object.
 
    if not p.exists():                      # If the target file does not exist, ensure parent dirs exist.
        # Best-effort: try recursive create, then single-level, ignore errors
        try:                                # First, attempt to create all missing directories recursively.
            makedirs(String(p))             # This may fail if p includes a filename; we catch that.
        except _:                           # Ignore failures (e.g., path points to a file).
            try:                            # Second attempt: try single-level mkdir (in case parent exists).
                mkdir(String(p))            # Create a single directory level if applicable.
            except _:                       # Ignore any error to keep function robust.
                pass                        # No-op on failure.

    # 2) If file exists, try to read it
    if p.exists():                          # Only attempt reading when the path is present.
        try:                                # Guard the read in case the file is corrupt or unsupported.
            var img = vision.read_image(path)  # Read the image; API may return (status, image) or similar tuple.
            # Sanity check: reject zero-sized images
            if img[1].width() > 0 and img[1].height() > 0:  # Validate loaded image dimensions.
                return img[1].copy()        # Return a safe copy of the loaded image.
            # Otherwise fall through to demo generation
        except _:                           # On any read error, proceed to demo image generation.
            # Fall through to demo generation on any error
            pass

    # 3) Fallback: generate demo image and try to save it
    var demo = make_demo_image()            # Create a synthetic demo image as a fallback/default.
    try:                                    # Try to persist the demo image at the requested path.
        vision.write_image(path, demo)      # Write the demo so users can see the generated asset.
    except _:                               # If writing fails (permissions, invalid path), ignore the error.
        # Ignore write errors; still return a valid image
        pass

    return demo.copy()                      # Always return a valid image (copy) to the caller.

# ----------------------- 2) Basic I/O & Colors -----------------------

fn basic_io_and_colors(img: vision.Image, outdir: String, show: Bool, pause_ms: Int) -> None:  # Demonstrate I/O and color ops.
    save(outdir, "01_original_bgr.png", img)        # Save the original BGR image.

    var gray = vision.bgr_to_gray(img)               # Convert BGR to single-channel grayscale.
    save(outdir, "02_gray.png", gray)                # Save the grayscale image.

    var rgb = vision.bgr_to_rgb(img)                 # Convert BGR to RGB color order.
    # Save RGB by converting back to BGR so current BGR-based writers remain compatible.
    var rgb_as_bgr = vision.rgb_to_bgr(rgb)          # Reorder channels back to BGR for writing.
    save(outdir, "03_rgb_saved_as_bgr.png", rgb_as_bgr)  # Save the converted image.

    var parts = vision.split3(img.clone())           # Split B, G, R into three single-channel images (rvalue tuple).
    var b = parts[0].copy()                          # Copy the B channel out of the tuple.
    var g = parts[1].copy()                          # Copy the G channel out of the tuple.
    var r = parts[2].copy()                          # Copy the R channel out of the tuple.

    var merged = vision.merge3(b, g, r)              # Merge channels back into a 3-channel image (BGR).
                                                     # (Optionally save/visualize 'merged' if needed.)

# ----------------------- Runnable -----------------------

fn main() -> None:                                   # Entry point with no arguments.
    var outdir = "outputs_io_colors"                 # Choose an output directory for generated files.
    ensure_outdir(outdir)                            # Make sure the output directory exists.

    # If there is no input, we build/write a demo image first.
    var input_path = outdir + "/input_demo1.png"     # Define the intended input image path.
    var img = load_image(input_path)                 # Load image or generate a demo if missing/invalid.

    basic_io_and_colors(img, outdir, False, 800)     # Run the demo pipeline (show/pause currently unused).
