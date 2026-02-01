# Project:      Momijo
# Module:       examples.resize_crop_rotate_flip                 # Logical module path within the package.
# File:         resize_crop_rotate_flip.mojo                     # Source filename.
# Path:         src/momijo/examples/resize_crop_rotate_flip.mojo # Full repo path per packaging rules.
#
# Description:  Demonstrates resize, crop, rotation, and flip operations using momijo.vision.
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
#   - Shows robust output-dir creation, safe image loading with fallback, and common geometry ops.

import momijo.vision as vision                # Import vision module under a short alias for image operations.
from pathlib import Path                      # Filesystem path handling.
from os import makedirs                       # Recursive directory creation (mkdir -p).
from os import mkdir                          # Single-level directory creation.
from collections.list import List             # List type (not strictly needed here, but available if used).

# ----------------------- Utility Helpers -----------------------

fn ensure_outdir(outdir: String) -> None:     # Ensure the output directory exists; create if missing.
    var p = Path(outdir)                      # Wrap the directory path in a Path object.
    if not p.exists():                        # Only attempt creation if it does not already exist.
        try:                                  # Guard against OS errors (permissions, races).
            makedirs(String(p))               # Recursively create the full directory tree.
        except _:                             # Swallow any exception to keep demo resilient.
            pass                              # No-op on failure.

fn _save_all(outdir: String, stem: String, img: vision.Image) -> None:  # Save 'img' as PNG with a stem name.
    try:                                  # Attempt to write; do not abort on failure.
        vision.write_image(outdir + "/" + stem + ".png", img)  # Write to "<outdir>/<stem>.png".
    except e:                              # Catch any I/O or codec error.
        print(e)                           # Print the error for visibility.
    print("Saved: " + outdir + "/" + stem + ".png")  # Confirm the saved path.

fn save(outdir: String, name: String, img: vision.Image) -> String:  # Save with explicit filename and return the path.
    var path = outdir + "/" + name          # Construct the full output path.
    try:                                     # Attempt to write the image.
        vision.write_image(path, img)        # Write using extension-inferred codec.
    except e:                                # Catch any write error.
        print(e)                             # Log the error but continue.
    print("Saved: " + path)                  # Log success/attempted path.
    return path                              # Return the path to the caller.

fn show_if(show: Bool, pause_ms: Int, title: String, img: vision.Image) -> None:  # Optional preview window.
    if show and vision.supports_windows():   # Only show if requested and windowing is supported.
        vision.imshow(title, img)            # Display the image in a window titled 'title'.
        vision.wait_key(pause_ms)            # Wait for a key event or timeout in milliseconds.

# ----------------------- Demo Image & Safe Load -----------------------

fn make_demo_image() -> vision.Image:        # Build a synthetic demo image with shapes and title text.
    var w = 800                               # Canvas width in pixels.
    var h = 500                               # Canvas height in pixels.
    var img = vision.zeros(h, w, 3, vision.UInt8())  # Create an HxWx3 uint8 image initialized to zeros (BGR).
    img = vision.fill(img, [UInt8(32), UInt8(32), UInt8(36)])  # Fill background with a dark BGR color.

    img = vision.rectangle(
