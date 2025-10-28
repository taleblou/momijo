# Project:      Momijo                                   # Project/repository name.
# Module:       examples.drawing_text                    # Logical module within the package.
# File:         drawing_and_text.mojo                    # Source filename.
# Path:         src/momijo/examples/drawing_and_text.mojo  # Full path from repo root.
#
# Description:  Demo of drawing primitives and text rendering using momijo.vision.
#
# Author(s):    Morteza Taleblou & Mitra Daneshmand      # File maintainers.
# Website:      https://taleblou.ir/                     # Project/author website.
# Repository:   https://github.com/taleblou/momijo       # Canonical source repository.
#
# License:      MIT License                              # License label (short form only).
# SPDX-License-Identifier: MIT                           # SPDX identifier for tooling.
# Copyright:    (c) 2025 Morteza Taleblou & Mitra Daneshmand  # Copyright notice.
#
# Notes:                                                   # Quick pointers for readers.
#   - Demonstrates: directory setup, robust image load fallback, drawing ops, text.
#   - I/O: saves PNG files at each step for easy visual verification.

import momijo.vision as vision                            # Import vision API under a short alias for image ops.
from pathlib import Path                                  # Path class for filesystem-safe path handling.
from os import makedirs                                   # Recursive directory creation (mkdir -p behavior).
from os import mkdir                                      # Single-level directory creation.
from collections.list import List                         # Typed dynamic list container.

# ----------------------- Utility Helpers -----------------------

fn ensure_outdir(outdir: String) -> None:                 # Ensure an output directory exists; create it if missing.
    var p = Path(outdir)                                  # Wrap the string path in a Path object.
    if not p.exists():                                    # Guard: only create when it does not already exist.
        try:                                              # Protected section to avoid throwing on race/permissions.
            makedirs(String(p))                           # Recursively create all missing directories.
        except _:                                         # Catch-all: ignore any creation error.
            pass                                          # No-op to keep demo resilient across environments.

fn _save_all(outdir: String, stem: String, img: vision.Image) -> None:  # Save an image as PNG with a name stem.
    try:                                                  # Attempt to write the file; avoid aborting on failure.
        vision.write_image(outdir + "/" + stem + ".png", img)  # Persist the image (format inferred by extension).
    except e:                                             # On any I/O/codec error, handle gracefully.
        print(e)                                          # Print the error for visibility during development.
    print("Saved: " + outdir + "/" + stem + ".png")       # Confirm the exact saved file path.

fn save(outdir: String, name: String, img: vision.Image) -> String:  # Save with explicit filename; return the path.
    var path = outdir + "/" + name                        # Build the output path string.
    try:                                                  # Guard the write to prevent exceptions from bubbling up.
        vision.write_image(path, img)                     # Write the image to disk.
    except e:                                             # Handle any failure (permissions, codec, etc.).
        print(e)                                          # Log the error for troubleshooting.
    print("Saved: " + path)                               # Emit a confirmation message.
    return path                                           # Return the final path to the caller.

fn show_if(show: Bool, pause_ms: Int, title: String, img: vision.Image) -> None:  # Optional GUI preview helper.
    if show and vision.supports_windows():                # Only show when requested and GUI support is present.
        vision.imshow(title, img)                         # Display the image in a window with a title.
        vision.wait_key(pause_ms)                         # Block for a short time to allow viewing.

# ----------------------- Demo / Safe Load -----------------------

fn make_demo_image() -> vision.Image:                     # Create a synthetic demo image to use when no input exists.
    var w = 800                                           # Desired canvas width in pixels.
    var h = 500                                           # Desired canvas height in pixels.
    var img = vision.zeros(h, w, 3, vision.UInt8())       # Allocate HxWx3 uint8 image initialized to zeros.
    img = vision.fill(img, [UInt8(32), UInt8(32), UInt8(36)])  # Fill background with a dark BGR color.

    img = vision.rectangle(img, 24, 24, 240, 140, (UInt8(0), UInt8(255), UInt8(255)), -1)  # Solid cyan rectangle.
    img = vision.rectangle(img, 280, 24, 520, 140, (UInt8(255), UInt8(0), UInt8(255)), 3)  # Magenta rectangle outline.

    img = vision.circle(img, 120, 310, 70, [UInt8(0), UInt8(128), UInt8(255)], -1)         # Filled orange-ish circle.
    img = vision.circle(img, 360, 300, 80, [UInt8(255), UInt8(255), UInt8(0)], 3)          # Yellow circle outline.

    img = vision.arrowed_line(img, 24, 180, 260, 250, [UInt8(0), UInt8(255), UInt8(0)], 2, 0.15)  # Green arrow.
    img = vision.line(img, 560, 70, 760, 70, [UInt8(255), UInt8(255), UInt8(255)], 2)             # White horizontal line.

    var pts_flat = [600, 40, 760, 40, 720, 140, 580, 140]   # Flattened polygon points: [x0,y0, x1,y1, ...].
    img = vision.fill_poly(img, pts_flat, [UInt8(0), UInt8(200), UInt8(80)])  # Filled greenish polygon.

    try:                                                  # Text rendering may fail if fonts are unavailable.
        img = vision.put_text(                            # Draw a text label on the image.
            img, "Drawing & Text Base",                   # Target image and the string to render.
            190, 200, vision.FONT_SIMPLEX(), 1.0,         # Baseline (x,y), font face, and scale.
            [UInt8(220), UInt8(220), UInt8(255)], 2, True # BGR color, thickness, anti-aliased flag.
        )
    except e:                                             # Catch any rendering-related errors.
        print(e)                                          # Log for diagnosis without failing the demo.
    return img.copy()                                     # Return a copy to avoid shared ownership issues.

fn load_image(path: String) -> vision.Image:              # Load an image or fall back to a generated demo image.
    # Ensures parent existence, reads the image if present, and falls back to demo on errors.

    var p = Path(path)                                    # Wrap the string path for filesystem checks.

    if not p.exists():                                    # If the target file is absent, ensure directories exist.
        try:                                              # First, try to create parents recursively (best-effort).
            makedirs(String(p))                           # May fail if 'p' includes a filename.
        except _:                                         # Ignore errors (will try a simpler mkdir next).
            try:                                          # Second attempt with single-level mkdir.
                mkdir(String(p))                          # Create just one directory level.
            except _:                                     # Ignore any error to keep behavior robust.
                pass                                      # No-op.

    if p.exists():                                        # If a file exists at 'path', try to read it.
        try:                                              # Protect against corrupt/unsupported files.
            var img = vision.read_image(path)             # Attempt to read; often returns (status, image).
            if img[1].width() > 0 and img[1].height() > 0:  # Basic sanity check for non-empty image.
                return img[1].copy()                      # Return a defensive copy of the loaded image.
        except _:                                         # On any error, fall through to demo generation.
            pass                                          # No-op; proceed to fallback.

    var demo = make_demo_image()                          # Fallback: create a synthetic demo image.
    try:                                                  # Try to persist the demo to the requested path.
        vision.write_image(path, demo)                    # Write demo image so users see an example asset.
    except _:                                             # Ignore write errors (e.g., permissions).
        pass                                              # Still return a valid image.
    return demo.copy()                                    # Always return a usable image to the caller.

# ----------------------- 4) Drawing & Text -----------------------
# Each modification is saved immediately as PNG for step-by-step inspection.

fn drawing_and_text(img: vision.Image, outdir: String) -> None:  # Showcase drawing primitives and text.
    var canvas = vision.full_like(img, UInt8(255))        # Start from a white canvas with same shape as input.
    _save_all(outdir, "11_00_canvas_white", canvas)       # Save the initial canvas.

    canvas = vision.line(canvas, 20, 20, 220, 20, [UInt8(0), UInt8(0), UInt8(0)], 2)  # Draw a black horizontal line.
    _save_all(outdir, "11_01_line", canvas)               # Save after drawing the line.

    canvas = vision.rectangle(canvas, 20, 40, 220, 160, (UInt8(255), UInt8(0), UInt8(0)), 2)  # Blue rectangle (BGR).
    _save_all(outdir, "11_02_rectangle", canvas)          # Save after drawing the rectangle.

    canvas = vision.circle(canvas, 320, 100, 60, [UInt8(0), UInt8(0), UInt8(255)], -1)  # Filled red circle (BGR).
    _save_all(outdir, "11_03_circle_filled", canvas)      # Save the result with the filled circle.

    var pts: List[List[Int]] = [[460, 40], [640, 40], [600, 160], [480, 160]]  # Polygon as list of [x,y] pairs.
    canvas = vision.fill_poly(canvas, pts, [UInt8(0), UInt8(255), UInt8(0)])   # Filled green polygon.
    _save_all(outdir, "11_04_polygon", canvas)            # Save after polygon fill.

    try:                                                  # Text rendering guarded to avoid aborting on errors.
        canvas = vision.put_text(                         # Draw a title on the canvas.
            canvas, "Shapes & Text",                      # Target image and text content.
            20, 220, vision.FONT_SIMPLEX(), 1.0,          # Baseline (x,y), font face, and scale.
            [UInt8(128), UInt8(0), UInt8(128)],           # Purple text color in BGR.
            2, True                                       # Thickness and anti-aliased flag.
        )
    except e:                                             # If rendering fails, continue gracefully.
        print(e)                                          # Log the error for debugging.
    _save_all(outdir, "11_05_text", canvas)               # Save the final canvas with text.

# ----------------------- Runnable -----------------------

fn main() -> None:                                        # Program entry point.
    var outdir = "outputs_drawing_and_text"               # Output directory for this demo.
    ensure_outdir(outdir)                                 # Create the directory if it does not exist.

    var input_path = outdir + "/input_demo.png"           # Intended input path (will be created on fallback).
    var base = load_image(input_path)                     # Load existing image or generate demo if missing.

    drawing_and_text(base, outdir)                        # Run the drawing/text demo and save each step.

    show_if(False, 1200, "Drawing & Text", base)          # Optional GUI preview (disabled by default).
