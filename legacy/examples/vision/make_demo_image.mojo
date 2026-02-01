# Project:      Momijo
# Module:       examples.demo_steps
# File:         demo_steps.mojo
# Path:         src/momijo/examples/demo_steps.mojo
#
# Description:  Step-by-step image drawing demo (I/O + shapes + text) with saved outputs.
#
# Author(s):    Morteza Taleblou & Mitra Daneshmand
# Website:      https://taleblou.ir/
# Repository:   https://github.com/taleblou/momijo
#
# License:      MIT License
# SPDX-License-Identifier: MIT
# Copyright:    (c) 2025 Morteza Taleblou & Mitra Daneshmand

import momijo.vision as vision                    # Import vision APIs under short alias for image ops.
from pathlib import Path                          # Path utility for filesystem-safe path handling.
from os import makedirs                           # Recursive directory creation helper.
from os import mkdir                              # Single-level directory creation helper.
from collections.list import List                 # Generic List for typed dynamic arrays.

# ----------------------- Utility Helpers -----------------------
fn ensure_outdir(outdir: String) -> None:         # Ensure an output directory exists; create if missing.
    var p = Path(outdir)                          # Wrap the string as a Path instance.
    if not p.exists():                            # If the directory does not exist...
        try:                                      # Attempt a best-effort creation without failing the program.
            makedirs(String(p))                   # Recursively create directories (no-op if already present).
        except _:                                 # Swallow any exception (permissions, race condition, etc.).
            pass                                  # Intentionally ignore to keep demo resilient.

fn _save_all(outdir: String, stem: String, img: vision.Image) -> None:  # Save an image as PNG using a stem filename.
    try:                                                              # Guard against I/O or codec errors.
        vision.write_image(outdir + "/" + stem + ".png", img)         # Write image with .png extension.
    except e:                                                         # On error, capture the exception object.
        print(e)                                                      # Log the error for debugging visibility.
    print("Saved: " + outdir + "/" + stem + ".png")                   # Confirm save path in the console.

fn save(outdir: String, name: String, img: vision.Image) -> String:   # Save image to a specific filename; return its path.
    var path = outdir + "/" + name                                    # Construct full destination path.
    try:                                                              # Guard the write to avoid crashing on failure.
        vision.write_image(path, img)                                 # Persist the image to disk (format by extension).
    except e:                                                         # Capture any thrown error during write.
        print(e)                                                      # Print error details; continue execution.
    print("Saved: " + path)                                           # Notify user of the saved file location.
    return path                                                       # Return path for chaining/verification.

fn show_if(show: Bool, pause_ms: Int, title: String, img: vision.Image) -> None:  # Conditionally show a window.
    if show and vision.supports_windows():                            # Only display if requested and supported.
        vision.imshow(title, img)                                     # Open a window and show the image.
        vision.wait_key(pause_ms)                                     # Block for a key press or timeout in milliseconds.

# ----------------------- 1) Input Preparation -----------------------
fn _u8(v: Int) -> UInt8:                                              # Clamp an Int to [0,255] and cast to UInt8.
    var x = v                                                         # Copy value to a mutable local.
    if x < 0: x = 0                                                   # Clamp lower bound.
    if x > 255: x = 255                                               # Clamp upper bound.
    return UInt8(x)                                                   # Return as 8-bit unsigned integer.

fn make_demo_image_and_save_steps(outdir: String) -> vision.Image:    # Build a demo image while saving each step.
    var w = 800                                                       # Canvas width in pixels.
    var h = 500                                                       # Canvas height in pixels.

    # Background
    var img = vision.zeros(h, w, 3, vision.UInt8())                   # Allocate HxWx3 uint8 image initialized to zero.
    img = vision.fill(img, [UInt8(32), UInt8(32), UInt8(36)])         # Fill background with a dark BGR tone.
    save(outdir, "step_01_background.png", img)                       # Save the initial background step.

    # Rectangles
    img = vision.rectangle(                                           # Draw the first rectangle (filled).
        img, 24, 24, 240, 140,                                       # Top-left (24,24), bottom-right (240,140).
        (UInt8(0), UInt8(255), UInt8(255)),                           # BGR color: cyan.
        -1                                                            # Thickness -1 indicates a filled rectangle.
    )
    save(outdir, "step_02_rect_filled.png", img)                      # Save state after filled rectangle.

    img = vision.rectangle(                                           # Draw the second rectangle (outline).
        img, 280, 24, 520, 140,                                       # Top-left (280,24), bottom-right (520,140).
        (UInt8(255), UInt8(0), UInt8(255)),                           # BGR color: magenta.
        3                                                             # Thickness 3 pixels (outline).
    )
    save(outdir, "step_03_rect_outline.png", img)                     # Save state after outlined rectangle.

    # Circles
    img = vision.circle(img, 120, 310, 70, [UInt8(0), UInt8(128), UInt8(255)], -1)  # Filled circle (center 120,310; r=70).
    save(outdir, "step_04_circle_filled.png", img)                    # Save state after filled circle.

    img = vision.circle(img, 360, 300, 80, [UInt8(255), UInt8(255), UInt8(0)], 3)   # Outlined circle (center 360,300; r=80).
    save(outdir, "step_05_circle_outline.png", img)                   # Save state after circle outline.

    # Arrow & line
    img = vision.arrowed_line(img, 24, 180, 260, 250, [UInt8(0), UInt8(255), UInt8(0)], 2, 0.15)  # Green arrow.
    save(outdir, "step_06_arrow.png", img)                            # Save state after arrow.

    img = vision.line(img, 560, 70, 760, 70, [UInt8(255), UInt8(255), UInt8(255)], 2)             # White horizontal line.
    save(outdir, "step_07_line.png", img)                             # Save state after line.

    # Polygon (x,y flattened: [[600,40],[760,40],[720,140],[580,140]])
    var pts: List[Int] = [600, 40, 760, 40, 720, 140, 580, 140]       # Define polygon vertices as interleaved ints.
    img = vision.fill_poly(img, pts, [UInt8(0), UInt8(200), UInt8(80)])  # Fill polygon with a greenish color.
    save(outdir, "step_08_polygon.png", img)                          # Save state after polygon.

    try:                                                              # Attempt to render text (font availability may vary).
        img = vision.put_text(                                        # Draw string using a specific font.
            img, "Demo",                                              # Target image and text content.
            190, 200, vision.FONT_SIMPLEX(), 1.2,                     # Baseline (190,200), font face, scale 1.2.
            [UInt8(220), UInt8(220), UInt8(255)], 2, True             # BGR color, thickness 2, anti-aliased on.
        )
    except e:                                                         # Catch any rendering exceptions.
        print(e)                                                      # Log and continue without failing the demo.
    save(outdir, "step_09_text.png", img)                             # Save state after text rendering.

    return img.copy()                                                 # Return a defensive copy to avoid aliasing issues.

# ----------------------- Runnable Example -----------------------
fn main() -> None:                                                    # Program entry point.
    var outdir = "outputs_demo"                                       # Output directory for all saved images.
    ensure_outdir(outdir)                                             # Create the directory if it does not exist.

    # Build demo image while saving each step
    var img = make_demo_image_and_save_steps(outdir)                  # Generate the image and write step outputs.

    # Final save & optional show
    save(outdir, "step_10_final.png", img)                            # Save the final composite image.
    show_if(False, 1200, "Momijo Vision Demo", img)                   # Optionally display; disabled by default.
