# Project:      Momijo                         # Project/repository name.
# Module:       examples.geometric_transforms  # Logical module path within the package.
# File:         geo_transforms.mojo            # Source filename.
# Path:         src/momijo/examples/geo_transforms.mojo  # Full path from repo root.
#
# Description:  Demo of geometric transforms (translate, affine, perspective) with robust I/O.
#
# Author(s):    Morteza Taleblou & Mitra Daneshmand  # File maintainers.
# Website:      https://taleblou.ir/                 # Project/author website.
# Repository:   https://github.com/taleblou/momijo   # Canonical repository.
#
# License:      MIT License                   # Short license tag (no full text here).
# SPDX-License-Identifier: MIT               # SPDX identifier required by policy.
# Copyright:    (c) 2025 Morteza Taleblou & Mitra Daneshmand  # Copyright.
#
# Notes:
#   - This example saves each step to disk to aid visual inspection.
#   - Window display is optional and guarded by supports_windows().

import momijo.vision as vision              # Import vision module for image ops, named 'vision' for brevity.
from pathlib import Path                    # Path utilities for filesystem-safe paths.
from os import makedirs                     # Recursive directory creation (mkdir -p behavior).
from os import mkdir                        # Single-level directory creation.
from collections.list import List           # Typed dynamic array for integer point lists.

# ----------------------- Utility Helpers -----------------------
fn ensure_outdir(outdir: String) -> None:   # Ensure an output directory exists; create if missing.
    var p = Path(outdir)                    # Wrap the string path in a Path object.
    if not p.exists():                      # If the path does not exist, attempt creation.
        try:                                # Guard against exceptions (permissions/race).
            makedirs(String(p))             # Recursively create directories for the given path.
        except _:                           # Swallow any exception to keep the example resilient.
            pass                            # No-op on failure (caller continues regardless).

fn _save_all(outdir: String, stem: String, img: vision.Image) -> None:  # Save image as PNG with a stem name.
    try:                                                                # Attempt to write the image file.
        vision.write_image(outdir + "/" + stem + ".png", img)           # Save using .png extension.
    except e:                                                            # On any write failure...
        print(e)                                                         # ...print the error for debugging.
    print("Saved: " + outdir + "/" + stem + ".{png}")                    # Status log; NOTE: likely a typo:
                                                                         # prints '.{png}' literally; consider '.png'.

fn save(outdir: String, name: String, img: vision.Image) -> String:     # Save image to a given filename; return path.
    var path = outdir + "/" + name                                       # Compose full file path.
    try:                                                                 # Attempt to write the image file.
        vision.write_image(path, img)                                    # Save image; format inferred by extension.
    except e:                                                            # Catch and log any error without raising.
        print(e)                                                         # Print error message.
    print("Saved: " + path)                                              # Status log for the saved file path.
    return path                                                          # Return the file path to the caller.

fn show_if(show: Bool, pause_ms: Int, title: String, img: vision.Image) -> None:  # Conditionally show windowed preview.
    if show and vision.supports_windows():                               # Only show if requested and backend available.
        vision.imshow(title, img)                                        # Display image in a window.
        vision.wait_key(pause_ms)                                        # Wait for a key event or timeout.

# ----------------------- Demo / Safe Load -----------------------

fn make_demo_image() -> vision.Image:                                    # Build a high-contrast synthetic image.
    # High-contrast scene for geometric transforms                      # Context comment for content choice.
    var w = 720                                                          # Canvas width in pixels.
    var h = 480                                                          # Canvas height in pixels.
    var img = vision.zeros(h, w, 3, vision.UInt8())                      # Allocate HxWx3 uint8 image initialized to 0.
    img = vision.fill(img, [UInt8(30), UInt8(30), UInt8(35)])            # Fill with a dark BGR background.

    img = vision.rectangle(img, 60, 60, 300, 220, (UInt8(0), UInt8(255), UInt8(255)), -1)  # Solid cyan rectangle.
    img = vision.circle(img, 520, 160, 70, [UInt8(0), UInt8(0), UInt8(255)], -1)           # Solid red disk.
    img = vision.arrowed_line(img, 80, 300, 300, 380, [UInt8(255), UInt8(255), UInt8(255)], 3, 0.15)  # White arrow.
 
    try:                                                                 # Text rendering may fail if fonts unavailable.
        # Text edges                                                     # Context: add edges via text for features.
        img = vision.put_text(                                           # Draw text onto the image.
            img, "Geometric Transform",                                  # Target image and string content.
            180, 360, vision.FONT_SIMPLEX(), 1.2,                        # Baseline (x,y), font, and scale.
            [UInt8(220), UInt8(220), UInt8(220)], 2, True                # BGR color, thickness, anti-aliased.
        )
    except e:                                                            # If put_text fails...
        print(e)                                                         # ...log the exception and continue.
    return img.copy()                                                    # Return a defensive copy (avoid shared state).

fn load_image(path: String) -> vision.Image:                             # Load image or generate demo fallback.
    # Ensure parent directory exists; then try to read the image.        # High-level function contract.
    # If reading fails or the image is invalid, generate a demo image,   # Fallback strategy description.
    # save, and return it.                                               # The demo is also saved when possible.

    var p = Path(path)                                                   # Wrap path string into a Path object.
 
    if not p.exists():                                                   # If the target path does not exist...
        # Best-effort: try recursive create, then single-level, ignore errors  # Directory creation strategy.
        try:                                                             # First attempt: recursive creation.
            makedirs(String(p))                                          # Create the full path; may fail if includes file.
        except _:                                                        # If that fails (e.g., path includes filename)...
            try:                                                         # Second attempt: single-level mkdir.
                mkdir(String(p))                                         # Create a single directory level.
            except _:                                                    # Ignore any error in both attempts.
                pass                                                     # No-op; proceed to fallback later.

    # 2) If file exists, try to read it                                  # Proceed to reading if path exists.
    if p.exists():                                                       # Only read when the path is present.
        try:                                                             # Guard read operation.
            var img = vision.read_image(path)                            # Attempt to read; often returns (ok, image).
            # Sanity check: reject zero-sized images                     # Validate that the image is non-empty.
            if img[1].width() > 0 and img[1].height() > 0:               # Width/height must both be > 0.
                return img[1].copy()                                     # Return a safe copy of the loaded image.
            # Otherwise fall through to demo generation                  # Invalid image ⇒ generate demo below.
        except _:                                                        # On any read error...
            # Fall through to demo generation on any error               # ...fall through to the fallback path.
            pass

    # 3) Fallback: generate demo image and try to save it                # Build and store a demo image.
    var demo = make_demo_image()                                         # Create a synthetic image.
    try:                                                                 # Attempt to persist the demo at 'path'.
        vision.write_image(path, demo)                                   # Save demo image to help users inspect.
    except _:                                                            # Ignore write failures (permissions, etc.).
        # Ignore write errors; still return a valid image                # The function must always return an image.
        pass

    return demo.copy()                                                   # Return a copy of the demo image.

# ----------------------- 9) Geometric Transforms (Affine / Perspective) -----------------------
# Every modification is saved immediately.                               # Design note: _save_all at each step.

fn geometric_transforms(img: vision.Image, outdir: String) -> None:      # Apply and save several geometric transforms.
    # 0) Original                                                        # Step label for clarity in outputs.
    _save_all(outdir, "36_00_original_bgr", img)                         # Save the input as the starting reference.

    var h = img.height()                                                 # Cache image height (pixels).
    var w = img.width()                                                  # Cache image width (pixels).

    # 1) Translation: shift right/down by (tx=40, ty=30)                 # Translate to demonstrate pure shift.
    #    Border mode uses reflection (BS variant if supported in your lib).  # Border handling note.
    var trans = vision.translate(img, 40, 30, vision.BORDER_REFLECT_BS())# Translate with reflect-border (BS flavor).
    _save_all(outdir, "37_01_translate", trans)                          # Save translated image.

    # 2) Affine: map 3 source points to 3 destination points             # Affine warp from 3 point correspondences.
    #    src: (0,0), (w-1,0), (0,h-1)                                    # Source control points (corners).
    #    dst: slight skew + shift                                        # Destination points produce skew/shift.
    var affine = vision.affine(                                          # Compute/apply affine transform.
        img,                                                             # Source image.
        ((0.0, 0.0), (Float64(w) - 1.0, 0.0), (0.0, Float64(h) - 1.0)),  # Source triangle (double precision).
        ((0.0, 0.0), (Float64(w) * 0.9, 40.0), (40.0, Float64(h) * 0.95)),# Destination triangle (skew/shift).
        vision.BORDER_REFLECT()                                          # Border handling: reflect (standard variant).
    )
    _save_all(outdir, "38_02_affine", affine)                            # Save affine-warped image.

    # 3) Perspective: map 4 source corners of an inner quadrilateral to full image rectangle  # Perspective warp.
    var persp = vision.perspective(                                      # Compute/apply 4-point perspective transform.
        img,                                                             # Source image.
        ((40.0, 40.0),                                                   # Source quad: top-left (inset from border).
         (Float64(w) - 40.0, 40.0),                                      # Source quad: top-right.
         (Float64(w) - 60.0, Float64(h) - 60.0),                         # Source quad: bottom-right (more inset).
         (60.0, Float64(h) - 60.0)),                                     # Source quad: bottom-left.
        ((0.0, 0.0),                                                     # Destination quad: full image top-left.
         (Float64(w) - 1.0, 0.0),                                        # Destination quad: top-right (max x).
         (Float64(w) - 1.0, Float64(h) - 1.0),                           # Destination quad: bottom-right (max x,y).
         (0.0, Float64(h) - 1.0)),                                       # Destination quad: bottom-left (max y).
        vision.BORDER_REFLECT()                                          # Border handling: reflect.
    )
    _save_all(outdir, "39_03_perspective", persp)                        # Save perspective-warped image.

# ----------------------- Runnable -----------------------

fn main() -> None:                                                       # Program entry point.
    var outdir = "outputs_geometric_transforms"                          # Directory where outputs will be written.
    ensure_outdir(outdir)                                                # Ensure the output directory exists.

    var input_path = outdir + "/input_demo.png"                          # Intended input path (also used for demo save).
    var img = load_image(input_path)                                     # Load existing image or generate a demo image.

    geometric_transforms(img, outdir)                                    # Run the transforms and save all results.

    show_if(False, 1200, "Geometric Transforms", img)                    # Optional window preview (currently disabled).
