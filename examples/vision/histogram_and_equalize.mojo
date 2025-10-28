# Project:      Momijo
# Module:       examples.geometric_transforms               # Logical module path inside the package.
# File:         geometric_transforms.mojo                   # Source file name.
# Path:         src/momijo/examples/geometric_transforms.mojo  # Full path from repository root per packaging rules.
#
# Description:  Demo of geometric transforms (translate, affine, perspective) with image I/O helpers.
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
#   - Shows robust output dir creation, safe image loading with fallback to synthetic demo.
#   - Saves each transformation step for easy inspection.

import momijo.vision as vision                 # Import vision API with a short alias for image operations.
from pathlib import Path                       # Filesystem path abstraction for portability and clarity.
from os import makedirs                        # Recursive directory creation (creates parents as needed).
from os import mkdir                           # Single-level directory creation (no parents).
from collections.list import List              # Typed dynamic array for polygon point storage.

# ----------------------- Utility Helpers -----------------------

fn ensure_outdir(outdir: String) -> None:      # Ensure an output directory exists (best-effort).
    var p = Path(outdir)                       # Convert string to Path for existence checks and creation.
    if not p.exists():                         # Only attempt creation if it is not present.
        try:                                   # Guard against errors (permissions/races).
            makedirs(String(p))                # Recursively create the path; ok if it already exists.
        except _:                              # Swallow any exception to keep examples resilient.
            pass                               # Intentionally ignore errors.

fn _save_all(outdir: String, stem: String, img: vision.Image) -> None:  # Save image as outdir/stem.png, log result.
    try:                                                           # Guard the write to avoid aborting on I/O errors.
        vision.write_image(outdir + "/" + stem + ".png", img)      # Write image (codec inferred by extension).
    except e:                                                      # On failure, capture the exception object.
        print(e)                                                   # Print error details for debugging.
    print("Saved: " + outdir + "/" + stem + ".png")                # Log the final saved path (fixed extension log).

fn save(outdir: String, name: String, img: vision.Image) -> String:   # Convenience: save with an explicit file name.
    var path = outdir + "/" + name                                   # Compose a full path by simple concatenation.
    try:                                                              # Guard the write operation.
        vision.write_image(path, img)                                 # Persist the image to disk.
    except e:                                                         # On error, do not throw from demo code.
        print(e)                                                      # Log the exception to stderr.
    print("Saved: " + path)                                           # Inform the user about the saved path.
    return path                                                       # Return the written path to the caller.

fn show_if(show: Bool, pause_ms: Int, title: String, img: vision.Image) -> None:  # Conditionally display a window.
    if show and vision.supports_windows():                            # Only attempt if display is supported and wanted.
        vision.imshow(title, img)                                     # Show an image window with the given title.
        vision.wait_key(pause_ms)                                     # Block for a short period to let the user view.

# ----------------------- Demo / Safe Load -----------------------

fn make_demo_image() -> vision.Image:                                 # Create a synthetic image with strong features.
    # High-contrast scene for geometric transforms                    # Rationale for chosen primitives and colors.
    var w = 720                                                       # Canvas width in pixels.
    var h = 480                                                       # Canvas height in pixels.
    var img = vision.zeros(h, w, 3, vision.UInt8())                   # Allocate HxWx3 uint8 image initialized to zero.
    img = vision.fill(img, [UInt8(30), UInt8(30), UInt8(35)])         # Fill with a dark BGR background.

    img = vision.rectangle(img, 60, 60, 300, 220, (UInt8(0), UInt8(255), UInt8(255)), -1)  # Solid cyan rectangle.
    img = vision.circle(img, 520, 160, 70, [UInt8(0), UInt8(0), UInt8(255)], -1)           # Solid red disk.
    img = vision.arrowed_line(                                                           # White arrow for direction cue.
        img, 80, 300, 300, 380, [UInt8(255), UInt8(255), UInt8(255)], 3, 0.15)

    try:                                                                                  # Text rendering may fail on fonts.
        # Text edges                                                                      # Intent: add edges for affine/persp tests.
        img = vision.put_text(                                                            # Draw overlay text.
            img,                                                                          # Target image.
            "Geometric Transform",                                                        # Content string.
            180, 360,                                                                     # Baseline origin (x, y).
            vision.FONT_SIMPLEX(),                                                        # Font face.
            1.2,                                                                          # Font scale.
            [UInt8(220), UInt8(220), UInt8(220)],                                         # Light gray BGR color.
            2,                                                                            # Thickness in pixels.
            True                                                                          # Anti-aliased rendering.
        )
    except e:                                                                              # Font or rendering backend may be missing.
        print(e)                                                                           # Report the issue, keep going.
    return img.copy()                                                                      # Return a defensive copy to avoid aliasing.

fn load_image(path: String) -> vision.Image:                                               # Robust loader with demo fallback.
    # Ensure parent directory exists; then try to read the image.                          # High-level behavior summary.
    # If reading fails or the image is invalid, generate a demo image, save, and return it.# Guarantees a valid result.

    var p = Path(path)                                                                     # Wrap the path string.

    if not p.exists():                                                                      # If the target path doesn't exist:
        # Best-effort: try recursive create, then single-level, ignore errors               # Two-pass creation strategy.
        try:                                                                                # First attempt: recursive parents.
            makedirs(String(p))                                                             # May fail if 'p' includes a file.
        except _:                                                                           # Ignore any error.
            try:                                                                            # Second attempt: flat mkdir.
                mkdir(String(p))                                                            # Works if 'p' is a directory path.
            except _:                                                                       # Ignore again to keep function robust.
                pass                                                                        # No-op.

    # 2) If file exists, try to read it                                                     # Proceed only if something is at 'p'.
    if p.exists():                                                                          # Check again after potential creation.
        try:                                                                                # Guard the read.
            var img = vision.read_image(path)                                               # Attempt to load; API may return tuple.
            # Sanity check: reject zero-sized images                                        # Validate image before using it.
            if img[1].width() > 0 and img[1].height() > 0:                                  # Confirm dimensions are positive.
                return img[1].copy()                                                        # Return a safe copy of the image.
            # Otherwise fall through to demo generation                                     # If invalid, continue to fallback.
        except _:                                                                           # Catch any decoding/IO error.
            # Fall through to demo generation on any error                                  # Intentionally proceed to fallback.
            pass

    # 3) Fallback: generate demo image and try to save it                                   # Ensure caller gets a valid image.
    var demo = make_demo_image()                                                            # Create synthetic content.
    try:                                                                                   # Try to persist for reproducibility.
        vision.write_image(path, demo)                                                      # Write demo to the expected location.
    except _:                                                                               # Permissions or path issues can occur.
        # Ignore write errors; still return a valid image                                   # The function promise is to return an image.
        pass

    return demo.copy()                                                                      # Return a copy of the demo image.

# ----------------------- 9) Geometric Transforms (Affine / Perspective) -----------------------
# Every modification is saved immediately.                                                   # Each step produces an output file.

fn geometric_transforms(img: vision.Image, outdir: String) -> None:                         # Apply and save transforms.
    # 0) Original                                                                            # Keep a reference output for comparison.
    _save_all(outdir, "36_00_original_bgr", img)                                            # Save the unmodified input.

    var h = img.height()                                                                     # Cache height to avoid repeated calls.
    var w = img.width()                                                                      # Cache width likewise.

    # 1) Translation: shift right/down by (tx=40, ty=30)                                     # Simple translation example.
    #    Border mode uses reflection (BS variant if supported in your lib).                  # Avoid black borders by reflect.
    var trans = vision.translate(img, 40, 30, vision.BORDER_REFLECT_BS())                    # Translate with reflective border.
    _save_all(outdir, "37_01_translate", trans)                                              # Save translated result.

    # 2) Affine: map 3 source points to 3 destination points                                 # Non-uniform linear transform.
    #    src: (0,0), (w-1,0), (0,h-1)                                                        # Reference triangle at image corners.
    #    dst: slight skew + shift                                                            # Adds shear and offset.
    var affine = vision.affine(                                                              # Compute/apply affine transform.
        img,                                                                                 # Source image.
        ((0.0, 0.0), (Float64(w) - 1.0, 0.0), (0.0, Float64(h) - 1.0)),                      # 3 source points.
        ((0.0, 0.0), (Float64(w) * 0.9, 40.0), (40.0, Float64(h) * 0.95)),                   # 3 destination points.
        vision.BORDER_REFLECT()                                                              # Border handling (reflect).
    )
    _save_all(outdir, "38_02_affine", affine)                                                # Save affine result.

    # 3) Perspective: map 4 source corners of an inner quadrilateral to full image rectangle # 4-point homography warp.
    var persp = vision.perspective(                                                          # Compute/apply perspective warp.
        img,                                                                                 # Source image.
        ((40.0, 40.0),                                                                       # Source quad: top-left.
         (Float64(w) - 40.0, 40.0),                                                          # Top-right.
         (Float64(w) - 60.0, Float64(h) - 60.0),                                             # Bottom-right (inset).
         (60.0, Float64(h) - 60.0)),                                                         # Bottom-left (inset).
        ((0.0, 0.0),                                                                         # Destination rect: top-left.
         (Float64(w) - 1.0, 0.0),                                                            # Top-right (max width).
         (Float64(w) - 1.0, Float64(h) - 1.0),                                               # Bottom-right (max extents).
         (0.0, Float64(h) - 1.0)),                                                           # Bottom-left.
        vision.BORDER_REFLECT()                                                              # Border handling (reflect).
    )
    _save_all(outdir, "39_03_perspective", persp)                                            # Save perspective result.

# ----------------------- Runnable -----------------------

fn main() -> None:                                                                           # Program entry point.
    var outdir = "outputs_geometric_transforms"                                              # Output directory for all results.
    ensure_outdir(outdir)                                                                    # Create it if missing.

    var input_path = outdir + "/input_demo.png"                                              # Location where input is expected.
    var img = load_image(input_path)                                                         # Load input or synthesize fallback.

    geometric_transforms(img, outdir)                                                        # Run and save all transform demos.

    show_if(False, 1200, "Geometric Transforms", img)                                        # Optional on-screen preview (disabled).
