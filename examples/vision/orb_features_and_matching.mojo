# Project:      Momijo
# Module:       examples.orb_matching
# File:         orb_matching.mojo
# Path:         src/momijo/examples/orb_matching.mojo
#
# Description:  ORB keypoint detection and brute-force Hamming matching demo
#               (rotation robustness) with simple I/O helpers.
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
#   - Demonstrates: directory setup, safe image load with fallback,
#     ORB detect+compute, rotation, BF matching, draw_matches.

import momijo.vision as vision                 # Import the vision module under a short alias for image operations.

from pathlib import Path                       # Import Path for file system path handling.
from os import makedirs                        # Import recursive directory creation.
from os import mkdir                           # Import single-level directory creation.
# from collections.list import List            # (Removed) Not used in this file; keeping imports minimal per standards.

# ----------------------- Utility Helpers -----------------------

fn ensure_outdir(outdir: String) -> None:      # Ensure the output directory exists; create it if missing.
    var p = Path(outdir)                       # Wrap the provided string as a Path.
    if not p.exists():                         # Check whether the directory already exists.
        try:                                   # Guard to avoid raising if a race or permission issue occurs.
            makedirs(String(p))                # Recursively create the directory tree.
        except _:                              # On any error (already exists, permissions), continue without failing.
            pass                               # Intentionally ignore errors for demo resilience.

fn _save_all(outdir: String, stem: String, img: vision.Image) -> None:  # Save an image as outdir/stem.png; log outcome.
    try:                                       # Attempt to write the image.
        vision.write_image(outdir + "/" + stem + ".png", img)  # Write PNG using the given stem.
    except e:                                  # If an error happens during write (I/O, codec), catch it.
        print(e)                               # Log the error to stdout.
    print("Saved: " + outdir + "/" + stem + ".png")  # Confirm the saved path (fixed from ".{png}" to ".png").

fn save(outdir: String, name: String, img: vision.Image) -> String:  # Save image to a specific name; return full path.
    var path = outdir + "/" + name             # Construct the full path string.
    try:                                       # Try to persist the image.
        vision.write_image(path, img)          # Write image to disk (format inferred from extension).
    except e:                                  # Catch any exception raised by write_image.
        print(e)                               # Print the error for visibility.
    print("Saved: " + path)                    # Log the saved path.
    return path                                # Return the saved file path to the caller.

fn show_if(show: Bool, pause_ms: Int, title: String, img: vision.Image) -> None:  # Conditionally show an image in a window.
    if show and vision.supports_windows():     # Only attempt GUI display if requested and supported.
        vision.imshow(title, img)              # Create/update a named window with the image.
        vision.wait_key(pause_ms)              # Wait for a key press or timeout (milliseconds).

# ----------------------- Demo / Safe Load -----------------------

fn make_demo_image() -> vision.Image:          # Create a synthetic demo image suitable for ORB features.
    var w = 500                                # Desired width of the demo image in pixels.
    var h = 350                                # Desired height of the demo image in pixels.
    var img = vision.zeros(h, w, 3, vision.UInt8())   # Allocate an HxWx3 uint8 BGR image initialized to zeros.
    img = vision.fill(img, [UInt8(60), UInt8(60), UInt8(70)])  # Fill background with a dark neutral color.

    img = vision.rectangle(                    # Draw a solid white rectangle to provide corners/edges for ORB.
        img, 40, 40, 180, 160, (UInt8(255), UInt8(255), UInt8(255)), -1
    )
    img = vision.circle(                       # Draw a solid red circle (BGR order) for additional features.
        img, 360, 120, 60, [UInt8(0), UInt8(0), UInt8(255)], -1
    )
    
    try:                                       # Draw some text; this can fail if fonts are not available.
        img = vision.put_text(                 # Render the label "ORB Demo" near the bottom.
            img, "ORB Demo",                   # Target image and text string.
            140, 300,                          # Text origin (x, y) baseline coordinates.
            vision.FONT_SIMPLEX(),             # Font face (simplex).
            1.0,                               # Font scale.
            [UInt8(200), UInt8(200), UInt8(255)],  # Text color (light BGR).
            2,                                 # Stroke thickness in pixels.
            True                               # Anti-aliasing enabled.
        )
    except e:                                  # If rendering fails, catch and continue.
        print(e)                               # Log the error but keep the demo usable.
    return img.copy()                          # Return a defensive copy to avoid shared ownership issues.

fn load_image(path: String) -> vision.Image:   # Load an image from disk; if not present/invalid, create and save a demo.
    # Strategy: ensure parent directories exist, try reading; on failure, generate demo, save, and return it.

    var p = Path(path)                         # Wrap the requested file path as a Path.

    if not p.exists():                         # If the file does not exist yet, try to ensure directories exist.
        try:                                   # First attempt: recursive mkdir (best effort).
            makedirs(String(p))                # Note: if 'p' points to a file, this may fail; we ignore errors below.
        except _:                              # Ignore errors from recursive mkdir.
            try:                               # Second attempt: create a single directory level if applicable.
                mkdir(String(p))               # May still fail; safe to ignore.
            except _:                          # Swallow any exception to keep flow simple.
                pass                           # No-op.

    if p.exists():                             # If the path now exists, try to read it as an image.
        try:
            var img = vision.read_image(path)  # Read image; API typically returns (status, image) or similar tuple.
            if img[1].width() > 0 and img[1].height() > 0:  # Validate non-zero dimensions.
                return img[1].copy()           # Return a copy of the successfully loaded image.
            # If invalid, fall through to demo generation below.
        except _:                              # On read failure (corrupt/unsupported), proceed to demo fallback.
            pass

    var demo = make_demo_image()               # Create a demo image as a fallback/default.
    try:
        vision.write_image(path, demo)         # Attempt to persist the demo at the requested path for user visibility.
    except _:                                  # Ignore write errors (permissions, invalid directory).
        pass                                   # Still return a valid image.
    return demo.copy()                         # Return the demo image copy.

# ----------------------- 12) ORB Features & Matching -----------------------

fn orb_features_and_matching(img: vision.Image, outdir: String) -> None:  # Run ORB detection and BF matching workflow.
    var gray = vision.bgr_to_gray(img)         # Convert source image to grayscale (ORB operates on single channel).
    _save_all(outdir, "45_00_gray", gray)      # Save grayscale image for inspection.

    var odac1 = vision.orb_detect_and_compute(gray, 500)  # Detect up to 500 ORB keypoints and compute descriptors.
    var kp1 = odac1[0].copy()                   # Extract keypoints (copy to own storage).
    var des1 = odac1[1].copy()                  # Extract descriptors (copy to own storage).

    var rotated = vision.rotate(                # Rotate the original image to test rotation robustness.
        img,                                    # Source image to rotate.
        25.0,                                   # Angle in degrees (positive = counterclockwise).
        1.0,                                    # Scale factor (1.0 keeps original size).
        Float64(img.width()) / 2.0,             # Rotation center x at image center.
        Float64(img.height()) / 2.0,            # Rotation center y at image center.
        vision.BORDER_REFLECT()                 # Border mode to fill exposed areas.
    )
    _save_all(outdir, "45_01_rotated", rotated) # Save the rotated image.

    var gray2 = vision.bgr_to_gray(rotated)     # Convert rotated image to grayscale.
    _save_all(outdir, "45_02_rotated_gray", gray2)  # Save rotated grayscale.

    var odac2 = vision.orb_detect_and_compute(gray2, 500)  # Detect+compute on rotated version.
    var kp2 = odac2[0].copy()                   # Extract rotated keypoints.
    var des2 = odac2[1].copy()                  # Extract rotated descriptors.

    if vision.valid_descriptors(des1)           # Ensure descriptor buffers are valid (non-empty, correct type).
       and vision.valid_descriptors(des2)       # Validate descriptors for the rotated image as well.
       and (vision.len_keypoints(kp1) > 0)      # Ensure we have at least one keypoint in the original.
       and (vision.len_keypoints(kp2) > 0):     # Ensure we have at least one keypoint in the rotated.
        var matches_i32 = vision.bf_match_hamming(des1, des2, True)  # Cross-checked BF match with Hamming distance.
        var matches = vision.to_int_triples(matches_i32)  # Convert to (queryIdx, trainIdx, distance) int triples.
        matches = vision.top_k_matches(matches, 50)       # Keep top-50 matches by score for clearer visualization.

        var matched = vision.draw_matches(      # Draw matched keypoints between original and rotated images.
            img, kp1, rotated, kp2, matches
        )
        _save_all(outdir, "45_03_orb_matches", matched)  # Save the match visualization.
    else:                                     # If prerequisites are not met, skip matching gracefully.
        print("[INFO] ORB: not enough keypoints/descriptors.")  # Diagnostic message.

# ----------------------- Runnable -----------------------

fn main() -> None:                              # Program entry point.
    var outdir = "outputs_orb_matching"         # Choose an output directory for generated artifacts.
    ensure_outdir(outdir)                       # Ensure that the directory exists.

    var input_path = outdir + "/input_demo.png" # Define an input image path under the output directory.
    var img = load_image(input_path)            # Load existing image or create and save a demo if missing/invalid.

    orb_features_and_matching(img, outdir)      # Run the ORB + matching pipeline and save outputs.

    show_if(False, 1200, "ORB Matching", img)   # Optionally display the image (disabled by default).
