# Project:      Momijo
# Module:       examples.morphology_contours           # Logical module name within the examples package.
# File:         morphology_contours.mojo               # Source file name.
# Path:         src/momijo/examples/morphology_contours.mojo   # Full path from repository root.
#
# Description:  Demo of morphology and contour operations with robust I/O helpers.
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
#   - Shows: grayscale + Otsu, erode/dilate/open/close/gradient, contour stats, connected components.

import momijo.vision as vision                 # Import the vision module for image processing operations.
from pathlib import Path                       # Path abstraction for filesystem usage.
from os import makedirs                        # Recursive directory creation (creates parents as needed).
from os import mkdir                           # Single-level directory creation.

# ----------------------- Utility Helpers -----------------------
fn ensure_outdir(outdir: String) -> None:      # Ensure an output directory exists; create if missing.
    var p = Path(outdir)                       # Wrap the string path in a Path object.
    if not p.exists():                         # If it does not exist, try to create it.
        try:                                   # Guard against exceptions to keep demo resilient.
            makedirs(String(p))                # Create directory tree recursively.
        except _:                              # On any failure (permissions, race, etc.), ignore.
            pass                               # No-op; caller can still proceed.

fn _save_all(outdir: String, stem: String, img: vision.Image) -> None:  # Save an image as outdir/stem.png.
    try:                                       # Attempt to write the image to disk.
        vision.write_image(outdir + "/" + stem + ".png", img)  # Format inferred from extension.
    except e:                                  # If writing fails (codec, I/O), report but continue.
        print(e)                               # Print the error for visibility.
    print("Saved: " + outdir + "/" + stem + ".png")  # Log the saved path (fixed extension message).

fn save(outdir: String, name: String, img: vision.Image) -> String:  # Save with explicit name; return full path.
    var path = outdir + "/" + name             # Join directory and filename.
    try:                                       # Attempt to write the image file.
        vision.write_image(path, img)          # Use vision I/O to persist the image.
    except e:                                  # Catch and log any error instead of throwing.
        print(e)                               # Print the exception for debugging.
    print("Saved: " + path)                    # Confirm the output path for the user.
    return path                                 # Return the path to the caller.

fn show_if(show: Bool, pause_ms: Int, title: String, img: vision.Image) -> None:  # Conditionally show a window.
    if show and vision.supports_windows():     # Only show if requested and the platform supports windows.
        vision.imshow(title, img)              # Display the image in a window titled 'title'.
        vision.wait_key(pause_ms)              # Wait for a key event or timeout in milliseconds.

# ----------------------- Demo / Safe Load -----------------------
fn make_demo_image() -> vision.Image:          # Build a synthetic image that yields clean contours.
    # Create a synthetic image with shapes that will produce good contours
    var w = 600                                # Width of the canvas in pixels.
    var h = 400                                # Height of the canvas in pixels.
    var img = vision.zeros(h, w, 3, vision.UInt8())  # Allocate HxWx3 uint8 image initialized to zeros (BGR).
    img = vision.fill(img, [UInt8(32), UInt8(32), UInt8(36)])  # Fill with a dark BGR background.

    img = vision.rectangle(                    # Draw a solid white rectangle (good sharp edges).
        img, 50, 50, 200, 200, (UInt8(255), UInt8(255), UInt8(255)), -1)
    img = vision.circle(                       # Draw a solid red circle (BGR: [0,0,255]).
        img, 350, 150, 60, [UInt8(0), UInt8(0), UInt8(255)], -1)
    img = vision.rectangle(                    # Draw a solid green rectangle (BGR: [0,255,0]).
        img, 250, 250, 500, 350, (UInt8(0), UInt8(255), UInt8(0)), -1)

    try:                                       # Draw text; may fail if font backend is unavailable.
        img = vision.put_text(                 # Render demo label onto the image.
            img, "Contours Demo",              # Text string content.
            120, 380,                          # Baseline origin (x, y).
            vision.FONT_SIMPLEX(),             # Font face or factory.
            0.8,                               # Font scale factor.
            [UInt8(220), UInt8(220), UInt8(255)],  # Light BGR color for visibility.
            2,                                 # Stroke thickness.
            True                               # Enable anti-aliasing if supported.
        )
    except e:                                  # If rendering fails, do not abort the demo.
        print(e)                               # Log the error for diagnostics.
    return img.copy()                          # Return a copy to avoid shared-ownership pitfalls.

fn load_image(path: String) -> vision.Image:   # Load an image; fallback to demo image on failure.
    # Ensure parent directory exists; then try to read the image.
    # If reading fails or the image is invalid, generate a demo image, save, and return it.

    var p = Path(path)                         # Wrap the path string for filesystem checks.

    if not p.exists():                         # If the file does not exist, try to ensure the directory.
        # Best-effort: try recursive create, then single-level, ignore errors
        try:
            makedirs(String(p))                # Attempt to create parent directories recursively.
        except _:
            try:
                mkdir(String(p))               # Fallback single-level directory creation.
            except _:
                pass                           # Ignore errors to keep behavior non-throwing.

    # 2) If file exists, try to read it
    if p.exists():                             # If the target exists, attempt to load it.
        try:
            var img = vision.read_image(path)  # Read image; API may return a tuple like (ok, image).
            # Sanity check: reject zero-sized images
            if img[1].width() > 0 and img[1].height() > 0:  # Validate the loaded image dimensions.
                return img[1].copy()           # Return a safe copy of the image.
            # Otherwise fall through to demo generation
        except _:
            # Fall through to demo generation on any error
            pass

    # 3) Fallback: generate demo image and try to save it
    var demo = make_demo_image()               # Build a synthetic fallback image.
    try:
        vision.write_image(path, demo)         # Persist demo to the requested path for user visibility.
    except _:
        # Ignore write errors; still return a valid image
        pass

    return demo.copy()                         # Return the fallback image regardless of write success.

# ----------------------- 6) Morphology & Contours -----------------------
# Each transformation is saved immediately.
fn morphology_and_contours(img: vision.Image, outdir: String) -> None:  # Run morphology and contour pipeline.
    # 0) Grayscale + Otsu threshold
    var gray = vision.bgr_to_gray(img)        # Convert input BGR image to single-channel grayscale.
    _save_all(outdir, "21_00_gray", gray)     # Save the grayscale result.

    var bw = vision.threshold_otsu(gray)      # Apply Otsu automatic threshold to get a binary image.
    _save_all(outdir, "21_01_otsu_bw", bw)    # Save the thresholded binary image.

    # 1) Morphological operations
    var kernel = vision.ones_u8(5, 5)         # Create a 5x5 uint8 kernel of ones for morphology ops.

    var erode = vision.erode(bw, kernel, 1)   # Erode: shrink foreground (remove noise).
    _save_all(outdir, "21_02_erode", erode)   # Save erosion result.

    var dilate = vision.dilate(bw, kernel, 1) # Dilate: grow foreground (fill gaps).
    _save_all(outdir, "21_03_dilate", dilate) # Save dilation result.

    var open_ = vision.morphology(bw, vision.MORPH_OPEN(), kernel)   # Open: erode then dilate (remove small noise).
    _save_all(outdir, "21_04_open", open_)                           # Save opening result.

    var close = vision.morphology(bw, vision.MORPH_CLOSE(), kernel)  # Close: dilate then erode (close small holes).
    _save_all(outdir, "21_05_close", close)                          # Save closing result.

    var grad = vision.morphology(bw, vision.MORPH_GRADIENT(), kernel)  # Morphological gradient: edges of objects.
    _save_all(outdir, "21_06_gradient", grad)                          # Save gradient result.

    # 2) Contour detection and drawing
    var contours = vision.find_contours(bw, external_only=True)  # Extract contours; external_only limits to outer ones.
    var cont_vis = vision.draw_contours(                         # Visualize contours over the original image.
        img, contours, vision.bgr_u8(0, 0, 255), 2)              # Draw in red (BGR) with thickness 2.
    _save_all(outdir, "21_07_contours_drawn", cont_vis)          # Save contour visualization.

    # 3) Bounding boxes + stats
    var cont_stats = img.copy()                  # Create a working copy to draw per-contour annotations.
    var i = 0                                    # Loop counter over contours.
    while i < vision.len_contours(contours):     # Iterate all detected contours.
        var c = vision.get_contour(contours, i)  # Fetch the i-th contour handle/object.
        var (x, y, w, h) = vision.bounding_rect(c)  # Axis-aligned bounding rectangle around the contour.
        var area = vision.contour_area(c)        # Compute contour area (may return a tuple or status).
        var peri = vision.arc_length(c, True)    # Compute contour perimeter; True ⇒ closed contour.

        cont_stats = vision.rectangle(           # Draw bounding box in cyan for visibility.
            cont_stats, x, y, x + w, y + h,
            [UInt8(0), UInt8(255), UInt8(255)], 2)

        var label = "A:" + String(Float64(area[0])) + " P:" + String(Float64(peri[0]))  # Build a text label.

        try:                                     # Attempt to render the label above the bounding box.
            cont_stats = vision.put_text(
                cont_stats,                      # Target image for annotation.
                label,                           # Label text (area and perimeter).
                x, y - 5,                        # Position slightly above the rectangle.
                vision.FONT_PLAIN(),             # Simpler font for small text.
                0.7,                             # Font scale (smaller).
                [UInt8(0), UInt8(255), UInt8(255)], 1, True)  # Cyan text, thin stroke, anti-aliased.
        except e:                                # If font rendering fails, keep going.
            print(e)                             # Log the error for debugging.

        i += 1                                   # Advance loop index.
    _save_all(outdir, "21_08_contour_stats", cont_stats)  # Save the annotated contour stats image.

    # 4) Connected components visualization
    var cc_out = vision.connected_components(bw)  # Label connected regions in the binary image.
    var num = cc_out[0]                           # Number of components (including background).
    var labels = cc_out[1].copy()                 # Label image where each pixel stores a component id.
    var colored = vision.label_to_color_image(labels, num)  # Map labels to distinct colors for visualization.
    _save_all(outdir, "21_09_connected_components", colored)  # Save the colored components image.

# ----------------------- Runnable -----------------------
fn main() -> None:                             # Entry point for this demo program.
    var outdir = "outputs_morphology_contours" # Output directory for generated artifacts.
    ensure_outdir(outdir)                      # Create the output directory if needed.

    var input_path = outdir + "/input_demo.png"  # Target path of the input (will be created if missing).
    var img = load_image(input_path)           # Load the image or generate/save a demo fallback.

    morphology_and_contours(img, outdir)       # Run the full pipeline and save each intermediate result.

    show_if(False, 1200, "Morphology & Contours", img)  # Optional GUI preview (disabled here).
