# Project:      Momijo
# Module:       examples.dataframe_string_ops
# File:         dataframe_string_ops.mojo
# Path:         src/momijo/examples/dataframe_string_ops.mojo
#
# Description:  Teaching demo of common string operations in momijo.dataframe
#               (title/upper/contains/slice/regex replace) and assign().
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
#   - Demonstrates: df.str_title/upper/contains/slice/replace_regex and df.assign.
#   - Avoids mixed-typed Dict by passing a literal mapping directly to df.assign.

import momijo.dataframe as df                # Import the dataframe API with a short alias.
from pathlib import Path                     # Path abstraction for filesystem operations.
from os import makedirs                      # Recursive directory creation utility.
from os import mkdir                         # Single-level directory creation utility.

from collections.list import List            # Typed dynamic array for demo data vectors.

# ---------- Utilities ----------

fn print_section(title: String) -> None:     # Pretty-print a visible section header.
    var line = String("=") * 80              # Build a divider line of '=' signs.
    print("\n" + line)                       # Leading newline + top divider.
    print(title)                             # The section title text.
    print(line)                              # Bottom divider to frame the title.

fn workdir_make() -> Path:                   # Ensure a working directory exists and return it.
    # Simple temp-like folder next to CWD for demo artifacts
    var p = Path("./_momijo_demo_outputs")   # Choose a deterministic outputs folder.
    if not p.exists():                       # If the path is missing, create it.
        try:                                 # First attempt: recursive mkdir (creates parents).
            # Try recursive create (parents)
            makedirs(String(p))              # Convert Path to String for OS call.
        except _:                            # On failure, try a simpler mkdir.
            # Fallback: single-level mkdir
            try:
                mkdir(String(p))             # Single-level directory creation.
            except _:                        # If it still fails, ignore to keep the demo resilient.
                # Ignore errors quietly for teaching demo
                pass
    print("[INFO] Working directory: " + String(p))  # Log the chosen folder for transparency.
    return p                                 # Return the Path even if creation was skipped/failed.

# ---------- 10) String Ops ----------

fn strings_ops(frame: df.DataFrame) -> df.DataFrame:   # Apply a set of string transformations and return a new frame.
    print_section("10) String Ops")        # Show a banner for this demo section.

    # Build new columns via assign() with a literal mapping to avoid mixed-typed Dict issues.
    var out = df.assign(                    # Return a new DataFrame with appended/overwritten columns.
        frame,                              # Base DataFrame we are augmenting.
        {
            # Title-case the 'city' strings (e.g., "helsinki" → "Helsinki").
            "city_title": df.str_title(frame, "city"),
            # Upper-case the 'city' strings (e.g., "Helsinki" → "HELSINKI").
            "city_upper": df.str_upper(frame, "city"),
            # Boolean vector: whether 'city' contains "u" (case-insensitive, NA treated as false).
            "city_has_u": df.str_contains(frame, "city", "u", case_insensitive=True, na_false=True),
            # First three characters of 'city' (prefix), safe slicing semantics.
            "city_prefix": df.str_slice(frame, "city", 0, 3),
            # First character of 'name' (initial).
            "name_initial": df.str_slice(frame, "name", 0, 1),
            # Regex replace vowels with '_' in 'city' (teaching example of regex ops).
            "city_replaced": df.str_replace_regex(frame, "city", "[aeiou]", "_")
        }
    )

    print(df.head(out, 6).to_string())     # Preview the first six rows of the augmented DataFrame.
    return out                              # Return the transformed DataFrame to the caller.

# ---------- main ----------

fn main() -> None:                          # Program entry point.
    var _ = workdir_make()                  # Ensure the working directory exists (and log it).

    # Demo frame: names and cities (with vowels for regex test)
    var names  : List[String]  = ["Alice","Bob","Cathy","Dan","Eve","Frank"]   # Example names.
    var ages   : List[Int]     = [25,31,29,40,22,35]                           # Example ages.
    var cities : List[String]  = ["Helsinki","Turku","Tampere","Oulu","Espoo","Helsinki"]  # Cities.
    var scores : List[Float64] = [88.5,75.0,92.0,66.0,79.0,85.5]               # Scores.
    var groups : List[String]  = ["A","B","A","B","A","B"]                     # Group labels.

    var pairs = df.make_pairs()                 # Start a (name, values) container for building a DataFrame.
    pairs = df.pairs_append(pairs, "name",  names)   # Append 'name' column.
    pairs = df.pairs_append(pairs, "age",   ages)    # Append 'age' column.
    pairs = df.pairs_append(pairs, "city",  cities)  # Append 'city' column.
    pairs = df.pairs_append(pairs, "score", scores)  # Append 'score' column.
    pairs = df.pairs_append(pairs, "group", groups)  # Append 'group' column.

    var frame = df.df_from_pairs(pairs)        # Build a DataFrame from the accumulated pairs.
    print_section("Input DataFrame")           # Section header for the input preview.
    print(df.head(frame, 6).to_string())       # Show the first six rows of input data.
    print("dtypes:\n" + df.df_dtypes(frame))   # Display column dtypes for quick verification.

    var out = strings_ops(frame)               # Apply string operations to create derived columns.

    print_section("Final DataFrame (string ops)")  # Header for the final result preview.
    print(df.head(out, 6).to_string())         # Show the first six rows of the final augmented DataFrame.
