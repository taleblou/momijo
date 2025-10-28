# Project:      Momijo
# Module:       examples.dataframe_selection_filtering
# File:         dataframe_selection_filtering.mojo
# Path:         src/momijo/examples/dataframe_selection_filtering.mojo
#
# Description:  Teaching demo for column selection, loc/iloc slicing, and boolean filtering.
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
#   - Demonstrates: df.select, df.loc/df.iloc, comparison masks, isin, mask_and, filter, head, to_string.

import momijo.dataframe as df                 # Import the dataframe module with a short alias for convenience.
from pathlib import Path                      # Path abstraction for robust filesystem path handling.
from os import makedirs                       # Recursive directory creation (creates parent directories).
from os import mkdir                          # Single-level directory creation (fallback if needed).

from collections.list import List             # Typed dynamic array used to build example columns.

# ---------- Utilities ----------

fn print_section(title: String) -> None:      # Pretty-print a section header for console readability.
    var line = String("=") * 80               # Build a horizontal rule consisting of 80 '=' characters.
    print("\n" + line)                        # Print a leading newline and the top rule.
    print(title)                              # Print the section title text provided by the caller.
    print(line)                               # Print the bottom rule to frame the section.

fn workdir_make() -> Path:                    # Ensure/create a working directory for demo artifacts; return its Path.
    # Simple temp-like folder next to CWD for demo artifacts
    var p = Path("./_momijo_demo_outputs")    # Choose a deterministic relative folder for outputs.
    if not p.exists():                        # If the folder does not yet exist, try to create it.
        try:                                  # Attempt a best-effort recursive creation first.
            # Try recursive create (parents)
            makedirs(String(p))               # Create all missing directories in the path.
        except _:                             # On failure (permissions/race), try a simpler mkdir.
            # Fallback: single-level mkdir
            try:
                mkdir(String(p))              # Create a single directory level.
            except _:                         # If it still fails, suppress to keep demo non-fatal.
                # Ignore errors quietly for teaching demo
                pass
    print("[INFO] Working directory: " + String(p))  # Log the resolved outputs folder path.
    return p                                   # Return the Path so callers can use it if desired.

# ---------- 3) Selection & Filtering ----------

fn selection_filtering(frame: df.DataFrame) -> df.DataFrame:  # Showcase selection/slicing/filtering on a DataFrame.
    print_section("3) Selection & Filtering")  # Section marker for this phase of the demo.

    # Column subset
    print("Select columns ['name','age']:\n" +          # Describe the operation for the user.
          df.select(frame, ["name","age"]).to_string()) # Select by explicit column names and render as string.

    # loc slice: rows 1..3, cols ["name","city"]
    var loc_view = df.loc(frame, rows=df.RowRange(1, 3), cols=["name","city"])  # Label-based slice for rows/cols.
    print("loc rows 1..3, cols name/city:\n" + loc_view.to_string())            # Print the loc slice result.

    # iloc by row list and column range
    var iloc_view = df.iloc(frame, row_indices=[0,2,4], col_range=df.ColRange(0, 3))  # Position-based slice.
    print("iloc rows [0,2,4], first 3 cols:\n" + iloc_view.to_string())               # Print the iloc view.

    # Boolean mask: (age >= 30) & city in {Helsinki, Turku}
    var m1 = df.col_ge(frame, "age", 30)                                  # Build mask where 'age' >= 30.
    var m2 = df.col_isin(frame, "city", ["Helsinki","Turku"])              # Build mask where 'city' ∈ set.
    var mask = df.mask_and(m1, m2)                                         # Combine masks with logical AND.
    var filtered = df.filter(frame, mask)                                  # Filter rows using the combined mask.
    print("filtered head():\n" + df.head(filtered, 5).to_string())         # Show first few filtered rows.

    return filtered                                                        # Return the filtered DataFrame.

# ---------- main ----------

fn main() -> None:                                  # Program entry point with no arguments.
    var _ = workdir_make()                          # Ensure the working directory exists (logged above).

    # Sample demo DataFrame
    var names  : List[String]  = ["Alice","Bob","Cathy","Dan","Eve"]       # Example people names.
    var ages   : List[Int]     = [25,31,29,40,22]                          # Example ages (integers).
    var cities : List[String]  = ["Tehran","Karaj","Isfahan","Tabriz","Mashad"]  # Example city names.

    var pairs = df.make_pairs()                         # Initialize a (name, values) accumulator for columns.
    pairs = df.pairs_append(pairs, "name",  names)      # Append the 'name' column.
    pairs = df.pairs_append(pairs, "age",   ages)       # Append the 'age' column.
    pairs = df.pairs_append(pairs, "city",  cities)     # Append the 'city' column.

    var frame = df.df_from_pairs(pairs)                 # Materialize a DataFrame from the column pairs.
    print_section("Input DataFrame")                    # Header for the input preview section.
    print(df.head(frame, 10).to_string())               # Show the first up-to-10 rows of the input data.

    var filtered = selection_filtering(frame)           # Run the selection/filtering demo and capture result.

    print_section("Final filtered DataFrame")           # Header for the final preview section.
    print(filtered.to_string())                         # Print the full filtered DataFrame.
