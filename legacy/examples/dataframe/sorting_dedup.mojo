# Project:      Momijo
# Module:       examples.dataframe_sorting_dedup
# File:         dataframe_sorting_dedup.mojo
# Path:         src/momijo/examples/dataframe_sorting_dedup.mojo
#
# Description:  Demo of sorting by multiple keys and dropping duplicates in momijo.dataframe.
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
#   - Shows df.sort_values(by=[...], ascending=[...]) and df.drop_duplicates(subset=..., keep=...).

import momijo.dataframe as df              # Import the dataframe API under the alias 'df'.
from pathlib import Path                   # Path abstraction for filesystem-safe path handling.
from os import makedirs                    # Recursive directory creation (creates parents).
from os import mkdir                       # Single-level directory creation (fallback).

from collections.list import List          # Typed dynamic array for demo data construction.

# ---------- Utilities ----------
fn print_section(title: String) -> None:   # Print a formatted section header for readability.
    var line = String("=") * 80            # Construct a divider line of '=' characters.
    print("\n" + line)                     # Newline + top divider to separate sections.
    print(title)                           # Print the provided section title text.
    print(line)                            # Bottom divider to frame the title.

fn workdir_make() -> Path:                 # Ensure a working output directory exists; return it.
    # Simple temp-like folder next to CWD for demo artifacts
    var p = Path("./_momijo_demo_outputs") # Choose a deterministic outputs folder relative to CWD.
    if not p.exists():                     # If the folder does not exist yet, try to create it.
        try:                               # First attempt: recursively create parents as needed.
            # Try recursive create (parents)
            makedirs(String(p))            # Convert Path to String for OS call compatibility.
        except _:                          # On failure (permissions, race), try single-level mkdir.
            # Fallback: single-level mkdir
            try:
                mkdir(String(p))           # Attempt to create just one directory level.
            except _:                      # If it still fails, ignore to keep the demo resilient.
                # Ignore errors quietly for teaching demo
                pass
    print("[INFO] Working directory: " + String(p))  # Log the effective working directory.
    return p                              # Return Path for potential downstream use.

# ---------- 9) Sorting & Duplicates ----------
fn sorting_dedup(frame: df.DataFrame) -> df.DataFrame:  # Perform sorting and duplicate removal demos.
    print_section("9) Sorting & Duplicates")            # Visible heading for this step.

    # Sort by city ascending, score descending
    var sorted_df = df.sort_values(                     # Sort rows by multiple keys.
        frame,                                          # Source DataFrame to sort.
        by=["city","score"],                            # Primary key: city; secondary key: score.
        ascending=[True, False]                         # City ↑ ascending; score ↓ descending.
    )
    print("sorted head():\n" + sorted_df.head( 5).to_string())  # Show top 5 rows post-sort.

    # Drop duplicate names (keep first occurrence)
    var dedup = df.drop_duplicates(                     # Remove duplicate rows by a subset of columns.
        frame,                                          # Input DataFrame (original order).
        subset=["name"],                                # Consider duplicates by the 'name' column.
        keep="first"                                    # Keep the first occurrence; drop subsequent ones.
    )
    print("drop_duplicates head():\n" + dedup.head(5).to_string())  # Preview deduplicated result.

    return sorted_df                                    # Return the sorted DataFrame (as function output).

# ---------- main ----------
fn main() -> None:                         # Program entry point.
    var _ = workdir_make()                 # Ensure working directory exists (and log it).

    # Demo frame with intentional duplicate names
    var names  : List[String]  = ["Alice","Bob","Cathy","Dan","Eve","Bob","Alice","Hank"]      # Example names (dupes).
    var ages   : List[Int]     = [25,31,29,40,22,35,25,31]                                      # Example ages.
    var cities : List[String]  = ["Helsinki","Turku","Tampere","Oulu","Espoo","Turku","Helsinki","Turku"]  # Cities.
    var scores : List[Float64] = [88.5,75.0,92.0,66.0,79.0,85.5,70.0,95.0]                      # Scores.
    var groups : List[String]  = ["A","B","A","B","A","B","A","B"]                              # Group labels.

    var pairs = df.make_pairs()            # Start a (column-name, values) container for DataFrame build.
    pairs = df.pairs_append(pairs, "name",  names)   # Add 'name' column.
    pairs = df.pairs_append(pairs, "age",   ages)    # Add 'age' column.
    pairs = df.pairs_append(pairs, "city",  cities)  # Add 'city' column.
    pairs = df.pairs_append(pairs, "score", scores)  # Add 'score' column.
    pairs = df.pairs_append(pairs, "group", groups)  # Add 'group' column.

    var frame = df.df_from_pairs(pairs)    # Construct a DataFrame from the accumulated pairs.
    print_section("Input DataFrame (with duplicates)") # Header for initial data preview.
    print(frame.head( 10).to_string())     # Show first 10 rows (includes intentional duplicates).
    print("dtypes:\n" + df.df_dtypes(frame)) # Display column data types for verification.

    var out = sorting_dedup(frame)         # Run the sorting & dedup demo; capture the sorted DataFrame.

    print_section("Final Sorted DataFrame (head 8)")  # Header for final preview.
    print(out.head(8).to_string())         # Show first 8 rows of the sorted DataFrame.
