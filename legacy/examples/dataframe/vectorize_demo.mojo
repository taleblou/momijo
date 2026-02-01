# Project:      Momijo
# Module:       examples.dataframe_vectorize
# File:         dataframe_vectorize.mojo
# Path:         src/momijo/examples/dataframe_vectorize.mojo
#
# Description:  Vectorized labeling demo using boolean masks (where) and numeric binning (cut_numeric).
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
#   - Demonstrates: df.where with a boolean mask, df.cut_numeric for binning, df.assign to add columns.
#   - Input: small mixed-type DataFrame (names, ages, cities, scores, groups).

import momijo.dataframe as df                 # Import the dataframe API as a short alias for readability.
from pathlib import Path                      # Import Path for safe filesystem path handling.
from collections.list import List             # Import List for typed dynamic arrays used as sample data.
from os import makedirs                       # Import recursive directory creation utility.
from os import mkdir                          # Import single-level directory creation as a fallback.

# ---------- Utilities ----------

fn print_section(title: String) -> None:      # Print a framed section header for console readability.
    var line = String("=") * 80               # Create a divider line of a fixed width using '=' chars.
    print("\n" + line)                        # Print a leading newline and the top divider.
    print(title)                              # Print the provided section title text.
    print(line)                               # Print the bottom divider to close the header block.

fn workdir_make() -> Path:                    # Ensure a local working directory exists and return it.
    # Simple temp-like folder next to CWD for demo artifacts
    var p = Path("./_momijo_demo_outputs")    # Choose a deterministic relative path for outputs.
    if not p.exists():                        # If it does not exist, attempt to create it.
        try:                                  # First, try recursive directory creation (parents included).
            makedirs(String(p))               # Convert Path to String for OS call compatibility.
        except _:                             # If recursive creation fails, try a single-level mkdir.
            try:
                mkdir(String(p))              # Attempt to create just one directory level.
            except _:                         # Ignore remaining errors to keep the demo resilient.
                # Ignore errors quietly for teaching demo
                pass
    print("[INFO] Working directory: " + String(p))  # Log the chosen working directory path.
    return p                                   # Return the Path so callers can reuse or join with filenames.

# ---------- 11) Vectorized labels ----------

fn vectorize_demo(frame: df.DataFrame) -> df.DataFrame:  # Build vectorized labels and return an augmented frame.
    print_section("11) Vectorized / map/apply (simplified)")  # Visible banner for this demo step.

    # Label "High"/"Low" by score threshold
    var high = df.col_ge(frame, "score", 85)  # Create a boolean mask where score >= 85.
    var label = df.where(                     # Map mask to strings without loops (vectorized).
        high,                                 # Condition: boolean column (same length as DataFrame).
        then="High",                          # Value for rows where condition is True.
        else_="Low"                           # Value for rows where condition is False.
    )

    # Bin ages into 3 bands
    var bands = df.cut_numeric(               # Bucketize numeric ages into labeled intervals.
        frame,                                # Source DataFrame to read the 'age' column from.
        col="age",                            # Target numeric column to bin.
        bins=[0,25,35,100],                   # Bin edges: [0,25), [25,35), [35,100].
        labels=["<=25","26-35","36+"]         # Labels assigned to each corresponding bin.
    )

    var out = df.assign(                      # Create a new DataFrame with additional derived columns.
        frame,                                # Base DataFrame to augment.
        {                                     # Mapping of new column names to their computed data.
            "score_label": label,             # Add the High/Low label per score threshold.
            "age_band": bands                  # Add the categorical age band per numeric bin.
        }
    )
    print(df.head(out, 6).to_string())        # Preview the first six rows of the augmented frame.
    return out                                 # Return the new DataFrame to the caller.

# ---------- main ----------

fn main() -> None:                            # Program entry point with no arguments.
    var _ = workdir_make()                    # Ensure the working directory exists (and log it).

    # Demo frame with numeric + categorical data
    var names  : List[String]  = ["Alice","Bob","Cathy","Dan","Eve","Frank"]          # Example names.
    var ages   : List[Int]     = [25,31,29,40,22,35]                                   # Example ages.
    var cities : List[String]  = ["Helsinki","Turku","Tampere","Oulu","Espoo","Helsinki"]  # Cities.
    var scores : List[Float64] = [88.5,75.0,92.0,66.0,79.0,85.5]                       # Test scores.
    var groups : List[String]  = ["A","B","A","B","A","B"]                             # Group labels.

    var pairs = df.make_pairs()                # Start a (column-name, values) container for DataFrame build.
    pairs = df.pairs_append(pairs, "name",  names)    # Append the 'name' column.
    pairs = df.pairs_append(pairs, "age",   ages)     # Append the 'age' column.
    pairs = df.pairs_append(pairs, "city",  cities)   # Append the 'city' column.
    pairs = df.pairs_append(pairs, "score", scores)   # Append the 'score' column.
    pairs = df.pairs_append(pairs, "group", groups)   # Append the 'group' column.

    var frame = df.df_from_pairs(pairs)        # Materialize a DataFrame from the accumulated pairs.
    print_section("Input DataFrame")           # Section header for input preview.
    print(df.head(frame, 6).to_string())       # Show the first six rows to confirm the input.

    var out = vectorize_demo(frame)            # Compute vectorized labels and get the augmented DataFrame.

    print_section("Final DataFrame (with labels)")  # Section header for the final preview.
    print(df.head(out, 8).to_string())         # Show the first eight rows including derived columns.
