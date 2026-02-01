# Project:      Momijo
# Module:       examples.dataframe_groupby_agg
# File:         dataframe_groupby_agg.mojo
# Path:         src/momijo/examples/dataframe_groupby_agg.mojo
#
# Description:  GroupBy/Aggregation and per-group transform (z-score) demo for momijo.dataframe.
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
#   - Demonstrates: df.groupby(..., aggs=...), groupby_transform(..., op="zscore"),
#     df.assign(...) to append derived columns, and pretty-printing with head()/to_string().

import momijo.dataframe as df                 # Import dataframe API under alias 'df' for brevity.
from pathlib import Path                      # Path abstraction for file-system safe paths.
from os import makedirs                       # Recursive directory creation utility.
from os import mkdir                          # Single-level directory creation utility.
from collections.list import List             # Typed dynamic array for example data.

# ---------- Utilities ----------

fn print_section(title: String) -> None:      # Print a bannered section header for console output.
    var line = String("=") * 80               # Build a fixed-width divider line of '=' characters.
    print("\n" + line)                        # Prepend a newline, then the top divider.
    print(title)                              # Print the provided section title.
    print(line)                               # Print the bottom divider.

fn workdir_make() -> Path:                    # Ensure a local working directory exists; return it.
    # Simple temp-like folder next to CWD for demo artifacts
    var p = Path("./_momijo_demo_outputs")    # Choose a deterministic outputs folder.
    if not p.exists():                        # If it does not exist yet, try to create it.
        try:                                  # First attempt: recursive creation (parents as needed).
            # Try recursive create (parents)
            makedirs(String(p))               # Convert Path to String for OS call.
        except _:                             # On failure (permissions, race), try a simpler mkdir.
            # Fallback: single-level mkdir
            try:
                mkdir(String(p))              # Attempt single-level directory creation.
            except _:                         # If it still fails, ignore to keep demo resilient.
                # Ignore errors quietly for teaching demo
                pass
    print("[INFO] Working directory: " + String(p))  # Log the resolved working directory.
    return p                                   # Return the Path for possible downstream use.

# ---------- 6) GroupBy / Agg ----------

fn groupby_agg(frame: df.DataFrame)  -> df.DataFrame:  # Perform groupby aggregations and a per-group transform.
    print_section("6) GroupBy / Agg")        # Visible header for this demo step.

    # Aggregations per city
    var g = df.groupby(                      # Group the frame by one or more keys and compute aggregations.
        frame,                               # Input DataFrame.
        by=["city"],                         # Grouping key: the 'city' column.
        aggs={                               # Aggregations mapping: column → list of operations.
            "age":   ["mean"],               # Average age per city.
            "score": ["max"],                # Maximum score per city.
            "name":  ["count"],              # Row count per city (using 'name' column as proxy).
            "group": ["nunique"]             # Number of distinct group labels per city.
        }
    )
    print("grouped by city:\n" + g.to_string())  # Pretty-print grouped result for inspection.

    # Transform example: z-score per city on 'score'
    var z = df.groupby_transform(            # Compute a per-row transform within city groups.
        frame,                               # Input DataFrame (original, not aggregated).
        by=["city"],                         # Grouping key: 'city'.
        col="score",                         # Target column for the transform.
        op="zscore"                          # Operation: standardized z-score within each group.
    )
    var t = df.assign(                       # Create a new DataFrame with an added derived column.
        frame,                               # Base DataFrame to augment.
        {"score_z_per_city": z.copy()}       # Column mapping: new column name → data (defensive copy).
    )

    print(df.head(t, 5).to_string())         # Show the first few rows with the new z-score column.
    return t                                 # Return the augmented DataFrame to the caller.

# ---------- main ----------

fn main() -> None:                           # Program entry point.
    var _ = workdir_make()                   # Ensure working directory exists (logged for the user).

    # Demo frame
    var names  : List[String]  = ["Alice","Bob","Cathy","Dan","Eve","Frank","Gina","Hank"]     # Example names.
    var ages   : List[Int]     = [25,31,29,40,22,35,29,31]                                      # Example ages.
    var cities : List[String]  = ["Helsinki","Turku","Tampere","Oulu","Espoo","Helsinki","Tampere","Turku"]  # Cities.
    var scores : List[Float64] = [88.5,75.0,92.0,66.0,79.0,85.5,90.0,70.0]                      # Test scores.
    var groups : List[String]  = ["A","B","A","B","A","B","A","B"]                              # Group labels.

    var pairs = df.make_pairs()              # Start a (name, values) container for building a DataFrame.
    pairs = df.pairs_append(pairs, "name",  names)   # Append 'name' column.
    pairs = df.pairs_append(pairs, "age",   ages)    # Append 'age' column.
    pairs = df.pairs_append(pairs, "city",  cities)  # Append 'city' column.
    pairs = df.pairs_append(pairs, "score", scores)  # Append 'score' column.
    pairs = df.pairs_append(pairs, "group", groups)  # Append 'group' column.

    var frame = df.df_from_pairs(pairs)      # Build a DataFrame from the accumulated pairs.
    print_section("Input DataFrame")         # Section header for the input preview.
    print(df.head(frame, 8).to_string())     # Show the first eight rows of the input data.
    print("dtypes:\n" + df.df_dtypes(frame)) # Show column data types for quick verification.
 
    var out = groupby_agg(frame)             # Run the groupby/aggregation/transform demo to get output.

    print_section("Final DataFrame (with score_z_per_city)")  # Header for the final result preview.
    print(out.head(8).to_string())           # Show the first eight rows of the augmented DataFrame.
