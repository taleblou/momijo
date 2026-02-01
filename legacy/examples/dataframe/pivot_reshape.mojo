# Project:      Momijo
# Module:       examples.dataframe_pivot_reshape
# File:         dataframe_pivot_reshape.mojo
# Path:         src/momijo/examples/dataframe_pivot_reshape.mojo
#
# Description:  Minimal pivot/reshape demo: groupby mean → pivot_table (+margins) → melt back to long form.
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
#   - Demonstrates: df.groupby(...), df.pivot_table(..., margins=True), df.melt(...), df.reset_index(...).

import momijo.dataframe as df          # Import dataframe API under alias 'df' for concise calls.
from pathlib import Path               # Path abstraction for filesystem-safe operations.
from os import makedirs                # Recursive directory creation utility (creates parents).
from os import mkdir                   # Single-level directory creation utility (fallback).

from collections.list import List      # Typed dynamic array used to build example columns.

# ---------- Utilities ----------
fn print_section(title: String) -> None:      # Print a bannered section header for readability.
    var line = String("=") * 80               # Build a fixed-width divider line.
    print("\n" + line)                        # Blank line + top divider.
    print(title)                              # Section title text.
    print(line)                               # Bottom divider.

fn workdir_make() -> Path:                    # Ensure a working directory exists; return it.
    # Simple temp-like folder next to CWD for demo artifacts
    var p = Path("./_momijo_demo_outputs")    # Deterministic local outputs folder.
    if not p.exists():                        # If it does not exist yet, try to create it.
        try:
            # Try recursive create (parents)
            makedirs(String(p))               # Create all missing parent directories.
        except _:
            # Fallback: single-level mkdir
            try:
                mkdir(String(p))              # Attempt a simple single-level creation.
            except _:
                # Ignore errors quietly for teaching demo
                pass
    print("[INFO] Working directory: " + String(p))  # Log the chosen output path.
    return p                                           # Return Path for optional downstream use.

# ---------- 8) Pivot / Reshape (minimal) ----------
fn pivot_reshape(frame: df.DataFrame) -> df.DataFrame:   # Compute grouped mean → pivot → melt.
    print_section("8) Pivot / Reshape")                  # Visible section header for this demo step.

    # 1) aggregate: score mean per (city, group)
    var small = df.groupby(                              # Group by two keys and aggregate.
        frame,                                           # Input DataFrame.
        by=["city","group"],                             # Grouping keys.
        aggs={ "score": ["mean"] }                       # Aggregations: mean of 'score'.
    )
    # Result column is typically named "score_mean"       # Name convention from aggregation.

    # 2) pivot: rows=city, cols=group, values=score_mean (+ margins/Total)
    var pv = df.pivot_table(                             # Build a city×group matrix of means.
        small,                                           # Source: aggregated table with score_mean.
        index="city",                                    # Rows become unique city values.
        columns="group",                                 # Columns become unique group values.
        values="score_mean",                             # Cell values come from the mean column.
        agg=df.Agg.mean(),                               # Safe explicit aggregator (type-checked).
        margins=True,                                    # Add overall totals row/column.
        margins_name="Total"                             # Name for totals row/column.
    )
    print("pivot head():\n" + df.head(pv, 6).to_string())  # Preview first rows of the pivot.

    # 3) melt back to long form
    var melted = df.melt(                                # Convert wide pivot back to long format.
        df.reset_index(pv),                              # Reset index so 'city' becomes a column.
        id_vars=(["city","index"]),                      # Keep city and the original index (from margins).
        var_name="group",                                # Name for the melted column headers.
        value_name="score_mean"                          # Name for the melted values.
    )
    print("melted head():\n" + df.head(melted, 5).to_string())  # Show a small sample of the long form.

    return pv                                            # Return the pivot table (caller can also use 'melted').

# ---------- main ----------
fn main() -> None:                                       # Program entry point.
    var _ = workdir_make()                               # Ensure working directory exists and is logged.

    # Demo DataFrame: name/age/city/score/group (8 rows)
    var names  : List[String]  = ["Alice","Bob","Cathy","Dan","Eve","Frank","Gina","Hank"]    # Example names.
    var ages   : List[Int]     = [25,31,29,40,22,35,29,31]                                     # Example ages.
    var cities : List[String]  = ["Helsinki","Turku","Tampere","Oulu","Espoo","Helsinki","Tampere","Turku"]  # Cities.
    var scores : List[Float64] = [88.5,75.0,92.0,66.0,79.0,85.5,90.0,70.0]                     # Test scores.
    var groups : List[String]  = ["A","B","A","B","A","B","A","B"]                             # Group labels.

    var pairs = df.make_pairs()                           # Start (name, values) container to build a DataFrame.
    pairs = df.pairs_append(pairs, "name",  names)        # Add 'name' column.
    pairs = df.pairs_append(pairs, "age",   ages)         # Add 'age' column.
    pairs = df.pairs_append(pairs, "city",  cities)       # Add 'city' column.
    pairs = df.pairs_append(pairs, "score", scores)       # Add 'score' column.
    pairs = df.pairs_append(pairs, "group", groups)       # Add 'group' column.

    var frame = df.df_from_pairs(pairs)                   # Create a DataFrame from the accumulated pairs.
    print_section("Input DataFrame")                      # Header for the input preview.
    print(df.head(frame, 8).to_string())                  # Show the entire 8-row sample.
    print("dtypes:\n" + df.df_dtypes(frame))              # Inspect column data types.

    var pv = pivot_reshape(frame)                         # Run the pivot/reshape demo and capture the pivot table.

    print_section("Final Pivot (with Total)")             # Header for final output preview.
    print(pv.to_string())                                 # Print the full pivot including totals.
