# Project:      Momijo
# Module:       examples.pd_dtypes_datetime
# File:         pd_dtypes_datetime.mojo
# Path:         src/momijo/examples/pd_dtypes_datetime.mojo
#
# Description:  Demo of dtype management and basic datetime handling in momijo.dataframe.
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
#   - Shows: assign(), to_datetime(), to_category(), dtype inspection, astype(), year extraction.

import momijo.dataframe as df                 # Import the dataframe API under alias df for brevity.
from pathlib import Path                      # Path abstraction for filesystem-safe paths.
from os import makedirs                       # Recursive directory creation (creates parents as needed).
from os import mkdir                          # Single-level directory creation (no parents).

from collections.list import List             # Typed dynamic array container used for demo data.

# ---------- Utilities ----------
fn print_section(title: String) -> None:      # Print a visible section banner around a title.
    var line = String("=") * 80               # Create a horizontal rule line (80 '=' characters).
    print("\n" + line)                        # Blank line + opening rule for spacing.
    print(title)                              # Print the section title text.
    print(line)                               # Closing rule to frame the section.

fn workdir_make() -> Path:                    # Ensure a working directory exists; return its Path.
    # Simple temp-like folder next to CWD for demo artifacts
    var p = Path("./_momijo_demo_outputs")    # Choose a predictable local folder name.
    if not p.exists():                        # If it does not exist, try to create it.
        try:                                  # First attempt: recursive creation (parents included).
            # Try recursive create (parents)
            makedirs(String(p))               # Create directory tree if missing.
        except _:                             # If recursive creation fails, try a simpler mkdir.
            # Fallback: single-level mkdir
            try:
                mkdir(String(p))              # Create a single folder (no parents).
            except _:
                # Ignore errors quietly for teaching demo
                pass                          # Suppress errors to keep the demo resilient.
    print("[INFO] Working directory: " + String(p))  # Inform the user where artifacts go.
    return p                                  # Return the Path so callers can use it.

# ---------- 5) Dtypes & (basic) Datetime ----------
fn dtypes_datetime(frame: df.DataFrame) -> df.DataFrame:  # Transform a DataFrame; return the result.
    print_section("5) Dtypes & Datetime")     # Visual section header in console output.
    var t = df.copy(frame)                    # Work on a copy to avoid mutating the input.

    # Assign a new 'joined' column as strings, then cast to datetime
    t = df.assign(t, {                        # Add/replace columns in a single call using a dict literal.
        "joined": ["2024-01-01","2024-03-05","2024-06-10","2024-07-15","2024-09-01","2024-10-12","2024-12-25","2025-01-01"]
    })
    t = df.to_datetime(t, "joined", fmt="%Y-%m-%d")  # Parse 'joined' strings to datetime with explicit format.

    # Categorical-like ordering for `group`
    t = df.to_category(t, "group", ordered=True, categories=["A","B"])  # Make 'group' an ordered category A<B.

    print("dtypes:\n" + df.df_dtypes(t))      # Show column dtypes after conversions.

    # Show first 3 years extracted from the datetime column
    var years = df.datetime_year_df(t, "joined")  # Derive a DataFrame with extracted year from 'joined'.
    print("year first 3:\n" , years.head(3).to_string())  # Preview the first 3 rows (stringified).

    # Cast age to float64
    t = df.astype(t, "age", "float64")        # Change dtype of 'age' column to float64.

    print(df.head(t, 4).to_string())          # Show the first 4 rows after dtype changes.
    return t                                   # Return the transformed DataFrame.

# ---------- main ----------
fn main() -> None:                             # Program entry point.
    var _ = workdir_make()                     # Ensure the working directory exists (log its path).

    # Build a demo DataFrame with 8 rows (to match the 8 'joined' dates above)
    var names  : List[String]  = ["Alice","Bob","Cathy","Dan","Eve","Frank","Gina","Hank"]  # Sample names.
    var ages   : List[Int]     = [25,31,29,40,22,35,29,31]                                   # Sample ages.
    var cities : List[String]  = ["Helsinki","Turku","Tampere","Oulu","Espoo","Helsinki","Tampere","Turku"]  # Cities.
    var scores : List[Float64] = [88.5,75.0,92.0,66.0,79.0,85.5,90.0,70.0]                   # Test scores.
    var groups : List[String]  = ["A","B","A","B","A","B","A","B"]                           # Group labels.

    var pairs = df.make_pairs()               # Start an empty (name, values) accumulator.
    pairs = df.pairs_append(pairs, "name",  names)   # Add name column.
    pairs = df.pairs_append(pairs, "age",   ages)    # Add age column.
    pairs = df.pairs_append(pairs, "city",  cities)  # Add city column.
    pairs = df.pairs_append(pairs, "score", scores)  # Add score column.
    pairs = df.pairs_append(pairs, "group", groups)  # Add group column.

    var frame = df.df_from_pairs(pairs)       # Materialize a DataFrame from the pairs.
    print_section("Input DataFrame")          # Section header for input preview.
    print(df.head(frame, 8).to_string())      # Print first 8 rows so they align with 8 demo dates.
    print("dtypes:\n" + df.df_dtypes(frame))  # Show initial dtypes before any transformations.

    var out = dtypes_datetime(frame)          # Run the dtype/datetime demo on the input frame.

    print_section("Final DataFrame (after dtypes/datetime)")  # Section header for final preview.
    print(out.head(8).to_string())            # Print first 8 rows of the transformed frame.
