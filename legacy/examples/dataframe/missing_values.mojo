# Project:      Momijo
# Module:       examples.dataframe_merge_join_concat
# File:         dataframe_merge_join_concat.mojo
# Path:         src/momijo/examples/dataframe_merge_join_concat.mojo
#
# Description:  Merge/Join/Concat demonstration for momijo.dataframe (left join, row/column concat).
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
#   - Demonstrates: df.merge(..., how=left), df.concat_rows(...), df.concat_cols(...),
#     df.assign(...), df.take_rows(...), and small lookup table via pairs.

import momijo.dataframe as df              # Import the dataframe API under a short alias for readability.
from pathlib import Path                   # Path abstraction for filesystem-safe path handling.
from os import makedirs                    # Recursive directory creation utility (creates parents if needed).
from collections.list import List          # Typed dynamic array used to hold example column data.

# ---------- Utilities ----------

fn print_section(title: String) -> None:   # Print a bannered section header for clearer console output.
    var line = String("=") * 80            # Create a horizontal rule of '=' characters (fixed width).
    print("\n" + line)                     # Print a leading newline and the top rule.
    print(title)                           # Print the provided section title.
    print(line)                            # Print the bottom rule.

fn workdir_make() -> Path:                 # Ensure an outputs directory exists and return its Path.
    # Simple temp-like folder next to CWD for demo artifacts
    var p = Path("./_momijo_demo_outputs") # Choose a deterministic folder relative to current directory.
    if not p.exists():                     # If the folder does not exist yet, create it recursively.
        try:
            makedirs(String(p))            # Recursively create directory tree; OK if it already exists.
        except _:                          # Ignore any error to keep the teaching demo resilient.
            pass
    print("[INFO] Working directory: " + String(p))  # Log where artifacts will be written.
    return p                               # Return the Path for potential downstream use.
    

# ---------- 7) Merge / Join / Concat ----------

fn merge_join_concat(frame: df.DataFrame) -> df.DataFrame:  # Perform left merge and row/column concatenations.
    print_section("7) Merge / Join / Concat")   # Visible header for this demo step.

    # City -> Region lookup table via pairs (remember to assign after pairs_append)
    var city_region_pairs = df.make_pairs()     # Start building a small lookup table using (name, values) pairs.
    city_region_pairs = df.pairs_append(        # Append first column: city names.
        city_region_pairs, "city",
        ["Helsinki","Turku","Tampere","Oulu","Espoo"]
    )
    city_region_pairs = df.pairs_append(        # Append second column: corresponding regions.
        city_region_pairs, "region",
        ["Uusimaa","Southwest","Pirkanmaa","North Ostrobothnia","Uusimaa"]
    )
    var city_region = df.df_from_pairs(city_region_pairs)  # Construct a DataFrame from the pairs container.

    # Left merge (join) on 'city'
    var left = df.merge(                        # Merge the input frame with the lookup table.
        frame,                                  # Left DataFrame (will keep all its rows).
        city_region,                            # Right DataFrame providing the 'region' info.
        on=["city"],                            # Join key: the 'city' column exists in both frames.
        how=df.Join().left                      # Join mode: left join (preserve all rows from 'frame').
    )
    print("left-merge head():\n" + df.head(left, 5).to_string())  # Preview first rows after the merge.

    # concat rows: duplicate first two rows and mark copies
    var extra = df.take_rows(left, [0, 1])      # Take the first two rows to create duplicates.
    extra = df.assign(                          # Modify the duplicates to mark them as copies.
        extra, {"name": df.col_str_concat(extra, "name", suffix="_copy")}  # Append '_copy' to the 'name'.
    )

    var stacked = df.concat_rows(               # Stack original (left) and duplicated (extra) rows vertically.
        [left, extra],                          # Provide frames to stack in order.
        ignore_index=True                       # Reindex the result to a clean 0..N-1 range.
    )
    print("concat rows head():\n" + df.head(stacked, 6).to_string())  # Show a preview of stacked rows.

    # concat columns: stitch 'name' and 'city' side-by-side from the original frame
    var name_city = df.concat_cols(             # Concatenate columns side-by-side (must align row counts).
        [df.select(frame, ["name"]), df.select(frame, ["city"])]
    )
    print("concat cols head():\n" + df.head(name_city, 5).to_string())  # Preview column-wise concatenation.

    return stacked                              # Return the row-stacked DataFrame for the caller.

# ---------- main ----------

fn main() -> None:                              # Program entry point.
    var _ = workdir_make()                      # Ensure working directory exists (logged above).

    # Demo DataFrame: name/age/city/score/group (8 rows)
    var names  : List[String]  = ["Alice","Bob","Cathy","Dan","Eve","Frank","Gina","Hank"]   # Example names.
    var ages   : List[Int]     = [25,31,29,40,22,35,29,31]                                    # Example ages.
    var cities : List[String]  = ["Helsinki","Turku","Tampere","Oulu","Espoo","Helsinki","Tampere","Turku"]  # Cities.
    var scores : List[Float64] = [88.5,75.0,92.0,66.0,79.0,85.5,90.0,70.0]                    # Test scores.
    var groups : List[String]  = ["A","B","A","B","A","B","A","B"]                            # Group labels.

    var pairs = df.make_pairs()                 # Start a (column-name, values) container to build a DataFrame.
    pairs = df.pairs_append(pairs, "name",  names)   # Add 'name' column.
    pairs = df.pairs_append(pairs, "age",   ages)    # Add 'age' column.
    pairs = df.pairs_append(pairs, "city",  cities)  # Add 'city' column.
    pairs = df.pairs_append(pairs, "score", scores)  # Add 'score' column.
    pairs = df.pairs_append(pairs, "group", groups)  # Add 'group' column.

    var frame = df.df_from_pairs(pairs)         # Build the input DataFrame from the provided columns.
    print_section("Input DataFrame")            # Section header for input preview.
    print(df.head(frame, 8).to_string())        # Show the first eight rows of the input.
    print("dtypes:\n" + df.df_dtypes(frame))    # Display column data types for verification.

    var out = merge_join_concat(frame)          # Execute the merge/join/concat workflow and capture the output.

    print_section("Final stacked DataFrame")    # Section header for final preview.
    print(out.head(12).to_string())             # Show the first twelve rows of the stacked DataFrame.
