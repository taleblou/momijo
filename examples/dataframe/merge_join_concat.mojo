# Project:      Momijo
# Module:       examples.dataframe_merge_join_concat
# File:         dataframe_merge_join_concat.mojo
# Path:         src/momijo/examples/dataframe_merge_join_concat.mojo
#
# Description:  Merge (join) and concat demonstrations for momijo.dataframe.
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
#   - Shows: left merge on a lookup table, row-wise concatenation with duplicated rows,
#            column-wise concatenation of selected columns.

import momijo.dataframe as df               # Import the dataframe API under alias 'df' for brevity.
from pathlib import Path                    # Path abstraction for safe filesystem handling.
from os import makedirs                     # Recursive directory creation helper (creates parents).
from collections.list import List           # Generic dynamic array for demo data containers.

# ---------- Utilities ----------

fn print_section(title: String) -> None:    # Print a bannered section header for console readability.
    var line = String("=") * 80             # Build a divider line of '=' repeated to a fixed width.
    print("\n" + line)                      # Print a leading newline and the top divider.
    print(title)                            # Print the provided section title.
    print(line)                             # Print the bottom divider.

fn workdir_make() -> Path:                  # Ensure a local working directory exists; return it.
    # Simple temp-like folder next to CWD for demo artifacts
    var p = Path("./_momijo_demo_outputs")  # Choose a deterministic outputs folder relative to CWD.
    if not p.exists():                      # If the folder does not exist yet, try to create it.
        try:
            makedirs(String(p))             # Recursively create the directory tree; ok if already exists.
        except _:
            pass                            # Ignore errors to keep this teaching demo resilient.
    print("[INFO] Working directory: " + String(p))  # Log where artifacts will be written.
    return p                                # Return the path for potential downstream use.
    

# ---------- 7) Merge / Join / Concat ----------

fn merge_join_concat(frame: df.DataFrame) -> df.DataFrame:  # Demonstrate merge/join and concatenations.
    print_section("7) Merge / Join / Concat")  # Visible header for this stage.

    # City -> Region lookup table via pairs (remember to assign after pairs_append)
    var city_region_pairs = df.make_pairs()     # Start an empty (name, values) collection for a tiny lookup.
    city_region_pairs = df.pairs_append(        # Add first column 'city' with five Finnish cities.
        city_region_pairs, "city",
        ["Helsinki","Turku","Tampere","Oulu","Espoo"]
    )
    city_region_pairs = df.pairs_append(        # Add second column 'region' matching each city by position.
        city_region_pairs, "region",
        ["Uusimaa","Southwest","Pirkanmaa","North Ostrobothnia","Uusimaa"]
    )
    var city_region = df.df_from_pairs(city_region_pairs)  # Materialize a DataFrame from the pairs.

    # Left merge (join) on 'city'
    var left = df.merge(                        # Perform a left join to enrich 'frame' with 'region'.
        frame, city_region, on=["city"], how=df.Join().left
    )
    print("left-merge head():\n" + df.head(left, 5).to_string())  # Preview the merged result.

    # concat rows: duplicate first two rows and mark copies
    var extra = df.take_rows(left, [0, 1])      # Select the first two rows to duplicate.
    extra = df.assign(                          # Modify 'name' in the duplicate rows to indicate copies.
        extra, {"name": df.col_str_concat(extra, "name", suffix="_copy")}
    )

    var stacked = df.concat_rows(               # Stack original and duplicated rows vertically.
        [left, extra], ignore_index=True
    )
    print("concat rows head():\n" + df.head(stacked, 6).to_string())  # Preview the stacked table.

    # concat columns: stitch 'name' and 'city' side-by-side from the original frame
    var name_city = df.concat_cols(             # Concatenate two single-column DataFrames horizontally.
        [df.select(frame, ["name"]), df.select(frame, ["city"])]
    )
    print("concat cols head():\n" + df.head(name_city, 5).to_string())  # Preview the column concat.

    return stacked                              # Return the vertically concatenated DataFrame.

# ---------- main ----------

fn main() -> None:                              # Program entry point.
    var _ = workdir_make()                      # Ensure working directory exists (logged above).

    # Demo DataFrame: name/age/city/score/group (8 rows)
    var names  : List[String]  = ["Alice","Bob","Cathy","Dan","Eve","Frank","Gina","Hank"]  # Example names.
    var ages   : List[Int]     = [25,31,29,40,22,35,29,31]                                   # Example ages.
    var cities : List[String]  = ["Helsinki","Turku","Tampere","Oulu","Espoo","Helsinki","Tampere","Turku"]  # Cities.
    var scores : List[Float64] = [88.5,75.0,92.0,66.0,79.0,85.5,90.0,70.0]                   # Test scores.
    var groups : List[String]  = ["A","B","A","B","A","B","A","B"]                           # Group labels.

    var pairs = df.make_pairs()                 # Initialize a (name, values) container to build a DataFrame.
    pairs = df.pairs_append(pairs, "name",  names)   # Add 'name' column.
    pairs = df.pairs_append(pairs, "age",   ages)    # Add 'age' column.
    pairs = df.pairs_append(pairs, "city",  cities)  # Add 'city' column.
    pairs = df.pairs_append(pairs, "score", scores)  # Add 'score' column.
    pairs = df.pairs_append(pairs, "group", groups)  # Add 'group' column.

    var frame = df.df_from_pairs(pairs)         # Create a DataFrame from the assembled pairs.
    print_section("Input DataFrame")            # Header for input preview.
    print(df.head(frame, 8).to_string())        # Show the first eight rows for context.
    print("dtypes:\n" + df.df_dtypes(frame))    # Display inferred data types for each column.

    var out = merge_join_concat(frame)          # Run merge/join/concat demo; capture the result.

    print_section("Final stacked DataFrame")    # Header for final preview.
    print(out.head(12).to_string())             # Show the first twelve rows of the stacked output.
