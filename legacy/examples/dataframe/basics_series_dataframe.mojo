# Project:      Momijo
# Module:       examples.dataframe_basics
# File:         dataframe_basics.mojo
# Path:         src/momijo/examples/dataframe_basics.mojo
#
# Description:  Teaching-style demo of Series/DataFrame basics in momijo.dataframe.
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
#   - Covers: Series construction, DataFrame from pairs, shape/dtypes/describe, select/rename,
#             set/reset index, and pretty-printing with head().

# Import the dataframe module under a short alias for convenience.
import momijo.dataframe as df
# Import List for typed dynamic arrays used to hold values and indices.
from collections.list import List 
# Import Path for filesystem-safe path handling.
from pathlib import Path
# Import recursive mkdir to create nested directories when needed.
from os import makedirs
# Import single-level mkdir for fallback creation.
from os import mkdir

# ---------- Utilities ----------

# Print a visible, sectioned title banner for console output readability.
fn print_section(title: String) -> None:
    # Build a horizontal rule line of '=' characters with a fixed width.
    var line = String("=") * 80
    # Print an empty line, then the rule to separate sections.
    print("\n" + line)
    # Print the section title text.
    print(title)
    # Print the closing rule line to frame the title.
    print(line)

# Create (if needed) and return a working directory path for demo artifacts.
fn workdir_make() -> Path:
    # Choose a predictable, local folder relative to the current working directory.
    var p = Path("./_momijo_demo_outputs")
    # If the path does not yet exist, attempt to create it.
    if not p.exists():
        try:
            # First attempt: recursive creation (creates parents when missing).
            makedirs(String(p))
        except _:
            # If recursive creation fails for any reason, try a single-level mkdir.
            try:
                mkdir(String(p))
            except _:
                # Final fallback: ignore errors to keep the demo non-crashing.
                pass
    # Inform the user where files will be written.
    print("[INFO] Working directory: " + String(p))
    # Return the Path so callers can use it or join child paths.
    return p

# ---------- 1) Basics: Series & DataFrame ----------

# Build a small DataFrame and print common information; return the frame for further use.
fn basics_series_dataframe() -> df.DataFrame:
    # Show a section header describing what this function demonstrates.
    print_section("1) Basics: Series & DataFrame")

    # Prepare an index for a toy Series (three labels).
    var idx:  List[String] = [String("a"), String("b"), String("c")]
    # Prepare corresponding integer values for the Series.
    var vals: List[Int]    = [10, 20, 30]

    # Construct a Series from the given index and values, and assign a name.
    var s = df.series_from_list(index=idx, values=vals, name=String("numbers"))
    # Print the Series index representation (for teaching/debugging).
    print("Series index: " + df.series_index(s))

    # Prepare example columns for a DataFrame: names...
    var names  : List[String]  = [String("Alice"), String("Bob"), String("Cathy"), String("Dan"),
                                  String("Eve"), String("Frank"), String("Gina"), String("Hank")]
    # ...ages (integers)...
    var ages   : List[Int]     = [25, 31, 29, 40, 22, 35, 29, 31]
    # ...cities (strings)...
    var cities : List[String]  = [String("Helsinki"), String("Turku"), String("Tampere"), String("Oulu"),
                                  String("Espoo"), String("Helsinki"), String("Tampere"), String("Turku")]
    # ...scores (floats)...
    var scores : List[Float64] = [88.5, 75.0, 92.0, 66.0, 79.0, 85.5, 90.0, 70.0]
    # ...and group labels (strings).
    var groups : List[String]  = [String("A"), String("B"), String("A"), String("B"),
                                  String("A"), String("B"), String("A"), String("B")]

    # Start with an empty column-name → values container (implementation-defined "pairs").
    var pairs = df.make_pairs()
    # Append each column to the pairs with its column name and corresponding values.
    pairs = df.pairs_append(pairs, String("name"),  names)
    pairs = df.pairs_append(pairs, String("age"),   ages)
    pairs = df.pairs_append(pairs, String("city"),  cities)
    pairs = df.pairs_append(pairs, String("score"), scores)
    pairs = df.pairs_append(pairs, String("group"), groups)

    # Build a DataFrame from the accumulated pairs.
    var frame = df.df_from_pairs(pairs)

    # Print the shape (rows, columns) of the DataFrame.
    print("shape: " + df.df_shape(frame))
    # Print a summary of each column's dtype.
    print("dtypes:\n" + df.df_dtypes(frame))
    # Print descriptive statistics for numeric columns (and applicable summaries).
    print("describe():\n" + df.df_describe(frame))
    # Demonstrate selecting a subset of columns and printing them.
    print("Select ...:\n" + df.select(frame, [String("name"), String("age")]).to_string())

    # Demonstrate column rename and index naming via a mapping and index_name parameter.
    var renamed = df.rename(
        frame,
        cols_map={String("name"): String("person"), String("score"): String("test_score")},
        index_name=String("row_id")
    )
    # Show the first three rows of the renamed frame.
    print("renamed head():\n" + df.head(renamed, 3).to_string())

    # Demonstrate setting an index to a specific column ("name").
    var idxd = df.set_index(frame, String("name"))
    # Show the first three rows with the new index applied.
    print("set_index('name') head:\n" + df.head(idxd, 3).to_string())
    # Demonstrate resetting the index back to a default RangeIndex-like structure and preview it.
    print("reset_index() head:\n" + df.head(df.reset_index(idxd), 3).to_string())

    # Return a defensive copy of the original frame for use by the caller.
    return frame.copy()

# ---------- main ----------

# Program entry point that prepares the workspace, builds a demo DataFrame, and prints a preview.
fn main() -> None:
    # Ensure the working directory exists (and log its path), even if unused later.
    var _ = workdir_make()
    # Build the teaching DataFrame and capture it.
    var frame = basics_series_dataframe()
    # Print the first five rows so the user can see the returned result.
    print("\nReturned frame head:\n" + df.head(frame, 5).to_string())
