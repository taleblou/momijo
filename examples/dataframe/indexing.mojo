# Project:      Momijo
# Module:       examples.dataframe_indexing_loc_iloc
# File:         dataframe_indexing_loc_iloc.mojo
# Path:         src/momijo/examples/dataframe_indexing_loc_iloc.mojo
#
# Description:  Demonstrates label/position indexing (loc/iloc) and scalar access (at/iat) in momijo.dataframe.
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
#   - Shows: constructing a DataFrame with explicit Index, loc/iloc slices and lists,
#             single-cell access via loc/iloc, scalar mutations via at/iat, dtype checks.

import momijo.dataframe as df                      # Import the dataframe API under a short alias for convenience.

# ---------------- indexing: loc / iloc / at / iat ---------------- #
fn pd_indexing_loc_iloc() -> None:                 # Define a demo function returning nothing (prints to stdout).
    print("\n=== pd_indexing_loc_iloc ===")        # Visual banner to mark the start of the demo.

    # Construct a simple DataFrame with explicit index labels
    var X = df.ToDataFrame({                       # Create a DataFrame from a dict-like mapping of column names → Series.
        "x": df.Series([0, 1, 2, 3, 4], dtype=df.int64()),     # Column 'x' as 64-bit integers.
        "y": df.Series(["a","b","c","d","e"], dtype=df.string())  # Column 'y' as strings.
    }, index=df.Index(["v","w","x","y","z"]))      # Provide an explicit Index with labels for rows.

    print("Initial X:")                            # Print a heading for the initial table dump.
    print(X.__str__())                             # Print the DataFrame string representation.

    # ---------- loc: label-based selection ----------
    print("\nloc slice rows v..x (inclusive), cols ['x']:")    # Explain the next operation (loc slice).
    var loc_slice = X.loc(                          # Perform a label-based selection using .loc.
        rows=df.slice_labels("v", "x", inclusive=True),  # Select rows from label 'v' through 'x' inclusive.
        cols=["x"]                                   # Select only column 'x'.
    )
    print(loc_slice.__str__())                      # Print the selection result.

    print("\nloc rows ['w','y'], cols ['x','y'] (label lists):")  # Next: list-of-labels selection.
    var loc_rows_cols = X.loc(rows=["w","y"], cols=["x","y"])     # Select given row labels and column labels.
    print(loc_rows_cols.__str__())                  # Print the result.

    print("\nloc single cell ['x','y'] (row label='x', col='y'):")  # Next: single-cell access via labels.
    var cell_xy = X.loc(row="x", col="y")          # Select a single scalar cell using row/col labels.
    print(cell_xy.__str__())                       # Print the scalar (or scalar-like wrapper) as string.

    # ---------- iloc: position-based selection ----------
    print("\niloc rows 1..3 (end-exclusive), col 0:")  # Explain positional slicing (end-exclusive).
    var iloc_slice = X.iloc(                       # Perform a position-based selection with .iloc.
        rows=df.slice_rows(1, 4),                  # Select row positions 1,2,3 (stop=4 not included).
        cols=df.slice_cols(0, 1)                   # Select only column position 0.
    )
    print(iloc_slice.__str__())                    # Print the selection result.

    print("\niloc rows [0,2,4], cols [0,1] (position lists):")  # Next: list-of-positions selection.
    var iloc_list = X.iloc(rows=[0,2,4], cols=[0,1])            # Select specified row/col positions.
    print(iloc_list.__str__())                   # Print the result.

    print("\niloc single cell [row=3, col=1]:")  # Next: single-cell access by positions.
    var cell_31 = X.iloc(row=3, col=1)           # Select scalar at (row index 3, col index 1).
    print(cell_31.__str__())                     # Print the scalar (or scalar-like wrapper).

    # ---------- Mutations with at / iat (scalar set) ----------
    # Note: use df.Value.* wrappers to set typed scalars
    X.set_at(label_row="w", col="x", v=df.Value.int32(999))  # Set scalar by label: row 'w', column 'x' to 999 (int32).
    X.set_iat(row=2, col=0,       v=df.Value.int32(-5))      # Set scalar by position: (2,0) to -5 (int32).

    print("\nAfter at/iat mutation:")            # Heading for mutation results.
    print(X.__str__())                           # Print the DataFrame to show in-place updates.

    # ---------- More loc/iloc variants ----------
    print("\nloc semi-open slice rows v..y (inclusive=False), cols ['y']:")  # Demonstrate half-open label slice.
    var loc_semi = X.loc(                        # Another label-based selection.
        rows=df.slice_labels("v", "y", inclusive=False),  # From 'v' up to but not including 'y' (labels 'v','w','x').
        cols=["y"]                               # Only column 'y'.
    )
    print(loc_semi.__str__())                    # Print the result.

    print("\niloc rectangular block rows 0..3, cols 0..2 (end-exclusive):")  # Rectangular block by positions.
    var iloc_block = X.iloc(                     # Position-based block selection.
        rows=df.slice_rows(0, 3),                # Rows 0,1,2.
        cols=df.slice_cols(0, 2)                 # Cols 0,1.
    )
    print(iloc_block.__str__())                  # Print the block.

    # ---------- Read-back mutations via at/iat ----------
    print("\nRead-back mutated scalars:")        # Heading for scalar read-back.
    var rb1 = X.at(label_row="w", col="x").get() # Read scalar by label; unwrap with .get() if Value-like.
    var rb2 = X.iat(row=2, col=0).get()          # Read scalar by position; unwrap with .get().
    print("X.at('w','x') =", rb1.__str__())      # Print the first read-back value.
    print("X.iat(2,0)    =", rb2.__str__())      # Print the second read-back value.

    # ---------- DTypes check ----------
    print("\nDTypes after all operations (should remain consistent):")  # Sanity-check dtypes.
    print(X.dtypes().__str__())                  # Print the dtype summary to verify no unintended coercions.

# ---------------- main ---------------- #
fn main() -> None:                               # Program entry point.
    pd_indexing_loc_iloc()                       # Run the indexing demo.
