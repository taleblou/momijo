# Project:      Momijo
# Module:       examples.pd_nullable
# File:         pd_nullable.mojo
# Path:         src/momijo/examples/pd_nullable.mojo
#
# Description:  Demonstrates nullable dtypes in momijo.dataframe (ints, strings, floats, bools).
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
#   - Shows: constructing Series with nullable dtypes and assembling into a DataFrame.
#   - Prints: table and its dtypes to verify nullability propagation.

import momijo.dataframe as df                 # Import the dataframe API under alias 'df' for brevity.
from collections.list import List             # Import List for typed dynamic arrays (used for index).

# ---------------- basics (nullable) ---------------- #

fn pd_basics_nullable() -> df.DataFrame:      # Build a small DataFrame using nullable dtypes and return it.
    print("\n=== pd_basics_nullable ===")     # Section header to separate this demo in console output.

    # Nullable integer
    var s_int  = df.Series(                   # Construct a Series of 32-bit integers allowing nulls.
        [1, 2, None, 4],                      # Data with one missing value at position 2.
        dtype=df.int32(nullable=True)         # Request a nullable int32 dtype.
    )
    # Nullable string
    var s_str  = df.Series(                   # Construct a Series of strings allowing nulls.
        ["a", None, "c", "d"],                # Data with one missing string value.
        dtype=df.string(nullable=True)        # Request a nullable string dtype.
    )
    # Nullable float
    var s_flt  = df.Series(                   # Construct a Series of 64-bit floats allowing nulls.
        [1.5, None, 3.14, 2.71],              # Floating-point data with a missing value.
        dtype=df.float64(nullable=True)       # Request a nullable float64 dtype.
    )
    # Nullable boolean
    var s_bool = df.Series(                   # Construct a Series of booleans allowing nulls.
        [True, None, False, True],            # Boolean data with a missing value.
        dtype=df.bool(nullable=True)          # Request a nullable boolean dtype.
    )

    var idx:List[String] = ["r0", "r1", "r2", "r3"]  # Explicit row index labels matching the data length.

    var tbl = df.ToDataFrame(                 # Assemble a DataFrame from a mapping of column names to Series.
        ({                                   # Begin inline dictionary of column name → Series.
            "ints":   s_int.copy(),          # Column 'ints' as a defensive copy of the nullable int Series.
            "strs":   s_str.copy(),          # Column 'strs' as a copy of the nullable string Series.
            "floats": s_flt.copy(),          # Column 'floats' as a copy of the nullable float Series.
            "bools":  s_bool.copy()          # Column 'bools' as a copy of the nullable boolean Series.
        }),
        idx                                   # Provide the explicit index for the resulting DataFrame.
    )

    return tbl                                # Return the constructed nullable DataFrame.

# ---------------- main ---------------- #

fn main() -> None:                            # Program entry point with no arguments.
    var tbl = pd_basics_nullable()            # Build the demo DataFrame showcasing nullable dtypes.

    print("nullable df:")                     # Label for the upcoming table printout.
    print(tbl.__str__())                      # Print the DataFrame using its string representation.

    print("dtypes:")                          # Label for the dtype information printout.
    print(tbl.dtypes().__str__())             # Print the DataFrame's dtypes to confirm nullability.
