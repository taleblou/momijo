# Project:      Momijo
# Module:       examples.dataframe_categorical_datetime_resample
# File:         dataframe_categorical_datetime_resample.mojo
# Path:         src/momijo/examples/dataframe_categorical_datetime_resample.mojo
#
# Description:  Categorical data with datetime index; resampling to calendar bins and basic dtype/column ops.
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
#   - Demonstrates: date_range(), Series construction with explicit dtypes,
#     building a DataFrame, setting a DatetimeIndex, resampling (sum/mean),
#     numeric_only behavior, and creating a derived column from the index.

import momijo.dataframe as df                    # Import the dataframe API under alias 'df' for brevity.

# ---------------- categorical / datetime / resample ---------------- #
fn pd_categorical_datetime_resample() -> df.DataFrame:  # Demo function returning the constructed DataFrame.
    print("\n=== pd_categorical_datetime_resample ===")  # Section banner for console readability.

    # Build a daily datetime range (8 days starting 2025-01-01)
    var ts = df.date_range(start="2025-01-01", periods=8, freq="D")  # Create a DatetimeIndex-like sequence.

    # Explicit labels instead of list("AABBCCDD") for clarity/portability
    var cat_raw = df.Series(                                     # Build a categorical-like string Series.
        ["A","A","B","B","C","C","D","D"],                       # Eight labels aligned with 'ts'.
        dtype=df.string()                                        # Explicit string dtype for portability.
    ) 

    var vals = df.Series([0,10,20,30,40,50,60,70], dtype=df.int32())  # Numeric Series with explicit int32 dtype.

    # Assemble and set datetime index
    var T = df.ToDataFrame({                                     # Construct a DataFrame from a dict of Series.
        "ts": ts.copy(),                                         # Copy timestamp sequence for safety.
        "cat": cat_raw.copy(),                                   # Copy category labels to avoid aliasing.
        "val": vals.copy()                                       # Copy numeric values likewise.
    }).set_index("ts")                                           # Set 'ts' as the index (DatetimeIndex).

    print("df (indexed by ts):")                                 # Explain the next printout.
    print(T.__str__())                                           # Print the DataFrame string representation.
    print("dtypes:")                                             # Label for the dtype display.
    print(T.dtypes().__str__())                                  # Print each column’s dtype information.

    # Resample to 2-day bins; numeric columns summed
    var R = T.resample("2D").sum(numeric_only=True)              # Resample by 2-day windows; sum numeric columns only.
    print("\nresample 2D (sum, numeric_only=True):")             # Explain the result being shown.
    print(R.__str__())                                           # Print the resampled-sum DataFrame.
    print("R.dtypes:")                                           # Label for resampled dtypes.
    print(R.dtypes().__str__())                                  # Show dtypes after aggregation.

    # Calendar ops from the DatetimeIndex
    T.set_column("dow", T.index())                               # Insert a derived column from the index (e.g., day-of-week).

    print("\nday names (from index):")                           # Section for the view of selected columns.
    print(T[["cat","val","dow"]].__str__())                      # Print subset with category, value, and derived index column.

    return T                                                     # Return the indexed DataFrame for further use.

# ---------------- main ---------------- #
fn main() -> None:                                               # Program entry point.
    var T = pd_categorical_datetime_resample()                   # Build and inspect the demo DataFrame.

    # Optional: show a quick resample-mean, just to contrast with sum
    var R_mean = T[["val"]].resample("2D").mean(numeric_only=True)  # Resample only 'val' and compute 2-day mean.
    print("\nresample 2D (mean, numeric_only=True):")            # Label the mean-aggregation output.
    print(R_mean.__str__())                                      # Print the resampled-mean DataFrame.
