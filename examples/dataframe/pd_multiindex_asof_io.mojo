# Project:      Momijo
# Module:       examples.dataframe_multiindex_asof_io
# File:         dataframe_multiindex_asof_io.mojo
# Path:         src/momijo/examples/dataframe_multiindex_asof_io.mojo
#
# Description:  MultiIndex construction/inspection, merge_asof variants, and CSV round-trip I/O.
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
#   - Demonstrates: MultiIndex.from_arrays → DataFrame index wiring, merge_asof (backward/forward/nearest),
#     to_datetime + sort_values preconditions, and CSV write/read round-trip.

import momijo.dataframe as df  # Import dataframe API under alias 'df' for brevity and clarity.

# ---------------- MultiIndex / merge_asof / CSV IO ---------------- #
fn pd_multiindex_asof_io(tmp_prefix: String = "pd_demo") raises -> df.DataFrame:  # Demo function; returns DataFrame and may raise.
    print("\n=== pd_multiindex_asof_io ===")  # Section header for console readability.

    # ---- MultiIndex dataframe ------------------------------------------------
    # Use explicit arrays (avoid list("ABABAB") for portability)
    var A   = df.Series(["A","B","A","B","A","B"], dtype=df.string())  # First level labels as a string Series.
    var I   = df.Series([1,2,3,1,2,3], dtype=df.int32())                # Second level labels as an int32 Series.
    var MI  = df.MultiIndex.from_arrays([A.copy(), I.copy()], names=["grp","idx"])  # Build MultiIndex from parallel arrays.

    # Convert MultiIndex -> (List[String] index values, String index name)
    var pair = MI.to_index_pair()              # Extract a pair: (materialized index values, combined index name).
    var idx_vals = pair[0].copy()              # Copy the index values (list-like) for safe reuse.
    var idx_name = pair[1].copy()              # Copy the index name (string) if needed elsewhere.

    var midf = df.ToDataFrame({"v": df.range(0, 6, dtype=df.int32())}, index=idx_vals)  # Create a DF with value column 'v' and MultiIndex values.

    print("multiindex df:")                    # Label for upcoming DataFrame dump.
    print(midf.__str__())                      # Print the DataFrame string representation.
    print("multiindex df dtypes:")             # Label for dtype information.
    print(midf.dtypes().__str__())             # Print each column's dtype.

    # Optional: show index summary (levels and names) if available in your API
    # print(midf.index().__str__())            # (Commented) Example: print index metadata when supported.

    # ---- merge_asof: inputs must be time-sorted --------------------------------
    # left
    var left = df.ToDataFrame({                # Construct the left table for as-of join.
        "t": df.Series(                        # Time column as strings initially.
            (["2025-01-01 00:00:00","2025-01-01 00:00:05","2025-01-01 00:00:10"]),
            dtype=df.string()
        ),
        "x": df.Series([1,2,3], dtype=df.int32())  # Payload column 'x'.
    })
    left = df.to_datetime(left, "t", String("%Y-%m-%d %H:%M:%S"))  # Parse 't' strings into datetimes using the given format.
    left = left.sort_values((["t"]))          # Sort by time; merge_asof requires monotonic time order.

    # right
    var right = df.ToDataFrame({               # Construct the right table for as-of join.
        "t": df.Series(                        # Time column as strings initially.
            (["2025-01-01 00:00:03","2025-01-01 00:00:09"]),
            dtype=df.string()
        ),
        "y": df.Series([10,99], dtype=df.int32())  # Payload column 'y'.
    })
    right = df.to_datetime(right, "t", String("%Y-%m-%d %H:%M:%S"))  # Parse 't' strings into datetimes.
    right = right.sort_values((["t"]))        # Sort by time to satisfy as-of join precondition.

    var masof_back = df.merge_asof(left, right, on="t", direction=df.AsOf().backward)  # As-of join: match last right.t ≤ left.t.
    print("\nmerge_asof (backward):")         # Label for backward result.
    print(masof_back.__str__())               # Print backward as-of join output.

    # (Optional) also demonstrate 'forward' and 'nearest' if supported
    var masof_fwd = df.merge_asof(left, right, on="t", direction=df.AsOf().forward)    # As-of join: first right.t ≥ left.t.
    print("\nmerge_asof (forward):")          # Label for forward result.
    print(masof_fwd.__str__())                # Print forward as-of join output.

    var masof_near = df.merge_asof(left, right, on="t", direction=df.AsOf().nearest)   # As-of join: closest right.t to left.t.
    print("\nmerge_asof (nearest):")          # Label for nearest result.
    print(masof_near.__str__())               # Print nearest as-of join output.

    # ---- CSV IO ----------------------------------------------------------------
    var C = df.ToDataFrame({                  # Build a simple DF to demonstrate CSV round-trip.
        "a": df.Series([1,2,3], dtype=df.int32()),  # Column 'a' as int32.
        "b": df.Series([4,5,6], dtype=df.int32())   # Column 'b' as int32.
    })

    # Write without index for a clean CSV
    var csv_path = tmp_prefix + ".csv"        # Choose output CSV path using the given prefix.
    C.to_csv(csv_path, index=False)           # Write to CSV without index to simplify round-trip.

    var rd = df.read_csv(csv_path)            # Read the CSV back into a new DataFrame.
    print("\nread csv:")                      # Label for CSV-read output.
    print(rd.__str__())                       # Print the loaded DataFrame.
    print("read csv dtypes:")                 # Label for dtype information of loaded data.
    print(rd.dtypes().__str__())              # Print dtypes after CSV round-trip.

    # Return something useful (e.g., the CSV round-trip result)
    return rd                                 # Return the DataFrame loaded from CSV as the demo artifact.

# ---------------- main ---------------- #
fn main() raises -> None:                     # Standard entry point; may raise to bubble errors to caller.
    var _ = pd_multiindex_asof_io("pd_demo")  # Invoke the demo with default prefix and ignore the returned DF here.
