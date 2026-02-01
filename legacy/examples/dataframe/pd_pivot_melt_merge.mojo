# Project:      Momijo                              # Repository/project name.
# Module:       examples.pd_pivot_melt_merge        # Logical module path within the package.
# File:         pd_pivot_melt_merge.mojo            # Source filename.
# Path:         src/momijo/examples/pd_pivot_melt_merge.mojo  # Repository-relative path.
#
# Description:  Demo of pivot / melt / merge operations using momijo.dataframe.  # One-line summary.
#
# Author(s):    Morteza Taleblou & Mitra Daneshmand # File maintainers/authors.
# Website:      https://taleblou.ir/                # Project or author website.
# Repository:   https://github.com/taleblou/momijo  # Canonical Git repository.
#
# License:      MIT License                         # License label (short form only).
# SPDX-License-Identifier: MIT                      # SPDX identifier required by policy.
# Copyright:    (c) 2025 Morteza Taleblou & Mitra Daneshmand  # Copyright.
#
# Notes:                                           # Optional quick notes for readers.
#   - Shows: pivot_table → wide, reset_index+melt → long, merge with metadata, sorting for readability.

import momijo.dataframe as df                      # Import the dataframe API under alias 'df' for brevity.

# ---------------- pivot / melt / merge ---------------- #
fn pd_pivot_melt_merge():                          # Define a demo function to showcase pivot/melt/merge.
    print("\n=== pd_pivot_melt_merge ===")         # Print a visible section header.

    # Tall/long table of store-month revenue
    var sales = df.ToDataFrame({                   # Build an example long-format DataFrame from a dict-like spec.
        "store":   df.Series(["A","A","B","B"], dtype=df.string()),            # Store ID column as string dtype.
        "month":   df.Series(["2025-01","2025-02","2025-01","2025-02"], dtype=df.string()),  # Month label as string.
        "revenue": df.Series([100, 120, 90, 130], dtype=df.int32())            # Revenue values as int32.
    })
    print("sales:")                                # Label before printing the table.
    print(sales.__str__())                         # Render the DataFrame to string explicitly and print it.
    print("sales.dtypes:")                         # Label before printing dtypes.
    print(sales.dtypes().__str__())                # Show each column's dtype as a string.

    # Pivot to wide format (stores as rows, months as columns)
    var wide = sales.pivot_table(                  # Create a pivot table (wide format).
        index=["store"],                           # Row index: one row per store.
        columns=["month"],                         # Column headers: one column per month.
        values="revenue",                          # Aggregated values come from the 'revenue' column.
        agg=df.Agg.sum(),                          # Aggregation function: sum revenues if duplicates exist.
        fill_value=df.Value.int32(0)               # Replace missing combinations with 0.
    )
    print("\npivot wide:")                         # Blank line + label for readability.
    print(wide.__str__())                          # Print the wide-format table.
    print("wide.dtypes:")                          # Label before printing dtypes of the wide table.
    print(wide.dtypes().__str__())                 # Show dtypes for the pivot result.

    # Back to long via reset_index + melt
    var long = wide.reset_index().melt(            # Convert back to long format for tidy operations.
        id_vars=["store"],                         # Keep 'store' as an identifier column (not melted).
        var_name="month",                          # Name for the variable column holding former column labels.
        value_name="revenue"                       # Name for the values column holding pivot cell values.
    )
    print("\nmelt back (long):")                   # Blank line + label for readability.
    print(long.__str__())                          # Print the melted (long) table.
    print("long.dtypes:")                          # Label before printing dtypes of the long table.
    print(long.dtypes().__str__())                 # Show dtypes for the melted result.

    # Add store metadata and merge (left join)
    var meta = df.ToDataFrame({                    # Build a small metadata DataFrame keyed by 'store'.
        "store": df.Series(["A","B"], dtype=df.string()),   # Store IDs present in metadata.
        "city":  df.Series(["Pori","Oulu"], dtype=df.string())  # City where each store is located.
    })
    var merged = long.merge(                       # Join the long table with metadata.
        meta,                                      # Right-hand DataFrame containing metadata.
        on=["store"],                              # Join key: 'store'.
        how=df.Join().left                         # Join type: left (preserve all rows from 'long').
    )

    # A tidy final sort for readability
    merged = merged.sort_values(                   # Sort rows for deterministic, readable output.
        ["store","month"],                         # Sort keys: first by store, then by month.
        ascending=[True, True]                     # Sort both keys in ascending order.
    )

    print("\nmeta:")                               # Label before printing metadata.
    print(meta.__str__())                          # Print the metadata table.
    print("\nmerged:")                             # Label before printing merged table.
    print(merged.__str__())                        # Print the merged and sorted result.
    print("merged.dtypes:")                        # Label before printing dtypes of the merged table.
    print(merged.dtypes().__str__())               # Show dtypes for the final table.
 

# ---------------- main ---------------- #
fn main() -> None:                                 # Program entry point; no CLI args.
    pd_pivot_melt_merge()                          # Run the pivot/melt/merge demonstration.
