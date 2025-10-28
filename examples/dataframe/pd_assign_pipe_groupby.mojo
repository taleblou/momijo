# Project:      Momijo
# Module:       examples.pd_assign_pipe_groupby
# File:         pd_assign_pipe_groupby.mojo
# Path:         src/momijo/examples/pd_assign_pipe_groupby.mojo
#
# Description:  Manual assign/pipe/groupby-style demo: derive z/bucket columns, sort by score, and rank by z.
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
#   - Implements: Newton–Raphson sqrt, z-score and bucket builders from Int lists, argsort(desc) for Float64,
#     stable reindex helpers, DataFrame construction, and top-k display by z.

import momijo.dataframe as df              # Import the dataframe API with a short alias for convenience.
from collections.list import List           # Import List (typed dynamic array) used throughout the demo.

# ---------- helpers: pure List utilities ----------

@always_inline                              # Hint the compiler to inline this small numeric helper.
fn _sqrt_nr(x: Float64) -> Float64:        # Newton–Raphson square root for non-negative Float64.
    if x <= 0.0:                           # Guard: return 0 for non-positive inputs to avoid division by zero.
        return 0.0                         # Zero result for x <= 0.
    var g = x                              # Initial guess set to x (converges quickly for positive x).
    var i = 0                              # Loop counter for fixed-iteration refinement.
    while i < 12:                          # Perform a small, constant number of NR iterations.
        g = 0.5 * (g + x / g)              # Newton update: g_{n+1} = (g + x/g)/2.
        i += 1                             # Increment iteration counter.
    return g                               # Return the approximated square root.

fn make_z_from_ints(scores: List[Int]) -> List[String]:  # Compute z-scores from Int scores; return as strings.
    var n = len(scores)                    # Number of observations.
    if n == 0:                             # Handle empty input defensively.
        var empty = List[String]()         # Allocate an empty list for symmetry.
        return empty.copy()                # Return a copy to avoid aliasing.

    var vals = List[Float64]()             # Temporary Float64 list for numeric calculations.
    vals.reserve(n)                        # Pre-reserve capacity for speed.
    var i = 0                              # Index for loops.
    while i < n:                           # Convert Int scores to Float64.
        vals.append(Float64(scores[i]))    # Append converted value.
        i += 1                             # Next index.

    var sumv = 0.0                         # Accumulator for the sum.
    i = 0                                  # Reset index.
    while i < n:                           # Sum all values.
        sumv += vals[i]                    # Add current value.
        i += 1                             # Next index.
    var mean = sumv / Float64(n)           # Arithmetic mean.

    var varsum = 0.0                       # Accumulator for squared deviations.
    i = 0                                  # Reset index.
    while i < n:                           # Compute sum of squared deviations.
        var dv = vals[i] - mean            # Deviation from mean.
        varsum += dv * dv                  # Accumulate squared deviation.
        i += 1                             # Next index.
    var std = _sqrt_nr(varsum / Float64(n))  # Population standard deviation via NR sqrt.

    var out = List[String]()               # Output strings (to match downstream string dtype).
    out.reserve(n)                         # Reserve capacity for efficiency.
    i = 0                                  # Reset index to populate output.
    if std == 0.0:                         # Degenerate case: all values equal.
        while i < n:                       # Fill with zeros to avoid division by zero.
            out.append(String(0.0))        # Append "0.0" for each observation.
            i += 1                         # Next index.
    else:                                  # Normal case: non-zero standard deviation.
        while i < n:                       # Compute standardized z = (x - mean) / std.
            out.append(String((vals[i] - mean) / std))  # Append z-score as string.
            i += 1                         # Next index.
    return out.copy()                      # Return a defensive copy (no shared ownership).

fn make_bucket_from_ints(scores: List[Int]) -> List[String]:  # Bucketize Int scores into coarse labels (as strings).
    var out = List[String]()               # Output list of bucket strings.
    out.reserve(len(scores))               # Reserve capacity to avoid reallocations.
    var i = 0                              # Loop index.
    var n = len(scores)                    # Cache length.
    while i < n:                           # Iterate through scores.
        var v = scores[i]                  # Current score.
        var b = 0                          # Default bucket.
        if v >= 90:                        # Bucket thresholds (simple example policy).
            b = 7                          # Top bucket for >= 90.
        elif v >= 80:                      # Next threshold.
            b = 5                          # Middle-high bucket.
        elif v >= 70:                      # Next threshold.
            b = 1                          # Middle-low bucket.
        else:                              # Otherwise...
            b = 0                          # Lowest bucket.
        out.append(String(b))              # Append bucket encoded as string.
        i += 1                             # Next index.
    return out.copy()                      # Return a defensive copy.

fn _argsort_desc_f64(vals: List[Float64]) -> List[Int]:  # Indices that would sort vals in descending order.
    var idx = List[Int]()                  # Index list 0..n-1.
    var n = len(vals)                      # Cache length.
    idx.reserve(n)                         # Reserve capacity.
    var i = 0                              # Build identity indices.
    while i < n:                           # Append 0..n-1.
        idx.append(i)                      # Add current index.
        i += 1                             # Next index.
    var j = 1                              # Insertion sort pass index (stable enough for small n).
    while j < n:                           # For each position j...
        var key_i = idx[j]                 # Candidate index being inserted.
        var key_v = vals[key_i]            # Its corresponding value.
        var k = j - 1                      # Scan left to find insertion point.
        while k >= 0 and vals[idx[k]] < key_v:  # Descending: move while left value is smaller.
            idx[k + 1] = idx[k]            # Shift larger index right.
            k -= 1                         # Move left.
        idx[k + 1] = key_i                 # Place candidate at the found position.
        j += 1                             # Next position.
    return idx.copy()                      # Return a defensive copy of the order.

fn _reindex_int(xs: List[Int], idx: List[Int]) -> List[Int]:  # Reorder Int list xs by index order 'idx'.
    var out = List[Int]()                  # Output list.
    out.reserve(len(idx))                  # Reserve target capacity.
    var i = 0                              # Loop index.
    var n = len(idx)                       # Number of indices.
    while i < n:                           # For each index in order...
        out.append(xs[idx[i]])             # Append the referenced element.
        i += 1                             # Next.
    return out.copy()                      # Return a defensive copy.

fn _reindex_str(xs: List[String], idx: List[Int]) -> List[String]:  # Reorder String list xs by 'idx'.
    var out = List[String]()               # Output list.
    out.reserve(len(idx))                  # Reserve capacity.
    var i = 0                              # Loop index.
    var n = len(idx)                       # Number of indices.
    while i < n:                           # Iterate over indices.
        out.append(xs[idx[i]])             # Append referenced string.
        i += 1                             # Next.
    return out.copy()                      # Return a defensive copy.

# ---------- demo: assign (manual) / pipe(sort) ----------

fn pd_assign_pipe_groupby() -> df.DataFrame:  # Build a DataFrame, derive columns, sort via argsort, and return.
    print("\n=== pd_assign_pipe_groupby ===") # Section banner for console readability.

    # Deterministic pseudo-randomlike input IDs (sequence 0..11).
    var ids_raw = df.range(0, 12, dtype=df.int32())  # Produce List[Int] in [0, 12).

    var scores_raw = List[Int]()             # Prepare a mutable list of integer scores.
    scores_raw.append(82)                    # Append literal score values (fixed dataset).
    scores_raw.append(65)
    scores_raw.append(98)
    scores_raw.append(74)
    scores_raw.append(91)
    scores_raw.append(59)
    scores_raw.append(87)
    scores_raw.append(93)
    scores_raw.append(78)
    scores_raw.append(88)
    scores_raw.append(69)
    scores_raw.append(95)

    var ids = df.Series(ids_raw, dtype=df.int32())     # Wrap ids_raw into a typed Series (Int32).
    var scores = df.Series(scores_raw, dtype=df.int32())  # Wrap scores_raw into a typed Series (Int32).

    # Derived columns: z-scores and coarse buckets computed from the raw Int scores.
    var z_list = make_z_from_ints(scores_raw)          # Compute z-scores (as List[String]).
    var bucket_list = make_bucket_from_ints(scores_raw)  # Compute bucket labels (as List[String]).
    var z = df.Series(z_list, dtype=df.string())       # Wrap z as a string Series.
    var bucket = df.Series(bucket_list, dtype=df.string())  # Wrap bucket as a string Series.

    var T = df.ToDataFrame({                           # Construct a DataFrame with all columns.
        "id": ids.copy(),                              # Use copies to avoid aliasing with inputs.
        "score": scores.copy(),
        "z": z.copy(),
        "bucket": bucket.copy()
    })

    print("Input (with derived columns):")             # Log the initial frame with derived columns.
    print(T.__str__())                                 # Print DataFrame textual representation.
    print("dtypes:")                                   # Label for dtype printout.
    print(T.dtypes().__str__())                        # Print column dtype information.

    # Pipe-like step: sort rows by 'score' descending using manual argsort order.
    var scores_f64 = List[Float64]()                   # Temporary float list for sorting key.
    scores_f64.reserve(len(scores_raw))                # Reserve capacity equal to input length.
    var i = 0                                          # Loop index.
    var n = len(scores_raw)                            # Cache element count.
    while i < n:                                       # Convert Int scores to Float64 for comparison.
        scores_f64.append(Float64(scores_raw[i]))      # Append converted value.
        i += 1                                         # Next index.
    var order = _argsort_desc_f64(scores_f64)          # Obtain indices that sort scores descending.

    var ids_sorted = df.Series(_reindex_int(ids_raw, order), dtype=df.int32())        # Reindex IDs by order.
    var scores_sorted = df.Series(_reindex_int(scores_raw, order), dtype=df.int32())  # Reindex scores by order.
    var z_sorted = df.Series(_reindex_str(z_list, order), dtype=df.string())          # Reindex z strings by order.
    var bucket_sorted = df.Series(_reindex_str(bucket_list, order), dtype=df.string())# Reindex bucket strings.

    var out = df.ToDataFrame({                         # Build the sorted DataFrame.
        "id": ids_sorted.copy(),                       # Copy each Series to avoid aliasing.
        "score": scores_sorted.copy(),
        "z": z_sorted.copy(),
        "bucket": bucket_sorted.copy()
    })

    print("\nAssigned + Piped (sorted by score desc):")# Log the sorted output table header.
    print(out.__str__())                               # Print the sorted DataFrame.
    print("out.dtypes:")                               # Label for the dtypes of the sorted DataFrame.
    print(out.dtypes().__str__())                      # Print dtypes for verification.

    return out                                         # Return the final, sorted DataFrame.

# ---------- main ----------

fn main() -> None:                                     # Program entry point.
    var out = pd_assign_pipe_groupby()                 # Build/run the demo and capture the resulting DataFrame.

    # Top-5 rows by z value in descending order (computed from the string column "z").
    var zr = out.col_values("z")                       # Extract column values for "z" as List[String].
    var zf = List[Float64]()                           # Destination vector for parsed float z values.
    var i = 0                                          # Loop index.
    var n = len(zr)                                    # Number of rows.
    zf.reserve(n)                                      # Reserve capacity to avoid reallocations.
    while i < n:                                       # Parse z strings defensively into floats.
        var v = 0.0                                    # Default to 0.0 on parse failure.
        try:
            v = Float64(zr[i])                         # Attempt to parse string to Float64.
        except _:
            v = 0.0                                    # On failure, keep default 0.0.
        zf.append(v)                                   # Append parsed (or default) value.
        i += 1                                         # Next element.
    var ordz = _argsort_desc_f64(zf)                   # Indices that sort z values descending.

    print("\nTop-5 z values (desc):")                  # Header for top-k display.
    var k = 0                                          # Counter for how many we have printed.
    while k < len(ordz) and k < 5:                     # Limit to 5 values or list length.
        print(String(zf[ordz[k]]))                     # Print the k-th largest z value.
        k += 1                                         # Next position.
