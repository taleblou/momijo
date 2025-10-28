# Project:      Momijo
# Module:       examples.dataframe_rolling_ewm
# File:         dataframe_rolling_ewm.mojo
# Path:         src/momijo/examples/dataframe_rolling_ewm.mojo
#
# Description:  Rolling/Expanding/EWM demonstrations for momijo.dataframe with debug prints.
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
#   - Shows: rolling mean/sum/std (nullable outputs), expanding mean, EWM mean/var using alpha vs span,
#            and building a nullable summary DataFrame for inspection.

import momijo.dataframe as df                     # Import the dataframe API under alias 'df'.
from collections.list import List                 # Import List for typed dynamic arrays used below.

# ---------------- pretty printers ---------------- #

fn list_f64_to_string(xs: List[Float64]) -> String:   # Convert a List[Float64] to a JSON-like string.
    var sb = String("[")                              # Start a string builder with an opening bracket.
    var n = len(xs)                                   # Cache the length to avoid repeated len() calls.
    var i = 0                                         # Loop counter.
    while i < n:                                      # Iterate through all elements.
        sb += String(xs[i])                           # Append current value as String.
        if i + 1 < n:                                 # If not the last element, add a comma+space.
            sb += String(", ")
        i += 1                                        # Increment loop counter.
    sb += String("]")                                 # Close the bracket.
    return sb                                         # Return the constructed string.

fn list_opt_f64_to_string(xs: List[Optional[Float64]]) -> String:  # Stringify nullable float list.
    var sb = String("[")                              # Begin with an opening bracket.
    var n = len(xs)                                   # Cache length.
    var i = 0                                         # Loop counter.
    while i < n:                                      # Iterate through elements.
        var o = xs[i]                                 # Fetch the Optional at index i.
        if o is None:                                 # If value is null/None...
            sb += String("null")                      # Append the literal "null".
        else:                                         # Otherwise...
            sb += String(o.value())                   # Append the inner Float64 value.
        if i + 1 < n:                                 # Add comma+space between items.
            sb += String(", ")
        i += 1                                        # Increment loop.
    sb += String("]")                                 # Close the bracket.
    return sb                                         # Return the final string.

fn to_opt_f64(values: List[Float64]) -> List[Optional[Float64]]:  # Wrap floats as Optional[Float64].
    var out = List[Optional[Float64]]()               # Allocate an output list of optionals.
    out.reserve(len(values))                          # Reserve capacity to avoid reallocations.
    var i = 0                                         # Loop counter.
    while i < len(values):                            # Iterate over input values.
        out.append(Optional[Float64](values[i]))      # Wrap each float in an Optional and append.
        i += 1                                        # Increment loop.
    return out.copy()                                 # Return a defensive copy (ownership safety).

# ---------------- tiny debug helpers ---------------- #

fn dbg_len(label: String, n: Int):                    # Print a debug line showing a length.
    print("[DBG] len(" + label + ") = " + String(n))  # Format: [DBG] len(label) = n

fn dbg_head_f64(label: String, xs: List[Float64], k: Int):  # Print first k float values.
    var n = len(xs)                                   # Compute total length.
    var m = k                                         # Desired head size.
    if n < m: m = n                                   # Clamp to available length.
    var sb = String(label + " head " + String(m) + ": [")  # Begin header and opening bracket.
    var i = 0                                         # Loop counter.
    while i < m:                                      # Iterate through the head slice.
        sb += String(xs[i])                           # Append value.
        if i + 1 < m: sb += String(", ")             # Add comma between items.
        i += 1                                        # Increment loop.
    sb += String("]")                                  # Close bracket.
    print(sb)                                         # Emit the debug line.

fn dbg_head_opt_f64(label: String, xs: List[Optional[Float64]], k: Int):  # Print first k optional floats.
    var n = len(xs)                                   # Compute total length.
    var m = k                                         # Desired head size.
    if n < m: m = n                                   # Clamp to available length.
    var sb = String(label + " head " + String(m) + ": [")  # Start header + bracket.
    var i = 0                                         # Loop counter.
    while i < m:                                      # Iterate through head portion.
        var o = xs[i]                                 # Get optional value.
        if o is None: sb += String("null")            # Print "null" for missing values.
        else: sb += String(o.value())                 # Otherwise print the contained float.
        if i + 1 < m: sb += String(", ")              # Comma between items.
        i += 1                                        # Increment.
    sb += String("]")                                  # Close bracket.
    print(sb)                                         # Emit debug string.

fn dbg_count_nulls(label: String, xs: List[Optional[Float64]]):  # Count nulls in an optional list.
    var n = len(xs)                                   # Total length.
    var c = 0                                         # Null counter.
    var i = 0                                         # Loop counter.
    while i < n:                                      # Iterate through values.
        if xs[i] is None: c += 1                      # Increment counter if element is None.
        i += 1                                        # Advance.
    print("[DBG] nulls(" + label + ") = " + String(c) + " / " + String(n))  # Print summary.

# ---------------- rolling / expanding / ewm ---------------- #

fn pd_rolling_expanding_ewm() -> df.DataFrame:       # Build a demo and return a summary DataFrame.
    print("\n=== pd_rolling_expanding_ewm ===")      # Section header for readability.

    # Base series: 0..9 as float64
    var s = df.range_f64(0, 10)                       # Produce [0.0, 1.0, ..., 9.0] as Float64.
    print("base series:")                             # Label for base series output.
    print(list_f64_to_string(s))                      # Print the series using the helper.
    dbg_len("s", len(s))                              # Debug: length of s.
    dbg_head_f64("s", s, 5)                           # Debug: first 5 values of s.

    # Rolling mean with window=3 (produces nulls for first 2 positions)
    var window = 3                                    # Window size for rolling operations.
    var roll3_mean = df.rolling_mean_f64(s, window)   # Nullable: first window-1 entries are None.
    print("\nrolling mean (window=3):")               # Label for rolling mean output.
    print(list_opt_f64_to_string(roll3_mean))         # Print nullable list as string.
    dbg_len("roll3_mean", len(roll3_mean))            # Debug: length.
    dbg_count_nulls("roll3_mean", roll3_mean)         # Debug: count leading/trailing nulls.
    dbg_head_opt_f64("roll3_mean", roll3_mean, 6)     # Debug: preview first 6 values.

    # Expanding mean with min_periods=1 (no leading nulls)
    var exp_mean = df.expanding_mean_f64(s, 1)        # Expanding mean; min_periods=1 yields no nulls.
    print("\nexpanding mean (tail 5):")               # Label for expanding mean output.
    print(list_f64_to_string(exp_mean))               # Print full vector (small size is fine).
    dbg_len("exp_mean", len(exp_mean))                # Debug: length.
    dbg_head_f64("exp_mean", exp_mean, 6)             # Debug: first 6 values.

    # EWM: IMPORTANT — are we passing alpha or span?
    var span = 3                                      # Choose a span to demonstrate API differences.
    print("\n[DBG] Calling ewm_mean_f64 with alpha-or-span? span=" + String(span) + " (WARNING: ewm_mean_f64 expects alpha in (0,1])")
                                                      # Warn that ewm_mean_f64 expects alpha, not span.
    var ewm_mean = df.ewm_mean_f64(s, span, False)    # Intentionally misuse: passing span (not alpha).
                                                      # Some implementations may return empty/incorrect output.
    print("\newm(span=3, adjust=False) mean (head 5):")  # Label for EWM with span misused as alpha.
    print(list_f64_to_string(ewm_mean))               # Print returned values (may look odd).
    dbg_len("ewm_mean(returned)", len(ewm_mean))      # Debug: length of returned vector.
    dbg_head_f64("ewm_mean(returned)", ewm_mean, 5)   # Debug: first 5 values.

    # Also try the correct span-based wrapper if available
    print("[DBG] Trying span-wrapper ewm_mean_f64_span (expected to work)")  # Announce correct call.
    var ewm_mean_span = df.ewm_mean_f64_span(s, span, False)  # Proper span-based wrapper (convert span→alpha).
    dbg_len("ewm_mean_span", len(ewm_mean_span))      # Debug: length should match s.
    dbg_head_f64("ewm_mean_span", ewm_mean_span, 5)   # Debug: preview first 5 values.

    # A few extra useful stats for completeness
    var roll3_sum = df.rolling_sum_f64(s, window)     # Rolling sum (nullable head).
    var roll3_std = df.rolling_std_f64(s, window, 0)  # Rolling std; bias flag 0 for sample/pop as defined.

    # Check nullable types for these:
    dbg_len("roll3_sum", len(roll3_sum))              # Debug length of rolling sum.
    dbg_len("roll3_std", len(roll3_std))              # Debug length of rolling std.
    dbg_head_opt_f64("roll3_sum", roll3_sum, 6)       # Preview first 6 rolling sums (nullable).
    dbg_head_opt_f64("roll3_std", roll3_std, 6)       # Preview first 6 rolling stds (nullable).
    dbg_count_nulls("roll3_sum", roll3_sum)           # Count nulls in rolling sum.
    dbg_count_nulls("roll3_std", roll3_std)           # Count nulls in rolling std.

    # EWM variance — both alpha-misuse and span-wrapper paths
    print("[DBG] Calling ewm_var_f64 with span as alpha (expect failure/empty)")  # Warn about misuse again.
    var ewm_var_bad = df.ewm_var_f64(s, span, False)  # Incorrect call: passing span as alpha.
    dbg_len("ewm_var(returned)", len(ewm_var_bad))    # Debug returned length (may be wrong/empty).
    dbg_head_f64("ewm_var(returned)", ewm_var_bad, 5) # Preview values for awareness.

    print("[DBG] Trying span-wrapper ewm_var_f64_span (expected to work)")    # Announce proper variant.
    var ewm_var = df.ewm_var_f64_span(s, span, False) # Correct call via span wrapper.
    dbg_len("ewm_var_span", len(ewm_var))             # Debug: should match s length.
    dbg_head_f64("ewm_var_span", ewm_var, 5)          # Preview first 5 values.

    # Build an aligned summary table (nullable float columns where needed)
    print("[DBG] Building summary via ToDataFrameNullable; verifying column lengths first...")  # Explain next step.
    var n_s = len(s)                                   # Gather lengths for validation logs.
    var n_rm = len(roll3_mean)
    var n_rs = len(roll3_sum)
    var n_rstd = len(roll3_std)
    var n_em = len(exp_mean)
    var n_emspan = len(ewm_mean_span)
    var n_evspan = len(ewm_var)

    print("[DBG] lens -> s=" + String(n_s) +          # Emit lengths to ensure alignment prior to framing.
          ", roll3_mean=" + String(n_rm) +
          ", roll3_sum=" + String(n_rs) +
          ", roll3_std=" + String(n_rstd) +
          ", exp_mean=" + String(n_em) +
          ", ewm_mean_span=" + String(n_emspan) +
          ", ewm_var_span=" + String(n_evspan))

    var summary = df.ToDataFrameNullable({            # Construct a DataFrame with nullable columns.
        "x":            to_opt_f64(s.copy()),         # Base x values as Optional for uniformity.
        "roll3_mean":   roll3_mean.copy(),            # Rolling mean (already nullable).
        "roll3_sum":    roll3_sum.copy(),             # Rolling sum (nullable).
        "roll3_std":    roll3_std.copy(),             # Rolling std (nullable).
        "exp_mean":     to_opt_f64(exp_mean.copy()),  # Wrap expanding mean into Optional.
        "ewm_mean_s3":  to_opt_f64(ewm_mean_span.copy()),  # EWM mean (span=3) wrapped as Optional.
        "ewm_var_s3":   to_opt_f64(ewm_var.copy())    # EWM variance (span=3) wrapped as Optional.
    })                                                # Close the mapping and call.

    print("[DBG] summary constructed")                 # Confirmation log.

    print("\nsummary (head):")                         # Label for head preview.
    print(summary.__str__())                           # Print the head (implementation may print full small DF).
    print("dtypes:")                                   # Label for dtypes preview.
    print(summary.dtypes().__str__())                  # Print dtypes for verification.

    return summary                                     # Return the assembled summary DataFrame.

# ---------------- main ---------------- #

fn main() -> None:                                     # Program entry point.
    var summary = pd_rolling_expanding_ewm()           # Run the demonstration and capture the DataFrame.

    print("\nsummary (tail):")                         # Label for final preview (tail or full, impl-dependent).
    print(summary.__str__())                           # Print again to show final state.
