# Project:      Momijo
# Module:       examples.dataframe_setitem_assign
# File:         dataframe_setitem_assign.mojo
# Path:         src/momijo/examples/dataframe_setitem_assign.mojo
#
# Description:  Demonstrates DataFrame column/row assignment modes:
#               by name/position, scalar broadcast, masked updates,
#               multi-assign (names/indices), and rhs frame reuse.
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
#   - Focus: __setitem__/set_column/set_columns/set_rows behaviors.
#   - Input: in-memory demo frame; no file I/O required.

import momijo.dataframe as df                 # Import the dataframe API under alias 'df'.
from collections.list import List             # Import List for typed dynamic arrays.

# ---------- tiny print helpers ----------
fn line(title: String):                       # Print a section banner for readability.
    var sep = String("=") * 80                # Build a separator line of '=' characters.
    print("\n" + sep)                         # Print a leading newline and the top separator.
    print(title)                              # Print the section title.
    print(sep)                                # Print the bottom separator.

fn print_kv(k: String, v: String):            # Print a key-value pair as "k: v".
    print(k + ": " + v)                       # Concatenate and print.

# ---------- build a demo DataFrame ----------
fn make_demo() -> df.DataFrame:               # Create a small typed DataFrame for the demo.
    var columns:List[String] = (["name", "age", "city", "score", "group"])  # Column names (strings).
    var data    = List[List[String]]()        # Column-major storage as List of String columns.
    data.append((["Alice","Bob","Cathy","Dan","Eve","Frank"]))               # name column.
    data.append((["25","31","29","40","22","35"]))                           # age column (as strings).
    data.append((["Helsinki","Turku","Tampere","Oulu","Espoo","Helsinki"]))  # city column.
    data.append((["88.5","75.0","92.0","66.0","79.0","85.5"]))               # score column (as strings).
    data.append((["A","B","A","B","A","B"]))                                 # group column.
    var index:List[String]= (["r0","r1","r2","r3","r4","r5"])                # Row labels as strings.
    var frame = df.DataFrame(columns, data, index, index_name="row_id")      # Construct the DataFrame.
    return frame                                                               # Return the demo frame.

# ---------- helpers to build typed lists quickly ----------
fn list_str2(a: String, b: String) -> List[String]:  # Build a List[String] with 2 items.
    var xs = List[String](); xs.append(a); xs.append(b); return xs.copy()    # Append and return a copy.

fn list_str5(a: String, b: String, c: String, d: String, e: String) -> List[String]:  # List of 5 strings.
    var xs = List[String](); xs.append(a); xs.append(b); xs.append(c); xs.append(d); xs.append(e); return xs.copy()

fn list_str6(a: String, b: String, c: String, d: String, e: String, f: String) -> List[String]:  # List of 6 strings.
    var xs = List[String](); xs.append(a); xs.append(b); xs.append(c); xs.append(d); xs.append(e); xs.append(f); return xs.copy()

fn list_bool6(a: Bool, b: Bool, c: Bool, d: Bool, e: Bool, f: Bool) -> List[Bool]:  # List[Bool] of length 6.
    var xs = List[Bool](); xs.append(a); xs.append(b); xs.append(c); xs.append(d); xs.append(e); xs.append(f); return xs.copy()

# ---------- main demo ----------
fn main():                                     # Program entry point (no args).
    var frame = make_demo()                    # Build the starting demo DataFrame.
    line("Initial frame")                      # Section: initial state.
    print(frame.shape_str())                   # Show shape as a formatted string.
    print(frame.__str__())                     # Pretty-print the full frame.

    # ============================================================
    # 1) __setitem__(name: String, src: Column)
    # ============================================================
    line("1) Set by name with Column (shape-safe replace/insert)")  # Section header.
    var age_col = frame["age"]               # Extract a Column view by name.
    age_col.rename("age")                    # Ensure the column has the intended name.
    frame.set_column(String("age"), age_col) # Replace/insert the 'age' column by name using a Column.
    print(frame["age"].__str__())            # Print the resulting 'age' column.

    # ============================================================
    # 2) __setitem__(name: String, values: List[String])
    # ============================================================
    line("2) Set by name with List[String] (preserve tag if exists)")  # Section header.
    var new_age = List[String]()            # Prepare replacement values for 'age'.
    new_age.append("26"); new_age.append("32"); new_age.append("30")   # Append first three values.
    new_age.append("41"); new_age.append("23"); new_age.append("36")   # Append next three values.
    var tag = 4                              # Default type tag (example fallback value).
    var pos = frame.find_col("age")          # Find the position of the 'age' column if it exists.
    if pos >= 0: tag = frame.cols[pos].tag   # If found, preserve its existing tag metadata.
    var col_age = df.col_from_list_with_tag(new_age, String("age"), tag)  # Build a Column with the tag.
    frame.set_column(String("age"), col_age) # Assign by name using a constructed Column.
    print(frame["age"].__str__())            # Print the updated 'age' column.

    # ============================================================
    # 3) __setitem__(name: String, scalar: String) -> broadcast
    # ============================================================
    line("3) Set by name with scalar (broadcast)")  # Section header.
    var c_group = df.make_broadcast_col_by_name(frame, "group", String("A"))  # Create a broadcast Column.
    frame.set_column(String("group"), c_group)       # Replace 'group' with broadcast scalar 'A'.
    print(frame["group"].__str__())                  # Print the updated 'group' column.

    # ============================================================
    # 4) __setitem__(idx: Int, src: Column)  (positional replace)
    # ============================================================
    line("4) Set by positional index with Column")   # Section header.
    var city_copy = frame["city"]                    # Extract the 'city' column.
    frame.set_column(Int(2), city_copy)              # Replace column at position 2 with it.
    print(frame[Int(2)].__str__())                   # Print column at position 2 to verify.

    # ============================================================
    # 5) __setitem__(names: List[String], rhs: DataFrame)
    #    (match by name; ignore missing; shape must match)
    # ============================================================
    line("5) Multi-assign by names from rhs frame (match by name)")   # Section header.
    var rhs_by_name = frame.loc(df.rows_all(), ["name", "city"])      # Slice rhs with selected columns.
    var targets_by_name: List[String] = ["name", "city"]               # Target columns to overwrite.
    frame.set_columns(targets_by_name, rhs_by_name)                    # Assign by matching names.
    print(frame.loc(df.rows_all(), list_str2("name","city")).__str__())# Print the two targets.

    # ============================================================
    # 6) __setitem__(indices: List[Int], rhs: DataFrame)
    #    (by position; keep target names; shape must match)
    # ============================================================
    line("6) Multi-assign by positions from rhs frame (positional match)")  # Section header.
    var rhs_pos = frame.loc(df.rows_all(), frame.col_names)            # rhs with same shape/cols as frame.
    var idxs: List[Int] = [0, 1, 2]                                    # Positions to overwrite.
    frame.set_columns(idxs, rhs_pos)                                   # Assign by index positions.
    print(frame.loc(df.rows_all(), list_str2("name","age")).__str__()) # Print a subset to verify.
    print(frame[Int(2)].__str__())                                     # Print column at pos 2.

    # ============================================================
    # 7) __setitem__(mask: List[Bool], scalar: String)
    #    (row-wise masked broadcast across all columns)
    # ============================================================
    line("7) Masked row assignment with scalar (broadcast across all columns)")  # Section header.
    var m: List[Bool] = [False, True, False, True, False, True]         # Mask selecting rows 1,3,5.
    frame.set_rows(m, String("<masked>"))                                # Broadcast scalar across masked rows.
    print(frame.__str__())                                               # Print full frame to verify.

    # ============================================================
    # 8) __setitem__(name: String, rhs: DataFrame)
    #    (prefer same-name col; else if rhs has exactly one column, use it)
    # ============================================================
    line("8) Set by name with rhs frame (same-name or single-column)")  # Section header.
    # Case (a): rhs has same-name column
    var rhs_same = frame.loc(df.rows_all(), ["city"])                   # rhs with the 'city' column.
    frame.set_column(String("city"), rhs_same)                          # Assign 'city' from rhs (by name).
    print(frame["city"].__str__())                                      # Verify 'city'.

    # Case (b): rhs has one column only (renamed on the fly)
    var rhs_one = frame.loc(df.rows_all(), ["name"])                    # rhs with single 'name' column.
    frame.set_column(String("alias"), rhs_one)                          # Assign into new/existing 'alias'.
    print(frame["alias"].__str__())                                     # Verify 'alias'.

    # ============================================================
    # 9) __setitem__(idx: Int, values: List[String])  (positional + list)
    # ============================================================
    line("9) Set by positional index with List[String] (preserve tag)") # Section header.
    var alias_vals = List[String]()                                     # Prepare new values for 'alias'.
    alias_vals.append("A1"); alias_vals.append("B1"); alias_vals.append("C1")  # First three values.
    alias_vals.append("D1"); alias_vals.append("E1"); alias_vals.append("F1")  # Next three values.
    var alias_pos = frame.find_col("alias")                             # Locate 'alias' column position.
    if alias_pos >= 0:                                                  # Only proceed if found.
        frame.set_column(Int(alias_pos), alias_vals)                    # Replace by position with list values.
        print(frame["alias"].__str__())                                 # Verify 'alias'.

    # ============================================================
    # 10) __setitem__(idx: Int, scalar: String)  (positional broadcast)
    # ============================================================
    line("10) Set by positional index with scalar (broadcast)")         # Section header.
    var score_pos = frame.find_col("score")                             # Locate 'score' column position.
    if score_pos >= 0:                                                  # Proceed if found.
        frame.set_column(Int(score_pos), String("0.0"))                 # Broadcast scalar into that column.
        print(frame["score"].__str__())                                 # Verify 'score'.

    # ============================================================
    # 11) names-assign: rhs has exactly one column -> reuse for all
    # ============================================================
    line("11) names-assign: single rhs col reused across targets")      # Section header.
    var only_city = frame.loc(df.rows_all(), ["city"])                  # rhs with single 'city' column.
    var targets_multi = list_str2("name", "group")                      # Target columns to overwrite.
    frame.set_columns(targets_multi, only_city)                         # Reuse rhs single column for all targets.
    print(frame.loc(df.rows_all(), targets_multi).__str__())            # Verify both targets.

    # ============================================================
    # 12) indices-assign: rhs has exactly one column -> reuse across indices
    # ============================================================
    line("12) indices-assign: single rhs col reused across positions")  # Section header.
    var onecol_rhs = frame.loc(df.rows_all(), ["alias"])                # rhs with single 'alias' column.
    var pos_multi = List[Int](); pos_multi.append(0); pos_multi.append(2)   # Positions 0 and 2 (name & city).
    frame.set_columns(pos_multi, onecol_rhs)                            # Reuse single rhs column across indices.
    print(frame.loc(df.rows_all(), list_str2("name","city")).__str__()) # Verify name & city.

    # ============================================================
    # 13) __setitem__(mask: List[Bool], rhs: DataFrame)
    #     (mode=1: rhs.ncols == self.ncols ; mode=2: rhs.ncols == 1)
    # ============================================================
    line("13) Masked row assignment from rhs frame (single-col broadcast then full positional)")  # Header.
    var mask2 = list_bool6(True, False, True, False, True, False)       # Mask selecting rows r0, r2, r4.
    var rhs_single = frame.loc(df.rows_all(), ["alias"])                # Single-column rhs.
    frame.set_rows(mask2, rhs_single)                                   # Broadcast that column across masked rows.
    print(frame.__str__())                                              # Verify full frame.

    var rhs_full = frame.loc(df.rows_all(), frame.col_names)            # Full-shape rhs (all columns).
    var mask3 = list_bool6(False, True, False, True, False, False)      # Mask selecting rows r1, r3.
    frame.set_rows(mask3, rhs_full)                                     # Row-wise positional replace on masked rows.
    print(frame.__str__())                                              # Verify changes.

    # ============================================================
    # 14) __setitem__(mask: List[Bool], row_values: List[String])
    #     row_values must match ncols()
    # ============================================================
    line("14) Masked row assignment with List[String] as a whole row")  # Section header.
    # frame now has 6 columns: name, age, city, score, group, alias
    var rowvals = list_str6("X","99","Nowhere","-1.0","Z","ALIAS_X")    # Provide one full-row of values.
    var mask4 = list_bool6(False, False, True, False, False, True)      # Apply to rows r2 and r5.
    frame.set_rows(mask4, rowvals)                                      # Replace those rows entirely.
    print(frame.__str__())                                              # Verify frame content.

    # Final sanity
    line("Final sanity")                                                # Section header.
    print_kv("shape", frame.shape_str())                                # Print the final shape.
    print(frame.__str__())                                              # Print the final frame.
