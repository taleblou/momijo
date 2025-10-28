# Project:      Momijo
# Module:       examples.dataframe_setitem
# File:         dataframe_setitem.mojo
# Path:         src/momijo/examples/dataframe_setitem.mojo
#
# Description:  Demonstration of DataFrame column/row assignment patterns:
#               set by name/position, broadcast scalars, masked row updates,
#               and multi-assign from rhs frames.
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
#   - Focus: __setitem__/set_column/set_columns/set_rows variants and behaviors.

import momijo.dataframe as df                # Import the dataframe API under alias 'df' for brevity.
from collections.list import List            # Import typed dynamic array container 'List'.

# ---------- tiny print helpers ----------

fn line(title: String) -> None:              # Print a section separator with a title for readability.
    var sep = String("=") * 80               # Build an 80-char separator line.
    print("\n" + sep)                        # Print a leading newline and the top separator.
    print(title)                             # Print the provided title.
    print(sep)                               # Print the bottom separator.

fn print_kv(k: String, v: String) -> None:   # Print a "key: value" line.
    print(k + ": " + v)                      # Concatenate key, delimiter, and value.

# ---------- build a demo DataFrame ----------

fn make_demo() -> df.DataFrame:              # Construct a small demo DataFrame for the examples.
    var columns:List[String] = (["name", "age", "city", "score", "group"])   # Column names in order.
    var data    = List[List[String]]()       # Allocate outer list to hold per-column string lists.
    data.append((["Alice","Bob","Cathy","Dan","Eve","Frank"]))               # name column values.
    data.append((["25","31","29","40","22","35"]))                           # age column values (as strings).
    data.append((["Helsinki","Turku","Tampere","Oulu","Espoo","Helsinki"]))  # city column values.
    data.append((["88.5","75.0","92.0","66.0","79.0","85.5"]))               # score column values (strings).
    data.append((["A","B","A","B","A","B"]))                                 # group column values.
    var index:List[String]= (["r0","r1","r2","r3","r4","r5"])                # Row index labels.
    var frame = df.DataFrame(columns, data, index, index_name="row_id")      # Build the DataFrame with index name.
    return frame                                                               # Return the constructed frame.

# ---------- helpers to build typed lists quickly ----------

fn list_str2(a: String, b: String) -> List[String]:     # Create a new List[String] with 2 items.
    var xs = List[String](); xs.append(a); xs.append(b); return xs.copy()    # Append and return a copy.

fn list_str5(a: String, b: String, c: String, d: String, e: String) -> List[String]:  # Build 5-item list.
    var xs = List[String](); xs.append(a); xs.append(b); xs.append(c); xs.append(d); xs.append(e); return xs.copy()

fn list_bool6(a: Bool, b: Bool, c: Bool, d: Bool, e: Bool, f: Bool) -> List[Bool]:     # Build 6-item bool list.
    var xs = List[Bool](); xs.append(a); xs.append(b); xs.append(c); xs.append(d); xs.append(e); xs.append(f); return xs.copy()

# ---------- main demo ----------

fn main() -> None:                           # Program entry point.
    var frame = make_demo()                  # Create the initial demo DataFrame.
    line("Initial frame")                    # Section title for initial state.
    print(frame.shape_str())                 # Print shape as a string ("(rows, cols)").
    print(frame.__str__())                   # Print the full table string representation.

    # ============================================================
    # 1) __setitem__(name: String, src: Column)
    # ============================================================
    line("1) Set by name with Column (shape-safe replace/insert)")  # Describe test section.
    var age_col = frame["age"]               # Copy out the "age" column as a Column object.
    age_col.rename("age")                    # Ensure the column's name metadata is aligned.
    frame.set_column(String("age"), age_col) # Replace the "age" column by name with the copied column.
    print(frame["age"].__str__())            # Print the resulting "age" column.

    # ============================================================
    # 2) __setitem__(name: String, values: List[String])
    # ============================================================
    line("2) Set by name with List[String] (tag preserved if exists)")  # Describe list-based set.
    var new_age = List[String]()            # Allocate a fresh list for new age values.
    new_age.append("26"); new_age.append("32"); new_age.append("30")     # Append first three values.
    new_age.append("41"); new_age.append("23"); new_age.append("36")     # Append next three values (total 6).
    # get target tag if column exists; otherwise use string tag = 4
    var tag = 4                              # Default tag (example: string column tag id).
    var pos = frame.find_col("age")          # Locate the "age" column position if it exists.
    if pos >= 0:                             # If found, copy its tag to preserve dtype/metadata.
        tag = frame.cols[pos].tag
    var col_age = df.col_from_list_with_tag(new_age, String("age"), tag)  # Build a Column with name and tag.
    frame.set_column(String("age"), col_age) # Replace "age" column by name with the new Column.
    print(frame["age"].__str__())            # Print the updated "age" column.

    # ============================================================
    # 3) __setitem__(name: String, scalar: String)  -> broadcast
    # ============================================================
    line("3) Set by name with scalar (broadcast)")     # Describe scalar broadcast update.
    var c_group = df.make_broadcast_col_by_name(frame, "group", String("A"))  # Build broadcast Column for "group".
    frame.set_column(String("group"), c_group)         # Set all "group" values to "A".
    print(frame["group"].__str__())                    # Print the updated "group" column.

    # ============================================================
    # 4) __setitem__(idx: Int, src: Column)  (positional replace)
    # ============================================================
    line("4) Set by positional index with Column")     # Describe positional set with Column.
    var city_copy = frame["city"]                      # Copy the "city" column.
    # position 2 is "city" in this demo; replace it with the same data to test path
    frame.set_column(Int(2), city_copy)                # Replace column at index 2 (positional) with copy.
    print(frame[Int(2)].__str__())                     # Print the column now at position 2.

    # ============================================================
    # 5) __setitem__(names: List[String], rhs: DataFrame)
    #    (match by name; ignore missing; shape must match)
    # ============================================================
    line("5) Multi-assign by names from rhs frame (match by name)")  # Describe multi-assign by names.
    var rhs_by_name = frame.loc(df.rows_all(), ["name", "city"])     # Build rhs with matching columns.
    var targets_by_name: List[String] = ["name", "city"]              # Target columns to update.
    frame.set_columns(targets_by_name, rhs_by_name)                   # Assign by matching names.
    print(frame.loc(df.rows_all(), list_str2("name","city")).__str__())  # Print selected columns to verify.

    # ============================================================
    # 6) __setitem__(indices: List[Int], rhs: DataFrame)
    #    (by position; keep target names; shape must match)
    # ============================================================
    line("6) Multi-assign by positions from rhs frame (positional match)")  # Describe positional multi-assign.
    # Build an rhs frame with at least first 3 columns so idx [0,1,2] is valid
    var rhs_pos = frame.loc(df.rows_all(), df.cols_all())             # Full-width rhs with same shape.
    var idxs: List[Int] = [0, 1, 2]                                   # Positional targets to update.
 
    frame.set_columns(idxs, rhs_pos)                                  # Assign columns at indices [0,1,2].
    print(frame.loc(df.rows_all(), list_str2("name","age")).__str__())# Verify some affected columns.
    print(frame[Int(2)].__str__())                                    # Print column at index 2 for inspection.

    # ============================================================
    # 7) __setitem__(mask: List[Bool], scalar: String)
    #    (row-wise masked broadcast across all columns)
    # ============================================================
    line("7) Masked row assignment with scalar (broadcast across all columns)")  # Describe masked broadcast.
    var m: List[Bool] = [False, True, False, True, False, True]       # Row mask selecting r1, r3, r5.
    frame.set_rows(m, String("<masked>"))                             # Replace all columns on masked rows.
    print(frame.__str__())                                            # Print frame to show masked replacements.

    # ============================================================
    # 8) __setitem__(name: String, rhs: DataFrame)
    #    (prefer same-name col; else if rhs has exactly one column, use it)
    # ============================================================
    line("8) Set by name with rhs frame (same-name or single-column)")  # Describe name-based rhs logic.
    # Case (a): rhs has same-name column
    var rhs_same = frame.loc(df.rows_all(), ["city"])                 # rhs with the same-name column "city".
    frame.set_column(String("city"), rhs_same)   # ✅ unambiguous       # Assign "city" from rhs with same name.
    print(frame["city"].__str__())                                    # Print updated "city" column.

    # Case (b): rhs has one column only (renamed on the fly)
    var rhs_one = frame.loc(df.rows_all(), ["name"])                  # rhs with single column "name".
    frame.set_column(String("alias"), rhs_one)                        # Assign into a new/other name "alias".
    print(frame["alias"].__str__())                                   # Print new "alias" column.

    # ============================================================
    # 9) __setitem__(idx: Int, values: List[String])  (positional + list)
    # ============================================================
    line("9) Set by positional index with List[String] (preserve tag)")  # Describe positional + list set.
    var alias_vals = List[String]()                                     # Prepare replacement values.
    alias_vals.append("A1"); alias_vals.append("B1"); alias_vals.append("C1")  # First 3 entries.
    alias_vals.append("D1"); alias_vals.append("E1"); alias_vals.append("F1")  # Next 3 entries (nrows=6).
    # Find "alias" position to demonstrate positional set
    var alias_pos = frame.find_col("alias")                             # Locate "alias" column index.
    if alias_pos >= 0:                                                  # If present, set by position.
        frame.set_column(Int(alias_pos), alias_vals)                    # Replace at found index using list.
        print(frame["alias"].__str__())                                 # Print updated "alias" column.

    # ============================================================
    # 10) __setitem__(idx: Int, scalar: String)  (positional broadcast)
    # ============================================================
    line("10) Set by positional index with scalar (broadcast)")         # Describe positional scalar broadcast.
    var score_pos = frame.find_col("score")                             # Find "score" column index.
    if score_pos >= 0:                                                  # If found, broadcast replace.
        frame.set_column(Int(score_pos), String("0.0"))                 # Set entire column to "0.0".
        print(frame["score"].__str__())                                 # Print updated "score" column.

    # ============================================================
    # 11) Enhanced names-assign: rhs has exactly one column -> reuse for all
    # ============================================================
    line("11) Enhanced names-assign: single rhs col reused across targets")  # Describe reuse of single rhs col.
    var only_city = frame.loc(df.rows_all(), ["city"])                   # rhs with exactly one column.
    var targets_multi = list_str2("name", "group")                       # Two target columns to update.
    frame.set_columns(targets_multi, only_city)                          # Copy rhs column into both targets.
    print(frame.loc(df.rows_all(), targets_multi).__str__())             # Verify both columns updated.

    # ============================================================
    # 12) Enhanced indices-assign: rhs has exactly one column -> reuse across indices
    # ============================================================
    line("12) Enhanced indices-assign: single rhs col reused across positions")  # Describe positional reuse.
    var onecol_rhs = frame.loc(df.rows_all(), ["alias"])                 # rhs with one column ("alias").
    var pos_multi = List[Int](); pos_multi.append(0); pos_multi.append(2)   # Positions to update (e.g., name & city).
    frame.set_columns(pos_multi, onecol_rhs)                             # Reuse single rhs col across positions.
    print(frame.loc(df.rows_all(), list_str2("name","city")).__str__())  # Verify updated columns.

    # ============================================================
    # 13) __setitem__(mask: List[Bool], rhs: DataFrame)
    #     (mode=1: rhs.ncols == self.ncols ; mode=2: rhs.ncols == 1)
    # ============================================================
    line("13) Masked row assignment from rhs frame (positional or single-col broadcast)")  # Describe masked rhs set.
    var mask2 = list_bool6(True, False, True, False, True, False)  # r0,r2,r4 rows selected.
    # mode=2: single-column rhs -> broadcast across columns on True rows
    var rhs_single = frame.loc(df.rows_all(), ["alias"])           # rhs with exactly one column.
    frame.set_rows(mask2, rhs_single)                              # Broadcast rhs column into masked rows.
    print(frame.__str__())                                         # Print to verify changes.

    # mode=1: same number of columns -> per-position replacement on True rows
    var rhs_full = frame.loc(df.rows_all(), df.cols_all())         # rhs with same number of columns as self.
    var mask3 = list_bool6(False, True, False, True, False, False) # r1,r3 rows selected.
    frame.set_rows(mask3, rhs_full)                                # Replace masked rows with rhs rows (positional).
    print(frame.__str__())                                         # Print to verify changes.

    # ============================================================
    # 14) __setitem__(mask: List[Bool], row_values: List[String])
    #     row_values must match ncols()
    # ============================================================
    line("14) Masked row assignment with List[String] as a whole row")  # Describe masked row-values set.
    var rowvals = list_str5("X","99","Nowhere","-1.0","Z")              # Exactly ncols()=5 values for one row.
    var mask4 = list_bool6(False, False, True, False, False, True)      # Mask selecting r2 and r5.
    frame.set_rows(mask3, rowvals)   # Note: mask3 used here while mask4 is defined above (left as-is intentionally).
    print(frame.__str__())                                               # Print to verify masked row updates.

    # Final sanity
    line("Final sanity")                              # Final section separator.
    print_kv("shape", frame.shape_str())              # Print final shape for quick check.
    print(frame.__str__())                            # Print final DataFrame state.
