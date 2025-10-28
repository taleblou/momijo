# Project:      Momijo
# Module:       examples.dataframe_indexing
# File:         dataframe_indexing.mojo
# Path:         src/momijo/examples/dataframe_indexing.mojo
#
# Description:  Demonstration of DataFrame indexing/selecting patterns: [], loc, iloc, masks, and slices.
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
#   - Shows: single/multi column selection, boolean row filtering,
#            label/positional slices, chaining loc/iloc, and simple sanity checks.

import momijo.dataframe as df              # Import the dataframe API under alias 'df' for concise calls.
from collections.list import List          # Import List to build typed arrays for demo inputs and masks.

# ---------- tiny printing helpers ----------
fn line(title: String):                    # Print a section header with separators around the given title.
    var sep = String("=") * 80             # Build a separator line of '=' characters with fixed width.
    print("\n" + sep)                      # Print a blank line followed by the top separator.
    print(title)                           # Print the section title text.
    print(sep)                             # Print the bottom separator.

fn show(title: String, s: String):         # Print a labeled single-line value for quick inspection.
    print("[" + title + "] " + s)          # Format as "[Title] value" for readability.

# Optional tiny asserts (string only, to avoid raising)
fn assert_eq_str(a: String, b: String, label: String):  # Compare two strings and log a failure message if unequal.
    if a != b:                                          # If the values do not match...
        print("[ASSERT FAIL] " + label + ": " + a + " != " + b)  # ...emit a non-throwing assertion message.

# Build the demo DataFrame
fn make_demo() -> df.DataFrame:            # Create a small DataFrame with string-backed columns and explicit index.
    var columns:List[String]=(["name", "age", "city", "score", "group"])  # Column names for the frame schema.
    var data    = List[List[String]]()     # 2D data container: list of column arrays (all strings here).
    data.append((["Alice","Bob","Cathy","Dan","Eve","Frank"]))            # name column values
    data.append((["25","31","29","40","22","35"]))                        # age column values (string-typed)
    data.append((["Helsinki","Turku","Tampere","Oulu","Espoo","Helsinki"]))  # city column values
    data.append((["88.5","75.0","92.0","66.0","79.0","85.5"]))            # score column values (string-typed)
    data.append((["A","B","A","B","A","B"]))                              # group column values
    var index: List[String]=(["r0","r1","r2","r3","r4","r5"])             # Explicit string index labels for rows.
    var frame = df.DataFrame(columns, data, index, index_name="row_id")   # Construct the DataFrame with metadata.
    return frame                                                           # Return the demo frame to callers.

# Build boolean mask score >= 85.0 using string-backed values
fn ge_85_mask(frame: df.DataFrame) -> List[Bool]:  # Compute a boolean mask where 'score' >= 85.0 (string to float).
    var n = frame.nrows()                          # Number of rows to iterate over.
    var mask = List[Bool]()                        # Allocate an output mask (Bool per row).
    mask.reserve(n)                                # Reserve capacity to avoid reallocations during append.
    var c = frame["score"]                         # Access the 'score' column (string-backed in this demo).
    var r = 0                                      # Row counter initializer.
    while r < n:                                   # Iterate across rows.
        var s = c.get_string(r)                    # Read the row value as a String.
        var ok = True                              # Track whether parsing succeeds.
        var val = 0.0                              # Placeholder for parsed float value.
        try:                                       # Attempt to parse the string as Float64.
            val = Float64(s)                       # Convert string to numeric value.
        except _:                                  # If parsing fails...
            ok = False                             # ...mark as not okay to exclude from threshold logic.
        if ok and val >= 85.0:                     # If parsed and meets threshold (>= 85.0)...
            mask.append(True)                      # ...set mask at this row to True.
        else:                                      # Otherwise...
            mask.append(False)                     # ...set mask to False.
        r += 1                                     # Advance to the next row.
    return mask.copy()                             # Return a defensive copy of the mask.

# ---------------------------------------------------------------------
# main demo
# ---------------------------------------------------------------------
fn main():                                         # Program entry point.
    var frame = make_demo()                        # Build the sample DataFrame for all demonstrations.

    line("Input frame")                            # Section header for the initial DataFrame.
    print(frame.shape_str())                       # Print the shape as a compact string, e.g. "(6, 5)".
    print(frame.__str__())                         # Print the formatted table (assumes DataFrame implements __str__).

    # =============== [] basic indexing =================
    line("[]: single column by name")              # Section: bracket indexing with a single column name.
    var name_col = frame["name"]                   # Select a single column by its name, returns a Column-like object.
    show("col name", name_col.get_name())          # Print the resolved column name.
    print(name_col.__str__())                      # Print the column contents in tabular/text form.

    line("[]: single column by index (0-based)")   # Section: bracket indexing with a 0-based position.
    var city_col = frame[Int(2)]                   # Select the 3rd column via explicit Int cast to disambiguate API.
    show("col name", city_col.get_name())          # Print the chosen column's name.
    print(city_col.__str__())                      # Print the selected column.

    line("[]: multi columns by names -> DataFrame")# Section: multi-select by a list of names.
    var by_names = frame[["city","score"]]         # Select two columns by name, returning a new DataFrame.
    print(by_names.shape_str())                    # Show the shape of the resulting 2-column frame.
    print(by_names.__str__())                      # Print the resulting frame.

    line("[]: multi columns by positions -> DataFrame")  # Section: multi-select by positions (ints).
    var tem:List[Int] = [4, 1, 0]                  # Column positions to extract (group, age, name).
    var by_pos = frame[tem]                         # Select columns by numeric positions, return DataFrame.
    print(by_pos.shape_str())                       # Show the shape of the positional selection.
    print(by_pos.__str__())                         # Print the selection.

    line("[]: boolean mask row filter")            # Section: bracket boolean mask filtering of rows.
    var mask: List[Bool]=([True, False, True, False, True, False])  # Keep r0, r2, r4 (3 rows).
    var even_rows = frame[mask]                    # Apply the mask: True rows are kept, False rows are dropped.
    print(even_rows.shape_str())                   # Show the shape after masking.
    print(even_rows.__str__())                     # Print the masked frame.

    line("[]: edge cases (missing/out-of-range/invalid mask)")  # Section: robustness checks.
    var missing_col = frame["does_not_exist"]      # Request a non-existing column → expect empty string-typed Column.
    show("missing col name", missing_col.get_name())  # Print its reported name (likely empty or placeholder).
    print(missing_col.__str__())                   # Print its representation (should indicate emptiness).

    var out_of_range_col = frame[Int(99)]          # Request out-of-range position → expect empty Column named "99".
    show("oor col name", out_of_range_col.get_name())  # Print the placeholder column name.
    print(out_of_range_col.__str__())              # Print its representation (should be empty).

    var btmp: List[Bool] = [True]                  # Deliberately wrong-length mask (1 vs 6 rows).
    var bad_mask = frame[btmp]                     # Apply invalid mask → expect empty frame with headers preserved.
    print(bad_mask.shape_str())                    # Show that the result is empty (0 rows).
    print(bad_mask.__str__())                      # Print the empty frame for clarity.

    # =============== loc: label-oriented rows =================
    line("loc: all rows, one column by name")      # Section: label-based row selection with column by name.
    var score_col = frame.loc(df.rows_all(), "score")  # Select 'score' over all rows using loc.
    show("col name", score_col.get_name())         # Confirm the selected column name.
    print(score_col.__str__())                     # Print the selected column.

    line("loc: all rows, some columns by name -> DataFrame")  # Section: loc with list of column labels.
    var city_group = frame.loc(df.rows_all(), ["city","group"])  # Select two columns over all rows.
    print(city_group.shape_str())                  # Show shape of the resulting frame.
    print(city_group.__str__())                    # Print the resulting frame.

    line("loc: all rows, columns by positions -> DataFrame")   # Section: loc with positional columns list.
    var age_score = frame.loc(df.rows_all(), [1, 3])           # Select columns by positions (age, score).
    print(age_score.shape_str())                   # Show shape.
    print(age_score.__str__())                     # Print the selection.

    line("loc: row positions + all columns")       # Section: loc with explicit row positions and all columns.
    var rpos: List[Int] = [0, 2, 5]                # Choose three row positions by label helper df.rows([...]).
    var rows_0_2_5 = frame.loc(df.rows(rpos), df.cols_all())  # Select those rows with all columns.
    print(rows_0_2_5.shape_str())                  # Show shape.
    print(rows_0_2_5.__str__())                    # Print the selection.

    line("loc: row positions + specific columns by name")  # Section: loc rows subset + named columns.
    var rpos2: List[Int] = [1, 4]                # Two row positions to include.
    var rows_1_4_name_score = frame.loc(df.rows(rpos2), ["name","score"])  # Subset of columns by name.
    print(rows_1_4_name_score.shape_str())         # Show shape.
    print(rows_1_4_name_score.__str__())           # Print.

    line("loc: row positions + single column")     # Section: loc returning a single column for selected rows.
    var rpos3: List[Int] = [0, 3]                  # Two row positions.
    var rows_0_3_city = frame.loc(df.rows(rpos3), "city")  # Return the 'city' Column over rows 0 and 3.
    show("col name", rows_0_3_city.get_name())     # Confirm column name.
    print(rows_0_3_city.__str__())                 # Print the column.

    line("loc: boolean mask rows + all columns")   # Section: loc with boolean mask for row selection.
    var mask_rows: List[Bool]=[False, True, False, True, False, True]  # Keep r1, r3, r5.
    var masked_df = frame.loc(df.rows(mask_rows), df.cols_all())  # Apply mask across all columns.
    print(masked_df.shape_str())                   # Show shape of the masked result.
    print(masked_df.__str__())                     # Print the masked DataFrame.

    line("loc: label slice on index (inclusive and open)")  # Section: label-based slicing on index.
    var incl = frame.loc(df.slice_labels("r1","r3", inclusive=True), df.cols_all())   # Inclusive slice r1..r3.
    print(incl.shape_str())                        # Show expected 3 rows.
    print(incl.__str__())                          # Print the slice.

    var open = frame.loc(df.slice_labels("r1","r3", inclusive=False), df.cols_all())  # Open slice r1..(r3-1) → r1,r2.
    print(open.shape_str())                        # Show expected 2 rows.
    print(open.__str__())                          # Print the slice.

    line("loc: label slice + single column")       # Section: label slice returning a single column.
    var slice_city = frame.loc(df.slice_labels("r2","r4", inclusive=True), "city")  # 'city' over r2..r4 inclusive.
    show("col name", slice_city.get_name())        # Confirm column name.
    print(slice_city.__str__())                    # Print the column.

    # =============== iloc: strictly positional =================
    line("iloc: all rows, single column by position")  # Section: iloc with one positional column.
    var col_pos_3 = frame.iloc(df.rows_all(), Int(3))  # Select the 4th column ("score") for all rows.
    show("col name", col_pos_3.get_name())             # Confirm selected column name.
    print(col_pos_3.__str__())                         # Print the column.

    line("iloc: all rows, multiple columns by positions -> DataFrame")  # Section: iloc with multiple columns.
    var cols_0_2_4 = frame.iloc(df.rows_all(), [0, 2, 4])  # Select name, city, group by positions.
    print(cols_0_2_4.shape_str())                  # Show shape of the result.
    print(cols_0_2_4.__str__())                    # Print the selection.

    line("iloc: row positions + all columns")      # Section: iloc with row positions and all columns.
    var rpos1: List[Int] = [5, 0, 2]               # Choose rows by positions (reordered).
    var rows_pos = frame.iloc(df.rows(rpos1), df.cols_all())  # Materialize those rows.
    print(rows_pos.shape_str())                    # Show shape.
    print(rows_pos.__str__())                      # Print the selection.

    line("iloc: row positions + single column by position")  # Section: iloc rows subset + single column.
    var rpos4: List[Int] = [1, 3, 5]              # Choose three rows.
    var rows_pos_score = frame.iloc(df.rows(rpos4), Int(3))  # Extract 'score' column (position 3).
    show("col name", rows_pos_score.get_name())    # Confirm column name.
    print(rows_pos_score.__str__())                # Print the column.

    line("iloc: row positional slice [start:stop) + all columns")  # Section: half-open positional slice.
    var mid_rows = frame.iloc(df.pslice(1, 4), df.cols_all())  # Rows 1,2,3 (stop-exclusive).
    print(mid_rows.shape_str())                    # Show shape = (3, 5).
    print(mid_rows.__str__())                      # Print the slice.

    line("iloc: row positional slice + specific columns")  # Section: slice rows + pick columns by positions.
    var mid_rows_cols = frame.iloc(df.pslice(1, 5), [0, 1])    # Rows 1..4 of ["name","age"].
    print(mid_rows_cols.shape_str())               # Show shape = (4, 2).
    print(mid_rows_cols.__str__())                 # Print the selection.

    line("iloc: row positional slice + single column")  # Section: slice rows + single column by pos.
    var last_three_age = frame.iloc(df.pslice(3, 6), Int(1))  # Rows 3..5 (inclusive of 5) for "age".
    show("col name", last_three_age.get_name())     # Confirm column name.
    print(last_three_age.__str__())                 # Print the column.

    # =============== combined workflows =================
    line("Workflow: score >= 85 mask + select subset of columns")  # Section: boolean mask + narrow view.
    var top_mask = ge_85_mask(frame)             # Build mask for high scorers.
    var top_df = frame.loc(df.rows(top_mask), ["name","city","score"])  # Keep a subset of columns.
    print(top_df.shape_str())                    # Show shape of filtered subset.
    print(top_df.__str__())                      # Print the filtered subset.

    line("Workflow: chain iloc then loc")        # Section: demonstrate chaining of iloc then loc.
    var block = frame.iloc(df.pslice(1, 5), df.cols_all())      # Positional slice of rows 1..4 (all columns).
    var reduced = block.loc(df.rows_all(), ["name","group"])    # Then pick two columns by name.
    print(reduced.shape_str())                   # Show shape of the reduced view.
    print(reduced.__str__())                     # Print the reduced view.

    line("Workflow: iloc edge handling")         # Section: out-of-range and empty selections.
    var empty_col = frame.iloc(df.rows_all(), Int(99))          # Request non-existent column by pos → empty.
    show("edge col name", empty_col.get_name())  # Print placeholder name (likely "99").
    print(empty_col.__str__())                   # Print its representation.

    var empty_slice = frame.iloc(df.pslice(9, 12), df.cols_all())  # Slice beyond length → empty DataFrame.
    print(empty_slice.shape_str())               # Show shape = (0, 5) expected.
    print(empty_slice.__str__())                 # Print the empty frame.

    # =============== sanity checks =================
    line("Sanity checks")                        # Section: lightweight consistency assertions.
    assert_eq_str(frame.shape_str(), "(6, 5)", "input shape")   # Verify initial shape of the demo frame.
    var nm = frame["name"]                        # Fetch 'name' Column for name check.
    assert_eq_str(nm.get_name(), "name", "[] name col")         # Ensure 'name' Column reports correct name.
    var sub = frame[["city","group"]]            # Build a small 2-column frame for column-count check.
    if sub.ncols() != 2: print("[ASSERT FAIL] sub.ncols != 2")  # Non-throwing assertion on number of columns.
    var sl = frame.iloc(df.pslice(2, 5), [0, 3]) # Slice rows 2..4 and columns [0,3] → shape should be (3,2).
    if sl.nrows() != 3 or sl.ncols() != 2: print("[ASSERT FAIL] iloc slice shape")  # Validate expected shape.
