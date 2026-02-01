# Project:      Momijo
# Module:       examples.io_csv_json
# File:         io_csv_json.mojo
# Path:         src/momijo/examples/io_csv_json.mojo
#
# Description:  Demo of CSV and JSON Lines round-trips with momijo.dataframe I/O.
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
#   - Demonstrates: df.to_csv/read_csv and df.to_json_lines/read_json_lines with a small DataFrame.

from pathlib import Path                           # Path abstraction for robust filesystem operations.
from os import makedirs                            # Recursive directory creation utility.
import momijo.dataframe as df                      # Import dataframe API under alias 'df'.
from collections.list import List                  # Typed dynamic array for columns and values.

# ---------- Utilities ----------

fn print_section(title: String) -> None:           # Print a visible banner around section titles.
    var line = String("=") * 80                    # Build a horizontal rule of '=' characters.
    print("\n" + line)                             # Blank line and top rule for spacing.
    print(title)                                   # Print the given section title.
    print(line)                                    # Bottom rule to frame the section.

fn workdir_make() -> Path:                         # Ensure and return a working directory path.
    # Simple temp-like folder next to CWD for demo artifacts
    var p = Path("./_momijo_demo_outputs")         # Choose a deterministic outputs directory.
    if not p.exists():                             # If the directory is missing, attempt to create it.
        try:
            makedirs(String(p))                    # Recursively create directories; OK if already exists.
        except _:
            pass                                   # Ignore errors to keep the teaching demo resilient.
    print("[INFO] Working directory: " + String(p))# Log the resolved directory path for the user.
    return p                                       # Return the Path for downstream use.

# ---------- 2) IO: CSV / JSON (line-delimited) ----------

fn io_demo(frame: df.DataFrame, outdir: Path) raises -> df.DataFrame:  # Round-trip CSV and JSON Lines; return read-back.
    print_section("2) IO: CSV / JSON")             # Section header for I/O demonstration.

    # CSV round-trip
    var csv_path = Path(String(outdir) + "/people.csv")  # Compose a CSV file path under the output directory.
    df.to_csv(frame, String(csv_path))              # Write the DataFrame to CSV.
    print("Wrote CSV -> " + String(csv_path))       # Inform the user about the CSV write location.

    var read_back = df.read_csv(String(csv_path))   # Read the CSV back into a DataFrame.
    print("Read CSV head():\n" + df.head(read_back, 3).to_string())  # Show a small preview of the CSV read-back.

    # JSON Lines round-trip
    var jsonl_path = Path(String(outdir) + "/people.jsonl")  # Compose a JSONL file path under the output directory.
    df.to_json_lines(frame, String(jsonl_path))     # Write the DataFrame as JSON Lines.
    print("Wrote JSON Lines -> " + String(jsonl_path))  # Inform the user about the JSONL write location.

    var read_jsonl = df.read_json_lines(String(jsonl_path))  # Read the JSON Lines back into a DataFrame.
    print("Read JSONL head():\n" + df.head(read_jsonl, 3).to_string())  # Show a small preview of the JSONL read-back.

    return read_back                                 # Return the CSV read-back DataFrame for a final check.

# ---------- main ----------

fn main() raises -> None:                            # Program entry point (declared as raising for I/O safety).
    var outdir = workdir_make()                      # Ensure outputs directory exists and obtain its Path.

    # Build a small demo DataFrame (columns: name, age, city)
    var names  : List[String] = ["Alice","Bob","Cathy","Dan"]   # Sample string column: person names.
    var ages   : List[Int]    = [25,31,29,40]                   # Sample integer column: ages.
    var cities : List[String] = ["Helsinki","Turku","Tampere","Oulu"]  # Sample string column: cities.

    var pairs = df.make_pairs()                       # Start an empty (name, values) container for building a frame.
    pairs = df.pairs_append(pairs, "name",  names)    # Append the 'name' column.
    pairs = df.pairs_append(pairs, "age",   ages)     # Append the 'age' column.
    pairs = df.pairs_append(pairs, "city",  cities)   # Append the 'city' column.

    var frame = df.df_from_pairs(pairs)               # Materialize a DataFrame from the accumulated pairs.
    print_section("Input DataFrame")                  # Section header for the input preview.
    print(df.head(frame, 10).to_string())             # Show up to 10 rows of the input data.
    print("dtypes:\n" + df.df_dtypes(frame))          # Print column data types for verification.

    var rd = io_demo(frame, outdir)                   # Perform I/O round-trips and get the CSV read-back frame.

    print_section("Final check (CSV read-back, head)")# Header for the final verification output.
    print(df.head(rd, 10).to_string())                # Show up to 10 rows from the read-back DataFrame.
