# MIT License
# SPDX-License-Identifier: MIT
# Project: momijo.dataframe
# File: examples/pd_io_csv_jsonl.mojo
# Description: IO demo for CSV and JSON Lines using the functional momijo.dataframe API.

from pathlib import Path
from os import makedirs
import momijo.dataframe as df
from collections.list import List

# ---------- Utilities ----------
fn print_section(title: String) -> None:
    var line = String("=") * 80
    print("\n" + line)
    print(title)
    print(line)

fn workdir_make() -> Path:
    # Simple temp-like folder next to CWD for demo artifacts
    var p = Path("./_momijo_demo_outputs")
    if not p.exists():
        try:
            makedirs(String(p))   # recursive create; ok if already exists
        except _:
            pass  # ignore errors quietly for teaching demo
    print("[INFO] Working directory: " + String(p))
    return p

# ---------- 2) IO: CSV / JSON (line-delimited) ----------
fn io_demo(frame: df.DataFrame, outdir: Path) -> df.DataFrame:
    print_section("2) IO: CSV / JSON")

    # CSV round-trip
    var csv_path = Path(String(outdir) + "/people.csv")
    df.to_csv(frame, String(csv_path))
    print("Wrote CSV -> " + String(csv_path))

    var read_back = df.read_csv(String(csv_path))
    print("Read CSV head():\n" + df.head(read_back, 3).to_string())

    # JSON Lines round-trip
    var jsonl_path = Path(String(outdir) + "/people.jsonl")
    df.to_json_lines(frame, String(jsonl_path))
    print("Wrote JSON Lines -> " + String(jsonl_path))

    var read_jsonl = df.read_json_lines(String(jsonl_path))
    print("Read JSONL head():\n" + df.head(read_jsonl, 3).to_string())

    return read_back

# ---------- main ----------
fn main() -> None:
    var outdir = workdir_make()

    # Build a small demo DataFrame (columns: name, age, city)
    var names  : List[String] = ["Alice","Bob","Cathy","Dan"]
    var ages   : List[Int]    = [25,31,29,40]
    var cities : List[String] = ["Helsinki","Turku","Tampere","Oulu"]

    var pairs = df.make_pairs()
    pairs = df.pairs_append(pairs, "name",  names)
    pairs = df.pairs_append(pairs, "age",   ages)
    pairs = df.pairs_append(pairs, "city",  cities)

    var frame = df.df_from_pairs(pairs)
    print_section("Input DataFrame")
    print(df.head(frame, 10).to_string())
    print("dtypes:\n" + df.df_dtypes(frame))

    var rd = io_demo(frame, outdir)

    print_section("Final check (CSV read-back, head)")
    print(df.head(rd, 10).to_string())
