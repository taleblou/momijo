# Project:      Momijo
# Module:       src.momijo.io.formats.csv
# File:         csv.mojo
# Path:         src/momijo/io/formats/csv.mojo
#
# Description:  Filesystem/IO helpers with Path-centric APIs and safe resource
#               management (binary/text modes and encoding clarity).
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
#   - Structs: CSVOptions
#   - Key functions: __init__, __copyinit__, __moveinit__, read_csv, write_csv
#   - Uses generic functions/types with explicit trait bounds.
#   - Performs file/Path IO; prefer context-managed patterns.


import os

struct CSVOptions:
    var delimiter: String
    var has_header: Bool
fn __init__(out self, delimiter: String = ",", has_header: Bool = True) -> None:
        self.delimiter = delimiter
        self.has_header = has_header
fn __copyinit__(out self, other: Self) -> None:
        self.delimiter = other.delimiter
        self.has_header = other.has_header
fn __moveinit__(out self, deinit other: Self) -> None:
        self.delimiter = other.delimiter
        self.has_header = other.has_header
# -----------------------------------------------------------------------------
# Read CSV
# -----------------------------------------------------------------------------
fn read_csv(path: String, options: CSVOptions = CSVOptions()) -> (List[String], List[List[String]]):
    if not os.path.exists(path):
        raise FileNotFoundError("CSV file not found: " + path)

    var f = open(path, "r")
    var lines = f.readlines()
    f.close()

    var header = List[String]()
    var rows = List[List[String]]()

    for (i, line) in enumerate(lines):
        var parts = line.strip().split(options.delimiter)
        if i == 0 and options.has_header:
            header = parts
        else:
            rows.append(parts)

    return (header, rows)

# -----------------------------------------------------------------------------
# Write CSV
# -----------------------------------------------------------------------------
fn write_csv(path: String, header: List[String], rows: List[List[String]], options: CSVOptions = CSVOptions()):
    var f = open(path, "w")

    if options.has_header and len(header) > 0:
        f.write(options.delimiter.join(header) + "\n")

    for row in rows:
        f.write(options.delimiter.join(row) + "\n")

    f.close()