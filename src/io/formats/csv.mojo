# ============================================================================
# MIT License
# Copyright (c) 2025 Morteza Talebou and Mitra Daneshmand
# Project: momijo  |  Source: https://github.com/taleblou/momijo
# This file is part of the Momijo project. See the LICENSE file at the repository root.
# SPDX-License-Identifier: MIT
# Website: https://taleblou.ir/
# Repository: https://github.com/taleblou/momijo
#
# Project: momijo.io.formats
# File: src/momijo/io/formats/csv.mojo
# ============================================================================

import os

# -----------------------------------------------------------------------------
# CSVOptions
# -----------------------------------------------------------------------------
struct CSVOptions:
    var delimiter: String
    var has_header: Bool

    fn __init__(out self, delimiter: String = ",", has_header: Bool = True):
        self.delimiter = delimiter
        self.has_header = has_header


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


 
