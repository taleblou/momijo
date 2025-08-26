# MIT License
# Copyright (c) 2025 Morteza Talebou and Mitra Daneshmand
# Project: momijo  |  Source: https://github.com/taleblou/momijo
# This file is part of the Momijo project. See the LICENSE file at the repository root.


# Minimal CSV reader/writer for SeriesF64 columns (no quotes/escapes; demo only)

from momijo.dataframe.series_f64 import SeriesF64
from momijo.dataframe.frame import DataFrame

fn read_csv(path: String, headers: List[String]) -> DataFrame:
    var cols = List[List[Float64]]()
    for _ in headers:
        cols.append([])

    let fh = open(path, "r")
    for line in fh:
        let L = line.strip()
        if len(L) == 0: 
            continue
        let parts = L.split(",")
        assert(len(parts) == len(headers), "CSV columns mismatch")
        for i in range(0, len(parts)):
            cols[i].append(Float64(parts[i]))
    fh.close()

    var series = List[SeriesF64]()
    for i in range(0, len(headers)):
        series.append(SeriesF64(headers[i], cols[i]))
    return DataFrame(series)

fn write_csv(df: DataFrame, path: String):
    let fh = open(path, "w")
    # header
    fh.write(",".join(df.column_names()) + "\n")
    let n = df.height()
    for i in range(0, n):
        var row = List[String]()
        for c in df.columns:
            row.append(String(c.values[i]))
        fh.write(",".join(row) + "\n")
    fh.close()
