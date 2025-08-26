# MIT License
# Copyright (c) 2025 Morteza Talebou and Mitra Daneshmand
# Project: momijo  |  Source: https://github.com/taleblou/momijo
# This file is part of the Momijo project. See the LICENSE file at the repository root.
# Minimal DataFrame using columnar SeriesF64

from momijo.dataframe.series_f64 import SeriesF64
from collections import BitSet

struct DataFrame(Copyable, Movable):
    var columns: List[SeriesF64]

    fn __init__(out self, columns: List[SeriesF64]):
        # Column lengths must match
        if len(columns) > 0:
            var n = columns[0].len()
            for c in columns:
                assert(c.len() == n, "All columns must have the same length")
        self.columns = columns

    fn height(self) -> Int:
        return 0 if len(self.columns) == 0 else self.columns[0].len()

    fn width(self) -> Int:
        return len(self.columns)

    fn column_names(self) -> List[String]:
        var names = List[String]()
        for c in self.columns:
            names.append(c.name)
        return names

    fn select_mean(self, name: String) -> Float64:
        for c in self.columns:
            if c.name == name:
                return c.mean()
        assert(False, "Column not found: " + name)
        return 0.0

    fn filter_gt(self, col: String, threshold: Float64) -> DataFrame:
        # Build mask
        let n = self.height()
        var mask = BitSet()  # grow-only; we'll set bits we keep
        for i in range(0, n):
            var keep = False
            for c in self.columns:
                if c.name == col and c.validity.is_set(i):
                    keep = (c.values[i] > threshold)
            if keep:
                mask.set(i)
        # Apply to all cols
        var out_cols = List[SeriesF64]()
        for c in self.columns:
            out_cols.append(c.filter(mask))
        return DataFrame(out_cols)

fn get_column(self, name: String) -> SeriesF64:
    for c in self.columns:
        if c.name == name:
            return c
    assert(False, "Column not found: " + name)
    return SeriesF64("empty", [])

fn filter_by_mask(self, mask: Bitmap) -> DataFrame:
    let n = self.height()
    assert(mask.nbits == n, "Mask length mismatch")
    var out_cols = List[SeriesF64]()
    for c in self.columns:
        # build filtered values
        var vals = List[Float64]()
        for i in range(0, n):
            if mask.is_set(i) and c.validity.is_set(i):
                vals.append(c.values[i])
        out_cols.append(SeriesF64(c.name, vals))
    return DataFrame(out_cols)

fn select_columns(self, cols: List[String]) -> DataFrame:
    var out_cols = List[SeriesF64]()
    for name in cols:
        out_cols.append(self.get_column(name))
    return DataFrame(out_cols)

}
