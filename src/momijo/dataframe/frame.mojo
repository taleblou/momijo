# MIT License
# Copyright (c) 2025 Morteza Talebou and Mitra Daneshmand
# Project: momijo  |  Source: https://github.com/taleblou/momijo
# This file is part of the Momijo project. See the LICENSE file at the repository root.
# Momijo 
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Morteza Talebou and Mitra Daneshmand
# Website: https://taleblou.ir/
# Repository: https://github.com/taleblou/momijo
#
# Project: momijo.dataframe
# File: src/momijo/dataframe/frame.mojo

from momijo.core.error import module
from momijo.dataframe.bitmap import Bitmap
from momijo.dataframe.column import ColumnTag, F64, I64, as_f64_or_nan, value_str
from momijo.dataframe.helpers import m, t
from momijo.tensor.tensor import index
from momijo.utils.timer import start
from momijo.vision.schedule.schedule import ColumnTag
from pathlib import Path
from pathlib.path import Path

struct DataFrame(Copyable, Movable):
    var names: List[String]
    var cols: List[Column]

    # -------- Constructors --------
fn __init__(out self) -> None:
        self.names = List[String]()
        self.cols = List[Column]()
fn __init__(out self, names: List[String], cols: List[Column]) -> None:
        self.names = List[String]()
        self.cols = List[Column]()
        if len(names) != len(cols):
            return
        var h = 0
        if len(cols) > 0:
            h = cols[0].len()
        var i = 0
        while i < len(cols):
            if cols[i].len() != h:
                return
            i += 1
        self.names = names
        self.cols = cols

    # Explicit copy constructor
fn __copyinit__(out self, other: Self) -> None:
        self.names = List[String]()
        var i = 0
        var n = len(other.names)
        while i < n:
            self.names.append(String(other.names[i]))
            i += 1
        self.cols = List[Column]()
        i = 0
        var m = len(other.cols)
        while i < m:
            self.cols.append(other.cols[i])
            i += 1

    # -------- Introspection --------
fn width(self) -> Int:
        return len(self.cols)
fn height(self) -> Int:
        if len(self.cols) == 0:
            return 0
        return self.cols[0].len()
fn shape(self) -> (Int, Int):
        return (self.height(), self.width())
fn is_empty(self) -> Bool:
        return self.width() == 0 or self.height() == 0
fn column_names(self) -> List[String]:
        var out = List[String]()
        var i = 0
        while i < len(self.names):
            out.append(self.names[i])
            i += 1
        return out
fn get_column_index(self, name: String) -> Int:
        var i = 0
        var n = len(self.names)
        while i < n:
            if self.names[i] == name:
                return i
            i += 1
        return -1
fn get_column(self, name: String) -> Column:
        var idx = self.get_column_index(name)
        if idx < 0:
            return Column()
        return self.cols[idx]
fn get_column_at(self, idx: Int) -> Column:
        if idx < 0 or idx >= len(self.cols):
            return Column()
        return self.cols[idx]

    # -------- Column management --------
fn try_add_column(mut self, c: Column) -> Bool:
        if len(self.cols) == 0:
            self.names.append(c.name())
            self.cols.append(c)
            return True
        if c.len() != self.height():
            return False
        self.names.append(c.name())
        self.cols.append(c)
        return True
fn add_column(mut self, c: Column) -> Bool:
        return self.try_add_column(c)
fn with_column(self, c: Column) -> DataFrame:
        var nm = c.name()
        var idx = self.get_column_index(nm)
        if idx < 0:
            var out = DataFrame(self.column_names(), self.cols)
            _ = out.try_add_column(c)
            return out
        if c.len() != self.height():
            return self
        var new_names = self.column_names()
        var new_cols = List[Column]()
        var i = 0
        var w = self.width()
        while i < w:
            if i == idx:
                new_cols.append(c)
            else:
                new_cols.append(self.cols[i])
            i += 1
        return DataFrame(new_names, new_cols)
fn rename_column(mut self, old_name: String, new_name: String) -> Bool:
        var idx = self.get_column_index(old_name)
        if idx < 0:
            return False
        self.names[idx] = new_name
        return True
fn drop_columns(self, drop: List[String]) -> DataFrame:
        var keep_names = List[String]()
        var keep_cols = List[Column]()
        var i = 0
        var w = self.width()
        while i < w:
            var to_drop = False
            var j = 0
            while j < len(drop):
                if self.names[i] == drop[j]:
                    to_drop = True
                    break
                j += 1
            if not to_drop:
                keep_names.append(self.names[i])
                keep_cols.append(self.cols[i])
            i += 1
        return DataFrame(keep_names, keep_cols)

    # -------- Row selection / slicing --------
fn select_columns(self, keep: List[String]) -> DataFrame:
        var new_names = List[String]()
        var new_cols = List[Column]()
        var i = 0
        var m = len(keep)
        while i < m:
            var idx = self.get_column_index(keep[i])
            if idx >= 0:
                new_names.append(self.names[idx])
                new_cols.append(self.cols[idx])
            i += 1
        return DataFrame(new_names, new_cols)
fn filter_by_mask(self, mask: Bitmap) -> DataFrame:
        var new_cols = List[Column]()
        var i = 0
        var w = len(self.cols)
        while i < w:
            new_cols.append(self.cols[i].gather(mask))
            i += 1
        return DataFrame(self.column_names(), new_cols)
fn take(self, idxs: List[Int]) -> DataFrame:
        var new_cols = List[Column]()
        var i = 0
        var w = len(self.cols)
        while i < w:
            new_cols.append(self.cols[i].take(idxs))
            i += 1
        return DataFrame(self.column_names(), new_cols)
fn slice_rows(self, start: Int, end: Int) -> DataFrame:
        var n = self.height()
        var s = start
        if s < 0:
            s = 0
        var e = end
        if e > n:
            e = n
        if e <= s:
            return DataFrame(self.column_names(), List[Column]())
        var idxs = List[Int]()
        var i = s
        while i < e:
            idxs.append(i)
            i += 1
        return self.take(idxs)
fn head(self, k: Int) -> DataFrame:
        var m = k
        if m < 0:
            m = 0
        var n = self.height()
        var e = m
        if e > n:
            e = n
        return self.slice_rows(0, e)
fn tail(self, k: Int) -> DataFrame:
        var n = self.height()
        var m = k
        if m < 0:
            m = 0
        var s = 0
        if m < n:
            s = n - m
        return self.slice_rows(s, n)

    # -------- Display --------
fn to_string(self, max_rows: Int = 20) -> String:
        var s = String("DataFrame[")
        s = s + String(self.height()) + String(" x ") + String(self.width()) + String("]\n")
        var i = 0
        var w = self.width()
        while i < w:
            s = s + self.names[i]
            if i + 1 < w:
                s = s + String("\t")
            i += 1
        s = s + String("\n")
        var r = 0
        var h = self.height()
        var shown = 0
        while r < h and shown < max_rows:
            i = 0
            while i < w:
                s = s + self.cols[i].value()_str(r)
                if i + 1 < w:
                    s = s + String("\t")
                i += 1
            s = s + String("\n")
            r += 1
            shown += 1
        if h > shown:
            s = s + String("... (") + String(h - shown) + String(" more rows)\n")
        return s

    # -------- Sorting --------
fn sort_by(self, key: String, ascending: Bool = True) -> DataFrame:
        var keys = List[String]()
        keys.append(key)
        var asc = List[Bool]()
        asc.append(ascending)
        return self.sort_by_multi(keys, asc)
fn sort_by_multi(self, keys: List[String], ascending: List[Bool]) -> DataFrame:
        var n = self.height()
        var order = List[Int]()
        var i = 0
        while i < n:
            order.append(i)
            i += 1
fn less(i: Int, j: Int) -> Bool:
            var k = 0
            while k < len(keys):
                var col_idx = self.get_column_index(keys[k])
                var asc_k = True
                if k < len(ascending):
                    asc_k = ascending[k]
                var c = self.cols[col_idx]
                var res = 0
                if c.tag == ColumnTag.F64() or c.tag == ColumnTag.I64():
                    var a = c.as_f64_or_nan(i)
                    var b = c.as_f64_or_nan(j)
                    if a < b: res = -1
                    elif a > b: res = 1
                elif c.tag == ColumnTag.BOOL():
                    var a_i = 0
                    var b_i = 0
                    if c.b.get(i): a_i = 1
                    if c.b.get(j): b_i = 1
                    if a_i < b_i: res = -1
                    elif a_i > b_i: res = 1
                else:
                    assert(c is not None, String("c is None"))
                    var a_s = c.value()_str(i)
                    var b_s = c.value()_str(j)
                    if a_s < b_s: res = -1
                    elif a_s > b_s: res = 1
                if res != 0:
                    if asc_k: return res < 0
                    else:     return res > 0
                k += 1
            # stable tiebreaker by original index
            return i < j

        i = 0
        while i < n:
            var best = i
            var j = i + 1
            while j < n:
                if less(order[j], order[best]):
                    best = j
                j += 1
            if best != i:
                var tmp = order[i]
                order[i] = order[best]
                order[best] = tmp
            i += 1
        return self.take(order)