# Project:      Momijo
# Module:       dataframe.categorical
# File:         categorical.mojo
# Path:         dataframe/categorical.mojo
#
# Description:  dataframe.categorical — Categorical module for Momijo DataFrame.
#               Implements core data structures, algorithms, and convenience APIs for production use.
#               Designed as a stable, composable building block within the Momijo public API.
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
#   - Structs: Categorical
#   - Key functions: __init__, __copyinit__, __len__, __str__, get, categories, codes_list, unique, values, value_counts, astype, reorder, remove_unused, to_category

from momijo.dataframe.column import Column,from_list_int
from momijo.dataframe.series_bool import *
from momijo.dataframe.series_str import *
from momijo.dataframe.series_f64 import *
from momijo.dataframe.series_i64 import *


struct Categorical(Copyable, Movable):
    var cats: List[String]
    var codes: List[Int]

    fn __init__(out self):
        self.cats = List[String]()
        self.codes = List[Int]()

    fn __init__(out self, values: List[String]):
        self.cats = List[String]()
        self.codes = List[Int]()
        var seen = Dict[String, Int]()
        var i = 0
        while i < len(values):
            var v = values[i]
            if v not in seen:
                seen[v] = len(self.cats)
                self.cats.append(v)
            self.codes.append(seen[v])
            i += 1

    fn __copyinit__(out self, other: Self):
        var cc = List[String]()
        var i = 0
        while i < len(other.cats):
            cc.append(other.cats[i])
            i += 1
        var cd = List[Int]()
        var j = 0
        while j < len(other.codes):
            cd.append(other.codes[j])
            j += 1
        self.cats = cc
        self.codes = cd

    fn __len__(self) -> Int:
        return len(self.codes)

    fn __str__(self) -> String:
        var s = String("Categorical(")
        s += String(len(self.codes)) + String(" items, ")
        s += String(len(self.cats)) + String(" categories)")
        return s

# Accessors
    fn get(self, i: Int) -> String:
        if i < 0 or i >= len(self.codes):
            return String("")
        var code = self.codes[i]
        if code < 0 or code >= len(self.cats):
            return String("")
        return self.cats[code]

    fn categories(self) -> List[String]:
        return self.cats

    fn codes_list(self) -> List[Int]:
        return self.codes

    fn unique(self) -> List[String]:
        return self.cats

    fn values(self) -> List[String]:
# Decode codes to values
        var out = List[String]()
        var i = 0
        while i < len(self.codes):
            var c = self.codes[i]
            if c >= 0 and c < len(self.cats):
                out.append(self.cats[c])
            else:
                out.append(String(""))
            i += 1
        return out

    fn value_counts(self) -> Dict[String, Int]:
        var counts = Dict[String, Int]()
        var i = 0
        while i < len(self.codes):
            var code = self.codes[i]
            var cat = self.cats[code]
            if cat in counts:
                counts[cat] = counts[cat] + 1
            else:
                counts[cat] = 1
            i += 1
        return counts

# Casting
    fn astype(self, target: String) -> List[String]:
# Currently supports: "string" -> decode to List[String]
# Other targets can be added later (e.g., int codes).
        if target == String("string") or target == String("str") or target == String("object"):
            return self.values()
# Default: return decoded values to be safe for downstream usage
        return self.values()

# Category operations
    fn reorder(self, new_cats: List[String]) -> Categorical:
# Produce a new Categorical with categories reordered to new_cats.
# Codes are remapped; values not in new_cats are mapped to -1 then to "" on decode.
        var rev = Dict[String, Int]()
        var i = 0
        while i < len(new_cats):
            rev[new_cats[i]] = i
            i += 1

        var new_codes = List[Int]()
        var j = 0
        while j < len(self.codes):
            var old_code = self.codes[j]
            var val = String("")
            if old_code >= 0 and old_code < len(self.cats):
                val = self.cats[old_code]
            if val in rev:
                new_codes.append(rev[val])
            else:
                new_codes.append(-1)  # unknown after reorder
            j += 1

        var out = Categorical()
        out.cats = new_cats
        out.codes = new_codes
        return out

    fn remove_unused(self) -> Categorical:
# Remove categories that are not referenced by any code.
        var used = Dict[Int, Int]()  # old_code -> new_code
        var new_cats = List[String]()

# Mark used categories
        var i = 0
        while i < len(self.codes):
            var c = self.codes[i]
            if c >= 0 and c < len(self.cats):
                if c not in used:
                    used[c] = len(new_cats)
                    new_cats.append(self.cats[c])
            i += 1

# Remap codes
        var new_codes = List[Int]()
        var j = 0
        while j < len(self.codes):
            var oc = self.codes[j]
            if oc in used:
                new_codes.append(used[oc])
            else:
                new_codes.append(-1)
            j += 1

        var out = Categorical()
        out.cats = new_cats
        out.codes = new_codes
        return out
 

# Build categorical (cats, codes) purely as tuple without importing Categorical
# fn to_category(col: Column) -> (List[String], List[Int]):
#     var cats = List[String]()
#     var index = Dict[String, Int]()
#     var codes = List[Int]()
#     var n = col.len()
#     var i = 0
#     while i < n:
#         var v = col.get_str(i)
#         if v in index:
#             codes.append(index[v])
#         else:
#             var cid = len(cats)
#             index[v] = cid
#             cats.append(v)
#             codes.append(cid)
#         i += 1
#     return (cats, codes)

# Column-level overload with options
# Helper: works when you already have a Column object
# Convert a string column to categorical integer codes and append as a new column.
# - If `categories` is non-empty: use its order to map; unseen values → -1.
# - Else: build categories on the fly by first occurrence order.
# Build/replace an integer code column from a string column with ordered categories.
fn to_category(df: DataFrame,
               col_name: String,
               new_name: String = String("category_code"),
               ordered: Bool = False,
               categories: List[String] = List[String]()) -> DataFrame:
    var out = df.copy()

    # resolve source column index
    var j = -1
    var c = 0
    while c < out.ncols():
        if out.col_names[c] == col_name:
            j = c
            break
        c += 1
    if j < 0:
        return out  # column not found → no-op

    # Build category list `cats`
    var cats = List[String]()
    if len(categories) > 0:
        # Use provided order, deduplicate preserving order
        var k = 0
        while k < len(categories):
            var v = categories[k]
            # linear "contains" (no Dict.contains_key dependency)
            var seen = False
            var q = 0
            while q < len(cats):
                if cats[q] == v:
                    seen = True
                    break
                q += 1
            if not seen:
                cats.append(v)
            k += 1
    else:
        # Discover from data, in first-seen order
        var n = out.cols[j].len()
        var r = 0
        while r < n:
            var v = out.cols[j].get_string(r)
            var seen = False
            var q = 0
            while q < len(cats):
                if cats[q] == v:
                    seen = True
                    break
                q += 1
            if not seen:
                cats.append(v)
            r += 1

    # Build codes
    var nrows = out.nrows()
    var codes = List[Int]()
    codes.reserve(nrows)

    var r2 = 0
    while r2 < nrows:
        var v = out.cols[j].get_string(r2)
        # index_of
        var idx = -1
        var q2 = 0
        while q2 < len(cats):
            if cats[q2] == v:
                idx = q2
                break
            q2 += 1
        codes.append(idx)  # -1 if unseen
        r2 += 1

    # Assemble SeriesI64 (or Int) and set into frame (replace or append)
    var s = SeriesI64()
    s.set_name(new_name)
    # If SeriesI64 expects Int64 values, ensure you push Int64; otherwise Int is fine:
    s.data = codes.copy()   # deep copy semantics for List[Int]
    # Optionally set validity to all true if you track it.
    # s.valid = Bitmap.full(nrows, True)

    # If a column named new_name exists, replace it; else append.
    var dst = -1
    var t = 0
    while t < out.ncols():
        if out.col_names[t] == new_name:
            dst = t
            break
        t += 1

    if dst >= 0:
        # replace in place via setter (avoid aliasing)
        out.cols[dst].set_i64_series(s)
    else:
        var col = Column()
        col.set_i64_series(s)
        out.cols.append(col.copy())
        out.col_names.append(new_name)
    return out
