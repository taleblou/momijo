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
fn to_category(df: DataFrame,
               col_name: String,
               new_name: String = "category_code",
               ordered: Bool = False,
               categories: List[String] = List[String]()) -> DataFrame:

    var col = df.get_column_by_name(col_name)

    var cats = List[String]()
    var index = Dict[String, Int]()
    var codes = List[Int]()

    if len(categories) > 0:
        var k = 0
        while k < len(categories):
            var c = categories[k]
            if c not in index:
                index[c] = len(cats)
                cats.append(c)
            k += 1

        var n = col.len()
        var i = 0
        while i < n:
            var v = col.get_str(i)
            var code = index.get(v, -1)  # safe access without raise
            codes.append(code)
            i += 1
    else:
        var n = col.len()
        var i = 0
        while i < n:
            var v = col.get_str(i)
            var code = index.get(v, -1)
            if code == -1:
                code = len(cats)
                index[v] = code
                cats.append(v)
            codes.append(code)
            i += 1

    var codes_col = from_list_int(codes, name=new_name)
    var out = df.copy()
    out.add_column(codes_col)

    return out
