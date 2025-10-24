# MIT License
# SPDX-License-Identifier: MIT
# Project: momijo.dataframe
# File: src/momijo/dataframe/multiindex.mojo
# Description: String-backed MultiIndex with factories, reordering, masking, and helpers.

from collections.list import List

struct MultiIndex(Copyable, Movable):
    var levels: List[List[String]]     # unique labels per level
    var codes:  List[List[Int]]        # per-row integer codes for each level
    var names:  List[String]           # level names (len == nlevels)

    # -------------------- core lifecycle --------------------
    fn __init__(out self):
        self.levels = List[List[String]]()
        self.codes  = List[List[Int]]()
        self.names  = List[String]()

    fn __copyinit__(out self, other: Self):
        self.levels = List[List[String]]()
        var i = 0
        while i < len(other.levels):
            self.levels.append(other.levels[i].copy())
            i += 1
        self.codes = List[List[Int]]()
        i = 0
        while i < len(other.codes):
            self.codes.append(other.codes[i].copy())
            i += 1
        self.names = other.names.copy()

    # -------------------- constants/helpers --------------------
    @staticmethod
    fn _sep() -> String:
        # field separator for composite key; choose a non-printable unlikely char
        return String("\x1f")

    @staticmethod
    fn _empty() -> Self:
        var mi = MultiIndex()
        return mi.copy()

    @staticmethod
    fn _all_lengths_equal(arrs: List[List[String]]) -> Bool:
        if len(arrs) == 0:
            return True
        var n = len(arrs[0])
        var i = 1
        while i < len(arrs):
            if len(arrs[i]) != n:
                return False
            i += 1
        return True

    @staticmethod
    fn _unique_in_order(xs: List[String]) -> List[String]:
        var out = List[String]()
        var i = 0
        while i < len(xs):
            var s = xs[i]
            var seen = False
            var j = 0
            while j < len(out):
                if out[j] == s:
                    seen = True
                    break
                j += 1
            if not seen:
                out.append(s)
            i += 1
        return out.copy()

    @staticmethod
    fn _index_of(xs: List[String], key: String) -> Int:
        var i = 0
        while i < len(xs):
            if xs[i] == key:
                return i
            i += 1
        return -1

    @staticmethod
    fn _join_comma(names: List[String]) -> String:
        var out = String("")
        var i = 0
        while i < len(names):
            if i > 0:
                out = out + String(",")
            out = out + names[i]
            i += 1
        return out

    # -------------------- factories --------------------

    # Build MI from row arrays per level: arrays[level][row] = label (String)
    @staticmethod
    fn from_arrays(arrays: List[List[String]], names: List[String]) -> Self:
        var L = len(arrays)
        if L == 0:
            var mi0 = MultiIndex()
            mi0.names = names.copy()
            return mi0.copy()
        if not MultiIndex._all_lengths_equal(arrays):
            return MultiIndex._empty()

        var nrows = len(arrays[0])
        var levels = List[List[String]]()
        var codes  = List[List[Int]]()

        var li = 0
        while li < L:
            var lvl = MultiIndex._unique_in_order(arrays[li])
            levels.append(lvl.copy())

            var cs = List[Int]()
            var r = 0
            while r < nrows:
                var label = arrays[li][r]
                var pos = MultiIndex._index_of(lvl, label)
                cs.append(pos)      # pos >= 0 by construction
                r += 1
            codes.append(cs.copy())
            li += 1

        var nm = List[String]()
        var i = 0
        while i < L and i < len(names):
            nm.append(names[i])
            i += 1
        while i < L:
            nm.append(String("level_" + String(i)))
            i += 1

        var mi = MultiIndex()
        mi.levels = levels.copy()
        mi.codes  = codes.copy()
        mi.names  = nm.copy()
        return mi.copy()

    # Build MI from list of row-wise tuples (each tuple = labels across levels) and names
    @staticmethod
    fn from_tuples(rows: List[List[String]], names: List[String]) -> Self:
        if len(rows) == 0:
            var mi0 = MultiIndex()
            mi0.names = names.copy()
            return mi0.copy()

        var L = len(rows[0])
        # split rows into per-level arrays
        var arrays = List[List[String]]()
        var li = 0
        while li < L:
            arrays.append(List[String]())
            li += 1

        var r = 0
        while r < len(rows):
            if len(rows[r]) != L:
                return MultiIndex._empty()
            var c = 0
            while c < L:
                arrays[c].append(rows[r][c])
                c += 1
            r += 1

        return MultiIndex.from_arrays(arrays.copy(), names.copy())

    # Cartesian product of levels inputs (each levels[i] = unique labels for level i)
    @staticmethod
    fn from_product(levels: List[List[String]], names: List[String]) -> Self:
        var L = len(levels)
        if L == 0:
            return MultiIndex._empty()

        # compute nrows = product of lengths; empty level → empty MI
        var nrows = 1
        var i = 0
        while i < L:
            var ln = len(levels[i])
            if ln == 0:
                return MultiIndex._empty()
            nrows = nrows * ln
            i += 1

        # build row-wise labels by nesting loops
        var arrays = List[List[String]]()
        i = 0
        while i < L:
            arrays.append(List[String]())
            i += 1

        var r = 0
        while r < nrows:
            var denom = nrows
            var li2 = 0
            while li2 < L:
                var ln2 = len(levels[li2])
                denom = denom // ln2
                var idx = (r // denom) % ln2
                arrays[li2].append(levels[li2][idx])
                li2 += 1
            r += 1

        return MultiIndex.from_arrays(arrays.copy(), names.copy())

    # -------------------- info & export --------------------
    fn nlevels(self) -> Int:
        return len(self.levels)

    fn nrows(self) -> Int:
        if len(self.codes) == 0:
            return 0
        return len(self.codes[0])

    fn levels_copy(self) -> List[List[String]]:
        var out = List[List[String]]()
        var i = 0
        while i < len(self.levels):
            out.append(self.levels[i].copy())
            i += 1
        return out.copy()

    fn codes_copy(self) -> List[List[Int]]:
        var out = List[List[Int]]()
        var i = 0
        while i < len(self.codes):
            out.append(self.codes[i].copy())
            i += 1
        return out.copy()

    fn names_copy(self) -> List[String]:
        return self.names.copy()

    fn get_level_values(self, level: Int) -> List[String]:
        var out = List[String]()
        if level < 0 or level >= len(self.codes):
            return out.copy()
        var n = len(self.codes[level])
        var i = 0
        while i < n:
            var code = self.codes[level][i]
            var s = String("")
            if code >= 0 and code < len(self.levels[level]):
                s = self.levels[level][code]
            out.append(s)
            i += 1
        return out.copy()

    fn values(self) -> List[String]:
        var nlevels = len(self.codes)
        var out = List[String]()
        if nlevels == 0:
            return out.copy()
        var nrows = len(self.codes[0])
        var r = 0
        while r < nrows:
            var key = String("")
            var lv = 0
            while lv < nlevels:
                var code = self.codes[lv][r]
                var label = String("")
                if code >= 0 and code < len(self.levels[lv]):
                    label = self.levels[lv][code]
                if lv > 0:
                    key = key + MultiIndex._sep()
                key = key + label
                lv += 1
            out.append(key)
            r += 1
        return out.copy()

    fn to_index_pair(self) -> (List[String], String):
        var vals = self.values()
        var name = MultiIndex._join_comma(self.names)
        return (vals.copy(), String(name))

    fn __str__(self) -> String:
        var s = String("MultiIndex(levels=[")
        var i = 0
        while i < len(self.levels):
            if i > 0: s += String(",")
            s += String("[")
            var j = 0
            while j < len(self.levels[i]):
                if j > 0: s += String(",")
                s += self.levels[i][j]
                j += 1
            s += String("]")
            i += 1
        s += String("], names=[")
        i = 0
        while i < len(self.names):
            if i > 0: s += String(",")
            s += self.names[i]
            i += 1
        s += String("])")
        return s

    # -------------------- renaming & reordering --------------------
    fn set_names(mut self, names: List[String]) -> Self:
        var L = self.nlevels()
        var nm = List[String]()
        var i = 0
        while i < L and i < len(names):
            nm.append(names[i])
            i += 1
        while i < L:
            nm.append(String("level_" + String(i)))
            i += 1
        self.names = nm.copy()
        return self.copy()

    fn rename_levels(mut self, mapper: List[(String, String)]) -> Self:
        # Replace names[i] when matches old-name in mapper; keep otherwise.
        var nm = self.names.copy()
        var i = 0
        while i < len(nm):
            var j = 0
            while j < len(mapper):
                if nm[i] == mapper[j][0]:
                    nm[i] = mapper[j][1]
                    break
                j += 1
            i += 1
        self.names = nm.copy()
        return self.copy()

    fn reorder_levels(mut self, order: List[Int]) -> Self:
        var L = self.nlevels()
        if len(order) != L:
            return self.copy()  # ignore bad request
        # bounds check
        var i = 0
        while i < L:
            if order[i] < 0 or order[i] >= L:
                return self.copy()
            i += 1

        var new_levels = List[List[String]]()
        var new_codes  = List[List[Int]]()
        var new_names  = List[String]()

        i = 0
        while i < L:
            var src = order[i]
            new_levels.append(self.levels[src].copy())
            new_codes.append(self.codes[src].copy())
            new_names.append(self.names[src])
            i += 1

        self.levels = new_levels.copy()
        self.codes  = new_codes.copy()
        self.names  = new_names.copy()
        return self.copy()

    fn swaplevel(mut self, a: Int, b: Int) -> Self:
        var L = self.nlevels()
        if a < 0 or b < 0 or a >= L or b >= L:
            return self.copy()
        if a == b:
            return self.copy()
        var order = List[Int]()
        var i = 0
        while i < L:
            order.append(i)
            i += 1
        var tmp = order[a]
        order[a] = order[b]
        order[b] = tmp
        return self.reorder_levels(order.copy())

    # -------------------- row selection --------------------
    fn take_rows(self, positions: List[Int]) -> Self:
        var L = self.nlevels()
        var n = self.nrows()
        var out = MultiIndex()
        out.levels = self.levels_copy()
        out.names  = self.names.copy()

        out.codes = List[List[Int]]()
        var lv = 0
        while lv < L:
            var cs = List[Int]()
            var i = 0
            while i < len(positions):
                var p = positions[i]
                var v = -1
                if p >= 0 and p < n:
                    v = self.codes[lv][p]
                cs.append(v)
                i += 1
            out.codes.append(cs.copy())
            lv += 1
        return out.copy()

    fn mask_rows(self, mask: List[Bool]) -> Self:
        var L = self.nlevels()
        var n = self.nrows()
        if len(mask) != n:
            return self.copy()
        var keep = List[Int]()
        var i = 0
        while i < n:
            if mask[i]:
                keep.append(i)
            i += 1
        return self.take_rows(keep.copy())

    # -------------------- code/label mapping --------------------
    fn level_index_of(self, level: Int, label: String) -> Int:
        if level < 0 or level >= self.nlevels():
            return -1
        return MultiIndex._index_of(self.levels[level], label)

    fn code_of(self, level: Int, row: Int) -> Int:
        if level < 0 or level >= len(self.codes):
            return -1
        if row < 0 or (len(self.codes[level]) == 0) or row >= len(self.codes[level]):
            return -1
        return self.codes[level][row]

    fn label_of(self, level: Int, row: Int) -> String:
        var c = self.code_of(level, row)
        if c < 0:
            return String("")
        if c >= len(self.levels[level]):
            return String("")
        return self.levels[level][c]

    # -------------------- checks --------------------
    fn is_lexsorted(self) -> Bool:
        # returns True if lexicographically non-decreasing by (code[0], code[1], ...)
        var n = self.nrows()
        var L = self.nlevels()
        if n <= 1:
            return True
        var i = 1
        while i < n:
            var lv = 0
            var ok = True
            var all_eq = True
            while lv < L:
                var a = self.codes[lv][i - 1]
                var b = self.codes[lv][i]
                if a < b:
                    all_eq = False
                    ok = True
                    break
                elif a > b:
                    ok = False
                    all_eq = False
                    break
                lv += 1
            if not ok:
                return False
            # if all equal so far, continue to next row
            i += 1
        return True

    fn is_unique(self) -> Bool:
        # true if no two rows have identical code tuples
        var n = self.nrows()
        var L = self.nlevels()
        var seen = List[String]()
        var i = 0
        while i < n:
            var key = String("")
            var lv = 0
            while lv < L:
                if lv > 0:
                    key = key + MultiIndex._sep()
                key = key + String(self.codes[lv][i])
                lv += 1
            # linear search (no Dict)
            var dup = False
            var j = 0
            while j < len(seen):
                if seen[j] == key:
                    dup = True
                    break
                j += 1
            if dup:
                return False
            seen.append(key)
            i += 1
        return True
