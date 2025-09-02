# MIT License
# Project: momijo.dataframe
# File: momijo/dataframe/series_str.mojo

from momijo.dataframe.bitmap import Bitmap

struct SeriesStr(Copyable, Movable):
    var name: String
    var values: List[String]
    var valid: Bitmap

    # -------------------------
    # Constructors
    # -------------------------
    fn __init__(out self):
        self.name = String("")
        self.values = List[String]()
        self.valid = Bitmap()

    fn __init__(out self, name: String, values: List[String]):
        self.name = name
        self.values = values
        self.valid = Bitmap(len(values), True)

    fn __init__(out self, name: String, values: List[String], valid: Bitmap):
        self.name = name
        self.values = values
        if len(values) != len(valid):
            self.valid = Bitmap(len(values), True)
        else:
            self.valid = valid

    fn __copyinit__(out self, other: Self):
        self.name = String(other.name)
        self.values = List[String]()
        var i = 0
        var n = len(other.values)
        while i < n:
            self.values.append(String(other.values[i]))
            i += 1
        self.valid = Bitmap(n, True)
        i = 0
        while i < n:
            if not other.valid.get(i):
                _ = self.valid.set(i, False)
            i += 1

    # -------------------------
    # Basics
    # -------------------------
    fn len(self) -> Int:
        return len(self.values)

    fn is_valid(self, i: Int) -> Bool:
        if i < 0 or i >= len(self.values):
            return False
        return self.valid.get(i)

    fn get(self, i: Int) -> String:
        if i < 0 or i >= len(self.values):
            return String("")
        return self.values[i]

    fn rename(mut self, new_name: String):
        self.name = new_name

    fn count_valid(self) -> Int:
        return self.valid.count_true()

    fn null_count(self) -> Int:
        return self.len() - self.count_valid()

    # -------------------------
    # Builders / Mutators
    # -------------------------
    @staticmethod
    fn full(name: String, n: Int, value: String, is_valid: Bool = True) -> SeriesStr:
        var vals = List[String]()
        var i = 0
        while i < n:
            vals.append(String(value))
            i += 1
        var mask = Bitmap(n, is_valid)
        return SeriesStr(name, vals, mask)

    fn append(mut self, value: String, is_valid: Bool = True):
        self.values.append(value)
        if len(self.valid) == 0 and len(self.values) == 1:
            self.valid = Bitmap(1, is_valid)
        else:
            var old_len = len(self.valid)
            if old_len < len(self.values):
                var tmp = Bitmap(old_len + 1, True)
                var i = 0
                while i < old_len:
                    if not self.valid.get(i):
                        _ = tmp.set(i, False)
                    i += 1
                if not is_valid:
                    _ = tmp.set(old_len, False)
                self.valid = tmp
            else:
                _ = self.valid.set(len(self.values) - 1, is_valid)

    fn extend(mut self, more: SeriesStr):
        var i = 0
        var n = len(more.values)
        while i < n:
            self.append(more.values[i], more.valid.get(i))
            i += 1

    fn set(mut self, i: Int, value: String, is_valid: Bool = True):
        if i < 0 or i >= self.len():
            return
        self.values[i] = value
        _ = self.valid.set(i, is_valid)

    fn set_null(mut self, i: Int):
        if i < 0 or i >= self.len():
            return
        _ = self.valid.set(i, False)

    # -------------------------
    # Selection
    # -------------------------
    fn gather(self, mask: Bitmap) -> SeriesStr:
        var out = List[String]()
        var i = 0
        var n = len(self.values)
        while i < n:
            if mask.get(i) and self.valid.get(i):
                out.append(self.values[i])
            i += 1
        return SeriesStr(self.name, out)

    fn take(self, idxs: List[Int]) -> SeriesStr:
        var out = List[String]()
        var i = 0
        var n = len(idxs)
        while i < n:
            var j = idxs[i]
            if j >= 0 and j < len(self.values) and self.valid.get(j):
                out.append(self.values[j])
            i += 1
        return SeriesStr(self.name, out)

    # -------------------------
    # Slicing / Views
    # -------------------------
    fn slice(self, start: Int, end: Int) -> SeriesStr:
        var n = self.len()
        var s = start
        if s < 0:
            s = 0
        var e = end
        if e > n:
            e = n
        if e <= s:
            return SeriesStr(self.name, List[String]())
        var out_vals = List[String]()
        var out_valid = Bitmap(e - s, True)
        var i = s
        var k = 0
        while i < e:
            out_vals.append(self.values[i])
            if not self.valid.get(i):
                _ = out_valid.set(k, False)
            i += 1
            k += 1
        return SeriesStr(self.name, out_vals, out_valid)

    fn head(self, k: Int) -> SeriesStr:
        var m = k
        if m < 0:
            m = 0
        var n = self.len()
        var e = m
        if e > n:
            e = n
        return self.slice(0, e)

    fn tail(self, k: Int) -> SeriesStr:
        var n = self.len()
        var m = k
        if m < 0:
            m = 0
        var s = 0
        if m < n:
            s = n - m
        return self.slice(s, n)

    # -------------------------
    # Conversion
    # -------------------------
    fn to_list(self) -> List[String]:
        var out = List[String]()
        var i = 0
        var n = self.len()
        while i < n:
            out.append(self.values[i])
            i += 1
        return out

    # -------------------------
    # Null-handling helper (optional)
    # -------------------------
    fn fill_null(self, fill_value: String) -> SeriesStr:
        var n = self.len()
        var out_vals = List[String]()
        var out_valid = Bitmap(n, True)
        var i = 0
        while i < n:
            if self.valid.get(i):
                out_vals.append(self.values[i])
            else:
                out_vals.append(String(fill_value))
            i += 1
        return SeriesStr(self.name, out_vals, out_valid)
