# MIT License
# Project: momijo.dataframe
# File: momijo/dataframe/series_f64.mojo

from momijo.dataframe.bitmap import Bitmap

struct SeriesF64(Copyable, Movable):
    var name: String
    var values: List[Float64]
    var valid: Bitmap

    # -------------------------
    # Constructors
    # -------------------------
    fn __init__(out self):
        self.name = String("")
        self.values = List[Float64]()
        self.valid = Bitmap()

    fn __init__(out self, name: String, values: List[Float64]):
        self.name = name
        self.values = values
        self.valid = Bitmap(len(values), True)

    fn __init__(out self, name: String, values: List[Float64], valid: Bitmap):
        self.name = name
        self.values = values
        if len(values) != len(valid):
            self.valid = Bitmap(len(values), True)
        else:
            self.valid = valid
 
    fn __copyinit__(out self, other: Self):
        self.name = String(other.name)
        self.values = List[Float64]()
        var i = 0
        var n = len(other.values)
        while i < n:
            self.values.append(other.values[i])
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

    fn get(self, i: Int) -> Float64:
        if i < 0 or i >= len(self.values):
            return 0.0
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
    fn full(name: String, n: Int, value: Float64, is_valid: Bool = True) -> SeriesF64:
        var vals = List[Float64]()
        var i = 0
        while i < n:
            vals.append(value)
            i += 1
        var mask = Bitmap(n, is_valid)
        return SeriesF64(name, vals, mask)

    fn append(mut self, value: Float64, is_valid: Bool = True):
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

    fn extend(mut self, more: SeriesF64):
        var i = 0
        var n = len(more.values)
        while i < n:
            self.append(more.values[i], more.valid.get(i))
            i += 1

    fn set(mut self, i: Int, value: Float64, is_valid: Bool = True):
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
    fn gather(self, mask: Bitmap) -> SeriesF64:
        var out = List[Float64]()
        var i = 0
        var n = len(self.values)
        while i < n:
            if mask.get(i) and self.valid.get(i):
                out.append(self.values[i])
            i += 1
        return SeriesF64(self.name, out)

    fn take(self, idxs: List[Int]) -> SeriesF64:
        var out = List[Float64]()
        var i = 0
        var n = len(idxs)
        while i < n:
            var j = idxs[i]
            if j >= 0 and j < len(self.values) and self.valid.get(j):
                out.append(self.values[j])
            i += 1
        return SeriesF64(self.name, out)

    fn slice(self, start: Int, end: Int) -> SeriesF64:
        var n = self.len()
        var s = start
        if s < 0:
            s = 0
        var e = end
        if e > n:
            e = n
        if e <= s:
            return SeriesF64(self.name, List[Float64]())
        var out_vals = List[Float64]()
        var out_valid = Bitmap(e - s, True)
        var i = s
        var k = 0
        while i < e:
            out_vals.append(self.values[i])
            if not self.valid.get(i):
                _ = out_valid.set(k, False)
            i += 1
            k += 1
        return SeriesF64(self.name, out_vals, out_valid)

    fn head(self, k: Int) -> SeriesF64:
        var m = k
        if m < 0:
            m = 0
        var n = self.len()
        var e = m
        if e > n:
            e = n
        return self.slice(0, e)

    fn tail(self, k: Int) -> SeriesF64:
        var n = self.len()
        var m = k
        if m < 0:
            m = 0
        var s = 0
        if m < n:
            s = n - m
        return self.slice(s, n)

    fn to_list(self) -> List[Float64]:
        var out = List[Float64]()
        var i = 0
        var n = self.len()
        while i < n:
            out.append(self.values[i])
            i += 1
        return out

    # -------------------------
    # Stats
    # -------------------------
    fn sum(self) -> Float64:
        var s: Float64 = 0.0
        var i = 0
        var n = len(self.values)
        while i < n:
            if self.valid.get(i):
                s = s + self.values[i]
            i += 1
        return s

    fn mean(self) -> Float64:
        var c = self.count_valid()
        if c == 0:
            return 0.0
        return self.sum() / Float64(c)

    fn min(self) -> Float64:
        var n = self.len()
        if n == 0:
            return 0.0
        var i = 0
        var found = False
        var mn: Float64 = 0.0
        while i < n and not found:
            if self.valid.get(i):
                mn = self.values[i]
                found = True
            i += 1
        if not found:
            return 0.0
        while i < n:
            if self.valid.get(i) and self.values[i] < mn:
                mn = self.values[i]
            i += 1
        return mn

    fn max(self) -> Float64:
        var n = self.len()
        if n == 0:
            return 0.0
        var i = 0
        var found = False
        var mx: Float64 = 0.0
        while i < n and not found:
            if self.valid.get(i):
                mx = self.values[i]
                found = True
            i += 1
        if not found:
            return 0.0
        while i < n:
            if self.valid.get(i) and self.values[i] > mx:
                mx = self.values[i]
            i += 1
        return mx

    # -------------------------
    # Element-wise ops (NULL-aware)
    # -------------------------
    fn add_scalar(self, x: Float64) -> SeriesF64:
        var n = self.len()
        var out_vals = List[Float64]()
        var out_valid = Bitmap(n, True)
        var i = 0
        while i < n:
            out_vals.append(self.values[i] + x)
            if not self.valid.get(i):
                _ = out_valid.set(i, False)
            i += 1
        return SeriesF64(self.name, out_vals, out_valid)

    fn sub_scalar(self, x: Float64) -> SeriesF64:
        var n = self.len()
        var out_vals = List[Float64]()
        var out_valid = Bitmap(n, True)
        var i = 0
        while i < n:
            out_vals.append(self.values[i] - x)
            if not self.valid.get(i):
                _ = out_valid.set(i, False)
            i += 1
        return SeriesF64(self.name, out_vals, out_valid)

    fn mul_scalar(self, x: Float64) -> SeriesF64:
        var n = self.len()
        var out_vals = List[Float64]()
        var out_valid = Bitmap(n, True)
        var i = 0
        while i < n:
            out_vals.append(self.values[i] * x)
            if not self.valid.get(i):
                _ = out_valid.set(i, False)
            i += 1
        return SeriesF64(self.name, out_vals, out_valid)

    fn div_scalar(self, x: Float64) -> SeriesF64:
        var n = self.len()
        var out_vals = List[Float64]()
        var out_valid = Bitmap(n, True)
        var i = 0
        while i < n:
            out_vals.append(self.values[i] / x)
            if not self.valid.get(i):
                _ = out_valid.set(i, False)
            i += 1
        return SeriesF64(self.name, out_vals, out_valid)

    fn add(self, other: SeriesF64) -> SeriesF64:
        var n = self.len()
        var m = other.len()
        var L = n
        if m < L:
            L = m
        var out_vals = List[Float64]()
        var out_valid = Bitmap(L, True)
        var i = 0
        while i < L:
            var ok = self.valid.get(i) and other.valid.get(i)
            if ok:
                out_vals.append(self.values[i] + other.values[i])
            else:
                out_vals.append(0.0)
                _ = out_valid.set(i, False)
            i += 1
        return SeriesF64(self.name, out_vals, out_valid)

    fn sub(self, other: SeriesF64) -> SeriesF64:
        var n = self.len()
        var m = other.len()
        var L = n
        if m < L:
            L = m
        var out_vals = List[Float64]()
        var out_valid = Bitmap(L, True)
        var i = 0
        while i < L:
            var ok = self.valid.get(i) and other.valid.get(i)
            if ok:
                out_vals.append(self.values[i] - other.values[i])
            else:
                out_vals.append(0.0)
                _ = out_valid.set(i, False)
            i += 1
        return SeriesF64(self.name, out_vals, out_valid)

    fn mul(self, other: SeriesF64) -> SeriesF64:
        var n = self.len()
        var m = other.len()
        var L = n
        if m < L:
            L = m
        var out_vals = List[Float64]()
        var out_valid = Bitmap(L, True)
        var i = 0
        while i < L:
            var ok = self.valid.get(i) and other.valid.get(i)
            if ok:
                out_vals.append(self.values[i] * other.values[i])
            else:
                out_vals.append(0.0)
                _ = out_valid.set(i, False)
            i += 1
        return SeriesF64(self.name, out_vals, out_
