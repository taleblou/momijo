# Project:      Momijo
# Module:       src.momijo.tensor.tensorbool
# File:         tensorbool.mojo
# Path:         src/momijo/tensor/tensorbool.mojo
#
# Description:  Bool-only tensor (Bool storage). Ambiguity-free constructors.
# License:      MIT

from momijo.tensor.utils import shape_product
from momijo.tensor.utils import compute_strides

struct BoolTensor(ExplicitlyCopyable, Movable):
    var _shape: List[Int]
    var _strides: List[Int]
    var _data: List[Bool]

    fn __copyinit__(out self, other: Self):
        self._shape = other._shape
        self._strides = other._strides
        self._data = List[Bool]()
        var i = 0
        while i < len(other._data):
            self._data.append(other._data[i])
            i += 1

    fn copy(self) -> Self:
        var out: BoolTensor = self
        return out

    # 1D
    fn __init__(out self, data: List[Bool]):
        self._shape = [len(data)]
        self._strides = compute_strides(self._shape)
        self._data = List[Bool]()
        var i = 0
        while i < len(data):
            self._data.append(data[i])
            i += 1

    # 2D
    fn __init__(out self, data: List[List[Bool]]):
        var rows = len(data)
        var cols = 0
        if rows > 0: cols = len(data[0])
        self._shape = [rows, cols]
        self._strides = compute_strides(self._shape)
        self._data = List[Bool]()
        var r = 0
        while r < rows:
            var c = 0
            while c < cols:
                self._data.append(data[r][c])
                c += 1
            r += 1

    # shape + fill
    fn __init__(out self, shape: List[Int], fill: Bool):
        self._shape = shape
        self._strides = compute_strides(shape)
        self._data = List[Bool]()
        var n = shape_product(shape)
        var i = 0
        while i < n:
            self._data.append(fill)
            i += 1

    fn shape(self) -> List[Int]: return self._shape
    fn __str__(self) -> String:  return "BoolTensor(shape=" + self._shape.__str__() + ")"


    # ----- Helpers -----
    fn _ensure_1d_len(self) -> Int:
        if len(self._shape) == 1:
            return self._shape[0]
        var n = 1
        var i = 0
        while i < len(self._shape):
            n = n * self._shape[i]
            i += 1
        return n

    fn _copy_flat(self) -> List[Bool]:
        var n = len(self._data)
        var out = List[Bool]()
        var i = 0
        while i < n:
            out.append(self._data[i])
            i += 1
        return out


    # ===== Algorithms (boolean) =====
    fn sort(self) -> BoolTensor:
        # False < True
        var n = len(self._data)
        var zeros = 0
        var i = 0
        while i < n:
            if not self._data[i]:
                zeros += 1
            i += 1
        var out = BoolTensor([n], False)
        out._data = List[Bool]()
        var k = 0
        while k < zeros:
            out._data.append(False); k += 1
        while k < n:
            out._data.append(True); k += 1
        return out

    fn unique(self, return_counts: Bool = True) -> (BoolTensor, IntTensor):
        var has_false = False
        var has_true = False
        var i = 0
        while i < len(self._data):
            if self._data[i]:
                has_true = True
            else:
                has_false = True
            i += 1
        var vals = List[Bool]()
        var cnts = List[Int]()
        if has_false:
            vals.append(False)
            var c0 = 0; i = 0
            while i < len(self._data):
                if not self._data[i]: c0 += 1
                i += 1
            cnts.append(c0)
        if has_true:
            vals.append(True)
            var c1 = 0; i = 0
            while i < len(self._data):
                if self._data[i]: c1 += 1
                i += 1
            cnts.append(c1)
        var ut = BoolTensor([len(vals)], False); ut._data = vals
        var ct = IntTensor([len(cnts)], 0);      ct._data = cnts
        return (ut, ct) if return_counts else (ut, IntTensor([0], 0))

    fn bincount(self) -> IntTensor:
        var c0 = 0; var c1 = 0
        var i = 0
        while i < len(self._data):
            if self._data[i]: c1 += 1
            else: c0 += 1
            i += 1
        var out = IntTensor([2], 0)
        out._data[0] = c0
        out._data[1] = c1
        return out

    fn histogram(self, bins: List[Int]) -> (IntTensor, IntTensor):
        # interpret False=0, True=1
        var ints = IntTensor([len(self._data)], 0)
        var i = 0
        while i < len(self._data):
            ints._data.append( (1 if self._data[i] else 0) )
            i += 1
        return ints.histogram(bins)

    fn digitize(self, bins: List[Int]) -> IntTensor:
        var ints = IntTensor([len(self._data)], 0)
        var i = 0
        while i < len(self._data):
            ints._data.append( (1 if self._data[i] else 0) )
            i += 1
        return ints.digitize(bins)

    fn set_union(self, other: BoolTensor) -> BoolTensor:
        var a_true = False; var a_false = False
        var b_true = False; var b_false = False
        var i = 0
        while i < len(self._data):
            if self._data[i]: a_true = True
            else: a_false = True
            i += 1
        i = 0
        while i < len(other._data):
            if other._data[i]: b_true = True
            else: b_false = True
            i += 1
        var vals = List[Bool]()
        if a_false or b_false: vals.append(False)
        if a_true or b_true:   vals.append(True)
        var out = BoolTensor([len(vals)], False); out._data = vals; return out

    fn set_intersection(self, other: BoolTensor) -> BoolTensor:
        var has_false = False
        var has_true = False
        var i = 0
        while i < len(self._data):
            has_false = has_false or (not self._data[i])
            has_true  = has_true  or self._data[i]
            i += 1
        var has_false_b = False
        var has_true_b = False
        i = 0
        while i < len(other._data):
            has_false_b = has_false_b or (not other._data[i])
            has_true_b  = has_true_b  or other._data[i]
            i += 1
        var vals = List[Bool]()
        if has_false and has_false_b: vals.append(False)
        if has_true and has_true_b:   vals.append(True)
        var out = BoolTensor([len(vals)], False); out._data = vals; return out

    fn set_difference(self, other: BoolTensor) -> BoolTensor:
        # Elements present in self but not in other (set semantics)
        var vals = List[Bool]()
        var has_self_false = False; var has_self_true = False
        var i = 0
        while i < len(self._data):
            if self._data[i]: has_self_true = True
            else: has_self_false = True
            i += 1
        var has_other_false = False; var has_other_true = False
        i = 0
        while i < len(other._data):
            if other._data[i]: has_other_true = True
            else: has_other_false = True
            i += 1
        if has_self_false and (not has_other_false): vals.append(False)
        if has_self_true  and (not has_other_true):  vals.append(True)
        var out = BoolTensor([len(vals)], False); out._data = vals; return out

    fn set_xor(self, other: BoolTensor) -> BoolTensor:
        var vals = List[Bool]()
        var has_self_false = False; var has_self_true = False
        var i = 0
        while i < len(self._data):
            if self._data[i]: has_self_true = True
            else: has_self_false = True
            i += 1
        var has_other_false = False; var has_other_true = False
        i = 0
        while i < len(other._data):
            if other._data[i]: has_other_true = True
            else: has_other_false = True
            i += 1
        # XOR set: elements present in exactly one set
        if (has_self_false != has_other_false): vals.append(False)
        if (has_self_true  != has_other_true):  vals.append(True)
        var out = BoolTensor([len(vals)], False); out._data = vals; return out


    # ----- reshape -----
    fn reshape(self, new_shape: List[Int]) -> Self:
        var n_old = len(self._data)
        var n_new = 1
        var i = 0
        while i < len(new_shape):
            n_new = n_new * new_shape[i]
            i += 1
        if not (n_new == n_old):
            pass
        var out: Self = self
        out._shape = new_shape
        out._strides = compute_strides(new_shape)
        return out


    # ----- creation: arange -----
    @staticmethod
    fn arange(start: Int, stop: Int, step: Int = 1) -> BoolTensor:
        if step == 0:
            step = 1
        var out = BoolTensor([False])
        out._data = List[Bool]()
        var i = start
        if step > 0:
            while i < stop:
                out._data.append(Bool(i != 0))
                i += step
        else:
            while i > stop:
                out._data.append(Bool(i != 0))
                i += step
        out._shape = [len(out._data)]
        out._strides = compute_strides(out._shape)
        return out


    # ----- moveaxis/swapaxes/roll -----
    fn moveaxis(self, source: Int, destination: Int) -> BoolTensor:
        var rank = len(self._shape)
        if not (source >= 0 and source < rank):
            pass
        if not (destination >= 0 and destination < rank):
            pass
        var perm = List[Int]()
        var i = 0
        while i < rank:
            if i != source:
                perm.append(i)
            i += 1
        perm.insert(destination, source)
        var new_shape = List[Int]()
        var k = 0
        while k < rank:
            new_shape.append(self._shape[perm[k]])
            k += 1
        var out = BoolTensor(new_shape, False)
        out._data = List[Bool]()
        out._data.resize(len(self._data))
        var n = self._total_size()
        var flat = 0
        while flat < n:
            var idxs = self._unravel_index(flat, self._shape)
            var new_idxs = List[Int]()
            var t = 0
            while t < rank:
                new_idxs.append(idxs[perm[t]])
                t += 1
            var new_flat = self._ravel_index(new_idxs, new_shape)
            out._data[new_flat] = self._data[flat]
            flat += 1
        return out

    fn swapaxes(self, a: Int, b: Int) -> BoolTensor:
        return self.moveaxis(a, b).moveaxis(b, a)

    fn roll(self, shift: Int, axis: Int = 0) -> BoolTensor:
        var rank = len(self._shape)
        if not (axis >= 0 and axis < rank):
            pass
        var dim = self._shape[axis]
        if dim == 0: return self
        var k = ((shift % dim) + dim) % dim
        if k == 0: return self
        var out = BoolTensor(self._shape, False)
        out._data = List[Bool]()
        out._data.resize(len(self._data))
        var n = self._total_size()
        var flat = 0
        while flat < n:
            var idxs = self._unravel_index(flat, self._shape)
            var dst = idxs
            dst[axis] = (idxs[axis] + k) % dim
            var new_flat = self._ravel_index(dst, self._shape)
            out._data[new_flat] = self._data[flat]
            flat += 1
        return out

    # ----- fliplr/flipud (2D) -----
    fn fliplr(self) -> BoolTensor:
        if not (len(self._shape) == 2):
            pass
        var rows = self._shape[0]
        var cols = self._shape[1]
        var out = BoolTensor([rows, cols], False)
        var r = 0
        while r < rows:
            var c = 0
            while c < cols:
                out._data[r * cols + (cols - 1 - c)] = self._data[r * cols + c]
                c += 1
            r += 1
        return out

    fn flipud(self) -> BoolTensor:
        if not (len(self._shape) == 2):
            pass
        var rows = self._shape[0]
        var cols = self._shape[1]
        var out = BoolTensor([rows, cols], False)
        var r = 0
        while r < rows:
            var c = 0
            while c < cols:
                out._data[(rows - 1 - r) * cols + c] = self._data[r * cols + c]
                c += 1
            r += 1
        return out

    # ----- sliding window (1D) -----
    fn sliding_window(self, window: Int, step: Int = 1) -> BoolTensor:
        if not (len(self._shape) == 1):
            pass
        var n = self._shape[0]
        if not (window > 0 and step > 0 and window <= n):
            pass
        var count = 1 + (n - window) / step
        var out = BoolTensor([count, window], False)
        var i = 0
        while i < count:
            var j = 0
            while j < window:
                out._data[i * window + j] = self._data[i * step + j]
                j += 1
            i += 1
        return out

    # ----- pad (1D/2D) -----
    fn pad(self, pads: List[Int], value: Bool = False) -> BoolTensor:
        if len(self._shape) == 1:
            if not (len(pads) == 2):
                pass
            var n = self._shape[0]
            var out = BoolTensor([pads[0] + n + pads[1]], False)
            var i = 0
            while i < len(out._data):
                out._data[i] = value
                i += 1
            i = 0
            while i < n:
                out._data[pads[0] + i] = self._data[i]
                i += 1
            return out
        elif len(self._shape) == 2:
            if not (len(pads) == 4):
                pass
            var rows = self._shape[0]; var cols = self._shape[1]
            var new_rows = pads[0] + rows + pads[1]
            var new_cols = pads[2] + cols + pads[3]
            var out = BoolTensor([new_rows, new_cols], False)
            var r = 0
            while r < new_rows:
                var c = 0
                while c < new_cols:
                    out._data[r * new_cols + c] = value
                    c += 1
                r += 1
            r = 0
            while r < rows:
                var c = 0
                while c < cols:
                    out._data[(pads[0] + r) * new_cols + (pads[2] + c)] = self._data[r * cols + c]
                    c += 1
                r += 1
            return out
        else:
            return self

    # ----- matmul/tensordot -----
    fn matmul(self, other: BoolTensor) -> IntTensor:
        if not (len(self._shape) == 2 and len(other._shape) == 2):
            pass
        var A = self._shape[0]; var B = self._shape[1]
        var B2 = other._shape[0]; var C = other._shape[1]
        if not (B == B2):
            pass
        var out = IntTensor([A, C], 0)
        var i = 0
        while i < A:
            var k = 0
            while k < C:
                var s = 0
                var j = 0
                while j < B:
                    var a = (1 if self._data[i * B + j] else 0)
                    var b = (1 if other._data[j * C + k] else 0)
                    s = s + (a * b)
                    j += 1
                out._data[i * C + k] = s
                k += 1
            i += 1
        return out

    fn tensordot(self, other: BoolTensor, axes: Int = 1) -> IntTensor:
        if axes == 1:
            if len(self._shape) == 1 and len(other._shape) == 1:
                var n = self._shape[0]
                if not (n == other._shape[0]):
                    pass
                var s = 0
                var i = 0
                while i < n:
                    var a = (1 if self._data[i] else 0)
                    var b = (1 if other._data[i] else 0)
                    s = s + (a * b)
                    i += 1
                var out = IntTensor([1], 0)
                out._data[0] = s
                return out
            if len(self._shape) == 2 and len(other._shape) == 2:
                return self.matmul(other)
        return IntTensor([0], 0)

    # ----- sum (count of trues) -----
    fn sum(self) -> Int:
        var s = 0
        var i = 0
        while i < len(self._data):
            if self._data[i]: s = s + 1
            i += 1
        return s


    # ----- sum along axis -> reduces that axis (counts of True) -----
            var new_shape = List[Int]()
        var i = 0
        while i < rank:
            if i != ax: new_shape.append(self._shape[i])
            i += 1

        if len(new_shape) == 0:
            var out0 = IntTensor([1], 0)
            out0._data[0] = self.sum()
            return out0

        var out = IntTensor(new_shape, 0)
        var out_elems = 1
        i = 0
        while i < len(new_shape):
            out_elems = out_elems * new_shape[i]
            i += 1

        var idx = 0
        while idx < out_elems:
            var idxs = self._unravel_index(idx, new_shape)
            var full = List[Int]()
            var d = 0; var p = 0
            while d < rank:
                if d == ax:
                    full.append(0)
                else:
                    full.append(idxs[p]); p += 1
                d += 1
            var dim_ax = self._shape[ax]
            var sumv = 0
            var k = 0
            while k < dim_ax:
                full[ax] = k
                var flat = self._ravel_index(full, self._shape)
                if self._data[flat]: sumv = sumv + 1
                k += 1
            out._data[idx] = sumv
            idx += 1
        return out

    fn sum(self, axis: Optional[Int] = None) -> IntTensor:
        if axis is None:
            var out = IntTensor([1], 0)
            var s = 0
            var i = 0
            while i < len(self._data):
                if self._data[i]: s = s + 1
                i += 1
            out._data[0] = s
            return out
        else:
            var ax = axis.value()
            var rank = len(self._shape)
            if ax < 0: ax = ax + rank
            var new_shape = List[Int]()
            var i = 0
            while i < rank:
                if i != ax: new_shape.append(self._shape[i])
                i += 1
            if len(new_shape) == 0:
                return self.sum(None)
            var out = IntTensor(new_shape, 0)
            var out_elems = 1
            i = 0
            while i < len(new_shape):
                out_elems = out_elems * new_shape[i]
                i += 1
            var idx = 0
            while idx < out_elems:
                var idxs = self._unravel_index(idx, new_shape)
                var full = List[Int]()
                var d = 0; var p = 0
                while d < rank:
                    if d == ax:
                        full.append(0)
                    else:
                        full.append(idxs[p]); p += 1
                    d += 1
                var dim_ax = self._shape[ax]
                var sumv = 0
                var k = 0
                while k < dim_ax:
                    full[ax] = k
                    var flat = self._ravel_index(full, self._shape)
                    if self._data[flat]: sumv = sumv + 1
                    k += 1
                out._data[idx] = sumv
                idx += 1
            return out


    # ----- pad with (before, after) pairs for each axis -----
    fn pad(self, pad_width: List[(Int, Int)], constant: Bool = 0) -> BoolTensor:
        var rank = len(self._shape)
        if len(pad_width) != rank:
            return self

        var new_shape = List[Int]()
        var i = 0
        while i < rank:
            var before = pad_width[i][0]
            var after  = pad_width[i][1]
            new_shape.append(self._shape[i] + before + after)
            i += 1

        var out = BoolTensor(new_shape, constant)

        var total = 1
        i = 0
        while i < rank:
            total = total * self._shape[i]
            i += 1

        var idx = 0
        while idx < total:
            var coords = self._unravel_index(idx, self._shape)
            var shifted = List[Int]()
            var d = 0
            while d < rank:
                shifted.append(coords[d] + pad_width[d][0])
                d += 1
            var src_val = self._data[idx]
            var dst_idx = out._ravel_index(shifted, new_shape)
            out._data[dst_idx] = src_val
            idx += 1

        return out


# === Added: typed gather for BoolTensor (1D/2D; axis=0/1 for 2D) ===
fn gather(a: BoolTensor, axis: Int, indices: IntTensor) -> BoolTensor:
    var shp = a.shape()
    if shp.len() == 1:
        var K = indices.shape()[0]
        var out = BoolTensor([K], false)
        var i = 0
        while i < K:
            var idx = indices[i]
            out._data[i] = a._data[idx]
            i += 1
        return out
    elif shp.len() == 2:
        var R = shp[0]
        var C = shp[1]
        var K = indices.shape()[0]
        if axis == 0:
            var out = BoolTensor([K, C], false)
            var i = 0
            while i < K:
                var r = indices[i]
                var j = 0
                while j < C:
                    out._data[i * C + j] = a._data[r * C + j]
                    j += 1
                i += 1
            return out
        elif axis == 1:
            var out = BoolTensor([R, K], false)
            var j2 = 0
            while j2 < K:
                var c = indices[j2]
                var i = 0
                while i < R:
                    out._data[i * K + j2] = a._data[i * C + c]
                    i += 1
                j2 += 1
            return out
        else:
            assert false, "gather: axis must be 0 or 1 for 2D tensors"
            return BoolTensor([0], false)
    else:
        assert false, "gather: only supports 1D or 2D tensors in this version"
        return BoolTensor([0], false)


# === Added: copy_tensor (materialized) for BoolTensor ===
fn copy_tensor(x: BoolTensor) -> BoolTensor:
    var out = BoolTensor(x.shape(), false)
    var i = 0
    while i < x._data.len():
        out._data[i] = x._data[i]
        i += 1
    return out
