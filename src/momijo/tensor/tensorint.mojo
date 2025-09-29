# Project:      Momijo
# Module:       src.momijo.tensor.tensorint
# File:         tensorint.mojo
# Path:         src/momijo/tensor/tensorint.mojo
#
# Description:  Int-only tensor (Int storage). Ambiguity-free constructors + astype to float.
# License:      MIT

from momijo.tensor.utils import shape_product
from momijo.tensor.utils import compute_strides
from momijo.tensor.dtype import DType
from momijo.tensor.tensorfloat import FloatTensor  # for astype to float

struct IntTensor(ExplicitlyCopyable, Movable):

    fn ensure_1d_len(self) -> Int:
        if len(self._shape) != 1:
            return 0
        return self._shape[0]


    var _shape: List[Int]
    var _strides: List[Int]
    var _data: List[Int]

    fn __copyinit__(out self, other: Self):
        self._shape = other._shape
        self._strides = other._strides
        self._data = List[Int]()
        var i = 0
        while i < len(other._data):
            self._data.append(other._data[i])
            i += 1

    fn copy(self) -> Self:
        var out: IntTensor = self
        return out

    # 1D
    fn __init__(out self, data: List[Int]):
        self._shape = [len(data)]
        self._strides = compute_strides(self._shape)
        self._data = List[Int]()
        var i = 0
        while i < len(data):
            self._data.append(data[i])
            i += 1

    # 2D
    fn __init__(out self, data: List[List[Int]]):
        var rows = len(data)
        var cols = 0
        if rows > 0: cols = len(data[0])
        self._shape = [rows, cols]
        self._strides = compute_strides(self._shape)
        self._data = List[Int]()
        var r = 0
        while r < rows:
            var c = 0
            while c < cols:
                var __row = data[r]
                var __val: Int = __row[c]
                self._data.append(__val)
                c += 1
            r += 1

    # 3D
    fn __init__(out self, data: List[List[List[Int]]]):
        var d0 = len(data)
        var d1 = 0
        var d2 = 0
        if d0 > 0:
            d1 = len(data[0])
            if d1 > 0:
                d2 = len(data[0][0])
        self._shape = [d0, d1, d2]
        self._strides = compute_strides(self._shape)
        self._data = List[Int]()
        var i = 0
        while i < d0:
            var j = 0
            while j < d1:
                var k = 0
                while k < d2:
                    self._data.append(data[i][j][k])
                    k += 1
                j += 1
            i += 1

    # 4D
    fn __init__(out self, data: List[List[List[List[Int]]]]):
        var d0 = len(data)
        var d1 = 0
        var d2 = 0
        var d3 = 0
        if d0 > 0:
            d1 = len(data[0])
            if d1 > 0:
                d2 = len(data[0][0])
                if d2 > 0:
                    d3 = len(data[0][0][0])
        self._shape = [d0, d1, d2, d3]
        self._strides = compute_strides(self._shape)
        self._data = List[Int]()
        var i = 0
        while i < d0:
            var j = 0
            while j < d1:
                var k = 0
                while k < d2:
                    var l = 0
                    while l < d3:
                        self._data.append(data[i][j][k][l])
                        l += 1
                    k += 1
                j += 1
            i += 1

    fn __init__(out self, shape: List[Int], fill: Int):
        self._shape = shape
        self._strides = compute_strides(shape)
        self._data = List[Int]()
        var n = shape_product(shape)
        var i = 0
        while i < n:
            self._data.append(fill)
            i += 1

    # API
    fn shape(self) -> List[Int]: return self._shape
    fn __str__(self) -> String:  return "IntTensor(shape=" + self._shape.__str__() + ")"

    # Helper: compute total number of elements 
        
    fn len(self) -> Int:
        var n = 1
        var i = 0
        while i < len(self._shape):
            n *= self._shape[i]
            i += 1
        return n

    fn __len__(self) -> Int:
        return self.len()

    # Private: linear index from N-D indices (row-major fallback if strides missing)
    fn lin_index_nd(self, idxs: List[Int]) -> Int:
        # handle scalars / empty shape
        var ndim = len(self._shape)
        if ndim == 0:
            return 0

        # build a clamped index vector ii of length ndim
        var ii = List[Int]()
        var d = 0
        while d < ndim:
            var v = 0
            if d < len(idxs):
                v = idxs[d]
            var dim = self._shape[d]
            if dim <= 0:
                ii.append(0)
            else:
                if v < 0:
                    v = 0
                if v >= dim:
                    v = dim - 1
                ii.append(v)
            d += 1

        # if explicit strides exist and match rank, use them
        if len(self._strides) == ndim:
            var base = 0
            var i = 0
            while i < ndim:
                base += ii[i] * self._strides[i]
                i += 1
            return base

        # fallback: row-major strides computed on the fly
        var base2 = 0
        var mul = 1
        var k = ndim - 1
        while k >= 0:
            base2 += ii[k] * mul
            mul *= (self._shape[k] if self._shape[k] > 0 else 1)
            k -= 1
        return base2


    # Scalar extraction when tensor has exactly one element
    fn item(self) -> Int:
        self.__len__() == 1, "IntTensor.item(): tensor must contain exactly one element"
        if not (self.__len__() == 1):
            pass
        return self._data[0]

    # 1D access
    fn item(self, i0: Int) -> Int:
        len(self._shape) == 1, "IntTensor.item(i): 1D tensor required"
        if not (len(self._shape) == 1):
            pass
        var li = self.lin_index_nd([i0])
        return self._data[li]

    # 2D access
    fn item(self, i0: Int, i1: Int) -> Int:
        len(self._shape) == 2, "IntTensor.item(i,j): 2D tensor required"
        if not (len(self._shape) == 2):
            pass
        var li = self.lin_index_nd([i0, i1])
        return self._data[li]

    # 3D access
    fn item(self, i0: Int, i1: Int, i2: Int) -> Int:
        len(self._shape) == 3, "IntTensor.item(i,j,k): 3D tensor required"
        if not (len(self._shape) == 3):
            pass
        var li = self.lin_index_nd([i0, i1, i2])
        return self._data[li]

    # 4D access
    fn item(self, i0: Int, i1: Int, i2: Int, i3: Int) -> Int:
        len(self._shape) == 4, "IntTensor.item(i,j,k,l): 4D tensor required"
        if not (len(self._shape) == 4):
            pass
        var li = self.lin_index_nd([i0, i1, i2, i3])
        return self._data[li]
    
    
    fn select(self, axis: Int, index: Int) -> IntTensor:
        var ndim = len(self._shape)
        ndim >= 2 and ndim <= 4, "IntTensor.select: rank must be 2..4"
        if not (ndim >= 2 and ndim <= 4):
            pass
        axis >= 0 and axis < ndim, "IntTensor.select: bad axis"
        if not (axis >= 0 and axis < ndim):
            pass
        index >= 0 and index < self.shape[axis], "IntTensor.select: index out of range"
        if not (index >= 0 and index < self.shape[axis]):
            pass

        var out_shape = List[Int]()
        var d = 0
        while d < ndim:
            if d != axis: out_shape.append(self.shape[d])
            d += 1

        var out = IntTensor.zeros(out_shape)

        if ndim == 2:
            var R = self.shape[0]
            var C = self.shape[1]
            if axis == 0:
                var j = 0
                while j < C:
                    out[j] = self[index, j]
                    j += 1
            else:
                var i = 0
                while i < R:
                    out[i] = self[i, index]
                    i += 1
        elif ndim == 3:
            var A = self.shape[0]
            var B = self.shape[1]
            var C = self.shape[2]
            if axis == 0:
                var i = 0
                while i < B:
                    var j = 0
                    while j < C:
                        out[i, j] = self[index, i, j]
                        j += 1
                    i += 1
            elif axis == 1:
                var i = 0
                while i < A:
                    var j = 0
                    while j < C:
                        out[i, j] = self[i, index, j]
                        j += 1
                    i += 1
            else:
                var i = 0
                while i < A:
                    var j = 0
                    while j < B:
                        out[i, j] = self[i, j, index]
                        j += 1
                    i += 1
        else:  # ndim == 4
            var A = self.shape[0]
            var B = self.shape[1]
            var C = self.shape[2]
            var D = self.shape[3]
            if axis == 0:
                var i = 0
                while i < B:
                    var j = 0
                    while j < C:
                        var k = 0
                        while k < D:
                            out[i, j, k] = self[index, i, j, k]
                            k += 1
                        j += 1
                    i += 1
            elif axis == 1:
                var i = 0
                while i < A:
                    var j = 0
                    while j < C:
                        var k = 0
                        while k < D:
                            out[i, j, k] = self[i, index, j, k]
                            k += 1
                        j += 1
                    i += 1
            elif axis == 2:
                var i = 0
                while i < A:
                    var j = 0
                    while j < B:
                        var k = 0
                        while k < D:
                            out[i, j, k] = self[i, j, index, k]
                            k += 1
                        j += 1
                    i += 1
            else:
                var i = 0
                while i < A:
                    var j = 0
                    while j < B:
                        var k = 0
                        while k < C:
                            out[i, j, k] = self[i, j, k, index]
                            k += 1
                        j += 1
                    i += 1

        return out
 
 
    # --- Getters ---
    fn __getitem__(self, i0: Int) -> Int:
        len(self._shape) == 1, "IntTensor[i]: 1D tensor required"
        if not (len(self._shape) == 1):
            pass
        return self._data[self.lin_index_nd([i0])]

    fn __getitem__(self, i0: Int, i1: Int) -> Int:
        len(self._shape) == 2, "IntTensor[i,j]: 2D tensor required"
        if not (len(self._shape) == 2):
            pass
        return self._data[self.lin_index_nd([i0, i1])]

    fn __getitem__(self, i0: Int, i1: Int, i2: Int) -> Int:
        len(self._shape) == 3, "IntTensor[i,j,k]: 3D tensor required"
        if not (len(self._shape) == 3):
            pass
        return self._data[self.lin_index_nd([i0, i1, i2])]

    fn __getitem__(self, i0: Int, i1: Int, i2: Int, i3: Int) -> Int:
        len(self._shape) == 4, "IntTensor[i,j,k,l]: 4D tensor required"
        if not (len(self._shape) == 4):
            pass
        return self._data[self.lin_index_nd([i0, i1, i2, i3])]

    # --- Setters ---
    fn __setitem__(mut self, i0: Int, value: Int):
        len(self._shape) == 1, "IntTensor[i]=v: 1D tensor required"
        if not (len(self._shape) == 1):
            pass
        self._data[self.lin_index_nd([i0])] = value

    fn __setitem__(mut self, i0: Int, i1: Int, value: Int):
        len(self._shape) == 2, "IntTensor[i,j]=v: 2D tensor required"
        if not (len(self._shape) == 2):
            pass
        self._data[self.lin_index_nd([i0, i1])] = value

    fn __setitem__(mut self, i0: Int, i1: Int, i2: Int, value: Int):
        len(self._shape) == 3, "IntTensor[i,j,k]=v: 3D tensor required"
        if not (len(self._shape) == 3):
            pass
        self._data[self.lin_index_nd([i0, i1, i2])] = value

    fn __setitem__(mut self, i0: Int, i1: Int, i2: Int, i3: Int, value: Int):
        len(self._shape) == 4, "IntTensor[i,j,k,l]=v: 4D tensor required"
        if not (len(self._shape) == 4):
            pass
        self._data[self.lin_index_nd([i0, i1, i2, i3])] = value

    # Casting to float64 to match calls like: a.astype(tensor.float64())
    fn astype(self, dtype: DType, copy: Bool = True) -> FloatTensor:
        # We only support float64 target in this minimal API.
        # If someone asks for int32, just return a float copy (safe superset).
        var out = FloatTensor(self._shape, 0.0)
        var n = len(self._data)
        var i = 0
        while i < n:
            out._data.append(Float64(self._data[i]))
            i += 1
        return out


    @staticmethod
    fn zeros(shape: List[Int]) -> IntTensor:
        return IntTensor(shape, 0)

    @staticmethod
    fn zeros_like(other: IntTensor) -> IntTensor:
        return IntTensor(other.shape, 0)

    @staticmethod
    fn ones(shape: List[Int]) -> IntTensor:
        return IntTensor(shape, 1)

    @staticmethod
    fn full(shape: List[Int], value: Int) -> IntTensor:
        return IntTensor(shape, value)
    # # ----- Indexing up to 4D -----
    # # Modified: return sub-tensor for rank>=2; keeps scalar via get1 for vectors
    # fn __getitem__(self, idx: Int) -> IntTensor:
    #     var shp = self.shape()
    #     var rank = len(shp)
    #     if rank == 0:
    #         return IntTensor([1], 0)
    #     if rank == 1:
    #         var out = IntTensor([1], 0)
    #         out._data[0] = self._data[idx]
    #         return out
    #     elif rank == 2:
    #         var C = shp[1]
    #         var out = IntTensor([C], 0)
    #         var j = 0
    #         while j < C:
    #             out._data[j] = self._data[idx * C + j]
    #             j += 1
    #         return out
    #     else:
    #         assert rank == 3, "IntTensor.__getitem__(Int): only rank 1..3 supported"
    #         var M = shp[1]
    #         var N = shp[2]
    #         var out = IntTensor([M, N], 0)
    #         var j = 0
    #         while j < M:
    #             var k = 0
    #             while k < N:
    #                 out[j, k] = self.get3(idx, j, k)
    #                 k += 1
    #             j += 1
    #         return out

    # fn __setitem__(mut self, idx: Int, value: Int):
    #         self._data[idx] = value

    # fn __getitem__(self, idx2: (Int, Int)) -> Int:
    #     var r = idx2[0]
    #     var c = idx2[1]
    #     var cols = self._shape[1]
    #     return self._data[r * cols + c]

    # fn __setitem__(mut self, idx2: (Int, Int), value: Int):
    #     var r = idx2[0]
    #     var c = idx2[1]
    #     var cols = self._shape[1]
    #     self._data[r * cols + c] = value

    # fn __getitem__(self, idx3: (Int, Int, Int)) -> Int:
    #     var i = idx3[0]
    #     var j = idx3[1]
    #     var k = idx3[2]
    #     var s1 = self._shape[1]
    #     var s2 = self._shape[2]
    #     var flat = ((i * s1) + j) * s2 + k
    #     return self._data[flat]

    # fn __setitem__(mut self, idx3: (Int, Int, Int), value: Int):
    #     var i = idx3[0]
    #     var j = idx3[1]
    #     var k = idx3[2]
    #     var s1 = self._shape[1]
    #     var s2 = self._shape[2]
    #     var flat = ((i * s1) + j) * s2 + k
    #     self._data[flat] = value

    # fn __getitem__(self, idx4: (Int, Int, Int, Int)) -> Int:
    #     var a = idx4[0]
    #     var b = idx4[1]
    #     var c = idx4[2]
    #     var d = idx4[3]
    #     var s1 = self._shape[1]
    #     var s2 = self._shape[2]
    #     var s3 = self._shape[3]
    #     var flat = (((a * s1) + b) * s2 + c) * s3 + d
    #     return self._data[flat]

    # fn __setitem__(mut self, idx4: (Int, Int, Int, Int), value: Int):
    #     var a = idx4[0]
    #     var b = idx4[1]
    #     var c = idx4[2]
    #     var d = idx4[3]
    #     var s1 = self._shape[1]
    #     var s2 = self._shape[2]
    #     var s3 = self._shape[3]
    #     var flat = (((a * s1) + b) * s2 + c) * s3 + d
    #     self._data[flat] = value

    # Convenience helpers for 2D
    fn row(self, r: Int) -> IntTensor:
        var cols = self._shape[1]
        var out = IntTensor([cols], 0)
        var c = 0
        while c < cols:
            out._data.append(self._data[r * cols + c])
            c += 1
        return out

    fn col(self, c: Int) -> IntTensor:
        var rows = self._shape[0]
        var cols = self._shape[1]
        var out = IntTensor([rows], 0)
        var r = 0
        while r < rows:
            out._data.append(self._data[r * cols + c])
            r += 1
        return out


    # ----- Helpers -----
    fn _ensure_1d_self.__len__() -> Int:
        if len(self._shape) == 1:
            return self._shape[0]
        var n = 1
        var i = 0
        while i < len(self._shape):
            n = n * self._shape[i]
            i += 1
        return n

    fn _copy_flat(self) -> List[Int]:
        var n = len(self._data)
        var out = List[Int]()
        var i = 0
        while i < n:
            out.append(self._data[i])
            i += 1
        return out


    # ===== Algorithms =====

    # Returns a sorted copy (ascending)
    fn sort(self) -> IntTensor:
        var n = len(self._data)
        var buf = self._copy_flat()
        # insertion sort (simple & stable)
        var i = 1
        while i < n:
            var key = buf[i]
            var j = i - 1
            while j >= 0 and buf[j] > key:
                buf[j + 1] = buf[j]
                j -= 1
            buf[j + 1] = key
            i += 1
        var out = IntTensor([n], 0)
        out._data = buf
        return out

    # unique with counts (assumes 1D view)
    fn unique(self, return_counts: Bool = True) -> (IntTensor, IntTensor):
        var n = self.ensure_1d_len()
        var buf = self._copy_flat()
        # sort first
        var sorted = IntTensor([n], 0)
        sorted._data = buf
        sorted = sorted.sort()
        # scan
        var uniques = List[Int]()
        var counts = List[Int]()
        if n == 0:
            return (IntTensor([0], 0), IntTensor([0], 0))
        var curr = sorted._data[0]
        var cnt = 1
        var i = 1
        while i < n:
            if sorted._data[i] == curr:
                cnt += 1
            else:
                uniques.append(curr)
                counts.append(cnt)
                curr = sorted._data[i]
                cnt = 1
            i += 1
        uniques.append(curr)
        counts.append(cnt)
        var ut = IntTensor([len(uniques)], 0)
        ut._data = uniques
        var ct = IntTensor([len(counts)], 0)
        ct._data = counts
        return (ut, ct) if return_counts else (ut, IntTensor([0], 0))

    # bincount over non-negative ints
    fn bincount(self) -> IntTensor:
        var n = self.ensure_1d_len()
        var maxv = 0
        var i = 0
        while i < n:
            if self._data[i] > maxv:
                maxv = self._data[i]
            i += 1
        var out = IntTensor([maxv + 1], 0)
        i = 0
        while i < n:
            var v = self._data[i]
            if v >= 0:
                if v >= len(out._data):
                    # resize not needed because max computed
                    pass
                out._data[v] = out._data[v] + 1
            i += 1
        return out

    # histogram with explicit bin edges (ascending). Returns (counts, bin_edges)
    fn histogram(self, bins: List[Int]) -> (IntTensor, IntTensor):
        # bins defines edges; length B => B-1 bins. We output counts size B-1
        var B = len(bins)
        if not (B >= 2):
            pass
        var counts = List[Int]()
        # REMOVED resize: counts.resize(B - 1)
        var n = B - 1
        var i = 0
        while i < n:
            counts.append(0)
            i += 1
        i = 0
        while i < len(self._data):
            var x = self._data[i]
            # place x into bin j where bins[j] <= x < bins[j+1]
            var j = 0
            while j < B - 1:
                if x >= bins[j] and x < bins[j + 1]:
                    counts[j] = counts[j] + 1
                    break
                j += 1
            i += 1
        var ct = IntTensor([B - 1], 0)
        ct._data = counts
        var be = IntTensor([B], 0)
        be._data = bins
        return (ct, be)

    # digitize: return bin index for each x (like numpy, right-open)
    fn digitize(self, bins: List[Int]) -> IntTensor:
        var out = IntTensor([len(self._data)], 0)
        var i = 0
        while i < len(self._data):
            var x = self._data[i]
            var idx = 0
            while idx < len(bins) and x >= bins[idx]:
                idx += 1
            out._data[i] = idx - 1  # index of left edge
            i += 1
        return out

    # ----- Set operations on 1D views (results sorted unique) -----
    fn set_union(self, other: IntTensor) -> IntTensor:
        var u1, c1 = self.unique(True)
        var u2, c2 = other.unique(True)
        var i = 0
        var j = 0
        var out = List[Int]()
        while i < len(u1._data) and j < len(u2._data):
            var a = u1._data[i]
            var b = u2._data[j]
            if a == b:
                out.append(a); i += 1; j += 1
            elif a < b:
                out.append(a); i += 1
            else:
                out.append(b); j += 1
        while i < len(u1._data):
            out.append(u1._data[i]); i += 1
        while j < len(u2._data):
            out.append(u2._data[j]); j += 1
        var t = IntTensor([len(out)], 0); t._data = out; return t

    fn set_intersection(self, other: IntTensor) -> IntTensor:
        var u1, c1 = self.unique(True)
        var u2, c2 = other.unique(True)
        var i = 0
        var j = 0
        var out = List[Int]()
        while i < len(u1._data) and j < len(u2._data):
            var a = u1._data[i]
            var b = u2._data[j]
            if a == b:
                out.append(a); i += 1; j += 1
            elif a < b:
                i += 1
            else:
                j += 1
        var t = IntTensor([len(out)], 0); t._data = out; return t

    fn set_difference(self, other: IntTensor) -> IntTensor:
        var u1, c1 = self.unique(True)
        var u2, c2 = other.unique(True)
        var i = 0
        var j = 0
        var out = List[Int]()
        while i < len(u1._data):
            var a = u1._data[i]
            while j < len(u2._data) and u2._data[j] < a:
                j += 1
            if j >= len(u2._data) or u2._data[j] != a:
                out.append(a)
            i += 1
        var t = IntTensor([len(out)], 0); t._data = out; return t

    fn set_xor(self, other: IntTensor) -> IntTensor:
        var u1, c1 = self.unique(True)
        var u2, c2 = other.unique(True)
        var i = 0
        var j = 0
        var out = List[Int]()
        while i < len(u1._data) or j < len(u2._data):
            if i < len(u1._data) and (j >= len(u2._data) or u1._data[i] < u2._data[j]):
                out.append(u1._data[i]); i += 1
            elif j < len(u2._data) and (i >= len(u1._data) or u2._data[j] < u1._data[i]):
                out.append(u2._data[j]); j += 1
            else:
                # equal -> skip both
                i += 1; j += 1
        var t = IntTensor([len(out)], 0) 
        t._data = out 
        return t
 
  


    # ----- Index helpers -----

    fn unravel_index(self, flat: Int, shape: List[Int]) -> List[Int]:
        var n = len(shape)
        var idxs = List[Int]()
        var i = 0
        while i < n:
            idxs.append(0)
            i += 1

        var total = 1
        i = 0
        while i < n:
            var s = shape[i]
            if s <= 0:
                s = 1
            total = total * s
            i += 1

        var rem = flat
        if rem < 0:
            rem = 0
        if total > 0 and rem >= total:
            rem = total - 1

        i = n - 1
        while i >= 0:
            var s = shape[i]
            if s <= 0:
                s = 1
            var qr = int_divmod(rem, s)
            var q = qr[0]
            var r = qr[1]
            idxs[i] = r
            rem = q
            i -= 1

        return idxs


    fn ravel_index(self, idxs: List[Int], shape: List[Int]) -> Int:
        var flat = 0
        var i = 0
        while i < len(shape):
            flat = flat * shape[i] + idxs[i]
            i += 1
        return flat

    fn total_size(self) -> Int:
        var n = 1
        var i = 0
        while i < len(self._shape):
            n = n * self._shape[i]
            i += 1
        return n


    fn moveaxis(self, source: Int, destination: Int) -> IntTensor:
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
        var out = IntTensor(new_shape, 0)
        out._data = List[Int]()
        # REMOVED resize: out._data.resize(len(self._data))
        var n = len(self._data)
        i = 0
        while i < n:
            out._data.append(0)
            i += 1
        n = self.total_size()
        var flat = 0
        var idxs:List[Int]
        while flat < n:
            idxs = self.unravel_index(flat, self._shape)
            var new_idxs = List[Int]()
            var t = 0
            while t < rank:
                new_idxs.append(idxs[perm[t]])
                t += 1
            var new_flat = self.ravel_index(new_idxs, new_shape)
            out._data[new_flat] = self._data[flat]
            flat += 1
        return out

    fn swapaxes(self, a: Int, b: Int) -> IntTensor:
        return self.moveaxis(a, b).moveaxis(b, a)


    fn roll(self, shift: Int, axis: Int = 0) -> IntTensor:
        var rank = len(self._shape)
        if not (axis >= 0 and axis < rank):
            pass
        var dim = self._shape[axis]
        if dim == 0: return self
        var k = ((shift % dim) + dim) % dim
        if k == 0: return self
        var out = IntTensor(self._shape, 0)
        out._data = List[Int]()
        var n: Int = len(self._data)
        out._data = List[Int]()
        var i = 0
        while i < n:
            out._data.append(0)
            i += 1
        n = self.total_size()
        var flat = 0

        var idxs:List[Int]
        while flat < n:
            idxs = self.unravel_index(flat, self._shape)
            var dst = idxs
            dst[axis] = (idxs[axis] + k) % dim
            var new_flat = self.ravel_index(dst, self._shape)
            out._data[new_flat] = self._data[flat]
            flat += 1
        return out


    fn fliplr(self) -> IntTensor:
        if not (len(self._shape) == 2):
            pass
        var rows = self._shape[0]
        var cols = self._shape[1]
        var out = IntTensor([rows, cols], 0)
        var r = 0
        while r < rows:
            var c = 0
            while c < cols:
                out._data[r * cols + (cols - 1 - c)] = self._data[r * cols + c]
                c += 1
            r += 1
        return out

    fn flipud(self) -> IntTensor:
        if not (len(self._shape) == 2):
            pass
        var rows = self._shape[0]
        var cols = self._shape[1]
        var out = IntTensor([rows, cols], 0)
        var r = 0
        while r < rows:
            var c = 0
            while c < cols:
                out._data[(rows - 1 - r) * cols + c] = self._data[r * cols + c]
                c += 1
            r += 1
        return out


    fn pad(self, pads: List[Int], value: Int = 0) -> IntTensor:
        if len(self._shape) == 1:
            if not (len(pads) == 2):
                pass
            var n = self._shape[0]
            var out = IntTensor([pads[0] + n + pads[1]], 0)
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
            var out = IntTensor([new_rows, new_cols], 0)
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
    fn arange(start: Int, stop: Int, step: Int = 1) -> IntTensor:
        var _step = step
        if _step == 0:
            _step = 1

        var data = List[Int]()   # ← explicit element type, no ambiguity
        var i = start
        if _step > 0:
            while i < stop:
                data.append(i)
                i += _step
        else:
            while i > stop:
                data.append(i)
                i += _step

        return IntTensor(data)   # if you have __init__(data: List[Int])



 

  

   


    fn sliding_window(self, window: Int, step: Int = 1) -> IntTensor:
        if not (len(self._shape) == 1):
            pass
        var n = self._shape[0]
        if not (window > 0 and step > 0 and window <= n):
            pass
        var count = 1 + (n - window) / step
        var out = IntTensor([count, window], 0.0)
        var i = 0
        while i < count:
            var j = 0
            while j < window:
                out._data[i * window + j] = self._data[i * step + j]
                j += 1
            i += 1
        return out


    fn matmul(self, other: IntTensor) -> IntTensor:
        if not (len(self._shape) == 2 and len(other._shape) == 2):
            pass
        var A = self._shape[0]; var B = self._shape[1]
        var B2 = other._shape[0]; var C = other._shape[1]
        if not (B == B2):
            pass
        var out = IntTensor([A, C], 0.0)
        var i = 0
        while i < A:
            var k = 0
            while k < C:
                var s = 0.0
                var j = 0
                while j < B:
                    s = s + self._data[i * B + j] * other._data[j * C + k]
                    j += 1
                out._data[i * C + k] = s
                k += 1
            i += 1
        return out


    fn tensordot(self, other: IntTensor, axes: Int = 1) -> IntTensor:
        if axes == 1:
            if len(self._shape) == 1 and len(other._shape) == 1:
                var n = self._shape[0]
                if not (n == other._shape[0]):
                    pass
                var s = 0.0
                var i = 0
                while i < n:
                    s = s + self._data[i] * other._data[i]
                    i += 1
                var out = IntTensor([1], 0.0)
                out._data[0] = s
                return out
            if len(self._shape) == 2 and len(other._shape) == 2:
                return self.matmul(other)
        return IntTensor([0], 0.0)        

    fn sum(self) -> Int:
        var s = 0.0
        var i = 0
        while i < len(self._data):
            s = s + self._data[i]
            i += 1
        return s

    fn mean(self, axis: Optional[Int] = None) -> IntTensor:
        if axis is None:
            var out = IntTensor([1], 0.0)
            var s = 0.0
            var i = 0
            while i < len(self._data):
                s = s + self._data[i]
                i += 1
            if len(self._data) > 0:
                out._data[0] = s / Float64(len(self._data))
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
                return self.mean(None)
            var out = IntTensor(new_shape, 0.0)
            var out_elems = 1
            i = 0
            while i < len(new_shape):
                out_elems = out_elems * new_shape[i]
                i += 1
            var idx = 0
            while idx < out_elems:
                var idxs = self.unravel_index(idx, new_shape)
                var full = List[Int]()
                var d = 0; var p = 0
                while d < rank:
                    if d == ax:
                        full.append(0)
                    else:
                        full.append(idxs[p]); p += 1
                    d += 1
                var dim_ax = self._shape[ax]
                var sumv = 0.0
                var k = 0
                while k < dim_ax:
                    full[ax] = k
                    var flat = self.ravel_index(full, self._shape)
                    sumv = sumv + self._data[flat]
                    k += 1
                out._data[idx] = sumv / IntTensor(dim_ax)
                idx += 1
            return out
 


    fn sum(self, axis: Optional[Int] = None) -> IntTensor:
        if axis is None:
            var out = IntTensor([1], 0)
            var s = 0
            var i = 0
            while i < len(self._data):
                s = s + self._data[i]
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

            var idxs:List[Int]
            while idx < out_elems:
                idxs = self.unravel_index(idx, new_shape)
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
                    var flat = self.ravel_index(full, self._shape)
                    sumv = sumv + self._data[flat]
                    k += 1
                out._data[idx] = sumv
                idx += 1
            return out


    # ----- pad with (before, after) pairs for each axis -----
    fn pad(self, pad_width: List[(Int, Int)], constant: Int = 0) -> IntTensor:
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

        var out = IntTensor(new_shape, constant)

        var total = 1
        i = 0
        while i < rank:
            total = total * self._shape[i]
            i += 1

        var idx = 0
        var coords:List[Int]
        while idx < total:
            coords = self.unravel_index(idx, self._shape)
            var shifted = List[Int]()
            var d = 0
            while d < rank:
                shifted.append(coords[d] + pad_width[d][0])
                d += 1
            var src_val = self._data[idx]
            var dst_idx = out.ravel_index(shifted, new_shape)
            out._data[dst_idx] = src_val
            idx += 1

        return out
    # Added: scalar accessor for vectors
    fn get1(self, idx: Int) -> Int:
        var shp = self.shape()
        len(shp) == 1
        if not (len(shp) == 1):
            pass
        return self._data[idx]

    fn dtype_name(self) -> String:
        return "int32"

    fn take(self, index: IntTensor) -> FloatTensor:
        var n = len(index._data)
        var out = FloatTensor([n], 0.0)

        var total = numel_from_shape(self._shape)
        var rm_self = compute_row_major_strides(self._shape)

        var t = 0
        while t < n:
            var p = index._data[t]
            if p < 0:
                p = p + total
            (0 <= p and p < total), "take: index out of range"

            var mi = _unravel_index(p, self._shape, rm_self)
            var src_lin = ravel_index(mi, self._strides)
            out._data[t] = self._data[src_lin]
            t += 1
        return out

# === Added: typed gather for IntTensor (supports 1D and 2D; axis=0/1) ===
fn gather(a: IntTensor, axis: Int, indices: IntTensor) -> IntTensor:
    var shp = a.shape()
    if len(shp) == 1:
        var K = indices.shape()[0]
        var out = IntTensor([K], 0)
        var i = 0
        while i < K:
            var idx = indices[i]
            out._data[i] = a._data[idx]
            i += 1
        return out
    elif len(shp) == 2:
        var R = shp[0]
        var C = shp[1]
        var K = indices.shape()[0]
        if axis == 0:
            var out = IntTensor([K, C], 0)
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
            var out = IntTensor([R, K], 0)
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
            false, "gather: axis must be 0 or 1 for 2D tensors"
            if not (false):
                pass
            return IntTensor([0], 0)
    else:
        false, "gather: only supports 1D or 2D tensors in this version"
        if not (false):
            pass
        return IntTensor([0], 0)




# === Added: read/write helpers for last-dim plane on 3D IntTensor ===
fn plane(a: IntTensor, dim: Int, index: Int) -> IntTensor:
    var shp = a.shape()
    len(shp) == 3
    if not (len(shp) == 3):
        pass
    var B = shp[0]
    var M = shp[1]
    var N = shp[2]
    dim == 2
    if not (dim == 2):
        pass
    index >= 0 and index < N
    if not (index >= 0 and index < N):
        pass
    var out = IntTensor([B, M], 0)
    var i = 0
    while i < B:
        var j = 0
        while j < M:
            out[i, j] = a.get3(i, j, index)
            j += 1
        i += 1
    return out

fn write_plane(mut a: IntTensor, dim: Int, index: Int, rhs: IntTensor) -> IntTensor:
    var shp = a.shape()
    len(shp) == 3
    if not (len(shp) == 3):
        pass
    var B = shp[0]
    var M = shp[1]
    var N = shp[2]
    dim == 2
    if not (dim == 2):
        pass
    index >= 0 and index < N
    if not (index >= 0 and index < N):
        pass
    var rsh = rhs.shape()
    len(rsh) == 2
    if not (len(rsh) == 2):
        pass
    rsh[0] == B
    if not (rsh[0] == B):
        pass
    (rsh[1] == 1) or (rsh[1] == M)
    if not ((rsh[1] == 1) or (rsh[1] == M)):
        pass

    var i = 0
    while i < B:
        var j = 0
        while j < M:
            var v = rhs[i, (0 if rsh[1] == 1 else j)]
            a[i, j, index] = v
            j += 1
        i += 1
    return a


fn int_divmod(a: Int, b: Int) -> (Int, Int):
    if b <= 0:
        return (0, a)
    var q = 0
    var r = a
    while r >= b:
        r -= b
        q += 1
    return (q, r)