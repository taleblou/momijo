# Project:      Momijo
# Module:       src.momijo.tensor.tensorfloat
# File:         tensorfloat.mojo
# Path:         src/momijo/tensor/tensorfloat.mojo
#
# Description:  Float-only tensor (Float64 storage). Ambiguity-free constructors + astype to int.
# License:      MIT

from momijo.tensor.utils import shape_product
from momijo.tensor.utils import compute_strides
from momijo.tensor.dtype import DType
from momijo.tensor.dtype import DType, float64
from collections.dict import Dict
from collections.list import List
from momijo.tensor.utils import shape_product
from momijo.tensor.utils import compute_strides
from pathlib.path import Path

from math import cosh
from math import sinh
from math import tanh
from math import sqrt
from math import log
from math import exp
from math import tan
from math import cos
from math import sin



struct FloatTensor(ExplicitlyCopyable, Movable):

 
    fn ensure_1d_len(self) -> Int:
        if len(self._shape) != 1:
            return 0
        return self._shape[0]


    var _shape: List[Int]
    var _strides: List[Int]
    var _data: List[Float64]

    fn __copyinit__(out self, other: Self):
        self._shape = other._shape
        self._strides = other._strides
        self._data = List[Float64]()
        var i = 0
        while i < len(other._data):
            self._data.append(other._data[i])
            i += 1

    fn copy(self) -> Self:
        var out: FloatTensor = self
        return out

    # 1D
    fn __init__(out self, data: List[Float64]):
        self._shape = [len(data)]
        self._strides = compute_strides(self._shape)
        self._data = List[Float64]()
        var i = 0
        while i < len(data):
            self._data.append(data[i])
            i += 1

    # 2D
    fn __init__(out self, data: List[List[Float64]]):
        var rows = len(data)
        var cols = 0
        if rows > 0: cols = len(data[0])
        self._shape = [rows, cols]
        self._strides = compute_strides(self._shape)
        self._data = List[Float64]()
        var r = 0
        while r < rows:
            var c = 0
            while c < cols:
                var __row = data[r]
                var __val: Float64 = __row[c]
                self._data.append(__val)
                c += 1
            r += 1

    # 3D
    fn __init__(out self, data: List[List[List[Float64]]]):
        var d0 = len(data)
        var d1 = 0
        var d2 = 0
        if d0 > 0:
            d1 = len(data[0])
            if d1 > 0:
                d2 = len(data[0][0])
        self._shape = [d0, d1, d2]
        self._strides = compute_strides(self._shape)
        self._data = List[Float64]()
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
    fn __init__(out self, data: List[List[List[List[Float64]]]]):
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
        self._data = List[Float64]()
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

    fn __init__(out self, shape: List[Int], fill: Float64):
        self._shape = shape
        self._strides = compute_strides(shape)
        self._data = List[Float64]()
        var n = shape_product(shape)
        var i = 0
        while i < n:
            self._data.append(fill)
            i += 1


     
    # -------------- Elementwise ops with broadcasting --------------------------
    # Helper: compute broadcasted shape of two shapes (NumPy rules)
    @staticmethod
    fn broadcast_shape(a: List[Int], b: List[Int]) -> List[Int]:
        var ra = len(a)
        var rb = len(b)
        var r = max(ra, rb)
        var out = List[Int]()
        # build from left to right after aligning right
        var i = 0
        while i < r:
            var ad = 1
            var bd = 1
            if i >= r - ra:
                ad = a[i - (r - ra)]
            if i >= r - rb:
                bd = b[i - (r - rb)]
            (ad == bd) or (ad == 1) or (bd == 1), "broadcast: incompatible dims"
            var od = ad if ad != 1 else bd
            out.append(od)
            i += 1
        return out

    # Internal: binary op with broadcasting
    fn binary_broadcast(self, other: FloatTensor, which: Int) -> FloatTensor:
        var a_shape = self._shape
        var b_shape = other._shape
        var out_shape = FloatTensor.broadcast_shape(a_shape, b_shape)
        var out = FloatTensor(out_shape, 0.0)

        var out_rm = compute_row_major_strides(out_shape)
        var total = out.numel()

        # align shapes to same rank by pre-padding 1s on the left
        var r = len(out_shape)
        var a_aligned = List[Int]()
        var b_aligned = List[Int]()
        var i = 0
        while i < r - len(a_shape):
            a_aligned.append(1)
            i += 1
        i = 0
        while i < len(a_shape):
            a_aligned.append(a_shape[i])
            i += 1
        i = 0
        while i < r - len(b_shape):
            b_aligned.append(1)
            i += 1
        i = 0
        while i < len(b_shape):
            b_aligned.append(b_shape[i])
            i += 1

        # compute per-dim effective strides (0 stride when broadcasting)
        var a_eff = List[Int]()
        var b_eff = List[Int]()
        i = 0
        while i < r:
            a_eff.append(0 if a_aligned[i] == 1 else self._strides[i - (r - len(self._shape))] if i >= r - len(self._shape) else 0)
            b_eff.append(0 if b_aligned[i] == 1 else other._strides[i - (r - len(other._shape))] if i >= r - len(other._shape) else 0)
            i += 1

        var lin = 0
        while lin < total:
            var md = unravel_index(lin, out_shape, out_rm)
            var ai = 0
            var bi = 0
            i = 0
            while i < r:
                ai += md[i] * a_eff[i]
                bi += md[i] * b_eff[i]
                i += 1
            var av = self._data[ai]
            var bv = other._data[bi]
            var cv = 0.0
            if which == 0:
                cv = av + bv
            elif which == 1:
                cv = av - bv
            elif which == 2:
                cv = av * bv
            else:
                cv = av / bv
            out._data[lin] = cv
            lin += 1
        return out

    # Public ops
    fn add(self, other: FloatTensor) -> FloatTensor:
        return self.binary_broadcast(other, 0)

    fn sub(self, other: FloatTensor) -> FloatTensor:
        return self.binary_broadcast(other, 1)

    fn mul(self, other: FloatTensor) -> FloatTensor:
        return self.binary_broadcast(other, 2)

    fn divide(self, other: FloatTensor) -> FloatTensor:
        return self.binary_broadcast(other, 3)
    # -------------- Divide aliases --------------------------------------------- 

    fn div(self, other: FloatTensor) -> FloatTensor:
        return self.binary_broadcast(other, 3)

    # Operator overloads
    fn __add__(self, other: FloatTensor) -> FloatTensor:
        return self.binary_broadcast(other, 0)

    fn __sub__(self, other: FloatTensor) -> FloatTensor:
        return self.binary_broadcast(other, 1)

    fn __mul__(self, other: FloatTensor) -> FloatTensor:
        return self.binary_broadcast(other, 2)

    fn __truediv__(self, other: FloatTensor) -> FloatTensor:
        return self.binary_broadcast(other, 3)

    fn __neg__(self) -> FloatTensor:
        var out = FloatTensor(self._shape, 0.0)
        var n = self.numel()
        var i = 0
        while i < n:
            out._data[i] = -self._data[i]
            i += 1
        out._strides = self._strides
        return out


    # -------------- Scalar math (elementwise) ----------------------------------
    fn sub_scalar(self, v: Float64) -> FloatTensor:
        var out = FloatTensor(self._shape, 0.0)
        var n = self.numel()
        var i = 0
        while i < n:
            out._data[i] = self._data[i] - v
            i += 1
        out._strides = self._strides
        return out

    fn mul_scalar(self, v: Float64) -> FloatTensor:
        var out = FloatTensor(self._shape, 0.0)
        var n = self.numel()
        var i = 0
        while i < n:
            out._data[i] = self._data[i] * v
            i += 1
        out._strides = self._strides
        return out

    fn div_scalar(self, v: Float64) -> FloatTensor:
        var out = FloatTensor(self._shape, 0.0)
        var n = self.numel()
        var i = 0
        while i < n:
            out._data[i] = self._data[i] / v
            i += 1
        out._strides = self._strides
        return out

    fn pow_scalar(self, v: Float64) -> FloatTensor:
        var out = FloatTensor(self._shape, 0.0)
        var n = self.numel()
        var i = 0
        while i < n:
            out._data[i] = pow(self._data[i], v)
            i += 1
        out._strides = self._strides
        return out

    fn abs(self) -> FloatTensor:
        var out = FloatTensor(self._shape, 0.0)
        var n = self.numel()
        var i = 0
        while i < n:
            var a = self._data[i]
            out._data[i] = -a if a < 0.0 else a
            i += 1
        out._strides = self._strides
        return out

    fn exp(self) -> FloatTensor:
        var out = FloatTensor(self._shape, 0.0)
        var n = self.numel()
        var i = 0
        while i < n:
            out._data[i] = exp(self._data[i])
            i += 1
        out._strides = self._strides
        return out

    fn log(self) -> FloatTensor:
        var out = FloatTensor(self._shape, 0.0)
        var n = self.numel()
        var i = 0
        while i < n:
            out._data[i] = log(self._data[i])
            i += 1
        out._strides = self._strides
        return out

    fn sqrt(self) -> FloatTensor:
        var out = FloatTensor(self._shape, 0.0)
        var n = self.numel()
        var i = 0
        while i < n:
            out._data[i] = sqrt(self._data[i])
            i += 1
        out._strides = self._strides
        return out



    # Elementwise tangent
    fn tan(self) -> FloatTensor:
        var out = FloatTensor(self._shape, 0.0)
        var i = 0
        while i < len(self._data):
            out._data[i] = tan(self._data[i])
            i += 1
        return out


    # Elementwise negation
    fn neg(self) -> FloatTensor:
        var out = FloatTensor(self._shape, 0.0)
        var i = 0
        while i < len(self._data):
            out._data[i] = 0.0 - self._data[i]
            i += 1
        return out


    # Elementwise ReLU
    fn relu(self) -> FloatTensor:
        var out = FloatTensor(self._shape, 0.0)
        var i = 0
        while i < len(self._data):
            var v = self._data[i]
            out._data[i] = v if v > 0.0 else 0.0
            i += 1
        return out


    # Elementwise Sigmoid
    fn sigmoid(self) -> FloatTensor:
        var out = FloatTensor(self._shape, 0.0)
        var i = 0
        while i < len(self._data):
            var v = self._data[i]
            out._data[i] = 1.0 / (1.0 + exp(0.0 - v))
            i += 1
        return out


    # Elementwise hyperbolic tangent
    fn tanh(self) -> FloatTensor:
        var out = FloatTensor(self._shape, 0.0)
        var i = 0
        while i < len(self._data):
            out._data[i] = tanh(self._data[i])
            i += 1
        return out


    # Elementwise hyperbolic sine
    fn sinh(self) -> FloatTensor:
        var out = FloatTensor(self._shape, 0.0)
        var i = 0
        while i < len(self._data):
            out._data[i] = sinh(self._data[i])
            i += 1
        return out


    # Elementwise hyperbolic cosine
    fn cosh(self) -> FloatTensor:
        var out = FloatTensor(self._shape, 0.0)
        var i = 0
        while i < len(self._data):
            out._data[i] = cosh(self._data[i])
            i += 1
        return out

    

    # Elementwise hyperbolic sin
    fn sin(self) -> FloatTensor:
        var out = FloatTensor(self._shape, 0.0)
        var n = len(self._data)
        var i = 0
        while i < n:
            out._data[i] = sin(self._data[i])
            i += 1
        return out



    # Elementwise hyperbolic cos
    fn cos(self) -> FloatTensor:
        var out = FloatTensor(self._shape, 0.0)
        var i = 0
        while i < len(self._data):
            out._data[i] = cos(self._data[i])
            i += 1
        return out

 
 

    # -------------- Clone (deep copy) -------------------------------------------
    fn clone(self) -> FloatTensor:
        var out = FloatTensor(self._shape, 0.0)
        var n = len(self._data)
        var i = 0
        while i < n:
            out._data[i] = self._data[i]
            i += 1
        # Preserve exact strides to keep view semantics identical
        out._strides = self._strides
        return out

    

    # -------------- PUT (scatter by flat indices) ------------------------------
    # Returns a NEW tensor with the given flat indices updated.
    fn put(self, index: IntTensor, value: Float64) -> FloatTensor:
        var out = self.clone()
        var n = len(index._data)
        var total = 1
        var i = 0
        while i < len(self._shape):
            total = total * self._shape[i]
            i += 1
        var rm = compute_row_major_strides(self._shape)
        var t = 0
        while t < n:
            var lin = index._data[t]
            0 <= lin and lin < total, "put: index out of range"
            if not (0 <= lin and lin < total):
                pass
            var mi = unravel_index(lin, self._shape, rm)
            var src_lin = ravel_index(mi, self._strides)
            out._data[src_lin] = value
            t += 1
        return out

    fn put(self, index: IntTensor, values: FloatTensor) -> FloatTensor:
        var out = self.clone()
        var n = len(index._data)
        n == values.numel(), "put: values length must match index length"
        if not (n == values.numel()):
            pass
        var total = 1
        var i = 0
        while i < len(self._shape):
            total = total * self._shape[i]
            i += 1
        var rm = compute_row_major_strides(self._shape)
        var t = 0
        while t < n:
            var lin = index._data[t]
            0 <= lin and lin < total, "put: index out of range"
            if not (0 <= lin and lin < total):
                pass
            var mi = unravel_index(lin, self._shape, rm)
            var src_lin = ravel_index(mi, self._strides)
            out._data[src_lin] = values._data[t]
            t += 1
        return out

    # -------------- reshape_like ------------------------------------------------
    fn reshape_like(self, other: FloatTensor) -> FloatTensor:
        return self.view(other._shape)   

    

    # -------------- view (reshape with -1 support; materialized) ---------------
 
    # Replace your existing `view` with this non-mutating version.
    # It allows calls like: x.view([-1, 4])
    fn view(self, shape: List[Int]) -> FloatTensor:
        # copy input shape so we can edit (infer -1) without requiring a mutable argument
        var new_shape = List[Int]()
        var i = 0
        while i < len(shape):
            new_shape.append(shape[i])
            i += 1

        # infer a single -1 dimension
        var infer_pos = -1
        var known = 1
        i = 0
        while i < len(new_shape):
            var d = new_shape[i]
            if d == -1:
                infer_pos = i
            else:
                known = known * d
            i += 1

        # total elements of self
        var total = 1
        i = 0
        while i < len(self._shape):
            total = total * self._shape[i]
            i += 1

        if infer_pos >= 0:
            new_shape[infer_pos] = total // known

        # materialize: flatten then reshape as row-major
        var flat = self.flatten()
        var out = FloatTensor(new_shape, 0.0)
        var n = 1
        i = 0
        while i < len(new_shape):
            n = n * new_shape[i]
            i += 1

        i = 0
        while i < n:
            out._data[i] = flat._data[i]
            i += 1
        out._strides = compute_row_major_strides(new_shape)
        return out
 


    # -------------- permute / transpose_axes -----------------------------------
    fn permute(self, dims: List[Int]) -> FloatTensor:
        len(dims) == len(self._shape), "permute: dims rank mismatch"
        var new_shape = List[Int]()
        var i = 0
        while i < len(dims):
            new_shape.append(self._shape[dims[i]])
            i += 1
        # materialize by index mapping
        var out = FloatTensor(new_shape, 0.0)
        var rm_dst = compute_row_major_strides(new_shape)
        var rm_src = compute_row_major_strides(self._shape)
        var total = out.numel()
        var lin = 0
        while lin < total:
            var md = unravel_index(lin, new_shape, rm_dst)
            # map md -> ms
            var ms = List[Int]()
            i = 0
            while i < len(md):
                ms.append( md[i] )
                i += 1
            # build src multi-index by inverse of dims
            var src = List[Int]()
            i = 0
            while i < len(dims):
                src.append(0)
                i += 1
            i = 0
            while i < len(dims):
                src[dims[i]] = md[i]
                i += 1
            var src_lin = ravel_index(src, self._strides)
            out._data[lin] = self._data[src_lin]
            lin += 1
        return out
 
    

    
    # -------------- cat / stack -------------------------------------------------
    @staticmethod
    fn cat(tensors: List[FloatTensor], dim: Int) -> FloatTensor:
        len(tensors) > 0, "cat: empty list"
        var rank = len(tensors[0]._shape)
        0 <= dim and dim < rank, "cat: bad dim"
        # compute new shape
        var new_shape = List[Int]()
        var i = 0
        while i < rank:
            if i == dim:
                var s = 0
                var k = 0
                while k < len(tensors):
                    s += tensors[k]._shape[i]
                    k += 1
                new_shape.append(s)
            else:
                new_shape.append(tensors[0]._shape[i])
            i += 1
        var out = FloatTensor(new_shape, 0.0)
        # copy blocks
        var offset = 0
        var k = 0
        while k < len(tensors):
            var t = tensors[k]
            var before = out.narrow(dim, offset, t._shape[dim])
            before.copy_from(t)
            offset += t._shape[dim]
            k += 1
        return out

    @staticmethod
    fn stack(tensors: List[FloatTensor], dim: Int) -> FloatTensor:
        len(tensors) > 0, "stack: empty list"
        var base = tensors[0]
        var new_shape = base._shape
        new_shape.insert(dim, 1)
        var expanded = List[FloatTensor]()
        var i = 0
        while i < len(tensors):
            expanded.append(tensors[i].unsqueeze(dim))
            i += 1
        return FloatTensor.cat(expanded, dim)

    # -------------- unbind / split_sizes ---------------------------------------
    fn unbind(self, dim: Int) -> List[FloatTensor]:
        var out = List[FloatTensor]()
        var i = 0
        while i < self._shape[dim]:
            out.append(self.narrow(dim, i, 1).squeeze_all())
            i += 1
        return out

    fn split_sizes(self, sizes: List[Int], dim: Int) -> List[FloatTensor]:
        var out = List[FloatTensor]()
        var off = 0
        var i = 0
        while i < len(sizes):
            var n = sizes[i]
            out.append(self.narrow(dim, off, n))
            off += n
            i += 1
        return out
 
    
    

    # -------------- Comparisons vs scalar -> IntTensor mask --------------------
    fn gt_scalar(self, v: Float64) -> FloatTensor:
        var out = FloatTensor(self._shape, 0)
        var n = self.numel()
        var i = 0
        while i < n:
            out._data[i] = 1 if self._data[i] > v else 0
            i += 1
        out._strides = self._strides
        return out

    fn ge_scalar(self, v: Float64) -> FloatTensor:
        var out = FloatTensor(self._shape, 0)
        var n = self.numel()
        var i = 0
        while i < n:
            out._data[i] = 1 if self._data[i] >= v else 0
            i += 1
        out._strides = self._strides
        return out

    fn lt_scalar(self, v: Float64) -> FloatTensor:
        var out = FloatTensor(self._shape, 0)
        var n = self.numel()
        var i = 0
        while i < n:
            out._data[i] = 1 if self._data[i] < v else 0
            i += 1
        out._strides = self._strides
        return out

    fn le_scalar(self, v: Float64) -> FloatTensor:
        var out = FloatTensor(self._shape, 0)
        var n = self.numel()
        var i = 0
        while i < n:
            out._data[i] = 1 if self._data[i] <= v else 0
            i += 1
        out._strides = self._strides
        return out

      

    # -------------- 1D sort / argsort / topk -----------------------------------
    fn sort_values(self) -> FloatTensor:
        len(self._shape) == 1, "sort_values: only 1D supported"
        var n = self._shape[0]
        var out = self.clone()
        var i = 0
        while i < n:
            var j = i + 1
            while j < n:
                if out._data[j] < out._data[i]:
                    var tmp = out._data[i]
                    out._data[i] = out._data[j]
                    out._data[j] = tmp
                j += 1
            i += 1
        return out

    fn argsort(self) -> IntTensor:
        len(self._shape) == 1, "argsort: only 1D supported"
        var n = self._shape[0]
        var idx = IntTensor([n], 0)
        var i = 0
        while i < n:
            idx._data[i] = i
            i += 1
        i = 0
        while i < n:
            var j = i + 1
            while j < n:
                if self._data[idx._data[j]] < self._data[idx._data[i]]:
                    var t = idx._data[i]
                    idx._data[i] = idx._data[j]
                    idx._data[j] = t
                j += 1
            i += 1
        idx._strides = self._strides
        return idx

    fn topk(self, k: Int) -> (FloatTensor, IntTensor):
        len(self._shape) == 1, "topk: only 1D supported"
        var n = self._shape[0]
        0 < k and k <= n, "topk: invalid k"
        var vals = FloatTensor([k], 0.0)
        var inds = IntTensor([k], 0)
        var t = 0
        while t < k:
            vals._data[t] = -1.0/0.0  # -inf
            inds._data[t] = -1
            t += 1
        var i = 0
        while i < n:
            var v = self._data[i]
            # insert into vals if v is larger than current minimum in top-k
            var pos = -1
            var m = 0
            var min_v = vals._data[0]
            var min_i = 0
            while m < k:
                if vals._data[m] < min_v:
                    min_v = vals._data[m]
                    min_i = m
                m += 1
            if v > min_v:
                pos = min_i
            if pos >= 0:
                vals._data[pos] = v
                inds._data[pos] = i
            i += 1
        return (vals, inds)

    # -------------- Matrix-vector multiply (2D @ 1D) ---------------------------
    fn matmul_vec(self, x: FloatTensor) -> FloatTensor:
        len(self._shape) == 2, "matmul_vec: self must be 2D"
        var m = self._shape[0]
        var n = self._shape[1]
        var xn = 0
        if len(x._shape) == 1:
            xn = x._shape[0]
        else:
            len(x._shape) == 2 and x._shape[1] == 1, "matmul_vec: x must be 1D or (n,1)"
            xn = x._shape[0]
        n == xn, "matmul_vec: dim mismatch"
        var y = FloatTensor([m], 0.0)
        var i = 0
        while i < m:
            var s = 0.0
            var j = 0
            while j < n:
                var a_idx = i * self._strides[0] + j * self._strides[1]
                var xv = x._data[j] if len(x._shape) == 1 else x._data[j * (x._strides[0])]
                s += self._data[a_idx] * xv
                j += 1
            y._data[i] = s
            i += 1
        return y

    # -------------- Linear solve (Gaussian elimination, square A) --------------
    fn solve(self, b: FloatTensor) -> FloatTensor:
        len(self._shape) == 2, "solve: A must be 2D square"
        self._shape[0] == self._shape[1], "solve: A must be square"
        var n = self._shape[0]
        var x = FloatTensor([n], 0.0)
        # Build augmented matrix [A | b]
        var aug = FloatTensor([n, n + 1], 0.0)
        var i = 0
        while i < n:
            var j = 0
            while j < n:
                var a_idx = i * self._strides[0] + j * self._strides[1]
                var aug_idx = i * aug._strides[0] + j * aug._strides[1]
                aug._data[aug_idx] = self._data[a_idx]
                j += 1
            var bv = b._data[i] if len(b._shape) == 1 else b._data[i * b._strides[0]]
            var rhs_idx = i * aug._strides[0] + n * aug._strides[1]
            aug._data[rhs_idx] = bv
            i += 1
        # Forward elimination with partial pivoting
        var k = 0
        while k < n:
            var piv = k
            var r = k + 1
            while r < n:
                var idx_rk = r * aug._strides[0] + k * aug._strides[1]
                var idx_pk = piv * aug._strides[0] + k * aug._strides[1]
                if abs(aug._data[idx_rk]) > abs(aug._data[idx_pk]):
                    piv = r
                r += 1
            # swap rows k and piv
            if piv != k:
                var c = k
                while c < n + 1:
                    var a1 = k * aug._strides[0] + c * aug._strides[1]
                    var a2 = piv * aug._strides[0] + c * aug._strides[1]
                    var tmp = aug._data[a1]
                    aug._data[a1] = aug._data[a2]
                    aug._data[a2] = tmp
                    c += 1
            var diag = aug._data[k * aug._strides[0] + k * aug._strides[1]]
            (diag != 0.0), "solve: singular matrix"
            var i2 = k + 1
            while i2 < n:
                var f = aug._data[i2 * aug._strides[0] + k * aug._strides[1]] / diag
                var c2 = k
                while c2 < n + 1:
                    var p1 = i2 * aug._strides[0] + c2 * aug._strides[1]
                    var p2 = k * aug._strides[0] + c2 * aug._strides[1]
                    aug._data[p1] = aug._data[p1] - f * aug._data[p2]
                    c2 += 1
                i2 += 1
            k += 1
        # Back substitution
        var i3 = n - 1
        while i3 >= 0:
            var s = aug._data[i3 * aug._strides[0] + n * aug._strides[1]]
            var j3 = i3 + 1
            while j3 < n:
                s -= aug._data[i3 * aug._strides[0] + j3 * aug._strides[1]] * x._data[j3]
                j3 += 1
            var d = aug._data[i3 * aug._strides[0] + i3 * aug._strides[1]]
            x._data[i3] = s / d
            i3 -= 1
        return x

    # -------------- Matrix inverse via solves (square A) ------------------------
    fn inv(self) -> FloatTensor:
        len(self._shape) == 2, "inv: A must be 2D square"
        self._shape[0] == self._shape[1], "inv: A must be square"
        var n = self._shape[0]
        var I = FloatTensor([n, n], 0.0)
        var r = 0
        while r < n:
            var c = 0
            while c < n:
                var idx = r * I._strides[0] + c * I._strides[1]
                I._data[idx] = 1.0 if r == c else 0.0
                c += 1
            r += 1
        var invA = FloatTensor([n, n], 0.0)
        var col = 0
        while col < n:
            var e = FloatTensor([n], 0.0)
            e._data[col] = 1.0
            var x = self.solve(e)
            var r2 = 0
            while r2 < n:
                invA._data[r2 * invA._strides[0] + col * invA._strides[1]] = x._data[r2]
                r2 += 1
            col += 1
        return invA

    # -------------- QR (classical Gram-Schmidt; square or tall) ----------------
    fn qr(self) -> (FloatTensor, FloatTensor):
        len(self._shape) == 2, "qr: A must be 2D"
        var m = self._shape[0]
        var n = self._shape[1]
        var Q = FloatTensor([m, n], 0.0)
        var R = FloatTensor([n, n], 0.0)
        var j = 0
        while j < n:
            # v = A[:, j]
            var i = 0
            while i < m:
                Q._data[i * Q._strides[0] + j * Q._strides[1]] = self._data[i * self._strides[0] + j * self._strides[1]]
                i += 1
            var k = 0
            while k < j:
                # R[k,j] = dot(Q[:,k], Q[:,j])
                var s = 0.0
                var t = 0
                while t < m:
                    var qk = Q._data[t * Q._strides[0] + k * Q._strides[1]]
                    var qj = Q._data[t * Q._strides[0] + j * Q._strides[1]]
                    s += qk * qj
                    t += 1
                R._data[k * R._strides[0] + j * R._strides[1]] = s
                # Q[:,j] -= R[k,j] * Q[:,k]
                t = 0
                while t < m:
                    var pos = t * Q._strides[0] + j * Q._strides[1]
                    Q._data[pos] = Q._data[pos] - s * Q._data[t * Q._strides[0] + k * Q._strides[1]]
                    t += 1
                k += 1
            # R[j,j] = norm(Q[:,j]); Q[:,j] /= R[j,j]
            var norm = 0.0
            i = 0
            while i < m:
                var v = Q._data[i * Q._strides[0] + j * Q._strides[1]]
                norm += v * v
                i += 1
            norm = sqrt(norm)
            (norm != 0.0), "qr: rank deficient"
            R._data[j * R._strides[0] + j * R._strides[1]] = norm
            i = 0
            while i < m:
                var pos2 = i * Q._strides[0] + j * Q._strides[1]
                Q._data[pos2] = Q._data[pos2] / norm
                i += 1
            j += 1
        return (Q, R)

    # -------------- SVD via eigen of A^T A (simple; square A) ------------------
    fn svd(self) -> (FloatTensor, FloatTensor, FloatTensor):
        len(self._shape) == 2, "svd: A must be 2D"
        self._shape[0] == self._shape[1], "svd: only square A supported here"
        var n = self._shape[0]
        # Compute C = A^T A (n x n)
        var C = FloatTensor([n, n], 0.0)
        var i = 0
        while i < n:
            var j = 0
            while j < n:
                var s = 0.0
                var k = 0
                while k < n:
                    var aik = self._data[k * self._strides[0] + i * self._strides[1]]
                    var ajk = self._data[k * self._strides[0] + j * self._strides[1]]
                    s += aik * ajk
                    k += 1
                C._data[i * C._strides[0] + j * C._strides[1]] = s
                j += 1
            i += 1
        # Power iteration with deflation to get eigenpairs
        var V = FloatTensor([n, n], 0.0)
        var S = FloatTensor([n], 0.0)
        var d = 0
        while d < n:
            # init v as e_d
            var v = FloatTensor([n], 0.0)
            v._data[d] = 1.0
            var it = 0
            while it < 40:
                # w = C * v
                var w = FloatTensor([n], 0.0)
                var r = 0
                while r < n:
                    var s2 = 0.0
                    var c2 = 0
                    while c2 < n:
                        s2 += C._data[r * C._strides[0] + c2 * C._strides[1]] * v._data[c2]
                        c2 += 1
                    w._data[r] = s2
                    r += 1
                # normalize w -> v
                var norm = 0.0
                r = 0
                while r < n:
                    norm += w._data[r] * w._data[r]
                    r += 1
                norm = sqrt(norm)
                if norm == 0.0:
                    break
                r = 0
                while r < n:
                    v._data[r] = w._data[r] / norm
                    r += 1
                it += 1
            # eigenvalue lambda ~ v^T C v
            var lam = 0.0
            var p = 0
            while p < n:
                var row = 0.0
                var q = 0
                while q < n:
                    row += C._data[p * C._strides[0] + q * C._strides[1]] * v._data[q]
                    q += 1
                lam += v._data[p] * row
                p += 1
            # store v in V[:,d]
            var r3 = 0
            while r3 < n:
                V._data[r3 * V._strides[0] + d * V._strides[1]] = v._data[r3]
                r3 += 1
            # deflate: C = C - lam * v v^T
            var r4 = 0
            while r4 < n:
                var c4 = 0
                while c4 < n:
                    var idx = r4 * C._strides[0] + c4 * C._strides[1]
                    C._data[idx] = C._data[idx] - lam * v._data[r4] * v._data[c4]
                    c4 += 1
                r4 += 1
            S._data[d] = sqrt(max(lam, 0.0))
            d += 1
        # U = A * V / S
        var U = FloatTensor([n, n], 0.0)
        var col = 0
        while col < n:
            var s = S._data[col]
            (s != 0.0), "svd: zero singular value"
            var j5 = 0
            while j5 < n:
                var sumv = 0.0
                var k5 = 0
                while k5 < n:
                    var aik = self._data[j5 * self._strides[0] + k5 * self._strides[1]]
                    var vkv = V._data[k5 * V._strides[0] + col * V._strides[1]]
                    sumv += aik * vkv
                    k5 += 1
                U._data[j5 * U._strides[0] + col * U._strides[1]] = sumv / s
                j5 += 1
            col += 1
        # Vh = V^T
        var Vh = V.transpose_axes(0, 1)
        return (U, S, Vh)





 
    fn __str__(self) -> String:  return "FloatTensor(shape=" + self._shape.__str__() + ")"

    # Helper: compute total number of elements
    fn self.__len__() -> Int:
        var n = 1
        var i = 0
        while i < len(self._shape):
            n *= self._shape[i]
            i += 1
        return n
 

    # Private: linear index from N-D indices (row-major fallback if strides missing)
    fn lin_index_nd(self, idxs: List[Int]) -> Int:
        var ndim = len(self._shape)
        len(idxs) == ndim, "FloatTensor: index rank mismatch"
        if not (len(idxs) == ndim):
            pass

        var d = 0
        while d < ndim:
            idxs[d] >= 0 and idxs[d] < self._shape[d], "FloatTensor: index out of range"
            if not (idxs[d] >= 0 and idxs[d] < self._shape[d]):
                pass
            d += 1

        var base = 0
        if len(self._strides) == ndim:
            var i = 0
            while i < ndim:
                base += idxs[i] * self._strides[i]
                i += 1
            return base
        else:
            # fallback: row-major (C-order)
            var idx = 0
            var j = 0
            while j < ndim:
                var mul = 1
                var k = j + 1
                while k < ndim:
                    mul *= self._shape[k]
                    k += 1
                idx += idxs[j] * mul
                j += 1
            return base + idx

    # Scalar extraction when tensor has exactly one element
    fn item(self) -> Float64:
        self.__len__() == 1, "FloatTensor.item(): tensor must contain exactly one element"
        if not (self.__len__() == 1):
            pass
        return self._data[0]

    # 1D access
    fn item(self, i0: Int) -> Float64:
        len(self._shape) == 1, "FloatTensor.item(i): 1D tensor required"
        if not (len(self._shape) == 1):
            pass
        var li = self.lin_index_nd([i0])
        return self._data[li]

    # 2D access
    fn item(self, i0: Int, i1: Int) -> Float64:
        len(self._shape) == 2, "FloatTensor.item(i,j): 2D tensor required"
        if not (len(self._shape) == 2):
            pass
        var li = self.lin_index_nd([i0, i1])
        return self._data[li]

    # 3D access
    fn item(self, i0: Int, i1: Int, i2: Int) -> Float64:
        len(self._shape) == 3, "FloatTensor.item(i,j,k): 3D tensor required"
        if not (len(self._shape) == 3):
            pass
        var li = self.lin_index_nd([i0, i1, i2])
        return self._data[li]

    # 4D access
    fn item(self, i0: Int, i1: Int, i2: Int, i3: Int) -> Float64:
        len(self._shape) == 4, "FloatTensor.item(i,j,k,l): 4D tensor required"
        if not (len(self._shape) == 4):
            pass
        var li = self.lin_index_nd([i0, i1, i2, i3])
        return self._data[li]


       # Helper: compute total number of elements
    fn len(self) -> Int:
        var n = 1
        var i = 0
        while i <len(self._shape):
            n *= self._shape[i]
            i += 1
        return n

    fn __len__(self) -> Int:
        return self.len()
    
    fn select(self, axis: Int, index: Int) -> FloatTensor:
        var ndim = len(self._shape)
        ndim >= 2 and ndim <= 4, "FloatTensor.select: rank must be 2..4"
        if not (ndim >= 2 and ndim <= 4):
            pass
        axis >= 0 and axis < ndim, "FloatTensor.select: bad axis"
        if not (axis >= 0 and axis < ndim):
            pass
        index >= 0 and index < self._shape[axis], "FloatTensor.select: index out of range"
        if not (index >= 0 and index < self._shape[axis]):
            pass

        var out_shape = List[Int]()
        var d = 0
        while d < ndim:
            if d != axis: out_shape.append(self._shape[d])
            d += 1

        var out = FloatTensor.zeros(out_shape)

        if ndim == 2:
            var R = self._shape[0]
            var C = self._shape[1]
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
            var A = self._shape[0]
            var B = self._shape[1]
            var C = self._shape[2]
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
            var A = self._shape[0]
            var B = self._shape[1]
            var C = self._shape[2]
            var D = self._shape[3]
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
    fn __getitem__(self, i0: Int) -> Float64:
        len(self._shape) == 1, "FloatTensor[i]: 1D tensor required"
        if not (len(self._shape) == 1):
            pass
        return self._data[self.lin_index_nd([i0])]

    fn __getitem__(self, i0: Int, i1: Int) -> Float64:
        len(self._shape) == 2, "FloatTensor[i,j]: 2D tensor required"
        if not (len(self._shape) == 2):
            pass
        return self._data[self.lin_index_nd([i0, i1])]

    fn __getitem__(self, i0: Int, i1: Int, i2: Int) -> Float64:
        len(self._shape) == 3, "FloatTensor[i,j,k]: 3D tensor required"
        if not (len(self._shape) == 3):
            pass
        return self._data[self.lin_index_nd([i0, i1, i2])]

    fn __getitem__(self, i0: Int, i1: Int, i2: Int, i3: Int) -> Float64:
        len(self._shape) == 4, "FloatTensor[i,j,k,l]: 4D tensor required"
        if not (len(self._shape) == 4):
            pass
        return self._data[self.lin_index_nd([i0, i1, i2, i3])]

    # --- Setters ---
    fn __setitem__(mut self, i0: Int, value: Float64):
        len(self._shape) == 1, "FloatTensor[i]=v: 1D tensor required"
        if not (len(self._shape) == 1):
            pass
        self._data[self.lin_index_nd([i0])] = value

    fn __setitem__(mut self, i0: Int, i1: Int, value: Float64):
        len(self._shape) == 2, "FloatTensor[i,j]=v: 2D tensor required"
        if not (len(self._shape) == 2):
            pass
        self._data[self.lin_index_nd([i0, i1])] = value

    fn __setitem__(mut self, i0: Int, i1: Int, i2: Int, value: Float64):
        len(self._shape) == 3, "FloatTensor[i,j,k]=v: 3D tensor required"
        if not (len(self._shape) == 3):
            pass
        self._data[self.lin_index_nd([i0, i1, i2])] = value

    fn __setitem__(mut self, i0: Int, i1: Int, i2: Int, i3: Int, value: Float64):
        len(self._shape) == 4, "FloatTensor[i,j,k,l]=v: 4D tensor required"
        if not (len(self._shape) == 4):
            pass
        self._data[self.lin_index_nd([i0, i1, i2, i3])] = value

    # Casting to int32 to match calls like: a.astype(tensor.int32())
    fn astype(self, dtype: DType, copy: Bool = True) -> IntTensor:
        # We only support float64 target in this minimal API.
        # If someone asks for int32, just return a float copy (safe superset).
        var out = IntTensor(self._shape, 0)
        var n = len(self._data)
        var i = 0
        while i < n:
            out._data.append(Float64(self._data[i]))
            i += 1
        return out


    @staticmethod
    fn zeros(shape: List[Int]) -> FloatTensor:
        # Works if your FloatTensor has a constructor (shape, fill: Float64)
        return FloatTensor(shape, 0.0)  

    @staticmethod
    fn zeros_like(other: FloatTensor) -> FloatTensor:
        return FloatTensor(other._shape, 0.0)

    @staticmethod
    fn ones(shape: List[Int]) -> FloatTensor:
        return FloatTensor(shape, 1.0)

    @staticmethod
    fn full(shape: List[Int], value: Float64) -> FloatTensor:
        return FloatTensor(shape, value)
    # # ----- Indexing up to 4D -----
    # # Modified: return sub-tensor for rank>=2; keeps scalar via get1 for vectors
    # fn __getitem__(self, idx: Int) -> FloatTensor:
    #     var shp = self._shape
    #     var rank = len(shp)
    #     if rank == 0:
    #         return FloatTensor([1], 0)
    #     if rank == 1:
    #         var out = FloatTensor([1], 0)
    #         out._data[0] = self._data[idx]
    #         return out
    #     elif rank == 2:
    #         var C = shp[1]
    #         var out = FloatTensor([C], 0)
    #         var j = 0
    #         while j < C:
    #             out._data[j] = self._data[idx * C + j]
    #             j += 1
    #         return out
    #     else:
    #         assert rank == 3, "FloatTensor.__getitem__(Int): only rank 1..3 supported"
    #         var M = shp[1]
    #         var N = shp[2]
    #         var out = FloatTensor([M, N], 0)
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
    fn row(self, r: Int) -> FloatTensor:
        var cols = self._shape[1]
        var out = FloatTensor([cols], 0)
        var c = 0
        while c < cols:
            out._data.append(self._data[r * cols + c])
            c += 1
        return out

    fn col(self, c: Int) -> FloatTensor:
        var rows = self._shape[0]
        var cols = self._shape[1]
        var out = FloatTensor([rows], 0)
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

    fn copy_flat(self) -> List[Float64]:
        var n = len(self._data)
        var out = List[Float64]()
        var i = 0
        while i < n:
            out.append(self._data[i])
            i += 1
        return out


    # ===== Algorithms =====

    # Returns a sorted copy (ascending)
    fn sort(self) -> FloatTensor:
        var n = len(self._data)
        var buf = self.copy_flat()
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
        var out = FloatTensor([n], 0.0)
        out._data = buf
        return out

    # unique with counts (assumes 1D view)
    # Paste INSIDE FloatTensor

    fn unique(self, return_counts: Bool = True) -> (FloatTensor, FloatTensor):
        # 1D only
        len(self._shape) == 1, "unique: only 1D supported"
        var n = self._shape[0]
        if n == 0:
            var empty = FloatTensor([0], 0.0)
            return (empty, empty if return_counts else empty)

        # sort a copy
        var sorted = self.sort_values()

        # scan
        var uniques = List[Float64]()
        var counts  = List[Int]()
        var curr = sorted._data[0]
        var cnt = 1
        var i = 1
        while i < n:
            var v = sorted._data[i]
            if v == curr:
                cnt += 1
            else:
                uniques.append(curr)
                counts.append(cnt)
                curr = v
                cnt = 1
            i += 1
        uniques.append(curr)
        counts.append(cnt)

        # pack outputs
        var m = len(uniques)
        var ut = FloatTensor([m], 0.0)
        i = 0
        while i < m:
            ut._data[i] = uniques[i]
            i += 1

        var ct = FloatTensor([m], 0.0)
        i = 0
        while i < m:
            ct._data[i] = Float64(counts[i])
            i += 1

        if return_counts:
            return (ut, ct)
        else:
            return (ut, FloatTensor([0], 0.0))

    # bincount over non-negative ints
    fn bincount(self) -> FloatTensor:
        len(self._shape) == 1, "bincount: only 1D supported"
        var n = self._shape[0]
        if n == 0:
            return FloatTensor([0], 0.0)

        var maxv = 0
        var i = 0
        while i < n:
            var vf = self._data[i]
            if vf == vf:                    # not NaN
                var vi = Int(vf)            # truncate toward zero
                if vi > maxv:
                    maxv = vi
            i += 1

        var out = FloatTensor([maxv + 1], 0.0)
        i = 0
        while i < n:
            var vf2 = self._data[i]
            if vf2 == vf2:                  # not NaN
                var vi2 = Int(vf2)
                if vi2 >= 0 and vi2 <= maxv:
                    out._data[vi2] = out._data[vi2] + 1.0
            i += 1
        return out

    # histogram with explicit bin edges (ascending). Returns (counts, bin_edges)
    fn histogram(self, bins: List[Float64]) -> (IntTensor, FloatTensor):
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
        var ct = FloatTensor([B - 1], 0)
        ct._data = counts
        var be = FloatTensor([B], 0)
        be._data = bins
        return (ct, be)

    # digitize: return bin index for each x (like numpy, right-open)
    fn digitize(self, bins: List[Float64]) -> IntTensor:
        var out = FloatTensor([len(self._data)], 0)
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
    fn set_union(self, other: FloatTensor) -> FloatTensor:
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
        var t = FloatTensor([len(out)], 0); t._data = out; return t

    fn set_intersection(self, other: FloatTensor) -> FloatTensor:
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
        var t = FloatTensor([len(out)], 0); t._data = out; return t

    fn set_difference(self, other: FloatTensor) -> FloatTensor:
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
        var t = FloatTensor([len(out)], 0); t._data = out; return t

    fn set_xor(self, other: FloatTensor) -> FloatTensor:
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
        var t = FloatTensor([len(out)], 0); t._data = out; return t
 
  



    fn total_size(self) -> Int:
        var n = 1
        var i = 0
        while i < len(self._shape):
            n = n * self._shape[i]
            i += 1
        return n


    fn moveaxis(self, source: Int, destination: Int) -> FloatTensor:
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
        var out = FloatTensor(new_shape, 0)
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

    fn swapaxes(self, a: Int, b: Int) -> FloatTensor:
        return self.moveaxis(a, b).moveaxis(b, a)


    fn roll(self, shift: Int, axis: Int = 0) -> FloatTensor:
        var rank = len(self._shape)
        if not (axis >= 0 and axis < rank):
            pass
        var dim = self._shape[axis]
        if dim == 0: return self
        var k = ((shift % dim) + dim) % dim
        if k == 0: return self
        var out = FloatTensor(self._shape, 0)
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


    fn fliplr(self) -> FloatTensor:
        if not (len(self._shape) == 2):
            pass
        var rows = self._shape[0]
        var cols = self._shape[1]
        var out = FloatTensor([rows, cols], 0)
        var r = 0
        while r < rows:
            var c = 0
            while c < cols:
                out._data[r * cols + (cols - 1 - c)] = self._data[r * cols + c]
                c += 1
            r += 1
        return out

    fn flipud(self) -> FloatTensor:
        if not (len(self._shape) == 2):
            pass
        var rows = self._shape[0]
        var cols = self._shape[1]
        var out = FloatTensor([rows, cols], 0)
        var r = 0
        while r < rows:
            var c = 0
            while c < cols:
                out._data[(rows - 1 - r) * cols + c] = self._data[r * cols + c]
                c += 1
            r += 1
        return out


    fn pad(self, pads: List[Int], value: Int = 0) -> FloatTensor:
        if len(self._shape) == 1:
            if not (len(pads) == 2):
                pass
            var n = self._shape[0]
            var out = FloatTensor([pads[0] + n + pads[1]], 0)
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
            var out = FloatTensor([new_rows, new_cols], 0)
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
    fn arange(start: Int, stop: Int, step: Int = 1) -> FloatTensor:
        var _step = step
        if _step == 0:
            _step = 1

        var out = FloatTensor(List[Float64]())  
        var i = start

        if _step > 0:
            while i < stop:
                out._data.append(Float64(i))
                i += _step
        else:
            while i > stop:
                out._data.append(Float64(i))
                i += _step

        out._shape = [len(out._data)]
        out._strides = compute_strides(out._shape)
        return out



   


    fn sliding_window(self, window: Int, step: Int = 1) -> FloatTensor:
        # Get length of axis 0; if rank != 1 still use axis 0 defensively
        var n = 0
        if len(self._shape) > 0:
            n = self._shape[0]

        # Ensure step is valid
        var _step = step
        if _step <= 0:
            _step = 1

        # Handle invalid input: return empty tensor [0,0]
        if window <= 0 or window > n or n == 0:
            var empty = FloatTensor(List[Float64]())
            empty._shape = [0, 0]
            empty._strides = compute_strides(empty._shape)
            return empty

        # Number of windows (integer division)
        var count = 1 + ((n - window) // _step)
        if count <= 0:
            var empty2 = FloatTensor(List[Float64]())
            empty2._shape = [0, 0]
            empty2._strides = compute_strides(empty2._shape)
            return empty2

        # Preallocate output data with zeros
        var total = count * window
        var data = List[Float64]()
        var k = 0
        while k < total:
            data.append(0.0)
            k += 1

        # Create output tensor
        var out = FloatTensor(data)
        out._shape = [count, window]
        out._strides = compute_strides(out._shape)

        # Fill with sliding windows
        var i = 0
        while i < count:
            var j = 0
            while j < window:
                out._data[i * window + j] = self._data[i * _step + j]
                j += 1
            i += 1

        return out


    fn matmul(self, other: FloatTensor) -> FloatTensor:
        if not (len(self._shape) == 2 and len(other._shape) == 2):
            pass
        var A = self._shape[0]; var B = self._shape[1]
        var B2 = other._shape[0]; var C = other._shape[1]
        if not (B == B2):
            pass
        var out = FloatTensor([A, C], 0.0)
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


    # tensordot with axes=1 for 1D·1D (dot) and 2D·2D (matmul); fallback returns empty tensor
    fn tensordot(self, other: FloatTensor, axes: Int = 1) -> FloatTensor:
        # 1D · 1D → scalar [1]
        if axes == 1 and len(self._shape) == 1 and len(other._shape) == 1:
            var n = 0
            if len(self._shape) > 0:
                n = self._shape[0]
            # size mismatch → return empty [0]
            if len(other._shape) == 0 or n != other._shape[0]:
                var empty = FloatTensor(List[Float64]())
                empty._shape = [0]
                empty._strides = compute_strides(empty._shape)
                return empty

            var s: Float64 = 0.0
            var i = 0
            while i < n:
                s = s + self._data[i] * other._data[i]
                i += 1

            var out_data = List[Float64]()
            out_data.append(s)
            var out = FloatTensor(out_data)
            out._shape = [1]
            out._strides = compute_strides(out._shape)
            return out

        # 2D · 2D → matrix multiply (delegates to matmul)
        if axes == 1 and len(self._shape) == 2 and len(other._shape) == 2:
            return self.matmul(other)

        # Fallback: unsupported case → empty [0]
        var fallback = FloatTensor(List[Float64]())
        fallback._shape = [0]
        fallback._strides = compute_strides(fallback._shape)
        return fallback


    # Sum over all elements (FloatTensor → Float64)
    fn sum(self) -> Float64:
        var s: Float64 = 0.0
        var i = 0
        var n = len(self._data)
        while i < n:
            s = s + self._data[i]
            i += 1
        return s


    # Optional: integer-accumulated sum if needed (truncates toward zero)
    fn sum_int(self) -> Int:
        var s: Int = 0
        var i = 0
        var n = len(self._data)
        while i < n:
            s = s + Int(self._data[i])
            i += 1
        return s


    # Mean over all elements or along a given axis; no asserts/exceptions; English comments only
    fn mean(self, axis: Optional[Int] = None) -> FloatTensor:
        # Scalar mean over all elements
        if axis is None:
            var n = len(self._data)
            var s: Float64 = 0.0
            var i = 0
            while i < n:
                s = s + self._data[i]
                i += 1

            var val: Float64 = 0.0
            if n > 0:
                val = s / Float64(n)

            var data = List[Float64]()
            data.append(val)

            var out = FloatTensor(data)
            out._shape = [1]                       # use scalar-as-[1]; adjust if your API wants []
            out._strides = compute_strides(out._shape)
            return out

        # Axis reduction
        var ax = axis.value()
        var rank = len(self._shape)

        # Normalize negative axis
        if ax < 0:
            ax = ax + rank

        # Clamp axis to valid range (no assert)
        if rank == 0:
            # no axes to reduce; fall back to scalar mean
            return self.mean(None)
        if ax < 0:
            ax = 0
        if ax >= rank:
            ax = rank - 1

        # Build output shape = input shape without axis ax
        var new_shape = List[Int]()
        var i = 0
        while i < rank:
            if i != ax:
                new_shape.append(self._shape[i])
            i += 1

        # If reducing all dims results in scalar, reuse scalar mean path
        if len(new_shape) == 0:
            return self.mean(None)

        # Compute number of output elements
        var out_elems = 1
        i = 0
        while i < len(new_shape):
            out_elems = out_elems * new_shape[i]
            i += 1

        # Preallocate output data with zeros
        var out_data = List[Float64]()
        var k = 0
        while k < out_elems:
            out_data.append(0.0)
            k += 1

        # Reduce along axis ax
        var idx = 0
        while idx < out_elems:
            # unravel idx in the reduced (new_shape) space
            var idxs = self.unravel_index(idx, new_shape)

            # build full indices by inserting a placeholder at ax
            var full = List[Int]()
            var d = 0
            var p = 0
            while d < rank:
                if d == ax:
                    full.append(0)
                else:
                    full.append(idxs[p])
                    p += 1
                d += 1

            var dim_ax = self._shape[ax]
            var sumv: Float64 = 0.0
            var t = 0
            while t < dim_ax:
                full[ax] = t
                var flat = self.ravel_index(full, self._shape)
                sumv = sumv + self._data[flat]
                t += 1

            out_data[idx] = sumv / Float64(dim_ax)
            idx += 1

        # Pack result tensor
        var out = FloatTensor(out_data)
        out._shape = new_shape
        out._strides = compute_strides(out._shape)
        return out

 


    fn sum(self, axis: Optional[Int] = None) -> FloatTensor:
        # Sum over all elements (scalar-as-[1])
        if axis is None:
            var n = len(self._data)
            var s: Float64 = 0.0
            var i = 0
            while i < n:
                s = s + self._data[i]
                i += 1

            var data = List[Float64]()
            data.append(s)

            var out = FloatTensor(data)
            out._shape = [1]  # represent scalar as [1]; adapt if your API differs
            out._strides = compute_strides(out._shape)
            return out

        # Axis reduction
        var ax = axis.value()
        var rank = len(self._shape)

        # Normalize and clamp axis without assertions
        if ax < 0:
            ax = ax + rank
        if rank == 0:
            return self.sum(None)
        if ax < 0:
            ax = 0
        if ax >= rank:
            ax = rank - 1

        # Build output shape by removing the reduced axis
        var new_shape = List[Int]()
        var i = 0
        while i < rank:
            if i != ax:
                new_shape.append(self._shape[i])
            i += 1

        # If reducing to scalar, reuse scalar path
        if len(new_shape) == 0:
            return self.sum(None)

        # Compute number of output elements
        var out_elems = 1
        i = 0
        while i < len(new_shape):
            out_elems = out_elems * new_shape[i]
            i += 1

        # Preallocate output storage
        var out_data = List[Float64]()
        var t = 0
        while t < out_elems:
            out_data.append(0.0)
            t += 1

        # Reduce along axis ax
        var idx = 0
        while idx < out_elems:
            # Unravel idx in reduced space (new_shape)
            var idxs = self.unravel_index(idx, new_shape)

            # Build full index by inserting a placeholder at ax
            var full = List[Int]()
            var d = 0
            var p = 0
            while d < rank:
                if d == ax:
                    full.append(0)
                else:
                    full.append(idxs[p])
                    p += 1
                d += 1

            var dim_ax = self._shape[ax]
            var sumv: Float64 = 0.0
            var k = 0
            while k < dim_ax:
                full[ax] = k
                var flat = self.ravel_index(full, self._shape)
                sumv = sumv + self._data[flat]
                k += 1

            out_data[idx] = sumv
            idx += 1

        # Pack result tensor
        var out = FloatTensor(out_data)
        out._shape = new_shape
        out._strides = compute_strides(out._shape)
        return out



    # ----- pad with (before, after) pairs for each axis -----
    fn pad(self, pad_width: List[(Int, Int)], constant: Int = 0) -> FloatTensor:
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

        var out = FloatTensor(new_shape, constant)

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
        var shp = self._shape
        len(shp) == 1
        if not (len(shp) == 1):
            pass
        return self._data[idx]

    @staticmethod
    fn linspace(start: Float64, stop: Float64, num: Int) -> FloatTensor:
        var out = FloatTensor([num], 0.0)
        if num <= 0:
            return out
        if num == 1:
            out._data[0] = start
            return out
        var step = (stop - start) / Float64(num - 1)
        var i = 0
        while i < num:
            out._data[i] = start + step * Float64(i)
            i += 1
        return out
    
    


    # ===== Public API on FloatTensor =====

    fn save_npy(self, path: String) -> None:
        var content = mn_dump(self._shape, self._data)
        write_text(path, content)

    @staticmethod
    fn load_npy(path: String) -> FloatTensor:
        var content = read_text(path)
        var (shape, vals) = mn_parse(content)
        var out = FloatTensor(shape, 0.0)
        var p = 0
        var n = len(vals)
        if len(out._data) < n: n = len(out._data)
        while p < n:
            out._data[p] = vals[p]
            p += 1
        return out

    @staticmethod
    fn save_npz(path: String, arrays: Dict[String, FloatTensor]) -> None:
        var builder = String("MNPZ v1\ncount: ")
        builder = builder + len(arrays).__str__() + "\n"

        for key in arrays.keys():
            var t_opt = arrays.get(key)        # Optional[FloatTensor]
            if t_opt is None:
                continue
            var t = t_opt.value()

            builder = builder + "name: " + key + "\n"
            builder = builder + mn_header(t._shape)
            builder = builder + "data:\n"

            var i = 0
            while i < len(t._data):
                builder = builder + t._data[i].__str__() + "\n"
                i += 1

        write_text(path, builder)


    @staticmethod
    fn load_npz(path: String) -> Dict[String, FloatTensor]:
        var text = read_text(path)
        var lines = split_lines(text)
        var out = Dict[String, FloatTensor]()
        var idx = 0

        # skip empties
        while idx < len(lines) and strip_str(lines[idx]).__len__() == 0:
            idx += 1
        if idx < len(lines) and strip_str(lines[idx]) == "MNPZ v1":
            idx += 1
        if idx < len(lines) and starts_with(strip_str(lines[idx]), "count:"):
            idx += 1

        while idx < len(lines):
            while idx < len(lines) and not starts_with(strip_str(lines[idx]), "name:"):
                idx += 1
            if idx >= len(lines):
                break

            var name_line = strip_str(lines[idx])
            idx += 1

            # extract name after "name:"
            var name = String("")
            var start = 5
            while start < name_line.__len__() and name_line[start] == ' ':
                start += 1
            var j = start
            while j < name_line.__len__():
                name = name + String(name_line[j])
                j += 1
            name = strip_str(name)

            # collect section
            var section = List[String]()
            while idx < len(lines) and not starts_with(strip_str(lines[idx]), "name:"):
                section.append(lines[idx])
                idx += 1

            # parse MNPY inside section
            var (shape, start_idx) = parse_mn_header(section)

            var k = start_idx
            while k < len(section) and strip_str(section[k]).__len__() == 0:
                k += 1
            if k < len(section) and starts_with(strip_str(section[k]), "data"):
                k += 1

            var vals = List[Float64]()
            while k < len(section):
                var sline = strip_str(section[k])
                if sline.__len__() > 0:
                    vals.append(to_float_simple(sline))
                k += 1

            if len(shape) == 0:
                shape = [len(vals)]

            var t = FloatTensor(shape, 0.0)
            var p = 0
            var n = len(vals)
            if len(t._data) < n:
                n = len(t._data)
            while p < n:
                t._data[p] = vals[p]
                p += 1

            out[name] = t

        return out



    @staticmethod
    fn save_csv(path: String, data: FloatTensor, header: List[String] = []) -> None:
        var lines = to_csv_lines(data._shape, data._data, header)
        var builder = String("")
        var i = 0
        while i < len(lines):
            builder = builder + lines[i] + "\n"
            i += 1
        write_text(path, builder)

    @staticmethod
    fn load_csv(path: String, skiprows: Int = 0) -> FloatTensor:
        var content = read_text(path)
        var (nrows, ncols, flat) = csv_parse(content, skiprows)
        if nrows == 0 and ncols == 0:
            return FloatTensor([0, 0], 0.0)
        var out = FloatTensor([nrows, ncols], 0.0)
        var p = 0
        while p < len(flat) and p < len(out._data):
            out._data[p] = flat[p]
            p += 1
        return out

 


    
    # === Added: copy_tensor (materialized) for FloatTensor ===
    fn copy_tensor(x: FloatTensor) -> FloatTensor:
        var out = FloatTensor(x.shape(), 0.0)
        var i = 0
        while i < x._data.len():
            out._data[i] = x._data[i]
            i += 1
        return out


    # Factory: build from a flat row-major list and a shape
    @staticmethod
    fn from_flat(flat: List[Float64], shape: List[Int]) -> FloatTensor:
        var sz = 1
        var i = 0
        while i < len(shape):
            sz = sz * shape[i]
            i += 1

        if sz != len(flat):
            # non-raising fallback
            return FloatTensor([0, 0], 0.0)

        var out = FloatTensor(shape, 0.0)   # or FloatTensor.zeros(shape) if you have it
        var j = 0
        while j < sz:
            out._data[j] = flat[j]
            j += 1
        return out


    # === Added: typed gather for FloatTensor (supports 1D and 2D; axis=0/1) ===
   # Module-level (or place with your other free functions)
    # --- module level (outside any impl) ---

 


    # Accept FloatTensor indices by converting, then delegate to the unique helper above.
    fn gather(a: FloatTensor, axis: Int, indices: FloatTensor) -> FloatTensor:
        return gather_int(a, axis, indices.to_int())

    
    fn to_int(self) -> IntTensor:
        # Truncate toward zero
        var out = IntTensor(self._shape, 0)
        var n = len(self._data)
        var i = 0
        while i < n:
            out._data[i] = Int(self._data[i])
            i += 1
        out._strides = self._strides
        return out

    fn to_int32(self) -> IntTensor:
        return self.to_int()

            # --- Introspection helpers ------------------------------------------------
    fn dtype_name(self) -> String:
        return "float64"

    fn shape_str(self) -> String:
        var s = String("[")
        var i = 0
        while i < len(self._shape):
            s = s + self._shape[i].__str__()
            if i + 1 < len(self._shape): s = s + ","
            i += 1
        s = s + "]"
        return s

    fn stride_str(self) -> String:
        var strides = compute_row_major_strides(self._shape)
        var s = String("[")
        var i = 0
        while i < len(strides):
            s = s + strides[i].__str__()
            if i + 1 < len(strides): s = s + ","
            i += 1
        s = s + "]"
        return s

    # --- Dtype conversions -----------------------------------------------------
    fn to_float64(self) -> FloatTensor:
        # If this tensor already stores Float64, this can return a materialized copy.
        var out = FloatTensor(self._shape, 0)
        var n = len(self._data)
        var i = 0
        while i < n:
            out._data[i] = self._data[i]  # widen/same as needed
            i += 1
        return out

    fn to_int16(self) -> IntTensor:
        # Requires IntTensor in your library.
        var out = IntTensor(self._shape, 0)
        var n = len(self._data)
        var i = 0
        while i < n:
            # Truncate toward zero; add clamp if you want saturating behavior.
            out._data[i] = Int(self._data[i])
            i += 1
        return out

    fn transpose(self, perm: List[Int]) -> FloatTensor:
        var rank = len(self._shape)
        len(perm) == rank, "transpose: perm length must match rank"

        # normalize and validate perm
        var used = List[Bool]()
        var i = 0
        while i < rank:
            used.append(False)
            i += 1

        var p = List[Int]()
        i = 0
        while i < rank:
            var ax = perm[i]
            if ax < 0:
                ax = ax + rank
            0 <= ax and ax < rank, "transpose: axis out of range"
            not used[ax], "transpose: repeated axis in perm"
            used[ax] = True
            p.append(ax)
            i += 1

        # new shape
        var new_shape = List[Int]()
        i = 0
        while i < rank:
            new_shape.append(self._shape[p[i]])
            i += 1

        # materialized transpose
        var out = FloatTensor(new_shape, 0.0)
        var rm_new = compute_row_major_strides(new_shape)

        # total elements
        var total = 1
        i = 0
        while i < rank:
            total = total * new_shape[i]
            i += 1

        var idx = 0
        while idx < total:
            # decode index in output
            var mi = unravel_index(idx, new_shape, rm_new)

            # map to source multi-index
            var src_mi = List[Int]()
            var j = 0
            while j < rank:
                src_mi.append(0)
                j += 1
            j = 0
            while j < rank:
                src_mi[p[j]] = mi[j]
                j += 1

            var src_lin = ravel_index(src_mi, self._strides)
            out._data[idx] = self._data[src_lin]
            idx += 1

        out._strides = rm_new
        return out


    fn is_contiguous(self) -> Bool:
        var expected = compute_row_major_strides(self._shape)
        var i = 0
        while i < len(expected):
            if self._strides[i] != expected[i]:
                return False
            i += 1
        return True


    # Inside struct FloatTensor
    fn contiguous(self) -> FloatTensor:
        # If already contiguous, just copy fast
        if self.is_contiguous():
            var out_fast = FloatTensor(self._shape, 0)
            var n_fast = len(self._data)
            var j = 0
            while j < n_fast:
                out_fast._data[j] = self._data[j]
                j += 1
            return out_fast

        # Otherwise, materialize into row-major order
        var out = FloatTensor(self._shape, 0)
        var rank = len(self._shape)
        var src_strides = self._strides              # assumes you store strides
        var dst_strides = compute_row_major_strides(self._shape)
        var total = numel_from_shape(self._shape)

        var idx = 0
        while idx < total:
            var mi = unravel_index(idx, self._shape, dst_strides)
            var src_lin = ravel_index(mi, src_strides)
            out._data[idx] = self._data[src_lin]
            idx += 1
        return out

    # --- 2D scalar get ------------------------------------------------------------
    fn get2(self, r: Int, c: Int) -> Float64:
        var rank = len(self._shape)
        rank == 2, "get2: tensor must be 2D"

        var rows = self._shape[0]
        var cols = self._shape[1]

        var rr = r
        if rr < 0: rr = rr + rows
        0 <= rr and rr < rows, "get2: row out of range"

        var cc = c
        if cc < 0: cc = cc + cols
        0 <= cc and cc < cols, "get2: col out of range"

        var lin = rr * self._strides[0] + cc * self._strides[1]
        return self._data[lin]


        # --- Slice along one axis -----------------------------------------------------
    fn slice(self, axis: Int, start: Int, end: Int, step: Int = 1) -> FloatTensor:
        var rank = len(self._shape)
        var ax = axis
        if ax < 0: ax = ax + rank
        0 <= ax and ax < rank, "slice: axis out of range"
        step > 0, "slice: step must be > 0"

        var L = self._shape[ax]

        var s = start
        var e = end
        if s < 0: s = s + L
        if e < 0: e = e + L
        if s < 0: s = 0
        if s > L: s = L
        if e < 0: e = 0
        if e > L: e = L
        if e < s: e = s

        var out_len = (e - s + step - 1) // step

        var new_shape = List[Int]()
        var i = 0
        while i < rank:
            var d = self._shape[i]
            if i == ax: d = out_len
            new_shape.append(d)
            i += 1

        var out = FloatTensor(new_shape, 0.0)
        var rm_dst = compute_row_major_strides(new_shape)
        var total = numel_from_shape(new_shape)

        var idx = 0
        while idx < total:
            var mi = unravel_index(idx, new_shape, rm_dst)

            var src_mi = List[Int]()
            var j = 0
            while j < rank:
                var pos = mi[j]
                if j == ax:
                    pos = s + pos * step
                src_mi.append(pos)
                j += 1
            var src_lin = ravel_index(src_mi, self._strides)
            out._data[idx] = self._data[src_lin]
            idx += 1
        out._strides = rm_dst
        return out


    # --- Elementwise modulo -------------------------------------------------------
    fn mod(self, rhs: Int) -> FloatTensor:
        return self.mod(Float64(rhs))

    fn mod(self, rhs: Float64) -> FloatTensor:
        rhs != 0.0, "mod: division by zero"
        var out = FloatTensor(self._shape, 0.0)
        var n = len(self._data)
        var i = 0
        while i < n:
            var a = self._data[i]
            var q = floor(a / rhs)
            out._data[i] = a - rhs * q
            i += 1
        return out


    # -------------- Equality vs scalar (returns IntTensor mask) ------------------
    fn eq_scalar(self, v: Int, eps: Float64 = 0.0) -> IntTensor:
        return self.eq_scalar(Float64(v), eps)

    fn eq_scalar(self, v: Float64, eps: Float64 = 1e-12) -> IntTensor:
        var out = IntTensor(self._shape, 0)
        var n = len(self._data)
        var i = 0
        while i < n:
            var ok = abs(self._data[i] - v) <= eps
            out._data[i] = 1 if ok else 0
            i += 1
        return out

    # -------------- Boolean select (flattened) -----------------------------------
    # Replace your methods with these versions (no `assert(...)`, English comments only).

    fn boolean_select(self, mask: IntTensor) -> FloatTensor:
        # Require same shape (no broadcasting)
        var rank = len(self._shape)
        rank == len(mask._shape), "boolean_select: rank mismatch"
        var i = 0
        while i < rank:
            self._shape[i] == mask._shape[i], "boolean_select: shape mismatch"
            i += 1

        var n = len(self._data)

        # Count selected
        var k = 0
        i = 0
        while i < n:
            if mask._data[i] != 0:
                k += 1
            i += 1

        var out = FloatTensor([k], 0.0)
        var j = 0
        i = 0
        while i < n:
            if mask._data[i] != 0:
                out._data[j] = self._data[i]
                j += 1
            i += 1
        return out


    fn size(self, dim: Optional[Int] = None) -> Int:
        if dim is None:
            return numel_from_shape(self._shape)
        var ax = dim.value()
        var rank = len(self._shape)
        if ax < 0:
            ax = ax + rank
        0 <= ax and ax < rank, "size(dim): axis out of range"
        return self._shape[ax]


    @staticmethod
    fn from_list_int32(data: List[List[Int]]) -> FloatTensor:
        # Determine shape
        var rows = len(data)
        var cols = 0
        if rows > 0:
            cols = len(data[0])

        # Validate no ragged rows
        var r = 0
        while r < rows:
            len(data[r]) == cols, "from_list_int32: ragged rows not allowed"
            r += 1

        # Materialize into FloatTensor (Float64 storage)
        var out = FloatTensor([rows, cols], 0.0)
        var i = 0
        r = 0
        while r < rows:
            var c = 0
            while c < cols:
                out._data[i] = Float64(data[r][c])
                i += 1
                c += 1
            r += 1
        return out


    # Dimension count
    fn ndim(self) -> Int:
        return len(self._shape)

    # Number of elements (all dims)
    fn numel(self) -> Int:
        return numel_from_shape(self._shape)

   

    # Optional: expose a safe copy of shape
    fn shape(self) -> List[Int]:
        var out = List[Int]()
        var i = 0
        while i < len(self._shape):
            out.append(self._shape[i])
            i += 1
        return out

   
    # Scatter values from `src` into a copy of `self` along `axis`
    # Requirements:
    #   - rank(index) == rank(src) == rank(self)
    #   - For every dim k ≠ axis: index.shape[k] == src.shape[k] == self.shape[k]
    #   - For axis dim: index.shape[axis] == src.shape[axis]
    #   - Each index value is in [-self.shape[axis], self.shape[axis]-1]
   # Paste INSIDE FloatTensor

    fn scatter(self, axis: Int, index: FloatTensor, src: FloatTensor) -> FloatTensor:
        var rank = len(self._shape)
        (len(index._shape) == rank and len(src._shape) == rank), "scatter: rank mismatch"

        var ax = axis
        if ax < 0:
            ax = ax + rank
        (0 <= ax and ax < rank), "scatter: axis out of range"

        var k = 0
        while k < rank:
            if k == ax:
                (index._shape[k] == src._shape[k]), "scatter: index/src size mismatch on axis"
            else:
                (index._shape[k] == self._shape[k]), "scatter: index must match self on non-axis dims"
                (src._shape[k] == self._shape[k]),   "scatter: src must match self on non-axis dims"
            k += 1

        var out = self.clone()

        var idx_strides = compute_row_major_strides(index._shape)
        var total = numel_from_shape(index._shape)
        var size_ax = self._shape[ax]

        var lin = 0
        while lin < total:
            var mi = unravel_index(lin, index._shape, idx_strides)

            var p_val = index._data[lin]
            if p_val < 0.0:
                p_val = p_val + Float64(size_ax)
            var p = Int(p_val)
            (0 <= p and p < size_ax), "scatter: index out of bounds"

            var out_mi = List[Int]()
            var t = 0
            while t < rank:
                if t == ax:
                    out_mi.append(p)
                else:
                    out_mi.append(mi[t])
                t += 1

            var out_lin = ravel_index(out_mi, out._strides)
            out._data[out_lin] = src._data[lin]  # last write wins
            lin += 1
        return out


    # Scatter a single scalar value using an index tensor
    fn scatter_fill(self, axis: Int, index: IntTensor, value: Float64) -> FloatTensor:
        var one = FloatTensor(index._shape, 0)
        var m = len(one._data)
        var j = 0
        while j < m:
            one._data[j] = value
            j += 1
        return self.scatter(axis, index, one)

 

    fn scatter_add(self, axis: Int, index: FloatTensor, src: FloatTensor) -> FloatTensor:
        var rank = len(self._shape)
        (len(index._shape) == rank and len(src._shape) == rank), "scatter_add: rank mismatch"

        var ax = axis
        if ax < 0:
            ax = ax + rank
        (0 <= ax and ax < rank), "scatter_add: axis out of range"

        var k = 0
        while k < rank:
            if k == ax:
                (index._shape[k] == src._shape[k]), "scatter_add: index/src size mismatch on axis"
            else:
                (index._shape[k] == self._shape[k]), "scatter_add: index must match self on non-axis dims"
                (src._shape[k] == self._shape[k]),   "scatter_add: src must match self on non-axis dims"
            k += 1

        var out = self.clone()

        var total = numel_from_shape(index._shape)
        var size_ax = self._shape[ax]

        var lin = 0
        while lin < total:
            # multi-index for current element in index/src
            var mi = unravel_index(lin, index._shape)

            # target coordinate along axis (Float -> Int, supports negative)
            var p_val = index._data[lin]
            if p_val < 0.0:
                p_val = p_val + Float64(size_ax)
            var p = Int(p_val)
            (0 <= p and p < size_ax), "scatter_add: index out of bounds"

            # build destination multi-index
            var dst_idx = List[Int]()
            var t = 0
            while t < rank:
                var pos = mi[t]
                if t == ax:
                    pos = p
                dst_idx.append(pos)
                t += 1

            var dst_lin = ravel_index(dst_idx, out._strides)
            out._data[dst_lin] = out._data[dst_lin] + src._data[lin]
            lin += 1

        return out

    # === Added: read/write helpers for last-dim plane on 3D FloatTensor ===
    fn plane(a: FloatTensor, dim: Int, index: Int) -> FloatTensor:
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
        var out = FloatTensor([B, M], 0)
        var i = 0
        while i < B:
            var j = 0
            while j < M:
                out[i, j] = a.get3(i, j, index)
                j += 1
            i += 1
        return out

    fn write_plane(mut a: FloatTensor, dim: Int, index: Int, rhs: FloatTensor) -> FloatTensor:
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
 
    # -------------- Add a scalar (non-mutating) ---------------------------------
    fn add_scalar(self, v: Int) -> FloatTensor:
        return self.add_scalar(Float64(v))

    fn add_scalar(self, v: Float64) -> FloatTensor:
        var out = FloatTensor(self._shape, 0.0)
        var n = len(self._data)
        var i = 0
        while i < n:
            out._data[i] = self._data[i] + v
            i += 1
        return out

  
     

    # -------------- Take (flat indices) -----------------------------------------
    # Gathers elements using *flat* indices (row-major), returning a 1D tensor.
    # Example:
    #   var flat = x.take(IntTensor.from_list_int32([0, 5, 9])) 


    # take with FloatTensor indices: convert to IntTensor first
    fn take(self, index: FloatTensor) -> FloatTensor:
        return self.take(index.to_int())


    @staticmethod
    fn from_list_float64(data: List[Float64]) -> FloatTensor:
        var shape = [len(data)]
        var out = FloatTensor(shape, 0.0)
        var i = 0
        while i < len(data):
            out._data[i] = data[i]
            i += 1
        return out

    @staticmethod
    fn from_list_int32(data: List[Int]) -> FloatTensor:
        var shape = [len(data)]
        var out = FloatTensor(shape, 0.0)
        var i = 0
        while i < len(data):
            out._data[i] = Float64(data[i])   # cast to float
            i += 1
        return out


 

    # -------------- Flatten (row-major materialization) -------------------------
    # Returns a 1D, row-major-contiguous copy of the tensor.
    fn flatten(self) -> FloatTensor:
        var total = 1
        var i = 0
        while i < len(self._shape):
            total = total * self._shape[i]
            i += 1
        var out = FloatTensor([total], 0.0)

        # Fast-path: data already laid out in row-major order
        var expected = compute_row_major_strides(self._shape)
        var row_major = True
        i = 0
        while i < len(self._shape):
            if self._strides[i] != expected[i]:
                row_major = False
            i += 1
        if row_major:
            i = 0
            while i < total:
                out._data[i] = self._data[i]
                i += 1
            return out

        # General case: respect source strides
        var src_strides = self._strides
        var dst_strides = expected
        var lin = 0
        while lin < total:
            var mi = unravel_index(lin, self._shape, dst_strides)
            var src_lin = ravel_index(mi, src_strides)
            out._data[lin] = self._data[src_lin]
            lin += 1
        return out

    # -------------- Take (flat indices) -----------------------------------------
    # Gathers elements using *flat* indices (row-major), returning a 1D tensor.
    # Example:
    #   var flat = x.take(IntTensor.from_list_int32([0, 5, 9]))
    fn take(self, index: IntTensor) -> FloatTensor:
        # Allow any index rank; we consume its flattened storage order
        var n = len(index._data)
        var out = FloatTensor([n], 0.0)
        var total = 1
        var i = 0
        while i < len(self._shape):
            total = total * self._shape[i]
            i += 1

        var dst_strides = compute_row_major_strides(self._shape)
        var t = 0
        while t < n:
            var lin = index._data[t]
            0 <= lin and lin < total, "take: index out of range"
            if not (0 <= lin and lin < total):
                pass
            var mi = unravel_index(lin, self._shape, dst_strides)
            var src_lin = ravel_index(mi, self._strides)
            out._data[t] = self._data[src_lin]
            t += 1
        return out


 
    fn put(self, index: IntTensor, values: FloatTensor) -> FloatTensor:
        var out = self.clone()
        var n = len(index._data)
        n == values.numel(), "put: values length must match index length"
        if not (n == values.numel()):
            pass
        var total = 1
        var i = 0
        while i < len(self._shape):
            total = total * self._shape[i]
            i += 1
        var rm = compute_row_major_strides(self._shape)
        var t = 0
        while t < n:
            var lin = index._data[t]
            0 <= lin and lin < total, "put: index out of range"
            if not (0 <= lin and lin < total):
                pass
            var mi = unravel_index(lin, self._shape, rm)
            var src_lin = ravel_index(mi, self._strides)
            out._data[src_lin] = values._data[t]
            t += 1
        return out

    # -------------- reshape_like ------------------------------------------------
    fn reshape_like(self, other: FloatTensor) -> FloatTensor:
        return self.view(other._shape)

    # -------------- set2 (2D setter, mutating) ---------------------------------
    fn set2(mut self, r: Int, c: Int, v: Float64):
        len(self._shape) == 2, "set2: tensor must be 2D"
        if not (len(self._shape) == 2):
            return
        0 <= r and r < self._shape[0], "row out of range"
        0 <= c and c < self._shape[1], "col out of range"
        var idx = r * self._strides[0] + c * self._strides[1]
        self._data[idx] = v

    # -------------- as_strided (materialized) ----------------------------------
    fn as_strided(self, shape: List[Int], strides: List[Int]) -> FloatTensor:
        len(shape) == len(strides), "as_strided: rank mismatch"
        var total = 1
        var i = 0
        while i < len(shape):
            total = total * shape[i]
            i += 1
        var out = FloatTensor(shape, 0.0)
        var lin = 0
        while lin < total:
            var mi = unravel_index(lin, shape, compute_row_major_strides(shape))
            var src_lin = 0
            i = 0
            while i < len(shape):
                src_lin += mi[i] * strides[i]
                i += 1
            out._data[lin] = self._data[src_lin]
            lin += 1
        return out

    

 
    fn transpose_axes(self, a: Int, b: Int) -> FloatTensor:
        var dims = List[Int]()
        var i = 0
        while i < len(self._shape):
            dims.append(i)
            i += 1
        var tmp = dims[a]
        dims[a] = dims[b]
        dims[b] = tmp
        return self.permute(dims)

    # -------------- unsqueeze / squeeze_all ------------------------------------
    fn unsqueeze(self, dim: Int) -> FloatTensor:
        var new_shape = List[Int]()
        var i = 0
        while i < len(self._shape):
            if i == dim:
                new_shape.append(1)
            new_shape.append(self._shape[i])
            i += 1
        if dim == len(self._shape):
            new_shape.append(1)
        return self.view(new_shape)

    fn squeeze_all(self) -> FloatTensor:
        var new_shape = List[Int]()
        var i = 0
        while i < len(self._shape):
            if self._shape[i] != 1:
                new_shape.append(self._shape[i])
            i += 1
        if len(new_shape) == 0:
            new_shape.append(1)
        return self.view(new_shape)

    # -------------- flatten_from(start_dim) ------------------------------------
    fn flatten_from(self, start_dim: Int) -> FloatTensor:
        var rank = len(self._shape)
        0 <= start_dim and start_dim < rank, "flatten_from: bad start_dim"
        var pre = List[Int]()
        var mid = 1
        var post = List[Int]()
        var i = 0
        while i < rank:
            if i < start_dim:
                pre.append(self._shape[i])
            elif i == start_dim:
                mid = 1
            else:
                pass
            i += 1
        i = start_dim
        while i < rank:
            mid = mid * self._shape[i]
            i += 1
        var new_shape = List[Int]()
        i = 0
        while i < len(pre):
            new_shape.append(pre[i])
            i += 1
        new_shape.append(mid)
        return self.view(new_shape)

    # -------------- repeat (tile) ----------------------------------------------
    fn repeat(self, reps: List[Int]) -> FloatTensor:
        len(reps) == len(self._shape), "repeat: reps rank mismatch"
        var new_shape = List[Int]()
        var i = 0
        while i < len(reps):
            new_shape.append(self._shape[i] * reps[i])
            i += 1
        var out = FloatTensor(new_shape, 0.0)
        var rm_dst = compute_row_major_strides(new_shape)
        var rm_src = compute_row_major_strides(self._shape)
        var total = out.numel()
        var lin = 0
        while lin < total:
            var md = unravel_index(lin, new_shape, rm_dst)
            # map to source by modulo
            var src = List[Int]()
            i = 0
            while i < len(md):
                var d = md[i] % self._shape[i]
                src.append(d)
                i += 1
            var src_lin = ravel_index(src, self._strides)
            out._data[lin] = self._data[src_lin]
            lin += 1
        return out
 
 
 
    @staticmethod
    fn randn(shape: List[Int]) -> FloatTensor:
        # Simple LCG + Box-Muller (not cryptographic)
        var out = FloatTensor(shape, 0.0)
        var n = out.numel()
        var seed: UInt64 = 0x9E3779B97F4A7C15
        var i = 0
        while i < n:
            # LCG to generate two uniforms in (0,1)
            seed = seed * 2862933555777941757 + 3037000493
            var u1 = Float64((seed >> 11) & 0xFFFFFFFF) / 4294967296.0
            seed = seed * 2862933555777941757 + 3037000493
            var u2 = Float64((seed >> 11) & 0xFFFFFFFF) / 4294967296.0
            var r = sqrt(-2.0 * log(max(u1, 1e-12)))
            var theta = 2.0 * 3.141592653589793 * u2
            out._data[i] = r * cos(theta)
            i += 1
        return out

    # ---- Paste INSIDE FloatTensor ----

    # Boolean-style bitwise ops for FloatTensor masks (nonzero ↔ 1.0)
    fn bit_and(self, other: FloatTensor) -> FloatTensor:
        len(self._shape) == len(other._shape), "bit_and: rank mismatch"
        var out = FloatTensor(self._shape, 0.0)
        var n = len(self._data)
        var i = 0
        while i < n:
            var a = self._data[i]
            var b = other._data[i]
            out._data[i] = 1.0 if (a != 0.0 and b != 0.0) else 0.0
            i += 1
        out._strides = self._strides
        return out

    fn bit_or(self, other: FloatTensor) -> FloatTensor:
        len(self._shape) == len(other._shape), "bit_or: rank mismatch"
        var out = FloatTensor(self._shape, 0.0)
        var n = len(self._data)
        var i = 0
        while i < n:
            var a = self._data[i]
            var b = other._data[i]
            out._data[i] = 1.0 if (a != 0.0 or b != 0.0) else 0.0
            i += 1
        out._strides = self._strides
        return out

    fn bit_xor(self, other: FloatTensor) -> FloatTensor:
        len(self._shape) == len(other._shape), "bit_xor: rank mismatch"
        var out = FloatTensor(self._shape, 0.0)
        var n = len(self._data)
        var i = 0
        while i < n:
            var a = self._data[i]
            var b = other._data[i]
            var ta = (a != 0.0)
            var tb = (b != 0.0)
            out._data[i] = 1.0 if ((ta and not tb) or (tb and not ta)) else 0.0
            i += 1
        out._strides = self._strides
        return out

    # where: condition is IntTensor (0/1); same-shape fast path (row-major)
    fn where(self, condition: IntTensor, other: FloatTensor) -> FloatTensor:
        len(self._shape) == len(other._shape), "where: rank mismatch"
        len(self._shape) == len(condition._shape), "where: rank mismatch"
        var i = 0
        while i < len(self._shape):
            self._shape[i] == other._shape[i], "where: shape mismatch"
            self._shape[i] == condition._shape[i], "where: shape mismatch"
            i += 1
        var out = FloatTensor(self._shape, 0.0)
        var n = len(self._data)
        i = 0
        while i < n:
            out._data[i] = self._data[i] if (condition._data[i] != 0) else other._data[i]
            i += 1
        out._strides = self._strides
        return out

    # min/max: return 1D tensor with a single value
    fn min(self) -> FloatTensor:
        var out = FloatTensor([1], 0.0)
        var n = len(self._data)
        n > 0, "min: empty tensor"
        var m = self._data[0]
        var i = 1
        while i < n:
            if self._data[i] < m:
                m = self._data[i]
            i += 1
        out._data[0] = m
        return out

    fn max(self) -> FloatTensor:
        var out = FloatTensor([1], 0.0)
        var n = len(self._data)
        n > 0, "max: empty tensor"
        var m = self._data[0]
        var i = 1
        while i < n:
            if self._data[i] > m:
                m = self._data[i]
            i += 1
        out._data[0] = m
        return out

    # sum over a dimension, with keepdim
    fn sum_dim_keep(self, axis: Int, keepdim: Bool) -> FloatTensor:
        var rank = len(self._shape)
        var ax = axis
        if ax < 0:
            ax = ax + rank
        0 <= ax and ax < rank, "sum_dim_keep: bad axis"

        # Build output shape
        var out_shape = List[Int]()
        var i = 0
        while i < rank:
            if i == ax:
                if keepdim:
                    out_shape.append(1)
            else:
                out_shape.append(self._shape[i])
            i += 1
        var out = FloatTensor(out_shape, 0.0)

        # Odometer over all dims except ax
        var idx = List[Int]()
        i = 0
        while i < rank:
            idx.append(0)
            i += 1

        # Helper to compute out linear index from idx
        fn out_lin(idx_ref: List[Int]) -> Int:
            var s = 0
            if keepdim:
                var d = 0
                while d < rank:
                    var od = d
                    var coeff = out._strides[od]
                    var val = idx_ref[d] if d != ax else 0
                    s = s + val * coeff
                    d += 1
            else:
                var d2 = 0
                while d2 < rank:
                    if d2 != ax:
                        var od2 = d2
                        if d2 > ax:
                            od2 = d2 - 1
                        s = s + idx_ref[d2] * out._strides[od2]
                    d2 += 1
            return s

        # Iterate
        var done = False
        while not done:
            # sum along axis
            var acc = 0.0
            var sdim = 0
            while sdim < self._shape[ax]:
                idx[ax] = sdim
                # source linear index
                var lin = 0
                i = 0
                while i < rank:
                    lin = lin + idx[i] * self._strides[i]
                    i += 1
                acc = acc + self._data[lin]
                sdim += 1
            # write to out
            var ol = out_lin(idx)
            out._data[ol] = acc

            # increment odometer excluding ax
            var d = rank - 1
            while True:
                if d == ax:
                    d -= 1
                    if d < 0:
                        done = True
                        break
                    else:
                        continue
                idx[d] = idx[d] + 1
                if idx[d] < self._shape[d]:
                    break
                idx[d] = 0
                d -= 1
                if d < 0:
                    done = True
                    break
        return out


    
 

    # to_float32: no-op cast for Float64-backed FloatTensor
    fn to_float32(self) -> FloatTensor:
        return self.clone()

  

    fn std_unbiased(self, unbiased: Bool) -> FloatTensor:
        # Standard deviation over all elements.
        # If unbiased == True  -> denominator = N - 1 (sample std)
        # If unbiased == False -> denominator = N     (population std)
        var n = 1
        var i = 0
        while i < len(self._shape):
            n = n * self._shape[i]
            i += 1

        var out = FloatTensor([1], 0.0)
        if n == 0:
            out._data[0] = 0.0 / 0.0  # NaN for empty
            return out

        # mean
        var s = 0.0
        var k = 0
        while k < n:
            s += self._data[k]
            k += 1
        var mu = s / Float64(n)

        # sum of squared deviations
        var ss = 0.0
        k = 0
        while k < n:
            var d = self._data[k] - mu
            ss += d * d
            k += 1

        if unbiased:
            (n > 1), "std_unbiased: need at least 2 elements for unbiased"
            if n <= 1:
                out._data[0] = 0.0 / 0.0  # NaN
                return out
            out._data[0] = sqrt(ss / Float64(n - 1))
        else:
            out._data[0] = sqrt(ss / Float64(n))

        return out
    

   

  

  

    # any/all on FloatTensor (treat nonzero & not-NaN as True)
    fn any(self) -> Bool:
        var n = len(self._data)
        var i = 0
        while i < n:
            var v = self._data[i]
            if v == v and v != 0.0:  # v==v filters out NaN
                return True
            i += 1
        return False

    fn all(self) -> Bool:
        var n = len(self._data)
        var i = 0
        while i < n:
            var v = self._data[i]
            if not (v == v and v != 0.0):  # any zero or NaN => False
                return False
            i += 1
        return True

    # logical_* on FloatTensor (returns IntTensor mask 0/1)
    fn logical_and(self, other: FloatTensor) -> IntTensor:
        len(self._shape) == len(other._shape), "logical_and: rank mismatch"
        var out = IntTensor(self._shape, 0)
        var n = len(self._data)
        var i = 0
        while i < n:
            var a = self._data[i]
            var b = other._data[i]
            var ta = (a == a and a != 0.0)
            var tb = (b == b and b != 0.0)
            out._data[i] = 1 if (ta and tb) else 0
            i += 1
        out._strides = self._strides
        return out

    fn logical_or(self, other: FloatTensor) -> IntTensor:
        len(self._shape) == len(other._shape), "logical_or: rank mismatch"
        var out = IntTensor(self._shape, 0)
        var n = len(self._data)
        var i = 0
        while i < n:
            var a = self._data[i]
            var b = other._data[i]
            var ta = (a == a and a != 0.0)
            var tb = (b == b and b != 0.0)
            out._data[i] = 1 if (ta or tb) else 0
            i += 1
        out._strides = self._strides
        return out

    fn logical_xor(self, other: FloatTensor) -> IntTensor:
        len(self._shape) == len(other._shape), "logical_xor: rank mismatch"
        var out = IntTensor(self._shape, 0)
        var n = len(self._data)
        var i = 0
        while i < n:
            var a = self._data[i]
            var b = other._data[i]
            var ta = (a == a and a != 0.0)
            var tb = (b == b and b != 0.0)
            out._data[i] = 1 if ((ta and not tb) or (tb and not ta)) else 0
            i += 1
        out._strides = self._strides
        return out

    # where overload accepting FloatTensor condition (nonzero => True)
    fn where(self, condition: FloatTensor, other: FloatTensor) -> FloatTensor:
        len(self._shape) == len(other._shape), "where: rank mismatch"
        len(self._shape) == len(condition._shape), "where: rank mismatch"
        var n = len(self._data)
        var out = FloatTensor(self._shape, 0.0)
        var i = 0
        while i < n:
            var c = condition._data[i]
            var take_self = (c == c and c != 0.0)
            out._data[i] = self._data[i] if take_self else other._data[i]
            i += 1
        out._strides = self._strides
        return out

    # item_f64: require exactly one element
    fn item_f64(self) -> Float64:
        var total = 1
        var i = 0
        while i < len(self._shape):
            total = total * self._shape[i]
            i += 1
        total == 1, "item_f64: tensor must contain exactly one element"
        return self.flatten()._data[0]

    # FloatTensor.nan() -> Float64 NaN literal
    @staticmethod
    fn nan() -> Float64:
        return 0.0 / 0.0  # NaN

    # nanmean along a dimension (drop NaNs). Returns shape with that dim removed.
    fn nanmean_dim(self, axis: Int) -> FloatTensor:
        var rank = len(self._shape)
        var ax = axis
        if ax < 0:
            ax = ax + rank
        0 <= ax and ax < rank, "nanmean_dim: bad axis"
        # build out shape (remove axis)
        var out_shape = List[Int]()
        var i = 0
        while i < rank:
            if i != ax:
                out_shape.append(self._shape[i])
            i += 1
        if len(out_shape) == 0:
            out_shape.append(1)
        var out = FloatTensor(out_shape, 0.0)

        # odometer index
        var idx = List[Int]()
        i = 0
        while i < rank:
            idx.append(0)
            i += 1

        fn out_lin(idx_ref: List[Int]) -> Int:
            var s = 0
            var d = 0
            var od = 0
            while d < rank:
                if d != ax:
                    s = s + idx_ref[d] * out._strides[od]
                    od += 1
                d += 1
            return s

        var done = False
        while not done:
            var sumv = 0.0
            var cnt = 0
            var t = 0
            while t < self._shape[ax]:
                idx[ax] = t
                var lin = 0
                var d2 = 0
                while d2 < rank:
                    lin = lin + idx[d2] * self._strides[d2]
                    d2 += 1
                var v = self._data[lin]
                if v == v:  # not NaN
                    sumv = sumv + v
                    cnt += 1
                t += 1
            var ol = out_lin(idx)
            out._data[ol] = sumv / Float64(cnt) if cnt > 0 else (0.0 / 0.0)  # NaN if empty

            # increment odometer excluding ax
            var d3 = rank - 1
            while True:
                if d3 == ax:
                    d3 -= 1
                    if d3 < 0:
                        done = True
                        break
                    else:
                        continue
                idx[d3] = idx[d3] + 1
                if idx[d3] < self._shape[d3]:
                    break
                idx[d3] = 0
                d3 -= 1
                if d3 < 0:
                    done = True
                    break
        return out


    # --- Paste INSIDE FloatTensor ---

    fn nansum(self) -> FloatTensor:
        # Sum of all elements, ignoring NaNs.
        # If all entries are NaN or tensor is empty -> returns 0.0
        var out = FloatTensor([1], 0.0)
        var s = 0.0
        var n = len(self._data)
        var i = 0
        while i < n:
            var v = self._data[i]
            if v == v:    # true iff not NaN
                s += v
            i += 1
        out._data[0] = s
        return out


    # --- Paste INSIDE FloatTensor ---

    @staticmethod
    fn eye(n: Int) -> FloatTensor:
        var out = FloatTensor([n, n], 0.0)
        var i = 0
        while i < n:
            var idx = i * out._strides[0] + i * out._strides[1]
            out._data[idx] = 1.0
            i += 1
        return out

    # Cholesky (lower-triangular): A = L * L^T ، فرض SPD
    fn cholesky(self) -> FloatTensor:
        len(self._shape) == 2, "cholesky: A must be 2D"
        self._shape[0] == self._shape[1], "cholesky: A must be square"
        var n = self._shape[0]
        var L = FloatTensor([n, n], 0.0)

        var i = 0
        while i < n:
            var j = 0
            while j <= i:
                # sum = A[i,j] - Σ_{k<j} L[i,k]*L[j,k]
                var a_idx = i * self._strides[0] + j * self._strides[1]
                var s = self._data[a_idx]
                var k = 0
                while k < j:
                    var Lik = L._data[i * L._strides[0] + k * L._strides[1]]
                    var Ljk = L._data[j * L._strides[0] + k * L._strides[1]]
                    s = s - Lik * Ljk
                    k += 1
                if i == j:
                    (s > 0.0), "cholesky: matrix not SPD (non-positive pivot)"
                    L._data[i * L._strides[0] + j * L._strides[1]] = sqrt(s)
                else:
                    var Ljj = L._data[j * L._strides[0] + j * L._strides[1]]
                    (Ljj != 0.0), "cholesky: zero diagonal encountered"
                    L._data[i * L._strides[0] + j * L._strides[1]] = s / Ljj
                j += 1
            i += 1

        return L

    
    # --- Paste INSIDE FloatTensor ---
    fn chunk(self, chunks: Int, dim: Int) -> List[FloatTensor]:
        var rank = len(self._shape)
        var ax = dim
        if ax < 0:
            ax = ax + rank
        (0 <= ax and ax < rank), "chunk: bad dim"
        (chunks > 0), "chunk: chunks must be > 0"

        var D = self._shape[ax]
        var base = D // chunks
        var rem = D % chunks

        var parts = List[FloatTensor]()
        var offset = 0
        var ci = 0
        while ci < chunks:
            # size for this chunk
            var size_ci = base
            if ci < rem:
                size_ci = size_ci + 1

            # shape of this piece
            var out_shape = List[Int]()
            var i = 0
            while i < rank:
                var d = self._shape[i]
                if i == ax:
                    d = size_ci
                out_shape.append(d)
                i += 1

            var out = FloatTensor(out_shape, 0.0)

            # copy data if non-empty
            if size_ci > 0:
                var total = numel_from_shape(out_shape)
                var rm_dst = compute_row_major_strides(out_shape)
                var lin = 0
                while lin < total:
                    var md = unravel_index(lin, out_shape, rm_dst)
                    md[ax] = md[ax] + offset
                    var src_lin = ravel_index(md, self._strides)
                    out._data[lin] = self._data[src_lin]
                    lin += 1
                out._strides = rm_dst

            parts.append(out)
            offset = offset + size_ci
            ci += 1

        return parts


    # helper: row-major strides for a given shape
    fn row_major_strides(shape: List[Int]) -> List[Int]:
        var r = len(shape)
        var st = List[Int]()
        var i = 0
        while i < r:
            st.append(0)
            i += 1
        if r == 0:
            return st
        st[r - 1] = 1
        var k = r - 2
        while k >= 0:
            st[k] = st[k + 1] * shape[k + 1]
            k -= 1
        return st

    # helper: unravel using provided row-major strides
    fn unravel(lin: Int, shape: List[Int], rm: List[Int]) -> List[Int]:
        var r = len(shape)
        var idx = List[Int]()
        var i = 0
        while i < r:
            var d = 0
            if rm[i] != 0:
                d = (lin // rm[i]) % (shape[i] if shape[i] != 0 else 1)
            idx.append(d)
            i += 1
        return idx

    # total elements from shape
    fn numel(self,shape: List[Int]) -> Int:
        var t = 1
        var i = 0
        while i < len(shape):
            t = t * shape[i]
            i += 1
        return t
 

    # ===== Paste INSIDE FloatTensor =====

    @staticmethod
    fn empty(shape: List[Int]) -> FloatTensor:
        # Materialize buffer (initialized to 0.0 for safety)
        return FloatTensor(shape, 0.0)

    @staticmethod
    fn rand(shape: List[Int]) -> FloatTensor:
        # Uniform U[0,1)
        var out = FloatTensor(shape, 0.0)
        var n = len(out._data)
        var seed: UInt64 = 0x9E3779B97F4A7C15
        var i = 0
        while i < n:
            seed = seed * 2862933555777941757 + 3037000493
            var u = Float64((seed >> 11) & 0xFFFFFFFF) / 4294967296.0
            out._data[i] = u
            i += 1
        return out

    @staticmethod
    fn normal(mean: Float64, std: Float64, shape: List[Int]) -> FloatTensor:
        # Box-Muller
        var out = FloatTensor(shape, 0.0)
        var n = len(out._data)
        var seed: UInt64 = 0xA24BAED5CF9A13B7
        var i = 0
        while i < n:
            seed = seed * 2862933555777941757 + 3037000493
            var u1 = Float64((seed >> 11) & 0xFFFFFFFF) / 4294967296.0
            seed = seed * 2862933555777941757 + 3037000493
            var u2 = Float64((seed >> 11) & 0xFFFFFFFF) / 4294967296.0
            var r = sqrt(-2.0 * log(max(u1, 1e-12)))
            var theta = 6.283185307179586 * u2
            var z = r * cos(theta)
            out._data[i] = mean + std * z
            i += 1
        return out

    # In-place uniform fill; returns self for chaining
    fn uniform(mut self, low: Float64, high: Float64) -> FloatTensor:
        var n = len(self._data)
        var seed: UInt64 = 0xD1B54A32D192ED03
        var i = 0
        while i < n:
            seed = seed * 2862933555777941757 + 3037000493
            var u = Float64((seed >> 11) & 0xFFFFFFFF) / 4294967296.0
            self._data[i] = low + (high - low) * u
            i += 1
        return self

    # For API parity with your example: uniform_ alias
    fn uniform_(mut self, low: Float64, high: Float64) -> FloatTensor:
        return self.uniform(low, high)

    # Bernoulli: treat tensor values as probabilities p in [0,1]
    fn bernoulli(self) -> IntTensor:
        var out = IntTensor(self._shape, 0)
        var n = len(self._data)
        var seed: UInt64 = 0x94D049BB133111EB
        var i = 0
        while i < n:
            var p = self._data[i]
            if p < 0.0: p = 0.0
            if p > 1.0: p = 1.0
            seed = seed * 2862933555777941757 + 3037000493
            var u = Float64((seed >> 11) & 0xFFFFFFFF) / 4294967296.0
            out._data[i] = 1 if u < p else 0
            i += 1
        out._strides = self._strides
        return out

    # Multinomial from probabilities (1D). If replacement==True uses IID draws.
    fn multinomial(self, num_samples: Int, replacement: Bool) -> IntTensor:
        len(self._shape) == 1, "multinomial: input must be 1D probs"
        var n = self._shape[0]
        num_samples >= 0, "multinomial: num_samples must be >= 0"
        var idxs = IntTensor([num_samples], 0)

        # compute sum
        var s = 0.0
        var i = 0
        while i < n:
            var p = self._data[i]
            s += (p if p > 0.0 else 0.0)
            i += 1
        (s > 0.0), "multinomial: sum of probs must be > 0"

        var seed: UInt64 = 0x2545F4914F6CDD1D

        if replacement:
            var t = 0
            while t < num_samples:
                seed = seed * 2862933555777941757 + 3037000493
                var u = Float64((seed >> 11) & 0xFFFFFFFF) / 4294967296.0
                var c = 0.0
                var j = 0
                while j < n:
                    var pj = self._data[j]
                    if pj > 0.0:
                        c += pj / s
                        if u <= c:
                            idxs._data[t] = j
                            break
                    j += 1
                t += 1
            return idxs
        else:
            # without replacement: simple O(n * k) scheme
            var picked = IntTensor([n], 0)
            var t2 = 0
            while t2 < num_samples:
                # renormalize over unpicked
                var s2 = 0.0
                i = 0
                while i < n:
                    if picked._data[i] == 0:
                        var pi = self._data[i]
                        if pi > 0.0: s2 += pi
                    i += 1
                (s2 > 0.0), "multinomial(no replacement): exhausted mass"
                seed = seed * 2862933555777941757 + 3037000493
                var u2 = Float64((seed >> 11) & 0xFFFFFFFF) / 4294967296.0
                var c2 = 0.0
                var j2 = 0
                while j2 < n:
                    if picked._data[j2] == 0:
                        var pj2 = self._data[j2]
                        if pj2 > 0.0:
                            c2 += pj2 / s2
                            if u2 <= c2:
                                idxs._data[t2] = j2
                                picked._data[j2] = 1
                                break
                    j2 += 1
                t2 += 1
            return idxs

    @staticmethod
    fn randperm(n: Int) -> IntTensor:
        n >= 0, "randperm: n must be >= 0"
        var out = IntTensor([n], 0)
        var i = 0
        while i < n:
            out._data[i] = i
            i += 1
        # Fisher–Yates shuffle
        var seed: UInt64 = 0x123456789ABCDEF
        var k = n - 1
        while k > 0:
            seed = seed * 2862933555777941757 + 3037000493
            var r = Int((seed >> 11) & 0x7FFFFFFF)
            var j = r % (k + 1)
            var tmp = out._data[k]
            out._data[k] = out._data[j]
            out._data[j] = tmp
            k -= 1
        return out


    # --- Paste INSIDE FloatTensor ---
 
    fn one_hot(self, num_classes: Int) -> FloatTensor:
        len(self._shape) == 1, "one_hot(FloatTensor): input must be 1D"
        num_classes > 0, "one_hot: num_classes must be > 0"
        var n = self._shape[0]
        var out = FloatTensor([n, num_classes], 0)
        var i = 0
        while i < n:
            var v = self._data[i]
            if v == v:                         # not NaN
                var cls = Int(v)               # truncate toward zero
                if cls < 0: cls = 0
                if cls >= num_classes: cls = num_classes - 1
                var pos = i * out._strides[0] + cls * out._strides[1]
                out._data[pos] = 1
            i += 1
        return out

    # Select elements where mask != 0 (IntTensor mask). Returns 1D tensor.
    fn masked_select(self, mask: IntTensor) -> FloatTensor:
        len(self._shape) == len(mask._shape), "masked_select: rank mismatch"
        var d = 0
        while d < len(self._shape):
            self._shape[d] == mask._shape[d], "masked_select: shape mismatch"
            d += 1
        var n = len(self._data)
        var cnt = 0
        var i = 0
        while i < n:
            if mask._data[i] != 0: cnt += 1
            i += 1
        var out = FloatTensor([cnt], 0.0)
        var j = 0
        i = 0
        while i < n:
            if mask._data[i] != 0:
                out._data[j] = self._data[i]
                j += 1
            i += 1
        return out

    # (Optional convenience) Float mask version: nonzero & not-NaN treated as True
    fn masked_select(self, mask: FloatTensor) -> FloatTensor:
        len(self._shape) == len(mask._shape), "masked_select: rank mismatch"
        var d = 0
        while d < len(self._shape):
            self._shape[d] == mask._shape[d], "masked_select: shape mismatch"
            d += 1
        var n = len(self._data)
        var cnt = 0
        var i = 0
        while i < n:
            var m = mask._data[i]
            if (m == m and m != 0.0): cnt += 1
            i += 1
        var out = FloatTensor([cnt], 0.0)
        var j = 0
        i = 0
        while i < n:
            var m2 = mask._data[i]
            if (m2 == m2 and m2 != 0.0):
                out._data[j] = self._data[i]
                j += 1
            i += 1
        return out




    # --- Paste INSIDE FloatTensor ---

     
    fn masked_fill(self, mask: IntTensor, value: Float64) -> FloatTensor:
        len(self._shape) == len(mask._shape), "masked_fill: rank mismatch"
        var d = 0
        while d < len(self._shape):
            self._shape[d] == mask._shape[d], "masked_fill: shape mismatch"
            d += 1
        var out = self.clone()
        var n = len(self._data)
        var i = 0
        while i < n:
            if mask._data[i] != 0:
                out._data[i] = value
            i += 1
        return out

  
    fn masked_fill(self, mask: FloatTensor, value: Float64) -> FloatTensor:
        len(self._shape) == len(mask._shape), "masked_fill: rank mismatch"
        var d = 0
        while d < len(self._shape):
            self._shape[d] == mask._shape[d], "masked_fill: shape mismatch"
            d += 1
        var out = self.clone()
        var n = len(self._data)
        var i = 0
        while i < n:
            var m = mask._data[i]
            if (m == m and m != 0.0):
                out._data[i] = value
            i += 1
        return out

     
    fn clamp(self, min_v: Float64, max_v: Float64) -> FloatTensor:
        (min_v <= max_v), "clamp: min_v must be <= max_v"
        var out = FloatTensor(self._shape, 0.0)
        var n = len(self._data)
        var i = 0
        while i < n:
            var v = self._data[i]
            if v < min_v: v = min_v
            if v > max_v: v = max_v
            out._data[i] = v
            i += 1
        out._strides = self._strides
        return out

    # --- Paste INSIDE IntTensor ---

    fn logical_not(self) -> IntTensor:
        # Elementwise NOT: 0 -> 1, nonzero -> 0
        var out = IntTensor(self._shape, 0)
        var n = len(self._data)
        var i = 0
        while i < n:
            out._data[i] = 1 if self._data[i] == 0 else 0
            i += 1
        out._strides = self._strides
        return out



    # ==== Paste INSIDE FloatTensor ====

    @staticmethod
    fn from_list_bool(data: List[Bool]) -> FloatTensor:
        var out = FloatTensor([len(data)], 0.0)
        var i = 0
        while i < len(data):
            out._data[i] = 1.0 if data[i] else 0.0
            i += 1
        return out

    @staticmethod
    fn complex(real: FloatTensor, imag: FloatTensor) -> FloatTensor:
        # Packs (real, imag) into last dim of size 2: shape = real.shape + [2]
        len(real._shape) == len(imag._shape), "complex: rank mismatch"
        var d = 0
        while d < len(real._shape):
            real._shape[d] == imag._shape[d], "complex: shape mismatch"
            d += 1
        var base_shape = real._shape
        var new_shape = List[Int]()
        d = 0
        while d < len(base_shape):
            new_shape.append(base_shape[d])
            d += 1
        new_shape.append(2)
        var out = FloatTensor(new_shape, 0.0)
        # iterate base elements
        var total = 1
        d = 0
        while d < len(base_shape):
            total = total * base_shape[d]
            d += 1
        var rm_base = compute_row_major_strides(base_shape)
        var lin = 0
        while lin < total:
            var mi = unravel_index(lin, base_shape, rm_base)
            var r_lin = ravel_index(mi, real._strides)
            var i_lin = ravel_index(mi, imag._strides)
            out._data[lin * 2 + 0] = real._data[r_lin]
            out._data[lin * 2 + 1] = imag._data[i_lin]
            lin += 1
        return out

    fn logsumexp(self, axis: Int) -> FloatTensor:
        var rank = len(self._shape)
        var ax = axis
        if ax < 0: ax = ax + rank
        0 <= ax and ax < rank, "logsumexp: bad axis"
        # out shape = remove axis
        var out_shape = List[Int]()
        var d = 0
        while d < rank:
            if d != ax: out_shape.append(self._shape[d])
            d += 1
        if len(out_shape) == 0: out_shape.append(1)
        var out = FloatTensor(out_shape, 0.0)

        # odometer excluding ax
        var idx = List[Int]()
        d = 0
        while d < rank:
            idx.append(0)
            d += 1

        fn out_lin(idx_ref: List[Int]) -> Int:
            var s = 0
            var od = 0
            var d = 0
            while d < rank:
                if d != ax:
                    s = s + idx_ref[d] * out._strides[od]
                    od += 1
                d += 1
            return s

        var done = False
        while not done:
            # max along axis for stability
            var m = -1.0/0.0
            var t = 0
            while t < self._shape[ax]:
                idx[ax] = t
                var lin = 0
                d = 0
                while d < rank:
                    lin = lin + idx[d] * self._strides[d]
                    d += 1
                var v = self._data[lin]
                if v > m: m = v
                t += 1
            # sum exp(x - m)
            var s = 0.0
            t = 0
            while t < self._shape[ax]:
                idx[ax] = t
                var lin2 = 0
                d = 0
                while d < rank:
                    lin2 = lin2 + idx[d] * self._strides[d]
                    d += 1
                s = s + exp(self._data[lin2] - m)
                t += 1
            var ol = out_lin(idx)
            out._data[ol] = m + log(s)

            # increment odometer excluding ax
            var dd = rank - 1
            while True:
                if dd == ax:
                    dd -= 1
                    if dd < 0: done = True; break
                    else: continue
                idx[dd] = idx[dd] + 1
                if idx[dd] < self._shape[dd]: break
                idx[dd] = 0
                dd -= 1
                if dd < 0: done = True; break
        return out

    # mean over an axis (no ternary; handles negative axis)
    fn mean_axis(self, axis: Int) -> FloatTensor:
        var rank = len(self._shape)
        var ax = axis
        if ax < 0:
            ax = ax + rank
        0 <= ax and ax < rank, "mean_axis: bad axis"
        var s = self.sum_dim_keep(ax, False)
        var denom = Float64(self._shape[ax])
        return s.div_scalar(denom)


    fn to_list(self) -> List[Float64]:
        # Flat list (row-major)
        var flat = self.flatten()
        var out = List[Float64]()
        var n = len(flat._data)
        var i = 0
        while i < n:
            out.append(flat._data[i])
            i += 1
        return out

    @staticmethod
    fn scalar_f64(v: Float64) -> FloatTensor:
        var out = FloatTensor([1], 0.0)
        out._data[0] = v
        return out

    fn fill_(mut self, v: Float64) -> FloatTensor:
        var n = len(self._data)
        var i = 0
        while i < n:
            self._data[i] = v
            i += 1
        return self

    fn clamp_(mut self, min_opt: Optional[Float64], max_opt: Optional[Float64]) -> FloatTensor:
        var has_min = not (min_opt is None)
        var has_max = not (max_opt is None)
        var mn: Float64 = 0.0
        var mx: Float64 = 0.0
        if has_min: mn = min_opt.value()
        if has_max: mx = max_opt.value()
        (not has_min) or (not has_max) or (mn <= mx), "clamp_: min must be <= max"
        var n = len(self._data)
        var i = 0
        while i < n:
            var v = self._data[i]
            if has_min and v < mn: v = mn
            if has_max and v > mx: v = mx
            self._data[i] = v
            i += 1
        return self

    fn can_broadcast_with(self, other: FloatTensor) -> Bool:
        var a = self._shape
        var b = other._shape
        var ra = len(a)
        var rb = len(b)
        var r = max(ra, rb)
        var i = 0
        while i < r:
            var ad = 1
            var bd = 1
            if i >= r - ra: ad = a[i - (r - ra)]
            if i >= r - rb: bd = b[i - (r - rb)]
            if not (ad == bd or ad == 1 or bd == 1):
                return False
            i += 1
        return True

 

    
    # Complex helpers assuming last dimension of size 2 stores (real, imag)
    fn real(self) -> FloatTensor:
        var rank = len(self._shape)
        rank >= 1, "real(): tensor rank must be >= 1"
        self._shape[rank - 1] == 2, "real(): last dim must be 2 (real, imag)"
        # output shape = all dims except last
        var out_shape = List[Int]()
        var d = 0
        while d < rank - 1:
            out_shape.append(self._shape[d])
            d += 1
        var out = FloatTensor(out_shape, 0.0)

        # odometer over first rank-1 dims
        var idx = List[Int]()
        d = 0
        while d < rank - 1:
            idx.append(0)
            d += 1

        var done = False
        while not done:
            var src_lin = 0
            d = 0
            while d < rank - 1:
                src_lin = src_lin + idx[d] * self._strides[d]
                d += 1
            # pick channel 0 (real)
            src_lin = src_lin + 0 * self._strides[rank - 1]

            var out_lin = 0
            d = 0
            while d < rank - 1:
                out_lin = out_lin + idx[d] * out._strides[d]
                d += 1

            out._data[out_lin] = self._data[src_lin]

            # increment odometer
            var k = rank - 2
            while True:
                if k < 0:
                    done = True
                    break
                idx[k] = idx[k] + 1
                if idx[k] < self._shape[k]:
                    break
                idx[k] = 0
                k = k - 1
        return out

    fn imag(self) -> FloatTensor:
        var rank = len(self._shape)
        rank >= 1, "imag(): tensor rank must be >= 1"
        self._shape[rank - 1] == 2, "imag(): last dim must be 2 (real, imag)"
        # output shape = all dims except last
        var out_shape = List[Int]()
        var d = 0
        while d < rank - 1:
            out_shape.append(self._shape[d])
            d += 1
        var out = FloatTensor(out_shape, 0.0)

        # odometer over first rank-1 dims
        var idx = List[Int]()
        d = 0
        while d < rank - 1:
            idx.append(0)
            d += 1

        var done = False
        while not done:
            var src_lin = 0
            d = 0
            while d < rank - 1:
                src_lin = src_lin + idx[d] * self._strides[d]
                d += 1
            # pick channel 1 (imag)
            src_lin = src_lin + 1 * self._strides[rank - 1]

            var out_lin = 0
            d = 0
            while d < rank - 1:
                out_lin = out_lin + idx[d] * out._strides[d]
                d += 1

            out._data[out_lin] = self._data[src_lin]

            # increment odometer
            var k = rank - 2
            while True:
                if k < 0:
                    done = True
                    break
                idx[k] = idx[k] + 1
                if idx[k] < self._shape[k]:
                    break
                idx[k] = 0
                k = k - 1
        return out


    fn narrow(self, dim: Int, start: Int, length: Int) -> FloatTensor:
        var rank = len(self._shape)
        var ax = dim
        if ax < 0:
            ax = ax + rank
        0 <= ax and ax < rank, "narrow: bad dim"
        0 <= start and start <= self._shape[ax], "narrow: bad start"
        0 <= length and start + length <= self._shape[ax], "narrow: bad length"

        var out_shape = List[Int]()
        var i = 0
        while i < rank:
            out_shape.append(self._shape[i])
            i += 1
        out_shape[ax] = length

        var out = FloatTensor(out_shape, 0.0)
        var total = 1
        i = 0
        while i < len(out_shape):
            total = total * out_shape[i]
            i += 1

        var rm_dst = compute_row_major_strides(out_shape)
        var lin = 0
        while lin < total:
            var md = unravel_index(lin, out_shape, rm_dst)
            md[ax] = md[ax] + start
            var src_lin = ravel_index(md, self._strides)
            out._data[lin] = self._data[src_lin]
            lin += 1
        out._strides = rm_dst
        return out

    fn copy_from(mut self, src: FloatTensor) -> FloatTensor:
        var same = (len(self._shape) == len(src._shape))
        var i = 0
        while same and i < len(self._shape):
            if self._shape[i] != src._shape[i]:
                same = False
            i += 1
        same, "copy_from: shape mismatch"
        var n = len(self._data)
        i = 0
        while i < n:
            self._data[i] = src._data[i]
            i += 1
        return self


    # Use the provided shape (build row-major strides from it)
    fn ravel_index(self, idxs: List[Int], shape: List[Int]) -> Int:
        var rm = compute_row_major_strides(shape)
        return ravel_index(idxs, rm)

    fn unravel_index(self, lin: Int, shape: List[Int]) -> List[Int]:
        var rm = compute_row_major_strides(shape)
        return unravel_index(lin, shape, rm)






# ===== Filesystem adapters (replace bodies with your Mojo file I/O) =====

fn write_text(path: String, text: String) -> Bool:
    """
    Write text to `path`. Returns True on success, False on any failure.
    """
    try:
        var f = open(Path(path), "w")
        f.write(text)
        f.close()
        return True
    except:
        return False

fn read_text(path: String) -> String:
    """
    Read whole text file from `path`. On failure returns empty String.
    """
    try:
        var f = open(Path(path), "r")
        var s = f.read()
        f.close()
        return s
    except:
        return String("")


# ===== String utilities =====

fn split_lines(text: String) -> List[String]:
    var lines = List[String]()
    var cur = ""
    var i = 0
    while i < text.__len__():
        var ch = text[i]
        if ch == '\n':
            lines.append(cur)
            cur = ""
        else:
            cur = cur + String(ch)
        i += 1
    if cur.__len__() > 0:
        lines.append(cur)
    return lines

fn join_csv_row(cells: List[String]) -> String:
    var row = ""
    var i = 0
    while i < len(cells):
        row = row + cells[i]
        if i + 1 < len(cells): row = row + ","
        i += 1
    return row


# ===== MNPY v1 (text) helpers =====

fn mn_header(shape: List[Int]) -> String:
    var s = "MNPY v1\nshape:"
    var i = 0
    while i < len(shape):
        s = s + " " + shape[i].__str__()
        if i + 1 < len(shape): s = s + ","
        i += 1
    s = s + "\n"
    return s

fn parse_mn_header(lines: List[String]) -> (List[Int], Int):
    var shape = List[Int]()
    var idx = 0
    while idx < len(lines):
        var Ls = strip_str(lines[idx])
        if Ls.__len__() == 0:
            idx += 1
            continue
        if Ls == "MNPY v1":
            idx += 1
            continue
        if starts_with(Ls, "shape:"):
            var nums = List[Int]()
            var cur = String("")
            var j = 6
            while j < Ls.__len__() and Ls[j] == ' ':
                j += 1
            while j < Ls.__len__():
                var ch = Ls[j]
                if ch == ',':
                    if strip_str(cur).__len__() > 0:
                        nums.append(to_int_simple(cur))
                    cur = String("")
                else:
                    cur = cur + String(ch)
                j += 1
            if strip_str(cur).__len__() > 0:
                nums.append(to_int_simple(cur))
            shape = nums
            return (shape, idx + 1)
        idx += 1
    return (shape, idx)

fn mn_dump(shape: List[Int], data: List[Float64]) -> String:
    var out = mn_header(shape)
    out = out + "data:\n"
    var i = 0
    while i < len(data):
        out = out + data[i].__str__() + "\n"
        i += 1
    return out

    
fn to_float64_simple(s_in: String) -> Float64:
    var s = strip_str(s_in)
    var n = s.__len__()
    if n == 0:
        return 0.0

    var sign: Float64 = 1.0
    var i = 0
    if n > 0 and char_str(s, 0) == "-":
        sign = -1.0
        i = 1
    elif n > 0 and char_str(s, 0) == "+":
        i = 1

    var acc: Float64 = 0.0
    while i < n and is_digit_str(char_str(s, i)):
        var dv = digit_value_str(char_str(s, i))
        if dv < 0:
            break
        acc = acc * 10.0 + Float64(dv)
        i += 1

    if i < n and char_str(s, i) == ".":
        i += 1
        var frac: Float64 = 0.0
        var base: Float64 = 1.0
        while i < n and is_digit_str(char_str(s, i)):
            var dv2 = digit_value_str(char_str(s, i))
            if dv2 < 0:
                break
            frac = frac * 10.0 + Float64(dv2)
            base = base * 10.0
            i += 1
        acc = acc + frac / base

    if i < n and (char_str(s, i) == "e" or char_str(s, i) == "E"):
        i += 1
        var esign: Int = 1
        if i < n and char_str(s, i) == "-":
            esign = -1
            i += 1
        elif i < n and char_str(s, i) == "+":
            i += 1
        var exp: Int = 0
        while i < n and is_digit_str(char_str(s, i)):
            var dv3 = digit_value_str(char_str(s, i))
            if dv3 < 0:
                break
            exp = exp * 10 + dv3
            i += 1
        var e: Int = esign * exp
        var factor: Float64 = 1.0
        if e > 0:
            var j = 0
            while j < e:
                factor = factor * 10.0
                j += 1
        elif e < 0:
            var j2 = 0
            while j2 < -e:
                factor = factor / 10.0
                j2 += 1
        acc = acc * factor

    return sign * acc


   


fn mn_parse(text: String) -> (List[Int], List[Float64]):
    var lines = split_lines(text)
    var (shape, start_idx) = parse_mn_header(lines)
    var k = start_idx
    while k < len(lines) and strip_str(lines[k]).__len__() == 0:
        k += 1
    if k < len(lines) and strip_str(lines[k]).startswith("data"):
        k += 1

    var vals = List[Float64]()
    while k < len(lines):
        var sline = strip_str(lines[k])
        if sline.__len__() > 0:
            vals.append(to_float64_simple(sline))
        k += 1

    if len(shape) == 0:
        shape = [len(vals)]
    return (shape, vals)



# ===== CSV helpers =====

fn to_csv_lines(shape: List[Int], data: List[Float64], header: List[String]) -> List[String]:
    var lines = List[String]()
    if len(header) > 0:
        lines.append(join_csv_row(header))

    var rank = len(shape)
    if rank == 0:
        if len(data) > 0:
            lines.append(data[0].__str__())
        else:
            lines.append("")
        return lines

    if rank == 1:
        var i2 = 0
        while i2 < len(data):
            lines.append(data[i2].__str__())
            i2 += 1
        return lines

    if rank == 2:
        var rows = shape[0]
        var cols = shape[1]
        var base = 0
        var r = 0
        while r < rows:
            var row_cells = List[String]()
            var c = 0
            while c < cols:
                row_cells.append(data[base + c].__str__())
                c += 1
            lines.append(join_csv_row(row_cells))
            base += cols
            r += 1
        return lines

    var last = shape[rank - 1]
    var rows_flat = 1
    var k = 0
    while k < rank - 1:
        rows_flat = rows_flat * shape[k]
        k += 1
    var base2 = 0
    var r2 = 0
    while r2 < rows_flat:
        var row_cells2 = List[String]()
        var c2 = 0
        while c2 < last:
            row_cells2.append(data[base2 + c2].__str__())
            c2 += 1
        lines.append(join_csv_row(row_cells2))
        base2 += last
        r2 += 1
    return lines

  

fn csv_parse(text: String, skiprows: Int) -> (Int, Int, List[Float64]):
    var lines = split_lines(text)
    var rows = List[List[Float64]]()
    var r = 0
    var seen = 0
    while r < len(lines):
        var L = strip_str(lines[r])
        r += 1
        if L.__len__() == 0:
            continue
        if seen < skiprows:
            seen += 1
            continue
        var vals = List[Float64]()
        var tok = String("")
        var j = 0
        while j < L.__len__():
            var ch = L[j]
            if ch == ',':
                var t = strip_str(tok)
                if t.__len__() > 0:
                    vals.append(to_float64_simple(t))
                else:
                    vals.append(0.0)
                tok = String("")
            else:
                tok = tok + String(ch)
            j += 1
        var tlast = strip_str(tok)
        if tlast.__len__() > 0:
            vals.append(to_float64_simple(tlast))
        if len(vals) > 0:
            rows.append(vals)
    var nrows = len(rows)
    if nrows == 0:
        return (0, 0, List[Float64]())
    var ncols = len(rows[0])
    var rr = 0
    while rr < nrows:
        var need = ncols - len(rows[rr])
        var k2 = 0
        while k2 < need:
            rows[rr].append(0.0)
            k2 += 1
        rr += 1
    var flat = List[Float64]()
    var rr2 = 0
    while rr2 < nrows:
        var cc = 0
        while cc < ncols:
            flat.append(rows[rr2][cc])
            cc += 1
        rr2 += 1
    return (nrows, ncols, flat)


fn is_space(ch: Character) -> Bool:
    return ch == ' ' or ch == '\t' or ch == '\n' or ch == '\r'
 

 

fn char_str(s: String, i: Int) -> String:
    var n = s.__len__()
    if i < 0 or i >= n:
        return String("")
    return String(s[i])

fn is_digit_str(ch: String) -> Bool:
    return (ch == "0") or (ch == "1") or (ch == "2") or (ch == "3") or (ch == "4") or
           (ch == "5") or (ch == "6") or (ch == "7") or (ch == "8") or (ch == "9")

fn digit_value_str(ch: String) -> Int:
    if ch == "0": return 0
    if ch == "1": return 1
    if ch == "2": return 2
    if ch == "3": return 3
    if ch == "4": return 4
    if ch == "5": return 5
    if ch == "6": return 6
    if ch == "7": return 7
    if ch == "8": return 8
    if ch == "9": return 9
    return -1

fn strip_str(s: String) -> String:
    var n = s.__len__()
    if n == 0:
        return s
    var i = 0
    # left trim
    while i < n and (char_str(s, i) == " " or char_str(s, i) == "\t" or char_str(s, i) == "\n" or char_str(s, i) == "\r"):
        i += 1
    # right trim
    var j = n - 1
    while j >= i and (char_str(s, j) == " " or char_str(s, j) == "\t" or char_str(s, j) == "\n" or char_str(s, j) == "\r"):
        j -= 1
    var out = String("")
    var k = i
    while k <= j:
        out = out + String(s[k])
        k += 1
    return out

fn to_float_simple(s_in: String) -> Float64:
    var s = strip_str(s_in)
    var n = s.__len__()
    if n == 0:
        return 0.0

    var sign: Float64 = 1.0
    var i = 0
    if char_str(s, 0) == "-":
        sign = -1.0
        i = 1
    elif char_str(s, 0) == "+":
        i = 1

    var int_part: Float64 = 0.0
    while i < n and is_digit_str(char_str(s, i)):
        var dv = digit_value_str(char_str(s, i))
        if dv < 0:
            break
        int_part = int_part * 10.0 + Float64(dv)
        i += 1

    var frac_part: Float64 = 0.0
    var scale: Float64 = 1.0
    if i < n and char_str(s, i) == ".":
        i += 1
        while i < n and is_digit_str(char_str(s, i)):
            var dv2 = digit_value_str(char_str(s, i))
            if dv2 < 0:
                break
            frac_part = frac_part * 10.0 + Float64(dv2)
            scale = scale * 10.0
            i += 1

    return sign * (int_part + frac_part / scale) 
 
fn starts_with(s: String, prefix: String) -> Bool:
    var n = s.__len__()
    var m = prefix.__len__()
    if m > n:
        return False
    var i = 0
    while i < m:
        if s[i] != prefix[i]:
            return False
        i += 1
    return True

fn to_int_simple(s_in: String) -> Int:
    var s = strip_str(s_in)
    var n = s.__len__()
    if n == 0:
        return 0
    var sign = 1
    var i = 0
    if char_str(s, 0) == "-":
        sign = -1
        i = 1
    elif char_str(s, 0) == "+":
        i = 1
    var acc: Int = 0
    while i < n and is_digit_str(char_str(s, i)):
        var dv = digit_value_str(char_str(s, i))
        if dv < 0:
            break
        acc = acc * 10 + dv
        i += 1
    return sign * acc


fn compute_row_major_strides(shape: List[Int]) -> List[Int]:
    var r = len(shape)
    var st = List[Int]()
    var i = 0
    while i < r:
        st.append(0)
        i += 1
    if r == 0:
        return st
    st[r - 1] = 1
    var k = r - 2
    while k >= 0:
        st[k] = st[k + 1] * shape[k + 1]
        k -= 1
    return st

 

fn unravel_index(lin: Int, shape: List[Int]) -> List[Int]:
    return unravel_index(lin, shape, compute_row_major_strides(shape))
 
fn unravel_index(lin: Int, shape: List[Int], rm: List[Int]) -> List[Int]:
        # Uses provided row-major strides `rm`.
        var r = len(shape)
        var idx = List[Int]()
        var i = 0
        while i < r:
            idx.append(0)
            i += 1
        i = 0
        while i < r:
            var stride = rm[i]
            var d = 0
            if stride != 0 and shape[i] != 0:
                d = (lin // stride) % shape[i]
            idx[i] = d
            i += 1
        return idx


fn ravel_index( idxs: List[Int], shape: List[Int]) -> Int:
    var flat = 0
    var i = 0
    while i < len(shape):
        flat = flat * shape[i] + idxs[i]
        i += 1
    return flat

fn gather_int(a: FloatTensor, axis: Int, indices: IntTensor) -> FloatTensor:
    var shp = a._shape
    var rank = len(shp)

    if rank == 1:
        var K = indices._shape[0]
        var out = FloatTensor([K], 0.0)
        var i = 0
        while i < K:
            var idx = indices._data[i]
            0 <= idx and idx < shp[0], "gather(1D): index out of range"
            out._data[i] = a._data[idx * a._strides[0]]
            i += 1
        return out

    elif rank == 2:
        0 <= axis and axis < 2, "gather(2D): axis must be 0 or 1"
        var R = shp[0]
        var C = shp[1]
        var K = indices._shape[0]

        if axis == 0:
            var out = FloatTensor([K, C], 0.0)
            var i = 0
            while i < K:
                var r = indices._data[i]
                0 <= r and r < R, "gather(rows): index out of range"
                var j = 0
                while j < C:
                    var dst = i * out._strides[0] + j * out._strides[1]
                    var src = r * a._strides[0] + j * a._strides[1]
                    out._data[dst] = a._data[src]
                    j += 1
                i += 1
            return out
        else:
            var out = FloatTensor([R, K], 0.0)
            var j2 = 0
            while j2 < K:
                var c = indices._data[j2]
                0 <= c and c < C, "gather(cols): index out of range"
                var i = 0
                while i < R:
                    var dst = i * out._strides[0] + j2 * out._strides[1]
                    var src = i * a._strides[0] + c * a._strides[1]
                    out._data[dst] = a._data[src]
                    i += 1
                j2 += 1
            return out

    else:
        0 == 1, "gather: only 1D or 2D tensors supported"
        return FloatTensor([0], 0.0)

# module-level helper
fn numel_from_shape(shape: List[Int]) -> Int:
    var t = 1
    var i = 0
    while i < len(shape):
        t = t * shape[i]
        i += 1
    return t


# module-level helper
fn floor(x: Float64) -> Float64:
    var i = Int(x)
    if Float64(i) > x:
        i = i - 1
    return Float64(i)
