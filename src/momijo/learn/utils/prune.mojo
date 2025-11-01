# MIT License
# SPDX-License-Identifier: MIT
# Project:      Momijo
# Module:       learn.prune.l1_unstructured
# File:         src/momijo/learn/prune/l1_unstructured.mojo
#
# Description:
#   L1 unstructured pruning utilities for Momijo Learn.
#   - PruneState: stores and applies binary masks to parameters (Linear.weight).
#   - tensor_sparsity: convenience metric (mean of zeros).
#
# Notes:
#   - English-only comments. No globals. var-only. Imports are explicit.
#   - This implementation targets Linear for now; extend as needed (e.g., Conv2d).

from collections.list import List
from momijo.tensor import tensor
from momijo.learn.nn.layers import Linear

# ----------------------------- Pruning state ----------------------------------
struct PruneState(Copyable, Movable):
    var has_mask: Bool
    var name: String
    var mask: tensor.Tensor[Float64]  # same shape as parameter

    fn __init__(out self):
        self.has_mask = False
        self.name = String("")
        self.mask = tensor.zeros([1])  # placeholder 1-element tensor

    fn _abs_values(self, w: tensor.Tensor[Float64]) -> tensor.Tensor[Float64]:
        # Return |w| flattened as [N]
        var shape = w.shape()
        var n = 1
        var d = 0
        while d < len(shape):
            n = n * shape[d]
            d = d + 1
        var out = tensor.zeros([n])
        var i = 0
        while i < n:
            var v = w._data[i]
            out._data[i] = v if v >= 0.0 else -v
            i = i + 1
        return out.copy()
 

    fn _kth_value(self, a: tensor.Tensor[Float64], k: Int) -> Float64:
        # Return the k-th smallest value (0-based) from 1D tensor a.
        var n = a.shape()[0]

        # Empty-guard (demo default)
        if n <= 0:
            return 0.0

        # Copy to a simple list for sorting
        var vals = List[Float64]()
        var i = 0
        while i < n:
            vals.append(a._data[i])
            i = i + 1

        # Insertion sort (sufficient for small demos)
        var j = 1
        while j < n:
            var key = vals[j]
            var m = j - 1
            while m >= 0 and vals[m] > key:
                vals[m + 1] = vals[m]
                m = m - 1
            vals[m + 1] = key
            j = j + 1

        # Clamp k into [0, n-1] and return once at the end
        var kk = k
        if kk < 0: kk = 0
        if kk >= n: kk = n - 1
        return vals[kk]

        

    fn _make_mask(self, w: tensor.Tensor[Float64], amount: Float64) -> tensor.Tensor[Float64]:
        # Build binary mask (1 keep, 0 prune) by thresholding |w| at the given sparsity.
        var absw = self._abs_values(w)           # [N]
        var n = absw.shape()[0]
        var n_prune = Int(amount * Float64(n))
        if n_prune < 0: n_prune = 0
        if n_prune > n: n_prune = n
        var thr = self._kth_value(absw, n_prune - 1) if n_prune > 0 else -1.0
        # If n_prune == 0, threshold is -1 so nothing is pruned.
        var mask = tensor.zeros(w.shape())       # same shape as w
        var i = 0
        while i < n:
            var keep = 1.0
            var av = absw._data[i]
            # prune values strictly below threshold; keep >= thr
            if n_prune > 0 and av < thr:
                keep = 0.0
            mask._data[i] = keep
            i = i + 1
        return mask.copy()

    # l1_unstructured: no assert, safe guards + early return
    fn l1_unstructured(mut self, mut lin: Linear, name: String, amount: Float64):
        # Support only "weight" for now; ignore others safely.
        if not (name == String("weight")):
            return

        # Clip amount to [0,1]
        var a = amount
        if a < 0.0: a = 0.0
        if a > 1.0: a = 1.0

        var w = lin.weight.copy()
        var mask = self._make_mask(w, a)

        # Apply in-place: w *= mask
        var n = w.shape()[0] * w.shape()[1]
        var i = 0
        while i < n:
            w._data[i] = w._data[i] * mask._data[i]
            i = i + 1

        lin.weight = w.copy()
        self.has_mask = True
        self.name = name
        self.mask = mask.copy()

    # remove: no assert, safe early return if name unsupported
    fn remove(mut self, mut lin: Linear, name: String):
        if not (name == String("weight")):
            return

        # No further action: weight already has mask applied permanently.
        self.has_mask = False
        self.name = String("")
        self.mask = tensor.zeros([1])


# ----------------------------- Utilities --------------------------------------
fn tensor_sparsity(x: tensor.Tensor[Float64]) -> Float64:
    # Mean of (x == 0). Avoid equality on floats in production; in demos it's OK.
    var n = 1
    var shape = x.shape()
    var d = 0
    while d < len(shape):
        n = n * shape[d]
        d = d + 1
    var zero = 0.0
    var i = 0
    while i < n:
        if x._data[i] == 0.0:
            zero = zero + 1.0
        i = i + 1
    return zero / Float64(n)
