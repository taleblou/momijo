# MIT License
# SPDX-License-Identifier: MIT
# Project:      Momijo
# Module:       momijo.utils.print
# File:         src/momijo/utils/print.mojo
#
# Description:
#   Safe, generic string helpers for Lists, Dicts, and Tensors.

from collections.list import List
from collections.dict import Dict
from momijo.tensor.tensor import Tensor

# ------------------------------ Generic stringify ------------------------------

@always_inline
fn ravel[T](x: T) -> String:
    # Relies on T.__str__ being available; String(x) triggers it.
    return String(x)

# ------------------------------ List -> String --------------------------------

fn to_string_list[T](xs: List[T]) -> String:
    var s = String("[")
    var i = 0
    var n = len(xs)
    while i < n:
        s = s + ravel[T](xs[i])
        if i < n - 1:
            s = s + ", "
        i += 1
    s = s + "]"
    return s

# ------------------------------ Dict -> String --------------------------------

fn to_string_dict[K, V](d: Dict[K, V]) -> String:
    var s = String("{")
    var keys = d.keys()
    var n = len(keys)
    var i = 0
    while i < n:
        var k = keys[i]
        # Assuming key exists; value() access per project Optional policy
        var v = d.get(k).value()
        s = s + ravel[K](k) + ": " + ravel[V](v)
        if i < n - 1:
            s = s + ", "
        i += 1
    s = s + "}"
    return s

# ------------------------------ Tensor -> String -------------------------------

# Generic Tensor pretty-short string
fn to_string_tensor[T](t: Tensor[T]) -> String:
    return "Tensor(shape=" + to_string_list[Int](t.shape()) + ", data=...)"

# Optional: IntTensor pretty (if your codebase defines IntTensor type alias/struct)
# If IntTensor is an alias for Tensor[Int], this overload will work. Otherwise, remove it.
fn to_string_inttensor(t: Tensor[Int]) -> String:
    return "IntTensor(shape=" + to_string_list[Int](t.shape()) + \
           ", data=" + to_string_list[Int](t.to_list()) + ")"
