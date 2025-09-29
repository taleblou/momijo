# MIT License
# Copyright (c) 2025 Morteza Talebou and Mitra Daneshmand
# Project: momijo/tensor
# File: src/momijo/tensor/printing.mojo

from momijo.tensor.tensor import Tensor
from momijo.tensor.tensorint import IntTensor
from momijo.tensor.tensorfloat import FloatTensor
from collections.dict import Dict
from collections.list import List
 

fn to_string[T](x: T) -> String:
    return String(x)


fn to_string_list[T](xs: List[T]) -> String:
    var s = "["
    var i = 0
    while i < xs.len():
        s += to_string(xs[i])
        if i < xs.len() - 1:
            s += ", "
        i += 1
    s += "]"
    return s


fn to_string_dict[K, V](d: Dict[K, V]) -> String:
    var s = "{"
    var keys = d.keys()
    var i = 0
    while i < keys.len():
        var k = keys[i]
        var v = d.get(k).value()
        s += to_string(k) + ": " + to_string(v)
        if i < keys.len() - 1:
            s += ", "
        i += 1
    s += "}"
    return s


fn to_string_tensor(t: Tensor) -> String:
    return "Tensor(shape=" + to_string_list(t.shape) + ", data=...)"


fn to_string_inttensor(t: IntTensor) -> String:
    return "IntTensor(shape=" + to_string_list(t.shape()) + ", data=" + to_string_list(t.to_list()) + ")"


fn to_string_floattensor(t: FloatTensor) -> String:
    return "FloatTensor(shape=" + to_string_list(t.shape()) + ", data=" + to_string_list(t.to_list()) + ")"
