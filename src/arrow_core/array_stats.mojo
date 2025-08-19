
# Momijo Arrow Core
# This file is part of the Momijo project. See the LICENSE file at the repository root.

from momijo.arrow_core.array import Array

fn sum_int(a: Array[Int]) -> Int:
    var s = 0
    var i = 0
    while i < a.len():
        s += a.get(i)
        i += 1
    return s

fn min_int(a: Array[Int]) -> Int raises:
    if a.len() == 0: raise Exception("min on empty array")
    var m = a.get(0)
    var i = 1
    while i < a.len():
        if a.get(i) < m: m = a.get(i)
        i += 1
    return m

fn max_int(a: Array[Int]) -> Int raises:
    if a.len() == 0: raise Exception("max on empty array")
    var m = a.get(0)
    var i = 1
    while i < a.len():
        if a.get(i) > m: m = a.get(i)
        i += 1
    return m

fn sum_f64(a: Array[Float64]) -> Float64:
    var s = 0.0
    var i = 0
    while i < a.len():
        s += a.get(i)
        i += 1
    return s

fn mean_f64(a: Array[Float64]) -> Float64 raises:
    if a.len() == 0: raise Exception("mean on empty array")
    return sum_f64(a) / Float64(a.len())
