# Project:      Momijo
# Module:       src.momijo.tensor.utils
# File:         utils.mojo
# Path:         src/momijo/tensor/utils.mojo
#
# Description:  Shape/stride helpers.
# License:      MIT

fn shape_product(shape: List[Int]) -> Int:
    var s: Int = 1
    var i = 0
    while i < len(shape):
        s = s * shape[i]
        i += 1
    return s

fn compute_strides(shape: List[Int]) -> List[Int]:
    var n = len(shape)
    var out = List[Int]()
    var i = 0
    while i < n:
        out.append(0)
        i += 1

    var acc: Int = 1
    var j = n - 1
    while j >= 0:
        out[j] = acc
        acc = acc * shape[j]
        j -= 1
    return out



# --- Flatten helpers for nested int lists ---
fn _flatten_2d_i32(data: List[List[Int]]) -> (Int, Int, List[Int]):
    var rows = len(data)
    var cols = 0
    if rows > 0: cols = len(data[0])
    var flat = List[Int]()
    var r = 0
    while r < rows:
        assert len(data[r]) == cols
        var c = 0
        while c < cols:
            flat.append(data[r][c])
            c += 1
        r += 1
    return (rows, cols, flat)

fn _flatten_3d_i32(data: List[List[List[Int]]]) -> (Int, Int, Int, List[Int]):
    var d0 = len(data)
    var d1 = 0
    var d2 = 0
    if d0 > 0:
        d1 = len(data[0])
        if d1 > 0:
            d2 = len(data[0][0])
    var flat = List[Int]()
    var i = 0
    while i < d0:
        assert len(data[i]) == d1
        var j = 0
        while j < d1:
            assert len(data[i][j]) == d2
            var k = 0
            while k < d2:
                flat.append(data[i][j][k])
                k += 1
            j += 1
        i += 1
    return (d0, d1, d2, flat)

fn _flatten_4d_i32(data: List[List[List[List[Int]]]]) -> (Int, Int, Int, Int, List[Int]):
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
    var flat = List[Int]()
    var i = 0
    while i < d0:
        assert len(data[i]) == d1
        var j = 0
        while j < d1:
            assert len(data[i][j]) == d2
            var k = 0
            while k < d2:
                assert len(data[i][j][k]) == d3
                var l = 0
                while l < d3:
                    flat.append(data[i][j][k][l])
                    l += 1
                k += 1
            j += 1
        i += 1
    return (d0, d1, d2, d3, flat)
