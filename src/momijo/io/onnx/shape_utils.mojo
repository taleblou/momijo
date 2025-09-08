# Project:      Momijo
# Module:       src.momijo.io.onnx.shape_utils
# File:         shape_utils.mojo
# Path:         src/momijo/io/onnx/shape_utils.mojo
#
# Description:  Filesystem/IO helpers with Path-centric APIs and safe resource
#               management (binary/text modes and encoding clarity).
#
# Author(s):    Morteza Taleblou & Mitra Daneshmand
# Website:      https://taleblou.ir/
# Repository:   https://github.com/taleblou/momijo
#
# License:      MIT License
# SPDX-License-Identifier: MIT
# Copyright:    (c) 2025 Morteza Taleblou & Mitra Daneshmand
#
# Notes:
#   - Structs: ShapeInfo
#   - Key functions: __init__, __copyinit__, __moveinit__, _broadcast_shapes, infer_add_shape, infer_matmul_shape, infer_relu_shape, infer_reshape_shape ...
#   - Uses generic functions/types with explicit trait bounds.


struct ShapeInfo:
    var shape: List[Int]
fn __init__(out self, shape: List[Int]) -> None:
        self.shape = shape
fn __copyinit__(out self, other: Self) -> None:
        self.shape = other.shape
fn __moveinit__(out self, deinit other: Self) -> None:
        self.shape = other.shape
# -----------------------------------------------------------------------------
# Broadcasting utility
# -----------------------------------------------------------------------------
fn _broadcast_shapes(a: List[Int], b: List[Int]) -> List[Int]:
    var out_shape = List[Int]()
    var la = len(a)
    var lb = len(b)
    var l = max(la, lb)
    for i in range(l):
        var da = a[la-1-i] if i < la else 1
        var db = b[lb-1-i] if i < lb else 1
        if da == db or da == 1 or db == 1:
            out_shape.insert(0, max(da, db))
        else:
            raise ValueError("Shapes not broadcastable: " + String(a) + " vs " + String(b))
    return out_shape

# -----------------------------------------------------------------------------
# Shape inference for common ops
# -----------------------------------------------------------------------------
fn infer_add_shape(a: ShapeInfo, b: ShapeInfo) -> ShapeInfo:
    return ShapeInfo(_broadcast_shapes(a.shape, b.shape))
fn infer_matmul_shape(a: ShapeInfo, b: ShapeInfo) -> ShapeInfo:
    if len(a.shape) != 2 or len(b.shape) != 2:
        raise ValueError("MatMul requires 2D tensors")
    if a.shape[1] != b.shape[0]:
        raise ValueError("MatMul shapes mismatch")
    return ShapeInfo([a.shape[0], b.shape[1]])
fn infer_relu_shape(a: ShapeInfo) -> ShapeInfo:
    return ShapeInfo(a.shape)
fn infer_reshape_shape(a: ShapeInfo, new_shape: List[Int]) -> ShapeInfo:
    var prod_old = 1
    for d in a.shape:
        prod_old *= d
    var prod_new = 1
    var infer_dim = -1
    for (i,d) in enumerate(new_shape):
        if d == -1:
            if infer_dim != -1:
                raise ValueError("Only one dimension can be inferred")
            infer_dim = i
        else:
            prod_new *= d
    if infer_dim != -1:
        if prod_old % prod_new != 0:
            raise ValueError("Inferred shape not divisible")
        new_shape[infer_dim] = prod_old // prod_new
    else:
        if prod_new != prod_old:
            raise ValueError("Reshape size mismatch")
    return ShapeInfo(new_shape)

# -----------------------------------------------------------------------------
# Generic shape inference
# -----------------------------------------------------------------------------
fn infer_shape(op: String, inputs: List[ShapeInfo], attrs: Dict[String, Any] = {}) -> ShapeInfo:
    if op == "Add":
        return infer_add_shape(inputs[0], inputs[1])
    elif op == "MatMul":
        return infer_matmul_shape(inputs[0], inputs[1])
    elif op == "Relu":
        return infer_relu_shape(inputs[0])
    elif op == "Reshape":
        return infer_reshape_shape(inputs[0], attrs["shape"])
    else:
        raise NotImplementedError("Shape inference not implemented for op: " + op)