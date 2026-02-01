# MIT License
# SPDX-License-Identifier: MIT
# Project:      Momijo
# Module:       momijo.learn.data.types.pair
# File:         src/momijo/learn/data/types/pair.mojo
#
# Description:
#   Generic (x, y) sample for image datasets:
#     - x: Tensor[Float32] in [C, H, W] layout, normalized by caller's pipeline
#     - y: class index (Int)

from momijo.tensor import tensor
from collections.list import List

struct Pair(Copyable, Movable):
    var x: tensor.Tensor[Float32]  # [C,H,W]
    var y: Int                     # class index

    fn __init__(out self, x: tensor.Tensor[Float32], y: Int):
        self.x = x.copy()
        self.y = y

    fn __copyinit__(out self, other: Self):
        self.x = other.x.copy()
        self.y = other.y

    @staticmethod
    fn empty() -> Self:
        # Minimal CHW tensor to preserve dimensionality guarantees
        var z = tensor.zeros([1, 1, 1])
        return Pair(z, 0)

    @staticmethod
    fn from_xy(x: tensor.Tensor[Float32], y: Int) -> Self:
        return Pair(x, y)

    fn shape(self) -> List[Int]:
        return self.x.shape()

    fn channels(self) -> Int:
        var shp = self.x.shape()
        return (shp[0] if len(shp) > 0 else 0)

    fn height(self) -> Int:
        var shp = self.x.shape()
        return (shp[1] if len(shp) > 1 else 0)

    fn width(self) -> Int:
        var shp = self.x.shape()
        return (shp[2] if len(shp) > 2 else 0)

    fn is_valid(self) -> Bool:
        var shp = self.x.shape()
        return (len(shp) == 3 and shp[0] > 0 and shp[1] > 0 and shp[2] > 0)

    fn __str__(self) -> String:
        return String("Pair(shape=") + self.x.shape().__str__() + String(", y=") + String(self.y) + String(")")
