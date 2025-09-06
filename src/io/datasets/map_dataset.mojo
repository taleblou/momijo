# ============================================================================
# MIT License
# Copyright (c) 2025 Morteza Talebou and Mitra Daneshmand
# Project: momijo  |  Source: https://github.com/taleblou/momijo
# This file is part of the Momijo project. See the LICENSE file at the repository root.
# SPDX-License-Identifier: MIT
# Website: https://taleblou.ir/
# Repository: https://github.com/taleblou/momijo
#
# Project: momijo.io.datasets
# File: src/momijo/io/datasets/map_dataset.mojo
# ============================================================================

# -----------------------------------------------------------------------------
# MapDataset: indexable dataset
# -----------------------------------------------------------------------------
struct MapDataset:
    var data: List[Any]
    var transform: Optional[fn(Any) -> Any]

    fn __init__(out self, data: List[Any], transform: Optional[fn(Any) -> Any] = None):
        self.data = data
        self.transform = transform

    fn __len__(self) -> Int:
        return len(self.data)

    fn __getitem__(self, idx: Int) -> Any:
        var item = self.data[idx]
        if self.transform is not None:
            item = self.transform(item)
        return item


# -----------------------------------------------------------------------------
# Example: subclass with custom logic
# -----------------------------------------------------------------------------
struct RangeSquaredDataset(MapDataset):
    var start: Int
    var end: Int

    fn __init__(out self, start: Int, end: Int):
        self.start = start
        self.end = end
        var arr = List[Int]()
        for i in range(start, end):
            arr.append(i)
        super.__init__(arr, fn(x: Any) -> Any: return x * x)


 


 
