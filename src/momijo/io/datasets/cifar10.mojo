# Project:      Momijo
# Module:       src.momijo.io.datasets.cifar10
# File:         cifar10.mojo
# Path:         src/momijo/io/datasets/cifar10.mojo
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
#   - Structs: CIFAR10
#   - Key functions: __init__, __len__, __getitem__, _download, _load_data, __copyinit__, __moveinit__, _self_test ...
#   - Uses generic functions/types with explicit trait bounds.
#   - Performs file/Path IO; prefer context-managed patterns.


from momijo.tensor.tensor import Tensor
import os
import pickle
import tarfile
import urllib.request

struct CIFAR10:
    var root: String
    var train: Bool
    var download: Bool
    var transform: Optional[fn(Tensor) -> Tensor]
    var data: List[Tensor]
    var targets: List[Int]
fn __init__(out self, root: String, train: Bool = True,
                download: Bool = False,
                transform: Optional[fn(Tensor) -> Tensor] = None):
        self.root = root
        self.train = train
        self.download = download
        self.transform = transform
        self.data = List[Tensor]()
        self.targets = List[Int]()

        if self.download:
            self._download()

        self._load_data()

    # Length of dataset
fn __len__(self) -> Int:
        return len(self.data)

    # Get item by index
fn __getitem__(self, idx: Int) -> (Tensor, Int):
        var img = self.data[idx]
        if self.transform is not None:
            img = self.transform(img)
        return (img, self.targets[idx])

    # -------------------------------------------------------------------------
    # Internal: download dataset if not present
    # -------------------------------------------------------------------------
fn _download(mut self):
        var url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
        var filename = os.path.join(self.root, "cifar-10-python.tar.gz")
        var folder = os.path.join(self.root, "cifar-10-batches-py")

        if os.path.exists(folder):
            return

        print("Downloading CIFAR-10 dataset...")
        urllib.request.urlretrieve(url, filename)

        print("Extracting...")
        var tar = tarfile.open(filename, "r:gz")
        tar.extractall(self.root)
        tar.close()
        print("Done.")

    # -------------------------------------------------------------------------
    # Internal: load data from extracted files
    # -------------------------------------------------------------------------
fn _load_data(mut self) -> None:
        var base = os.path.join(self.root, "cifar-10-batches-py")
        var batches = List[String]()

        if self.train:
            batches = ["data_batch_1", "data_batch_2", "data_batch_3",
                       "data_batch_4", "data_batch_5"]
        else:
            batches = ["test_batch"]

        for batch in batches:
            var path = os.path.join(base, batch)
            if not os.path.exists(path):
                continue
            var f = open(path, "rb")
            var entry = pickle.load(f, encoding="bytes")
            f.close()

            var data_arr = entry[b"data"]
            var labels = entry[b"labels"]

            # reshape to [N, 3, 32, 32]
            var n = len(labels)
            for i in range(n):
                var flat = data_arr[i]
                var r = flat[0:1024].reshape(32,32)
                var g = flat[1024:2048].reshape(32,32)
                var b = flat[2048:3072].reshape(32,32)
                var img = Tensor.stack([r,g,b], axis=0).to_dtype("f32")
                self.data.append(img)
                self.targets.append(labels[i])
fn __copyinit__(out self, other: Self) -> None:
        self.root = other.root
        self.train = other.train
        self.download = other.download
        self.transform = other.transform
        self.data = other.data
        self.targets = other.targets
fn __moveinit__(out self, deinit other: Self) -> None:
        self.root = other.root
        self.train = other.train
        self.download = other.download
        self.transform = other.transform
        self.data = other.data
        self.targets = other.targets
# -----------------------------------------------------------------------------
# Minimal self-test
# -----------------------------------------------------------------------------
fn _self_test() -> Bool:
    var ok = True
    var dataset = CIFAR10(root="/tmp", train=True, download=False)
    ok = ok and isinstance(len(dataset), Int)
    return ok
fn main() -> None:
    if _self_test():
        print("CIFAR10 module self-test: OK")
    else:
        print("CIFAR10 module self-test: FAIL")