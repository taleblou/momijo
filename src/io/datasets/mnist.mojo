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
# File: src/momijo/io/datasets/mnist.mojo
# ============================================================================

import os
import struct
import urllib.request
import gzip

from momijo.tensor.tensor import Tensor

# -----------------------------------------------------------------------------
# MNIST dataset loader
# -----------------------------------------------------------------------------
struct MNIST:
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

    fn __len__(self) -> Int:
        return len(self.data)

    fn __getitem__(self, idx: Int) -> (Tensor, Int):
        var img = self.data[idx]
        if self.transform is not None:
            img = self.transform(img)
        return (img, self.targets[idx])

    # -------------------------------------------------------------------------
    # Internal: download dataset
    # -------------------------------------------------------------------------
    fn _download(mut self):
        var base_url = "http://yann.lecun.com/exdb/mnist/"
        var files = [
            "train-images-idx3-ubyte.gz",
            "train-labels-idx1-ubyte.gz",
            "t10k-images-idx3-ubyte.gz",
            "t10k-labels-idx1-ubyte.gz"
        ]
        os.makedirs(self.root, exist_ok=True)
        for fname in files:
            var path = os.path.join(self.root, fname)
            if not os.path.exists(path):
                print("Downloading " + fname + "...")
                urllib.request.urlretrieve(base_url + fname, path)

    # -------------------------------------------------------------------------
    # Internal: load data from MNIST files
    # -------------------------------------------------------------------------
    fn _load_data(mut self):
        if self.train:
            images_path = os.path.join(self.root, "train-images-idx3-ubyte.gz")
            labels_path = os.path.join(self.root, "train-labels-idx1-ubyte.gz")
        else:
            images_path = os.path.join(self.root, "t10k-images-idx3-ubyte.gz")
            labels_path = os.path.join(self.root, "t10k-labels-idx1-ubyte.gz")

        # Read labels
        var f = gzip.open(labels_path, "rb")
        var magic, num = struct.unpack(">II", f.read(8))
        var labels = List[Int]()
        for i in range(num):
            var (val,) = struct.unpack("B", f.read(1))
            labels.append(val)
        f.close()

        # Read images
        var f2 = gzip.open(images_path, "rb")
        var magic2, num2, rows, cols = struct.unpack(">IIII", f2.read(16))
        var images = List[Tensor]()
        for i in range(num2):
            var buf = f2.read(rows * cols)
            var arr = List[List[Float32]]()
            for r in range(rows):
                var row = List[Float32]()
                for c in range(cols):
                    var val: UInt8 = buf[r*cols+c]
                    row.append(Float32(val) / 255.0)
                arr.append(row)
            var tensor = Tensor.from_array(arr).unsqueeze(0)  # [1,28,28]
            images.append(tensor)
        f2.close()

        self.data = images
        self.targets = labels


# -----------------------------------------------------------------------------
# Minimal self-test
# -----------------------------------------------------------------------------

fn _self_test() -> Bool:
    var ok = True
    # NOTE: assumes MNIST files already downloaded in /tmp
    var dataset = MNIST(root="/tmp", train=True, download=False)
    ok = ok and len(dataset) > 0
    var (img, label) = dataset[0]
    ok = ok and img.shape_as_list() == [1,28,28]
    ok = ok and label >= 0 and label <= 9
    return ok


fn main():
    if _self_test():
        print("MNIST module self-test: OK")
    else:
        print("MNIST module self-test: FAIL")
