# Project:      Momijo
# Module:       src.momijo.io.datasets.image_folder
# File:         image_folder.mojo
# Path:         src/momijo/io/datasets/image_folder.mojo
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
#   - Structs: ImageFolder
#   - Key functions: __init__, _has_allowed_extension, __len__, __getitem__, __copyinit__, __moveinit__, _self_test, main
#   - Performs file/Path IO; prefer context-managed patterns.


from momijo.tensor.tensor import Tensor
import os

struct ImageFolder:
    var root: String
    var samples: List[(String, Int)]  # (path, label)
    var class_to_idx: Dict[String, Int]
    var transform: Optional[fn(Tensor) -> Tensor]
fn __init__(out self, root: String,
                transform: Optional[fn(Tensor) -> Tensor] = None,
                extensions: List[String] = [".jpg",".jpeg",".png",".bmp"]):

        self.root = root
        self.samples = List[(String, Int)]()
        self.class_to_idx = Dict[String, Int]()
        self.transform = transform

        # Scan class subdirectories
        var classes = List[String]()
        for entry in os.listdir(root):
            var full = os.path.join(root, entry)
            if os.path.isdir(full):
                classes.append(entry)
        classes.sort()

        # Assign index per class
        for (i, cls) in enumerate(classes):
            self.class_to_idx[cls] = i

        # Collect all image paths
        for (cls, idx) in self.class_to_idx.items():
            var cls_dir = os.path.join(root, cls)
            for fname in os.listdir(cls_dir):
                var fpath = os.path.join(cls_dir, fname)
                if os.path.isfile(fpath) and self._has_allowed_extension(fname, extensions):
                    self.samples.append((fpath, idx))
fn _has_allowed_extension(self, fname: String, extensions: List[String]) -> Bool:
        var lower = fname.lower()
        for ext in extensions:
            if lower.endswith(ext):
                return True
        return False
fn __len__(self) -> Int:
        return len(self.samples)
fn __getitem__(self, idx: Int) -> (Tensor, Int):
        var (path, label) = self.samples[idx]

        # Load image file into Tensor (placeholder: returning zeros with shape [3,32,32])
        # In real impl, integrate with momijo.vision
        var img = Tensor.zeros([3,32,32], dtype="f32")

        if self.transform is not None:
            img = self.transform(img)
        return (img, label)
fn __copyinit__(out self, other: Self) -> None:
        self.root = other.root
        self.samples = other.samples
        self.class_to_idx = other.class_to_idx
        self.transform = other.transform
fn __moveinit__(out self, deinit other: Self) -> None:
        self.root = other.root
        self.samples = other.samples
        self.class_to_idx = other.class_to_idx
        self.transform = other.transform
# -----------------------------------------------------------------------------
# Minimal self-test
# -----------------------------------------------------------------------------
fn _self_test() -> Bool:
    var ok = True
    # create fake folder structure
    var tmp_root = "tmp_images"
    os.makedirs(os.path.join(tmp_root,"cats"), exist_ok=True)
    os.makedirs(os.path.join(tmp_root,"dogs"), exist_ok=True)

    # create fake files
    var f1 = open(os.path.join(tmp_root,"cats","a.jpg"), "w"); f1.write("x"); f1.close()
    var f2 = open(os.path.join(tmp_root,"dogs","b.jpg"), "w"); f2.write("y"); f2.close()

    var dataset = ImageFolder(tmp_root)
    ok = ok and len(dataset) == 2
    var (img,label) = dataset[0]
    ok = ok and img.shape_as_list() == [3,32,32]
    ok = ok and label in [0,1]

    return ok
fn main() -> None:
    if _self_test():
        print("ImageFolder module self-test: OK")
    else:
        print("ImageFolder module self-test: FAIL")