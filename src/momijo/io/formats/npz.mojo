# MIT License
# Copyright (c) 2025 Morteza Talebou and Mitra Daneshmand
# Project: momijo  |  Source: https://github.com/taleblou/momijo
# This file is part of the Momijo project. See the LICENSE file at the repository root.
# Momijo 
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Morteza Talebou and Mitra Daneshmand
# Website: https://taleblou.ir/
# Repository: https://github.com/taleblou/momijo
#
# Project: momijo.io.formats
# File: src/momijo/io/formats/npz.mojo

from momijo.tensor.tensor import Tensor
import numpy as np
import os
import zipfile

fn save_npz(path: String, arrays: Dict[String, Tensor]) -> None:
    var tmp_files = List[String]()
    var zf = zipfile.ZipFile(path, "w")
    for (name, tensor) in arrays.items():
        var npy_path = name + ".npy"
        var arr = tensor.to_numpy()
        np.save(npy_path, arr)
        zf.write(npy_path, arcname=npy_path)
        tmp_files.append(npy_path)
    zf.close()
    # cleanup temp files
    for f in tmp_files:
        os.remove(f)

# -----------------------------------------------------------------------------
# Load tensors from .npz archive
# -----------------------------------------------------------------------------
fn load_npz(path: String) -> Dict[String, Tensor]:
    if not os.path.exists(path):
        raise FileNotFoundError("NPZ file not found: " + path)

    var result = Dict[String, Tensor]()
    var zf = zipfile.ZipFile(path, "r")
    for fname in zf.namelist():
        if fname.endswith(".npy"):
            var arr = np.load(zf.open(fname))
            var t = Tensor.from_numpy(arr)
            var key = fname.replace(".npy","")
            result[key] = t
    zf.close()
    return result