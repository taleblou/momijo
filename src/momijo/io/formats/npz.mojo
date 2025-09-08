# Project:      Momijo
# Module:       src.momijo.io.formats.npz
# File:         npz.mojo
# Path:         src/momijo/io/formats/npz.mojo
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
#   - Key functions: save_npz, load_npz
#   - Performs file/Path IO; prefer context-managed patterns.


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