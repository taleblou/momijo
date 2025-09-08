# Project:      Momijo
# Module:       src.momijo.io.formats.npy
# File:         npy.mojo
# Path:         src/momijo/io/formats/npy.mojo
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
#   - Key functions: save_npy, load_npy


from momijo.tensor.tensor import Tensor
import numpy as np
import os

fn save_npy(path: String, tensor: Tensor) -> None:
    var arr = tensor.to_numpy()  # assuming Tensor <-> numpy bridge
    np.save(path, arr)

# -----------------------------------------------------------------------------
# Load tensor from .npy file
# -----------------------------------------------------------------------------
fn load_npy(path: String) -> Tensor:
    if not os.path.exists(path):
        raise FileNotFoundError("NPY file not found: " + path)
    var arr = np.load(path, allow_pickle=False)
    return Tensor.from_numpy(arr)