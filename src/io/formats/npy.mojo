# ============================================================================
# MIT License
# Copyright (c) 2025 Morteza Talebou and Mitra Daneshmand
# Project: momijo  |  Source: https://github.com/taleblou/momijo
# This file is part of the Momijo project. See the LICENSE file at the repository root.
# SPDX-License-Identifier: MIT
# Website: https://taleblou.ir/
# Repository: https://github.com/taleblou/momijo
#
# Project: momijo.io.formats
# File: src/momijo/io/formats/npy.mojo
# ============================================================================

import os
import struct
import numpy as np

from momijo.tensor.tensor import Tensor

# -----------------------------------------------------------------------------
# Save tensor to .npy file
# -----------------------------------------------------------------------------
fn save_npy(path: String, tensor: Tensor):
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


 