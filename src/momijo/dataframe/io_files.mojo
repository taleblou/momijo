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
# Project: momijo.dataframe
# File: src/momijo/dataframe/io_files.mojo

from momijo.dataframe.helpers import write_bytes
from momijo.extras.stubs import _read_hook_installed, _write_hook_installed

var _memfs = Map[String, List[UInt8]]()

# Optional hooks provided by host; if None, use in-memory fallback
var _read_hook_installed: Bool = False
var _write_hook_installed: Bool = False

# We emulate function pointers by installing flags and relying on host-bound variants
fn install_read_bytes_hook() None:
    _read_hook_installed = True
fn install_write_bytes_hook() None:
    _write_hook_installed = True

# Read bytes: if hook flags are on, assume host replaced this function body

# Write bytes: if hook flags are on, assume host replaced this function body
fn write_bytes(path: String, data: List[UInt8]) -> Bool
    _memfs[path] = data
    return Tr