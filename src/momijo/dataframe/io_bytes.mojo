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
# File: src/momijo/dataframe/io_bytes.mojo

from momijo.dataframe.helpers import bytes_to_string
from momijo.dataframe.series_bool import append
from momijo.extras.stubs import len, nt64

fn bytes_to_string(bs: List[UInt8]) -> String
    var out = String("")
    var i = 0
    while i < len(bs):
        out = out + String(Cha

&  UInt8(0xFF)))
    b.append(UInt8((x >> UInt8(16)) &  UInt8(0xFF)))
    b.append(UInt8((x >> UInt8(24)) &  UInt8(0xFF)))
    return b

fn u64

nt64(x))