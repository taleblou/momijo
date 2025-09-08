# Project:      Momijo
# Module:       src.momijo.dataframe.io_bytes
# File:         io_bytes.mojo
# Path:         src/momijo/dataframe/io_bytes.mojo
#
# Description:  src.momijo.dataframe.io_bytes — focused Momijo functionality with a stable public API.
#               Composable building blocks intended for reuse.
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
#   - Key functions: bytes_to_string


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