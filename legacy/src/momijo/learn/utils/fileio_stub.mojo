# MIT License
# SPDX-License-Identifier: MIT
# Project: momijo
# File: src/momijo/learn/utils/fileio_stub.mojo
# Description: Minimal real file I/O using Path.read_text/write_text.

from pathlib.path import Path
from momijo.tensor import tensor
from collections.list import List
from momijo.learn.api.serialization import _string_to_u8,_u8_to_string


fn write_all_bytes(path: String, data: tensor.Tensor[UInt8]) -> Bool:
    try:
        var p = Path(path)
        var s = _u8_to_string(data)   # bytes → متن ASCII
        p.write_text(s)
        return True
    except _:
        return False

fn read_all_bytes(path: String) -> tensor.Tensor[UInt8]:
    try:
        var p = Path(path)
        if not p.exists():
            return tensor.zeros_u8([0])
        var s = p.read_text()         # متن ASCII
        return _string_to_u8(s)       # متن → bytes
    except e:
        print(e)
        return tensor.zeros_u8([0])
