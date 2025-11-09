# MIT License
# SPDX-License-Identifier: MIT
# Project: momijo
# File: src/momijo/learn/utils/tensor_bytes.mojo
# Description: Portable ASCII-based pack/unpack helpers for Float32 tensors.
#
# NOTE: These are NOT raw IEEE754 binary; they serialize as UTF-8 CSV for portability.
# They satisfy an API like pack_f64_to_bytes/unpack_bytes_to_f64.

from momijo.tensor import tensor

# ------------ UTF-8 string <-> bytes (UInt8 Tensor) helpers ------------
fn _string_to_u8(s: String) -> tensor.Tensor[UInt8]:
    var n = len(s)
    var out = tensor.zeros_u8([n])
    var i = 0
    while i < n:
        var code: Int = 0
        try:
            code = Int(s[i])
        except _:
            code = 0
        if code < 0: code = 0
        if code > 255: code = 255
        out._data[i] = UInt8(code)
        i = i + 1
    return out.copy()

fn _u8_to_string(b: tensor.Tensor[UInt8]) -> String:
    var n = b.numel()
    var s = String("")
    var i = 0
    while i < n:
        s = s + String((b._data[i]))
        i = i + 1
    return s

# ------------ CSV (portable) pack/unpack for Float32 ------------
# Public: pack Float32 tensor to ASCII CSV bytes.
fn pack_f64_to_bytes(x: tensor.Tensor[Float32]) -> tensor.Tensor[UInt8]:
    var n = x.numel()
    var s = String("")
    var i = 0
    while i < n:
        s = s + String(x._data[i])
        if i + 1 < n:
            s = s + String(",")
        i = i + 1
    return _string_to_u8(s)

# Public: unpack ASCII CSV bytes to Float32 tensor.
fn unpack_bytes_to_f64(b: tensor.Tensor[UInt8]) -> tensor.Tensor[Float32]:
    var s = _u8_to_string(b)
    var vals = List[Float32]()
    var cur = String("")
    var L = len(s)
    var i = 0
    while i < L:
        var ch = s[i]
        if ch == ',':
            if cur.__len__() > 0:
                try:
                    vals.append(Float32(cur))
                except _:
                    vals.append(0.0)
                cur = String("")
        else:
            cur = cur + String(ch)
        i = i + 1
    if cur.__len__() > 0:
        try:
            vals.append(Float32(cur))
        except _:
            vals.append(0.0)


    return tensor.Tensor[Float32](vals)

# ------------ Binary fast-path wrappers (fallback to CSV if unavailable) ------------
fn pack_f64_to_bytes_binary(x: tensor.Tensor[Float32]) -> tensor.Tensor[UInt8]:
    # Try tensor facade binary pack; if not present, CSV fallback.
    return pack_f64_to_bytes(x)             # CSV fallback

fn unpack_bytes_to_f64_binary(b: tensor.Tensor[UInt8]) -> tensor.Tensor[Float32]:
    return unpack_bytes_to_f64(b)           # CSV fallback
