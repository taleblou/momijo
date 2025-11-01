# MIT License
# SPDX-License-Identifier: MIT
# Project: momijo
# File: src/momijo/learn/utils/checkpoint_fileio.mojo
# Description: File I/O wrappers for checkpoints. Uses project-level I/O helpers when available.
#
# Expected project utilities (if present):
#   - write_all_bytes(path: String, data: tensor.Tensor[UInt8]) -> Bool
#   - read_all_bytes(path: String) -> tensor.Tensor[UInt8]
#   - tensor.pack_f64_to_bytes(x: tensor.Tensor[Float64]) -> tensor.Tensor[UInt8]
#   - tensor.unpack_bytes_to_f64(b: tensor.Tensor[UInt8]) -> tensor.Tensor[Float64]
#
# If pack/unpack are not available, we fall back to a simple UTF-8 text format
# (comma-separated floats) for the blob for portability.

from momijo.tensor import tensor
from momijo.learn.api.sequential import Sequential
from momijo.learn.utils.checkpoint import make_checkpoint, apply_checkpoint
from momijo.learn.utils.tensor_bytes import pack_f64_to_bytes, unpack_bytes_to_f64
from momijo.learn.utils.fileio_stub import write_all_bytes, read_all_bytes

fn _has_pack_helpers() -> Bool:
    # Heuristic: try to call into helpers in a try block (compile-time/link-time availability varies).
    try:
        var t = tensor.zeros([1])
        var b = tensor.pack_f64_to_bytes(t)
        var t2 = tensor.unpack_bytes_to_f64(b)
        _ = t2
        return True
    except _:
        return False

fn _string_to_u8(s: String) -> tensor.Tensor[UInt8]:
    var n = len(s)
    var out = tensor.zeros_u8([n])
    var i = 0
    while i < n:
        out._data[i] = UInt8(s[i])
        i += 1
    return out

fn _u8_to_string(b: tensor.Tensor[UInt8]) -> String:
    var n = b.numel()
    var s = String("")
    var i = 0
    while i < n:
        s = s + String((b._data[i]))
        i += 1
    return s

fn _floats_to_csv(x: tensor.Tensor[Float64]) -> tensor.Tensor[UInt8]:
    var n = x.numel()
    var s = String("")
    var i = 0
    while i < n:
        s = s + String(x._data[i])
        if i + 1 < n: s = s + String(",")
        i += 1
    return _string_to_u8(s)

fn _csv_to_floats(b: tensor.Tensor[UInt8]) -> tensor.Tensor[Float64]:
    var s = _u8_to_string(b)
    # Simple CSV split on ','
    var out = tensor.zeros([0])
    var cur = String("")
    var i = 0
    while i < len(s):
        var ch = s[i]
        if ch == ',':
            # append cur
            try:
                var v = Float64(cur)
                out = tensor.append_scalar(out, v)
            except _:
                out = tensor.append_scalar(out, 0.0)
            cur = String("")
        else:
            cur = cur + String((ch))
        i += 1
    if len(cur) > 0:
        try:
            var v2 = Float64(cur)
            out = tensor.append_scalar(out, v2)
        except _:
            out = tensor.append_scalar(out, 0.0)
    return out

fn save_checkpoint_files(net: Sequential, header_path: String, blob_path: String) -> Bool:
    var pair = make_checkpoint(net)
    var header = pair[0]; var blob = pair[1]
    try:
        var ok1 = write_all_bytes(header_path, _string_to_u8(header))
        var ok2 = False
        if _has_pack_helpers():
            ok2 = write_all_bytes(blob_path, pack_f64_to_bytes_binary(blob))
        else:
            ok2 = write_all_bytes(blob_path, _floats_to_csv(blob))
        return ok1 and ok2
    except _:
        return False

fn load_checkpoint_files(mut net: Sequential, header_path: String, blob_path: String) -> Bool:
    try:
        var hdr_bytes = read_all_bytes(header_path)
        var header = _u8_to_string(hdr_bytes)
        var blob_bytes = read_all_bytes(blob_path)
        var blob = tensor.zeros([0])
        if _has_pack_helpers():
            blob = unpack_bytes_to_f64_binary(blob_bytes)
        else:
            blob = _csv_to_floats(blob_bytes)
        return apply_checkpoint(net, header, blob)
    except _:
        return False
