# MIT License
# SPDX-License-Identifier: MIT
# Project:      Momijo
# Module:       learn.api.serialization
# File:         src/momijo/learn/api/serialization.mojo
#
# Description:
#   Save/Load helpers compatible with learn/utils checkpoints.
#   Provides:
#     - save/load/load_state_dict for Sequential models (load returns (ok, header, blob)).
#     - save_linear/load_linear/load_state_dict_linear for a single Linear layer.
#
# Notes:
#   - File I/O relies on learn/utils/fileio_stub; replace its stubs with a real implementation.
from pathlib.path import Path
from momijo.tensor import tensor
from momijo.learn.nn.layers import Linear
from momijo.learn.api.sequential import Sequential
from momijo.learn.utils.checkpoint import make_checkpoint, apply_checkpoint
from momijo.learn.utils.checkpoint_fileio import save_checkpoint_files, load_checkpoint_files
from momijo.learn.utils import write_all_bytes, read_all_bytes, pack_f64_to_bytes_binary, unpack_bytes_to_f64_binary

# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------
fn _header_path(base: String) -> String:
    return base + String(".header.json")

fn _blob_path(base: String) -> String:
    return base + String(".state.bin")

fn _tensor_to_csv(x: tensor.Tensor[Float32]) -> String:
    var n = x.numel()
    var s = String("")
    var i = 0
    while i < n:
        s = s + String(x._data[i])
        if i + 1 < n: s = s + String(",")
        i = i + 1
    return s

fn _csv_to_list(s: String) -> List[Float32]:
    from collections.list import List
    var vals = List[Float32]()
    var cur = String("")
    var L = len(s)
    var i = 0
    while i < L:
        var ch = s[i]
        if ch == ',':
            if cur.__len__() > 0:
                try: vals.append(Float32(cur))
                except _: vals.append(0.0)
                cur = String("")
        else:
            cur = cur + String(ch)
        i = i + 1
    if cur.__len__() > 0:
        try: vals.append(Float32(cur))
        except _: vals.append(0.0)
    return vals.copy()

fn _u8_to_string(b: tensor.Tensor[UInt8]) -> String:
    var n = b.numel()
    var s = String("")
    var i = 0
    while i < n:
        s = s + String((b._data[i]))
        i = i + 1
    return s

fn _string_to_u8(s: String) -> tensor.Tensor[UInt8]:
    var n = len(s)
    var out = tensor.zeros_u8([n])
    var i = 0
    while i < n:
        var code: Int = 0
        try:
            code = Int(s[i])
        except _:
            code = 63                      # '?'
        if code < 0: code = 0
        if code > 255: code = 255
        out._data[i] = UInt8(code)
        i = i + 1
    return out.copy()

# Fallback CSV reader for Float32 blobs if binary packers are unavailable.
fn _csv_to_floats(bytes: tensor.Tensor[UInt8]) -> tensor.Tensor[Float32]:
    var s = _u8_to_string(bytes)


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

# -----------------------------------------------------------------------------
# Internal: flatten/apply for Linear (weight then bias)
# -----------------------------------------------------------------------------
fn _flatten_linear(m: Linear) -> tensor.Tensor[Float32]:
    var n = m.weight.numel() + m.bias_t.numel()
    var out = tensor.zeros([n])
    var k = 0
    var i = 0
    while i < m.weight.numel():
        out._data[k] = m.weight._data[i]; k = k + 1; i = i + 1
    i = 0
    while i < m.bias_t.numel():
        out._data[k] = m.bias_t._data[i]; k = k + 1; i = i + 1
    return out.copy()

fn _apply_linear(mut m: Linear, blob: tensor.Tensor[Float32]) -> Bool:
    var nW = m.weight.numel()
    var nB = m.bias_t.numel()
    if blob.numel() != (nW + nB): return False
    var k = 0
    var i = 0
    while i < nW: m.weight._data[i] = blob._data[k]; k = k + 1; i = i + 1
    i = 0
    while i < nB: m.bias_t._data[i] = blob._data[k]; k = k + 1; i = i + 1
    return True

# -----------------------------------------------------------------------------
# Public API — Sequential models
# -----------------------------------------------------------------------------

# Save a Sequential to files next to `base` (writes base.header.json, base.state.bin).
fn save(net: Sequential, base: String) -> Bool:
    var hp = _header_path(base)
    var bp = _blob_path(base)
    return save_checkpoint_files(net, hp, bp)

# Load raw state (header, blob) from files, without applying to a net.
fn load(base: String) -> (Bool, String, tensor.Tensor[Float32]):
    var hp = _header_path(base)
    var bp = _blob_path(base)
    try:
        var p_h = Path(hp)
        var p_b = Path(bp)
        if not p_h.exists() or not p_b.exists():
            return (False, String(""), tensor.zeros([0]))

        var header = p_h.read_text()
        var csv    = p_b.read_text()

        # parse CSV string
        var vals = _csv_to_list(csv)
        var blob = tensor.Tensor[Float32](vals)
        return (True, header, blob.copy())
    except _:
        return (False, String(""), tensor.zeros([0]))




# Apply a loaded state onto an existing Sequential with matching architecture.
fn load_state_dict(mut net: Sequential, header: String, blob: tensor.Tensor[Float32]) -> Bool:
    return apply_checkpoint(net, header, blob)

# Convenience: directly read from files and apply to net.
fn load_into(mut net: Sequential, base: String) -> Bool:
    var hp = _header_path(base)
    var bp = _blob_path(base)
    return load_checkpoint_files(mut net, hp, bp)

# -----------------------------------------------------------------------------
# Public API — Linear-only helpers (simple, single-layer use-cases)
# -----------------------------------------------------------------------------

# Save a single Linear layer's parameters next to `base` using our own small format.
fn save_linear(m: Linear, base: String) -> Bool:
    var hp = _header_path(base)
    var bp = _blob_path(base)

    var header = String("{\"type\":\"Linear\",\"in\":") + String(m.in_features) +
                 String(",\"out\":") + String(m.out_features) + String("}")
    var blob = _flatten_linear(m)
    var csv = _tensor_to_csv(blob)


    try:
        var p_h = Path(hp);  p_h.write_text(header)
        var p_b = Path(bp);  p_b.write_text(csv)
        return True
    except _:
        return False


# Load Linear-only files and return (ok, header, blob).
fn load_linear(base: String) -> (Bool, String, tensor.Tensor[Float32]):
    return load(base)

# Apply a flattened blob (weight then bias) to a Linear layer.
fn load_state_dict_linear(mut m: Linear, blob: tensor.Tensor[Float32]) -> Bool:
    return _apply_linear( m, blob)
