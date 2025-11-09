# MIT License
# SPDX-License-Identifier: MIT
# Project:      momijo
# File:         src/momijo/learn/utils/checkpoint.mojo
# Description:  Stateless checkpoint (Linear + Conv2d) -> (header JSON string, flat blob).

from momijo.tensor import tensor
from momijo.learn.api.sequential import Sequential, NNSequential
from momijo.learn.nn.conv import Conv2d

from pathlib.path import Path
from collections.list import List

# ------------------------------------------------------------------------------
# Build a flat tensor blob of parameters in traversal order, with a compact JSON
# header describing per-layer param sizes. Supports Linear and Conv2d.
# Returns: (header_json, flat_blob[Float32])
# ------------------------------------------------------------------------------
fn make_checkpoint(net: Sequential) -> (String, tensor.Tensor[Float32]):
    var total = 0
    var i = 0
    var n = net.len()
    while i < n:
        var m = net.get(i)
        if m.tag == 0:
            # Linear: weight + (optional) bias_t
            total = total + m.linear.weight.numel()
            total = total + m.linear.bias_t.numel()
        elif m.tag == 7:
            # Conv2d: weight + (optional) bias
            total = total + m.conv2d.weight.numel()
            total = total + m.conv2d.bias.numel()
        i = i + 1

    var flat = tensor.zeros([total])

    var header = String("{\"params\":[")
    var first = True

    i = 0
    var k = 0
    while i < n:
        var m2 = net.get(i)
        if m2.tag == 0:
            # Linear
            var W = m2.linear.weight
            var B = m2.linear.bias_t
            var nW = W.numel()
            var nB = B.numel()

            # copy W
            var j = 0
            while j < nW:
                flat._data[k + j] = W._data[j]
                j = j + 1
            k = k + nW

            # copy B (may be zero-length)
            j = 0
            while j < nB:
                flat._data[k + j] = B._data[j]
                j = j + 1
            k = k + nB

            # header entry
            if not first:
                header = header + String(",")
            header = header + String("{\"type\":\"Linear\",\"w\":") + String(nW) + String(",\"b\":") + String(nB) + String("}")
            first = False

        elif m2.tag == 7:
            # Conv2d
            var Wc = m2.conv2d.weight
            var Bc = m2.conv2d.bias
            var nWc = Wc.numel()
            var nBc = Bc.numel()

            # copy Wc
            var j2 = 0
            while j2 < nWc:
                flat._data[k + j2] = Wc._data[j2]
                j2 = j2 + 1
            k = k + nWc

            # copy Bc (may be zero-length)
            j2 = 0
            while j2 < nBc:
                flat._data[k + j2] = Bc._data[j2]
                j2 = j2 + 1
            k = k + nBc

            # header entry
            if not first:
                header = header + String(",")
            header = header + String("{\"type\":\"Conv2d\",\"w\":") + String(nWc) + String(",\"b\":") + String(nBc) + String("}")
            first = False
        i = i + 1

    header = header + String("]}")
    return (header, flat)

# ------------------------------------------------------------------------------
# Apply a flat tensor blob back into net (Linear + Conv2d) in traversal order.
# Ignores header structure (relies on the same ordering and sizes).
# Returns: True on success.
# ------------------------------------------------------------------------------
fn apply_checkpoint(mut net: Sequential, header: String, blob: tensor.Tensor[Float32]) -> Bool:
    _ = header
    var i = 0
    var k = 0
    var n = net.len()

    while i < n:
        var m = net.get(i)
        if m.tag == 0:
            # Linear
            var nW = m.linear.weight.numel()
            var nB = m.linear.bias_t.numel()

            var j = 0
            while j < nW:
                m.linear.weight._data[j] = blob._data[k + j]
                j = j + 1
            k = k + nW

            j = 0
            while j < nB:
                m.linear.bias_t._data[j] = blob._data[k + j]
                j = j + 1
            k = k + nB

            net.set(i, m)

        elif m.tag == 7:
            # Conv2d
            var nWc = m.conv2d.weight.numel()
            var nBc = m.conv2d.bias.numel()

            var j2 = 0
            while j2 < nWc:
                m.conv2d.weight._data[j2] = blob._data[k + j2]
                j2 = j2 + 1
            k = k + nWc

            j2 = 0
            while j2 < nBc:
                m.conv2d.bias._data[j2] = blob._data[k + j2]
                j2 = j2 + 1
            k = k + nBc

            net.set(i, m)
        i = i + 1

    return True

# ------------------------------------------------------------------------------
# Internal helpers (no-raise, defensive where possible)
# ------------------------------------------------------------------------------
fn _escape_json_string(s: String) -> String:
    # Minimal JSON string escaper (quotes, backslash, newline, tab)
    var out = String("")
    var i = 0
    while i < s.__len__():
        var ch = s[i]
        if ch == String("\\")[0]:
            out = out + String("\\\\")
        elif ch == String("\"")[0]:
            out = out + String("\\\"")
        elif ch == String("\n")[0]:
            out = out + String("\\n")
        elif ch == String("\t")[0]:
            out = out + String("\\t")
        else:
            out = out + String(ch)
        i = i + 1
    return out

fn _now_iso8601() -> String:
    # Deterministic timestamp for tests
    return String("1970-01-01T00:00:00Z")

# Non-raising write; no parent-mkdir (portable)
fn _write_text(p: Path, s: String) -> None:
    try:
        p.write_text(s)
    except e:
        pass

# Non-raising read; returns "" on failure
fn _read_text(p: Path) -> String:
    if not p.exists():
        return String("")
    try:
        return p.read_text()
    except e:
        return String("")

# Safe substring for String
fn _substr(s: String, start_idx: Int, end_idx: Int) -> String:
    var n = s.__len__()
    var a = start_idx
    var b = end_idx
    if a < 0: a = 0
    if b > n: b = n
    if b <= a: return String("")
    var out = String("")
    var i = a
    while i < b:
        out = out + String(s[i])
        i = i + 1
    return out

# Derived sibling paths for a given base path string
fn _header_path(base: String) -> Path:
    return Path(base + String(".json"))

fn _state_path(base: String) -> Path:
    return Path(base + String(".state.json"))

fn _blob_path(base: String) -> Path:
    return Path(base + String(".bin"))

# Tiny extractors from header JSON (string find; no full JSON parse)
fn _extract_state_path_from_header(header_json: String) -> String:
    var key = String("\"state\":\"")
    var idx = header_json.find(key)
    if idx < 0:
        return String("")
    var start = idx + key.__len__()
    var end = header_json.find(String("\""), start)
    if end < 0:
        return String("")
    return _substr(header_json, start, end)

fn _extract_model_class_from_header(header_json: String) -> String:
    var key = String("\"model_class\":\"")
    var idx = header_json.find(key)
    if idx < 0:
        return String("")
    var start = idx + key.__len__()
    var end = header_json.find(String("\""), start)
    if end < 0:
        return String("")
    return _substr(header_json, start, end)

# ------------------------------------------------------------------------------
# Public API for state JSON (header + state JSON; blob reserved for future)
# Sequential-only (simple, non-raising wrappers)
# ------------------------------------------------------------------------------
fn save_state_dict(model: Sequential, path: String) -> None:
    var base_path = String(path)
    var header_path = _header_path(base_path)
    var state_path  = _state_path(base_path)
    var blob_path   = _blob_path(base_path)

    var state_json = model.state_dict()
    var model_class = String("Model")

    var header_json =
        String("{") +
        String("\"format\":\"MNP\",") +
        String("\"version\":1,") +
        String("\"created_at\":\"") + _escape_json_string(_now_iso8601()) + String("\",") +
        String("\"model_class\":\"") + _escape_json_string(model_class) + String("\",") +
        String("\"files\":{") +
            String("\"state\":\"") + _escape_json_string(state_path.__str__()) + String("\",") +
            String("\"blob\":\"")  + _escape_json_string(blob_path.__str__())  + String("\"") +
        String("},") +
        String("\"blob_meta\":{") +
            String("\"dtype\":\"float32\",") +
            String("\"endian\":\"little\",") +
            String("\"total_elems\":0") +
        String("}") +
        String("}")

    _write_text(header_path, header_json)
    _write_text(state_path, state_json)


# 2-arg: save_state_dict(NNSequential, path)
fn save_state_dict(model: NNSequential, path: String) -> None:
    # delegate to Sequential version
    save_state_dict(model.seq, path)

# 3-arg: save_state_dict(NNSequential, dir, stem)
fn save_state_dict(model: NNSequential, dir_: String, stem: String) -> None:
    var base = _join_dir_stem(dir_, stem)
    save_state_dict(model.seq, base)




fn load_state_dict(mut model: Sequential, path: String) -> Bool:
    var base_path = String(path)
    var header = _header_path(base_path)

    if header.exists():
        var header_json = _read_text(header)
        var state_str = _extract_state_path_from_header(header_json)
        if state_str.__len__() == 0:
            return False
        var state_path = Path(state_str)
        if not state_path.exists():
            var local_state = _state_path(base_path)
            if not local_state.exists():
                return False
            state_path = local_state
        var state_json = _read_text(state_path)
        model.load_state_dict(state_json)
        return True

    # Fallback: maybe `path` is the state JSON directly
    var direct = Path(base_path)
    if direct.exists():
        var state_json2 = _read_text(direct)
        model.load_state_dict(state_json2)
        return True

    return False


# 2-arg: load_state_dict(NNSequential, path)
fn load_state_dict(mut model: NNSequential, path: String) -> Bool:
    return load_state_dict( model.seq, path)

# 3-arg: load_state_dict(NNSequential, dir, stem)
fn load_state_dict(mut model: NNSequential, dir_: String, stem: String) -> Bool:
    var base = _join_dir_stem(dir_, stem)
    return load_state_dict( model.seq, base)

fn load_state_dict(mut model: Sequential, dir_: String, stem: String) -> Bool:
    var base = _join_dir_stem(dir_, stem)
    return load_state_dict( model, base)

# Reads `<path>.json` and returns model_class if present; else "".
fn read_model_class_from_header(path: String) -> String:
    var header = _header_path(path)
    if not header.exists():
        return String("")
    var header_json = _read_text(header)
    return _extract_model_class_from_header(header_json)


# Keep both exists_checkpoint arities:
# 1-arg already exists: fn exists_checkpoint(path: String) -> Bool
# Add 2-arg: exists_checkpoint(dir, stem)
fn exists_checkpoint(dir_: String, stem: String) -> Bool:
    var base = _join_dir_stem(dir_, stem)
    return exists_checkpoint(base)
# Returns True if a valid header and state file are present.
fn exists_checkpoint(path: String) -> Bool:
    var header = _header_path(path)
    if not header.exists():
        return False
    var header_json = _read_text(header)
    var state_str = _extract_state_path_from_header(header_json)
    if state_str.__len__() == 0:
        return False
    var state_path = Path(state_str)
    if state_path.exists():
        return True
    var local_state = _state_path(path)
    return local_state.exists()

# Convenience: build base path from (dir, stem) and reuse exists_checkpoint(base)
fn exists_checkpoint_at(dir_: String, stem: String) -> Bool:
    var base = dir_
    if base.__len__() > 0:
        var last = base[base.__len__() - 1]
        if last != String("/")[0]:
            base = base + String("/")
    base = base + stem
    return exists_checkpoint(base)


# ------------------------------------------------------------------------------
# Convenience overloads: dir + stem  → build base path and reuse core API
# Place these at the end of the file.
# ------------------------------------------------------------------------------

fn _join_dir_stem(dir_: String, stem: String) -> String:
    var base = dir_
    if base.__len__() > 0:
        var last = base[base.__len__() - 1]
        if last != String("/")[0]:
            base = base + String("/")
    return base + stem
