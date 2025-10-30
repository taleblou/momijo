# MIT License
# SPDX-License-Identifier: MIT
# Project:      Momijo
# Module:       learn.utils.checkpoint
# File:         src/momijo/learn/utils/checkpoint.mojo
#
# Description:
#   Checkpoint utilities for Momijo Learn.
#   Stable header+payload layout:
#     - <base>.json       : header (JSON)
#     - <base>.state.json : model state (JSON, by model contract)
#     - <base>.bin        : reserved for future Float32 blob
#
# Author(s):    Morteza Taleblou & Mitra Daneshmand
# Website:      https://taleblou.ir/
# Repository:   https://github.com/taleblou/momijo

from pathlib.path import Path
from collections.list import List

# ------------------------------------------------------------
# Internal helpers (no-raise, defensive where possible)
# ------------------------------------------------------------

# Minimal JSON string escaper (quotes, backslash, newline, tab)
fn _escape_json_string(s: String) -> String:
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

# Placeholder timestamp; keep deterministic for tests
fn _now_iso8601() -> String:
    return String("1970-01-01T00:00:00Z")

fn _ensure_parent_dir(p: Path) -> None:
    var parent = p.parent()
    if not parent.exists():
        # try to create parents; ignore failures for now
        parent.mkdir(parents=True)

fn _write_text(p: Path, s: String) -> None:
    _ensure_parent_dir(p)
    p.write_text(s)

fn _read_text(p: Path) -> String:
    return p.read_text()

# Derived sibling paths for a given base path string
fn _header_path(base: String) -> Path:
    return Path(base + String(".json"))

fn _state_path(base: String) -> Path:
    return Path(base + String(".state.json"))

fn _blob_path(base: String) -> Path:
    return Path(base + String(".bin"))

# Tiny extractor for `"state":"<...>"` from header JSON
fn _extract_state_path_from_header(header_json: String) -> String:
    var key = String("\"state\":\"")
    var idx = header_json.find(key)
    if idx < 0:
        return String("")
    var start = idx + key.__len__()
    var end = header_json.find(String("\""), start)
    if end < 0:
        return String("")
    return header_json.slice(start, end)

# Optional: tiny extractor for `"model_class":"<...>"` (not required for load)
fn _extract_model_class_from_header(header_json: String) -> String:
    var key = String("\"model_class\":\"")
    var idx = header_json.find(key)
    if idx < 0:
        return String("")
    var start = idx + key.__len__()
    var end = header_json.find(String("\""), start)
    if end < 0:
        return String("")
    return header_json.slice(start, end)

# ------------------------------------------------------------
# Public API
# ------------------------------------------------------------

# Save a model state to disk with MNP-like layout.
# Writes:
#   - <path>.json        : header JSON
#   - <path>.state.json  : state JSON (as provided by model.state_dict())
#
# Contract for model:
#   - state_dict(self) -> String            # JSON string
#   - load_state_dict(mut self, String)     # for future load
fn save_state_dict(model, path: String) -> None:
    var base_path = String(path)
    var header_path = _header_path(base_path)
    var state_path  = _state_path(base_path)
    var blob_path   = _blob_path(base_path)      # reserved for future Float32 blob

    # Pull model state as JSON string (model is responsible for JSON correctness)
    var state_json = model.state_dict()

    # If your model exposes a class/name accessor, replace "Model" below accordingly
    var model_class = String("Model")

    # Compose header JSON
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
            String("\"dtype\":\"float32\",") +   # reserved for future
            String("\"endian\":\"little\",") +
            String("\"total_elems\":0") +
        String("}") +
        String("}")

    # Write header and state
    _write_text(header_path, header_json)
    _write_text(state_path, state_json)

    # Future: when tensor storage is ready, write concatenated Float32 to blob_path
    # and update blob_meta.total_elems plus per-tensor offsets in the header.

# Load a model state saved by `save_state_dict`.
# Behavior:
#   1) If <path>.json exists, parse it and load referenced <state> JSON.
#   2) Else, if `path` itself is a file, treat it as the state JSON file directly.
# Returns True on success, False otherwise.
fn load_state_dict(model, path: String) -> Bool:
    var base_path = String(path)
    var header = _header_path(base_path)

    if header.exists():
        var header_json = _read_text(header)
        var state_str = _extract_state_path_from_header(header_json)
        if state_str.__len__() == 0:
            return False
        var state_path = Path(state_str)

        if not state_path.exists():
            # If absolute path moved, try local sibling: <base>.state.json
            var local_state = _state_path(base_path)
            if not local_state.exists():
                return False
            state_path = local_state

        var state_json = _read_text(state_path)
        model.load_state_dict(state_json)
        return True

    # Fallback: no header; maybe user passed the state file directly
    var direct = Path(base_path)
    if direct.exists():
        var state_json2 = _read_text(direct)
        model.load_state_dict(state_json2)
        return True

    return False

# ------------------------------------------------------------
# Optional helpers (header inspection)
# ------------------------------------------------------------

# Reads `<path>.json` and returns model_class if present; else returns "".
fn read_model_class_from_header(path: String) -> String:
    var header = _header_path(path)
    if not header.exists():
        return String("")
    var header_json = _read_text(header)
    return _extract_model_class_from_header(header_json)

# Returns True if a valid header and state file are present (not parsing state).
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
    # fallback to sibling
    var local_state = _state_path(path)
    return local_state.exists()
