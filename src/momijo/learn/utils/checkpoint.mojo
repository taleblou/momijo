# Project:      Momijo
# Module:       learn.utils.checkpoint
# File:         utils/checkpoint.mojo
# Path:         src/momijo/learn/utils/checkpoint.mojo
#
# Description:  Checkpoint utilities for Momijo Learn.
#               Provides stable save/load of model state with a header+payload
#               design. By default writes a JSON header and a separate model
#               state JSON. The format mirrors an MNP layout (JSON header +
#               binary blob), so adding a Float32 blob later is straightforward.
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
#   - Types/Functions: save_state_dict, load_state_dict
#   - Header fields: format, version, created_at, model_class, files
#   - Model contract: model.state_dict() -> String, model.load_state_dict(String)
#   - Future: add Float32 blob file (files.blob) and fill offsets in header.

from pathlib.path import Path
from collections.list import List

# ------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------

fn _escape_json_string(s: String) -> String:
    # Minimal JSON string escaper (quotes, backslash, newline, tab)
    var out = String("")
    var i = 0
    while i < s.__len__():
        var ch = s[i]
        if ch == String("\\")[0]:
            out += String("\\\\")
        elif ch == String("\"")[0]:
            out += String("\\\"")
        elif ch == String("\n")[0]:
            out += String("\\n")
        elif ch == String("\t")[0]:
            out += String("\\t")
        else:
            out += String(ch)
        i = i + 1
    return out

fn _now_iso8601() -> String:
    # Placeholder for timestamp; replace with real clock when available
    # Using a fixed-format stub to keep files deterministic in tests
    return String("1970-01-01T00:00:00Z")

fn _ensure_parent_dir(p: Path):
    var parent = p.parent()
    if not parent.exists():
        parent.mkdir(parents=True)

fn _write_text(p: Path, s: String):
    _ensure_parent_dir(p)
    p.write_text(s)

fn _read_text(p: Path) -> String:
    return p.read_text()

# ------------------------------------------------------------
# Public API
# ------------------------------------------------------------

/// Save a model state to disk with an MNP-like layout.
/// - Writes a header JSON at `<path>.json`
/// - Writes the model's state JSON at `<path>.state.json`
///
/// Contract:
///   model must implement:
///     - state_dict(self) -> String       (JSON string)
///     - load_state_dict(mut self, String)
fn save_state_dict(model, path: String):
    var base = Path(path)
    var header_path = Path(path + String(".json"))
    var state_path  = Path(path + String(".state.json"))
    # Future binary blob path (Float32 concat):
    var blob_path   = Path(path + String(".bin"))  # not written yet

    # Pull model state as JSON string (caller model is responsible for JSON correctness)
    var state_json = model.state_dict()

    # Compose header JSON (strings escaped to be safe)
    var model_class = String("Model")  # if model has a name accessor, use it here
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
            String("\"dtype\":\"float32\",") +          # reserved for future
            String("\"endian\":\"little\",") +
            String("\"total_elems\":0") +
        String("}") +
        String("}")

    # Write header and state
    _write_text(header_path, header_json)
    _write_text(state_path, state_json)

    # Note: when tensor storage is ready, write concatenated Float32 to blob_path
    # and update blob_meta.total_elems + per-tensor offsets in the header.


/// Load a model state from disk created by `save_state_dict`.
/// Reads `<path>.json` header, then loads `<path>.state.json` into the model.
///
/// Returns: Bool indicating success.
fn load_state_dict(model, path: String) -> Bool:
    var header_path = Path(path + String(".json"))
    if not header_path.exists():
        # Fallback: allow passing direct state file without header
        var direct_state_path = Path(path)
        if direct_state_path.exists():
            var state_json = _read_text(direct_state_path)
            model.load_state_dict(state_json)
            return True
        return False

    var header_json = _read_text(header_path)

    # Extremely small parser to extract "state" file path from header.
    # We avoid a full JSON parser to keep dependencies minimal.
    var key = String("\"state\":\"")
    var idx = header_json.find(key)
    if idx < 0:
        return False
    var start = idx + key.__len__()
    var end = header_json.find(String("\""), start)
    if end < 0:
        return False
    var state_path_str = header_json.slice(start, end)

    var state_path = Path(state_path_str)
    if not state_path.exists():
        # Try sibling path derived from base name if absolute path was moved
        var base = Path(path)
        var local_state = Path(base.__str__() + String(".state.json"))
        if local_state.exists():
            state_path = local_state
        else:
            return False

    var state_json = _read_text(state_path)
    model.load_state_dict(state_json)
    return True
