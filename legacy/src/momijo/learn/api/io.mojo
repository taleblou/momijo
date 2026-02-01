# MIT License
# SPDX-License-Identifier: MIT
# Project:      Momijo
# Module:       learn.api.io
# File:         src/momijo/learn/api/io.mojo
#
# Description:
#   High-level save/load façade for Momijo Learn models. Delegates to
#   momijo.learn.utils.checkpoint (MNP format: JSON header + binary blob).
#   Keeps a stable Keras/PyTorch-like API while remaining backend-agnostic.
#
# Author(s):    Morteza Taleblou & Mitra Daneshmand
# Website:      https://taleblou.ir/
# Repository:   https://github.com/taleblou/momijo

from pathlib.path import Path
from momijo.learn.utils.checkpoint import save_state_dict
from momijo.learn.utils.checkpoint import load_state_dict
from momijo.learn.api.model import Model  # Generic fallback container

# -----------------------------------------------------------------------------
# Optional trait to enable typed overloads (static safety when available)
# -----------------------------------------------------------------------------

trait SerializableModel:
    fn state_dict(self) -> String
    fn load_state_dict(mut self, state: String)

# -----------------------------------------------------------------------------
# Small path helpers (OS-agnostic, conservative behavior)
# -----------------------------------------------------------------------------

# Normalize a user-provided file path. If the last path component has no dot,
# conservatively append ".mnp" as the default Momijo checkpoint extension.
fn _normalize_and_default_ext(path: String) -> Path:
    var p = Path(path)
    # Detect missing extension by scanning the final component.
    var name = p.name()
    var has_dot = False
    var i = 0
    while i < Int(len(name)):
        if name[i] == '.':
            has_dot = True
            break
        i = i + 1
    if has_dot:
        return p
    # Append ".mnp" to the last component
    var parent = p.parent()
    var with_ext = String(name) + String(".mnp")
    var out_p = parent / with_ext
    return out_p

# Best-effort: if parent directory exists without throwing.
fn _ensure_parent_dir(path_obj: Path):
    var parent = path_obj.parent()
    if parent.as_string() == String(""):
        return  # no parent to create
    if parent.exists():
        return
    try:
        parent.mkdir()
    except _:
        # Avoid escalation in restricted environments; saving may still succeed if
        # the FS allows implicit parent creation or current working dir is suitable.
        pass

# -----------------------------------------------------------------------------
# Public API: Save
# -----------------------------------------------------------------------------
# Requirements on `model` (duck-typed path):
#   - model.state_dict() -> String
# Prefer saving the architecture config at higher levels 

fn save_model(model, path: String):
    var p = _normalize_and_default_ext(path)
    _ensure_parent_dir(p)
    # Delegate persistence to the central checkpoint utility (MNP format).
    save_state_dict(model, String(p.as_string()))

# Overload returning Bool to indicate best-effort success without raising.
fn save_model_ok(model, path: String) -> Bool:
    var p = _normalize_and_default_ext(path)
    _ensure_parent_dir(p)
    try:
        save_state_dict(model, String(p.as_string()))
        return True
    except _:
        return False

# Typed overloads when the model implements SerializableModel (static safety).
fn save_model(model: SerializableModel, path: String):
    var p = _normalize_and_default_ext(path)
    _ensure_parent_dir(p)
    save_state_dict(model, String(p.as_string()))

fn save_model_ok(model: SerializableModel, path: String) -> Bool:
    var p = _normalize_and_default_ext(path)
    _ensure_parent_dir(p)
    try:
        save_state_dict(model, String(p.as_string()))
        return True
    except _:
        return False

# -----------------------------------------------------------------------------
# Public API: Load
# -----------------------------------------------------------------------------
# Behavior (generic path):
#   - Creates a generic Model() container and asks checkpoint utils to populate it. 
#   - In production, prefer constructing the exact model then calling its
#     load_state_dict(...) yourself to guarantee shape/topology compatibility.

fn load_model(path: String) -> Model:
    var p = _normalize_and_default_ext(path)
    var m = Model()
    load_state_dict(m, String(p.as_string()))
    return m

# Overload returning (ok, Model) to avoid raising in strict pipelines.
fn load_model_ok(path: String) -> (Bool, Model):
    var p = _normalize_and_default_ext(path)
    var m = Model()
    try:
        load_state_dict(m, String(p.as_string()))
        return (True, m)
    except _:
        return (False, m)

# Typed load into an existing SerializableModel instance (preferred pattern).
# Returns Bool to indicate success; the caller supplies the concrete model.
fn load_into(mut model: SerializableModel, path: String) -> Bool:
    var p = _normalize_and_default_ext(path)
    try:
        load_state_dict(model, String(p.as_string()))
        return True
    except _:
        return False

# Convenience constructor: allocate a default Model and try to load it.
# Useful for quick scripts where a generic container suffices.
fn load_new(path: String) -> (Bool, Model):
    return load_model_ok(path)
