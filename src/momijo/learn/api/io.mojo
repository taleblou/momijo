# Project:      Momijo
# Module:       learn.api.io
# File:         api/io.mojo
# Path:         src/momijo/learn/api/io.mojo
#
# Description:  learn.api.io — High-level save/load façade for Momijo Learn models.
#               Delegates persistence to utils.checkpoint (MNP format: JSON header + blob).
#               Keeps API stable for Keras-like and PyTorch-like workflows.
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
#   - Functions: save_model, load_model
#   - Uses: momijo.learn.utils.checkpoint (save_state_dict/load_state_dict)
#   - Duck-typed model: expects `state_dict()` and `load_state_dict(state: String)`

from pathlib.path import Path
from momijo.learn.utils.checkpoint import (
    save_state_dict,
    load_state_dict,
)
from momijo.learn.api.model import Model  # Generic fallback container

# Normalize a user-provided file path.
fn _normalize_path(path: String) -> Path:
    var p = Path(path)
    return p

# Save a model in Momijo Checkpoint (MNP) format.
# Requirements on `model`:
#   - has method: state_dict() -> String    (JSON or serialized state)
# Prefer saving architecture config separately at higher API levels.
fn save_model(model, path: String):
    var p = _normalize_path(path)
    # Delegate to utils.checkpoint; it is responsible for MNP JSON header + blob handling.
    save_state_dict(model, String(p.as_string()))

# Load a model from Momijo Checkpoint (MNP) format.
# Behavior:
#   - Creates a generic `Model()` as a container and calls `load_state_dict` on it.
#   - In real projects, prefer passing/using a concrete architecture then calling
#     `load_state_dict` yourself, to ensure shapes/topology match.
# Returns:
#   - A `Model` instance populated with the loaded state (as far as supported).
fn load_model(path: String) -> Model:
    var p = _normalize_path(path)
    var m = Model()
    load_state_dict(m, String(p.as_string()))
    return m
