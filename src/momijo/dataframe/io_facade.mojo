# Project:      Momijo 
# Module:       dataframe.io_facade
# File:         io_facade.mojo
# Path:         dataframe/io_facade.mojo
#
# Description:  dataframe.io_facade — High-level I/O facade for Momijo DataFrame.
#               Thin wrappers around JSON (columnar / JSON Lines) helpers and
#               string-based MNP helpers, providing a stable entry-point API.
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
#   - Structs: (none) — facade exposes free functions for convenience.
#   - Key functions:
#       * to_json(df, path), read_json(path)
#       * to_json_lines(df, path), read_json_lines(path)
#       * to_json_string(df), from_json_string(s)
#       * to_json_lines_string(df), from_json_lines_string(s)
#       * to_mnp_string(df), from_mnp_string(s)
#   - Static methods present: N/A.
#

from pathlib import Path
import momijo.dataframe as mdf
from momijo.dataframe.io_json_min import to_json_string as _to_json_str
from momijo.dataframe.io_json_min import from_json_string as _from_json_str
from momijo.dataframe.io_json_min import to_json_lines_string as _to_jsonl_str
from momijo.dataframe.io_json_min import from_json_lines_string as _from_jsonl_str
from momijo.dataframe.io_json_min import write_json as _write_json
from momijo.dataframe.io_json_min import read_json as _read_json
from momijo.dataframe.io_pickle_mnp import to_mnp_string as _to_mnp_str
from momijo.dataframe.io_pickle_mnp import from_mnp_string as _from_mnp_str

# ---------- File-based JSON ----------

fn to_json(df: mdf.DataFrame, path: String) -> Bool:
    try:
        return _write_json(df, Path(path), False, False)
    except:
        return False

fn read_json(path: String) -> mdf.DataFrame:
    try:
        return _read_json(Path(path))
    except:
        # Fallback: return an empty DataFrame
        return mdf.read_csv_string(String("empty\n"))

fn to_json_lines(df: mdf.DataFrame, path: String) -> Bool:
    try:
        # orient_records=True, lines=True -> JSON Lines (one object per line)
        return _write_json(df, Path(path), True, True)
    except:
        return False

fn read_json_lines(path: String) -> mdf.DataFrame:
    try:
        # If backend supports auto-detect, this will work; otherwise, parser handles lines
        return _read_json(Path(path))
    except:
        return mdf.read_csv_string(String("empty\n"))

# ---------- String-based JSON ----------

fn to_json_string(df: mdf.DataFrame) -> String:
    return _to_json_str(df)

fn from_json_string(s: String) -> mdf.DataFrame:
    return _from_json_str(s)

fn to_json_lines_string(df: mdf.DataFrame) -> String:
    return _to_jsonl_str(df)

fn from_json_lines_string(s: String) -> mdf.DataFrame:
    return _from_jsonl_str(s)

# ---------- String-based MNP (pickle-like) ----------

fn to_mnp_string(df: mdf.DataFrame) -> String:
    return _to_mnp_str(df)

fn from_mnp_string(s: String) -> mdf.DataFrame:
    return _from_mnp_str(s)
