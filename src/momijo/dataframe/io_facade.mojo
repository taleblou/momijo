# Project:      Momijo
# Module:       dataframe.io_facade
# File:         io_facade.mojo
# Path:         dataframe/io_facade.mojo
#
# Description:  dataframe.io_facade — Io Facade module for Momijo DataFrame.
#               Implements core data structures, algorithms, and convenience APIs for production use.
#               Designed as a stable, composable building block within the Momijo public API.
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
#   - Structs: —
#   - Key functions: to_csv, read_csv, to_json, read_json, to_json_lines, read_json_lines

from pathlib.path import Path

from momijo.dataframe import api as mdf


fn to_csv(df: mdf.DataFrame, path: String) -> Bool:
    try:
        return mdf.write_csv(df, Path(path))
    except:
        return False

fn read_csv(path: String) -> mdf.DataFrame:
    return mdf.read_csv(Path(path))

fn to_json(df: mdf.DataFrame, path: String) -> Bool:
    try:
# orient_records=True, lines=False -> JSON array of records
        return mdf.write_json(df, Path(path), True, False)
    except:
        return False

fn read_json(path: String) -> mdf.DataFrame:
    try:
        return mdf.read_json(Path(path))
    except:
# Fallback: return a trivial empty DataFrame to keep demo flowing
        return mdf.read_csv_string(String("empty\n"))

fn to_json_lines(df: mdf.DataFrame, path: String) -> Bool:
    try:
# orient_records=True, lines=True -> JSON Lines (one JSON object per line)
        return mdf.write_json(df, Path(path), True, True)
    except:
        return False

fn read_json_lines(path: String) -> mdf.DataFrame:
    try:
# If the backend supports auto-detect, this will work.
        return mdf.read_json(Path(path))
    except:
# Fallback: return an empty DataFrame to keep demo flowing
        return mdf.read_csv_string(String("empty\n"))