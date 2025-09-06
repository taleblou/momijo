# ============================================================================
# MIT License
# Copyright (c) 2025 Morteza Talebou and Mitra Daneshmand
# Project: momijo  |  Source: https://github.com/taleblou/momijo
# This file is part of the Momijo project. See the LICENSE file at the repository root.
# SPDX-License-Identifier: MIT
# Website: https://taleblou.ir/
# Repository: https://github.com/taleblou/momijo
#
# Project: momijo.io
# File: src/momijo/io/file_system.mojo
# ============================================================================

# Core Mojo / stdlib imports
from pathlib import Path
from os import listdir, remove
from os.path import isfile, isdir, join, exists

# -----------------------------------------------------------------------------
# FileSystem Utilities
# -----------------------------------------------------------------------------

struct FileSystem:

    # Check if path exists
    @staticmethod
    fn exists(path: String) -> Bool:
        return exists(path)

    # Check if path is a file
    @staticmethod
    fn is_file(path: String) -> Bool:
        return isfile(path)

    # Check if path is a directory
    @staticmethod
    fn is_dir(path: String) -> Bool:
        return isdir(path)

    # Read text content from file
    @staticmethod
    fn read_text(path: String) -> String:
        if not isfile(path):
            return String("")
        var f = open(path, "r")
        var data = f.read()
        f.close()
        return String(data)

    # Write text content to file (overwrite)
    @staticmethod
    fn write_text(path: String, content: String):
        var f = open(path, "w")
        f.write(content)
        f.close()

    # Append text to file
    @staticmethod
    fn append_text(path: String, content: String):
        var f = open(path, "a")
        f.write(content)
        f.close()

    # Delete file
    @staticmethod
    fn delete(path: String):
        if isfile(path):
            remove(path)

    # List files in directory
    @staticmethod
    fn list_files(path: String) -> List[String]:
        if not isdir(path):
            return []
        var files = listdir(path)
        var result = List[String]()
        for f in files:
            var full_path = join(path, f)
            if isfile(full_path):
                result.append(String(full_path))
        return result

    # List directories in directory
    @staticmethod
    fn list_dirs(path: String) -> List[String]:
        if not isdir(path):
            return []
        var files = listdir(path)
        var result = List[String]()
        for f in files:
            var full_path = join(path, f)
            if isdir(full_path):
                result.append(String(full_path))
        return result


# -----------------------------------------------------------------------------
# Minimal self-test
# -----------------------------------------------------------------------------

fn _self_test() -> Bool:
    var ok = True
    var tmp_file = "tmp_test.txt"

    # Test write
    FileSystem.write_text(tmp_file, "hello")
    ok = ok and FileSystem.exists(tmp_file)
    ok = ok and FileSystem.is_file(tmp_file)

    # Test read
    var content = FileSystem.read_text(tmp_file)
    ok = ok and content == String("hello")

    # Test append
    FileSystem.append_text(tmp_file, " world")
    var content2 = FileSystem.read_text(tmp_file)
    ok = ok and content2 == String("hello world")

    # Test list_files
    var files = FileSystem.list_files(".")
    ok = ok and len(files) > 0

    # Cleanup
    FileSystem.delete(tmp_file)
    ok = ok and not FileSystem.exists(tmp_file)

    return ok


fn main():
    if _self_test():
        print("FileSystem module self-test: OK")
    else:
        print("FileSystem module self-test: FAIL")
