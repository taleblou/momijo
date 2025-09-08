# Project:      Momijo
# Module:       src.momijo.io.file_system
# File:         file_system.mojo
# Path:         src/momijo/io/file_system.mojo
#
# Description:  Filesystem/IO helpers with Path-centric APIs and safe resource
#               management (binary/text modes and encoding clarity).
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
#   - Structs: FileSystem
#   - Key functions: exists, is_file, is_dir, read_text, write_text, append_text, delete, list_files ...
#   - Static methods present.
#   - Performs file/Path IO; prefer context-managed patterns.


from os import listdir, remove
from os.path import exists, isdir, isfile, join

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
fn write_text(path: String, content: String) -> None:
        var f = open(path, "w")
        f.write(content)
        f.close()

    # Append text to file
    @staticmethod
fn append_text(path: String, content: String) -> None:
        var f = open(path, "a")
        f.write(content)
        f.close()

    # Delete file
    @staticmethod
fn delete(path: String) -> None:
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
fn __init__(out self, ) -> None:
        pass
fn __copyinit__(out self, other: Self) -> None:
        pass
fn __moveinit__(out self, deinit other: Self) -> None:
        pass
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
fn main() -> None:
    if _self_test():
        print("FileSystem module self-test: OK")
    else:
        print("FileSystem module self-test: FAIL")