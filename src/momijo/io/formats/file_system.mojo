# Project:      Momijo
# Module:       src.momijo.io.formats.file_system
# File:         file_system.mojo
# Path:         src/momijo/io/formats/file_system.mojo
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


import os

struct FileSystem:

    # Check if path exists
    @staticmethod
fn exists(path: String) -> Bool:
        return os.path.exists(path)

    # Check if path is a file
    @staticmethod
fn is_file(path: String) -> Bool:
        return os.path.isfile(path)

    # Check if path is a directory
    @staticmethod
fn is_dir(path: String) -> Bool:
        return os.path.isdir(path)

    # Read text content from file
    @staticmethod
fn read_text(path: String) -> String:
        if not os.path.isfile(path):
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
        if os.path.isfile(path):
            os.remove(path)

    # List files in directory
    @staticmethod
fn list_files(path: String) -> List[String]:
        if not os.path.isdir(path):
            return []
        var files = os.listdir(path)
        var result = List[String]()
        for f in files:
            var full_path = os.path.join(path, f)
            if os.path.isfile(full_path):
                result.append(String(full_path))
        return result

    # List directories in directory
    @staticmethod
fn list_dirs(path: String) -> List[String]:
        if not os.path.isdir(path):
            return []
        var files = os.listdir(path)
        var result = List[String]()
        for f in files:
            var full_path = os.path.join(path, f)
            if os.path.isdir(full_path):
                result.append(String(full_path))
        return result

    # Copy file
    @staticmethod
fn copy_file(src: String, dst: String) -> None:
        var src_f = open(src, "rb")
        var data = src_f.read()
        src_f.close()
        var dst_f = open(dst, "wb")
        dst_f.write(data)
        dst_f.close()

    # Move file
    @staticmethod
fn move_file(src: String, dst: String) -> None:
        FileSystem.copy_file(src, dst)
        FileSystem.delete(src)
fn __init__(out self, ) -> None:
        pass
fn __copyinit__(out self, other: Self) -> None:
        pass
fn __moveinit__(out self, deinit other: Self) -> None:
        pass