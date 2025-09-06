# ============================================================================
# MIT License
# Copyright (c) 2025 Morteza Talebou and Mitra Daneshmand
# Project: momijo  |  Source: https://github.com/taleblou/momijo
# This file is part of the Momijo project. See the LICENSE file at the repository root.
# SPDX-License-Identifier: MIT
# Website: https://taleblou.ir/
# Repository: https://github.com/taleblou/momijo
#
# Project: momijo.io.formats
# File: src/momijo/io/formats/file_system.mojo
# ============================================================================

import os

# -----------------------------------------------------------------------------
# FileSystem Utilities
# -----------------------------------------------------------------------------
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
    fn copy_file(src: String, dst: String):
        var src_f = open(src, "rb")
        var data = src_f.read()
        src_f.close()
        var dst_f = open(dst, "wb")
        dst_f.write(data)
        dst_f.close()

    # Move file
    @staticmethod
    fn move_file(src: String, dst: String):
        FileSystem.copy_file(src, dst)
        FileSystem.delete(src)


 
