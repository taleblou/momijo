# MIT License
# Copyright (c) 2025 Morteza Talebou and Mitra Daneshmand
# Project: momijo  |  Source: https://github.com/taleblou/momijo
# This file is part of the Momijo project. See the LICENSE file at the repository root.
# Module: momijo.enum.unique
# Minimal enum utilities implemented in Mojo.
# Project: momijo.enum
# MIT License
# Copyright (c) 2025 Morteza Talebou (https://taleblou.ir/)
# Momijo Enum
# This file is part of the Momijo project. See the LICENSE file at the repository root.

#
# Copyright (c) 2025 Morteza Taleblou (https:#taleblou.ir/)
# All rights reserved.
#
fn ensure_unique_tags(tags: List[UInt64]) -> Bool:
    var n = len(tags)
    for i in range(0, n):
        for j in range(i+1, n):
            if tags[i] == tags[j]:
                return False
    return True

# Does: utility function in enum module.
# Inputs: names.
# Returns: result value or status.
fn ensure_unique_names(names: List[String]) -> Bool:
    var n = len(names)
    for i in range(0, n):
        for j in range(i+1, n):
            if names[i] == names[j]:
                return False
    return True