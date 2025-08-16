#
# Copyright (c) 2025 Morteza Taleblou (https://taleblou.ir/)
# All rights reserved.
#
fn words_needed(size_bytes: UInt64) -> UInt64:
    var w = size_bytes / 8
    if (size_bytes % 8) != 0: w += 1
    if w > 4: w = 4
    return w
