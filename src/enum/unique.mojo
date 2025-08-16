#
# Copyright (c) 2025 Morteza Taleblou (https://taleblou.ir/)
# All rights reserved.
#
fn ensure_unique_tags(tags: List[UInt64]) -> Bool:
    var n = len(tags)
    for i in range(0, n):
        for j in range(i+1, n):
            if tags[i] == tags[j]:
                return False
    return True

fn ensure_unique_names(names: List[String]) -> Bool:
    var n = len(names)
    for i in range(0, n):
        for j in range(i+1, n):
            if names[i] == names[j]:
                return False
    return True
